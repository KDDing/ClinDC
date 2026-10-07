from __future__ import annotations

import torch
from torch import nn


class DrugDiseaseDrugModel(nn.Module):
    """Inductive KG/text model with a permutation-invariant drug-pair decoder."""

    def __init__(
        self,
        num_drugs: int,
        embedding_dim: int,
        hidden_dim: int,
        kg_buckets: int,
        text_buckets: int,
        drug_mode: str = "id_kg",
        disease_mode: str = "kg_text",
        fusion: str = "gated",
        pair_mode: str = "symmetric",
        structure_dim: int = 0,
        architecture: str = "standard",
        decoder_dropout_1: float = 0.2,
        decoder_dropout_2: float = 0.1,
        modality_dropout: float = 0.0,
        pair_modalities=("q", "m", "g"),
        pair_fusion: str = "attention",
        pair_include_virtual: bool = True,
        pair_attention_disease: bool = True,
        pair_representation: str = "symmetric",
        bidirectional_inference: bool = False,
    ):
        super().__init__()
        self.drug_mode = drug_mode
        self.disease_mode = disease_mode
        self.fusion = fusion
        self.pair_mode = pair_mode
        self.structure_dim = structure_dim
        self.architecture = architecture
        self.modality_dropout = modality_dropout
        self.pair_modalities = tuple(pair_modalities)
        self.pair_fusion = pair_fusion
        self.pair_include_virtual = pair_include_virtual
        self.pair_attention_disease = pair_attention_disease
        self.pair_representation = pair_representation
        self.bidirectional_inference = bidirectional_inference
        self.drug_id = nn.Embedding(num_drugs, embedding_dim)
        self.kg_bag = nn.EmbeddingBag(
            kg_buckets, embedding_dim, mode="mean", padding_idx=0
        )
        self.text_bag = nn.EmbeddingBag(
            text_buckets, embedding_dim, mode="mean", padding_idx=0
        )
        self.drug_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid()
        )
        if structure_dim:
            self.structure_projection = nn.Sequential(
                nn.Linear(structure_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
            )
            self.structure_kg_gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid()
            )
            self.id_structure_kg_gate = nn.Linear(
                embedding_dim * 3, embedding_dim * 3
            )
        self.disease_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid()
        )
        if architecture == "invariant_virtual" or (
            architecture == "pair_level_multimodal" and pair_include_virtual
        ):
            self.invariant_attention = nn.Sequential(
                nn.Linear(embedding_dim * 3, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, 1),
            )
        if architecture == "pair_level_multimodal":
            if "m" in self.pair_modalities and not structure_dim:
                raise ValueError("pair_level_multimodal requires molecular features")
            if not self.pair_modalities:
                raise ValueError("pair_level_multimodal requires at least one modality")
            unknown = set(self.pair_modalities) - {"q", "m", "g"}
            if unknown:
                raise ValueError(f"Unknown pair modalities: {sorted(unknown)}")
            if pair_fusion not in {"attention", "mean"}:
                raise ValueError(f"Unknown pair fusion: {pair_fusion}")
            if pair_representation not in {"symmetric", "ordered"}:
                raise ValueError(
                    f"Unknown pair representation: {pair_representation}"
                )
            modality_pair_terms = (
                2 if pair_representation == "ordered"
                else 3 + int(pair_include_virtual)
            )
            modality_pair_dim = embedding_dim * modality_pair_terms
            self.modality_pair_norms = nn.ModuleList(
                [nn.LayerNorm(modality_pair_dim) for _ in self.pair_modalities]
            )
            if pair_fusion == "attention":
                attention_dim = modality_pair_dim + (
                    embedding_dim if pair_attention_disease else 0
                )
                self.modality_attention = nn.Sequential(
                    nn.Linear(attention_dim, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, 1),
                )
        pair_terms = (
            modality_pair_terms
            if architecture == "pair_level_multimodal"
            else (4 if architecture == "invariant_virtual" else (
                3 if pair_mode == "symmetric" else 2
            ))
        )
        pair_dim = embedding_dim * pair_terms
        self.decoder = nn.Sequential(
            nn.Linear(pair_dim + embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(decoder_dropout_1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(decoder_dropout_2),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.drug_id.weight, std=0.02)

    @staticmethod
    def _bag(layer, packed):
        values, offsets = packed
        return layer(values, offsets)

    def encode_drug(
        self, ids, kg_tokens, structure=None, structure_valid=None
    ):
        learned = self.drug_id(ids)
        kg = self._bag(self.kg_bag, kg_tokens)
        if self.drug_mode == "id_only":
            return learned
        if self.drug_mode == "kg_only":
            return kg
        if self.drug_mode in {"structure_kg", "id_structure_kg"}:
            if structure is None or structure_valid is None:
                raise ValueError(f"{self.drug_mode} requires structural features")
            molecular = self.structure_projection(structure)
            valid = structure_valid.unsqueeze(-1)
            if self.drug_mode == "structure_kg":
                gate = self.structure_kg_gate(torch.cat([molecular, kg], dim=-1))
                # Missing structures fall back exactly to the KG representation.
                return valid * (gate * molecular + (1.0 - gate) * kg) + (
                    1.0 - valid
                ) * kg
            components = torch.stack([learned, molecular, kg], dim=1)
            logits = self.id_structure_kg_gate(
                torch.cat([learned, molecular, kg], dim=-1)
            ).reshape(-1, 3, learned.shape[-1])
            unavailable = torch.zeros_like(logits, dtype=torch.bool)
            unavailable[:, 1, :] = (structure_valid <= 0).view(-1, 1)
            if self.training and self.modality_dropout > 0:
                unavailable[:, 0, :] = (
                    torch.rand(len(ids), 1, device=ids.device)
                    < self.modality_dropout
                )
                unavailable[:, 1, :] |= (
                    torch.rand(len(ids), 1, device=ids.device)
                    < self.modality_dropout
                )
            logits = logits.masked_fill(
                unavailable, torch.finfo(logits.dtype).min
            )
            weights = torch.softmax(logits, dim=1)
            return (weights * components).sum(dim=1)
        if self.fusion == "mean":
            return 0.5 * (learned + kg)
        gate = self.drug_gate(torch.cat([learned, kg], dim=-1))
        return gate * learned + (1.0 - gate) * kg

    def encode_disease(self, kg_tokens, text_tokens):
        kg = self._bag(self.kg_bag, kg_tokens)
        text = self._bag(self.text_bag, text_tokens)
        if self.disease_mode == "text_only":
            return text
        if self.disease_mode == "kg_only":
            return kg
        if self.fusion == "mean":
            return 0.5 * (kg + text)
        gate = self.disease_gate(torch.cat([kg, text], dim=-1))
        # For unmapped diseases the KG signature is token 0. Padding embeddings
        # are fixed at zero, so the representation naturally falls back to text.
        return gate * kg + (1.0 - gate) * text

    def _invariant_virtual(self, h1, h2, disease):
        score_1 = self.invariant_attention(
            torch.cat([h1, disease, h1 * disease], dim=-1)
        )
        score_2 = self.invariant_attention(
            torch.cat([h2, disease, h2 * disease], dim=-1)
        )
        weights = torch.softmax(torch.cat([score_1, score_2], dim=-1), dim=-1)
        return weights[:, :1] * h1 + weights[:, 1:] * h2

    def _symmetric_pair(self, h1, h2, disease, include_virtual=False):
        terms = [h1 + h2, torch.abs(h1 - h2), h1 * h2]
        if include_virtual:
            terms.append(self._invariant_virtual(h1, h2, disease))
        return torch.cat(terms, dim=-1)

    def _raw_drug_modalities(
        self, ids, kg_tokens, structure=None, structure_valid=None
    ):
        result = {}
        if "q" in self.pair_modalities:
            result["q"] = self.drug_id(ids)
        if "m" in self.pair_modalities:
            if structure is None or structure_valid is None:
                raise ValueError(
                    "Morgan pair modality requires structural features"
                )
            molecular = self.structure_projection(structure)
            result["m"] = molecular * structure_valid.unsqueeze(-1)
        if "g" in self.pair_modalities:
            result["g"] = self._bag(self.kg_bag, kg_tokens)
        return result

    @staticmethod
    def _swap_drugs(batch):
        swapped = dict(batch)
        suffixes = ("", "_kg", "_structure", "_structure_valid")
        for suffix in suffixes:
            left = f"drug_1{suffix}"
            right = f"drug_2{suffix}"
            if left in batch and right in batch:
                swapped[left], swapped[right] = batch[right], batch[left]
        return swapped

    def _forward_once(self, batch):
        hd = self.encode_disease(batch["disease_kg"], batch["disease_text"])
        if self.architecture == "pair_level_multimodal":
            modalities_1 = self._raw_drug_modalities(
                batch["drug_1"], batch["drug_1_kg"],
                batch.get("drug_1_structure"),
                batch.get("drug_1_structure_valid"),
            )
            modalities_2 = self._raw_drug_modalities(
                batch["drug_2"], batch["drug_2_kg"],
                batch.get("drug_2_structure"),
                batch.get("drug_2_structure_valid"),
            )
            pair_values = []
            for norm, modality in zip(
                self.modality_pair_norms, self.pair_modalities
            ):
                a, b = modalities_1[modality], modalities_2[modality]
                raw_pair = (
                    torch.cat([a, b], dim=-1)
                    if self.pair_representation == "ordered"
                    else self._symmetric_pair(
                        a, b, hd, include_virtual=self.pair_include_virtual
                    )
                )
                pair_values.append(norm(raw_pair))
            pairs = torch.stack(pair_values, dim=1)
            if self.pair_fusion == "attention":
                attention_input = pairs
                if self.pair_attention_disease:
                    disease_context = hd.unsqueeze(1).expand(
                        -1, pairs.shape[1], -1
                    )
                    attention_input = torch.cat(
                        [pairs, disease_context], dim=-1
                    )
                logits = self.modality_attention(attention_input).squeeze(-1)
            else:
                logits = torch.zeros(
                    pairs.shape[:2], device=pairs.device, dtype=pairs.dtype
                )
            unavailable = torch.zeros_like(logits, dtype=torch.bool)
            if "m" in self.pair_modalities:
                structure_available = (
                    (batch["drug_1_structure_valid"] > 0)
                    & (batch["drug_2_structure_valid"] > 0)
                )
                unavailable[:, self.pair_modalities.index("m")] = (
                    ~structure_available
                )
            logits = logits.masked_fill(
                unavailable, torch.finfo(logits.dtype).min
            )
            weights = torch.softmax(logits, dim=1).unsqueeze(-1)
            pair = (weights * pairs).sum(dim=1)
            return self.decoder(torch.cat([pair, hd], dim=-1)).squeeze(-1)

        h1 = self.encode_drug(
            batch["drug_1"],
            batch["drug_1_kg"],
            batch.get("drug_1_structure"),
            batch.get("drug_1_structure_valid"),
        )
        h2 = self.encode_drug(
            batch["drug_2"],
            batch["drug_2_kg"],
            batch.get("drug_2_structure"),
            batch.get("drug_2_structure_valid"),
        )
        if self.pair_mode == "ordered":
            pair = torch.cat([h1, h2], dim=-1)
        else:
            pair = self._symmetric_pair(
                h1, h2, hd,
                include_virtual=self.architecture == "invariant_virtual",
            )
        return self.decoder(torch.cat([pair, hd], dim=-1)).squeeze(-1)

    def forward(self, batch):
        forward_score = self._forward_once(batch)
        if self.training or not self.bidirectional_inference:
            return forward_score
        reverse_score = self._forward_once(self._swap_drugs(batch))
        return 0.5 * (forward_score + reverse_score)
