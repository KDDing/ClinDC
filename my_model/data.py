from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


def stable_bucket(text: str, buckets: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "little") % (buckets - 1)


def text_tokens(text: str, buckets: int) -> List[int]:
    normalized = f"  {' '.join(text.casefold().split())}  "
    tokens = {
        stable_bucket(normalized[i : i + n], buckets)
        for n in (3, 4, 5)
        for i in range(max(0, len(normalized) - n + 1))
    }
    return sorted(tokens) or [0]


@dataclass(frozen=True)
class Example:
    drug_1: int
    disease: int
    drug_2: int
    evidence_count: int


class TaskData:
    def __init__(self, triples_path: Path):
        rows = []
        with triples_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        drug_names = sorted(
            {r["drug_1"] for r in rows} | {r["drug_2"] for r in rows}
        )
        disease_names = sorted({r["disease"] for r in rows})
        self.drug_to_idx = {name: i for i, name in enumerate(drug_names)}
        self.disease_to_idx = {name: i for i, name in enumerate(disease_names)}
        self.idx_to_drug = drug_names
        self.idx_to_disease = disease_names
        self.splits: Dict[str, List[Example]] = defaultdict(list)
        self.all_positive: Set[Tuple[int, int, int]] = set()
        self.disease_kg_keys: Dict[int, Set[Tuple[str, str]]] = defaultdict(set)
        self.mapped_diseases: Set[int] = set()

        for row in rows:
            d1 = self.drug_to_idx[row["drug_1"]]
            d2 = self.drug_to_idx[row["drug_2"]]
            disease = self.disease_to_idx[row["disease"]]
            example = Example(d1, disease, d2, int(row["evidence_count"]))
            self.splits[row["split"]].append(example)
            self.all_positive.add(self.canonical_key(d1, disease, d2))
            ids = [x for x in row.get("kg_disease_ids", "").split("|") if x]
            sources = [x for x in row.get("kg_disease_sources", "").split("|") if x]
            for source in sources:
                for entity_id in ids:
                    self.disease_kg_keys[disease].add((source, entity_id))
            if ids:
                self.mapped_diseases.add(disease)

        self.train_drugs = {
            drug
            for ex in self.splits["train"]
            for drug in (ex.drug_1, ex.drug_2)
        }
        self.train_diseases = {ex.disease for ex in self.splits["train"]}
        self.train_pairs = {
            (min(ex.drug_1, ex.drug_2), max(ex.drug_1, ex.drug_2))
            for ex in self.splits["train"]
        }

    @staticmethod
    def canonical_key(d1: int, disease: int, d2: int) -> Tuple[int, int, int]:
        return (min(d1, d2), disease, max(d1, d2))

    def strata(self, ex: Example) -> List[str]:
        labels = ["overall"]
        new_drug = ex.drug_1 not in self.train_drugs or ex.drug_2 not in self.train_drugs
        new_disease = ex.disease not in self.train_diseases
        pair = (min(ex.drug_1, ex.drug_2), max(ex.drug_1, ex.drug_2))
        if new_drug:
            labels.append("drug_cold")
        if new_disease:
            labels.append("disease_cold")
        if pair not in self.train_pairs:
            labels.append("pair_cold")
        if not new_drug and not new_disease:
            labels.append("fully_transductive")
        labels.append(
            "kg_mapped" if ex.disease in self.mapped_diseases else "disease_unmapped"
        )
        return labels


class KGSignatures:
    """Stream kg.csv and build bounded hashed one-hop signatures."""

    def __init__(
        self,
        data: TaskData,
        kg_path: Path,
        buckets: int,
        max_tokens: int,
        progress_callback: Optional[Callable[[int], None]] = None,
        signature_mode: str = "full",
    ):
        if signature_mode == "none":
            self.drug_tokens = [[0] for _ in data.idx_to_drug]
            self.disease_tokens = [[0] for _ in data.idx_to_disease]
            return
        target_drugs = set(data.idx_to_drug)
        disease_key_to_idx = {
            key: disease
            for disease, keys in data.disease_kg_keys.items()
            for key in keys
        }
        drug_sets: Dict[int, Set[int]] = defaultdict(set)
        disease_sets: Dict[int, Set[int]] = defaultdict(set)

        with kg_path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 1):
                if progress_callback is not None and row_number % 1_000_000 == 0:
                    progress_callback(row_number)
                relation = row["relation"]
                for side, other, direction in (("x", "y", "out"), ("y", "x", "in")):
                    entity_type = row[f"{side}_type"]
                    entity_id = row[f"{side}_id"].strip()
                    source = row[f"{side}_source"].strip()
                    destination = None
                    if entity_type == "drug" and entity_id in target_drugs:
                        destination = drug_sets[data.drug_to_idx[entity_id]]
                    elif entity_type == "disease":
                        disease_idx = disease_key_to_idx.get((source, entity_id))
                        if disease_idx is not None:
                            destination = disease_sets[disease_idx]
                    if destination is None or signature_mode == "none":
                        continue
                    other_type = row[f"{other}_type"]
                    other_source = row[f"{other}_source"]
                    other_id = row[f"{other}_id"]
                    destination.add(
                        stable_bucket(
                            f"r={relation}|dir={direction}|type={other_type}",
                            buckets,
                        )
                    )
                    if signature_mode == "full":
                        destination.add(
                            stable_bucket(
                                f"r={relation}|nbr={other_type}:{other_source}:{other_id}",
                                buckets,
                            )
                        )

        self.drug_tokens = [
            sorted(drug_sets.get(i, {0}))[:max_tokens]
            for i in range(len(data.idx_to_drug))
        ]
        self.disease_tokens = [
            sorted(disease_sets.get(i, {0}))[:max_tokens]
            for i in range(len(data.idx_to_disease))
        ]


def pack_token_lists(token_lists: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    values: List[int] = []
    offsets: List[int] = []
    for tokens in token_lists:
        offsets.append(len(values))
        values.extend(tokens or [0])
    return torch.tensor(values, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class BatchBuilder:
    def __init__(
        self,
        data: TaskData,
        kg: KGSignatures,
        text_buckets: int,
        negative_ratio: int,
        seed: int,
        drug_features_path: Optional[Path] = None,
        negative_drug_pool: Optional[Sequence[int]] = None,
    ):
        self.data = data
        self.kg = kg
        self.negative_ratio = negative_ratio
        self.rng = random.Random(seed)
        self.negative_drug_pool = (
            tuple(range(len(data.idx_to_drug)))
            if negative_drug_pool is None
            else tuple(int(index) for index in negative_drug_pool)
        )
        if not self.negative_drug_pool:
            raise ValueError("negative_drug_pool must not be empty")
        self.disease_text = [
            text_tokens(name, text_buckets) for name in data.idx_to_disease
        ]
        self.drug_features = None
        self.drug_feature_valid = None
        self.drug_feature_metadata = {}
        if drug_features_path is not None:
            archive = np.load(drug_features_path, allow_pickle=False)
            feature_ids = [str(value) for value in archive["drug_ids"]]
            feature_index = {drug: i for i, drug in enumerate(feature_ids)}
            missing_ids = [
                drug for drug in data.idx_to_drug if drug not in feature_index
            ]
            if missing_ids:
                raise ValueError(
                    f"Drug feature file is missing {len(missing_ids)} dataset IDs; "
                    f"first: {missing_ids[:5]}"
                )
            order = [feature_index[drug] for drug in data.idx_to_drug]
            self.drug_features = torch.from_numpy(
                archive["features"][order].astype(np.float32, copy=False)
            )
            self.drug_feature_valid = torch.from_numpy(
                archive["valid"][order].astype(np.float32, copy=False)
            )
            if "metadata" in archive:
                import json

                self.drug_feature_metadata = json.loads(str(archive["metadata"]))

    @property
    def drug_feature_dim(self) -> int:
        return 0 if self.drug_features is None else int(self.drug_features.shape[1])

    def negative(self, ex: Example) -> Example:
        for _ in range(100):
            replacement = self.rng.choice(self.negative_drug_pool)
            if replacement == ex.drug_1:
                continue
            key = self.data.canonical_key(ex.drug_1, ex.disease, replacement)
            if key not in self.data.all_positive:
                return Example(ex.drug_1, ex.disease, replacement, 1)
        raise RuntimeError("Unable to sample a filtered negative")

    def build(self, positives: Sequence[Example], include_negatives: bool = True):
        examples: List[Example] = []
        labels: List[float] = []
        for ex in positives:
            examples.append(ex)
            labels.append(1.0)
            if include_negatives:
                for _ in range(self.negative_ratio):
                    examples.append(self.negative(ex))
                    labels.append(0.0)
        return self.collate(examples), torch.tensor(labels, dtype=torch.float)

    def collate(self, examples: Sequence[Example]):
        """Convert an explicit list of triples into model inputs."""
        d1 = torch.tensor([x.drug_1 for x in examples], dtype=torch.long)
        d2 = torch.tensor([x.drug_2 for x in examples], dtype=torch.long)
        disease = torch.tensor([x.disease for x in examples], dtype=torch.long)
        evidence = torch.tensor([x.evidence_count for x in examples], dtype=torch.float)
        inputs = {
            "drug_1": d1,
            "drug_2": d2,
            "disease": disease,
            "drug_1_kg": pack_token_lists([self.kg.drug_tokens[i] for i in d1]),
            "drug_2_kg": pack_token_lists([self.kg.drug_tokens[i] for i in d2]),
            "disease_kg": pack_token_lists([self.kg.disease_tokens[i] for i in disease]),
            "disease_text": pack_token_lists([self.disease_text[i] for i in disease]),
            "evidence": evidence,
        }
        if self.drug_features is not None:
            inputs.update(
                {
                    "drug_1_structure": self.drug_features[d1],
                    "drug_2_structure": self.drug_features[d2],
                    "drug_1_structure_valid": self.drug_feature_valid[d1],
                    "drug_2_structure_valid": self.drug_feature_valid[d2],
                }
            )
        return inputs


def batches(items: Sequence[Example], batch_size: int, rng: random.Random):
    indices = list(range(len(items)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [items[i] for i in indices[start : start + batch_size]]
