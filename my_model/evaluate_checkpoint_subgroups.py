from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch

from ablations import ABLATIONS
from data import BatchBuilder, KGSignatures, TaskData
from model import DrugDiseaseDrugModel
from train import evaluate_full_candidates


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def build_from_checkpoint(checkpoint_path, data, kg_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg = checkpoint["config"]
    if checkpoint.get("adapted_method") or checkpoint.get("baseline"):
        raise ValueError("This publication package evaluates ClinDC checkpoints only")
    variant_name = checkpoint.get("variant_name", "main_no_id_without_virtual_node")
    if variant_name not in ABLATIONS:
        raise ValueError(f"Checkpoint is not a packaged ClinDC variant: {variant_name}")
    variant = checkpoint.get("variant", ABLATIONS[variant_name])
    drug_features = checkpoint.get("drug_features")

    signature_mode = variant["kg_signature_mode"]
    kg = KGSignatures(
        data,
        kg_path,
        cfg["kg_hash_buckets"],
        cfg["max_kg_tokens_per_entity"],
        signature_mode=signature_mode,
    )
    builder = BatchBuilder(
        data,
        kg,
        cfg["text_hash_buckets"],
        cfg["negative_ratio"],
        cfg["seed"],
        drug_features_path=Path(drug_features) if drug_features else None,
    )

    model = DrugDiseaseDrugModel(
            len(data.idx_to_drug),
            cfg["embedding_dim"],
            cfg["hidden_dim"],
            cfg["kg_hash_buckets"],
            cfg["text_hash_buckets"],
            drug_mode=variant["drug_mode"],
            disease_mode=variant["disease_mode"],
            fusion=variant["fusion"],
            pair_mode=variant["pair_mode"],
            structure_dim=builder.drug_feature_dim,
            architecture=variant.get("architecture", "standard"),
            decoder_dropout_1=cfg.get("decoder_dropout_1", 0.2),
            decoder_dropout_2=cfg.get("decoder_dropout_2", 0.1),
            modality_dropout=cfg.get("modality_dropout", 0.0),
            pair_modalities=variant.get("pair_modalities", ("q", "m", "g")),
            pair_fusion=variant.get("pair_fusion", "attention"),
            pair_include_virtual=variant.get("pair_include_virtual", True),
            pair_attention_disease=variant.get("pair_attention_disease", True),
            pair_representation=variant.get("pair_representation", "symmetric"),
            bidirectional_inference=variant.get(
                "bidirectional_inference", False
            ),
        )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), builder, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints on disease-cold and top-disease subsets."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--triples",
        type=Path,
        default=PROJECT_ROOT
        / "data/dataset_v2_ontology/drug_disease_drug_kg_mapped.csv",
    )
    parser.add_argument(
        "--kg", type=Path, default=PROJECT_ROOT / "data/kg.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-diseases", type=int, default=6)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    data = TaskData(args.triples)
    model, builder, checkpoint = build_from_checkpoint(
        args.checkpoint, data, args.kg, device
    )

    disease_counts = Counter(ex.disease for ex in data.splits["test"])
    top_disease_ids = [
        disease for disease, _ in disease_counts.most_common(args.top_diseases)
    ]
    disease_cold = [
        ex for ex in data.splits["valid"] if ex.disease not in data.train_diseases
    ]
    result = {
        "method": args.method,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": checkpoint["config"]["seed"],
        "best_epoch": checkpoint.get("best_epoch"),
        "candidate_protocol": "full_filtered",
        "disease_cold_partition": "validation",
        "disease_cold": evaluate_full_candidates(
            model,
            disease_cold,
            data,
            builder,
            args.query_batch_size,
            device,
            stage_name=f"{args.method} disease-cold validation",
        ),
        "top_test_diseases": {},
    }
    for disease in top_disease_ids:
        examples = [
            ex for ex in data.splits["test"] if ex.disease == disease
        ]
        name = data.idx_to_disease[disease]
        result["top_test_diseases"][name] = evaluate_full_candidates(
            model,
            examples,
            data,
            builder,
            args.query_batch_size,
            device,
            stage_name=f"{args.method} test disease={name}",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for disease, metrics in result["top_test_diseases"].items():
        rows.append(
            {
                "method": args.method,
                "disease": disease,
                "n": metrics["strata"]["overall"]["n"],
                "mrr": metrics["mrr"],
                "hits@1": metrics["hits@1"],
                "hits@3": metrics["hits@3"],
                "hits@10": metrics["hits@10"],
            }
        )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
