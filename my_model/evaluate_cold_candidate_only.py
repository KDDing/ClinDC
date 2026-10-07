"""Evaluate warm-to-cold Test-1 queries over the fixed cold-drug candidate set."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data import Example, TaskData
from evaluate_checkpoint_subgroups import build_from_checkpoint
from train import evaluate_full_candidates, move


ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def macro_metrics(model, data, builder, cold_indices, batch_size, device, method):
    positives = collections.defaultdict(set)
    known = collections.defaultdict(set)
    for split, examples in data.splits.items():
        for ex in examples:
            known[(ex.drug_1, ex.disease)].add(ex.drug_2)
            known[(ex.drug_2, ex.disease)].add(ex.drug_1)
            if split == "test1":
                # The prepared split guarantees drug_1=warm and drug_2=cold.
                positives[(ex.drug_1, ex.disease)].add(ex.drug_2)
    queries = sorted(positives)
    rows = []
    model.eval()
    for start in range(0, len(queries), batch_size):
        batch_queries = queries[start : start + batch_size]
        examples, labels, offsets = [], [], [0]
        for anchor, disease in batch_queries:
            positive_targets = positives[(anchor, disease)]
            all_known = known[(anchor, disease)]
            for candidate in cold_indices:
                if candidate in all_known and candidate not in positive_targets:
                    continue
                examples.append(Example(anchor, disease, candidate, 1))
                labels.append(1 if candidate in positive_targets else 0)
            offsets.append(len(examples))
        scores = model(move(builder.collate(examples), device)).detach().cpu().numpy()
        labels_array = np.asarray(labels, dtype=np.int8)
        for index, (anchor, disease) in enumerate(batch_queries):
            lo, hi = offsets[index], offsets[index + 1]
            y_true, y_score = labels_array[lo:hi], scores[lo:hi]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                raise RuntimeError("Cold-only query lacks both positive and unlabeled candidates")
            rows.append({
                "method": method,
                "anchor_drug": data.idx_to_drug[anchor],
                "disease": data.idx_to_disease[disease],
                "n_candidates": len(y_true),
                "n_positives": int(y_true.sum()),
                "auroc": float(roc_auc_score(y_true, y_score)),
                "aupr": float(average_precision_score(y_true, y_score)),
            })
    return {
        "num_queries": len(rows),
        "macro_auroc": float(np.mean([row["auroc"] for row in rows])),
        "macro_aupr": float(np.mean([row["aupr"] for row in rows])),
    }, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--triples", type=Path, required=True)
    parser.add_argument("--cold-drugs", type=Path, required=True)
    parser.add_argument("--kg", type=Path, default=ROOT / "data/kg.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    data = TaskData(args.triples)
    device = torch.device(args.device)
    model, builder, checkpoint = build_from_checkpoint(args.checkpoint, data, args.kg, device)
    with args.cold_drugs.open(encoding="utf-8", newline="") as handle:
        cold_ids = {row["drug_id"] for row in csv.DictReader(handle)}
    missing = sorted(cold_ids - set(data.drug_to_idx))
    if missing:
        raise RuntimeError(f"Cold drugs absent from vocabulary: {missing[:5]}")
    cold_indices = sorted(data.drug_to_idx[drug] for drug in cold_ids)
    if any(any(int(token) != 0 for token in builder.kg.drug_tokens[index]) for index in cold_indices):
        raise RuntimeError("At least one cold candidate has nonzero Drug KG tokens")

    ranking = evaluate_full_candidates(
        model, data.splits["test1"], data, builder, args.query_batch_size, device,
        stage_name=f"{args.method} Test-1 cold-candidate-only",
        candidate_pool=cold_indices, tie_policy="average",
    )
    macro, rows = macro_metrics(
        model, data, builder, cold_indices, args.query_batch_size, device, args.method
    )
    result = {
        "method": args.method,
        "seed": int(checkpoint["config"]["seed"]),
        "protocol": "Test-1 warm-to-cold; fixed 156 KG-unmapped structure-valid candidates; filtered; average tie rank",
        "num_cold_candidates": len(cold_indices),
        "num_test_triples": len(data.splits["test1"]),
        "ranking": ranking,
        "macro": macro,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
