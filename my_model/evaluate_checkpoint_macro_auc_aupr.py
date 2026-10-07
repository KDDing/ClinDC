from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data import TaskData
from evaluate_checkpoint_subgroups import build_from_checkpoint
from train import log, move


ROOT = Path(__file__).resolve().parents[1]


def query_maps(data: TaskData, split: str):
    evaluation = collections.defaultdict(set)
    known = collections.defaultdict(set)
    for partition, examples in data.splits.items():
        for ex in examples:
            known[(ex.drug_1, ex.disease)].add(ex.drug_2)
            known[(ex.drug_2, ex.disease)].add(ex.drug_1)
            if partition == split:
                evaluation[(ex.drug_1, ex.disease)].add(ex.drug_2)
                evaluation[(ex.drug_2, ex.disease)].add(ex.drug_1)
    return evaluation, known


@torch.no_grad()
def evaluate(model, data, builder, split, query_batch_size, device, method):
    model.eval()
    positives, known = query_maps(data, split)
    queries = sorted(positives)
    rows = []
    started = time.perf_counter()
    log(
        f"Starting {method} macro AUROC/AUPR on {split}: "
        f"{len(queries):,} bidirectional anchor-disease queries"
    )
    for start in range(0, len(queries), query_batch_size):
        batch_queries = queries[start : start + query_batch_size]
        examples, labels, offsets = [], [], [0]
        for anchor, disease in batch_queries:
            evaluation_partners = positives[(anchor, disease)]
            all_known_partners = known[(anchor, disease)]
            for candidate in range(len(data.idx_to_drug)):
                if candidate == anchor:
                    continue
                if candidate in all_known_partners and candidate not in evaluation_partners:
                    continue
                examples.append(
                    type(data.splits[split][0])(anchor, disease, candidate, 1)
                )
                labels.append(1 if candidate in evaluation_partners else 0)
            offsets.append(len(examples))
        scores = model(move(builder.collate(examples), device)).detach().cpu().numpy()
        label_array = np.asarray(labels, dtype=np.int8)
        for index, (anchor, disease) in enumerate(batch_queries):
            lo, hi = offsets[index], offsets[index + 1]
            y_true = label_array[lo:hi]
            y_score = scores[lo:hi]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                raise RuntimeError("A query lacks a positive or an unlabeled candidate")
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "anchor_drug": data.idx_to_drug[anchor],
                    "disease": data.idx_to_disease[disease],
                    "n_candidates": len(y_true),
                    "n_positives": int(y_true.sum()),
                    "auroc": float(roc_auc_score(y_true, y_score)),
                    "aupr": float(average_precision_score(y_true, y_score)),
                }
            )
        completed = min(start + len(batch_queries), len(queries))
        if completed % 250 == 0 or completed == len(queries):
            elapsed = time.perf_counter() - started
            log(f"{method}: {completed:,}/{len(queries):,} queries, {completed / elapsed:.1f}/s")
    result = {
        "method": method,
        "split": split,
        "candidate_protocol": "full_filtered_positive_unlabeled",
        "query_unit": "unique bidirectional (anchor_drug, disease, ?)",
        "known_positive_filter": (
            "Known partners outside the evaluated partition are excluded; "
            "partners in the evaluated partition are positive labels."
        ),
        "num_queries": len(rows),
        "num_positive_query_candidate_pairs": sum(row["n_positives"] for row in rows),
        "macro_auroc": float(np.mean([row["auroc"] for row in rows])),
        "macro_aupr": float(np.mean([row["aupr"] for row in rows])),
    }
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triples", type=Path, default=ROOT / "data/dataset_v2_ontology/drug_disease_drug_kg_mapped.csv")
    parser.add_argument("--kg", type=Path, default=ROOT / "data/kg.csv")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    data = TaskData(args.triples)
    device = torch.device(args.device)
    model, builder, checkpoint = build_from_checkpoint(args.checkpoint, data, args.kg, device)
    result, rows = evaluate(model, data, builder, args.split, args.query_batch_size, device, args.method)
    result.update({"checkpoint": str(args.checkpoint.resolve()), "seed": int(checkpoint["config"]["seed"]), "best_epoch": checkpoint.get("best_epoch")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
