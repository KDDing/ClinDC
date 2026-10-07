"""Full filtered candidate ranking for a selected validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import TaskData
from evaluate_checkpoint_subgroups import build_from_checkpoint
from train import evaluate_full_candidates


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--triples", type=Path, required=True)
    parser.add_argument("--kg", type=Path, default=ROOT / "data/kg.csv")
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    data = TaskData(args.triples)
    device = torch.device(args.device)
    model, builder, checkpoint = build_from_checkpoint(
        args.checkpoint, data, args.kg, device
    )
    metrics = evaluate_full_candidates(
        model, data.splits[args.split], data, builder,
        args.query_batch_size, device,
        stage_name=f"{args.method} {args.split}",
    )
    result = {
        "method": args.method,
        "split": args.split,
        "seed": checkpoint["config"]["seed"],
        "best_epoch": checkpoint.get("best_epoch"),
        "candidate_protocol": "full filtered ranking",
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
