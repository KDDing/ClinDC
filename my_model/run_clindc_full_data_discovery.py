"""Fit ClinDC on every CDCDB positive and rank a fixed discovery query."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED_EPOCHS = {0: 17, 1: 13, 2: 18, 3: 19, 4: 5, 5: 8, 6: 14, 7: 6, 8: 14, 42: 17}


def run_one(args: argparse.Namespace, seed: int) -> dict:
    run_dir = args.output_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pt"
    command = [
        sys.executable, str(HERE / "train.py"),
        "--variant", "main_no_id_without_virtual_node",
        "--triples", str(args.triples), "--kg", str(args.kg),
        "--drug-features", str(args.drug_features),
        "--output", str(run_dir), "--seed", str(seed),
        "--max-epochs", str(SEED_EPOCHS[seed]),
        "--embedding-dim", "128", "--hidden-dim", "512",
        "--negative-ratio", "30", "--evidence-weight-alpha", "0.60",
        "--learning-rate", "0.001", "--weight-decay", "0.00001",
        "--decoder-dropout-1", "0.2", "--decoder-dropout-2", "0.1",
        "--modality-dropout", "0.0", "--device", args.device,
        "--fixed-training-only",
    ]
    (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    if not (args.resume and checkpoint.is_file()):
        with (run_dir / "train.log").open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
        if completed.returncode:
            raise RuntimeError(f"training failed; see {run_dir / 'train.log'}")
    return {"seed": seed, "epochs": SEED_EPOCHS[seed], "checkpoint": str(checkpoint.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", type=Path, default=ROOT / "data/dataset_v2_ontology/drug_disease_drug_kg_mapped.csv")
    parser.add_argument("--kg", type=Path, default=ROOT / "data/kg.csv")
    parser.add_argument("--drug-features", type=Path, default=ROOT / "data/drug_structures/v2_mapped/morgan.npz")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/clindc_full_data_hctz_hypertension")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(9)) + [42])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.seeds) - set(SEED_EPOCHS))
    if unknown:
        parser.error(f"No preselected epoch count for seeds: {unknown}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {executor.submit(run_one, args, seed): seed for seed in args.seeds}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                row = {**future.result(), "status": "complete", "error": ""}
            except Exception as exc:
                row = {"seed": seed, "epochs": SEED_EPOCHS[seed], "checkpoint": "", "status": "failed", "error": str(exc)}
            rows.append(row)
            with (args.output_root / "training_status.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("seed", "epochs", "checkpoint", "status", "error"))
                writer.writeheader(); writer.writerows(sorted(rows, key=lambda x: x["seed"]))
            print(json.dumps(row), flush=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "ClinDC (main_no_id_without_virtual_node)",
        "training_protocol": "from-scratch fixed-epoch fit on union of original train/valid/test positives",
        "epoch_selection": "per-seed epoch counts previously selected using the original validation partition",
        "seed_epochs": {str(k): v for k, v in SEED_EPOCHS.items() if k in args.seeds},
        "hyperparameters": {"embedding_dim": 128, "hidden_dim": 512, "negative_ratio": 30, "evidence_weight_alpha": 0.60},
        "query": {"anchor_drug": "DB00999", "anchor_name": "hydrochlorothiazide", "disease": "hypertension"},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
