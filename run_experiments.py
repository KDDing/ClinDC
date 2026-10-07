"""Reproduce the paper's ClinDC sensitivity, ablation, and cold-drug runs.

This entry point trains every model from scratch. It never reuses a checkpoint
from the historical ``main_no_id`` model, which included a virtual node.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "my_model"
DATA = ROOT / "data"
SEEDS = (*range(9), 42)
FINAL_VARIANT = "main_no_id_without_virtual_node"
ABLATIONS = {
    "ClinDC": FINAL_VARIANT,
    "w_o_Morgan": "clindc_without_morgan",
    "w_o_KG": "clindc_without_all_kg",
    "w_o_disease_lexical": "clindc_disease_kg_only",
    "ordered_bidirectional": "clindc_ordered_bidirectional",
}
STRUCTURE_FREE = {"w_o_Morgan"}
TRIPLES = {
    "chronological": DATA / "dataset_v2_ontology/drug_disease_drug_kg_mapped.csv",
    "ontology": DATA / "dataset_v2_ontology_aware_ldo/drug_disease_drug_kg_mapped.csv",
    "cold": DATA / "dataset_v2_structure_only_kg_unmapped/drug_disease_drug.csv",
}
KG = DATA / "kg.csv"
MORGAN = DATA / "drug_structures/v2_mapped/morgan.npz"
COLD_IDS = DATA / "dataset_v2_structure_only_kg_unmapped/cold_drugs.csv"


def experiments(stage: str, settings: list[str], seeds: list[int], names: list[str] | None):
    if stage == "dimensions":
        # One-factor-at-a-time, not a 4 x 4 Cartesian grid.
        configs = {(128, 512)}
        configs.update((value, 512) for value in (64, 128, 256, 512))
        configs.update((128, value) for value in (128, 256, 512, 1024))
        for embedding, hidden in sorted(configs):
            label = f"embedding_{embedding}_hidden_{hidden}"
            if names and label not in names:
                continue
            for seed in seeds:
                yield "chronological", label, seed, ("--variant", FINAL_VARIANT), embedding, hidden
        return
    for setting in settings:
        methods = (
            {"ClinDC": ("--variant", FINAL_VARIANT)}
            if stage == "clindc"
            else {name: ("--variant", variant) for name, variant in ABLATIONS.items()}
            if stage == "ablations"
            else {"ClinDC": ("--variant", FINAL_VARIANT), "w_o_Morgan": ("--variant", "clindc_without_morgan")}
        )
        for label, selector in methods.items():
            if names and label not in names:
                continue
            for seed in seeds:
                yield setting, label, seed, selector, 128, 512


def run_command(command: list[str], log: Path, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        finished = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if finished.returncode:
        raise RuntimeError(f"Command failed ({finished.returncode}): {log}")


def one_experiment(args, spec) -> dict | None:
    setting, label, seed, selector, embedding, hidden = spec
    run_dir = args.output / args.stage / setting / label / f"seed_{seed}"
    checkpoint = run_dir / "best_model.pt"
    metrics_path = run_dir / "metrics.json"
    train = [
        sys.executable, str(CODE / "train.py"), *selector,
        "--triples", str(TRIPLES[setting]), "--kg", str(KG),
        "--output", str(run_dir), "--seed", str(seed),
        "--max-epochs", "30", "--embedding-dim", str(embedding),
        "--hidden-dim", str(hidden), "--negative-ratio", "30",
        "--evidence-weight-alpha", "0.60", "--learning-rate", "0.001",
        "--weight-decay", "0.00001", "--patience", "5",
        "--decoder-dropout-1", "0.2", "--decoder-dropout-2", "0.1",
        "--modality-dropout", "0.0", "--evaluation-mode", "full",
        "--full-eval-query-batch-size", str(args.query_batch_size),
        "--device", args.device, "--validation-only",
    ]
    if label not in STRUCTURE_FREE:
        train += ["--drug-features", str(MORGAN)]
    if args.stage == "cold":
        train += ["--train-negatives-train-drugs-only", "--validation-candidates-train-drugs-only"]
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "train_command.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    if not (args.resume and checkpoint.is_file() and metrics_path.is_file()):
        run_command(train, run_dir / "train.log", args.dry_run)
    if args.dry_run:
        return None
    train_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if train_metrics.get("test_evaluated") is not False:
        raise RuntimeError(f"Training unexpectedly accessed the test split: {metrics_path}")
    if args.stage == "cold":
        output = run_dir / "cold_evaluation.json"
        command = [
            sys.executable, str(CODE / "evaluate_cold_candidate_only.py"),
            "--checkpoint", str(checkpoint), "--method", label,
            "--triples", str(TRIPLES[setting]), "--cold-drugs", str(COLD_IDS),
            "--kg", str(KG), "--output", str(output),
            "--query-batch-size", str(args.query_batch_size), "--device", args.device,
        ]
        if not (args.resume and output.is_file()):
            run_command(command, run_dir / "cold_evaluation.log", False)
        cold = json.loads(output.read_text(encoding="utf-8"))
        return {
            "stage": args.stage, "setting": setting, "method": label, "seed": seed,
            "validation_mrr": train_metrics["best_valid_mrr"],
            "test1_macro_auroc": cold["macro"]["macro_auroc"],
            "test1_macro_aupr": cold["macro"]["macro_aupr"],
        }
    split = "valid" if args.stage == "dimensions" else "test"
    ranking_path = run_dir / f"{split}_ranking.json"
    macro_path = run_dir / f"{split}_macro.json"
    rank_command = [
        sys.executable, str(CODE / "evaluate_checkpoint_ranking.py"),
        "--checkpoint", str(checkpoint), "--triples", str(TRIPLES[setting]),
        "--kg", str(KG), "--split", split, "--method", label,
        "--output", str(ranking_path), "--query-batch-size", str(args.query_batch_size),
        "--device", args.device,
    ]
    macro_command = [
        sys.executable, str(CODE / "evaluate_checkpoint_macro_auc_aupr.py"),
        "--checkpoint", str(checkpoint), "--triples", str(TRIPLES[setting]),
        "--kg", str(KG), "--split", split, "--method", label,
        "--output", str(macro_path), "--query-batch-size", str(args.query_batch_size),
        "--device", args.device,
    ]
    if not (args.resume and ranking_path.is_file()):
        run_command(rank_command, run_dir / f"{split}_ranking.log", False)
    if not (args.resume and macro_path.is_file()):
        run_command(macro_command, run_dir / f"{split}_macro.log", False)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))["metrics"]
    macro = json.loads(macro_path.read_text(encoding="utf-8"))
    return {
        "stage": args.stage, "setting": setting, "method": label, "seed": seed,
        "embedding_dim": embedding, "hidden_dim": hidden,
        "validation_mrr": train_metrics["best_valid_mrr"],
        f"{split}_mrr": ranking["mrr"],
        f"{split}_hits@1": ranking["hits@1"],
        f"{split}_hits@3": ranking["hits@3"],
        f"{split}_hits@10": ranking["hits@10"],
        f"{split}_macro_auroc": macro["macro_auroc"],
        f"{split}_macro_aupr": macro["macro_aupr"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dimensions", "clindc", "ablations", "cold"))
    parser.add_argument("--setting", choices=("chronological", "ontology", "both"), default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--names", nargs="+", help="Optional exact model/configuration names")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction")
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    settings = (["chronological", "ontology"] if args.setting == "both" else [args.setting])
    if args.stage == "dimensions":
        settings = ["chronological"]
    if args.stage == "cold":
        settings = ["cold"]
    required = [KG, MORGAN, *(TRIPLES[setting] for setting in settings)]
    if args.stage == "cold":
        required.append(COLD_IDS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing packaged inputs: " + ", ".join(missing))
    specs = list(experiments(args.stage, settings, args.seeds, args.names))
    if not specs:
        parser.error("No experiment matched --names")
    print(f"Planned runs: {len(specs)}; final ClinDC variant: {FINAL_VARIANT}")
    rows = []
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec[:3]}", flush=True)
        result = one_experiment(args, spec)
        if result is not None:
            rows.append(result)
            output = args.output / args.stage / "results.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if args.dry_run:
        print("Dry run complete; no training or evaluation was performed.")


if __name__ == "__main__":
    main()
