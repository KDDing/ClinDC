from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from ablations import ABLATIONS
from data import BatchBuilder, KGSignatures, TaskData, batches
from model import DrugDiseaseDrugModel


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def move(batch, device):
    result = {}
    for key, value in batch.items():
        if isinstance(value, tuple):
            result[key] = tuple(x.to(device) for x in value)
        else:
            result[key] = value.to(device)
    return result


def average_precision(labels, scores):
    ranked = sorted(zip(scores, labels), reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hit = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            hit += 1
            total += hit / rank
    return total / positives


@torch.no_grad()
def evaluate_sampled(
    model,
    examples,
    data,
    builder,
    batch_size,
    eval_negatives,
    device,
    stage_name="evaluation",
    eval_seed=1042,
):
    model.eval()
    log(
        f"Starting {stage_name}: {len(examples):,} positives, "
        f"{eval_negatives} sampled negatives per positive"
    )
    started = time.perf_counter()
    old_ratio = builder.negative_ratio
    old_rng_state = builder.rng.getstate()
    builder.rng.seed(eval_seed)
    builder.negative_ratio = eval_negatives
    reciprocal_ranks, scores, labels = [], [], []
    stratum_rr = collections.defaultdict(list)
    for start in range(0, len(examples), batch_size):
        positives = examples[start : start + batch_size]
        # Candidate groups are emitted in positive-then-negatives order. Score
        # the whole chunk in one GPU call while retaining per-query ranks.
        batch, y = builder.build(positives, include_negatives=True)
        group_size = eval_negatives + 1
        logits = (
            model(move(batch, device))
            .cpu()
            .reshape(len(positives), group_size)
        )
        for row_index, positive in enumerate(positives):
            row_scores = logits[row_index]
            positive_score = row_scores[0]
            rank = 1 + int((row_scores[1:] > positive_score).sum().item())
            rr = 1.0 / rank
            reciprocal_ranks.append(rr)
            for stratum in data.strata(positive):
                stratum_rr[stratum].append(rr)
        scores.extend(logits.flatten().tolist())
        labels.extend(y.tolist())
        completed = min(start + len(positives), len(examples))
        if completed % 500 == 0 or completed == len(examples):
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed else 0.0
            log(
                f"{stage_name} progress: {completed:,}/{len(examples):,} "
                f"({100 * completed / len(examples):.1f}%), "
                f"{rate:.1f} positives/s"
            )
    builder.negative_ratio = old_ratio
    builder.rng.setstate(old_rng_state)
    metrics = {
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "hits@1": sum(x == 1.0 for x in reciprocal_ranks) / len(reciprocal_ranks),
        "hits@3": sum(x >= 1 / 3 for x in reciprocal_ranks) / len(reciprocal_ranks),
        "hits@10": sum(x >= 0.1 for x in reciprocal_ranks) / len(reciprocal_ranks),
        "auprc_sampled": average_precision(labels, scores),
    }
    metrics["strata"] = {
        name: {
            "n": len(values),
            "mrr": sum(values) / len(values),
            "hits@1": sum(x == 1.0 for x in values) / len(values),
            "hits@3": sum(x >= 1 / 3 for x in values) / len(values),
            "hits@10": sum(x >= 0.1 for x in values) / len(values),
        }
        for name, values in sorted(stratum_rr.items())
        if values
    }
    log(f"Finished {stage_name} in {time.perf_counter() - started:.1f}s")
    return metrics


@torch.no_grad()
def evaluate_full_candidates(
    model,
    examples,
    data,
    builder,
    query_batch_size,
    device,
    stage_name="evaluation",
    candidate_pool=None,
    tie_policy="optimistic",
):
    """Filtered ranking against every eligible drug in the dataset vocabulary."""
    model.eval()
    log(
        f"Starting {stage_name}: {len(examples):,} positives, full filtered "
        f"ranking over {len(data.idx_to_drug):,} drugs"
    )
    started = time.perf_counter()
    reciprocal_ranks = []
    stratum_rr = collections.defaultdict(list)
    candidate_counts = []
    for start in range(0, len(examples), query_batch_size):
        positives = examples[start : start + query_batch_size]
        explicit_examples = []
        offsets = [0]
        for positive in positives:
            # Put the target first, then score every drug that is neither the
            # anchor nor another known positive for this disease-conditioned
            # unordered pair query.
            group = [positive]
            for candidate in (
                range(len(data.idx_to_drug))
                if candidate_pool is None
                else candidate_pool
            ):
                if candidate in (positive.drug_1, positive.drug_2):
                    continue
                key = data.canonical_key(
                    positive.drug_1, positive.disease, candidate
                )
                if key in data.all_positive:
                    continue
                group.append(
                    type(positive)(
                        positive.drug_1,
                        positive.disease,
                        candidate,
                        1,
                    )
                )
            explicit_examples.extend(group)
            offsets.append(len(explicit_examples))
            candidate_counts.append(len(group))

        logits = model(
            move(builder.collate(explicit_examples), device)
        ).cpu()
        for row_index, positive in enumerate(positives):
            row_scores = logits[offsets[row_index] : offsets[row_index + 1]]
            positive_score = row_scores[0]
            greater = int((row_scores[1:] > positive_score).sum().item())
            if tie_policy == "average":
                equal = int((row_scores[1:] == positive_score).sum().item())
                rank = 1.0 + greater + 0.5 * equal
            elif tie_policy == "optimistic":
                rank = 1 + greater
            else:
                raise ValueError(f"Unknown tie policy: {tie_policy}")
            rr = 1.0 / rank
            reciprocal_ranks.append(rr)
            for stratum in data.strata(positive):
                stratum_rr[stratum].append(rr)

        completed = min(start + len(positives), len(examples))
        if completed % 250 == 0 or completed == len(examples):
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed else 0.0
            log(
                f"{stage_name} progress: {completed:,}/{len(examples):,} "
                f"({100 * completed / len(examples):.1f}%), "
                f"{rate:.1f} positives/s"
            )

    metrics = {
        "candidate_protocol": "full_filtered",
        "tie_policy": tie_policy,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "hits@1": sum(x == 1.0 for x in reciprocal_ranks) / len(reciprocal_ranks),
        "hits@3": sum(x >= 1 / 3 for x in reciprocal_ranks) / len(reciprocal_ranks),
        "hits@10": sum(x >= 0.1 for x in reciprocal_ranks) / len(reciprocal_ranks),
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "candidate_count_mean": sum(candidate_counts) / len(candidate_counts),
    }
    metrics["strata"] = {
        name: {
            "n": len(values),
            "mrr": sum(values) / len(values),
            "hits@1": sum(x == 1.0 for x in values) / len(values),
            "hits@3": sum(x >= 1 / 3 for x in values) / len(values),
            "hits@10": sum(x >= 0.1 for x in values) / len(values),
        }
        for name, values in sorted(stratum_rr.items())
        if values
    }
    log(f"Finished {stage_name} in {time.perf_counter() - started:.1f}s")
    return metrics


def evaluate(
    model,
    examples,
    data,
    builder,
    batch_size,
    eval_negatives,
    device,
    stage_name="evaluation",
    eval_seed=1042,
    evaluation_mode="full",
    full_eval_query_batch_size=16,
    candidate_pool=None,
):
    if evaluation_mode == "full":
        return evaluate_full_candidates(
            model,
            examples,
            data,
            builder,
            full_eval_query_batch_size,
            device,
            stage_name=stage_name,
            candidate_pool=candidate_pool,
        )
    return evaluate_sampled(
        model,
        examples,
        data,
        builder,
        batch_size,
        eval_negatives,
        device,
        stage_name=stage_name,
        eval_seed=eval_seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triples",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed_cdcdb"
        / "drug_disease_drug.csv",
    )
    parser.add_argument("--kg", type=Path, default=PROJECT_ROOT / "data" / "kg.csv")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "PyTorch device. Defaults to CUDA and fails explicitly when CUDA "
            "is unavailable; pass --device cpu only for an intentional CPU run."
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--variant", choices=sorted(ABLATIONS),
        default="main_no_id_without_virtual_node",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--eval-negatives", type=int)
    parser.add_argument(
        "--evaluation-mode",
        choices=("full", "sampled"),
        default="full",
        help="Use full filtered candidate ranking (default) or sampled ranking.",
    )
    parser.add_argument("--full-eval-query-batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--negative-ratio", type=int)
    parser.add_argument(
        "--train-negatives-train-drugs-only",
        action="store_true",
        help="Restrict corrupting drugs to identities observed in training positives.",
    )
    parser.add_argument(
        "--validation-candidates-train-drugs-only",
        action="store_true",
        help="Exclude held-out drug identities from validation candidate ranking.",
    )
    parser.add_argument("--evidence-weight-alpha", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--decoder-dropout-1", type=float)
    parser.add_argument("--decoder-dropout-2", type=float)
    parser.add_argument("--modality-dropout", type=float)
    parser.add_argument(
        "--lr-scheduler", choices=("none", "plateau"), default="none"
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Stop after model selection on the validation set. This prevents "
            "grid searches from evaluating the test partition."
        ),
    )
    parser.add_argument(
        "--fixed-training-only",
        action="store_true",
        help=(
            "Train from scratch for exactly --max-epochs on the union of all "
            "positive splits, save the final checkpoint, and skip validation/test. "
            "Use only after the epoch count and hyperparameters were selected "
            "without consulting the final fit data."
        ),
    )
    parser.add_argument(
        "--drug-features",
        type=Path,
        help="Aligned NPZ molecular features; required by structure variants.",
    )
    args = parser.parse_args()
    if args.fixed_training_only and args.validation_only:
        parser.error("--fixed-training-only and --validation-only are mutually exclusive")
    log("Starting drug-disease-drug training pipeline")
    log(f"Script directory: {SCRIPT_DIR}")
    log(f"Triples file: {args.triples.resolve()}")
    log(f"KG file: {args.kg.resolve()}")
    log(f"Config file: {args.config.resolve()}")
    log(f"Output directory: {args.output.resolve()}")
    for label, path in (
        ("config", args.config),
        ("triples", args.triples),
        ("kg", args.kg),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path.resolve()}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_epochs is not None:
        cfg["epochs"] = args.max_epochs
    if args.eval_negatives is not None:
        cfg["eval_negatives"] = args.eval_negatives
    if args.embedding_dim is not None:
        cfg["embedding_dim"] = args.embedding_dim
    if args.hidden_dim is not None:
        cfg["hidden_dim"] = args.hidden_dim
    if args.negative_ratio is not None:
        cfg["negative_ratio"] = args.negative_ratio
    if args.evidence_weight_alpha is not None:
        cfg["evidence_weight_alpha"] = args.evidence_weight_alpha
    for argument, key in (
        (args.learning_rate, "learning_rate"),
        (args.weight_decay, "weight_decay"),
        (args.patience, "patience"),
        (args.decoder_dropout_1, "decoder_dropout_1"),
        (args.decoder_dropout_2, "decoder_dropout_2"),
        (args.modality_dropout, "modality_dropout"),
    ):
        if argument is not None:
            cfg[key] = argument
    cfg.setdefault("decoder_dropout_1", 0.2)
    cfg.setdefault("decoder_dropout_2", 0.1)
    cfg.setdefault("modality_dropout", 0.0)
    cfg["lr_scheduler"] = args.lr_scheduler
    cfg["evaluation_mode"] = args.evaluation_mode
    cfg["full_eval_query_batch_size"] = args.full_eval_query_batch_size
    for key in ("embedding_dim", "hidden_dim", "negative_ratio"):
        if cfg[key] <= 0:
            parser.error(f"{key} must be positive, got {cfg[key]}")
    if cfg["evidence_weight_alpha"] < 0:
        parser.error(
            "evidence_weight_alpha must be non-negative, "
            f"got {cfg['evidence_weight_alpha']}"
        )
    if cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0 or cfg["patience"] <= 0:
        parser.error("learning rate/patience must be positive and weight decay non-negative")
    for key in ("decoder_dropout_1", "decoder_dropout_2", "modality_dropout"):
        if not 0 <= cfg[key] < 1:
            parser.error(f"{key} must be in [0, 1), got {cfg[key]}")
    if cfg["full_eval_query_batch_size"] <= 0:
        parser.error("--full-eval-query-batch-size must be positive")
    variant = ABLATIONS[args.variant]
    uses_structure = "structure" in variant["drug_mode"]
    if uses_structure and args.drug_features is None:
        parser.error(f"--drug-features is required for variant {args.variant}")
    if args.drug_features is not None and not args.drug_features.is_file():
        parser.error(f"drug feature file does not exist: {args.drug_features.resolve()}")
    log("Configuration loaded")
    log(f"ClinDC variant: {args.variant} - {json.dumps(variant)}")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            "CUDA was requested but is unavailable. Run outside GPU-isolated "
            "sandboxes or pass --device cpu only for an intentional CPU run."
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        device_description = torch.cuda.get_device_name(device)
    else:
        device_description = f"CPU ({torch.get_num_threads()} PyTorch threads)"
    log(f"PyTorch version: {torch.__version__}")
    log(f"Running on device: {device} - {device_description}")
    log(f"Smoke-test mode: {'ON' if args.smoke_test else 'OFF'}")

    started = time.perf_counter()
    log("Loading processed CDCDB triples")
    data = TaskData(args.triples)
    log(
        f"Triples loaded in {time.perf_counter() - started:.1f}s: "
        f"{len(data.splits['train']):,} train, "
        f"{len(data.splits['valid']):,} valid, "
        f"{len(data.splits['test']):,} test; "
        f"{len(data.idx_to_drug):,} drugs, "
        f"{len(data.idx_to_disease):,} diseases"
    )
    started = time.perf_counter()
    log("Scanning kg.csv and building bounded one-hop entity signatures")
    kg = KGSignatures(
        data,
        args.kg,
        cfg["kg_hash_buckets"],
        cfg["max_kg_tokens_per_entity"],
        progress_callback=lambda rows: log(f"KG scan progress: {rows:,} rows"),
        signature_mode=variant["kg_signature_mode"],
    )
    log(f"KG signatures built in {time.perf_counter() - started:.1f}s")
    log("Building disease text features and negative sampler")
    builder = BatchBuilder(
        data,
        kg,
        cfg["text_hash_buckets"],
        cfg["negative_ratio"],
        cfg["seed"],
        drug_features_path=args.drug_features,
        negative_drug_pool=(
            sorted(data.train_drugs)
            if args.train_negatives_train_drugs_only
            else None
        ),
    )
    cfg["train_negatives_train_drugs_only"] = bool(
        args.train_negatives_train_drugs_only
    )
    cfg["validation_candidates_train_drugs_only"] = bool(
        args.validation_candidates_train_drugs_only
    )
    if uses_structure:
        coverage = float(builder.drug_feature_valid.mean().item())
        log(
            f"Drug structural features: {args.drug_features.resolve()}, "
            f"dimension={builder.drug_feature_dim}, coverage={coverage:.2%}, "
            f"metadata={json.dumps(builder.drug_feature_metadata, sort_keys=True)}"
        )
    log("Initializing model")
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
            decoder_dropout_1=cfg["decoder_dropout_1"],
            decoder_dropout_2=cfg["decoder_dropout_2"],
            modality_dropout=cfg["modality_dropout"],
            pair_modalities=variant.get("pair_modalities", ("q", "m", "g")),
            pair_fusion=variant.get("pair_fusion", "attention"),
            pair_include_virtual=variant.get("pair_include_virtual", True),
            pair_attention_disease=variant.get("pair_attention_disease", True),
            pair_representation=variant.get("pair_representation", "symmetric"),
            bidirectional_inference=variant.get(
                "bidirectional_inference", False
            ),
        ).to(device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    log(f"Model initialized with {trainable_parameters:,} trainable parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
        )
        if cfg["lr_scheduler"] == "plateau"
        else None
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    args.output.mkdir(parents=True, exist_ok=True)
    train_examples = data.splits["train"]
    valid_examples = data.splits["valid"]
    if args.fixed_training_only:
        train_examples = [
            example
            for split_examples in data.splits.values()
            for example in split_examples
        ]
        log(
            "Fixed full-data fit enabled: training on the union of all splits "
            f"({len(train_examples):,} positives) for exactly {cfg['epochs']} epochs"
        )
    if args.smoke_test:
        train_examples = train_examples[:64]
        valid_examples = valid_examples[:8]
        cfg["epochs"] = 1
        cfg["eval_negatives"] = 5
        log("Smoke-test limits applied: 64 train, 8 valid, 8 test, 1 epoch")

    best_mrr, best_epoch, stale = -1.0, None, 0
    train_rng = random.Random(cfg["seed"])
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_started = time.perf_counter()
        expected_batches = math.ceil(len(train_examples) / cfg["batch_size"])
        log(
            f"Epoch {epoch}/{cfg['epochs']} started "
            f"({expected_batches:,} positive batches)"
        )
        model.train()
        total_loss = total_items = 0
        for batch_number, positives in enumerate(
            batches(train_examples, cfg["batch_size"], train_rng), 1
        ):
            batch, labels = builder.build(positives)
            evidence = batch["evidence"].to(device)
            labels = labels.to(device)
            logits = model(move(batch, device))
            raw_loss = loss_fn(logits, labels)
            weights = torch.ones_like(raw_loss)
            positive_mask = labels == 1
            if variant["use_evidence_weight"]:
                weights[positive_mask] += cfg["evidence_weight_alpha"] * torch.log1p(
                    evidence[positive_mask]
                ).clamp(max=2.0)
            loss = (raw_loss * weights).mean()
            if hasattr(model, "regularization_loss"):
                loss = loss + model.regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_items += len(labels)
            if batch_number % 25 == 0 or batch_number == expected_batches:
                log(
                    f"Epoch {epoch} training progress: "
                    f"{batch_number:,}/{expected_batches:,} batches, "
                    f"running loss={total_loss / total_items:.6f}"
                )

        if args.fixed_training_only:
            log(
                f"Epoch {epoch} finished in {time.perf_counter() - epoch_started:.1f}s; "
                "validation intentionally skipped for the full-data fit"
            )
            continue

        metrics = evaluate(
            model,
            valid_examples,
            data,
            builder,
            cfg["batch_size"],
            cfg["eval_negatives"],
            device,
            stage_name=f"validation epoch {epoch}",
            eval_seed=cfg["seed"] + 10_000,
            evaluation_mode=cfg["evaluation_mode"],
            full_eval_query_batch_size=cfg["full_eval_query_batch_size"],
            candidate_pool=(
                sorted(data.train_drugs)
                if args.validation_candidates_train_drugs_only
                else None
            ),
        )
        log(
            "Epoch result: "
            + json.dumps(
                {"epoch": epoch, "loss": total_loss / total_items, **metrics}
            )
        )
        log(f"Epoch {epoch} finished in {time.perf_counter() - epoch_started:.1f}s")
        if scheduler is not None:
            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(metrics["mrr"])
            current_lr = optimizer.param_groups[0]["lr"]
            log(f"Learning rate after scheduler: {current_lr:.8g}")
            if current_lr < previous_lr:
                log(f"Plateau scheduler reduced LR from {previous_lr:.8g}")
        if metrics["mrr"] > best_mrr:
            best_mrr, best_epoch, stale = metrics["mrr"], epoch, 0
            log(f"New best validation MRR={best_mrr:.6f}; saving checkpoint")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "variant_name": args.variant,
                    "variant": variant,
                    "best_epoch": best_epoch,
                    "drug_to_idx": data.drug_to_idx,
                    "disease_to_idx": data.disease_to_idx,
                    "drug_features": (
                        str(args.drug_features.resolve())
                        if args.drug_features is not None
                        else None
                    ),
                    "drug_feature_metadata": builder.drug_feature_metadata,
                },
                args.output / "best_model.pt",
            )
        else:
            stale += 1
            log(
                f"Validation MRR did not improve; "
                f"early-stopping counter={stale}/{cfg['patience']}"
            )
            if stale >= cfg["patience"]:
                log("Early stopping triggered")
                break

    if args.fixed_training_only:
        best_epoch = cfg["epochs"]
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg,
                "variant_name": args.variant,
                "variant": variant,
                "best_epoch": best_epoch,
                "drug_to_idx": data.drug_to_idx,
                "disease_to_idx": data.disease_to_idx,
                "drug_features": (
                    str(args.drug_features.resolve())
                    if args.drug_features is not None
                    else None
                ),
                "drug_feature_metadata": builder.drug_feature_metadata,
                "training_protocol": "fixed_epoch_all_positive_splits",
                "num_training_positives": len(train_examples),
            },
            args.output / "best_model.pt",
        )
        best_mrr = None

    result_common = {
        "variant": args.variant,
        "seed": cfg["seed"],
        "best_valid_mrr": best_mrr,
        "best_epoch": best_epoch,
        "training_protocol": (
            "fixed_epoch_all_positive_splits"
            if args.fixed_training_only
            else "validation_selected"
        ),
        "num_training_positives": len(train_examples),
        "config": cfg,
        "drug_features": (
            str(args.drug_features.resolve())
            if args.drug_features is not None
            else None
        ),
        "drug_feature_metadata": builder.drug_feature_metadata,
        "drug_structure_coverage": (
            float(builder.drug_feature_valid.mean().item())
            if builder.drug_feature_valid is not None
            else None
        ),
    }
    if args.fixed_training_only:
        (args.output / "metrics.json").write_text(
            json.dumps({**result_common, "validation_evaluated": False, "test_evaluated": False}, indent=2),
            encoding="utf-8",
        )
        log(
            "Fixed full-data fit completed; final checkpoint saved after "
            f"epoch {best_epoch}"
        )
        return
    if args.validation_only:
        (args.output / "metrics.json").write_text(
            json.dumps({**result_common, "test_evaluated": False}, indent=2),
            encoding="utf-8",
        )
        log(
            "Validation-only run completed; test partition was not evaluated. "
            f"Best validation MRR={best_mrr:.6f} at epoch {best_epoch}"
        )
        return

    # The checkpoint is created locally by this training run and also contains
    # dictionaries/configuration in addition to tensor weights.
    log("Loading best checkpoint for final test evaluation")
    checkpoint = torch.load(
        args.output / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    test_examples = data.splits["test"][:8] if args.smoke_test else data.splits["test"]
    test_metrics = evaluate(
        model,
        test_examples,
        data,
        builder,
        cfg["batch_size"],
        cfg["eval_negatives"],
        device,
        stage_name="test",
        eval_seed=cfg["seed"] + 20_000,
        evaluation_mode=cfg["evaluation_mode"],
        full_eval_query_batch_size=cfg["full_eval_query_batch_size"],
    )
    (args.output / "metrics.json").write_text(
        json.dumps(
            {**result_common, "test_evaluated": True, "test": test_metrics},
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Metrics saved to {(args.output / 'metrics.json').resolve()}")
    log("Test result: " + json.dumps({"test": test_metrics}))
    log("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
