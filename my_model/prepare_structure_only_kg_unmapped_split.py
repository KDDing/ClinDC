"""Build a strict structure-only split from KG-unmapped drugs with Morgan data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from data import KGSignatures, TaskData


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/dataset_v2_ontology/drug_disease_drug_kg_mapped.csv")
    parser.add_argument("--kg", type=Path, default=ROOT / "data/kg.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "data/drug_structures/v2_mapped/morgan.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/dataset_v2_structure_only_kg_unmapped")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = TaskData(args.source)
    signatures = KGSignatures(data, args.kg, 131071, 256, signature_mode="full")
    archive = np.load(args.features, allow_pickle=False)
    validity = {str(drug): bool(valid) for drug, valid in zip(archive["drug_ids"], archive["valid"])}
    cold = {
        data.idx_to_drug[index]
        for index, tokens in enumerate(signatures.drug_tokens)
        if not any(int(token) != 0 for token in tokens)
        and validity.get(data.idx_to_drug[index], False)
    }
    with args.source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    warm_rows, test1_rows, test2_rows = [], [], []
    for original in rows:
        row = dict(original)
        c1, c2 = row["drug_1"] in cold, row["drug_2"] in cold
        if c1 and c2:
            row["drug_1"], row["drug_2"] = sorted((row["drug_1"], row["drug_2"]))
            row["split"] = "test2"
            test2_rows.append(row)
        elif c1 or c2:
            if c1:  # Always orient Test-1 as warm anchor -> cold target.
                row["drug_1"], row["drug_2"] = row["drug_2"], row["drug_1"]
            row["split"] = "test1"
            test1_rows.append(row)
        else:
            warm_rows.append(row)

    rng = random.Random(args.seed)
    order = list(range(len(warm_rows)))
    rng.shuffle(order)
    drug_counts = Counter(drug for row in warm_rows for drug in (row["drug_1"], row["drug_2"]))
    disease_counts = Counter(row["disease"] for row in warm_rows)
    target_valid = round(len(warm_rows) * args.validation_fraction)
    valid_indices = set()
    for index in order:
        if len(valid_indices) >= target_valid:
            break
        row = warm_rows[index]
        d1, d2, disease = row["drug_1"], row["drug_2"], row["disease"]
        if drug_counts[d1] <= 1 or drug_counts[d2] <= 1 or disease_counts[disease] <= 1:
            continue
        valid_indices.add(index)
        drug_counts[d1] -= 1
        drug_counts[d2] -= 1
        disease_counts[disease] -= 1

    train_rows, valid_rows = [], []
    for index, row in enumerate(warm_rows):
        row["split"] = "valid" if index in valid_indices else "train"
        (valid_rows if index in valid_indices else train_rows).append(row)

    train_diseases = {row["disease"] for row in train_rows}
    excluded_test_only_disease = [
        row for row in test1_rows + test2_rows if row["disease"] not in train_diseases
    ]
    for row in excluded_test_only_disease:
        row["split"] = "excluded"
    test1_rows = [row for row in test1_rows if row["disease"] in train_diseases]
    test2_rows = [row for row in test2_rows if row["disease"] in train_diseases]
    output_rows = train_rows + valid_rows + test1_rows + test2_rows + excluded_test_only_disease

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "drug_disease_drug.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    with (args.output / "cold_drugs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["drug_id", "morgan_valid", "drug_kg_masked"])
        writer.writeheader()
        writer.writerows(
            {"drug_id": drug, "morgan_valid": 1, "drug_kg_masked": 1}
            for drug in sorted(cold)
        )

    train_drugs = {drug for row in train_rows for drug in (row["drug_1"], row["drug_2"])}
    report = {
        "definition": "all drugs with valid Morgan fingerprint and no effective non-padding PrimeKG one-hop token are fixed cold drugs",
        "seed": args.seed,
        "validation_fraction_of_warm_only_triples": args.validation_fraction,
        "num_cold_drugs": len(cold),
        "num_train_drugs": len(train_drugs),
        "counts": {
            "source": len(rows), "train": len(train_rows), "valid": len(valid_rows),
            "test1_exactly_one_cold": len(test1_rows), "test2_both_cold": len(test2_rows),
            "excluded_test_only_disease": len(excluded_test_only_disease),
        },
        "integrity": {
            "cold_absent_from_train_positives": not any(drug in train_drugs for drug in cold),
            "validation_contains_only_train_drugs": all(
                row["drug_1"] in train_drugs and row["drug_2"] in train_drugs for row in valid_rows
            ),
            "test_diseases_seen_in_train": all(
                row["disease"] in train_diseases for row in test1_rows + test2_rows
            ),
            "test1_exactly_one_cold": all(
                (row["drug_1"] in cold) + (row["drug_2"] in cold) == 1 for row in test1_rows
            ),
            "test2_both_cold": all(
                row["drug_1"] in cold and row["drug_2"] in cold for row in test2_rows
            ),
        },
        "training_negative_pool": "training-positive drugs only; enforced by train CLI flag",
        "test1_orientation": "warm drug is drug_1 anchor; cold drug is drug_2 target",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
