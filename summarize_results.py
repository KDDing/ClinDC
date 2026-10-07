"""Aggregate ten-seed experiment results and paired exact sign-flip tests."""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from collections import defaultdict
from pathlib import Path


def exact_sign_flip(differences: list[float]) -> float:
    observed = abs(sum(differences))
    n = len(differences)
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, differences)))
        >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=n)
    )
    return extreme / (2 ** n)


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.__getitem__)
    adjusted = {}
    running = 0.0
    size = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, values[name] * (size - index)))
        adjusted[name] = running
    return adjusted


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    with args.results.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("The results file is empty")
    metrics = [
        field for field in rows[0]
        if field in ("validation_mrr", "test1_macro_auroc", "test1_macro_aupr")
        or field.startswith(("valid_", "test_"))
    ]
    by_setting_method = defaultdict(dict)
    for row in rows:
        key = (row["setting"], row["method"])
        seed = int(row["seed"])
        if seed in by_setting_method[key]:
            parser.error(f"Duplicate seed {seed} for {key}")
        by_setting_method[key][seed] = row
    summaries = []
    for (setting, method), seed_rows in sorted(by_setting_method.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in seed_rows.values() if row.get(metric)]
            if not values:
                continue
            summaries.append({
                "setting": setting, "method": method, "metric": metric,
                "n_seeds": len(values), "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else "",
            })
    pvalues = []
    for setting in sorted({row["setting"] for row in rows}):
        clin = by_setting_method.get((setting, "ClinDC"))
        if not clin:
            continue
        competitors = {
            method: seeds for (group, method), seeds in by_setting_method.items()
            if group == setting and method != "ClinDC"
        }
        for metric in metrics:
            raw = {}
            differences = {}
            for method, other in competitors.items():
                matched = sorted(set(clin) & set(other))
                if len(matched) < 2 or any(
                    not clin[seed].get(metric) or not other[seed].get(metric)
                    for seed in matched
                ):
                    continue
                diff = [float(clin[seed][metric]) - float(other[seed][metric]) for seed in matched]
                raw[method] = exact_sign_flip(diff)
                differences[method] = (len(matched), statistics.mean(diff))
            corrected = holm(raw)
            for method in sorted(raw):
                count, mean_diff = differences[method]
                pvalues.append({
                    "setting": setting, "metric": metric, "comparison": f"ClinDC - {method}",
                    "n_paired_seeds": count, "mean_paired_difference": mean_diff,
                    "p_exact_two_sided": raw[method], "p_holm": corrected[method],
                })
    stem = args.results.with_suffix("")
    summary_path = stem.with_name(stem.name + "_summary.csv")
    pvalues_path = stem.with_name(stem.name + "_paired_tests.csv")
    write_csv(summary_path, summaries)
    write_csv(pvalues_path, pvalues)
    print(f"Wrote {summary_path}")
    if pvalues:
        print(f"Wrote {pvalues_path}")


if __name__ == "__main__":
    main()
