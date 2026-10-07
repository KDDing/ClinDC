#!/usr/bin/env python3
"""HIPPIE PPI analysis for ClinDC's filtered essential-HTN Top 5.

The disease module is the user-provided DisGeNET essential-hypertension gene
set (ScoreGDA >= 0.5), drug targets are the same STITCH medium-confidence
targets used by the companion enrichment analysis, and the PPI
network is the complete HIPPIE graph retained at the configured confidence
threshold.  Crucially, the script separates pair-level proximity from the
candidate's incremental contribution beyond hydrochlorothiazide (HCTZ).
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ppi_utils import (
    degree_bins,
    load_independent_ppi,
    multisource_distances,
)
from analyze_case_study_enrichment import ANCHOR, DRUGS, bh_adjust, load_top5


ROOT = Path(__file__).resolve().parents[1]
KEGG_OUT = ROOT / "outputs" / "clindc_essential_hypertension_top5_enrichment"
RANKING = (
    ROOT / "outputs" / "clindc_full_data_hctz_essential_hypertension_tiered_novelty"
    / "top30_consensus_target_network.csv"
)
HIPPIE = (
    ROOT / "data" / "external_hypertension_validation" / "raw"
    / "hippie_v2.4_2026-04-09.txt.gz"
)
OUT = ROOT / "outputs" / "clindc_essential_hypertension_top5_hippie"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer fields for empty output: {path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_targets(path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    targets: dict[str, set[str]] = collections.defaultdict(set)
    symbols: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gene = row["entrez_id"]
            targets[row["drugbank_id"]].add(gene)
            symbols[gene] = row["gene_symbol"]
    return targets, symbols


def load_disease_genes(path: Path) -> tuple[set[str], dict[str, str]]:
    genes, symbols = set(), {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gene = row["entrez_id"]
            genes.add(gene)
            symbols[gene] = row["gene_symbol"]
    return genes, symbols


def mean_distance(targets: set[str], distances: dict[str, int]) -> float:
    values = [distances[target] for target in targets if target in distances]
    return float(np.mean(values)) if values else math.nan


def sample_degree_matched(
    template: set[str],
    node_bin: dict[str, int],
    pools: dict[int, list[str]],
    rng: np.random.Generator,
    excluded: set[str] | None = None,
) -> set[str]:
    """Sample distinct nodes while matching every template node's degree bin."""
    sampled: set[str] = set()
    excluded = set() if excluded is None else set(excluded)
    for target in sorted(template):
        pool = pools[node_bin[target]]
        chosen = None
        # Rejection sampling avoids rebuilding a degree-bin-sized list for every
        # target in every permutation. Degree bins are much larger than a drug's
        # target set, so collisions are rare.
        for _ in range(100):
            node = pool[int(rng.integers(len(pool)))]
            if node not in sampled and node not in excluded:
                chosen = node
                break
        if chosen is None:
            chosen = next(
                (node for node in pool if node not in sampled and node not in excluded),
                None,
            )
        if chosen is None:
            chosen = next(node for node in pool if node not in sampled)
        sampled.add(chosen)
    return sampled


def degree_matched_proximity(
    targets: set[str],
    distances: dict[str, int],
    graph: dict[str, set[str]],
    node_bin: dict[str, int],
    pools: dict[int, list[str]],
    rng: np.random.Generator,
    permutations: int,
) -> dict:
    present = targets & graph.keys()
    observed = mean_distance(present, distances)
    if not present or not math.isfinite(observed):
        return {
            "observed": math.nan, "null_mean": math.nan, "null_sd": math.nan,
            "z": math.nan, "p_lower": math.nan, "present": len(present),
        }
    null = np.asarray([
        mean_distance(
            sample_degree_matched(present, node_bin, pools, rng), distances
        )
        for _ in range(permutations)
    ], dtype=float)
    null = null[np.isfinite(null)]
    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=1))
    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z": (observed - null_mean) / null_sd if null_sd > 0 else math.nan,
        "p_lower": (1 + int(np.sum(null <= observed))) / (len(null) + 1),
        "present": len(present),
    }


def incremental_test(
    anchor_targets: set[str],
    candidate_targets: set[str],
    distances: dict[str, int],
    graph: dict[str, set[str]],
    node_bin: dict[str, int],
    pools: dict[int, list[str]],
    rng: np.random.Generator,
    permutations: int,
) -> tuple[dict, list[float]]:
    """Test added proximity while holding HCTZ and shared targets fixed."""
    anchor = anchor_targets & graph.keys()
    candidate_specific = (candidate_targets - anchor_targets) & graph.keys()
    anchor_distance = mean_distance(anchor, distances)
    observed_union = mean_distance(anchor | candidate_specific, distances)
    observed_delta = anchor_distance - observed_union
    if not candidate_specific or not math.isfinite(observed_delta):
        return {
            "candidate_specific_ppi_target_count": len(candidate_specific),
            "observed_delta_anchor_minus_union": observed_delta,
            "null_delta_mean": math.nan,
            "null_delta_sd": math.nan,
            "incremental_z": math.nan,
            "incremental_upper_tail_p": math.nan,
        }, []
    null_delta = []
    for _ in range(permutations):
        random_specific = sample_degree_matched(
            candidate_specific, node_bin, pools, rng, excluded=anchor
        )
        random_union_distance = mean_distance(anchor | random_specific, distances)
        null_delta.append(anchor_distance - random_union_distance)
    null = np.asarray(null_delta, dtype=float)
    null = null[np.isfinite(null)]
    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=1))
    return {
        "candidate_specific_ppi_target_count": len(candidate_specific),
        "observed_delta_anchor_minus_union": observed_delta,
        "null_delta_mean": null_mean,
        "null_delta_sd": null_sd,
        "incremental_z": (observed_delta - null_mean) / null_sd if null_sd > 0 else math.nan,
        "incremental_upper_tail_p": (
            1 + int(np.sum(null >= observed_delta))
        ) / (len(null) + 1),
    }, null.tolist()


def disease_distance_matrix(
    disease_genes: set[str], graph: dict[str, set[str]]
) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Precompute disease-gene-to-node distances for fast coverage permutations."""
    nodes = sorted(graph)
    node_index = {node: i for i, node in enumerate(nodes)}
    matrix = np.full((len(disease_genes), len(nodes)), 32767, dtype=np.int16)
    for row, disease_gene in enumerate(sorted(disease_genes)):
        distances = multisource_distances(graph, {disease_gene})
        for node, distance in distances.items():
            matrix[row, node_index[node]] = distance
    return nodes, node_index, matrix


def disease_coverage_distance(
    targets: set[str], node_index: dict[str, int], matrix: np.ndarray
) -> float:
    indices = [node_index[target] for target in targets if target in node_index]
    if not indices:
        return math.nan
    closest = matrix[:, indices].min(axis=1)
    reachable = closest[closest < 32767]
    return float(np.mean(reachable)) if len(reachable) else math.nan


def incremental_coverage_test(
    anchor_targets: set[str],
    candidate_targets: set[str],
    graph: dict[str, set[str]],
    node_bin: dict[str, int],
    pools: dict[int, list[str]],
    node_index: dict[str, int],
    matrix: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> tuple[dict, list[float]]:
    """Disease-centric incremental coverage; adding targets cannot increase distance."""
    anchor = anchor_targets & graph.keys()
    candidate = candidate_targets & graph.keys()
    candidate_specific = candidate - anchor
    anchor_distance = disease_coverage_distance(anchor, node_index, matrix)
    candidate_distance = disease_coverage_distance(candidate, node_index, matrix)
    union_distance = disease_coverage_distance(anchor | candidate, node_index, matrix)
    delta = anchor_distance - union_distance
    base = {
        "anchor_disease_coverage_distance": anchor_distance,
        "candidate_disease_coverage_distance": candidate_distance,
        "union_disease_coverage_distance": union_distance,
        "coverage_delta_anchor_minus_union": delta,
    }
    if not candidate_specific or not math.isfinite(delta):
        return {
            **base, "coverage_null_delta_mean": math.nan,
            "coverage_null_delta_sd": math.nan, "coverage_incremental_z": math.nan,
            "coverage_incremental_upper_tail_p": math.nan,
        }, []
    null_delta = []
    for _ in range(permutations):
        random_specific = sample_degree_matched(
            candidate_specific, node_bin, pools, rng, excluded=anchor
        )
        random_distance = disease_coverage_distance(
            anchor | random_specific, node_index, matrix
        )
        null_delta.append(anchor_distance - random_distance)
    null = np.asarray(null_delta, dtype=float)
    null = null[np.isfinite(null)]
    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=1))
    return {
        **base,
        "coverage_null_delta_mean": null_mean,
        "coverage_null_delta_sd": null_sd,
        "coverage_incremental_z": (
            (delta - null_mean) / null_sd if null_sd > 0 else math.nan
        ),
        "coverage_incremental_upper_tail_p": (
            1 + int(np.sum(null >= delta))
        ) / (len(null) + 1),
    }, null.tolist()


def closest_distance_to_set(
    source: str, targets: set[str], graph: dict[str, set[str]]
) -> float:
    if source in targets:
        return 0.0
    seen = {source}
    queue = collections.deque([(source, 0)])
    while queue:
        node, distance = queue.popleft()
        for neighbor in graph.get(node, ()):
            if neighbor in seen:
                continue
            if neighbor in targets:
                return float(distance + 1)
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return math.nan


def symmetric_between_distance(
    first: set[str], second: set[str], graph: dict[str, set[str]]
) -> float:
    first, second = first & graph.keys(), second & graph.keys()
    if not first or not second:
        return math.nan
    values = [closest_distance_to_set(x, second, graph) for x in first]
    values += [closest_distance_to_set(x, first, graph) for x in second]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def within_distance(targets: set[str], graph: dict[str, set[str]]) -> float:
    targets = targets & graph.keys()
    if len(targets) < 2:
        return math.nan
    values = [closest_distance_to_set(x, targets - {x}, graph) for x in targets]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def add_bh(rows: list[dict], p_field: str, q_field: str) -> None:
    valid = [i for i, row in enumerate(rows) if math.isfinite(float(row[p_field]))]
    adjusted = bh_adjust([float(rows[i][p_field]) for i in valid]) if valid else []
    for row in rows:
        row[q_field] = math.nan
    for index, value in zip(valid, adjusted):
        rows[index][q_field] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, default=RANKING)
    parser.add_argument("--kegg-output", type=Path, default=KEGG_OUT)
    parser.add_argument("--hippie", type=Path, default=HIPPIE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--hippie-min-score", type=float, default=0.73)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    top5 = load_top5(args.ranking)
    targets, target_symbols = load_targets(
        args.kegg_output / "stitch_medium_confidence_targets.csv"
    )
    disease_genes, disease_symbols = load_disease_genes(
        args.kegg_output / "essential_hypertension_disgenet_scoregda_ge0p5.csv"
    )
    symbols = {**target_symbols, **disease_symbols}
    graph, graph_counts = load_independent_ppi(args.hippie, args.hippie_min_score)
    disease_in_graph = disease_genes & graph.keys()
    distances = multisource_distances(graph, disease_in_graph)
    node_bin, pools = degree_bins(graph)
    _, node_index, disease_matrix = disease_distance_matrix(disease_in_graph, graph)
    rng = np.random.default_rng(args.seed)

    anchor = targets.get(ANCHOR, set())
    anchor_test = degree_matched_proximity(
        anchor, distances, graph, node_bin, pools, rng, args.permutations
    )
    summary_rows, null_rows, coverage_rows = [], [], []
    for rank_row in top5:
        candidate = rank_row["candidate_drug"]
        candidate_targets = targets.get(candidate, set())
        union = anchor | candidate_targets
        candidate_test = degree_matched_proximity(
            candidate_targets, distances, graph, node_bin, pools, rng, args.permutations
        )
        union_test = degree_matched_proximity(
            union, distances, graph, node_bin, pools, rng, args.permutations
        )
        incremental, null_delta = incremental_test(
            anchor, candidate_targets, distances, graph, node_bin, pools, rng,
            args.permutations,
        )
        coverage_incremental, coverage_null_delta = incremental_coverage_test(
            anchor, candidate_targets, graph, node_bin, pools, node_index,
            disease_matrix, rng, args.permutations,
        )
        anchor_ppi, candidate_ppi = anchor & graph.keys(), candidate_targets & graph.keys()
        d_ab = symmetric_between_distance(anchor_ppi, candidate_ppi, graph)
        d_aa = within_distance(anchor_ppi, graph)
        d_bb = within_distance(candidate_ppi, graph)
        separation = (
            d_ab - 0.5 * (d_aa + d_bb)
            if all(math.isfinite(x) for x in (d_ab, d_aa, d_bb)) else math.nan
        )
        shared = anchor & candidate_targets
        nonoverlap_disease = disease_in_graph
        anchor_without_direct = anchor_ppi - nonoverlap_disease
        candidate_without_direct = candidate_ppi - nonoverlap_disease
        union_without_direct = (anchor_ppi | candidate_ppi) - nonoverlap_disease
        row = {
            "consensus_rank": int(rank_row["consensus_rank"]),
            "candidate_drug": candidate,
            "candidate_name": DRUGS[candidate]["name"],
            "disease_gene_count": len(disease_genes),
            "disease_genes_in_hippie": len(disease_in_graph),
            "anchor_target_count": len(anchor),
            "anchor_ppi_target_count": len(anchor_ppi),
            "candidate_target_count": len(candidate_targets),
            "candidate_ppi_target_count": len(candidate_ppi),
            "union_ppi_target_count": len(anchor_ppi | candidate_ppi),
            "shared_target_count": len(shared),
            "shared_target_symbols": "|".join(sorted(symbols.get(x, x) for x in shared)),
            "target_jaccard": len(shared) / len(union) if union else math.nan,
            "anchor_mean_distance": anchor_test["observed"],
            "anchor_proximity_z": anchor_test["z"],
            "anchor_proximity_p": anchor_test["p_lower"],
            "candidate_mean_distance": candidate_test["observed"],
            "candidate_proximity_null_mean": candidate_test["null_mean"],
            "candidate_proximity_z": candidate_test["z"],
            "candidate_proximity_p": candidate_test["p_lower"],
            "union_mean_distance": union_test["observed"],
            "union_proximity_null_mean": union_test["null_mean"],
            "union_proximity_z": union_test["z"],
            "union_proximity_p": union_test["p_lower"],
            **incremental,
            **coverage_incremental,
            "anchor_nonoverlap_mean_distance": mean_distance(anchor_without_direct, distances),
            "candidate_nonoverlap_mean_distance": mean_distance(candidate_without_direct, distances),
            "union_nonoverlap_mean_distance": mean_distance(union_without_direct, distances),
            "symmetric_cross_target_distance_dab": d_ab,
            "anchor_within_target_distance_daa": d_aa,
            "candidate_within_target_distance_dbb": d_bb,
            "network_separation_sab": separation,
        }
        summary_rows.append(row)
        null_rows.extend({
            "candidate_drug": candidate,
            "candidate_name": DRUGS[candidate]["name"],
            "permutation": i + 1,
            "null_target_centric_delta": value,
            "null_disease_coverage_delta": (
                coverage_null_delta[i] if i < len(coverage_null_delta) else ""
            ),
        } for i, value in enumerate(null_delta))
        for drug, genes in ((ANCHOR, anchor), (candidate, candidate_targets)):
            for gene in sorted(genes, key=int):
                coverage_rows.append({
                    "candidate_drug": candidate,
                    "candidate_name": DRUGS[candidate]["name"],
                    "source_drug": drug,
                    "source_drug_name": DRUGS[drug]["name"],
                    "entrez_id": gene,
                    "gene_symbol": symbols.get(gene, gene),
                    "in_hippie": gene in graph,
                    "is_essential_hypertension_gene": gene in disease_genes,
                    "distance_to_disease_module": distances.get(gene, ""),
                    "is_shared_drug_target": gene in shared,
                })

    for p_field, q_field in (
        ("candidate_proximity_p", "candidate_proximity_fdr_bh"),
        ("union_proximity_p", "union_proximity_fdr_bh"),
        ("incremental_upper_tail_p", "incremental_upper_tail_fdr_bh"),
        ("coverage_incremental_upper_tail_p", "coverage_incremental_upper_tail_fdr_bh"),
    ):
        add_bh(summary_rows, p_field, q_field)

    write_csv(args.output / "ppi_proximity_incremental_summary.csv", summary_rows)
    write_csv(args.output / "ppi_target_gene_coverage.csv", coverage_rows)
    write_csv(
        args.output / "ppi_incremental_null_distributions.csv", null_rows,
        ["candidate_drug", "candidate_name", "permutation",
         "null_target_centric_delta", "null_disease_coverage_delta"],
    )

    report = {
        "analysis_date": "2026-08-25",
        "disease_context": "Essential hypertension",
        "disease_gene_definition": (
            "User-supplied data/EH_genes.xlsx; disease exactly Essential Hypertension; "
            "ScoreGDA >= 0.5 (inclusive)"
        ),
        "drug_target_definition": "STITCH v5 human links; combined_score >= 400",
        "ppi": {
            "source": "HIPPIE v2.4 (2026-04-09 local snapshot)",
            "minimum_score": args.hippie_min_score,
            "graph_policy": "complete HIPPIE graph after confidence filtering; no PrimeKG-edge exclusion",
            **graph_counts,
        },
        "network_metric": (
            "Mean shortest-path distance from PPI-mapped drug targets to the nearest "
            "PPI-mapped essential-hypertension disease gene; smaller is closer"
        ),
        "incremental_test": (
            "HCTZ targets are fixed and candidate-specific targets are replaced with "
            "distinct HIPPIE nodes from matching log-degree deciles. Both target-centric "
            "delta and disease-centric coverage delta are reported. The latter is "
            "d(D,HCTZ)-d(D,HCTZ union candidate), so positive delta indicates broader "
            "disease-module coverage and cannot decrease merely because targets are added."
        ),
        "permutations": args.permutations,
        "permutation_seed": args.seed,
        "multiple_testing": (
            "BH correction across finite candidate p-values within each test family; four evaluable candidates for disease-centric incremental coverage. Families: candidate proximity, "
            "union proximity, target-centric incremental contribution, and disease-centric "
            "incremental coverage"
        ),
        "network_separation": (
            "s_AB=d_AB-(d_AA+d_BB)/2 using symmetric closest distances between drug "
            "target sets and self-excluding closest distances within each set; descriptive"
        ),
        "counts": {
            "disease_genes": len(disease_genes),
            "disease_genes_in_hippie": len(disease_in_graph),
            "top_candidates": len(top5),
        },
        "source_sha256": {
            args.hippie.name: sha256(args.hippie),
            "stitch_medium_confidence_targets.csv": sha256(
                args.kegg_output / "stitch_medium_confidence_targets.csv"
            ),
            "essential_hypertension_disgenet_scoregda_ge0p5.csv": sha256(
                args.kegg_output / "essential_hypertension_disgenet_scoregda_ge0p5.csv"
            ),
            args.ranking.name: sha256(args.ranking),
        },
        "limitations": [
            "STITCH combined scores can contain indirect, database, or text-mined evidence and do not necessarily indicate direct binding.",
            "PPI proximity is mechanistic prioritization evidence, not proof of combination efficacy, safety, or clinical novelty.",
            "HIPPIE and STITCH may share upstream biological databases with resources used elsewhere; the PPI graph is independent of model inference but not guaranteed source-disjoint from PrimeKG.",
            "Candidates without mapped STITCH targets cannot be evaluated by this PPI analysis.",
        ],
    }
    (args.output / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
