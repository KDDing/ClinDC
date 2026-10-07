"""Rank the HCTZ-hypertension query and run target/network/pathway analyses."""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hypergeom

from build_ontology_aware_ldo import (
    ancestors, canonical_branch, depth_from_root, disease_ids, parse_obo,
)
from data import Example, TaskData
from evaluate_checkpoint_subgroups import build_from_checkpoint
from train import move


ROOT = Path(__file__).resolve().parents[1]


ESSENTIAL_HTN_MANDATORY_LABELS = {
    "essential hypertension",
    "primary hypertension",
    "idiopathic hypertension",
    "essential hypertension genetic",
    "genetic essential hypertension",
    "hypertension essential salt sensitive",
    "hypertension",
    "hypertension htn",
    "high blood pressure",
    "high blood pressure hypertension",
    "arterial hypertension",
    "systemic arterial hypertension",
    "htn",
    "essential hypertension mild to moderate",
}

ESSENTIAL_HTN_CONSERVATIVE_LABELS = {
    "resistant hypertension",
    "drug resistant hypertension",
    "hypertension resistant to conventional therapy",
    "systolic hypertension",
    "diastolic hypertension",
    "chronic hypertension",
    "pediatric hypertension",
    "hypertension with metabolic syndrome",
    "hypertension and dyslipidemia",
    "essential hypertension dyslipidemia",
    "hypertension with hyperlipidemia",
    "left ventricular hypertrophy hypertension",
    "hypertension hypertrophy left ventricular",
    "hypertensive nephrosclerosis",
}


def normalize_disease_label(value: str) -> str:
    """Normalize punctuation without conflating distinct hypertension types."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def essential_hypertension_tiered_labels(rows: list[dict]) -> tuple[set[str], dict[str, str]]:
    """Return observed labels excluded for disease-context repurposing novelty."""
    observed = {row["disease"] for row in rows}
    categories: dict[str, str] = {}
    for label in observed:
        normalized = normalize_disease_label(label)
        if normalized in ESSENTIAL_HTN_MANDATORY_LABELS:
            categories[label] = "mandatory"
        elif normalized in ESSENTIAL_HTN_CONSERVATIVE_LABELS:
            categories[label] = "conservative"
    return set(categories), categories


def bh_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), 1):
        rank = len(values) - reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted


def load_kg(path: Path, target_drugs: set[str], disease_keys: set[tuple[str, str]]):
    drug_targets: dict[str, set[str]] = collections.defaultdict(set)
    disease_genes: set[str] = set()
    ppi: dict[str, set[str]] = collections.defaultdict(set)
    gene_terms: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    term_names: dict[tuple[str, str], str] = {}
    drug_names: dict[str, str] = {}
    all_targetable_genes: set[str] = set()
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            rel = row["relation"]
            x_id, y_id = row["x_id"].strip(), row["y_id"].strip()
            if row["x_type"] == "drug" and x_id in target_drugs:
                drug_names.setdefault(x_id, row["x_name"].strip() or x_id)
            if row["y_type"] == "drug" and y_id in target_drugs:
                drug_names.setdefault(y_id, row["y_name"].strip() or y_id)
            if rel == "drug_protein":
                if row["x_type"] == "drug" and row["y_type"] == "gene/protein":
                    all_targetable_genes.add(y_id)
                    if x_id in target_drugs: drug_targets[x_id].add(y_id)
                elif row["y_type"] == "drug" and row["x_type"] == "gene/protein":
                    all_targetable_genes.add(x_id)
                    if y_id in target_drugs: drug_targets[y_id].add(x_id)
            elif rel == "disease_protein":
                if row["x_type"] == "disease" and (row["x_source"].strip(), x_id) in disease_keys:
                    disease_genes.add(y_id)
                elif row["y_type"] == "disease" and (row["y_source"].strip(), y_id) in disease_keys:
                    disease_genes.add(x_id)
            elif rel == "protein_protein":
                ppi[x_id].add(y_id); ppi[y_id].add(x_id)
            elif rel in {"bioprocess_protein", "pathway_protein"}:
                category = "GO Biological Process" if rel == "bioprocess_protein" else "Reactome Pathway"
                if row["x_type"] == "gene/protein":
                    gene, term, name = x_id, y_id, row["y_name"].strip()
                else:
                    gene, term, name = y_id, x_id, row["x_name"].strip()
                key = (category, term)
                gene_terms[gene].add(key); term_names.setdefault(key, name or term)
    return drug_targets, disease_genes, ppi, gene_terms, term_names, drug_names, all_targetable_genes


def multisource_distances(graph: dict[str, set[str]], sources: set[str]) -> dict[str, int]:
    distance = {node: 0 for node in sources if node in graph}
    queue = collections.deque(distance)
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, ()):
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1; queue.append(neighbor)
    return distance


def mean_nearest(left: set[str], right: set[str], graph: dict[str, set[str]]) -> float:
    distances = multisource_distances(graph, right)
    values = [distances[x] for x in left if x in distances]
    return float(np.mean(values)) if values else math.nan


def similar_disease_filter(rows: list[dict], ontology: Path, kg_path: Path,
                           query: str, depth: int):
    parents, names = parse_obo(ontology)
    ids = disease_ids(rows)
    # The ontology-aware CDCDB mapping sometimes points to a PrimeKG
    # MONDO_grouped node rather than a directly parseable MONDO identifier
    # (hypertension is one such case). Resolve these grouped nodes through
    # their disease_disease links to ordinary MONDO terms before assigning a
    # canonical branch.
    key_to_diseases: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in rows:
        sources = [x for x in row.get("kg_disease_sources", "").split("|") if x]
        entity_ids = [x for x in row.get("kg_disease_ids", "").split("|") if x]
        for source in sources:
            for entity_id in entity_ids:
                key_to_diseases[(source, entity_id)].add(row["disease"])
    resolved: dict[str, set[str]] = collections.defaultdict(set)
    with kg_path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["relation"] != "disease_disease":
                continue
            xkey = (row["x_source"].strip(), row["x_id"].strip())
            ykey = (row["y_source"].strip(), row["y_id"].strip())
            if xkey in key_to_diseases and ykey[0] == "MONDO":
                term = f"MONDO:{int(ykey[1]):07d}" if ykey[1].isdigit() else ykey[1]
                for disease in key_to_diseases[xkey]: resolved[disease].add(term)
            if ykey in key_to_diseases and xkey[0] == "MONDO":
                term = f"MONDO:{int(xkey[1]):07d}" if xkey[1].isdigit() else xkey[1]
                for disease in key_to_diseases[ykey]: resolved[disease].add(term)
    for disease, terms in resolved.items():
        disease_stems = {
            token[:7] for token in disease.casefold().replace(",", " ").split()
            if len(token) >= 5
        }
        name_matched = {
            term for term in terms if term in parents and any(
                token[:7] in disease_stems
                for token in names.get(term, "").casefold().replace(",", " ").split()
                if len(token) >= 5
            )
        }
        if name_matched:
            best = max(
                name_matched,
                key=lambda term: (
                    difflib.SequenceMatcher(
                        None, disease.casefold(), names.get(term, "").casefold()
                    ).ratio(),
                    -len(names.get(term, "")),
                ),
            )
            ids[disease].add(best)
    acache: dict[str, set[str]] = {}; dcache: dict[str, int] = {}
    branch = canonical_branch(ids[query], parents, acache, dcache, depth)
    query_terms = sorted(ids.get(query, ()))
    query_term = max(
        query_terms,
        key=lambda term: difflib.SequenceMatcher(
            None, query.casefold(), names.get(term, "").casefold()
        ).ratio(),
    ) if query_terms else None
    similar = {query}
    if query_term is not None:
        similar.update(
            disease for disease, terms in ids.items()
            if any(term == query_term or query_term in ancestors(term, parents, acache)
                   for term in terms)
        )
    return branch, names.get(branch, branch), similar, query_terms, query_term


@torch.no_grad()
def rank_seed(checkpoint: Path, data: TaskData, kg_path: Path, anchor: int,
              disease: int, candidates: list[int], device: torch.device):
    model, builder, payload = build_from_checkpoint(checkpoint, data, kg_path, device)
    model.eval()
    examples = [Example(anchor, disease, candidate, 1) for candidate in candidates]
    scores = []
    for start in range(0, len(examples), 512):
        scores.extend(model(move(builder.collate(examples[start:start + 512]), device)).cpu().tolist())
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(order), dtype=int); ranks[order] = np.arange(1, len(order) + 1)
    return ranks.tolist(), scores, int(payload["config"]["seed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=ROOT / "outputs/clindc_full_data_hctz_hypertension")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Directory containing seed_*/best_model.pt; defaults to --run-root.",
    )
    parser.add_argument("--triples", type=Path, default=ROOT / "data/dataset_v2_ontology/drug_disease_drug_kg_mapped.csv")
    parser.add_argument(
        "--novelty-triples",
        type=Path,
        default=ROOT / "data/dataset_v2_ontology/drug_disease_drug.csv",
        help="Complete CDCDB triple table used only to exclude previously observed pairs.",
    )
    parser.add_argument("--kg", type=Path, default=ROOT / "data/kg.csv")
    parser.add_argument("--ontology", type=Path, default=ROOT / "data/ontologies/mondo-2026-07-06.obo")
    parser.add_argument("--anchor", default="DB00999")
    parser.add_argument("--disease", default="hypertension")
    parser.add_argument("--branch-depth", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--novelty-filter",
        choices=("ontology_descendants", "essential_hypertension_tiered"),
        default="ontology_descendants",
        help=(
            "Candidate exclusion scope. The tiered option removes observed "
            "essential/primary/generic and conservative related hypertension labels, "
            "but retains etiologically distinct hypertension contexts."
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint_root = args.checkpoint_root or args.run_root
    device = torch.device(args.device)
    data = TaskData(args.triples)
    with args.triples.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.novelty_triples.open(encoding="utf-8", newline="") as handle:
        novelty_rows = list(csv.DictReader(handle))
    if args.anchor not in data.drug_to_idx or args.disease not in data.disease_to_idx:
        raise ValueError("The requested anchor or disease is absent from the modeling vocabulary")
    branch, branch_label, similar, query_resolved_terms, query_ontology_term = similar_disease_filter(
        rows, args.ontology, args.kg, args.disease, args.branch_depth
    )
    exclusion_categories = {label: "ontology_descendant" for label in similar}
    if args.novelty_filter == "essential_hypertension_tiered":
        if normalize_disease_label(args.disease) != "essential hypertension":
            raise ValueError(
                "essential_hypertension_tiered requires --disease 'essential hypertension'"
            )
        similar, exclusion_categories = essential_hypertension_tiered_labels(novelty_rows)
        exclusion_source_rows = novelty_rows
    else:
        exclusion_source_rows = rows
    excluded = {args.anchor}
    exclusion_evidence: dict[str, set[str]] = collections.defaultdict(set)
    for row in exclusion_source_rows:
        pair = {row["drug_1"], row["drug_2"]}
        if args.anchor in pair and row["disease"] in similar:
            partner = next(iter(pair - {args.anchor}), args.anchor)
            excluded.add(partner); exclusion_evidence[partner].add(row["disease"])
    candidate_ids = [i for i, name in enumerate(data.idx_to_drug) if name not in excluded]
    anchor_idx = data.drug_to_idx[args.anchor]; disease_idx = data.disease_to_idx[args.disease]
    seed_rows = []
    for checkpoint in sorted(checkpoint_root.glob("seed_*/best_model.pt"), key=lambda p: int(p.parent.name.split("_")[-1])):
        ranks, scores, seed = rank_seed(checkpoint, data, args.kg, anchor_idx, disease_idx, candidate_ids, device)
        for candidate, rank, score in zip(candidate_ids, ranks, scores):
            seed_rows.append({"seed": seed, "candidate_drug": data.idx_to_drug[candidate], "rank": rank, "score": score})
    seeds = sorted({row["seed"] for row in seed_rows})
    if seeds != list(range(9)) + [42]:
        raise RuntimeError(f"Expected seeds 0-8 and 42, found {seeds}")
    by_drug: dict[str, list[dict]] = collections.defaultdict(list)
    for row in seed_rows: by_drug[row["candidate_drug"]].append(row)
    summary = []
    for drug, values in by_drug.items():
        ranks = [x["rank"] for x in values]
        summary.append({"candidate_drug": drug, "mean_rank": float(np.mean(ranks)),
                        "rank_sd": float(np.std(ranks, ddof=1)), "median_rank": float(np.median(ranks)),
                        "best_rank": min(ranks), "worst_rank": max(ranks),
                        "top_k_frequency": sum(x <= args.top_k for x in ranks)})
    summary.sort(key=lambda x: (x["mean_rank"], x["rank_sd"], x["candidate_drug"]))
    top = summary[:args.top_k]
    target_drugs = {args.anchor} | {x["candidate_drug"] for x in top}
    disease_keys = data.disease_kg_keys[disease_idx]
    drug_targets, disease_genes, ppi, gene_terms, term_names, drug_names, background = load_kg(
        args.kg, target_drugs, disease_keys
    )
    disease_distance = multisource_distances(ppi, disease_genes)
    target_rows = []
    for rank_index, item in enumerate(top, 1):
        drug = item["candidate_drug"]; a = drug_targets[args.anchor]; b = drug_targets[drug]
        union, overlap = a | b, a & b
        proximity = [disease_distance[g] for g in union if g in disease_distance]
        d_ab = mean_nearest(a, b, ppi); d_aa = mean_nearest(a, a, ppi); d_bb = mean_nearest(b, b, ppi)
        target_rows.append({**item, "consensus_rank": rank_index, "candidate_name": drug_names.get(drug, drug),
                            "anchor_target_count": len(a), "candidate_target_count": len(b),
                            "shared_target_count": len(overlap),
                            "target_jaccard": len(overlap) / len(union) if union else math.nan,
                            "anchor_specific_targets": "|".join(sorted(a - b)),
                            "candidate_specific_targets": "|".join(sorted(b - a)),
                            "shared_targets": "|".join(sorted(overlap)),
                            "union_target_count": len(union),
                            "union_disease_gene_overlap": len(union & disease_genes),
                            "union_mean_disease_distance": float(np.mean(proximity)) if proximity else math.nan,
                            "union_reachable_target_fraction": len(proximity) / len(union) if union else math.nan,
                            "target_set_distance": d_ab,
                            "network_separation": d_ab - 0.5 * (d_aa + d_bb) if all(np.isfinite([d_ab, d_aa, d_bb])) else math.nan})
    # Over-representation against genes represented by PrimeKG drug-target edges.
    term_to_genes: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for gene, terms in gene_terms.items():
        if gene in background:
            for term in terms: term_to_genes[term].add(gene)
    enrichment = []
    M = len(background)
    for item in target_rows:
        genes = (drug_targets[args.anchor] | drug_targets[item["candidate_drug"]]) & background
        if not genes: continue
        for term, annotated in term_to_genes.items():
            overlap = genes & annotated
            if not overlap: continue
            pvalue = float(hypergeom.sf(len(overlap) - 1, M, len(annotated), len(genes)))
            enrichment.append({"candidate_drug": item["candidate_drug"], "candidate_name": item["candidate_name"],
                               "category": term[0], "term_id": term[1], "term_name": term_names[term],
                               "query_gene_count": len(genes), "background_gene_count": M,
                               "term_background_count": len(annotated), "overlap_count": len(overlap),
                               "overlap_genes": "|".join(sorted(overlap)), "p_value": pvalue})
    for drug in {x["candidate_drug"] for x in enrichment}:
        indices = [i for i, x in enumerate(enrichment) if x["candidate_drug"] == drug]
        qvals = bh_adjust([enrichment[i]["p_value"] for i in indices])
        for i, qvalue in zip(indices, qvals): enrichment[i]["fdr_bh"] = qvalue
    enrichment.sort(key=lambda x: (x["candidate_drug"], x.get("fdr_bh", 1), x["p_value"]))
    args.run_root.mkdir(parents=True, exist_ok=True)
    def write_csv(path: Path, values: list[dict]):
        if not values: return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=values[0].keys()); writer.writeheader(); writer.writerows(values)
    write_csv(args.run_root / "per_seed_candidate_ranks.csv", seed_rows)
    write_csv(args.run_root / "top30_consensus_target_network.csv", target_rows)
    write_csv(args.run_root / "top30_go_reactome_enrichment.csv", enrichment)
    write_csv(args.run_root / "excluded_similar_disease_partners.csv", [
        {
            "drug": drug,
            "drug_name": drug_names.get(drug, drug),
            "similar_diseases": "|".join(sorted(ds)),
            "exclusion_tiers": "|".join(sorted({exclusion_categories[d] for d in ds})),
        }
        for drug, ds in sorted(exclusion_evidence.items())
    ])
    report = {
        "query": {"anchor_drug": args.anchor, "anchor_name": drug_names.get(args.anchor, "hydrochlorothiazide"), "disease": args.disease},
        "checkpoint_root": str(checkpoint_root),
        "novelty_evidence_triples": str(args.novelty_triples),
        "seeds": seeds, "aggregation": "ascending arithmetic mean of within-seed candidate ranks",
        "candidate_count_after_filter": len(candidate_ids), "top_k": args.top_k,
        "ontology_filter": {"novelty_filter": args.novelty_filter,
                            "branch_depth": args.branch_depth, "canonical_branch": branch,
                            "canonical_branch_label": branch_label, "similar_diseases": sorted(similar),
                            "similar_disease_tiers": {
                                label: exclusion_categories[label] for label in sorted(similar)
                            },
                            "query_terms_resolved_from_primekg": query_resolved_terms,
                            "query_ontology_term": query_ontology_term,
                            "filter_rule": (
                                "observed mandatory and conservative essential-hypertension-related labels"
                                if args.novelty_filter == "essential_hypertension_tiered"
                                else "query MONDO term and its descendants"
                            ),
                            "excluded_known_anchor_partners": len(exclusion_evidence)},
        "target_definition": "PrimeKG drug_protein neighbors",
        "disease_module_definition": "PrimeKG disease_protein neighbors for the mapped hypertension entity",
        "ppi_definition": "PrimeKG protein_protein graph",
        "network_proximity": "mean shortest path from union drug targets to the hypertension disease module",
        "enrichment": {"sources": ["PrimeKG GO bioprocess_protein", "PrimeKG Reactome pathway_protein"],
                       "test": "one-sided hypergeometric", "multiple_testing": "BH within each candidate pair",
                       "background": "genes appearing in PrimeKG drug_protein edges"},
        "counts": {"anchor_targets": len(drug_targets[args.anchor]), "disease_module_genes": len(disease_genes),
                   "ppi_nodes": len(ppi), "enrichment_background_genes": len(background),
                   "enrichment_rows": len(enrichment)},
        "limitations": ["All CDCDB positives were used for fitting; rankings are discovery priorities, not held-out performance.",
                        "Disease-context novelty does not imply global pair novelty: pairs observed only under etiologically distinct diseases remain eligible.",
                        "Orange Book rows lack disease labels in CDCDB and require a separate indication-aware external novelty audit.",
                        "PrimeKG associations and shortest paths provide mechanistic support, not evidence of clinical efficacy or synergy."],
    }
    (args.run_root / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
