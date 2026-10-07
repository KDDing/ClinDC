#!/usr/bin/env python3
"""STITCH target and KEGG enrichment analysis for ClinDC's essential-HTN Top 5.

Drug target sets are intersected with the supplied DisGeNET essential-
hypertension genes (ScoreGDA >= 0.5), then tested for KEGG over-representation
with within-pair false-discovery-rate correction.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external_case_study" / "raw"
RANKING = (
    ROOT / "outputs" / "clindc_full_data_hctz_essential_hypertension_tiered_novelty"
    / "top30_consensus_target_network.csv"
)
DISEASE_GENES = ROOT / "data" / "EH_genes.xlsx"
HGNC = ROOT / "data" / "external_hypertension_validation" / "raw" / "hgnc_complete_set_2026-08-22.txt"
OUT = ROOT / "outputs" / "clindc_essential_hypertension_top5_enrichment"

ANCHOR = "DB00999"
DRUGS = {
    "DB00999": {"name": "Hydrochlorothiazide", "pubchem_cid": 3639},
    "DB00695": {"name": "Furosemide", "pubchem_cid": 3440},
    "DB00590": {"name": "Doxazosin", "pubchem_cid": 3157},
    "DB00421": {"name": "Spironolactone", "pubchem_cid": 5833},
    "DB01294": {"name": "Bismuth subsalicylate", "pubchem_cid": 16682734},
    "DB00575": {"name": "Clonidine", "pubchem_cid": 2803},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_top5(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[:5]
    expected = list(DRUGS)[1:]
    observed = [row["candidate_drug"] for row in rows]
    if observed != expected:
        raise ValueError(f"Top-5 mismatch: expected {expected}, found {observed}")
    return rows


def load_hgnc(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    symbol_to_entrez: dict[str, str] = {}
    entrez_to_symbol: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            symbol, entrez = row.get("symbol", ""), row.get("entrez_id", "")
            if symbol and entrez:
                symbol_to_entrez[symbol] = entrez
                entrez_to_symbol[entrez] = symbol
    return symbol_to_entrez, entrez_to_symbol


def load_string_symbols(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mapping[row["#string_protein_id"]] = row["preferred_name"]
    return mapping


def stitch_ids(cid: int) -> set[str]:
    return {f"CIDm{cid:08d}", f"CIDs{cid:08d}"}


def load_stitch_targets(
    links_path: Path,
    protein_symbols: dict[str, str],
    symbol_to_entrez: dict[str, str],
    minimum_score: int,
) -> tuple[dict[str, set[str]], list[dict]]:
    chemical_to_drug = {
        chemical: drug
        for drug, meta in DRUGS.items()
        for chemical in stitch_ids(meta["pubchem_cid"])
    }
    targets: dict[str, set[str]] = defaultdict(set)
    best: dict[tuple[str, str], int] = {}
    with gzip.open(links_path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            drug = chemical_to_drug.get(row["chemical"])
            score = int(row["combined_score"])
            if drug is None or score < minimum_score:
                continue
            symbol = protein_symbols.get(row["protein"], "")
            entrez = symbol_to_entrez.get(symbol, "")
            if not entrez:
                continue
            targets[drug].add(entrez)
            key = (drug, entrez)
            best[key] = max(best.get(key, 0), score)
    rows = [
        {
            "drugbank_id": drug,
            "drug_name": DRUGS[drug]["name"],
            "pubchem_cid": DRUGS[drug]["pubchem_cid"],
            "entrez_id": gene,
            "gene_symbol": next(
                (symbol for symbol, entrez in symbol_to_entrez.items() if entrez == gene),
                gene,
            ),
            "maximum_stitch_combined_score": score,
        }
        for (drug, gene), score in sorted(best.items())
    ]
    return targets, rows


def load_disease_genes(
    path: Path,
    symbol_to_entrez: dict[str, str],
    minimum_score: float,
) -> tuple[set[str], dict[str, str], list[dict], dict]:
    frame = pd.read_excel(path, sheet_name=0)
    required = {"gene", "disease", "ScoreGDA"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing EH_genes.xlsx columns: {sorted(missing)}")
    frame["ScoreGDA"] = pd.to_numeric(frame["ScoreGDA"], errors="coerce")
    disease_values = {
        " ".join(str(value).casefold().split())
        for value in frame["disease"].dropna()
    }
    if disease_values != {"essential hypertension"}:
        raise ValueError(f"Unexpected disease values: {sorted(disease_values)}")
    retained = frame.loc[frame["ScoreGDA"].ge(minimum_score)].copy()
    retained["normalized_symbol"] = retained["gene"].astype(str).str.strip().str.upper()
    if retained["normalized_symbol"].duplicated().any():
        duplicates = retained.loc[
            retained["normalized_symbol"].duplicated(False), "normalized_symbol"
        ].tolist()
        raise ValueError(f"Duplicate retained gene symbols: {duplicates}")
    unmapped = sorted(set(retained["normalized_symbol"]) - set(symbol_to_entrez))
    if unmapped:
        raise ValueError(f"Retained symbols absent from HGNC Entrez mapping: {unmapped}")
    rows = []
    genes, symbols = set(), {}
    for _, row in retained.sort_values(
        ["ScoreGDA", "normalized_symbol"], ascending=[False, True]
    ).iterrows():
        symbol = row["normalized_symbol"]
        entrez = symbol_to_entrez[symbol]
        genes.add(entrez)
        symbols[entrez] = symbol
        rows.append({
            "gene_symbol": symbol,
            "entrez_id": entrez,
            "disease": str(row["disease"]).strip(),
            "ScoreGDA": float(row["ScoreGDA"]),
            "evidence_timeline": str(row.get("Evidence timeline", "")).strip(),
            "gene_full_name": str(row.get("gene full nama", "")).strip(),
        })
    audit = {
        "source_row_count": int(len(frame)),
        "retained_row_count": int(len(retained)),
        "unique_retained_gene_count": len(genes),
        "minimum_score_gda_inclusive": minimum_score,
        "missing_gene_count": int(frame["gene"].isna().sum()),
        "missing_score_count": int(frame["ScoreGDA"].isna().sum()),
        "disease_values": sorted(disease_values),
    }
    return genes, symbols, rows, audit


def load_kegg(raw: Path, min_size: int, max_size: int):
    names: dict[str, str] = {}
    with (raw / "kegg_human_pathways_2026-08-25.txt").open(encoding="utf-8") as handle:
        for line in handle:
            pathway, name = line.rstrip("\n").split("\t", 1)
            names[pathway.replace("hsa", "path:hsa", 1)] = name.removesuffix(" - Homo sapiens (human)")
    pathways: dict[str, set[str]] = defaultdict(set)
    with (raw / "kegg_human_pathway_gene_links_2026-08-25.txt").open(encoding="utf-8") as handle:
        for line in handle:
            pathway, gene = line.rstrip("\n").split("\t")
            pathways[pathway].add(gene.removeprefix("hsa:"))
    retained = {
        pathway: genes for pathway, genes in pathways.items()
        if min_size <= len(genes) <= max_size
    }
    background = set().union(*retained.values())
    return retained, names, background


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--ranking", type=Path, default=RANKING)
    parser.add_argument("--disease-genes", type=Path, default=DISEASE_GENES)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--minimum-stitch-score", type=int, default=400)
    parser.add_argument("--minimum-score-gda", type=float, default=0.5)
    parser.add_argument("--minimum-pathway-size", type=int, default=10)
    parser.add_argument("--maximum-pathway-size", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    top5 = load_top5(args.ranking)
    symbol_to_entrez, entrez_to_symbol = load_hgnc(HGNC)
    protein_symbols = load_string_symbols(args.raw / "9606.protein.info.v12.0.txt.gz")
    targets, target_rows = load_stitch_targets(
        args.raw / "9606.protein_chemical.links.v5.0.tsv.gz",
        protein_symbols,
        symbol_to_entrez,
        args.minimum_stitch_score,
    )
    disease_genes, disease_symbols, disease_gene_rows, disease_gene_audit = load_disease_genes(
        args.disease_genes, symbol_to_entrez, args.minimum_score_gda
    )
    pathways, pathway_names, background = load_kegg(
        args.raw, args.minimum_pathway_size, args.maximum_pathway_size
    )
    symbols = {**entrez_to_symbol, **disease_symbols}

    intersections: list[dict] = []
    enrichment: list[dict] = []
    summaries: list[dict] = []
    for rank_row in top5:
        candidate = rank_row["candidate_drug"]
        anchor_overlap = targets.get(ANCHOR, set()) & disease_genes
        candidate_overlap = targets.get(candidate, set()) & disease_genes
        query = (anchor_overlap | candidate_overlap) & background
        for drug, genes in ((ANCHOR, anchor_overlap), (candidate, candidate_overlap)):
            for gene in sorted(genes, key=int):
                intersections.append({
                    "consensus_rank": int(rank_row["consensus_rank"]),
                    "candidate_drug": candidate,
                    "candidate_name": DRUGS[candidate]["name"],
                    "source_drug": drug,
                    "source_drug_name": DRUGS[drug]["name"],
                    "entrez_id": gene,
                    "gene_symbol": symbols.get(gene, gene),
                })
        pair_rows = []
        for pathway, genes in pathways.items():
            overlap = query & genes
            # Keep the entire prespecified pathway family, including zero hits.
            p_value = float(hypergeom.sf(
                len(overlap) - 1, len(background), len(genes), len(query)
            )) if query else 1.0
            pair_rows.append({
                "consensus_rank": int(rank_row["consensus_rank"]),
                "candidate_drug": candidate,
                "candidate_name": DRUGS[candidate]["name"],
                "pathway_id": pathway.removeprefix("path:"),
                "pathway_name": pathway_names.get(pathway, pathway),
                "query_gene_count": len(query),
                "background_gene_count": len(background),
                "pathway_background_count": len(genes),
                "overlap_count": len(overlap),
                "overlap_entrez": "|".join(sorted(overlap, key=int)),
                "overlap_symbols": "|".join(sorted(symbols.get(gene, gene) for gene in overlap)),
                "fold_enrichment": (len(overlap) / len(query)) / (len(genes) / len(background)) if query else 0.0,
                "p_value": p_value,
            })
        q_values = bh_adjust([row["p_value"] for row in pair_rows])
        for row, q_value in zip(pair_rows, q_values):
            row["fdr_bh_within_pair"] = q_value
        pair_rows.sort(key=lambda row: (row["fdr_bh_within_pair"], row["p_value"]))
        enrichment.extend(pair_rows)
        summaries.append({
            "consensus_rank": int(rank_row["consensus_rank"]),
            "candidate_drug": candidate,
            "candidate_name": DRUGS[candidate]["name"],
            "anchor_stitch_target_count": len(targets.get(ANCHOR, set())),
            "candidate_stitch_target_count": len(targets.get(candidate, set())),
            "anchor_disease_overlap_count": len(anchor_overlap),
            "candidate_disease_overlap_count": len(candidate_overlap),
            "candidate_unique_disease_overlap_count": len(candidate_overlap - anchor_overlap),
            "candidate_unique_disease_overlap_symbols": "|".join(
                sorted(symbols.get(gene, gene) for gene in candidate_overlap - anchor_overlap)
            ),
            "merged_kegg_query_gene_count": len(query),
            "merged_kegg_query_symbols": "|".join(sorted(symbols.get(gene, gene) for gene in query)),
            "overlapping_pathway_hypothesis_count": sum(row["overlap_count"] > 0 for row in pair_rows),
            "tested_pathway_hypothesis_count": len(pair_rows),
            "fdr_significant_pathway_count": sum(
                row["fdr_bh_within_pair"] < 0.05 for row in pair_rows
            ),
            "multi_gene_fdr_significant_pathway_count": sum(
                row["fdr_bh_within_pair"] < 0.05 and row["overlap_count"] >= 2
                for row in pair_rows
            ),
            "single_gene_query_warning": len(query) == 1,
        })

    write_csv(args.output / "stitch_medium_confidence_targets.csv", target_rows)
    write_csv(
        args.output / "essential_hypertension_disgenet_scoregda_ge0p5.csv",
        disease_gene_rows,
    )
    write_csv(
        args.output / "drug_essential_hypertension_gene_intersections.csv",
        intersections,
        ["consensus_rank", "candidate_drug", "candidate_name", "source_drug",
         "source_drug_name", "entrez_id", "gene_symbol"],
    )
    write_csv(
        args.output / "kegg_enrichment_all.csv", enrichment,
        ["consensus_rank", "candidate_drug", "candidate_name", "pathway_id",
         "pathway_name", "query_gene_count", "background_gene_count",
         "pathway_background_count", "overlap_count", "overlap_entrez",
         "overlap_symbols", "fold_enrichment", "p_value", "fdr_bh_within_pair"],
    )
    write_csv(args.output / "top5_enrichment_summary.csv", summaries)
    significant = [row for row in enrichment if row["fdr_bh_within_pair"] < 0.05]
    write_csv(
        args.output / "kegg_enrichment_fdr05.csv", significant,
        list(enrichment[0]) if enrichment else ["consensus_rank", "candidate_drug"],
    )

    source_files = [
        args.raw / "9606.protein_chemical.links.v5.0.tsv.gz",
        args.raw / "9606.protein.info.v12.0.txt.gz",
        args.raw / "kegg_human_pathways_2026-08-25.txt",
        args.raw / "kegg_human_pathway_gene_links_2026-08-25.txt",
        args.disease_genes,
        args.ranking,
    ]
    report = {
        "method": "ClinDC case-study STITCH target and KEGG enrichment analysis",
        "faithful_components": [
            "STITCH medium-confidence threshold: combined_score >= 400",
            "intersect each drug target set with hypertension genes",
            "merge anchor and partner disease-gene intersections",
            "KEGG one-sided hypergeometric over-representation test",
            "BH FDR correction across all eligible KEGG pathways within each drug pair, including zero-overlap pathways with p=1; significance q < 0.05",
        ],
        "adaptation": (
            "The user-supplied essential-hypertension DisGeNET export was filtered "
            "at ScoreGDA >= 0.5 (inclusive), as explicitly requested."
        ),
        "clusterprofiler_compatibility": "KEGG pathway sizes restricted to 10--500 genes",
        "counts": {
            "essential_hypertension_disgenet_genes": len(disease_genes),
            "kegg_background_genes": len(background),
            "eligible_kegg_pathways": len(pathways),
        },
        "disease_gene_audit": disease_gene_audit,
        "top5_summary": summaries,
        "robustness_interpretation": (
            f"Across the five pairs, {sum(row['multi_gene_fdr_significant_pathway_count'] for row in summaries)} "
            "pair-specific pathway tests pass FDR q < 0.05 with at least two hit genes. "
            "Candidate-specific contribution must still be distinguished from HCTZ-only genes."
        ),
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "limitations": [
            "This is a computational mechanism analysis, not independent evidence of clinical efficacy.",
            "STRING v12 protein names map STITCH v5 ENSP identifiers; unmapped proteins are omitted.",
            "The DisGeNET release and export date of EH_genes.xlsx are not recorded and remain to be verified.",
            "ScoreGDA association does not by itself establish a causal disease gene.",
            "STITCH combined scores integrate multiple evidence channels and do not necessarily represent direct binding.",
            "Pathway enrichment is mechanistic hypothesis generation, not evidence of efficacy or synergy.",
        ],
    }
    (args.output / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
