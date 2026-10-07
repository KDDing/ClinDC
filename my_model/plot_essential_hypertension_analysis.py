#!/usr/bin/env python3
"""Publication figures for ClinDC essential-hypertension analyses.

Figure contract
---------------
Core conclusions:
1. KEGG significance must be separated from candidate-specific disease-gene
   contribution because some pair-level terms are inherited from HCTZ.
2. Spironolactone and clonidine add disease-module coverage beyond HCTZ under
   degree-matched HIPPIE permutation tests.
Archetype: quantitative grids; double-column, editable SVG/PDF plus TIFF/PNG.
Data policy: all FDR-significant KEGG terms are shown; all five candidates and
all 40,000 evaluable PPI null draws are retained.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

# Mandatory publication settings are declared before any figure is created.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
KEGG_DIR = ROOT / "outputs" / "clindc_essential_hypertension_top5_enrichment"
PPI_DIR = ROOT / "outputs" / "clindc_essential_hypertension_top5_hippie"
OUT = ROOT / "outputs"

CANDIDATES = [
    "Furosemide",
    "Doxazosin",
    "Spironolactone",
    "Bismuth subsalicylate",
    "Clonidine",
]
SHORT = {
    "Furosemide": "Furosemide",
    "Doxazosin": "Doxazosin",
    "Spironolactone": "Spironolactone",
    "Bismuth subsalicylate": "Bismuth\nsubsalicylate",
    "Clonidine": "Clonidine",
}

BLUE = "#0F4D92"
BLUE_LIGHT = "#8DB6D8"
TEAL = "#42949E"
VIOLET = "#7564A8"
GOLD = "#D99524"
GREY = "#B8B8B8"
DARK = "#303030"


def apply_style() -> None:
    # Mandatory editable-text and font rules from the figure specification.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def panel_label(
    ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04,
    fontsize: float = 9,
) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=fontsize, fontweight="bold", color="black",
    )


def export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(
        stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight', facecolor='white'
    )
    fig.savefig(
        stem.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white'
    )
    plt.close(fig)


def load_candidate_specific_genes() -> dict[str, set[str]]:
    intersections = pd.read_csv(
        KEGG_DIR / "drug_essential_hypertension_gene_intersections.csv"
    )
    result: dict[str, set[str]] = {}
    for candidate in CANDIDATES:
        subset = intersections.loc[intersections["candidate_name"].eq(candidate)]
        anchor = set(subset.loc[
            subset["source_drug_name"].eq("Hydrochlorothiazide"), "gene_symbol"
        ])
        candidate_genes = set(subset.loc[
            subset["source_drug_name"].eq(candidate), "gene_symbol"
        ])
        result[candidate] = candidate_genes - anchor
    return result


def plot_enrichment() -> None:
    candidate_specific = load_candidate_specific_genes()
    enrichment = pd.read_csv(KEGG_DIR / "kegg_enrichment_fdr05.csv")
    enrichment = enrichment.loc[
        enrichment["candidate_name"].isin(CANDIDATES)
    ].copy()
    enrichment["candidate_incremental_genes"] = enrichment.apply(
        lambda row: "|".join(sorted(
            set(str(row["overlap_symbols"]).split("|"))
            & candidate_specific[row["candidate_name"]]
        )),
        axis=1,
    )
    enrichment["has_candidate_increment"] = enrichment[
        "candidate_incremental_genes"
    ].ne("")
    enrichment["minus_log10_q"] = -np.log10(
        enrichment["fdr_bh_within_pair"].clip(lower=np.finfo(float).tiny)
    )
    pathway_order = (
        enrichment.groupby("pathway_name")["fdr_bh_within_pair"]
        .min().sort_values(ascending=True).index.tolist()
    )
    enrichment.to_csv(
        OUT / "clindc_essential_hypertension_kegg_figure_source_data.csv",
        index=False,
    )

    # A shared candidate axis makes the compact overlap panel read as the
    # mechanistic key to the pathway-level hero panel below it.
    fig = plt.figure(figsize=(7.2, 6.65))
    grid = fig.add_gridspec(
        2, 1, height_ratios=[0.22, 0.78], hspace=0.24,
        left=0.22, right=0.985, top=0.965, bottom=0.215,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_legend = fig.add_axes([0.215, 0.018, 0.43, 0.072])
    ax_cbar = fig.add_axes([0.685, 0.045, 0.285, 0.022])

    x = np.arange(len(CANDIDATES))
    counts = [len(candidate_specific[candidate]) for candidate in CANDIDATES]
    for xi, count in zip(x, counts):
        if count:
            ax_a.plot([xi, xi], [0, count], color=BLUE_LIGHT, lw=2.2, zorder=1)
            ax_a.scatter(
                xi, count, s=40, facecolor=TEAL, edgecolor="white",
                linewidth=0.7, zorder=3,
            )
        else:
            ax_a.scatter(
                xi, 0, s=28, facecolor="white", edgecolor="#A0A0A0",
                linewidth=0.9, zorder=3,
            )
    ax_a.axhline(0, color="#9A9A9A", linewidth=0.7, zorder=0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([])
    ax_a.set_xlim(-0.55, len(CANDIDATES) - 0.45)
    ax_a.set_ylim(-0.25, 4.05)
    ax_a.set_yticks([0, 1, 2, 3])
    ax_a.set_ylabel("Candidate-specific\ndisease genes, $n$")
    ax_a.set_title("Incremental disease-gene overlap beyond HCTZ", loc="left", fontweight="bold")
    ax_a.grid(axis="y", color="#E8E8E8", linewidth=0.55, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.spines["bottom"].set_visible(False)
    ax_a.tick_params(axis="x", length=0)
    for xi, candidate, count in zip(x, CANDIDATES, counts):
        genes = ", ".join(sorted(candidate_specific[candidate])) or "None"
        ax_a.text(
            xi, count + (0.30 if count else 0.24), genes,
            va="bottom", ha="center", fontsize=6.0,
            color=DARK if count else "#858585",
            fontstyle="italic" if count else "normal",
        )
    panel_label(ax_a, "a", x=-0.075, y=1.02)

    x_map = {candidate: i for i, candidate in enumerate(CANDIDATES)}
    y_map = {pathway: i for i, pathway in enumerate(pathway_order[::-1])}
    norm = mpl.colors.Normalize(
        vmin=float(enrichment["minus_log10_q"].min()),
        vmax=float(enrichment["minus_log10_q"].max()),
    )
    cmap = mpl.colormaps["Blues"]
    for _, row in enrichment.iterrows():
        x_value = x_map[row["candidate_name"]]
        y_value = y_map[row["pathway_name"]]
        color = cmap(norm(row["minus_log10_q"]))
        size = 18 + 16 * float(row["overlap_count"])
        if row["has_candidate_increment"]:
            ax_b.scatter(
                x_value, y_value, s=size, facecolor=color, edgecolor=DARK,
                linewidth=0.45, zorder=3,
            )
        else:
            ax_b.scatter(
                x_value, y_value, s=size, facecolor="white", edgecolor=color,
                linewidth=1.25, zorder=3,
            )
    ax_b.set_xticks(range(len(CANDIDATES)))
    ax_b.set_xticklabels([SHORT[candidate] for candidate in CANDIDATES], rotation=30, ha="right")
    ax_b.set_yticks(range(len(pathway_order)))
    ax_b.set_yticklabels(pathway_order[::-1])
    ax_b.set_xlim(-0.55, len(CANDIDATES) - 0.45)
    ax_b.set_ylim(-0.65, len(pathway_order) - 0.35)
    for row_index in range(len(pathway_order)):
        if row_index % 2 == 0:
            ax_b.axhspan(
                row_index - 0.5, row_index + 0.5,
                color="#F6F8FA", zorder=-2,
            )
    ax_b.grid(axis="x", color="#E3E7EA", linewidth=0.55, zorder=0)
    ax_b.tick_params(length=0)
    ax_b.set_title("FDR-significant KEGG pathways", loc="left", fontweight="bold")
    panel_label(ax_b, "b", x=-0.075, y=1.02)

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar,
        orientation="horizontal",
    )
    colorbar.set_label(r"$-\log_{10}$(BH-adjusted $q$)")
    colorbar.outline.set_linewidth(0.6)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_LIGHT,
               markeredgecolor=DARK, markersize=5.5, label="Candidate-specific gene contributes"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=BLUE, markersize=5.5, label="HCTZ/shared genes only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B8D6EA",
               markeredgecolor=DARK, markersize=np.sqrt(18 + 16), label="1 overlap gene"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B8D6EA",
               markeredgecolor=DARK, markersize=np.sqrt(18 + 48), label="3 overlap genes"),
    ]
    ax_legend.set_axis_off()
    ax_legend.legend(
        handles=legend, loc="center left", bbox_to_anchor=(0.0, 0.48),
        ncol=2, columnspacing=0.9, handletextpad=0.45,
    )
    export(fig, OUT / "clindc_essential_hypertension_kegg_enrichment")


def plot_ppi() -> None:
    # Figure 4 is drawn at its final single-column width, avoiding a
    # downscaled two-column image while leaving the supplementary KEGG figure unchanged.
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    summary = pd.read_csv(PPI_DIR / "ppi_proximity_incremental_summary.csv")
    summary = summary.set_index("candidate_name").loc[CANDIDATES].reset_index()
    null = pd.read_csv(PPI_DIR / "ppi_incremental_null_distributions.csv")
    summary.to_csv(
        OUT / "clindc_essential_hypertension_ppi_figure_source_data.csv",
        index=False,
    )

    fig = plt.figure(figsize=(3.55, 4.9))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.43, 0.57])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.11, top=0.94, hspace=0.74)

    x = np.arange(len(CANDIDATES))
    anchor = summary["anchor_disease_coverage_distance"].to_numpy(float)
    union = summary["union_disease_coverage_distance"].to_numpy(float)
    for xi, anchor_value, union_value in zip(x, anchor, union):
        ax_a.plot([xi, xi], [anchor_value, union_value], color="#A7A7A7", lw=1.2, zorder=1)
    ax_a.scatter(x, anchor, s=28, facecolor="white", edgecolor="#777777", lw=1.0,
                 label="HCTZ alone", zorder=3)
    ax_a.scatter(x, union, s=31, facecolor=BLUE, edgecolor="white", lw=0.5,
                 label="HCTZ + candidate", zorder=4)
    ax_a.text(
        x[3], 2.27, "No target\n(score ≥400)", ha="center", va="top",
        fontsize=8, color="#777777",
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([SHORT[candidate] for candidate in CANDIDATES], rotation=50, ha="right")
    ax_a.set_ylabel("Coverage distance")
    ax_a.set_ylim(1.40, 2.44)
    ax_a.set_title("a   Disease-module coverage", loc="left", fontweight="bold")
    ax_a.grid(axis="y", color="#E8E8E8", linewidth=0.6)

    evaluable = [candidate for candidate in CANDIDATES if candidate in set(null["candidate_name"])]
    sns.violinplot(
        data=null, x="candidate_name", y="null_disease_coverage_delta",
        order=evaluable, color="#D9E5EE", inner=None, cut=0, linewidth=0.7,
        density_norm="width", ax=ax_b,
    )
    for collection in ax_b.collections:
        collection.set_edgecolor("#7D8F9C")
        collection.set_alpha(0.9)
    observed = summary.set_index("candidate_name").loc[
        evaluable, "coverage_delta_anchor_minus_union"
    ].to_numpy(float)
    q_values = summary.set_index("candidate_name").loc[
        evaluable, "coverage_incremental_upper_tail_fdr_bh"
    ].to_numpy(float)
    ax_b.scatter(
        np.arange(len(evaluable)), observed, marker="D", s=38,
        facecolor=GOLD, edgecolor=DARK, linewidth=0.65, zorder=5,
        label="Observed candidate gain",
    )
    ax_b.axhline(0, color="#666666", lw=0.8, ls="--")
    for xi, value, q_value in zip(range(len(evaluable)), observed, q_values):
        ax_b.text(
            xi, 0.94, f"$q={q_value:.4f}$", ha="center", va="top",
            fontsize=8, fontweight="bold" if q_value < 0.05 else "normal",
        )
    ax_b.set_xticks(np.arange(len(evaluable)))
    ax_b.set_xticklabels([SHORT[candidate].replace("\n", " ") for candidate in evaluable], rotation=35, ha="right")
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Incremental coverage gain")
    ax_b.set_ylim(-0.05, 1.0)
    ax_b.set_title(
        "b   Degree-matched null distributions",
        loc="left", fontweight="bold",
    )
    ax_b.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    export(fig, OUT / "clindc_essential_hypertension_hippie_ppi")


def main() -> None:
    apply_style()
    plot_enrichment()
    plot_ppi()


if __name__ == "__main__":
    main()
