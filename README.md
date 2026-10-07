# ClinDC

This repository contains the ClinDC implementation, experiment runners, processed prediction datasets, and Morgan features. The processed inputs are stored directly under `data/` in the repository. PrimeKG and other third-party source files must be obtained from their original providers.

The final ClinDC combines Morgan fingerprints and PrimeKG drug context with disease lexical and graph context, then scores drug pairs using an order-invariant representation.

## 1. Obtain the data

The 12 processed prediction files are included in this repository under `data/`. They include the experiment splits, their preprocessing metadata, and the aligned Morgan fingerprints. The repository does not include the PrimeKG graph, the MONDO source ontology, or the third-party case-study source files.

Download the PrimeKG graph from the original [Zitnik Lab PrimeKG project page](https://zitniklab.hms.harvard.edu/projects/PrimeKG/), which links to its Harvard Dataverse deposit. Place the matching CSV at `data/kg.csv`. PrimeKG is not included in this repository.

## 2. Environment

The project was developed with Python 3.11 in the `yuxun` conda environment. Python package versions are listed in `requirements.txt`. Use a PyTorch build compatible with your NVIDIA driver and verify CUDA before training:

```bash
conda run -n yuxun python -c "import torch; assert torch.cuda.is_available()"
conda run -n yuxun python run_experiments.py clindc --setting chronological --seeds 42 --dry-run
```

The dry run verifies input paths and prints commands without fitting a model. Run commands from the directory containing this README; generated checkpoints and metrics go under `outputs/`.

## 3. Processed inputs in this repository

| Path under `data/` | Use |
| --- | --- |
| `dataset_v2_ontology/drug_disease_drug_kg_mapped.csv` | Chronological train/validation/test split: 20,314 / 2,890 / 2,922 triples. |
| `dataset_v2_ontology_aware_ldo/drug_disease_drug_kg_mapped.csv` | Ontology-aware leave-disease-out split: 19,594 / 2,613 / 3,919 triples; `ontology_groups.csv` records disease groups. |
| `drug_structures/v2_mapped/morgan.npz` | Aligned 2,048-bit Morgan fingerprints, with manifest and audit table. |
| `dataset_v2_structure_only_kg_unmapped/` | Structure-only drug-cold split and 156 cold-drug identifiers. |
| `dataset_v2_ontology/drug_disease_drug.csv` | Complete processed CDCDB table used for case-study novelty filtering. |

The precomputed ontology-aware split and `ontology_groups.csv` can be used for model experiments without the MONDO source file. To rebuild disease alignment or ontology groups from scratch, obtain the matching [MONDO release](https://mondo.monarchinitiative.org/pages/download/) and place it at `data/ontologies/mondo-2026-07-06.obo`.

## 4. Reproduce ClinDC experiments

All model runs use seeds `0 1 2 3 4 5 6 7 8 42`. The selected reference configuration has embedding dimension 128, decoder hidden dimension 512, training negative ratio 30, and evidence-weight coefficient 0.60. Checkpoint selection uses validation MRR. Test ranking scores all eligible candidates and filters known positive unordered pairs; macro AUROC/AUPR are computed over anchor-drug–disease queries.

```bash
# validation sensitivity analysis of the two dimensions.
conda run -n yuxun python run_experiments.py dimensions --resume

# ClinDC on chronological and ontology-aware splits.
conda run -n yuxun python run_experiments.py clindc --setting both --resume

# Reported ClinDC ablations on both splits.
conda run -n yuxun python run_experiments.py ablations --setting both --resume

# Structure-only one-drug-cold control (fixed 156-candidate pool).
conda run -n yuxun python run_experiments.py cold --resume

# Aggregate seed means, standard deviations, and paired tests.
conda run -n yuxun python summarize_results.py outputs/reproduction/clindc/results.csv
conda run -n yuxun python summarize_results.py outputs/reproduction/ablations/results.csv
```

## 5. Essential-hypertension case study

ClinDC ranks partners for hydrochlorothiazide in the essential-hypertension context and applies disease-context novelty filtering. This ranking uses the repository's processed inputs plus PrimeKG; for the ontology-based filtering option, it also needs the matching MONDO source file.

The independent target/pathway and PPI analyses require additional source files that **are not included in this repository**. Obtain the appropriate versions directly from [STITCH](https://stitch.embl.de/) (`data/external_case_study/raw/9606.protein_chemical.links.v5.0.tsv.gz`), [STRING](https://string-db.org/) (`data/external_case_study/raw/9606.protein.info.v12.0.txt.gz`), [KEGG](https://www.kegg.jp/) (`data/external_case_study/raw/kegg_human_pathway_gene_links_2026-08-25.txt` and `kegg_human_pathways_2026-08-25.txt`), [HGNC](https://hgnc.genenames.org/) (`data/external_hypertension_validation/raw/hgnc_complete_set_2026-08-22.txt`), and [HIPPIE](https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/download.php) (`data/external_hypertension_validation/raw/hippie_v2.4_2026-04-09.txt.gz`). The disease-gene input `data/EH_genes.xlsx` was derived from a user-supplied [DisGeNET](https://www.disgenet.com/) export using ScoreGDA ≥ 0.5; it is not redistributed. Users must obtain it under the provider's terms and reproduce the filtering described in the Supplementary Information.

After obtaining these inputs, run:

```bash
conda run -n yuxun python my_model/run_clindc_full_data_discovery.py --resume
conda run -n yuxun python my_model/analyze_clindc_hctz_hypertension.py \
  --checkpoint-root outputs/clindc_full_data_hctz_hypertension \
  --run-root outputs/clindc_full_data_hctz_essential_hypertension_tiered_novelty \
  --disease "essential hypertension" --novelty-filter essential_hypertension_tiered --top-k 30
conda run -n yuxun python my_model/analyze_case_study_enrichment.py
conda run -n yuxun python my_model/analyze_case_study_ppi.py
conda run -n yuxun python my_model/plot_essential_hypertension_analysis.py
```




