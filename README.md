# ProSIFT: A Nextflow Pipeline for Discovery Proteomics Analysis and Multi-Database Protein Annotation

ProSIFT (PROtein Statistical Integration and Filtering Tool) is an end-to-end Nextflow DSL2 pipeline that takes a protein-level abundance matrix from a discovery proteomics search (e.g., DIA-NN, MaxQuant, Proteome Discoverer) through input validation, UniProt ID mapping, pre-normalization QC, normalization, missingness-aware imputation, differential abundance testing (limma + DEqMS), functional enrichment (ORA and preranked GSEA via gseapy against MSigDB), and parallel annotation against five external databases (UniProt, PubMed, DisGeNET, DGIdb, CTD). It emits a structured results directory of Parquet tables, CSVs, static and interactive diagnostic plots, and per-module HTML reports intended to feed a downstream SQLite annotation database and interactive exploration frontend.

## Lab

Developed in the [Blanco-Suarez Lab](https://www.blancosuarezlab.com/) at San Diego State University. Manuscript in preparation.

## Biological Context

Discovery proteomics produces wide, sparse abundance matrices where the interesting biology is often hidden behind heavy missingness, peptide-count-dependent variance, and a long list of proteins with no obvious functional link to the phenotype under study. A typical case-control experiment yields a ranked list of differentially abundant proteins, but translating that list into mechanistic hypotheses requires integrating orthogonal evidence: pathway enrichment, disease associations, drug-target status, chemical interactions, and the existing primary literature. ProSIFT automates that integration for a pairwise contrast study in mouse tissue, preserving provenance from raw abundances through statistical calls to annotated, queryable per-protein profiles.

## Pipeline Overview

```
                       abundance.csv + metadata.csv + params.yml
                                          |
                                          v
                    +-------------------------------------------+
                    |   Module 01: Validation & ID Mapping      |
                    |  validate -> detection filter -> UniProt  |
                    |             (+ BioMart mouse->human)      |
                    +-------------------------------------------+
                               |                          \
                               v                           \
                 +------------------------+                 \
                 |   Module 02: Pre-norm   |                 \
                 |   QC/EDA (PCA, boxplots,|                  \
                 |   correlations, flags)  |                   \
                 +------------------------+                    \
                               |                                \
                               v                                 v
                 +------------------------+            +----------------------+
                 |   Module 03: Normalize  |           |  Module 06: Database |
                 |   (median/quantile/VSN) |           |  Queries (parallel)  |
                 |           ||            |           |   - UniProt          |
                 |   Module 03: Impute     |           |   - PubMed (PMI)     |
                 |   (MinProb + KNN,       |           |   - DisGeNET         |
                 |    MNAR/MAR-aware)      |           |   - DGIdb            |
                 +------------------------+            |   - CTD (offline)    |
                               |                       +----------------------+
                               v
                 +------------------------+
                 |   Module 03b: QC       |
                 |   Report Assembly      |
                 |   (consolidated HTML)  |
                 +------------------------+
                               |
                               v
                 +------------------------+
                 |   Module 04: Diff.     |
                 |   Abundance (limma +   |
                 |   DEqMS via rpy2)      |
                 +------------------------+
                               |
                               v
                 +------------------------+
                 |   Module 05: Enrich.   |
                 |   (ORA + preranked     |
                 |   GSEA, MSigDB, rrvgo) |
                 +------------------------+
                               |
                               v
                     results/{run_id}/*.parquet, *.csv, *.html, *.png
```

All eight modules run under Nextflow (DSL2) with a single `nextflow run` invocation. The only manual step is a one-time bulk download of the CTD chemical-gene interactions file (`CTD_chem_gene_ixns.tsv.gz`), which is too large and too infrequently updated to hit via an API on every run. Interactive setup helpers (`scripts/generate_run_config.py`, `scripts/prepare_prosift_input.py`) are provided to turn a raw search-engine output into the per-run samplesheet, metadata, and params files, but are not part of the Nextflow workflow itself.

## Getting Started

### Requirements

The pipeline targets Python 3.12, R 4.3+, and Nextflow DSL2. All dependencies, including the R packages needed by rpy2 (limma, DEqMS, rrvgo, GO.db, org.Mm.eg.db, GOSemSim), are pinned in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate prosift
```

### Configuration

Three things need to exist before a run:

1. **Samplesheet** (`samplesheet.csv`): one row per contrast, pointing to an abundance matrix, a metadata file, and a params file. Run:
   ```bash
   python scripts/generate_run_config.py -i abundance.csv -m metadata.csv -o run_config.yml
   python scripts/prepare_prosift_input.py --config run_config.yml
   ```
   to produce draft per-run inputs from a master abundance file and sample-level metadata.
2. **API keys** for NCBI E-utilities (optional, raises PubMed rate limit from 3/sec to 10/sec) and DisGeNET (required for the DisGeNET module). Copy `.env.example` to `.env` and fill in both, then `source .env && export NCBI_API_KEY DISGENET_API_KEY` before launching Nextflow. `.env` is gitignored.
3. **CTD bulk file**: download `CTD_chem_gene_ixns.tsv.gz` from http://ctdbase.org/downloads/ and place it in the cache directory referenced by `databases.ctd.cache_dir` in `params.yml`.

### Usage

Local (conda profile):

```bash
nextflow run main.nf -profile conda,local --samplesheet samplesheet.csv --outdir results
```

HPC cluster (Torque/PBS):

```bash
nextflow run main.nf -profile conda,pbs --samplesheet samplesheet.csv --outdir results
```

To resume after a failure or partial run without re-running completed processes:

```bash
nextflow run main.nf -profile conda,local --samplesheet samplesheet.csv -resume
```

### Input Format

The abundance matrix is a wide CSV with a `protein_id` column (UniProt accessions), one abundance column per sample (numeric), and optionally a matched set of peptide-count columns used by DEqMS for variance modeling. The metadata file is one row per sample, with `sample_id` matching the abundance column headers and a grouping column referenced in `params.yml`. Module 01 cross-validates sample IDs between the two files and proceeds only on the matched working set.

## Project Structure

```
ProSIFT/
├── main.nf                          Nextflow entry point
├── nextflow.config                  Executor profiles (local, pbs, conda, singularity)
├── conf/base.config                 Per-process resource allocations and retry policy
├── workflows/prosift.nf             Top-level DSL2 workflow; wires all modules together
├── modules/local/                   One Nextflow module per process (15 total)
├── bin/                             Per-process Python scripts (Nextflow auto-adds to PATH)
│   ├── validate_inputs.py           Module 01: schema + cross-validation
│   ├── filter_proteins.py           Module 01: per-group detection filter
│   ├── missingness_report.py        Module 01: advisory missingness plots
│   ├── uniprot_mapping.py           Module 01: UniProt REST + BioMart ortholog mapping
│   ├── prenorm_qc.py                Module 02: pre-normalization QC/EDA + HTML report
│   ├── normalize.py                 Module 03: log2 + median/quantile/VSN
│   ├── impute.py                    Module 03: MNAR/MAR-aware MinProb + KNN
│   ├── qc_report_assembly.py        Module 03b: consolidated QC HTML
│   ├── differential_abundance.py    Module 04: limma + DEqMS via rpy2
│   ├── enrichment.py                Module 05: ORA + preranked GSEA; rrvgo reduction
│   ├── query_uniprot.py             Module 06: UniProt REST annotations
│   ├── query_pubmed.py              Module 06: PubMed co-occurrence + PMI score
│   ├── query_disgenet.py            Module 06: DisGeNET gene-disease associations
│   ├── query_dgidb.py               Module 06: DGIdb drug-gene interactions (GraphQL)
│   ├── query_ctd.py                 Module 06: CTD chemical-gene interactions (offline)
│   ├── prosift_cache.py             Shared per-protein JSON cache utility
│   └── prosift_plot_utils.py        Shared plotting helpers (boxplots, PCA, clustermap)
├── scripts/                         Interactive setup helpers (run once per study)
│   ├── generate_run_config.py       Column detection + run definition -> run_config.yml
│   ├── prepare_prosift_input.py     run_config.yml -> per-run metadata + params
│   └── prepare_ctx_data.py          Dataset-specific conversion for the CTX synaptosome study
├── tests/test_prosift_cache.py      pytest suite for the shared cache layer
├── environment.yml                  Conda env specification (pinned)
├── .env.example                     API key template (real .env is gitignored)
├── databases/                       Local bulk-download databases, e.g. CTD (not tracked)
├── prosift_inputs/                  Per-study inputs: samplesheets, metadata, params.yml (not tracked)
├── results/                         Pipeline outputs, one subdirectory per run (not tracked)
├── work/                            Nextflow work directory (not tracked)
├── project_documentation/           Project documentation (not tracked)
└── CTX_synaptosome_project_data/    Unpublished source data (not tracked)
```

## Key Design Decisions

**Detection filter is a hard gate, fold-change is a flag.** Proteins that fail per-group detection thresholds are removed outright (Module 01), but fold-change magnitude is attached as a column on the differential abundance table rather than used to subset significant hits. This keeps the statistical test (adjusted p-value on the moderated t-statistic) as the single inclusion criterion downstream, avoids the inflated false-negative rate of a joint p+|FC| cutoff on small-n studies, and lets the user re-threshold fold change at the frontend without re-running the pipeline.

**Dual imputation driven by the detection filter, not by observed missingness patterns.** Proteins flagged SINGLE-GROUP by Module 01 (present in one group, absent in the other) are treated as missing-not-at-random in their absent group and imputed with MinProb (left-censored draw); proteins flagged PASSED with sporadic NaN are treated as missing-at-random and imputed with protein-wise KNN. This follows the DEP package convention and avoids the failure mode of KNN-only pipelines, which fabricate small, plausible values for proteins that are genuinely absent in one condition and thereby collapse real biological signal. A per-cell imputation mask is emitted so downstream reports can visually separate observed from imputed values.

**limma + DEqMS rather than a Welch t-test per protein.** With typical n = 3-5 per group, per-protein variance estimates are unstable and multiple testing over ~5000 proteins amplifies that noise. limma's empirical-Bayes shrinkage borrows strength across proteins, and DEqMS adds a peptide-count-aware variance correction that is particularly important for label-free and DIA data where low-peptide proteins have systematically inflated residual variance. Implemented via rpy2 rather than re-implemented in Python to preserve the reference implementation and its downstream compatibility with the Bioconductor ecosystem (e.g., rrvgo for GO-term redundancy reduction in Module 05).

**Two-layer ID mapping: native-organism symbols for analysis, human orthologs for annotation.** The pipeline supports mouse studies but most external disease, drug, and literature databases (DisGeNET, DGIdb, portions of CTD) are human-centric. Module 01 performs primary UniProt mapping in the native organism, then runs an Ensembl BioMart ortholog query to attach human gene symbols and Entrez IDs. Module 06 gates database queries on ortholog presence and labels each hit with `query_organism` so downstream consumers can distinguish mouse-native evidence from cross-species inferences.

**Application-level per-protein JSON cache beneath Nextflow's process cache.** Nextflow's `-resume` caches at the process level, which does not help when only a handful of proteins have changed between runs of the same study. The shared `prosift_cache` module writes one JSON per protein per database with age-based invalidation (default 30 days), keyed on the API-specific identifier. This makes incremental expansion of a study (adding a new contrast, adding a new database, re-querying after a DisGeNET version bump) cheap in API budget and respectful of rate limits, without giving up Nextflow's coarser provenance guarantees.

## Tools and Libraries

Nextflow (DSL2, pbs (Torque) and local executors), Python 3.12, pandas, numpy, pyarrow (Parquet intermediates), pyyaml, requests, tqdm, scipy, scikit-learn (KNNImputer, IterativeImputer), gseapy (enrich, prerank), rpy2, R 4.3 with Bioconductor limma, DEqMS, rrvgo, GOSemSim, GO.db, org.Mm.eg.db, matplotlib, seaborn, plotly, python-kaleido (static PNG export), pytest, Ensembl BioMart (XML REST), UniProt REST (/uniprotkb/search, /idmapping), NCBI E-utilities (ESearch), DisGeNET REST, DGIdb GraphQL v5.0, CTD bulk TSV, MSigDB GMT libraries, conda, Singularity/Apptainer.

## Author

**Reina Hastings** - [GitHub](https://github.com/reinahastings) - Blanco-Suarez Lab, San Diego State University
