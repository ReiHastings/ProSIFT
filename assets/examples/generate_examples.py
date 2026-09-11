#!/usr/bin/env python3
# title: generate_examples.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-09-01
# last modified: 2026-09-01
#
# purpose:
#   Generates the shipped minimal example dataset used by `nextflow run main.nf
#   -profile test`. Emits a small synthetic abundance matrix with deliberately
#   planted differential-abundance effects, the matching metadata, a tiny GMT
#   built from the example's own gene symbols, and a ground-truth table naming
#   which proteins were spiked and in which direction. The data is fully
#   synthetic and regenerable, so the example is provably not real lab data and
#   effect sizes can be retuned without hand-editing CSVs.
#
# inputs:
#   None (all parameters are constants below; --seed overrides the RNG seed)
#
# outputs:
#   assets/examples/minimal/abundance.csv           wide matrix, raw intensities
#   assets/examples/minimal/metadata.csv            sample_id + condition
#   assets/examples/minimal/example_gene_sets.gmt   4 synthetic gene sets
#   assets/examples/minimal/ground_truth.csv        planted effects, for checking
#   assets/examples/minimal/samplesheet.csv         one-row samplesheet (relative paths)
#   assets/examples/minimal/params.yml              example params (absolute GMT path)
#   assets/examples/minimal/ProSIFT_input_template.xlsx  human-facing template
#
# status:
#   2026-09-01 -- THE GENERATED EXAMPLE DOES NOT RUN. Reviewed and failed the
#   same day. Blocking defect: 40 proteins against the 50-protein floor at
#   bin/filter_proteins.py:583. See assets/examples/KNOWN_ISSUES.md for the
#   full finding list before changing anything here.
#
# usage example:
#   python assets/examples/generate_examples.py --outdir assets/examples/minimal

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. Example study design
# ============================================================
# Two groups, three replicates each. This is the smallest design that still
# exercises DEqMS variance moderation and leaves room for a per-group detection
# threshold of 2 (see qc.min_detections_per_group in the example params.yml).

GROUPS: Dict[str, List[str]] = {
    'CTRL':  ['CTRL_1', 'CTRL_2', 'CTRL_3'],
    'TREAT': ['TREAT_1', 'TREAT_2', 'TREAT_3'],
}
SAMPLES: List[str] = [s for ids in GROUPS.values() for s in ids]


# ============================================================
# 2. Protein panel
# ============================================================
# Real mouse gene symbols grouped into four coherent functional blocks.
#
# [FOLLOW-UP NEEDED] The accessions paired with these symbols are NOT synthetic.
# build_panel() emits P00001-P00040, which is the historical cytochrome c
# accession block (P00001 = CYC_HUMAN, P00009 = CYC_MOUSE). They resolve against
# UniProt, they carry the wrong gene symbols for this example, and they overlap
# the GMT below in zero genes. Earlier comments here claimed they were synthetic
# placeholders; that was wrong. See assets/examples/KNOWN_ISSUES.md E-3.
#
# Nor is the example offline: UNIPROT_MAPPING is not gated by databases.enabled
# and is a hard dependency of Module 04. See KNOWN_ISSUES.md E-2.

GENE_BLOCKS: Dict[str, List[str]] = {
    'EXAMPLE_SYNAPTIC_VESICLE_CYCLE': [
        'Snap25', 'Syn1', 'Syt1', 'Stx1a', 'Vamp2',
        'Sypl1', 'Dlg4', 'Nrxn1', 'Nlgn2', 'Cplx1',
        'Rab3a', 'Syngap1',
    ],
    'EXAMPLE_MITOCHONDRIAL_RESPIRATION': [
        'Ndufa9', 'Sdha', 'Uqcrc1', 'Cox4i1', 'Atp5f1a',
        'Cs', 'Idh3a', 'Mdh2', 'Aco2', 'Slc25a4',
    ],
    'EXAMPLE_ASTROCYTE_MARKERS': [
        'Gfap', 'Aqp4', 'Slc1a2', 'Slc1a3', 'S100b',
        'Aldh1l1', 'Vim', 'Sox9',
    ],
    'EXAMPLE_CYTOSKELETON_AND_GLYCOLYSIS': [
        'Actb', 'Tubb3', 'Map2', 'Nefl', 'Nefm',
        'Sptbn1', 'Ank2', 'Camk2a', 'Gapdh', 'Eno2',
    ],
}


# ============================================================
# 3. Planted effects (the ground truth)
# ============================================================
# [FOLLOW-UP NEEDED] Concentrating effects in whole blocks was intended to give
# ORA and GSEA something coherent to find. It is a DEFECT: it shifts 18/40 (45%)
# of proteins coordinately, which breaks median normalization's assumption that
# most proteins are unchanged (see .claude/review-domain-checklist.md). Measured
# result: up-block +3.15 not +2.5, down-block -1.47 not -2.0, and a +0.3 to +0.4
# log2 bias on every null protein. Fix: spike 15-20%, balanced up and down, with
# within-block variation. See KNOWN_ISSUES.md E-5.
#
# Effect sizes are large on purpose. With n=3 per group and BH correction over
# ~40 tests, a subtle effect would not survive, and a test profile that produces
# zero significant hits is a poor smoke test.

EFFECTS: Dict[str, float] = {
    'EXAMPLE_ASTROCYTE_MARKERS':        2.5,   # up in TREAT
    'EXAMPLE_MITOCHONDRIAL_RESPIRATION': -2.0,  # down in TREAT
}

# Baseline abundance model (log2 scale) and technical noise.
BASELINE_MEAN_LOG2 = 22.0
BASELINE_SD_LOG2 = 1.8
REPLICATE_NOISE_SD = 0.35

# Missingness. MAR is uniform at random; MNAR is applied preferentially to the
# low-abundance tail, which is the pattern Module 03's mixed imputation expects.
MAR_RATE = 0.05
MNAR_LOW_ABUNDANCE_QUANTILE = 0.20
MNAR_RATE_IN_TAIL = 0.30


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate the ProSIFT minimal example dataset.'
    )
    parser.add_argument('--outdir', default='assets/examples/minimal',
                        help='Output directory (default: assets/examples/minimal)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (default: 42). Change to reroll the data.')
    parser.add_argument('--no-xlsx', action='store_true',
                        help='Skip writing the .xlsx template (openpyxl not needed)')
    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def build_panel() -> pd.DataFrame:
    '''
    Build the protein panel: one row per protein with a synthetic accession,
    its real gene symbol, and the functional block it belongs to.

    [FOLLOW-UP NEEDED] These accessions are NOT made up. P00001-P00040 are real
    UniProt cytochrome c entries. Replace with a namespace verified to return
    zero UniProt hits. See assets/examples/KNOWN_ISSUES.md E-3.
    '''
    rows = []
    idx = 0
    for block, genes in GENE_BLOCKS.items():
        for gene in genes:
            # Deterministic synthetic accession, e.g. P00001 ... P00040.
            accession = f'P{idx + 1:05d}'
            rows.append({'protein_id': accession, 'gene_symbol': gene,
                         'gene_set': block})
            idx += 1
    return pd.DataFrame(rows)


def simulate_abundances(
    panel: pd.DataFrame,
    rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Simulate log2 abundances and peptide counts.

    Algorithm:
      1. Draw a per-protein baseline log2 intensity.
      2. Add the planted group effect to TREAT samples for spiked blocks.
      3. Add independent replicate noise to every cell.
      4. Punch missing values (MAR everywhere, MNAR in the low-abundance tail).
      5. Draw peptide counts that scale with abundance, zeroed where missing.

    Returns (log2_matrix, peptide_counts), both indexed like `panel`.
    '''
    n_proteins = len(panel)

    # --- Step 1: per-protein baseline ---
    baseline = rng.normal(BASELINE_MEAN_LOG2, BASELINE_SD_LOG2, n_proteins)

    # --- Step 2 + 3: group effect and replicate noise ---
    effect_per_protein = panel['gene_set'].map(EFFECTS).fillna(0.0).to_numpy()

    log2_vals = np.zeros((n_proteins, len(SAMPLES)))
    for j, sample in enumerate(SAMPLES):
        is_treat = sample in GROUPS['TREAT']
        shift = effect_per_protein if is_treat else 0.0
        log2_vals[:, j] = (baseline + shift
                           + rng.normal(0.0, REPLICATE_NOISE_SD, n_proteins))

    # --- Step 4: missingness ---
    # MAR: uniform random dropout across the whole matrix.
    mar_mask = rng.random(log2_vals.shape) < MAR_RATE

    # MNAR: extra dropout confined to the low-abundance tail, defined on the
    # per-protein baseline so the pattern is abundance-dependent by construction.
    low_cut = np.quantile(baseline, MNAR_LOW_ABUNDANCE_QUANTILE)
    in_tail = (baseline <= low_cut)[:, None]
    mnar_mask = in_tail & (rng.random(log2_vals.shape) < MNAR_RATE_IN_TAIL)

    missing = mar_mask | mnar_mask

    # Guard: never let a protein fall below the example's own detection
    # threshold (2 per group), otherwise Module 01 filters it out and the
    # ground-truth table would name proteins that never reach the test.
    for i in range(n_proteins):
        for group_samples in GROUPS.values():
            cols = [SAMPLES.index(s) for s in group_samples]
            n_missing = missing[i, cols].sum()
            if len(cols) - n_missing < 2:
                # Restore cells until the group has 2 detections.
                for c in cols:
                    if missing[i, c]:
                        missing[i, c] = False
                        if len(cols) - missing[i, cols].sum() >= 2:
                            break

    log2_vals[missing] = np.nan

    # --- Step 5: peptide counts ---
    # DEqMS weights by peptide/PSM count, so the counts must covary with
    # abundance for the example to exercise the weighting realistically.
    scaled = (baseline - baseline.min()) / max(np.ptp(baseline), 1e-9)
    lam = 2.0 + 18.0 * scaled
    counts = rng.poisson(lam[:, None], size=log2_vals.shape).astype(float)
    counts = np.clip(counts, 1, None)
    counts[missing] = 0  # no detection means no peptides

    log2_df = pd.DataFrame(log2_vals, columns=SAMPLES)
    counts_df = pd.DataFrame(counts.astype(int), columns=SAMPLES)
    return log2_df, counts_df


def write_abundance(
    panel: pd.DataFrame,
    log2_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    outdir: str
) -> str:
    '''
    Write the wide abundance CSV in ProSIFT's expected layout: protein_id,
    then abundance_<sample> columns, then peptide_count_<sample> columns, and
    nothing else.

    Values are written as RAW intensities (2 ** log2), matching
    input.abundance_type: 'raw' in the example params.yml.
    '''
    # Deliberately nothing but protein_id + the two prefixed column blocks:
    # this file is the canonical reference for what an abundance CSV looks
    # like, so it carries no annotation columns. The demonstration that extra
    # columns are safely ignored lives in the .xlsx template instead, which is
    # the authoring aid rather than the reference.
    out = pd.DataFrame({'protein_id': panel['protein_id']})

    raw = np.power(2.0, log2_df)
    for s in SAMPLES:
        out[f'abundance_{s}'] = raw[s].round(2)
    for s in SAMPLES:
        out[f'peptide_count_{s}'] = counts_df[s]

    path = os.path.join(outdir, 'abundance.csv')
    out.to_csv(path, index=False)
    return path


def write_metadata(outdir: str) -> str:
    '''Write one row per sample: sample_id plus the grouping column.'''
    rows = [{'sample_id': s, 'condition': g}
            for g, ids in GROUPS.items() for s in ids]
    path = os.path.join(outdir, 'metadata.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_gmt(panel: pd.DataFrame, outdir: str) -> str:
    '''
    Write a tiny GMT built from the example's own gene symbols.

    Format is standard GMT: set name, description, then tab-delimited members.
    Sets are small (8-12 genes), so the example params.yml lowers
    enrichment.min_gene_set_size accordingly.
    '''
    path = os.path.join(outdir, 'example_gene_sets.gmt')
    with open(path, 'w', encoding='utf-8') as f:
        for block, genes in GENE_BLOCKS.items():
            desc = 'ProSIFT synthetic example gene set'
            f.write('\t'.join([block, desc] + genes) + '\n')
    return path


def write_ground_truth(panel: pd.DataFrame, outdir: str) -> str:
    '''
    Write the record of which proteins were spiked and by how much.

    [FOLLOW-UP NEEDED] The expected_significant column is WRONG. It claims 18
    proteins; the pipeline calls 19, because median normalization biases the
    null proteins upward and pushes P00034 over fc_threshold. Planted effect and
    called effect need to be separate columns, with the latter generated from an
    actual pipeline run. See assets/examples/KNOWN_ISSUES.md E-6.
    '''
    gt = panel.copy()
    gt['planted_log2fc'] = gt['gene_set'].map(EFFECTS).fillna(0.0)
    gt['expected_significant'] = gt['planted_log2fc'] != 0.0
    gt['expected_direction'] = np.select(
        [gt['planted_log2fc'] > 0, gt['planted_log2fc'] < 0],
        ['up_in_TREAT', 'down_in_TREAT'],
        default='none'
    )
    path = os.path.join(outdir, 'ground_truth.csv')
    gt.to_csv(path, index=False)
    return path


def write_samplesheet(outdir: str) -> str:
    '''
    Write the example samplesheet with RELATIVE paths.

    Relative entries are resolved against the samplesheet's own directory by
    workflows/prosift.nf, so this file works from any launch directory and
    survives being copied or rsynced to the cluster.
    '''
    path = os.path.join(outdir, 'samplesheet.csv')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('run_id,abundance,metadata,params\n')
        f.write('example_CTRL_vs_TREAT,abundance.csv,metadata.csv,params.yml\n')
    return path


def write_params_yml(outdir: str) -> str:
    '''
    Write the example params.yml.

    Note on the GMT path: enrichment.py reads enrichment.gene_set_libraries
    directly from params.yml at runtime and Nextflow does not stage those files
    (see workflows/prosift.nf), so the path must be absolute and must resolve on
    the machine that runs the pipeline. It is therefore written as an absolute
    path resolved AT GENERATION TIME, which is why this file is regenerated
    rather than hand-edited after cloning the repo to a new location.
    '''
    gmt_abs = os.path.abspath(os.path.join(outdir, 'example_gene_sets.gmt'))
    content = f'''# ProSIFT params -- minimal shipped example
# GENERATED by assets/examples/generate_examples.py. Do not hand-edit;
# re-run the generator instead (the GMT path below is machine-specific).

project:
  name: "example_CTRL_vs_TREAT"
  organism: "mouse"

input:
  abundance_matrix: "abundance.csv"
  metadata: "metadata.csv"
  format: "csv"
  protein_id_column: "protein_id"
  # REQUIRED and never guessed. The generator emits raw intensities (2 ** log2).
  # Setting this wrong silently changes which values are treated as missing:
  # on 'raw', values <= 0 become NaN; on 'log2'/'normalized', 0 is a real value.
  abundance_type: "raw"
  abundance_prefix: "abundance_"
  peptide_count_prefix: "peptide_count_"

design:
  group_column: "condition"
  covariates: []
  batch_column: null
  contrasts:
    - "TREAT_vs_CTRL"

qc:
  min_samples_per_group: 2
  min_detections_per_group: 2
  min_detections_present_group: null

databases:
  # Empty on purpose: the example runs fully offline. The synthetic accessions
  # (P00001 ...) would not resolve against UniProt anyway. Add database names
  # here only when pointing this params file at real data.
  enabled: []
  query_scope: 'all'
  cache_dir: './prosift_cache/databases'
  cache_days: 30
  force_requery: false

normalization:
  method: "median"

imputation:
  mode: "mixed"
  mnar_method: "minprob"
  mar_method: "knn"
  minprob_quantile: 0.01
  minprob_scale: 0.3
  # k must be < the number of samples; the example has only 6.
  knn_k: 3
  random_seed: 42

differential_abundance:
  method: "deqms"
  significance:
    fdr_threshold: 0.05
    fc_threshold: 1.0

enrichment:
  run_ora:  true
  run_gsea: true
  gene_set_libraries:
    - "{gmt_abs}"
  background:           "detected"
  gsea_ranking:         "t_statistic"
  # Lowered from the usual 15/500: the example's sets are 8-12 genes, so the
  # default minimum would filter every set out and Module 05 would return empty.
  min_gene_set_size:    3
  max_gene_set_size:    500
  fdr_threshold:        0.05
  plot_top_n:           10
  plot_top_gsea_traces: 4
'''
    path = os.path.join(outdir, 'params.yml')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def write_xlsx_template(
    panel: pd.DataFrame,
    log2_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    outdir: str
) -> str:
    '''
    Write the human-facing .xlsx template: a documentation sheet plus the data
    sheets.

    IMPORTANT: ProSIFT cannot read .xlsx. validate_inputs.infer_separator()
    branches on file extension and falls through to reading the file as UTF-8
    text, so handing an .xlsx to the pipeline raises a decode error rather than
    a clean validation message. The workbook is a authoring aid only; the sheets
    must be exported to CSV before running. The README sheet says so first.
    '''
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()

    # --- Sheet 1: read me first ---
    ws = wb.active
    ws.title = 'READ ME FIRST'
    notes = [
        ('ProSIFT input template', True),
        ('', False),
        ('STEP 1. Fill in the "abundance" and "metadata" sheets.', False),
        ('STEP 2. Export EACH sheet separately to CSV (File > Save As > CSV).', False),
        ('STEP 3. Point samplesheet.csv at the exported CSV files.', False),
        ('', False),
        ('ProSIFT CANNOT read .xlsx files. You must export to CSV.', True),
        ('', False),
        ('Rules that will break the run if ignored:', True),
        ('  - Do NOT add comment or note ROWS to the abundance sheet. Every row', False),
        ('    is read as a protein. Two note rows collide as duplicate IDs and', False),
        ('    the run fails.', False),
        ('  - Extra note COLUMNS are fine. Anything not starting with', False),
        ('    "abundance_" or "peptide_count_" is ignored.', False),
        ('  - Every sample with an abundance_ column needs a matching', False),
        ('    peptide_count_ column, or omit peptide counts entirely.', False),
        ('  - sample_id in the metadata sheet must equal the abundance column', False),
        ('    name with the "abundance_" prefix removed, exactly.', False),
        ('  - Set input.abundance_type in params.yml to match your data', False),
        ('    (raw / log2 / normalized). ProSIFT will not guess, and getting it', False),
        ('    wrong silently changes which values are treated as missing.', False),
        ('  - Watch for Excel turning gene symbols into dates (SEPT7, MARCH1).', False),
        ('    Format those cells as Text before pasting.', False),
    ]
    for i, (text, bold) in enumerate(notes, start=1):
        cell = ws.cell(row=i, column=1, value=text)
        if bold:
            cell.font = Font(bold=True)
    ws.column_dimensions['A'].width = 78

    # --- Sheet 2: abundance ---
    ws2 = wb.create_sheet('abundance')
    raw = np.power(2.0, log2_df)
    header = (['protein_id', 'notes']
              + [f'abundance_{s}' for s in SAMPLES]
              + [f'peptide_count_{s}' for s in SAMPLES])
    ws2.append(header)
    for i in range(len(panel)):
        row = [panel['protein_id'].iloc[i],
               f"{panel['gene_symbol'].iloc[i]} ({panel['gene_set'].iloc[i]})"]
        row += [None if pd.isna(raw[s].iloc[i]) else round(float(raw[s].iloc[i]), 2)
                for s in SAMPLES]
        row += [int(counts_df[s].iloc[i]) for s in SAMPLES]
        ws2.append(row)
    for c in ws2[1]:
        c.font = Font(bold=True)

    # --- Sheet 3: metadata ---
    ws3 = wb.create_sheet('metadata')
    ws3.append(['sample_id', 'condition'])
    for g, ids in GROUPS.items():
        for s in ids:
            ws3.append([s, g])
    for c in ws3[1]:
        c.font = Font(bold=True)

    path = os.path.join(outdir, 'ProSIFT_input_template.xlsx')
    wb.save(path)
    return path


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    panel = build_panel()
    log2_df, counts_df = simulate_abundances(panel, rng)

    written = [
        write_abundance(panel, log2_df, counts_df, args.outdir),
        write_metadata(args.outdir),
        write_gmt(panel, args.outdir),
        write_ground_truth(panel, args.outdir),
        write_samplesheet(args.outdir),
        write_params_yml(args.outdir),
    ]
    if not args.no_xlsx:
        written.append(write_xlsx_template(panel, log2_df, counts_df, args.outdir))

    n_spiked = int((panel['gene_set'].map(EFFECTS).fillna(0.0) != 0).sum())
    print(f'Generated {len(panel)} proteins x {len(SAMPLES)} samples '
          f'(seed={args.seed}); {n_spiked} proteins carry a planted effect.')
    for p in written:
        print(f'  wrote {p}')


if __name__ == '__main__':
    main()
