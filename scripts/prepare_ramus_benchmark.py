#!/usr/bin/env python3
# title: prepare_ramus_benchmark.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-09-04
# last modified: 2026-09-04
#
# purpose:
#   Converts the MaxQuant reprocessing of the Ramus 2016 UPS1/yeast spike-in
#   benchmark (PRIDE PXD001819, reprocessed and deposited as PXD022169) into
#   ProSIFT input files.
#
#   The dataset is 9 UPS1 spike levels (50 to 50000 amol) x 3 replicates = 27
#   LC-MS runs against a constant yeast background. UPS1 proteins are the known
#   true positives; yeast proteins are the known true negatives and should show
#   no differential abundance. Because MaxQuant's proteinGroups.txt carries
#   per-run peptide counts, no peptide-count column has to be fabricated.
#
#   Writes one shared master abundance matrix covering all 27 samples, plus a
#   per-contrast metadata file and params.yml. Module 01 subsets the master
#   matrix to the samples named in each metadata file, so per-contrast abundance
#   copies are unnecessary.
#
# inputs:
#   - proteinGroups.txt   MaxQuant protein-level output (tab-delimited)
#
# outputs:
#   - {outdir}/ramus_abundance.csv                    master matrix, 27 samples
#   - {outdir}/ramus_ground_truth.csv                 protein_id + species label
#   - {outdir}/{run_id}/{run_id}_metadata.csv         per-contrast sample table
#   - {outdir}/{run_id}/{run_id}_params.yml           per-contrast parameters
#   - {outdir}/samplesheet_ramus.csv                  Nextflow samplesheet
#
# usage example:
#   python scripts/prepare_ramus_benchmark.py \
#     --protein-groups benchmark_datasets/ramus_dataset/PXD001819_MQ/combined/txt/proteinGroups.txt \
#     --outdir prosift_inputs/ramus_benchmark
#
#   copy/paste: python scripts/prepare_ramus_benchmark.py --protein-groups benchmark_datasets/ramus_dataset/PXD001819_MQ/combined/txt/proteinGroups.txt --outdir prosift_inputs/ramus_benchmark
#
#   Notes:
#   - Default abundance metric is MaxQuant 'Intensity' (un-normalized), so that
#     ProSIFT's own normalization step has real loading differences to correct.
#     Pass --lfq to use 'LFQ intensity' instead, which is already normalized by
#     MaxLFQ; if you do, set normalization.method to 'none' in the params files
#     to avoid normalizing twice.
#   - Contrasts default to four regimes spanning the spike ladder. Override with
#     --contrast HIGH:LOW (repeatable), using the amol level names.

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:
    print('Error: PyYAML is required. Install with: pip install pyyaml',
          file=sys.stderr)
    sys.exit(1)


# ============================================================
# Constants
# ============================================================

# The nine UPS1 spike levels in the deposit, in amol, as they appear inside the
# MaxQuant experiment names (e.g. '50000amol_R1').
SPIKE_LEVELS = ['50', '125', '250', '500', '2500',
                '5000', '12500', '25000', '50000']

# Default contrasts, chosen to span four distinct detection/effect regimes:
#   50000 vs 25000  ->  2x,    all UPS1 detected 3/3 in both groups
#   50000 vs  5000  ->  10x,   near-complete detection
#   50000 vs    50  ->  1000x, mostly presence/absence
#    2500 vs   500  ->  5x,    the detection boundary, exercises the anchor gate
DEFAULT_CONTRASTS = [('50000', '25000'), ('50000', '5000'),
                     ('50000', '50'), ('2500', '500')]

# Group labels used in every contrast. These become design-matrix column names
# in R, so they must be syntactically valid R names: a label like '50000amol'
# starts with a digit and would be rejected by parse_and_validate_contrasts().
GROUP_HIGH = 'high'
GROUP_LOW = 'low'

# MaxQuant flags marking rows that are not real protein measurements.
DISCARD_FLAGS = ['Reverse', 'Potential contaminant', 'Only identified by site']


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build ProSIFT inputs from the Ramus UPS1 MaxQuant output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--protein-groups', required=True, type=Path,
                        dest='protein_groups',
                        help='Path to MaxQuant proteinGroups.txt')
    parser.add_argument('--outdir', required=True, type=Path,
                        help='Output directory for ProSIFT input files')
    parser.add_argument('--lfq', action='store_true',
                        help="Use 'LFQ intensity' instead of 'Intensity'")
    parser.add_argument('--contrast', action='append', default=None,
                        metavar='HIGH:LOW',
                        help='Contrast as two amol level names, e.g. 50000:500. '
                             'Repeatable. Defaults to four spanning contrasts.')
    parser.add_argument('--organism', default='yeast',
                        help='Organism recorded in params.yml (default: yeast)')
    return parser.parse_args()


def resolve_contrasts(raw: 'list[str] | None') -> 'list[tuple[str, str]]':
    '''Parse --contrast arguments, or return the defaults.

    Fails loudly on an unknown spike level so a typo cannot silently produce a
    run with no samples.
    '''
    if not raw:
        return DEFAULT_CONTRASTS

    contrasts: list[tuple[str, str]] = []
    for item in raw:
        if ':' not in item:
            raise SystemExit(
                f"ERROR: contrast '{item}' must be in HIGH:LOW form, e.g. 50000:500"
            )
        high, low = item.split(':', 1)
        for level in (high, low):
            if level not in SPIKE_LEVELS:
                raise SystemExit(
                    f"ERROR: unknown spike level '{level}' in contrast '{item}'. "
                    f'Valid levels: {", ".join(SPIKE_LEVELS)}'
                )
        if high == low:
            raise SystemExit(
                f"ERROR: contrast '{item}' compares a level against itself."
            )
        contrasts.append((high, low))
    return contrasts


# ============================================================
# Step 1 -- Read and filter proteinGroups.txt
# ============================================================

def load_protein_groups(path: Path) -> pd.DataFrame:
    '''Read proteinGroups.txt and drop rows that are not real measurements.

    MaxQuant marks decoy hits ('Reverse'), contaminant-database hits
    ('Potential contaminant') and identifications supported only by a modified
    site ('Only identified by site') with a '+'. None of these are quantitative
    protein measurements, and leaving them in would contaminate both the
    true-negative set and the multiple-testing burden.
    '''
    if not path.is_file():
        raise SystemExit(f'ERROR: proteinGroups file not found: {path}')

    df = pd.read_csv(path, sep='\t', low_memory=False)
    n_start = len(df)

    for flag in DISCARD_FLAGS:
        if flag not in df.columns:
            raise SystemExit(
                f"ERROR: expected MaxQuant column '{flag}' not found. "
                f'Is {path.name} really a proteinGroups.txt?'
            )
        df = df[df[flag] != '+']

    print(f'  Read {n_start} protein groups, kept {len(df)} after dropping '
          f'reverse / contaminant / site-only rows')
    return df.reset_index(drop=True)


# ============================================================
# Step 2 -- Map MaxQuant run names to ProSIFT sample IDs
# ============================================================

def build_sample_map(df: pd.DataFrame, intensity_prefix: str) -> 'dict[str, str]':
    '''Return {maxquant_experiment_name: prosift_sample_id}.

    Two runs in this deposit carry a trailing underscore in their experiment
    name ('25000amol_R2_', '2500amol_R3_'), an artifact of the original raw file
    names. Strip it so sample IDs are uniform, and verify the result stays
    unique so the strip cannot silently collide two samples.
    '''
    experiments = [c[len(intensity_prefix):] for c in df.columns
                   if c.startswith(intensity_prefix) and c != intensity_prefix.strip()]
    if not experiments:
        raise SystemExit(
            f"ERROR: no per-run columns found with prefix '{intensity_prefix}'."
        )

    sample_map = {exp: exp.rstrip('_') for exp in experiments}

    if len(set(sample_map.values())) != len(sample_map):
        raise SystemExit(
            'ERROR: stripping trailing underscores collapsed two run names into '
            'one sample ID. Sample IDs must be unique.'
        )

    print(f'  Mapped {len(sample_map)} MaxQuant runs to sample IDs')
    return sample_map


def level_of(sample_id: str) -> str:
    '''Extract the amol spike level from a sample ID like "12500amol_R2".'''
    return sample_id.split('amol_')[0]


def extract_accession(protein_group: str) -> str:
    '''Reduce one MaxQuant protein group to a single representative accession.

    Two FASTA header conventions are mixed in this search, because the yeast
    background and the UPS1 spike-in came from separately built databases:

      yeast : 'sp|A5Z2X5|YP010_YEAST'      -> 3 fields, accession is field 2
      UPS1  : 'O00762ups|UBE2C_HUMAN_UPS'  -> 2 fields, accession is field 1

    A naive split on '|' taking the first field would return the literal 'sp'
    for every yeast protein and collapse 994 distinct proteins into one ID, so
    the field count has to drive the choice. The 'ups' suffix is preserved
    because build_ground_truth() uses it to label true positives.

    Semicolon-delimited groups are reduced to their first member first, which
    is the same representative-accession rule Module 01 applies.
    '''
    first_member = str(protein_group).split(';')[0].strip()
    fields = first_member.split('|')

    if len(fields) >= 3:
        return fields[1].strip()
    if len(fields) == 2:
        return fields[0].strip()
    return first_member


# ============================================================
# Step 3 -- Build the master abundance matrix
# ============================================================

def build_abundance(
    df: pd.DataFrame,
    sample_map: 'dict[str, str]',
    intensity_prefix: str,
) -> pd.DataFrame:
    '''Assemble the ProSIFT master abundance matrix.

    Column contract (Module 01):
      protein_id              representative accession
      abundance_{sample}      one per run
      peptide_count_{sample}  one per run, same sample IDs

    Two data conventions matter here:

    1. Protein groups. MaxQuant reports a semicolon-delimited group in
       'Majority protein IDs'. extract_accession() reduces each group to one
       representative accession, matching Module 01's first-member rule while
       also unwrapping the two different FASTA header formats in this search.
       Uniqueness is asserted here rather than left for validate_inputs.py to
       hard-stop on later.

    2. Missing values. MaxQuant writes 0, not blank, for a protein it did not
       quantify in a run. We leave the zeros in place: with
       abundance_type = 'raw', Module 01 converts non-positive values to NaN and
       records a provenance mask so imputation can treat them as
       below-detection (MNAR) rather than random dropout. Converting here would
       throw that provenance away.
    '''
    pep_prefix = 'Razor + unique peptides '

    out = pd.DataFrame()
    out['protein_id'] = df['Majority protein IDs'].map(extract_accession)

    dupes = out['protein_id'].duplicated(keep=False)
    if dupes.any():
        examples = out.loc[dupes, 'protein_id'].unique()[:5].tolist()
        raise SystemExit(
            f'ERROR: {int(dupes.sum())} duplicate protein IDs after taking the '
            f'first accession of each group. Examples: {examples}'
        )

    for exp, sample_id in sample_map.items():
        int_col = f'{intensity_prefix}{exp}'
        pep_col = f'{pep_prefix}{exp}'
        if pep_col not in df.columns:
            raise SystemExit(
                f"ERROR: no peptide count column '{pep_col}' for run '{exp}'. "
                f'Module 01 requires a peptide count column per abundance column.'
            )
        out[f'abundance_{sample_id}'] = df[int_col].values
        out[f'peptide_count_{sample_id}'] = df[pep_col].values

    # Drop proteins quantified in no run at all. MaxQuant emits these when a
    # protein is identified by MS/MS but never assigned an intensity; Module 01
    # hard-stops on an all-NA row, so remove them here with an explicit count
    # rather than letting the pipeline fail on them.
    abund_cols = [c for c in out.columns if c.startswith('abundance_')]
    all_zero = (out[abund_cols] == 0).all(axis=1)
    if all_zero.any():
        print(f'  Dropped {int(all_zero.sum())} protein(s) with zero intensity '
              f'in all {len(abund_cols)} runs')
        out = out[~all_zero].reset_index(drop=True)

    print(f'  Built abundance matrix: {len(out)} proteins x '
          f'{len(abund_cols)} samples')
    return out


# ============================================================
# Step 4 -- Ground truth
# ============================================================

def build_ground_truth(abundance: pd.DataFrame) -> pd.DataFrame:
    '''Label each protein UPS1 (true positive) or yeast (true negative).

    UPS1 accessions in this search carry a 'ups' suffix (e.g. 'P02787ups'),
    which is how the UPS1 FASTA was built. Every protein that is not UPS1 comes
    from the yeast background database and is expected to be invariant across
    spike levels.
    '''
    truth = pd.DataFrame({'protein_id': abundance['protein_id']})
    is_ups = truth['protein_id'].str.contains('ups', case=False, na=False)
    truth['species'] = pd.Series(['UPS1' if u else 'yeast' for u in is_ups])
    truth['expected_differential'] = is_ups

    n_ups = int(is_ups.sum())
    if n_ups != 48:
        print(f'  WARNING: expected 48 UPS1 proteins, found {n_ups}')
    print(f'  Ground truth: {n_ups} UPS1 true positives, '
          f'{len(truth) - n_ups} yeast true negatives')
    return truth


# ============================================================
# Step 5 -- Per-contrast metadata and params
# ============================================================

def build_metadata(sample_ids: 'list[str]', high: str, low: str) -> pd.DataFrame:
    '''Return sample_id + spike_level for the samples in one contrast.'''
    rows = []
    for sid in sample_ids:
        level = level_of(sid)
        if level == high:
            rows.append({'sample_id': sid, 'spike_level': GROUP_HIGH,
                         'amol': int(level)})
        elif level == low:
            rows.append({'sample_id': sid, 'spike_level': GROUP_LOW,
                         'amol': int(level)})

    meta = pd.DataFrame(rows)
    for label, level in [(GROUP_HIGH, high), (GROUP_LOW, low)]:
        n = int((meta['spike_level'] == label).sum())
        if n == 0:
            raise SystemExit(
                f"ERROR: no samples found for spike level {level} amol."
            )
    return meta


def build_params(
    run_id: str,
    high: str,
    low: str,
    organism: str,
    abundance_relpath: str,
    use_lfq: bool,
) -> dict:
    '''Assemble the params.yml content for one contrast.

    Deliberate settings for a benchmark run:
      - abundance_type 'raw' so Module 01 converts MaxQuant's 0 to NaN.
      - normalization 'none' when using LFQ intensity, which MaxLFQ already
        normalized; 'median' otherwise.
      - databases and enrichment switched off: the sample is a yeast lysate
        with human spike-ins, so disease/drug annotation and GO enrichment
        carry no meaning here and would only burn API calls.
    '''
    import math
    expected_log2fc = round(math.log2(int(high) / int(low)), 4)

    return {
        'project': {
            'name': run_id,
            'organism': organism,
            'notes': (
                f'Ramus 2016 UPS1/yeast benchmark (PXD001819, MaxQuant output '
                f'from PXD022169). Spike {high} amol vs {low} amol. '
                f'Expected UPS1 log2 fold change: {expected_log2fc}. '
                f'Yeast background expected invariant (log2 FC 0).'
            ),
        },
        'input': {
            'abundance_matrix': abundance_relpath,
            'metadata': f'{run_id}_metadata.csv',
            'format': 'csv',
            'protein_id_column': 'protein_id',
            'abundance_type': 'raw',
            'abundance_prefix': 'abundance_',
            'peptide_count_prefix': 'peptide_count_',
        },
        'design': {
            'group_column': 'spike_level',
            'covariates': [],
            'batch_column': None,
            'contrasts': [f'{GROUP_HIGH}_vs_{GROUP_LOW}'],
        },
        'qc': {
            'min_samples_per_group': 2,
            'min_detections_per_group': 2,
            'min_detections_present_group': None,
        },
        'databases': {
            'enabled': [],
            'query_scope': 'all',
            'cache_dir': './prosift_cache/databases',
            'cache_days': 30,
            'force_requery': False,
        },
        'normalization': {
            'method': 'none' if use_lfq else 'median',
        },
        'imputation': {
            'mode': 'mixed',
            'mnar_method': 'minprob',
            'mar_method': 'knn',
            'single_method': 'minprob',
            'minprob_quantile': 0.01,
            'minprob_scale': 0.3,
            'knn_k': 10,
            'left_censored_downshift': 1.8,
            'left_censored_width': 0.3,
            'random_seed': 42,
        },
        'differential_abundance': {
            'method': 'deqms',
            'significance': {
                'fdr_threshold': 0.05,
                'fc_threshold': 1.0,
            },
        },
        'enrichment': {
            'run_ora': False,
            'run_gsea': False,
            'gene_set_libraries': [],
            'background': 'detected',
            'gsea_ranking': 't_statistic',
            'min_gene_set_size': 15,
            'max_gene_set_size': 500,
            'fdr_threshold': 0.05,
            'plot_top_n': 20,
            'plot_top_gsea_traces': 10,
        },
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    contrasts = resolve_contrasts(args.contrast)
    intensity_prefix = 'LFQ intensity ' if args.lfq else 'Intensity '

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'ProSIFT input preparation -- Ramus UPS1 benchmark')
    print(f'  Abundance metric: {intensity_prefix.strip()}')

    # --- 1. Read and filter ---
    df = load_protein_groups(args.protein_groups)

    # --- 2. Sample IDs ---
    sample_map = build_sample_map(df, intensity_prefix)

    # --- 3. Master abundance matrix ---
    abundance = build_abundance(df, sample_map, intensity_prefix)
    abundance_path = outdir / 'ramus_abundance.csv'
    abundance.to_csv(abundance_path, index=False)
    print(f'  Wrote {abundance_path}')

    # --- 4. Ground truth ---
    truth = build_ground_truth(abundance)
    truth_path = outdir / 'ramus_ground_truth.csv'
    truth.to_csv(truth_path, index=False)
    print(f'  Wrote {truth_path}')

    # --- 5. Per-contrast metadata and params ---
    sample_ids = sorted(sample_map.values())
    samplesheet_rows = []

    for high, low in contrasts:
        run_id = f'ramus_{high}_vs_{low}'
        run_dir = outdir / run_id
        run_dir.mkdir(exist_ok=True)

        meta = build_metadata(sample_ids, high, low)
        meta_path = run_dir / f'{run_id}_metadata.csv'
        meta.to_csv(meta_path, index=False)

        params = build_params(
            run_id, high, low, args.organism,
            abundance_relpath='../ramus_abundance.csv',
            use_lfq=args.lfq,
        )
        params_path = run_dir / f'{run_id}_params.yml'
        with open(params_path, 'w', encoding='utf-8') as handle:
            handle.write(f'# ProSIFT params -- {run_id}\n')
            handle.write(f'# Generated by prepare_ramus_benchmark.py on '
                         f'{date.today().isoformat()}\n\n')
            yaml.safe_dump(params, handle, sort_keys=False, default_flow_style=False)

        n_high = int((meta['spike_level'] == GROUP_HIGH).sum())
        n_low = int((meta['spike_level'] == GROUP_LOW).sum())
        print(f'  {run_id}: {n_high} high + {n_low} low samples')

        samplesheet_rows.append({
            'run_id': run_id,
            'abundance': str(abundance_path.resolve()),
            'metadata': str(meta_path.resolve()),
            'params': str(params_path.resolve()),
        })

    # --- 6. Samplesheet ---
    samplesheet_path = outdir / 'samplesheet_ramus.csv'
    pd.DataFrame(samplesheet_rows).to_csv(samplesheet_path, index=False)
    print(f'  Wrote {samplesheet_path} ({len(samplesheet_rows)} runs)')
    print('Done.')


if __name__ == '__main__':
    main()
