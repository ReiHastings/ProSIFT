# title: prepare_prosift_input.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-03-23
# last modified: 2026-07-13  (emit complete params.yml: databases, normalization,
#                imputation, differential_abundance, enrichment; derive contrasts;
#                --gmt and --organism args; PubMed off by default)
#
# purpose:
#   Reads a run_config.yml (produced by generate_run_config.py) and the
#   master abundance file it references. For each run, writes two files into
#   a per-run subdirectory:
#     - {run}_metadata.csv    (sample_id + group column for this run's samples)
#     - {run}_params.yml      (draft ProSIFT parameters pointing to master abundance)
#
#   All runs share the master abundance file. Module 01's cross-validation step
#   subsets the abundance matrix to the samples listed in the metadata file, so
#   per-run abundance copies are unnecessary.
#
# inputs:
#   - run_config.yml (references master abundance CSV)
#
# outputs:
#   - Per-run subdirectories with {run}_metadata.csv and {run}_params.yml
#   - run_config.yml copied to outdir for provenance
#
# usage example:
#   python scripts/prepare_prosift_input.py \
#     --config prosift_inputs/run_config.yml \
#     --gmt databases/m5.go.bp.v2026.1.Mm.symbols.gmt \
#           databases/m2.cp.reactome.v2026.1.Mm.symbols.gmt \
#     --organism mouse --outdir prosift_inputs
#
#   Notes:
#   - --gmt is REQUIRED (one or more GMT paths). enrichment.py reads these paths
#     directly from params.yml at runtime (Nextflow does not stage them), so pass
#     ABSOLUTE paths -- or relative paths that will be made absolute here -- that
#     resolve on the machine that runs the pipeline. Like samplesheet.csv, the
#     GMT paths are machine-specific; regenerate (or pass cluster paths) when the
#     target machine changes.
#   - PubMed is OFF by default: it needs study-specific search_terms the generator
#     cannot infer. Add terms under databases.pubmed and re-add 'pubmed' to
#     databases.enabled to activate it.

import argparse
import csv
import os
import shutil
import sys
from collections import OrderedDict
from datetime import date

try:
    import yaml
except ImportError:
    print('Error: PyYAML is required. Install with: pip install pyyaml')
    sys.exit(1)


# ============================================================
# Loading and validation
# ============================================================

def load_config(config_path):
    '''Load and return the run config YAML.'''
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(base_dir, filename):
    '''Resolve a filename relative to a base directory.'''
    return os.path.join(base_dir, filename)


def load_csv(filepath):
    '''Load a CSV file and return (headers, rows) where rows is a list of dicts.'''
    with open(filepath, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows


def validate_config(config, headers, config_dir):
    '''Validate that the config references valid columns and files.'''
    source = config['source']
    runs = config['runs']

    # check source files exist
    abund_path = resolve_path(config_dir, source['abundance_file'])
    if not os.path.isfile(abund_path):
        print(f'Error: abundance file not found: {abund_path}')
        sys.exit(1)

    # check protein_id column exists
    protein_id_col = source['protein_id_column']
    if protein_id_col not in headers:
        print(f'Error: protein ID column "{protein_id_col}" not found in abundance file.')
        sys.exit(1)

    # check all sample columns exist
    abund_prefix = source.get('abundance_prefix', '')
    pep_prefix = source.get('peptide_count_prefix', None)

    for run_name, run_def in runs.items():
        for sample_id in run_def['samples']:
            abund_col = abund_prefix + sample_id
            if abund_col not in headers:
                print(f'Error: run "{run_name}": abundance column "{abund_col}" not found.')
                sys.exit(1)
            if pep_prefix:
                pep_col = pep_prefix + sample_id
                if pep_col not in headers:
                    print(f'Error: run "{run_name}": peptide count column "{pep_col}" not found.')
                    sys.exit(1)


# ============================================================
# Per-run output writers
# ============================================================

def write_samplesheet(filepath, samplesheet_rows):
    '''
    Write samplesheet.csv for Nextflow input.

    Each row has: run_id, abundance, metadata, params -- all absolute paths.
    Absolute paths are used so Nextflow can be run from any working directory.
    Regenerate this file after rsyncing to a new machine (absolute paths
    will differ on the cluster).
    '''
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'abundance', 'metadata', 'params'])
        writer.writeheader()
        writer.writerows(samplesheet_rows)


def write_run_metadata(filepath, sample_group_map, group_column):
    '''Write the per-run metadata CSV.'''
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', group_column])
        for sample_id, group in sample_group_map.items():
            writer.writerow([sample_id, group])


def derive_contrasts(groups):
    '''Derive DA contrasts from the ordered distinct group values of a run.

    For a two-group run the reference is the first-appearing group, so the
    contrast is "<second>_vs_<first>" (e.g. groups [WT, KO] -> "KO_vs_WT",
    matching the canonical genotype convention). For a non-binary run the
    direction is ambiguous, so return no contrasts and let the user fill them in.
    '''
    if len(groups) == 2:
        return [f'{groups[1]}_vs_{groups[0]}']
    return []


def write_run_params(filepath, run_name, group_column, groups, abund_prefix,
                     pep_prefix, master_abund_relpath, organism, gmt_paths):
    '''Write a complete params.yml for one run.

    Emits every block the pipeline consumes (Modules 01-06) with canonical
    defaults. Study-specific fields are handled explicitly:
      - design.contrasts are derived from the group order (see derive_contrasts)
      - enrichment.gene_set_libraries come from --gmt (absolute paths)
      - PubMed is left out of databases.enabled by default because it needs
        study-specific search terms the generator cannot infer
    Module 07-08 parameters are still TBD.
    '''
    today = date.today().isoformat()
    pep_line = (f'  peptide_count_prefix: "{pep_prefix}"'
                if pep_prefix else '  peptide_count_prefix: null')

    # --- design.contrasts: derived from group order (first group = reference) ---
    contrasts = derive_contrasts(groups)
    if contrasts:
        contrasts_block = ('  contrasts:\n'
                           + '\n'.join(f'    - "{c}"' for c in contrasts)
                           + '\n    # auto-derived as <second group>_vs_<first group>'
                           ' (first = reference); verify direction')
    else:
        contrasts_block = ('  contrasts: []\n'
                           f'  # REQUIRED: {len(groups)} groups found -- add contrasts'
                           ' as "<test>_vs_<reference>"')

    # --- enrichment.gene_set_libraries: absolute GMT paths from --gmt ---
    gmt_block = '\n'.join(f'    - "{os.path.abspath(p)}"' for p in gmt_paths)

    content = f'''# ProSIFT params -- {run_name}
# Generated by prepare_prosift_input.py on {today}

project:
  name: "{run_name}"
  organism: "{organism}"

input:
  abundance_matrix: "{master_abund_relpath}"  # shared master file
  metadata: "{run_name}_metadata.csv"
  format: "csv"
  protein_id_column: "protein_id"
  abundance_type: "raw"          # raw | log2 (raw values <=0 are set to NaN at validation)
  abundance_prefix: "{abund_prefix}"
{pep_line}

design:
  group_column: "{group_column}"
  covariates: []
  batch_column: null
{contrasts_block}

qc:
  min_samples_per_group: 2
  min_detections_per_group: 2
  # Anchor for presence/absence (SINGLE-GROUP) and PARTIAL proteins: at least
  # one group must reach this many detections or the protein is removed
  # (WEAK-ANCHOR), preventing MNAR-imputation-driven results from resting on
  # too few real measurements. null = require the anchor group to be FULLY
  # detected (recommended); an integer sets a fixed minimum instead.
  min_detections_present_group: null

databases:
  enabled:                              # which databases to query (remove to skip)
    - uniprot
    - disgenet
    - dgidb
    - ctd
    # - pubmed                          # OFF by default: set databases.pubmed.search_terms, then re-add here
  query_scope: 'all'                    # 'all' proteins (only option currently)
  cache_dir: './prosift_cache/databases' # application-level cache (outside work dir)
  cache_days: 30                        # re-query entries older than this
  force_requery: false                  # true = ignore cache, re-query everything
  pubmed:
    search_terms: []                    # REQUIRED before enabling pubmed -- study-specific co-occurrence terms
    normalization: 'pmi'                # score normalization (only pmi currently)
    min_pubs_for_score: 5               # min total publications to compute PMI
  disgenet:
    min_score: 0.1                      # min GDA score for output (applied at output time)
  api_keys:
    pubmed: 'NCBI_API_KEY'              # env var name for NCBI API key
    disgenet: 'DISGENET_API_KEY'        # env var name for DisGeNET API key

normalization:
  method: "median"              # options: median, quantile, vsn, none

imputation:
  mode: "mixed"                 # options: mixed (DEP-style MNAR+MAR), single (one method for all)
  mnar_method: "minprob"        # mixed mode only; currently only minprob
  mar_method: "knn"             # mixed mode only; currently only knn
  single_method: "minprob"      # single mode only; options: minprob, knn, left_censored
  minprob_quantile: 0.01        # quantile of sample distribution to center MinProb imputation
  minprob_scale: 0.3            # multiplier on sample SD for MinProb imputation width
  knn_k: 10                     # number of nearest neighbors for KNN imputation
  left_censored_downshift: 1.8  # SDs below mean for left-censored center (single mode only)
  left_censored_width: 0.3      # multiplier on sample SD for left-censored width (single mode only)
  random_seed: 42               # seed for MinProb random draws (ensures deterministic output)

differential_abundance:
  method: "deqms"              # options: deqms, limma
  significance:
    fdr_threshold: 0.05        # BH-adjusted p-value cutoff
    fc_threshold: 1.0          # absolute log2 FC cutoff (0 = disabled)

enrichment:
  run_ora:  true
  run_gsea: true
  gene_set_libraries:          # GMT paths from --gmt (absolute; machine-specific, like samplesheet.csv)
{gmt_block}
  background:           "detected"   # ORA background: all detected proteins in this run
  gsea_ranking:         "t_statistic"
  min_gene_set_size:    15
  max_gene_set_size:    500
  fdr_threshold:        0.05
  plot_top_n:           20
  plot_top_gsea_traces: 10

# --- Modules below not yet implemented ---
# Module 07 (results assembly) and Module 08 (frontend) parameters TBD.
'''
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Split master ProSIFT input files into per-run directories.'
    )
    parser.add_argument(
        '--config', '-c', required=True,
        help='Path to run_config.yml'
    )
    parser.add_argument(
        '--outdir', '-o', default=None,
        help='Output directory (default: same directory as config file)'
    )
    parser.add_argument(
        '--gmt', nargs='+', required=True, metavar='GMT',
        help='One or more GMT gene-set library files, written to '
             'enrichment.gene_set_libraries as absolute paths. REQUIRED: '
             'enrichment.py reads these paths directly at runtime, so they must '
             'resolve on the machine that runs the pipeline (regenerate for the '
             'cluster, like samplesheet.csv).'
    )
    parser.add_argument(
        '--organism', default='mouse',
        help='Organism written to project.organism (default: mouse)'
    )
    args = parser.parse_args()

    # Soft check: warn (do not fail) if a GMT path is absent here -- it may be a
    # path for a different target machine (e.g. cluster paths generated locally).
    for g in args.gmt:
        if not os.path.isfile(g):
            print(f'  Warning: --gmt path not found on this machine (ok if targeting '
                  f'another machine): {g}')

    config_path = args.config
    if not os.path.isfile(config_path):
        print(f'Error: config file not found: {config_path}')
        sys.exit(1)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    outdir = args.outdir if args.outdir else config_dir

    # ----------------------------------------------------------
    # 1. Load config
    # ----------------------------------------------------------
    config = load_config(config_path)
    source = config['source']
    runs = config['runs']

    abund_prefix = source.get('abundance_prefix', '')
    pep_prefix = source.get('peptide_count_prefix', None)

    # ----------------------------------------------------------
    # 2. Validate master abundance file headers
    # ----------------------------------------------------------
    abund_path = resolve_path(config_dir, source['abundance_file'])
    print(f'Loading {os.path.basename(abund_path)}...')
    headers, rows = load_csv(abund_path)
    print(f'  {len(rows)} proteins, {len(headers)} columns')

    validate_config(config, headers, config_dir)
    print('  Validation passed.')

    # ----------------------------------------------------------
    # 3. Process each run
    # ----------------------------------------------------------
    print(f'\nGenerating {len(runs)} runs in {outdir}/\n')

    samplesheet_rows = []

    for run_name, run_def in runs.items():
        sample_group_map = OrderedDict(run_def['samples'])
        group_column = run_def['group_column']
        # distinct group values in order of first appearance (for contrasts)
        groups = list(OrderedDict.fromkeys(sample_group_map.values()))

        # create run directory
        run_dir = os.path.join(outdir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # relative path from run directory to master abundance file
        master_abund_relpath = os.path.relpath(
            os.path.abspath(abund_path), os.path.abspath(run_dir)
        )

        meta_path   = os.path.join(run_dir, f'{run_name}_metadata.csv')
        params_path = os.path.join(run_dir, f'{run_name}_params.yml')

        write_run_metadata(meta_path, sample_group_map, group_column)
        write_run_params(params_path, run_name, group_column, groups, abund_prefix,
                         pep_prefix, master_abund_relpath, args.organism, args.gmt)

        # collect absolute paths for samplesheet
        samplesheet_rows.append({
            'run_id':   run_name,
            'abundance': os.path.abspath(abund_path),
            'metadata':  os.path.abspath(meta_path),
            'params':    os.path.abspath(params_path),
        })

        # summary
        groups = OrderedDict()
        for sid, grp in sample_group_map.items():
            groups.setdefault(grp, []).append(sid)
        group_summary = ', '.join(f'{len(v)} {k}' for k, v in groups.items())
        print(f'  {run_name}: {group_summary} ({group_column})')

    # ----------------------------------------------------------
    # 4. Copy run_config.yml for provenance
    # ----------------------------------------------------------
    config_dest = os.path.join(outdir, 'run_config.yml')
    if os.path.abspath(config_path) != os.path.abspath(config_dest):
        shutil.copy2(config_path, config_dest)
        print(f'\nCopied run_config.yml to {outdir}/')

    # ----------------------------------------------------------
    # 5. Write samplesheet.csv for Nextflow
    # ----------------------------------------------------------
    samplesheet_path = os.path.join(outdir, 'samplesheet.csv')
    write_samplesheet(samplesheet_path, samplesheet_rows)
    print(f'Written samplesheet.csv ({len(samplesheet_rows)} runs) to {outdir}/')
    print('  Note: samplesheet uses absolute paths -- regenerate after rsyncing to a new machine.')

    print(f'\nDone. {len(runs)} runs written to {outdir}/')


if __name__ == '__main__':
    main()
