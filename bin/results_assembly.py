#!/usr/bin/env python3
"""
Title:         results_assembly.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-07-10
Last Modified: 2026-09-29
Purpose:       Module 07 Results Assembly. The convergence point of the pipeline.
               Reads the Parquet/CSV outputs of the analytical spine (Modules
               01-05) and the database query layer (Module 06), assembles them
               into a single SQLite database (13 tables), and writes CSV exports
               plus a human-readable assembly summary. This is a pure ETL step:
               no statistics, no API calls, only type coercion and reshaping.
Inputs:
  --mapping-table         {run_id}.id_mapping.parquet             (Module 01 UNIPROT_MAPPING)
  --detection-table       {run_id}.detection_filter_table.csv     (Module 01 FILTER_PROTEINS)
  --sample-flags          {run_id}.sample_flags.parquet           (Module 02 PRENORM_QC)
  --imputed-matrix        {run_id}.imputed_matrix.parquet         (Module 03 IMPUTE)
  --imputation-mask       {run_id}.imputation_mask.parquet        (Module 03 IMPUTE)
  --diff-abundance        {run_id}.diff_abundance_results.parquet (Module 04)
  --enrichment-results    {run_id}.enrichment_results.parquet     (Module 05)
  --protein-term-mapping  {run_id}.protein_term_mapping.parquet   (Module 05)
  --uniprot               {run_id}.uniprot_annotations.parquet    (Module 06 QUERY_UNIPROT)
  --pubmed                {run_id}.pubmed_cooccurrence.parquet    (Module 06 QUERY_PUBMED)
  --disgenet              {run_id}.disgenet_associations.parquet  (Module 06 QUERY_DISGENET)
  --dgidb                 {run_id}.dgidb_interactions.parquet     (Module 06 QUERY_DGIDB)
  --ctd                   {run_id}.ctd_interactions.parquet       (Module 06 QUERY_CTD)
  --params                {run_id}_params.yml                     (pipeline configuration)
Outputs (all under --outdir):
  prosift_results.db                        SQLite database, 13 tables
  {run_id}.diff_abundance_results.csv       Full differential abundance results
  {run_id}.enrichment_results.csv           Full enrichment results
  {run_id}.significant_proteins.csv         Proteins significant in >= 1 contrast
  {run_id}.assembly_summary.txt             Row counts, FK checks, param snapshot
Usage:
  results_assembly.py \
      --mapping-table        CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \
      --detection-table      CTXcyto_WT_vs_CTXcyto_KO.detection_filter_table.csv \
      --sample-flags         CTXcyto_WT_vs_CTXcyto_KO.sample_flags.parquet \
      --imputed-matrix       CTXcyto_WT_vs_CTXcyto_KO.imputed_matrix.parquet \
      --imputation-mask      CTXcyto_WT_vs_CTXcyto_KO.imputation_mask.parquet \
      --diff-abundance       CTXcyto_WT_vs_CTXcyto_KO.diff_abundance_results.parquet \
      --enrichment-results   CTXcyto_WT_vs_CTXcyto_KO.enrichment_results.parquet \
      --protein-term-mapping CTXcyto_WT_vs_CTXcyto_KO.protein_term_mapping.parquet \
      --uniprot              CTXcyto_WT_vs_CTXcyto_KO.uniprot_annotations.parquet \
      --pubmed               CTXcyto_WT_vs_CTXcyto_KO.pubmed_cooccurrence.parquet \
      --disgenet             CTXcyto_WT_vs_CTXcyto_KO.disgenet_associations.parquet \
      --dgidb                CTXcyto_WT_vs_CTXcyto_KO.dgidb_interactions.parquet \
      --ctd                  CTXcyto_WT_vs_CTXcyto_KO.ctd_interactions.parquet \
      --params               CTXcyto_WT_vs_CTXcyto_KO_params.yml \
      --run-id               CTXcyto_WT_vs_CTXcyto_KO \
      --outdir               .
"""

import argparse
import datetime
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('results_assembly')


# ============================================================
# CONSTANTS
# ============================================================

# Prefix on the per-sample abundance columns of the imputed matrix. The bare
# sample_id is recovered by stripping this prefix.
ABUNDANCE_PREFIX = 'abundance_'

# Child tables carrying a protein_id foreign key. Checked for referential
# integrity against the proteins table during assembly (Section 4, acceptance
# criterion 3). Reported, not enforced: SQLite snapshot DB, single writer.
FK_CHILD_TABLES = [
    'differential_abundance',
    'sample_abundances',
    'protein_term_mapping',
    'uniprot_annotations',
    'pubmed_cooccurrence',
    'disease_associations',
    'drug_interactions',
    'chemical_interactions',
]


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    p = argparse.ArgumentParser(
        description='Module 07: assemble pipeline outputs into a SQLite database.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Analytical spine (Modules 01-05) ---
    p.add_argument('--mapping-table', required=True,
                   help='Module 01 id_mapping.parquet')
    p.add_argument('--detection-table', required=True,
                   help='Module 01 detection_filter_table.csv')
    p.add_argument('--sample-flags', required=True,
                   help='Module 02 sample_flags.parquet')
    p.add_argument('--imputed-matrix', required=True,
                   help='Module 03 imputed_matrix.parquet')
    p.add_argument('--imputation-mask', required=True,
                   help='Module 03 imputation_mask.parquet')
    p.add_argument('--diff-abundance', required=True,
                   help='Module 04 diff_abundance_results.parquet')
    p.add_argument('--enrichment-results', required=True,
                   help='Module 05 enrichment_results.parquet')
    p.add_argument('--protein-term-mapping', required=True,
                   help='Module 05 protein_term_mapping.parquet')
    # --- Database query layer (Module 06). Always present; empty if disabled. ---
    p.add_argument('--uniprot', required=True,
                   help='Module 06 uniprot_annotations.parquet')
    p.add_argument('--pubmed', required=True,
                   help='Module 06 pubmed_cooccurrence.parquet')
    p.add_argument('--disgenet', required=True,
                   help='Module 06 disgenet_associations.parquet')
    p.add_argument('--dgidb', required=True,
                   help='Module 06 dgidb_interactions.parquet')
    p.add_argument('--ctd', required=True,
                   help='Module 06 ctd_interactions.parquet')
    # --- Provenance and output ---
    p.add_argument('--params', required=True,
                   help='Pipeline params.yml (provenance capture)')
    p.add_argument('--run-id', required=True,
                   help='Run identifier (contrast label), e.g. CTXcyto_WT_vs_CTXcyto_KO')
    p.add_argument('--outdir', default='.',
                   help='Output directory for the database and exports')
    p.add_argument('--db-name', default='prosift_results.db',
                   help='Filename for the SQLite database')
    p.add_argument('--pipeline-version', default='unknown',
                   help='Pipeline version string for run_metadata provenance')
    p.add_argument('--run-date', default=None,
                   help='ISO 8601 run date for run_metadata (default: now). '
                        'Set explicitly for deterministic testing.')
    return p.parse_args()


# ============================================================
# SQLITE SCHEMA HELPERS
# ============================================================

def _sqlite_type(dtype) -> str:
    '''
    Map a pandas dtype to a SQLite column type via NumPy kind codes.

    'b' (bool) and integer kinds -> INTEGER, float kinds -> REAL, everything
    else (object strings, datetimes) -> TEXT. Pandas nullable extension dtypes
    (Int64, Float64, boolean) report the same kind codes as their NumPy
    counterparts, so this handles them without special-casing.
    '''
    kind = dtype.kind
    if kind in ('b', 'i', 'u'):
        return 'INTEGER'
    if kind == 'f':
        return 'REAL'
    return 'TEXT'


def _prep_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalize a DataFrame for SQLite insertion.

    Converts boolean columns (both NumPy bool and pandas nullable boolean) to
    nullable Int64 so they land as 0/1/NULL integers with a deterministic
    INTEGER column type, rather than relying on the sqlite3 bool adapter.
    '''
    out = df.copy()
    for col in out.columns:
        dtype_str = str(out[col].dtype)
        if out[col].dtype == bool or dtype_str == 'boolean':
            out[col] = out[col].astype('Int64')
    return out


def create_and_load(conn: sqlite3.Connection, name: str, df: pd.DataFrame,
                    pk: list | None = None) -> int:
    '''
    Create a table from a DataFrame's inferred schema and load its rows.

    Column types come from the DataFrame dtypes (see _sqlite_type), so
    pass-through tables automatically absorb upstream schema evolution (for
    example the Module 05 rrvgo cluster columns, or Module 06 column drift)
    without editing this script. An optional composite PRIMARY KEY is declared
    for the structured tables. Identifiers are double-quoted so reserved words
    (for example the "group" column of sample_abundances) are safe.

    Returns the number of rows inserted.
    '''
    df = _prep_for_sqlite(df)

    # Step 1: build the column definitions from dtypes.
    col_defs = [f'"{col}" {_sqlite_type(df[col].dtype)}' for col in df.columns]

    # Step 2: optionally append a composite primary key clause.
    pk_clause = ''
    if pk:
        quoted = ', '.join(f'"{c}"' for c in pk)
        pk_clause = f', PRIMARY KEY ({quoted})'

    # Step 3: (re)create the table, then append the rows.
    conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    conn.execute(f'CREATE TABLE "{name}" ({", ".join(col_defs)}{pk_clause})')
    df.to_sql(name, conn, if_exists='append', index=False)
    log.info('  table %-24s %7d rows, %2d cols', name, len(df), df.shape[1])
    return len(df)


# ============================================================
# INPUT LOADING
# ============================================================

def load_inputs(args: argparse.Namespace) -> dict:
    '''Read every input file. Returns a dict of DataFrames plus the params dict.'''
    log.info('Reading inputs')
    data = {
        'mapping': pd.read_parquet(args.mapping_table),
        'detection': pd.read_csv(args.detection_table),
        'flags': pd.read_parquet(args.sample_flags),
        'imputed': pd.read_parquet(args.imputed_matrix),
        'mask': pd.read_parquet(args.imputation_mask),
        'diff_abundance': pd.read_parquet(args.diff_abundance),
        'enrichment': pd.read_parquet(args.enrichment_results),
        'protein_term': pd.read_parquet(args.protein_term_mapping),
        'uniprot': pd.read_parquet(args.uniprot),
        'pubmed': pd.read_parquet(args.pubmed),
        'disgenet': pd.read_parquet(args.disgenet),
        'dgidb': pd.read_parquet(args.dgidb),
        'ctd': pd.read_parquet(args.ctd),
    }
    with open(args.params) as fh:
        data['params'] = yaml.safe_load(fh)
    return data


# ============================================================
# DERIVED-COLUMN LOGIC (proteins table)
# ============================================================

def derive_imputation_fraction(mask: pd.DataFrame) -> pd.Series:
    '''
    Per-protein fraction of abundance values that were imputed.

    For each protein: count of cells whose imputation status is not 'observed'
    (that is, 'mnar' or 'mar') divided by the number of samples. Indexed by
    protein_id.
    '''
    sample_cols = [c for c in mask.columns if c != 'protein_id']
    n_samples = len(sample_cols)
    imputed_counts = (mask[sample_cols] != 'observed').sum(axis=1)
    frac = imputed_counts / n_samples
    return pd.Series(frac.values, index=mask['protein_id'].values, name='imputation_fraction')


def build_proteins(data: dict) -> pd.DataFrame:
    '''
    Build the contrast-independent proteins table.

    Identity and ortholog columns come from the Module 01 mapping table;
    detection_category is left-joined from the Module 01 detection filter table;
    imputation_fraction is derived (Section 4.3). Sample-level QC flags are NOT
    projected onto proteins here -- they are loaded verbatim as the
    sample_qc_flags table (Section 4.3) and surfaced per-protein by the Module 08
    profile card, which joins a protein's observed samples against that table.
    '''
    mapping = data['mapping']
    detection = data['detection']

    # Step 1: identity + ortholog columns. Module 01 emits gene_symbol_mouse;
    # the schema names it gene_symbol.
    proteins = mapping[[
        'protein_id', 'gene_symbol_mouse', 'entrez_id_mouse', 'ensembl_gene_mouse',
        'human_ortholog_symbol', 'human_ortholog_entrez', 'ortholog_mapping_status',
    ]].rename(columns={'gene_symbol_mouse': 'gene_symbol'}).copy()

    # Step 2: detection_category via left join (Module 01 names it filter_status).
    det = detection[['protein_id', 'filter_status']].rename(
        columns={'filter_status': 'detection_category'})
    proteins = proteins.merge(det, on='protein_id', how='left')

    # Step 3: derived per-protein column.
    frac = derive_imputation_fraction(data['mask'])
    proteins['imputation_fraction'] = proteins['protein_id'].map(frac).astype('float64')

    # Step 4: fixed column order (matches spec Section 4.2).
    return cast(pd.DataFrame, proteins[[
        'protein_id', 'gene_symbol', 'entrez_id_mouse', 'ensembl_gene_mouse',
        'human_ortholog_symbol', 'human_ortholog_entrez', 'ortholog_mapping_status',
        'detection_category', 'imputation_fraction',
    ]])


# ============================================================
# RESHAPE LOGIC (sample_abundances table)
# ============================================================

def build_sample_abundances(data: dict) -> pd.DataFrame:
    '''
    Pivot the wide imputed matrix into a long protein-by-sample table.

    Joins the per-cell imputation status (from the mask) and the group label
    (from the sample flags table, which carries a resolved 'group' column).
    '''
    imputed = data['imputed']
    mask = data['mask']
    flags = data['flags']

    # Step 1: melt abundance columns to long form and recover the bare sample_id.
    # Guard against a silent empty table if the Module 03 abundance-column prefix
    # ever changes (the pipeline standardises on 'abundance_').
    abund_cols = [c for c in imputed.columns if c.startswith(ABUNDANCE_PREFIX)]
    if not abund_cols:
        raise ValueError(
            f"No '{ABUNDANCE_PREFIX}' columns in the imputed matrix "
            f'(found {list(imputed.columns)}); cannot build sample_abundances.')
    long = imputed.melt(id_vars='protein_id', value_vars=abund_cols,
                        var_name='sample_col', value_name='abundance')
    long['sample_id'] = long['sample_col'].str.slice(len(ABUNDANCE_PREFIX))
    long = long.drop(columns='sample_col')

    # Step 2: attach the group label from the sample flags table.
    sample_group = dict(zip(flags['sample_id'], flags['group'], strict=False))
    long['group'] = long['sample_id'].map(sample_group)

    # Step 3: attach the per-cell imputation status from the mask (long form).
    mask_long = mask.melt(id_vars='protein_id', var_name='sample_id',
                          value_name='imputation_status')
    long = long.merge(mask_long, on=['protein_id', 'sample_id'], how='left')

    return cast(pd.DataFrame,
                long[['protein_id', 'sample_id', 'abundance', 'group', 'imputation_status']])


# ============================================================
# METADATA TABLES
# ============================================================

def build_run_metadata(data: dict, args: argparse.Namespace,
                       n_proteins: int, n_samples: int) -> pd.DataFrame:
    '''Build the single-row run_metadata table.'''
    params = data['params']
    run_date = args.run_date or datetime.datetime.now().isoformat(timespec='seconds')
    groups = sorted(data['flags']['group'].dropna().unique().tolist())
    contrasts = params.get('design', {}).get('contrasts', [])
    organism = params.get('project', {}).get('organism', None)

    return pd.DataFrame([{
        'run_id': args.run_id,
        'pipeline_version': args.pipeline_version,
        'run_date': run_date,
        'organism': organism,
        'contrasts': json.dumps(contrasts),
        'groups': json.dumps(groups),
        'n_proteins': n_proteins,
        'n_samples': n_samples,
    }])


def _value_type(value) -> str:
    '''Type hint string for a params.yml value (bool checked before int).'''
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'float'
    if isinstance(value, list):
        return 'list'
    if isinstance(value, dict):
        return 'dict'
    if value is None:
        return 'null'
    return 'string'


def _flatten_params(node, prefix: str = '') -> list:
    '''
    Flatten a nested params dict into (dot_key, value) pairs.

    Dicts recurse; lists are stored whole (not indexed). Scalars terminate.
    '''
    items = []
    for key, value in node.items():
        dot_key = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_params(value, dot_key))
        else:
            items.append((dot_key, value))
    return items


def build_run_parameters(data: dict) -> pd.DataFrame:
    '''Build the key-value run_parameters table from the full params.yml.'''
    rows = []
    for key, value in _flatten_params(data['params']):
        vtype = _value_type(value)
        vstr = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        rows.append({'key': key, 'value': vstr, 'value_type': vtype})
    return pd.DataFrame(rows, columns=['key', 'value', 'value_type'])


# ============================================================
# CSV EXPORTS
# ============================================================

def write_csv_exports(data: dict, proteins: pd.DataFrame,
                      outdir: Path, run_id: str) -> None:
    '''Write the three CSV exports (diff abundance, enrichment, significant proteins).'''
    da = data['diff_abundance']
    enr = data['enrichment']

    # Full differential abundance and enrichment results.
    da.to_csv(outdir / f'{run_id}.diff_abundance_results.csv', index=False)
    enr.to_csv(outdir / f'{run_id}.enrichment_results.csv', index=False)

    # Significant proteins: DA rows where significant is truthy, joined with
    # protein identity columns.
    sig = da[da['significant'].astype('boolean').fillna(False)].copy()
    identity = proteins[[
        'protein_id', 'gene_symbol', 'human_ortholog_symbol', 'detection_category',
    ]]
    # Drop identity columns already present in the DA table (for example
    # gene_symbol) so the join does not create _x/_y duplicates.
    overlap = [c for c in identity.columns if c != 'protein_id' and c in sig.columns]
    identity = identity.drop(columns=overlap)
    sig_out = sig.merge(identity, on='protein_id', how='left')
    keep = [
        'protein_id', 'gene_symbol', 'human_ortholog_symbol', 'detection_category',
        'contrast', 'numerator', 'denominator', 'log2_fc', 'deqms_pvalue',
        'deqms_adj_pvalue', 'direction', 'significant',
    ]
    keep = [c for c in keep if c in sig_out.columns]
    sig_out[keep].to_csv(outdir / f'{run_id}.significant_proteins.csv', index=False)
    log.info('  CSV exports: diff_abundance (%d), enrichment (%d), significant (%d)',
             len(da), len(enr), len(sig_out))


# ============================================================
# FK INTEGRITY CHECK
# ============================================================

def check_fk_integrity(conn: sqlite3.Connection) -> list:
    '''
    Report referential integrity of each protein_id child table against proteins.

    Returns a list of (table, n_orphans) for tables that have orphan protein_ids
    (child rows whose protein_id is absent from proteins). Empty list means all
    tables are clean.
    '''
    valid = {r[0] for r in conn.execute('SELECT protein_id FROM proteins')}
    orphans = []
    for table in FK_CHILD_TABLES:
        child_ids = {r[0] for r in conn.execute(
            f'SELECT DISTINCT protein_id FROM "{table}"') if r[0] is not None}
        missing = child_ids - valid
        if missing:
            orphans.append((table, len(missing)))
    return orphans


# ============================================================
# ASSEMBLY SUMMARY
# ============================================================

def write_assembly_summary(outdir: Path, run_id: str, row_counts: dict,
                           orphans: list, disabled_dbs: list,
                           data: dict, args: argparse.Namespace) -> None:
    '''Write the human-readable assembly summary text file.'''
    run_date = args.run_date or datetime.datetime.now().isoformat(timespec='seconds')
    lines = []
    lines.append('=' * 60)
    lines.append('ProSIFT Module 07 -- Results Assembly Summary')
    lines.append('=' * 60)
    lines.append(f'Run ID:           {run_id}')
    lines.append(f'Assembly date:    {run_date}')
    lines.append(f'Pipeline version: {args.pipeline_version}')
    lines.append(f'Database:         {args.db_name}')
    lines.append('')
    lines.append('Table row counts')
    lines.append('-' * 60)
    for table, n in row_counts.items():
        lines.append(f'  {table:24s} {n:>10d}')
    lines.append('')
    lines.append('Foreign key integrity (protein_id child tables)')
    lines.append('-' * 60)
    if not orphans:
        lines.append('  PASS -- all child protein_id values resolve to proteins.')
    else:
        for table, n in orphans:
            lines.append(f'  WARNING -- {table}: {n} orphan protein_id value(s)')
    lines.append('')
    lines.append('Database query layer')
    lines.append('-' * 60)
    enabled = data['params'].get('databases', {}).get('enabled', [])
    lines.append(f'  Enabled in params:  {", ".join(enabled) if enabled else "(none)"}')
    if disabled_dbs:
        lines.append(f'  Empty tables:       {", ".join(disabled_dbs)}')
    else:
        lines.append('  Empty tables:       (none)')
    lines.append('')
    lines.append('=' * 60)
    (outdir / f'{run_id}.assembly_summary.txt').write_text('\n'.join(lines) + '\n')


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    '''Assemble the SQLite database and CSV exports from upstream outputs.'''
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    db_path = outdir / args.db_name

    # --- 1. Read all inputs ---
    data = load_inputs(args)

    # --- 2. Build the derived / reshaped tables ---
    proteins = build_proteins(data)
    sample_abundances = build_sample_abundances(data)
    n_proteins = len(proteins)
    n_samples = data['flags']['sample_id'].nunique()
    run_metadata = build_run_metadata(data, args, n_proteins, n_samples)
    run_parameters = build_run_parameters(data)

    # --- 3. Create the database and load all 13 tables ---
    log.info('Writing SQLite database: %s', db_path)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    row_counts = {}
    try:
        # Core tables
        row_counts['proteins'] = create_and_load(conn, 'proteins', proteins, pk=['protein_id'])
        row_counts['differential_abundance'] = create_and_load(
            conn, 'differential_abundance', data['diff_abundance'],
            pk=['protein_id', 'contrast'])
        row_counts['sample_abundances'] = create_and_load(
            conn, 'sample_abundances', sample_abundances,
            pk=['protein_id', 'sample_id'])
        # Sample QC dimension: the Module 02 sample_flags table verbatim
        # (one row per sample: group + the four boolean flags + n_flags). The
        # Module 08 profile card joins a protein's observed samples against this
        # to list the specific flags fired, and why, rather than a per-protein
        # count (Section 4.3; supersedes the 2026-07-10 qc_flag_count decision).
        row_counts['sample_qc_flags'] = create_and_load(
            conn, 'sample_qc_flags', data['flags'], pk=['sample_id'])
        # Enrichment tables (pass-through)
        row_counts['enrichment_results'] = create_and_load(
            conn, 'enrichment_results', data['enrichment'])
        row_counts['protein_term_mapping'] = create_and_load(
            conn, 'protein_term_mapping', data['protein_term'])
        # Annotation tables (pass-through; empty if the database was disabled)
        row_counts['uniprot_annotations'] = create_and_load(
            conn, 'uniprot_annotations', data['uniprot'], pk=['protein_id'])
        row_counts['pubmed_cooccurrence'] = create_and_load(
            conn, 'pubmed_cooccurrence', data['pubmed'])
        row_counts['disease_associations'] = create_and_load(
            conn, 'disease_associations', data['disgenet'])
        row_counts['drug_interactions'] = create_and_load(
            conn, 'drug_interactions', data['dgidb'])
        row_counts['chemical_interactions'] = create_and_load(
            conn, 'chemical_interactions', data['ctd'])
        # Metadata tables
        row_counts['run_metadata'] = create_and_load(
            conn, 'run_metadata', run_metadata, pk=['run_id'])
        row_counts['run_parameters'] = create_and_load(
            conn, 'run_parameters', run_parameters, pk=['key'])

        conn.commit()

        # --- 4. FK integrity check ---
        orphans = check_fk_integrity(conn)
        if orphans:
            for table, n in orphans:
                log.warning('FK: %s has %d orphan protein_id value(s)', table, n)
        else:
            log.info('FK integrity: all child protein_id values resolve to proteins')
    finally:
        conn.close()

    # --- 5. CSV exports ---
    write_csv_exports(data, proteins, outdir, args.run_id)

    # --- 6. Assembly summary ---
    # A database is reported as an empty table when its annotation table has no
    # rows (Module 06 emits a schema-correct empty Parquet when disabled).
    db_table_map = {
        'uniprot': 'uniprot_annotations',
        'pubmed': 'pubmed_cooccurrence',
        'disgenet': 'disease_associations',
        'dgidb': 'drug_interactions',
        'ctd': 'chemical_interactions',
    }
    disabled_dbs = [name for name, table in db_table_map.items()
                    if row_counts.get(table, 0) == 0]
    write_assembly_summary(outdir, args.run_id, row_counts, orphans,
                           disabled_dbs, data, args)

    log.info('Assembly complete: %d tables, %d proteins, %d samples',
             len(row_counts), n_proteins, n_samples)
    return 0


if __name__ == '__main__':
    sys.exit(main())
