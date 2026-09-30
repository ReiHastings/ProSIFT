#!/usr/bin/env python3
# title: test_results_assembly.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-10
# last modified: 2026-07-10
#
# purpose:
#   Unit and integration tests for Module 07 (bin/results_assembly.py). Unit
#   tests pin the pure builder/derivation functions (imputation_fraction,
#   params flattening, table reshaping). The integration test
#   runs the full CLI against synthetic inputs and asserts on the assembled
#   SQLite database: 13 tables, row counts, primary keys, foreign-key
#   integrity, derived-column values, and the disabled-database empty-table
#   case.
#
# inputs:
#   None (all inputs are synthesised in-memory / written to tmp_path)
#
# outputs:
#   None (pytest assertions)
#
# usage example:
#   pytest tests/test_results_assembly.py
#   copy/paste: pytest tests/test_results_assembly.py -q

import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Add bin/ to path so we can import results_assembly directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import results_assembly as ra

SCRIPT = Path(__file__).resolve().parent.parent / 'bin' / 'results_assembly.py'

# Synthetic sample layout: 2 WT + 2 KO.
SAMPLES = ['S_WT-1', 'S_WT-2', 'S_KO-1', 'S_KO-2']
GROUPS = ['WT', 'WT', 'KO', 'KO']


# ============================================================
# Synthetic-input builders (mirror real upstream schemas)
# ============================================================

def _mask_frame():
    '''
    Imputation mask (wide). Designed cells per protein:
      P1 observed everywhere            -> imp_frac 0.0
      P2 observed only in unflagged     -> imp_frac 0.5
      P3 observed incl. one flagged     -> imp_frac 0.5
      P4 SINGLE-GROUP (KO only observed)-> imp_frac 0.5
      P5 all imputed                    -> imp_frac 1.0
    Sample flag layout (see _flags_frame): S_WT-1 fires extreme_median,
    S_KO-1 fires pca_outlier. So the two flag types are on different samples.
    '''
    o, m = 'observed', 'mar'
    rows = {
        'protein_id':  ['P1', 'P2', 'P3', 'P4', 'P5'],
        'S_WT-1':      [o,    m,    o,    m,    m],
        'S_WT-2':      [o,    o,    o,    m,    m],
        'S_KO-1':      [o,    m,    o,    o,    m],
        'S_KO-2':      [o,    o,    m,    o,    m],
    }
    return pd.DataFrame(rows)


# The four sample-level QC flag columns produced by Module 02 PRENORM_QC.
_FLAG_COLS = [
    'flag_low_detection', 'flag_extreme_median',
    'flag_pca_outlier', 'flag_low_correlation',
]


def _flags_frame():
    df = pd.DataFrame({
        'sample_id': SAMPLES,
        'group': GROUPS,
        'flag_low_detection':   [False, False, False, False],
        'flag_extreme_median':  [True,  False, False, False],   # S_WT-1
        'flag_pca_outlier':     [False, False, True,  False],   # S_KO-1
        'flag_low_correlation': [False, False, False, False],
    })
    df['n_flags'] = df[_FLAG_COLS].sum(axis=1).astype('int64')
    return df


def _mapping_frame():
    return pd.DataFrame({
        'protein_id': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'uniprot_accession': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'gene_symbol_mouse': ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'],
        'entrez_id_mouse': ['11', '22', '33', '44', '55'],
        'ensembl_gene_mouse': ['ENSM1', 'ENSM2', 'ENSM3', 'ENSM4', 'ENSM5'],
        'human_ortholog_symbol': ['GENE1', 'GENE2', None, 'GENE4', 'GENE5'],
        'human_ortholog_entrez': ['111', '222', None, '444', '555'],
        'ortholog_mapping_status': ['one_to_one', 'one_to_one', 'no_ortholog',
                                    'one_to_one', 'one_to_one'],
        'mapping_status': ['mapped'] * 5,
        'mapping_notes': [''] * 5,
    })


def _detection_frame():
    # 6 pre-filter proteins; P6 is filtered out (absent from the spine).
    return pd.DataFrame({
        'protein_id': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'detections_WT': [2, 2, 2, 0, 1, 0],
        'detections_KO': [2, 2, 1, 2, 1, 0],
        'filter_status': ['PASSED', 'PASSED', 'PASSED', 'SINGLE-GROUP',
                          'PARTIAL', 'ABSENT'],
    })


def _imputed_frame():
    ab = {f'abundance_{s}': [20.0 + i for i in range(5)] for s in SAMPLES}
    pc = {f'peptide_count_{s}': [2, 2, 1, 1, 1] for s in SAMPLES}
    return pd.DataFrame({'protein_id': ['P1', 'P2', 'P3', 'P4', 'P5'], **ab, **pc})


def _diff_abundance_frame():
    return pd.DataFrame({
        'protein_id': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'gene_symbol': ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'],
        'log2_fc': [2.0, -0.1, 3.0, -2.5, 0.0],
        'avg_abundance': [20.0, 21.0, 22.0, 23.0, 24.0],
        'limma_t': [5.0, 0.2, 6.0, -4.0, 0.0],
        'limma_pvalue': [0.001, 0.8, 0.0005, 0.002, 0.99],
        'limma_adj_pvalue': [0.01, 0.9, 0.005, 0.02, 0.99],
        'deqms_t': [5.5, 0.2, 6.5, -4.5, 0.0],
        'deqms_pvalue': [0.0008, 0.8, 0.0003, 0.0015, 0.99],
        'deqms_adj_pvalue': [0.008, 0.9, 0.004, 0.015, 0.99],
        'n_peptides': pd.array([6, 4, 5, 3, 2], dtype='Int64'),
        'significant': [True, False, True, True, False],
        'direction': ['up', 'ns', 'up', 'down', 'ns'],
        'contrast': ['KO_vs_WT'] * 5,
        'numerator': ['KO'] * 5,
        'denominator': ['WT'] * 5,
    })


def _enrichment_frame():
    return pd.DataFrame({
        'term_id': ['GOBP_X', 'GOBP_Y'],
        'term_name': ['GOBP_X', 'GOBP_Y'],
        'library': ['GO_BP', 'GO_BP'],
        'analysis_type': ['ORA', 'GSEA'],
        'contrast': ['KO_vs_WT', 'KO_vs_WT'],
        'pvalue': [0.01, 0.02],
        'adj_pvalue': [0.05, 0.06],
        'enrichment_score': [1.5, -1.2],
        'odds_ratio': [3.0, None],
        'combined_score': [10.0, None],
        'gene_set_size': pd.array([20, 30], dtype='Int64'),
        'overlap_size': pd.array([3, 5], dtype='Int64'),
        'overlap_genes': ['Gene1;Gene3', 'Gene1;Gene4'],
    })


def _protein_term_frame():
    return pd.DataFrame({
        'gene_symbol': ['Gene1', 'Gene3'],
        'protein_id': ['P1', 'P3'],
        'term_id': ['GOBP_X', 'GOBP_X'],
        'term_name': ['GOBP_X', 'GOBP_X'],
        'library': ['GO_BP', 'GO_BP'],
        'in_significant_set': [True, True],
        'is_leading_edge': [False, True],
        'contrast': ['KO_vs_WT', 'KO_vs_WT'],
    })


def _uniprot_frame():
    return pd.DataFrame({
        'protein_id': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'protein_name': [f'name{i}' for i in range(5)],
        'function_description': [f'func{i}' for i in range(5)],
        'subcellular_location': ['Cytoplasm'] * 5,
        'tissue_expression': ['brain'] * 5,
        'keywords': ['kw'] * 5,
        'uniprot_query_status': ['success'] * 5,
    })


def _pubmed_frame():
    return pd.DataFrame({
        'protein_id': ['P1', 'P2'],
        'search_term': ['ketamine', 'ketamine'],
        'mouse_symbol_used': ['Gene1', 'Gene2'],
        'human_symbol_used': ['GENE1', 'GENE2'],
        'mouse_hit_count': pd.array([1, 0], dtype='Int64'),
        'human_hit_count': pd.array([2, 0], dtype='Int64'),
        'mouse_total_pubs': pd.array([10, 5], dtype='Int64'),
        'human_total_pubs': pd.array([12, 5], dtype='Int64'),
        'normalized_score': pd.array([1.3, None], dtype='Float64'),
        'pubmed_query_status': ['success', 'success'],
    })


def _disgenet_frame():
    return pd.DataFrame({
        'protein_id': ['P1'],
        'human_symbol_queried': ['GENE1'],
        'disease_id': ['C0001'],
        'disease_name': ['DiseaseA'],
        'disease_type': ['[disease]'],
        'gda_score': pd.array([0.4], dtype='Float64'),
        'evidence_index': pd.array([1.0], dtype='Float64'),
        'n_publications': pd.array([3], dtype='Int64'),
        'source': ['Strong'],
        'disgenet_query_status': ['success'],
    })


def _dgidb_empty_frame():
    # Empty but schema-correct: simulates a disabled DGIdb query.
    cols = ['protein_id', 'human_symbol_queried', 'drug_name', 'drug_concept_id',
            'interaction_type', 'interaction_score', 'approval_status',
            'n_sources', 'sources', 'pmids', 'dgidb_query_status']
    return pd.DataFrame({c: pd.Series(dtype='object') for c in cols})


def _ctd_frame():
    return pd.DataFrame({
        'protein_id': ['P1'],
        'gene_symbol_queried': ['Gene1'],
        'query_organism': ['mouse'],
        'chemical_name': ['ChemX'],
        'chemical_mesh_id': ['D0001'],
        'chemical_cas_rn': ['1-2-3'],
        'interaction_text': ['increases expression'],
        'interaction_actions': ['increases^expression'],
        'n_publications': pd.array([2], dtype='Int64'),
        'pmids': ['111;222'],
        'ctd_query_status': ['success'],
    })


def _params_dict():
    return {
        'project': {'name': 'SYN_RUN', 'organism': 'mouse'},
        'design': {'group_column': 'genotype', 'batch_column': None,
                   'contrasts': ['KO_vs_WT']},
        'databases': {'enabled': ['uniprot', 'pubmed', 'disgenet', 'ctd'],
                      'pubmed': {'search_terms': ['ketamine', 'TBI'],
                                 'min_pubs_for_score': 5}},
        'normalization': {'method': 'median'},
        'differential_abundance': {'significance': {'fdr_threshold': 0.05}},
    }


@pytest.fixture
def synthetic_inputs(tmp_path):
    '''Write all 14 input files to tmp_path and return the input-path dict.'''
    rid = 'SYN_RUN'
    paths = {}
    writers = {
        'mapping': (_mapping_frame(), 'id_mapping.parquet'),
        'flags': (_flags_frame(), 'sample_flags.parquet'),
        'imputed': (_imputed_frame(), 'imputed_matrix.parquet'),
        'mask': (_mask_frame(), 'imputation_mask.parquet'),
        'diff': (_diff_abundance_frame(), 'diff_abundance_results.parquet'),
        'enrich': (_enrichment_frame(), 'enrichment_results.parquet'),
        'ptm': (_protein_term_frame(), 'protein_term_mapping.parquet'),
        'uniprot': (_uniprot_frame(), 'uniprot_annotations.parquet'),
        'pubmed': (_pubmed_frame(), 'pubmed_cooccurrence.parquet'),
        'disgenet': (_disgenet_frame(), 'disgenet_associations.parquet'),
        'dgidb': (_dgidb_empty_frame(), 'dgidb_interactions.parquet'),
        'ctd': (_ctd_frame(), 'ctd_interactions.parquet'),
    }
    for key, (df, suffix) in writers.items():
        p = tmp_path / f'{rid}.{suffix}'
        df.to_parquet(p)
        paths[key] = p
    # Detection table is CSV.
    det_p = tmp_path / f'{rid}.detection_filter_table.csv'
    _detection_frame().to_csv(det_p, index=False)
    paths['detection'] = det_p
    # params.yml
    params_p = tmp_path / f'{rid}_params.yml'
    params_p.write_text(yaml.safe_dump(_params_dict()))
    paths['params'] = params_p
    paths['run_id'] = rid
    return paths


@pytest.fixture
def assembled_db(synthetic_inputs, tmp_path):
    '''Run the full CLI and return (db_path, outdir).'''
    outdir = tmp_path / 'out'
    p = synthetic_inputs
    cmd = [
        sys.executable, str(SCRIPT),
        '--mapping-table', str(p['mapping']),
        '--detection-table', str(p['detection']),
        '--sample-flags', str(p['flags']),
        '--imputed-matrix', str(p['imputed']),
        '--imputation-mask', str(p['mask']),
        '--diff-abundance', str(p['diff']),
        '--enrichment-results', str(p['enrich']),
        '--protein-term-mapping', str(p['ptm']),
        '--uniprot', str(p['uniprot']),
        '--pubmed', str(p['pubmed']),
        '--disgenet', str(p['disgenet']),
        '--dgidb', str(p['dgidb']),
        '--ctd', str(p['ctd']),
        '--params', str(p['params']),
        '--run-id', p['run_id'],
        '--outdir', str(outdir),
        '--pipeline-version', 'test-1.0',
        '--run-date', '2026-07-10T00:00:00',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f'assembly failed:\n{result.stderr}'
    return outdir / 'prosift_results.db', outdir


# ============================================================
# Section 1: Derived-column unit tests
# ============================================================

def test_imputation_fraction_values():
    frac = ra.derive_imputation_fraction(_mask_frame())
    assert frac['P1'] == pytest.approx(0.0)
    assert frac['P2'] == pytest.approx(0.5)
    assert frac['P5'] == pytest.approx(1.0)


# ============================================================
# Section 2: Params flattening unit tests
# ============================================================

def test_flatten_params_nested_and_lists():
    flat = dict(ra._flatten_params(_params_dict()))
    assert flat['project.organism'] == 'mouse'
    assert flat['design.contrasts'] == ['KO_vs_WT']            # list stored whole
    assert flat['databases.pubmed.min_pubs_for_score'] == 5    # deep nesting
    assert flat['design.batch_column'] is None


def test_value_type_bool_checked_before_int():
    assert ra._value_type(True) == 'bool'
    assert ra._value_type(1) == 'int'
    assert ra._value_type(1.0) == 'float'
    assert ra._value_type(['a']) == 'list'
    assert ra._value_type(None) == 'null'
    assert ra._value_type('x') == 'string'


def test_build_run_parameters_serialises_lists_as_json():
    df = ra.build_run_parameters({'params': _params_dict()})
    row = df[df['key'] == 'design.contrasts'].iloc[0]
    assert row['value_type'] == 'list'
    assert row['value'] == '["KO_vs_WT"]'


# ============================================================
# Section 3: Table-builder unit tests
# ============================================================

def test_build_proteins_columns_and_detection_join():
    data = {'mapping': _mapping_frame(), 'detection': _detection_frame(),
            'mask': _mask_frame(), 'flags': _flags_frame()}
    proteins = ra.build_proteins(data)
    assert list(proteins.columns) == [
        'protein_id', 'gene_symbol', 'entrez_id_mouse', 'ensembl_gene_mouse',
        'human_ortholog_symbol', 'human_ortholog_entrez', 'ortholog_mapping_status',
        'detection_category', 'imputation_fraction',
    ]
    # gene_symbol_mouse was renamed; P4 is SINGLE-GROUP in the detection table.
    p4 = proteins[proteins['protein_id'] == 'P4'].iloc[0]
    assert p4['gene_symbol'] == 'Gene4'
    assert p4['detection_category'] == 'SINGLE-GROUP'
    # Only spine proteins (P1-P5); the filtered-out P6 must not appear.
    assert set(proteins['protein_id']) == {'P1', 'P2', 'P3', 'P4', 'P5'}


def test_build_sample_abundances_shape_and_group():
    data = {'imputed': _imputed_frame(), 'mask': _mask_frame(), 'flags': _flags_frame()}
    sa = ra.build_sample_abundances(data)
    assert list(sa.columns) == ['protein_id', 'sample_id', 'abundance', 'group',
                                'imputation_status']
    assert len(sa) == 5 * 4                       # proteins x samples
    # Group label resolved from sample_flags.
    assert set(sa[sa['sample_id'] == 'S_WT-1']['group']) == {'WT'}
    # Imputation status joined from the mask (P2 in S_WT-1 was 'mar').
    cell = sa[(sa['protein_id'] == 'P2') & (sa['sample_id'] == 'S_WT-1')].iloc[0]
    assert cell['imputation_status'] == 'mar'


# ============================================================
# Section 4: Integration tests (full CLI -> SQLite)
# ============================================================

EXPECTED_TABLES = {
    'proteins', 'differential_abundance', 'sample_abundances', 'sample_qc_flags',
    'enrichment_results', 'protein_term_mapping',
    'uniprot_annotations', 'pubmed_cooccurrence', 'disease_associations',
    'drug_interactions', 'chemical_interactions',
    'run_metadata', 'run_parameters',
}


def test_all_thirteen_tables_created(assembled_db):
    db_path, _ = assembled_db
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert tables >= EXPECTED_TABLES
    assert len(EXPECTED_TABLES) == 13


def test_row_counts(assembled_db):
    db_path, _ = assembled_db
    conn = sqlite3.connect(db_path)

    def n(table):
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    assert n('proteins') == 5
    assert n('differential_abundance') == 5           # proteins x 1 contrast
    assert n('sample_abundances') == 20               # 5 proteins x 4 samples
    assert n('drug_interactions') == 0                # disabled -> empty
    assert n('run_metadata') == 1
    conn.close()


def test_primary_keys_declared(assembled_db):
    db_path, _ = assembled_db
    conn = sqlite3.connect(db_path)

    def pk(table):
        return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")') if r[5] > 0]

    assert pk('proteins') == ['protein_id']
    assert pk('differential_abundance') == ['protein_id', 'contrast']
    assert pk('sample_abundances') == ['protein_id', 'sample_id']
    assert pk('run_metadata') == ['run_id']
    assert pk('run_parameters') == ['key']
    conn.close()


def test_fk_integrity_clean(assembled_db):
    db_path, _ = assembled_db
    conn = sqlite3.connect(db_path)
    valid = {r[0] for r in conn.execute('SELECT protein_id FROM proteins')}
    for table in ra.FK_CHILD_TABLES:
        child = {r[0] for r in conn.execute(
            f'SELECT DISTINCT protein_id FROM "{table}"') if r[0] is not None}
        assert child <= valid, f'{table} has orphan protein_ids'
    conn.close()


def test_derived_columns_in_db(assembled_db):
    db_path, _ = assembled_db
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT protein_id, imputation_fraction '
                     'FROM proteins', conn).set_index('protein_id')
    conn.close()
    # qc_flag_count was dropped (2026-07-13); QC flags now live in the
    # sample_qc_flags table and are surfaced per-protein by the Module 08 card.
    assert 'qc_flag_count' not in df.columns
    assert df.loc['P1', 'imputation_fraction'] == pytest.approx(0.0)
    assert df.loc['P5', 'imputation_fraction'] == pytest.approx(1.0)


def test_sample_qc_flags_table(assembled_db):
    db_path, _ = assembled_db
    conn = sqlite3.connect(db_path)
    # One row per sample, keyed by sample_id, carrying group + the four flags.
    df = pd.read_sql('SELECT * FROM sample_qc_flags', conn).set_index('sample_id')
    pk = [r[1] for r in conn.execute('PRAGMA table_info("sample_qc_flags")') if r[5] > 0]
    conn.close()
    assert pk == ['sample_id']
    assert len(df) == 4                                   # 4 synthetic samples
    for col in _FLAG_COLS + ['group', 'n_flags']:
        assert col in df.columns
    # S_WT-1 fires extreme_median; S_KO-1 fires pca_outlier (see _flags_frame).
    assert bool(df.loc['S_WT-1', 'flag_extreme_median']) is True
    assert bool(df.loc['S_KO-1', 'flag_pca_outlier']) is True


def test_run_metadata_content(assembled_db):
    db_path, _ = assembled_db
    conn = sqlite3.connect(db_path)
    row = pd.read_sql('SELECT * FROM run_metadata', conn).iloc[0]
    conn.close()
    assert row['run_id'] == 'SYN_RUN'
    assert row['organism'] == 'mouse'
    assert row['contrasts'] == '["KO_vs_WT"]'
    assert row['n_proteins'] == 5
    assert row['n_samples'] == 4
    assert row['pipeline_version'] == 'test-1.0'


def test_reserved_word_group_column_queryable(assembled_db):
    # 'group' is a SQLite reserved word; the column must still be selectable.
    db_path, _ = assembled_db
    conn = sqlite3.connect(db_path)
    counts = dict(conn.execute(
        'SELECT "group", COUNT(*) FROM sample_abundances GROUP BY "group"'))
    conn.close()
    assert counts == {'WT': 10, 'KO': 10}


def test_disabled_db_empty_table_and_summary(assembled_db):
    db_path, outdir = assembled_db
    conn = sqlite3.connect(db_path)
    # The empty (disabled) DGIdb parquet -> empty drug_interactions table that
    # still has its full schema.
    cols = [r[1] for r in conn.execute('PRAGMA table_info(drug_interactions)')]
    assert 'drug_name' in cols and 'approval_status' in cols
    assert conn.execute('SELECT COUNT(*) FROM drug_interactions').fetchone()[0] == 0
    conn.close()
    summary = (outdir / 'SYN_RUN.assembly_summary.txt').read_text()
    assert 'dgidb' in summary            # reported as an empty table


def test_csv_exports_written(assembled_db):
    _, outdir = assembled_db
    da = pd.read_csv(outdir / 'SYN_RUN.diff_abundance_results.csv')
    sig = pd.read_csv(outdir / 'SYN_RUN.significant_proteins.csv')
    assert len(da) == 5
    # Significant rows: P1, P3, P4 (significant == True in the fixture).
    assert set(sig['protein_id']) == {'P1', 'P3', 'P4'}
    # Significant export carries identity columns joined from proteins.
    assert 'gene_symbol' in sig.columns and 'detection_category' in sig.columns



def test_orientation_columns_reach_db_and_csv(assembled_db):
    # The numerator/denominator columns make the log2_fc sign self-describing;
    # they must survive into both the SQLite table and the significant export.
    db_path, outdir = assembled_db
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        'SELECT DISTINCT contrast, numerator, denominator FROM differential_abundance'
    ).fetchall()
    conn.close()
    assert rows == [('KO_vs_WT', 'KO', 'WT')]
    sig = pd.read_csv(outdir / 'SYN_RUN.significant_proteins.csv')
    assert {'numerator', 'denominator'} <= set(sig.columns)
    assert (sig['numerator'] == 'KO').all() and (sig['denominator'] == 'WT').all()
