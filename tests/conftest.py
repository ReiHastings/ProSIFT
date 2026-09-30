#!/usr/bin/env python3
# title: conftest.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-24
# last modified: 2026-04-24
#
# purpose:
#   Shared pytest fixtures for the ProSIFT test suite. Fixtures defined here
#   are auto-discovered by every test file under tests/ -- no import needed.
#   Test files reference a fixture by its parameter name; pytest finds the
#   matching @pytest.fixture in this file and passes the object in.
#
#   Convention: start narrow. Add a fixture here only when two or more test
#   files would reuse it. Fixtures used by one file belong in that file.
#
#   Sample IDs follow the CTXcyto benchmark convention so tests read against
#   familiar labels. Random values are seeded for determinism.
#
# inputs:
#   None (fixtures build inputs in-memory)
#
# outputs:
#   None (provides fixture objects to the test runner)
#
# usage example:
#   # In any test file under tests/:
#   def test_my_function(synthetic_filtered_matrix, synthetic_metadata,
#                        minimal_params):
#       # fixtures are already built; go straight to the test
#       ...

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ============================================================
# CONSTANTS: shared across fixtures
# ============================================================

# Project root (parent of tests/). Used by tests that need to locate bin/
# scripts for subprocess-based smoke tests.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = PROJECT_ROOT / 'bin'

# Sample layout: 3 WT + 3 KO, matching the CTXcyto_WT_vs_CTXcyto_KO benchmark.
_SAMPLE_IDS = [
    'CTXcyto_WT-1', 'CTXcyto_WT-2', 'CTXcyto_WT-3',
    'CTXcyto_KO-1', 'CTXcyto_KO-2', 'CTXcyto_KO-3',
]
_GROUPS = ['WT', 'WT', 'WT', 'KO', 'KO', 'KO']

# Protein count for synthetic fixtures. Small enough to keep tests fast and
# readable; large enough that statistical summaries are meaningful.
_N_PROTEINS = 20


# ============================================================
# Section 1: Seeded random number generation
# ============================================================

@pytest.fixture
def rng():
    '''
    Seeded numpy RNG for reproducible synthetic data within a single test.

    Returns a fresh Generator each test (function scope), so tests do not
    interfere with each other. Use this when you need extra randomness
    beyond what the prebuilt fixtures provide.
    '''
    return np.random.default_rng(seed=42)


# ============================================================
# Section 2: Synthetic inputs matching Module 01 outputs
# ============================================================
# These fixtures mirror the shape of files produced by Module 01 and consumed
# by Modules 02+. All three are deterministic (seeded) so tests can assert
# exact values.

@pytest.fixture
def synthetic_filtered_matrix():
    '''
    Minimal filtered_matrix.parquet shape, as produced by Module 01
    FILTER_PROTEINS and consumed by Module 02 PRENORM_QC.

    Schema:
      protein_id                 -- UniProt-like accession (str)
      <sample_id>                -- log2 abundance (float; no prefix, matches
                                    the CTXcyto benchmark convention)
      peptide_count_<sample_id>  -- per-sample peptide counts (int)

    Shape: 20 proteins x 6 samples. Log2 values centered on 22 (realistic
    DIA-NN range). No missing values by default; tests that need NaN should
    inject them from the returned frame.
    '''
    rng = np.random.default_rng(seed=42)
    protein_ids = [f'P{i:05d}' for i in range(_N_PROTEINS)]

    # Abundance columns: log2 scale, small inter-sample variation
    abund_data = {
        sid: rng.normal(loc=22.0, scale=2.0, size=_N_PROTEINS)
        for sid in _SAMPLE_IDS
    }

    # Peptide count columns: integer counts 1-15
    pep_data = {
        f'peptide_count_{sid}': rng.integers(low=1, high=15, size=_N_PROTEINS)
        for sid in _SAMPLE_IDS
    }

    return pd.DataFrame({
        'protein_id': protein_ids,
        **abund_data,
        **pep_data,
    })


@pytest.fixture
def synthetic_metadata():
    '''
    Minimal validated_metadata.parquet, as produced by Module 01
    VALIDATE_INPUTS. One row per sample; 'group' column supplies the
    condition label that Module 02 reads via params.design.group_column.
    '''
    return pd.DataFrame({
        'sample_id': _SAMPLE_IDS,
        'group': _GROUPS,
    })


@pytest.fixture
def synthetic_id_mapping():
    '''
    Minimal id_mapping.parquet shape, as produced by Module 01
    UNIPROT_MAPPING (v3 schema: protein_id as the join key, not input_id).

    Module 02 currently loads this input but does not consume it beyond the
    read_parquet call. The fixture provides a valid-shaped stub so smoke
    tests can run the full CLI. Expand this schema when Module 01 or other
    modules need richer mapping fixtures.
    '''
    protein_ids = [f'P{i:05d}' for i in range(_N_PROTEINS)]
    return pd.DataFrame({
        'protein_id': protein_ids,
        'uniprot_accession': protein_ids,
        'gene_symbol': [f'Gene{i}' for i in range(_N_PROTEINS)],
        'ortholog_mapping_status': ['not_applicable'] * _N_PROTEINS,
    })


# ============================================================
# Section 3: Minimal params dict
# ============================================================

@pytest.fixture
def minimal_params():
    '''
    Minimum viable params dict, matching the fields Module 02 reads.

    Real params.yml files have many more entries (input validation,
    normalization, imputation, enrichment, database configs); this fixture
    covers only what PRENORM_QC touches:
        input.abundance_prefix  (empty string = bare sample IDs)
        input.abundance_type    ('log2' = no internal log2 transform needed)
        design.group_column     ('group' matches synthetic_metadata)

    Tests that need other fields should extend the dict from this fixture
    rather than redefining it from scratch.
    '''
    return {
        'input': {
            'abundance_prefix': '',
            'abundance_type': 'log2',
        },
        'design': {
            'group_column': 'group',
        },
    }
