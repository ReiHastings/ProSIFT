#!/usr/bin/env python3
# title: test_validate_inputs.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-08
# last modified: 2026-07-08
#
# purpose:
#   Integration tests for Module 01 VALIDATE_INPUTS (bin/validate_inputs.py),
#   focused on the 2026-07-08 non-positive abundance policy: on raw data, values
#   <= 0 are converted to NaN, a provenance mask is emitted, and the paired
#   peptide count is zeroed. These run the real CLI against the committed
#   benchmark fixture (whose answer key was validated independently), so a
#   failure here means the module diverged from the decided contract.
#
# inputs:
#   tests/fixtures/benchmark/benchmark_abundance.csv (+ metadata, params, truth)
#
# outputs:
#   Test results (stdout via pytest). Intermediate parquet written to tmp_path.
#
# usage example:
#   pytest tests/test_validate_inputs.py -v
#
#   copy/paste: pytest tests/test_validate_inputs.py -v

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN = PROJECT_ROOT / 'bin' / 'validate_inputs.py'
BENCH = PROJECT_ROOT / 'tests' / 'fixtures' / 'benchmark'


# ============================================================
# Helpers
# ============================================================

def _run_validate(abundance, metadata, params, run_id, outdir):
    '''Invoke the validate_inputs CLI as a subprocess; return CompletedProcess.'''
    return subprocess.run(
        [sys.executable, str(BIN),
         '--abundance', str(abundance),
         '--metadata', str(metadata),
         '--params', str(params),
         '--run-id', run_id,
         '--outdir', str(outdir)],
        capture_output=True, text=True,
    )


@pytest.fixture(scope='module')
def validated(tmp_path_factory):
    '''Run validate_inputs once on the benchmark; return loaded outputs.'''
    outdir = tmp_path_factory.mktemp('validate')
    proc = _run_validate(
        BENCH / 'benchmark_abundance.csv',
        BENCH / 'benchmark_metadata.csv',
        BENCH / 'benchmark_params.yml',
        'bench', outdir,
    )
    assert proc.returncode == 0, f'validate_inputs failed:\n{proc.stderr}'
    matrix = pd.read_parquet(outdir / 'bench.validated_matrix.parquet').set_index('protein_id')
    mask = pd.read_parquet(outdir / 'bench.nonpositive_mask.parquet').set_index('protein_id')
    return {'matrix': matrix, 'mask': mask, 'outdir': outdir}


# ============================================================
# Section 1: non-positive conversion
# ============================================================

class TestNonPositiveConversion:

    def test_zero_converted_to_nan(self, validated):
        '''EDGE_ZERO's literal 0.0 in KO-1 becomes NaN in the validated matrix.'''
        assert pd.isna(validated['matrix'].loc['EDGE_ZERO', 'abundance_KO-1'])

    def test_negative_converted_to_nan(self, validated):
        '''EDGE_NEG's negative in WT-2 becomes NaN in the validated matrix.'''
        assert pd.isna(validated['matrix'].loc['EDGE_NEG', 'abundance_WT-2'])

    def test_positive_values_untouched(self, validated):
        '''A quantified positive value is preserved unchanged.'''
        assert validated['matrix'].loc['EDGE_ZERO', 'abundance_KO-2'] > 0
        assert validated['matrix'].loc['EDGE_NEG', 'abundance_WT-1'] > 0


# ============================================================
# Section 2: provenance mask
# ============================================================

class TestNonPositiveMask:

    def test_mask_columns_match_matrix_abundance_columns(self, validated):
        '''
        The mask must use the same prefixed abundance column names as the matrix
        so a downstream consumer can align by direct intersection (review finding
        #1). Bare sample ids would silently fail to join.
        '''
        matrix_abund = [c for c in validated['matrix'].columns
                        if c.startswith('abundance_')]
        assert list(validated['mask'].columns) == matrix_abund

    def test_mask_marks_exactly_the_converted_cells(self, validated):
        '''The mask is True at EDGE_ZERO/KO-1 and EDGE_NEG/WT-2, False elsewhere.'''
        mask = validated['mask']
        assert bool(mask.loc['EDGE_ZERO', 'abundance_KO-1']) is True
        assert bool(mask.loc['EDGE_NEG', 'abundance_WT-2']) is True
        # Total True cells equals the two designed non-positive values.
        assert int(mask.to_numpy().sum()) == 2

    def test_mask_false_for_missing_but_not_nonpositive(self, validated):
        '''An originally-blank cell (EDGE_MNAR_WT) is missing but was not
        converted, so the mask marks it False (missing != non-positive).'''
        assert bool(validated['mask'].loc['EDGE_MNAR_WT', 'abundance_WT-1']) is False


# ============================================================
# Section 3: paired peptide-count zeroing
# ============================================================
# This code path REPAIRS an inconsistent input (abundance non-positive but
# peptide count positive). It therefore MUST be tested with a deliberately
# inconsistent input. The committed benchmark is self-consistent (peptide 0
# wherever abundance is not quantified), so a test against it would pass whether
# or not the zeroing runs -- it cannot exercise the repair. We build the
# inconsistent state directly here.

_WT = ['WT-1', 'WT-2', 'WT-3']
_KO = ['KO-1', 'KO-2', 'KO-3']


def _write_inconsistent_input(path):
    '''
    Write a tiny raw-abundance CSV in which two cells hold a non-positive
    abundance paired with a POSITIVE peptide count -- the exact inconsistency
    the validation zeroing exists to repair. Every other cell is quantified, so
    no column or row becomes empty. Sample ids match benchmark_metadata.csv.
    '''
    samples = _WT + _KO

    def base_row(pid):
        return {'protein_id': pid,
                **{f'abundance_{s}': 1000.0 for s in samples},
                **{f'peptide_count_{s}': 4 for s in samples}}

    rows = [base_row('P1')]                 # fully quantified control

    r2 = base_row('P2')
    r2['abundance_KO-1'] = 0.0              # literal zero abundance ...
    r2['peptide_count_KO-1'] = 5            # ... but claims 5 peptides
    rows.append(r2)

    r3 = base_row('P3')
    r3['abundance_WT-2'] = -3.0             # negative abundance ...
    r3['peptide_count_WT-2'] = 7            # ... but claims 7 peptides
    rows.append(r3)

    pd.DataFrame(rows).to_csv(path, index=False)


class TestPeptideCountConsistency:

    def test_zeroing_repairs_positive_count_at_nonpositive_cell(self, tmp_path):
        '''
        Discriminating test: the input carries peptide counts 5 and 7 at two
        non-positive abundance cells. Only the validation zeroing can drive them
        to 0, so this assertion FAILS if that code path breaks -- unlike a test
        against the self-consistent benchmark, where the cells are already 0.
        '''
        abund = tmp_path / 'inconsistent_abundance.csv'
        _write_inconsistent_input(abund)
        proc = _run_validate(
            abund, BENCH / 'benchmark_metadata.csv',
            BENCH / 'benchmark_params.yml', 'incon', tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        matrix = pd.read_parquet(
            tmp_path / 'incon.validated_matrix.parquet'
        ).set_index('protein_id')

        # Precondition: the non-positive abundances were converted to NaN.
        assert pd.isna(matrix.loc['P2', 'abundance_KO-1'])
        assert pd.isna(matrix.loc['P3', 'abundance_WT-2'])
        # The actual assertion under test: the positive counts were zeroed.
        assert matrix.loc['P2', 'peptide_count_KO-1'] == 0   # was 5
        assert matrix.loc['P3', 'peptide_count_WT-2'] == 0   # was 7
        # A quantified cell's peptide count is left untouched.
        assert matrix.loc['P1', 'peptide_count_KO-1'] == 4


# ============================================================
# Section 4: abundance_type enforcement + scale gating
# ============================================================

class TestAbundanceTypeHandling:

    def test_missing_abundance_type_errors(self, tmp_path):
        '''abundance_type is required; its absence is a hard error, not a
        silent default to raw (which would destroy zeros by default).'''
        params = yaml.safe_load((BENCH / 'benchmark_params.yml').read_text())
        del params['input']['abundance_type']
        p = tmp_path / 'no_type_params.yml'
        p.write_text(yaml.safe_dump(params))
        proc = _run_validate(
            BENCH / 'benchmark_abundance.csv', BENCH / 'benchmark_metadata.csv',
            p, 'notype', tmp_path,
        )
        assert proc.returncode != 0

    def test_log2_preserves_zero_and_mask_all_false(self, tmp_path):
        '''
        With abundance_type='log2', a 0 is a legitimate value: it must be
        preserved and the mask must be all-False (conversion is raw-only).
        '''
        params = yaml.safe_load((BENCH / 'benchmark_params.yml').read_text())
        params['input']['abundance_type'] = 'log2'
        p = tmp_path / 'log2_params.yml'
        p.write_text(yaml.safe_dump(params))
        proc = _run_validate(
            BENCH / 'benchmark_abundance.csv', BENCH / 'benchmark_metadata.csv',
            p, 'log2run', tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        matrix = pd.read_parquet(tmp_path / 'log2run.validated_matrix.parquet').set_index('protein_id')
        mask = pd.read_parquet(tmp_path / 'log2run.nonpositive_mask.parquet').set_index('protein_id')
        # The literal 0.0 survives as a real value under log2 semantics.
        assert matrix.loc['EDGE_ZERO', 'abundance_KO-1'] == 0.0
        assert int(mask.to_numpy().sum()) == 0



# ============================================================
# Contrast orientation (run_id order vs contrast order)
# ============================================================

sys.path.insert(0, str(PROJECT_ROOT / 'bin'))
from validate_inputs import run_id_order_reversed  # noqa: E402


class TestRunIdOrderReversed:

    @pytest.mark.parametrize('run_id,num,den', [
        ('CTXcyto_WT_vs_CTXcyto_KO', 'KO', 'WT'),
        ('CTXcyto_WT_vs_HIPcyto_WT', 'HIP', 'CTX'),
        ('HIPcyto_WT_vs_HIPsynap_WT', 'synap', 'cyto'),
        ('WT_vs_WT2', 'WT2', 'WT'),        # nested names: whole token wins
        ('wt_vs_ko', 'KO', 'WT'),          # case-insensitive
    ])
    def test_production_names_are_flagged(self, run_id, num, den):
        assert run_id_order_reversed(run_id, num, den)

    @pytest.mark.parametrize('run_id,num,den', [
        ('CTXcyto_KO_vs_CTXcyto_WT', 'KO', 'WT'),   # same order: fine
        ('bench', 'KO', 'WT'),                      # no _vs_ in run name
        ('KO_WT_vs_other', 'KO', 'WT'),             # both groups on one side
        ('A_vs_B', 'KO', 'WT'),                     # groups not mentioned
        ('Batch1_vs_Area2', 'A', 'B'),              # 'A' must not match 'Area2'
    ])
    def test_non_reversed_or_ambiguous_not_flagged(self, run_id, num, den):
        assert not run_id_order_reversed(run_id, num, den)


class TestContrastOrientationCli:

    def _run(self, run_id, tmp_path):
        proc = _run_validate(
            BENCH / 'benchmark_abundance.csv', BENCH / 'benchmark_metadata.csv',
            BENCH / 'benchmark_params.yml', run_id, tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        report = (tmp_path / f'{run_id}.validation_report_part1.txt').read_text()
        return proc, report

    def test_reversed_run_id_warns_but_passes(self, tmp_path):
        # Benchmark contrast is KO_vs_WT; the run name puts WT first.
        proc, report = self._run('WT_vs_KO', tmp_path)
        assert 'KO_vs_WT: positive log2 FC / NES = higher in KO than WT' in report
        assert 'opposite order' in report
        assert 'opposite order' in proc.stderr

    def test_matching_run_id_does_not_warn(self, tmp_path):
        proc, report = self._run('KO_vs_WT', tmp_path)
        assert 'CONTRAST ORIENTATION' in report
        assert 'opposite order' not in report
        assert 'opposite order' not in proc.stderr



def test_orientation_check_tolerates_null_design():
    # 'design:' present but empty parses to None; the check must not crash.
    from validate_inputs import Report, check_contrast_orientation
    report = Report('RUN')
    check_contrast_orientation('WT_vs_KO', {'design': None}, report)
    assert report.warning_count == 0
