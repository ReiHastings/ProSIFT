#!/usr/bin/env python3
# title: test_differential_abundance.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-09
# last modified: 2026-09-29
#
# purpose:
#   Unit tests for Module 04 DIFFERENTIAL_ABUNDANCE (bin/differential_abundance.py).
#   The statistical fit itself (limma + DEqMS) runs in R via rpy2 and is an
#   integration concern for a cluster run (see NOTE below); it is not exercised
#   here. The bulk of the module's decision logic is pure Python and is fully
#   covered:
#     - parse_and_validate_contrasts : contrast-string parsing + validation
#     - summarize_peptide_counts      : per-protein min-nonzero peptide count
#                                       (the DEqMS variance predictor)
#     - assemble_results              : R-output -> ProSIFT schema, significance
#                                       call (FDR + FC), direction assignment
#     - extract_abundance_and_peptide_cols / build_group_map : I/O parsing
#
#   assemble_results is the centerpiece: it decides which proteins are called
#   significant and their direction. We fabricate an R-output-shaped frame (no R
#   needed) and check the calls exactly.
#
#   Not tested here: the plot_* Plotly functions, write_summary_txt, main, and
#   the R fit (_run_one_contrast_r / limma+DEqMS). The R fit is validated by a
#   cluster integration run (test_differential_abundance_r.py), not this unit
#   file. rpy2 is loaded lazily inside _run_one_contrast_r, so importing the
#   module here is side-effect-free (no embedded R started).
#
# inputs:
#   None (tests build inputs in-memory).
#
# outputs:
#   Test results (stdout via pytest).
#
# usage example:
#   pytest tests/test_differential_abundance.py -v
#
#   copy/paste: pytest tests/test_differential_abundance.py -v

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add bin/ to path. differential_abundance imports cleanly (rpy2 is loaded
# lazily inside _run_one_contrast_r), so no import-time workaround is needed;
# the pure functions tested here never touch R.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from differential_abundance import (
    assemble_results,
    build_group_map,
    extract_abundance_and_peptide_cols,
    normalize_quarantine_samples,
    parse_and_validate_contrasts,
    parse_bool_param,
    plot_ma,
    plot_volcano,
    split_contrast,
    summarize_peptide_counts,
    validate_quarantine,
    write_provenance,
    write_summary_txt,
)

# ============================================================
# SHARED HELPERS
# ============================================================

def _metadata():
    return pd.DataFrame({
        'sample_id': ['WT-1', 'WT-2', 'WT-3', 'KO-1', 'KO-2', 'KO-3'],
        'genotype': ['WT', 'WT', 'WT', 'KO', 'KO', 'KO'],
    })


def _groups():
    '''
    The samples-present group labels for _metadata(), i.e. what main() passes to
    parse_and_validate_contrasts as `unique_groups`. Kept as a plain list because
    that is the real argument type: the validator sees the R factor levels, not
    the metadata frame.
    '''
    return ['KO', 'WT']


def _params(contrasts, fdr=0.05, fc=1.0):
    return {
        'design': {'group_column': 'genotype', 'contrasts': contrasts},
        'differential_abundance': {'significance': {'fdr_threshold': fdr,
                                                    'fc_threshold': fc}},
    }


# ============================================================
# Section 1: parse_and_validate_contrasts
# ============================================================

class TestParseContrasts:

    def test_valid_contrast_parsed(self):
        parsed = parse_and_validate_contrasts(_params(['KO_vs_WT']), _groups())
        assert parsed == [('KO_vs_WT', 'KO', 'WT', 'KO - WT')]

    def test_multiple_contrasts(self):
        parsed = parse_and_validate_contrasts(
            _params(['KO_vs_WT', 'WT_vs_KO']), _groups())
        assert len(parsed) == 2
        assert parsed[1] == ('WT_vs_KO', 'WT', 'KO', 'WT - KO')

    def test_missing_delimiter_raises(self):
        with pytest.raises(ValueError, match="_vs_"):
            parse_and_validate_contrasts(_params(['KOWT']), _groups())

    def test_empty_denominator_raises(self):
        with pytest.raises(ValueError, match='denominator is empty'):
            parse_and_validate_contrasts(_params(['KO_vs_']), _groups())

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match='not found'):
            parse_and_validate_contrasts(_params(['KO_vs_XX']), _groups())

    def test_no_contrasts_raises(self):
        with pytest.raises(ValueError, match='No contrasts defined'):
            parse_and_validate_contrasts(_params([]), _groups())


# ============================================================
# Section 2: summarize_peptide_counts (DEqMS variance predictor)
# ============================================================

class TestSummarizePeptideCounts:

    def test_min_of_nonzero_excludes_zeros(self):
        '''Per protein: minimum of nonzero counts; zeros (imputed positions) skip.'''
        pep = pd.DataFrame(
            {'s1': [3, 0], 's2': [0, 7], 's3': [5, 2]},
            index=pd.Index(['P1', 'P2'], name='protein_id'),
        )
        out = summarize_peptide_counts(pep)
        assert out['P1'] == 3          # min(3, 5), the 0 excluded
        assert out['P2'] == 2          # min(7, 2)

    def test_all_positive(self):
        pep = pd.DataFrame({'s1': [4], 's2': [5], 's3': [6]},
                           index=pd.Index(['P1'], name='protein_id'))
        assert summarize_peptide_counts(pep)['P1'] == 4

    def test_returns_int64(self):
        pep = pd.DataFrame({'s1': [3], 's2': [5]},
                           index=pd.Index(['P1'], name='protein_id'))
        assert summarize_peptide_counts(pep).dtype == np.int64

    @pytest.mark.xfail(
        reason=(
            'FINDING (2026-07-09): summarize_peptide_counts assumes every protein '
            'has >= 1 nonzero peptide count (Module 01 removes ABSENT proteins). '
            'An all-zero row hits np.nanmin over an all-NaN slice -> NaN + '
            'RuntimeWarning (escalated to error under pytest); in production the '
            'NaN.astype(int64) yields a garbage sentinel (-2^63) silently. '
            'Low reachability (a retained protein has detections, hence peptides), '
            'but unguarded. A len/all-zero guard would make it explicit.'
        ),
        strict=False,
    )
    def test_all_zero_row_does_not_corrupt(self):
        '''Adversarial probe: an all-zero peptide row. Correct behavior is a
        sensible value or a clear error, not a silent garbage int.'''
        pep = pd.DataFrame({'s1': [0], 's2': [0]},
                           index=pd.Index(['P1'], name='protein_id'))
        out = summarize_peptide_counts(pep)
        assert out['P1'] >= 0          # not a -2^63 garbage sentinel


# ============================================================
# Section 3: assemble_results (significance + direction centerpiece)
# ============================================================

class TestAssembleResults:

    def _deqms_raw(self):
        '''An R-output-shaped frame as DEqMS would return (no R needed).'''
        return pd.DataFrame({
            'protein_id':   ['P1', 'P2', 'P3', 'P4'],
            'logFC':        [2.0, -1.5, 2.0, 0.5],
            'AveExpr':      [20.0, 21.0, 19.0, 22.0],
            't':            [5.0, -4.0, 3.0, 4.0],
            'P.Value':      [1e-4, 1e-3, 0.10, 1e-3],
            'adj.P.Val':    [1e-3, 5e-3, 0.20, 5e-3],
            'B':            [3.0, 2.0, -1.0, 2.0],
            'sca.t':        [5.5, -4.5, 3.2, 4.2],
            'sca.P.Value':  [8e-5, 8e-4, 0.09, 8e-4],
            'sca.adj.pval': [1e-2, 1e-3, 0.20, 1e-2],
            'count':        [3, 5, 2, 4],
        })

    def _id_mapping(self):
        return pd.DataFrame({'protein_id': ['P1', 'P2', 'P3', 'P4'],
                             'gene_symbol': ['Gene1', 'Gene2', 'Gene3', 'Gene4']})

    def test_schema_and_column_order(self):
        out = assemble_results(self._deqms_raw(), self._id_mapping(),
                               _params(['KO_vs_WT']), 'DEqMS', 'KO_vs_WT')
        assert list(out.columns) == [
            'protein_id', 'gene_symbol', 'log2_fc', 'avg_abundance',
            'limma_t', 'limma_pvalue', 'limma_adj_pvalue',
            'deqms_t', 'deqms_pvalue', 'deqms_adj_pvalue',
            'n_peptides', 'significant', 'pvalue_undetermined',
            'direction', 'contrast', 'numerator', 'denominator',
        ]
        assert (out['contrast'] == 'KO_vs_WT').all()
        assert (out['numerator'] == 'KO').all()
        assert (out['denominator'] == 'WT').all()
        assert out.set_index('protein_id').loc['P1', 'n_peptides'] == 3

    def test_significance_and_direction_calls(self):
        '''
        With FDR<0.05 and |FC|>1, using the DEqMS adj p-value:
          P1: adj 0.01, FC +2   -> significant, up
          P2: adj 0.001, FC -1.5 -> significant, down
          P3: adj 0.20         -> ns (fails FDR)
          P4: adj 0.01, FC 0.5  -> ns (fails FC)
        '''
        out = assemble_results(self._deqms_raw(), self._id_mapping(),
                               _params(['KO_vs_WT']), 'DEqMS', 'KO_vs_WT'
                               ).set_index('protein_id')
        assert bool(out.loc['P1', 'significant']) is True
        assert out.loc['P1', 'direction'] == 'up'
        assert out.loc['P2', 'direction'] == 'down'
        assert bool(out.loc['P3', 'significant']) is False
        assert out.loc['P3', 'direction'] == 'ns'
        assert bool(out.loc['P4', 'significant']) is False      # fails FC
        assert out.loc['P4', 'direction'] == 'ns'

    def test_significance_uses_deqms_pvalue_when_deqms(self):
        '''
        A protein whose DEqMS adj p-value passes but whose limma adj p-value fails
        must be called significant (DEqMS is the primary p-value for method=DEqMS).
        '''
        raw = self._deqms_raw()
        raw.loc[raw['protein_id'] == 'P1', 'adj.P.Val'] = 0.30       # limma fails
        raw.loc[raw['protein_id'] == 'P1', 'sca.adj.pval'] = 0.01    # DEqMS passes
        out = assemble_results(raw, self._id_mapping(), _params(['KO_vs_WT']),
                               'DEqMS', 'KO_vs_WT').set_index('protein_id')
        assert bool(out.loc['P1', 'significant']) is True

    def test_limma_only_fills_deqms_columns_na(self):
        '''method=limma: DEqMS columns and n_peptides are NA; significance uses
        the limma adj p-value.'''
        raw = self._deqms_raw()[['protein_id', 'logFC', 'AveExpr', 't',
                                 'P.Value', 'adj.P.Val', 'B']]
        out = assemble_results(raw, self._id_mapping(), _params(['KO_vs_WT']),
                               'limma', 'KO_vs_WT').set_index('protein_id')
        assert pd.isna(out.loc['P1', 'deqms_t'])
        assert pd.isna(out.loc['P1', 'n_peptides'])
        assert bool(out.loc['P1', 'significant']) is True           # limma adj 1e-3

    def test_fc_threshold_zero_disables_fc_filter(self):
        '''fc_threshold=0 -> only FDR matters; a small-FC protein passing FDR is
        significant.'''
        raw = self._deqms_raw()
        out = assemble_results(raw, self._id_mapping(),
                               _params(['KO_vs_WT'], fc=0.0), 'DEqMS', 'KO_vs_WT'
                               ).set_index('protein_id')
        assert bool(out.loc['P4', 'significant']) is True          # FC no longer filters

    def test_gene_symbol_mouse_column_preferred(self):
        '''gene_symbol_mouse is used when present (ortholog-mapped runs).'''
        idmap = pd.DataFrame({'protein_id': ['P1', 'P2', 'P3', 'P4'],
                              'gene_symbol_mouse': ['M1', 'M2', 'M3', 'M4']})
        out = assemble_results(self._deqms_raw(), idmap, _params(['KO_vs_WT']),
                               'DEqMS', 'KO_vs_WT').set_index('protein_id')
        assert out.loc['P1', 'gene_symbol'] == 'M1'

    def test_missing_gene_column_yields_na(self):
        idmap = pd.DataFrame({'protein_id': ['P1', 'P2', 'P3', 'P4']})
        out = assemble_results(self._deqms_raw(), idmap, _params(['KO_vs_WT']),
                               'DEqMS', 'KO_vs_WT').set_index('protein_id')
        assert pd.isna(out.loc['P1', 'gene_symbol'])


# ============================================================
# Section 4: I/O parsing
# ============================================================

class TestIOParsing:

    def test_splits_abundance_and_peptides(self):
        matrix = pd.DataFrame({
            'protein_id': ['P1', 'P2'],
            'abundance_WT-1': [20.0, 21.0],
            'abundance_KO-1': [22.0, 23.0],
            'peptide_count_WT-1': [3, 5],
            'peptide_count_KO-1': [4, 6],
        })
        params = {'input': {'abundance_prefix': 'abundance_'}}
        abund, pep, sample_ids = extract_abundance_and_peptide_cols(matrix, params)
        assert sample_ids == ['WT-1', 'KO-1']
        assert list(abund.columns) == ['WT-1', 'KO-1']
        assert list(pep.columns) == ['peptide_count_WT-1', 'peptide_count_KO-1']

    def test_no_abundance_columns_raises(self):
        matrix = pd.DataFrame({'protein_id': ['P1'], 'peptide_count_WT-1': [3]})
        with pytest.raises(ValueError, match='No abundance columns'):
            extract_abundance_and_peptide_cols(matrix, {'input': {}})

    def test_build_group_map_missing_column_raises(self):
        meta = pd.DataFrame({'sample_id': ['WT-1'], 'condition': ['WT']})
        with pytest.raises(ValueError, match="group_column 'genotype' not found"):
            build_group_map(meta, {'design': {'group_column': 'genotype'}})


# ============================================================
# Section 5: validate_quarantine (mean-shift quarantine, Section 4.9)
# ============================================================

import fnmatch  # noqa: E402  (grouped with the quarantine tests it supports)

_QSAMPLES  = ['WT-1', 'WT-2', 'WT-3', 'KO-1', 'KO-2', 'KO-3']
_QGROUPMAP = {s: ('WT' if s.startswith('WT') else 'KO') for s in _QSAMPLES}
_QGROUPS   = [_QGROUPMAP[s] for s in _QSAMPLES]
_QCONTRAST = [('KO_vs_WT', 'KO', 'WT', 'KO - WT')]


class TestValidateQuarantine:

    def test_empty_no_quarantine_ok(self):
        # no exception
        validate_quarantine([], 'full', _QSAMPLES, _QGROUPMAP, _QCONTRAST, 2)

    def test_valid_single_quarantine_ok(self):
        validate_quarantine(['WT-3'], 'quarantined', _QSAMPLES,
                            _QGROUPMAP, _QCONTRAST, 2)

    def test_bad_primary_analysis_value(self):
        with pytest.raises(ValueError, match='primary_analysis'):
            validate_quarantine([], 'nope', _QSAMPLES, _QGROUPMAP, _QCONTRAST, 2)

    def test_unknown_sample_raises(self):
        with pytest.raises(ValueError, match='not found'):
            validate_quarantine(['WT-9'], 'full', _QSAMPLES,
                                _QGROUPMAP, _QCONTRAST, 2)

    def test_quarantined_primary_requires_samples(self):
        with pytest.raises(ValueError, match='requires a non-empty'):
            validate_quarantine([], 'quarantined', _QSAMPLES,
                                _QGROUPMAP, _QCONTRAST, 2)

    def test_group_below_min_after_quarantine_raises(self):
        # quarantining 2 of 3 WT leaves n=1 < min 2
        with pytest.raises(ValueError, match='after quarantine'):
            validate_quarantine(['WT-2', 'WT-3'], 'quarantined', _QSAMPLES,
                                _QGROUPMAP, _QCONTRAST, 2)

    def test_group_at_min_after_quarantine_ok(self):
        # quarantining 1 of 3 WT leaves n=2 == min 2 -> allowed
        validate_quarantine(['WT-3'], 'quarantined', _QSAMPLES,
                            _QGROUPMAP, _QCONTRAST, 2)


# ============================================================
# Section 6: write_provenance (always-written integrity record)
# ============================================================

class TestWriteProvenance:

    def test_no_quarantine_records_none(self, tmp_path):
        write_provenance('RUN', tmp_path, [], 'full',
                         _QSAMPLES, _QGROUPS, 'DEqMS', False)
        text = (tmp_path / 'RUN.analysis_provenance.txt').read_text()
        assert 'none' in text                       # "Quarantined samples: none"
        assert 'WT=3' in text and 'KO=3' in text
        assert text.count('WT=') == 1               # only the 'full' n-per-group line
        assert 'robust=FALSE' in text
        assert 'Benjamini-Hochberg (pinned explicitly)' in text

    def test_with_quarantine_records_both(self, tmp_path):
        write_provenance('RUN', tmp_path, ['WT-3'], 'quarantined',
                         _QSAMPLES, _QGROUPS, 'DEqMS', True)
        text = (tmp_path / 'RUN.analysis_provenance.txt').read_text()
        assert 'WT-3' in text
        assert 'quarantined' in text.lower()
        assert 'robust=TRUE' in text
        assert 'She & Owen 2011' in text
        # both n-per-group lines present: full WT=3, quarantined WT=2
        assert text.count('WT=') == 2
        assert 'WT=3' in text and 'WT=2' in text


# ============================================================
# Section 7: sensitivity-filename glob uniqueness (M3 guardrail)
# ============================================================

class TestSensitivityFilenameGlob:
    """The sensitivity outputs must NOT be captured by the primary emit globs,
    or Nextflow would route two files into a single-file downstream input."""

    RUN = 'HIPcyto_WT_vs_HIPcyto_KO'
    CON = 'KO_vs_WT'

    def test_results_table_globs_are_single_file(self):
        files = [
            f'{self.RUN}.diff_abundance_results.parquet',
            f'{self.RUN}.diff_abundance_results.sensitivity.parquet',
            f'{self.RUN}.diff_abundance_results.csv',
            f'{self.RUN}.diff_abundance_results.sensitivity.csv',
        ]
        pq  = [f for f in files if fnmatch.fnmatch(f, '*.diff_abundance_results.parquet')]
        csv = [f for f in files if fnmatch.fnmatch(f, '*.diff_abundance_results.csv')]
        assert pq  == [f'{self.RUN}.diff_abundance_results.parquet']
        assert csv == [f'{self.RUN}.diff_abundance_results.csv']

    def test_plot_globs_are_single_file(self):
        files = [
            f'{self.RUN}.{self.CON}.volcano_plot.png',
            f'{self.RUN}.{self.CON}.volcano_plot.sensitivity.png',
            f'{self.RUN}.{self.CON}.ma_plot.png',
            f'{self.RUN}.{self.CON}.ma_plot.sensitivity.png',
        ]
        volc = [f for f in files if fnmatch.fnmatch(f, '*.volcano_plot.png')]
        ma   = [f for f in files if fnmatch.fnmatch(f, '*.ma_plot.png')]
        assert volc == [f'{self.RUN}.{self.CON}.volcano_plot.png']
        assert ma   == [f'{self.RUN}.{self.CON}.ma_plot.png']


# ============================================================
# Section 8: parameter coercion helpers (config hardening)
# ============================================================

class TestNormalizeQuarantineSamples:

    def test_none_returns_empty(self):
        assert normalize_quarantine_samples(None) == []

    def test_empty_list_returns_empty(self):
        assert normalize_quarantine_samples([]) == []

    def test_bare_string_becomes_single_element(self):
        # a YAML scalar must NOT explode into characters
        assert normalize_quarantine_samples('HIPcyto_WT-3') == ['HIPcyto_WT-3']

    def test_list_passthrough(self):
        assert normalize_quarantine_samples(['A', 'B']) == ['A', 'B']

    def test_duplicates_removed_order_preserved(self):
        assert normalize_quarantine_samples(['B', 'A', 'B', 'A']) == ['B', 'A']

    def test_tuple_accepted(self):
        assert normalize_quarantine_samples(('A', 'B')) == ['A', 'B']

    def test_bad_type_raises(self):
        with pytest.raises(ValueError, match='must be a list'):
            normalize_quarantine_samples(42)


class TestParseBoolParam:

    def test_true_bool(self):
        assert parse_bool_param(True) is True

    def test_false_bool(self):
        assert parse_bool_param(False) is False

    def test_quoted_false_string_is_false(self):
        # the footgun: bool("false") would be True; parse_bool_param must not
        assert parse_bool_param('false') is False

    def test_quoted_true_string_is_true(self):
        assert parse_bool_param('true') is True

    def test_one_string_is_true(self):
        assert parse_bool_param('1') is True

    def test_none_uses_default(self):
        assert parse_bool_param(None, default=False) is False
        assert parse_bool_param(None, default=True) is True


# ============================================================
# Section 9: write_summary_txt dual (primary + sensitivity) branch
# ============================================================

def _summary_frame(n_up, n_dn, n=10):
    """A minimal assemble_results-shaped frame for summary rendering."""
    rows = []
    for i in range(n):
        sig = i < (n_up + n_dn)
        direction = 'ns' if not sig else ('up' if i < n_up else 'down')
        rows.append({
            'protein_id': f'P{i}', 'gene_symbol': f'G{i}',
            'log2_fc': 2.0 if direction == 'up' else (-2.0 if direction == 'down' else 0.1),
            'avg_abundance': 20.0,
            'limma_t': 1.0, 'limma_pvalue': 0.01, 'limma_adj_pvalue': 0.02,
            'deqms_t': 1.0, 'deqms_pvalue': 0.001,
            'deqms_adj_pvalue': 0.001 if sig else 0.5,
            'n_peptides': 5, 'significant': sig, 'direction': direction,
            'contrast': 'KO_vs_WT',
        })
    return pd.DataFrame(rows)


class TestWriteSummaryDual:

    def test_dual_summary_renders_primary_and_sensitivity(self, tmp_path):
        primary = [('KO_vs_WT', 'KO', 'WT', _summary_frame(3, 2))]   # 5 sig
        sens    = [('KO_vs_WT', 'KO', 'WT', _summary_frame(1, 0))]   # 1 sig
        write_summary_txt(
            primary, 'RUN', 'DEqMS', _params(['KO_vs_WT']), tmp_path,
            sensitivity_results=sens, primary_key='quarantined',
            quarantine_samples=['HIPcyto_WT-3'],
        )
        text = (tmp_path / 'RUN.diff_abundance_summary.txt').read_text()
        assert 'PRIMARY = quarantined' in text
        assert 'SENSITIVITY ANALYSIS (full)' in text
        assert 'HIPcyto_WT-3' in text
        assert '5 / 10' in text            # primary count
        assert '1 / 10' in text            # sensitivity count

    def test_single_analysis_summary_has_no_sensitivity(self, tmp_path):
        primary = [('KO_vs_WT', 'KO', 'WT', _summary_frame(3, 2))]
        write_summary_txt(
            primary, 'RUN', 'DEqMS', _params(['KO_vs_WT']), tmp_path,
        )
        text = (tmp_path / 'RUN.diff_abundance_summary.txt').read_text()
        assert 'SENSITIVITY ANALYSIS' not in text
        assert 'PRIMARY =' not in text


# ============================================================
# CONTRAST ORIENTATION (numerator/denominator made explicit)
# ============================================================

class TestContrastOrientation:
    '''
    The fitted contrast is numerator - denominator, so 'up' / positive log2 FC
    must be labelled as higher in the numerator everywhere it is reported.
    '''

    def test_split_contrast_first_vs(self):
        assert split_contrast('KO_vs_WT') == ('KO', 'WT')
        # Only the FIRST '_vs_' splits, matching parse_and_validate_contrasts.
        assert split_contrast('A_vs_B_vs_C') == ('A', 'B_vs_C')

    @pytest.mark.parametrize('bad', ['KOWT', '_vs_WT', 'KO_vs_'])
    def test_split_contrast_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            split_contrast(bad)

    def test_orientation_follows_label_not_run_order(self):
        # The reversed label must flip the orientation columns (negative control
        # for the schema test above, which only sees KO_vs_WT).
        raw = TestAssembleResults()._deqms_raw()
        out = assemble_results(raw, TestAssembleResults()._id_mapping(),
                               _params(['WT_vs_KO']), 'DEqMS', 'WT_vs_KO')
        assert (out['numerator'] == 'WT').all()
        assert (out['denominator'] == 'KO').all()

    def test_summary_states_direction(self, tmp_path):
        # Build the tuple the way main() does (via parse_and_validate_contrasts
        # -> split_contrast), so the test fails if the split is ever reversed.
        (label, num, den, _), = parse_and_validate_contrasts(
            _params(['KO_vs_WT']), ['KO', 'WT'])
        primary = [(label, num, den, _summary_frame(3, 2))]
        write_summary_txt(primary, 'WT_vs_KO_run', 'DEqMS',
                          _params(['KO_vs_WT']), tmp_path)
        text = (tmp_path / 'WT_vs_KO_run.diff_abundance_summary.txt').read_text()
        assert 'log2 FC > 0 = higher in KO than WT' in text
        assert 'Up (higher in KO):' in text
        assert 'Down (higher in WT):' in text

    def test_summary_rows_align_with_long_group_names(self, tmp_path):
        # Production-length names must neither run into the value nor shift it.
        primary = [('CTXcyto_KO_vs_CTXcyto_WT', 'CTXcyto_KO', 'CTXcyto_WT',
                    _summary_frame(3, 2))]
        write_summary_txt(primary, 'RUN', 'DEqMS',
                          _params(['CTXcyto_KO_vs_CTXcyto_WT']), tmp_path)
        lines = (tmp_path / 'RUN.diff_abundance_summary.txt').read_text().splitlines()
        down = next(l for l in lines if l.startswith('  Down (higher in CTXcyto_WT):'))
        assert 'CTXcyto_WT): ' in down                  # at least one space
        short = [('KO_vs_WT', 'KO', 'WT', _summary_frame(3, 2))]
        write_summary_txt(short, 'RUN2', 'DEqMS', _params(['KO_vs_WT']), tmp_path)
        lines2 = (tmp_path / 'RUN2.diff_abundance_summary.txt').read_text().splitlines()
        up  = next(l for l in lines2 if l.startswith('  Up (higher in KO):'))
        sig = next(l for l in lines2 if l.startswith('Significant proteins:'))
        # Short labels share the value column with the fixed-width rows (26).
        assert up[26:27].isdigit() and sig[26:27].isdigit()

    def test_plots_label_groups(self):
        df = _summary_frame(2, 1)
        vol = plot_volcano(df, 'KO_vs_WT', 'RUN', _params(['KO_vs_WT']))
        ma  = plot_ma(df, 'KO_vs_WT', 'RUN', _params(['KO_vs_WT']))
        assert 'higher in KO' in vol.layout.xaxis.title.text
        assert 'higher in KO' in ma.layout.yaxis.title.text
        # The full statement (both groups) is carried in the subtitle.
        assert 'higher in KO than WT' in vol.layout.title.text
        assert 'higher in KO than WT' in ma.layout.title.text
        names = {t.name for t in vol.data}
        assert {'Higher in KO', 'Higher in WT'} <= names



# ============================================================
# Section: BH pinned explicitly in the embedded R (handoff Piece C item 5)
# ============================================================

class TestBhPinnedInRSource:
    '''
    Static guard on the embedded R code (the fit itself needs R, see
    tests/test_differential_abundance_r.py). DEqMS::outputResult() takes no
    adjust.method, so BH must be re-applied with p.adjust on both adjusted
    columns; limma::topTable must pass adjust.method = "BH". Numerical identity
    with the package defaults was checked against DEqMS 1.28.0 on 2026-09-29.
    '''

    _SRC = (Path(__file__).resolve().parent.parent / 'bin'
            / 'differential_abundance.py').read_text()

    def test_toptable_pins_bh(self):
        assert 'adjust.method = "BH"' in self._SRC

    def test_deqms_columns_recomputed_with_bh(self):
        assert 'res$adj.P.Val    <- stats::p.adjust(res$P.Value,     method = "BH")' in self._SRC
        assert 'res$sca.adj.pval <- stats::p.adjust(res$sca.P.Value, method = "BH")' in self._SRC
