#!/usr/bin/env python3
# title: test_differential_abundance_gate.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-24
# last modified: 2026-07-24
#
# purpose:
#   Permanent empirical-gate tests for Module 04 DIFFERENTIAL_ABUNDANCE
#   (bin/differential_abundance.py). Encodes the APPROVED Piece C review
#   invariants M04-1 .. M04-4 as runnable pytest checks, each paired with a
#   negative control that proves the check has teeth (it must fail on known-bad
#   input). All four gates are pure Python (no rpy2 / embedded R), so they run
#   on a dev machine; the R statistical fit itself is covered separately by the
#   cluster integration file tests/test_differential_abundance_r.py.
#
#   Gate map:
#     M04-1  contrast sign symmetry (metamorphic)   -> TestContrastSignSymmetry
#     M04-2  peptide-count floor (invariant)        -> TestPeptideCountFloor
#     M04-3  no introduced NaN / NaN surfaced       -> TestNoIntroducedNaN
#     M04-4  operator-char group label rejected     -> TestOperatorCharGroupLabel
#
#   Two of the gates document CONFIRMED LATENT FINDINGS in current source and are
#   marked xfail(strict=True), matching the repo convention for the two existing
#   xfails in test_differential_abundance.py:
#     - M04-3 finding: a NaN primary adjusted p-value is silently classified 'ns'
#       instead of being surfaced/counted.
#     - M04-4 finding: parse_and_validate_contrasts does not reject group labels
#       containing R-operator characters (hyphen/space/+/:) which the R contrast
#       string then misparses.
#   When source is fixed these xfails will XPASS and (being strict) fail, which is
#   the intended signal to update the gate.
#
#   The existing xfail
#   test_differential_abundance.py::TestSummarizePeptideCounts::test_all_zero_row_does_not_corrupt
#   already probes the all-zero peptide-count corruption for M04-2; this file does
#   NOT duplicate it. The M04-2 gate here asserts the positive floor invariant on
#   valid input and demonstrates teeth on a hand-built degenerate value, and
#   references the existing xfail for the all-zero source behavior.
#
# inputs:
#   None (tests build inputs in-memory).
#
# outputs:
#   Test results (stdout via pytest).
#
# usage example:
#   pytest tests/test_differential_abundance_gate.py -v
#
#   copy/paste: pytest tests/test_differential_abundance_gate.py -q

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add bin/ to path. differential_abundance imports cleanly (rpy2 is loaded
# lazily inside _run_one_contrast_r); the functions exercised here never touch R.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from differential_abundance import (  # noqa: E402
    assemble_results,
    is_syntactic_r_name,
    parse_and_validate_contrasts,
    summarize_peptide_counts,
)


# ============================================================
# SHARED HELPERS
# ============================================================

def _params(fdr: float = 0.05, fc: float = 1.0) -> dict:
    '''Minimal params dict for assemble_results (only significance is read).'''
    return {
        'differential_abundance': {
            'significance': {'fdr_threshold': fdr, 'fc_threshold': fc},
        },
    }


def _deqms_raw() -> pd.DataFrame:
    '''
    An R-output-shaped DEqMS frame for the KO - WT fit (no R needed).
      P1: adj 0.01, FC +2.0  -> significant up
      P2: adj 0.001, FC -1.5 -> significant down
      P3: adj 0.20           -> ns (fails FDR)
      P4: adj 0.01, FC +0.5  -> ns (fails FC)
    '''
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


def _flip_direction(raw: pd.DataFrame) -> pd.DataFrame:
    '''
    Return the KO - WT frame as the WT - KO fit would produce it: negate every
    sign-bearing statistic (logFC, limma t, DEqMS sca.t); p-values, adjusted
    p-values, average expression, and peptide counts are direction-invariant.
    '''
    out = raw.copy()
    for col in ('logFC', 't', 'sca.t'):
        out[col] = -out[col]
    return out


def _id_mapping() -> pd.DataFrame:
    return pd.DataFrame({'protein_id': ['P1', 'P2', 'P3', 'P4'],
                         'gene_symbol': ['Gene1', 'Gene2', 'Gene3', 'Gene4']})


def _assert_sign_symmetry(out_a: pd.DataFrame, out_b: pd.DataFrame,
                          tol: float = 1e-9) -> None:
    '''
    The property under test (shared by the check and its negative control):
    for matched proteins, out_a.log2_fc == -out_b.log2_fc and the two t-columns
    flip sign too. Raises AssertionError when the relation does not hold.
    '''
    a = out_a.set_index('protein_id')
    b = out_b.set_index('protein_id')
    common = a.index.intersection(b.index)
    assert len(common) > 0, 'no overlapping proteins to compare'
    for col in ('log2_fc', 'deqms_t', 'limma_t'):
        va = a.loc[common, col].astype(float).to_numpy()
        vb = b.loc[common, col].astype(float).to_numpy()
        assert np.allclose(va, -vb, atol=tol, equal_nan=True), (
            f'{col} is not sign-symmetric between the two contrast directions'
        )


def _assert_peptide_floor(series: pd.Series) -> None:
    '''
    The property under test: every summarized peptide count is >= 1 (DEqMS's
    count-variance predictor is undefined for a zero/garbage floor). Raises
    AssertionError otherwise.
    '''
    vals = series.astype('int64')
    assert (vals >= 1).all(), (
        f'peptide-count floor violated: min={int(vals.min())} (< 1)'
    )


def _assert_no_introduced_nan(out: pd.DataFrame, cols: list[str]) -> None:
    '''
    The property under test: none of the listed output columns contains a NaN.
    Used with clean input (must pass) and with NaN-injected input (must fail).
    '''
    for col in cols:
        assert not out[col].isna().any(), f'unexpected NaN in {col}'


# ============================================================
# M04-1: Contrast sign symmetry (metamorphic, local)
# ============================================================

class TestContrastSignSymmetry:
    '''
    For the same input, WT_vs_KO log2_fc == -(KO_vs_WT log2_fc) within float
    tolerance, and deqms_t / limma_t flip sign too. assemble_results is a
    pass-through for the sign-bearing statistics, so the symmetry must survive
    schema assembly and drive the up/down direction flip.
    '''

    def test_sign_symmetry_and_direction_flip(self):
        raw_ko_wt = _deqms_raw()
        raw_wt_ko = _flip_direction(raw_ko_wt)

        out_ko = assemble_results(raw_ko_wt, _id_mapping(), _params(),
                                  'DEqMS', 'KO_vs_WT')
        out_wt = assemble_results(raw_wt_ko, _id_mapping(), _params(),
                                  'DEqMS', 'WT_vs_KO')

        # Sign-bearing statistics flip cleanly between the two directions.
        _assert_sign_symmetry(out_ko, out_wt)

        # And the significance direction flips accordingly for the spiked pair.
        ko = out_ko.set_index('protein_id')
        wt = out_wt.set_index('protein_id')
        assert ko.loc['P1', 'direction'] == 'up'
        assert wt.loc['P1', 'direction'] == 'down'
        assert ko.loc['P2', 'direction'] == 'down'
        assert wt.loc['P2', 'direction'] == 'up'

    def test_negative_control_direction_ignoring_builder_fails(self):
        '''
        NEGATIVE CONTROL (teeth): a builder that ignores contrast direction emits
        the SAME signed frame for both directions. The symmetry check must reject
        it (x is not -x for nonzero x). If this control ever passes, the symmetry
        check has no teeth.
        '''
        raw_ko_wt = _deqms_raw()
        out_ko = assemble_results(raw_ko_wt, _id_mapping(), _params(),
                                  'DEqMS', 'KO_vs_WT')
        # Stub builder ignores direction -> identical output for 'WT_vs_KO'.
        out_stub = out_ko.copy()
        out_stub['contrast'] = 'WT_vs_KO'
        with pytest.raises(AssertionError):
            _assert_sign_symmetry(out_ko, out_stub)


# ============================================================
# M04-2: Peptide-count floor (invariant, local)
# ============================================================

class TestPeptideCountFloor:
    '''
    summarize_peptide_counts must return >= 1 for every retained protein (the
    per-protein minimum of nonzero peptide counts). The all-zero-row corruption
    is already pinned by the existing xfail in test_differential_abundance.py
    (TestSummarizePeptideCounts::test_all_zero_row_does_not_corrupt); this gate
    does NOT duplicate it. Here we assert the positive floor on valid input and
    demonstrate the floor check's teeth on a hand-built degenerate value.
    '''

    def test_floor_holds_for_valid_retained_proteins(self):
        # Every protein has at least one nonzero (detected) peptide count, as
        # Module 01's detection filter guarantees for retained proteins.
        pep = pd.DataFrame(
            {'s1': [3, 0, 7], 's2': [0, 4, 2], 's3': [5, 6, 0]},
            index=pd.Index(['P1', 'P2', 'P3'], name='protein_id'),
        )
        out = summarize_peptide_counts(pep)
        _assert_peptide_floor(out)
        assert out['P1'] == 3 and out['P2'] == 4 and out['P3'] == 2

    def test_negative_control_sub_one_value_fails_floor(self):
        '''
        NEGATIVE CONTROL (teeth): the int64 garbage sentinel that an all-zero row
        would produce (np.nanmin over an all-NaN slice -> NaN -> astype(int64) ->
        -2**63) is a sub-1 value. The floor check must reject it. This is the
        exact degenerate value the existing xfail flags at the source level; here
        it proves _assert_peptide_floor has teeth without re-running the source
        path (which raises a RuntimeWarning under pytest).
        '''
        garbage = pd.Series([np.int64(-2 ** 63)],
                            index=pd.Index(['P1'], name='protein_id'),
                            name='n_peptides')
        with pytest.raises(AssertionError):
            _assert_peptide_floor(garbage)


# ============================================================
# M04-3: No introduced NaN; NaN adj p-value must be surfaced
# ============================================================

class TestNoIntroducedNaN:
    '''
    Where inputs had no NaN, assemble_results must not introduce NaN into
    log2_fc or the primary adjusted p-value (DEqMS -> deqms_adj_pvalue). A NaN
    adjusted p-value must be surfaced/counted, not silently classified 'ns'.
    '''

    _CRITICAL = ['log2_fc', 'deqms_adj_pvalue']

    def test_no_nan_introduced_on_clean_input(self):
        out = assemble_results(_deqms_raw(), _id_mapping(), _params(),
                               'DEqMS', 'KO_vs_WT')
        _assert_no_introduced_nan(out, self._CRITICAL)

    def test_negative_control_injected_nan_is_detected(self):
        '''
        NEGATIVE CONTROL (teeth): inject a NaN into the raw DEqMS adjusted p-value.
        assemble_results carries it into deqms_adj_pvalue, so the null-scan must
        detect it (fail). If this control passes, the scan is blind to NaN.
        '''
        raw = _deqms_raw()
        raw.loc[raw['protein_id'] == 'P1', 'sca.adj.pval'] = np.nan
        out = assemble_results(raw, _id_mapping(), _params(), 'DEqMS', 'KO_vs_WT')
        with pytest.raises(AssertionError):
            _assert_no_introduced_nan(out, self._CRITICAL)

    def test_nan_adj_pvalue_surfaced_not_silent_ns(self):
        # Fixed 2026-07-24: assemble_results now flags a protein whose primary
        # adjusted p-value is NaN in a dedicated boolean column
        # `pvalue_undetermined` (and logs a count), instead of letting
        # 'NaN < threshold' -> False fold it silently into 'ns' with no trace.
        # `direction` deliberately stays in {up, down, ns} (additive-column design,
        # 2026-07-24 decision), so the flag is the surfacing mechanism.
        raw = _deqms_raw()
        raw.loc[raw['protein_id'] == 'P1', 'sca.adj.pval'] = np.nan
        out = assemble_results(raw, _id_mapping(), _params(), 'DEqMS',
                               'KO_vs_WT').set_index('protein_id')
        # The NaN row is surfaced by the flag, is not counted significant, and its
        # direction remains a valid three-value enum member.
        assert bool(out.loc['P1', 'pvalue_undetermined'])
        assert not bool(out.loc['P1', 'significant'])
        assert out.loc['P1', 'direction'] == 'ns'
        # A protein with a real p-value is NOT flagged (the flag is specific).
        assert not bool(out.loc['P2', 'pvalue_undetermined'])

    def test_direction_enum_stays_three_valued(self):
        # The additive-column design keeps the documented direction contract:
        # even with a NaN-adj-p row present, no fourth value leaks out.
        raw = _deqms_raw()
        raw.loc[raw['protein_id'] == 'P1', 'sca.adj.pval'] = np.nan
        out = assemble_results(raw, _id_mapping(), _params(), 'DEqMS', 'KO_vs_WT')
        assert set(out['direction'].unique()).issubset({'up', 'down', 'ns'})


# ============================================================
# M04-4: Non-syntactic-R-name group label rejected (boundary, local)
# ============================================================

def _present_groups(genotypes: list[str]) -> list[str]:
    '''
    The samples-present group labels for a run, i.e. main()'s `unique_groups` and
    the R factor's levels vector. This is the argument parse_and_validate_contrasts
    actually takes: it validates the levels R will see, not the metadata column.
    '''
    return sorted(set(genotypes))


class TestNonSyntacticGroupLabel:
    '''
    Group labels become design-matrix column names in the R fit, and limma's
    makeContrasts() rejects any level name for which make.names(x) != x, checking
    the FULL levels vector (every group with samples, not only the contrasted
    ones). So a label that is not a syntactically valid R name (hyphen, space,
    digit-leading, punctuation, reserved word) must be rejected up front with a
    clear Python error, not left to abort the fit later with a cryptic R message.

    The 2026-07-24 fix validates ALL groups against `is_syntactic_r_name`, not
    only the two named in a contrast. The 2026-08-31 follow-up scopes that check
    to the SAMPLES-PRESENT groups (main()'s `unique_groups`), which is exactly R's
    levels vector, rather than the whole metadata column. Consequences pinned
    below: an unused-but-present group is still rejected (R would see it), while a
    metadata-only group with no samples in the run is no longer rejected (R never
    sees it), and a contrast naming such a group is now caught here in Python
    instead of dying inside makeContrasts() on an undefined variable.
    '''

    def test_negative_control_clean_label_is_accepted(self):
        '''
        NEGATIVE CONTROL / companion (teeth against over-eager rejection): clean
        R-name labels must still parse. If the guard rejected these it would be
        too aggressive.
        '''
        params = {'design': {'group_column': 'genotype',
                             'contrasts': ['KO_vs_WT']}}
        groups = _present_groups(['KO', 'KO', 'KO', 'WT', 'WT', 'WT'])
        parsed = parse_and_validate_contrasts(params, groups)
        assert parsed == [('KO_vs_WT', 'KO', 'WT', 'KO - WT')]

    @pytest.mark.parametrize('bad_group', ['WT-A', '5xFAD', 'WT[1]', 'Pool ref', 'TRUE'])
    def test_non_syntactic_contrast_group_rejected(self, bad_group):
        # A non-syntactic group named in a contrast is rejected.
        params = {'design': {'group_column': 'genotype',
                             'contrasts': [f'KO_vs_{bad_group}']}}
        groups = _present_groups(['KO', 'KO', 'KO', bad_group, bad_group, bad_group])
        with pytest.raises(ValueError):
            parse_and_validate_contrasts(params, groups)

    def test_non_syntactic_UNUSED_but_present_group_still_rejected(self):
        # A non-syntactic group that HAS samples in the run but is named in NO
        # contrast is still rejected: it gets a design-matrix column, so it is in
        # R's levels vector and makeContrasts() would reject the whole vector.
        # Pins the 2026-07-24 decision to validate all levels (correcting the
        # earlier "scope to contrast groups only" proposal), which the 2026-08-31
        # samples-present narrowing must NOT undo.
        params = {'design': {'group_column': 'genotype',
                             'contrasts': ['KO_vs_WT']}}
        groups = _present_groups(['KO', 'KO', 'WT', 'WT', 'Pool-ref', 'Pool-ref'])
        with pytest.raises(ValueError):
            parse_and_validate_contrasts(params, groups)

    def test_non_syntactic_group_absent_from_run_is_ignored(self):
        # Mirror image of the test above, and the point of the 2026-08-31 fix: a
        # non-syntactic group that exists in the metadata but contributes NO
        # samples to this run (e.g. a stripped 'Pool-ref' QC group) is not in R's
        # levels vector, so R never sees it and it must not block the fit.
        params = {'design': {'group_column': 'genotype',
                             'contrasts': ['KO_vs_WT']}}
        # 'Pool-ref' is deliberately NOT in the samples-present list.
        groups = _present_groups(['KO', 'KO', 'KO', 'WT', 'WT', 'WT'])
        parsed = parse_and_validate_contrasts(params, groups)
        assert parsed == [('KO_vs_WT', 'KO', 'WT', 'KO - WT')]

    def test_contrast_naming_group_absent_from_run_is_rejected(self):
        # The latent bug the narrowing also closes. Under the old metadata-wide
        # check, a contrast naming a syntactically fine group that has no samples
        # in the run passed validation, then died in R: that group has no
        # design-matrix column, so makeContrasts() hits an undefined variable.
        # It must now fail here, in Python, with a message naming the group.
        params = {'design': {'group_column': 'genotype',
                             'contrasts': ['KO_vs_Pool']}}
        groups = _present_groups(['KO', 'KO', 'KO', 'WT', 'WT', 'WT'])
        with pytest.raises(ValueError, match='not found'):
            parse_and_validate_contrasts(params, groups)


class TestRNameParity:
    '''
    is_syntactic_r_name must equal `make.names(x) == x` in R. The expected values
    below are R's actual behavior (make.names appends 'X'/'.' or a dot to fix
    non-syntactic names, and appends a dot to reserved words), used as a static
    oracle since embedded R is not importable locally.
    '''

    # (name, make.names(name) == name in R)
    _BATTERY = [
        ('WT', True), ('KO', True), ('HET', True),
        ('WT.2', True), ('HET_1', True), ('group1', True), ('.foo', True),
        ('WT-A', False),     # hyphen
        ('5xFAD', False),    # digit-leading
        ('3xTg', False),     # digit-leading
        ('WT[1]', False),    # bracket
        ('Pool ref', False), # space
        ('WT+KO', False),    # plus
        ('.5x', False),      # dot followed by digit
        ('TRUE', False),     # reserved word
        ('NA', False),       # reserved word
        ('if', False),       # reserved word
        ('', False),         # empty
    ]

    @pytest.mark.parametrize('name,expected', _BATTERY)
    def test_parity_with_make_names(self, name, expected):
        assert is_syntactic_r_name(name) is expected
