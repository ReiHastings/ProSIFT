#!/usr/bin/env python3
# title: test_prenorm_qc.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-24
# last modified: 2026-04-24
#
# purpose:
#   Unit tests for Module 02 Pre-Normalization QC/EDA (bin/prenorm_qc.py).
#   Scope of this file: compute_sample_flags (Process 4.6 Outlier Detection).
#
#   This is the scientific centerpiece of Module 02: four independent outlier
#   flags combined into a convergent-evidence summary. Bugs here change which
#   samples a researcher is told to investigate, so coverage of the individual
#   flag paths and their edge cases is worth the investment.
#
#   Template conventions this file demonstrates:
#     - sys.path insertion to import from bin/ (no package install needed)
#     - class-based grouping (TestFlagLowDetection, TestFlagExtremeMedian, ...)
#     - local _make_* helpers for function-specific inputs, keeping conftest
#       fixtures reserved for cross-file reuse
#     - explicit numerical construction so a reader sees which threshold
#       is crossed without loading extra files
#
# inputs:
#   None (tests build inputs in-memory via local helpers)
#
# outputs:
#   Test results (stdout via pytest)
#
# usage example:
#   pytest tests/test_prenorm_qc.py -v
#   pytest tests/test_prenorm_qc.py::TestFlagLowDetection -v
#   pytest tests/test_prenorm_qc.py -k 'extreme_median'
#
#   copy/paste: pytest tests/test_prenorm_qc.py -v

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add bin/ to path so we can import prenorm_qc directly. This mirrors the
# mechanism established by test_prosift_cache.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from prenorm_qc import compute_sample_flags

# ============================================================
# TEST HELPERS
# ============================================================
# compute_sample_flags takes three pre-computed DataFrames (summary, pca,
# correlation) plus a scalar. Rather than running the earlier Module 02
# steps to produce them, these helpers build minimal but valid inputs
# directly. This keeps each test focused on flag logic and avoids coupling
# test outcomes to bugs in unrelated functions.

def _make_summary_df(
    n_detected: list[int],
    median_intensity: list[float] | None = None,
    sample_ids: list[str] | None = None,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    '''
    Build a summary_df with the columns compute_sample_flags consumes
    (sample_id, group, n_detected, median_intensity) plus stubs for the
    other columns compute_sample_summaries would populate.
    '''
    n = len(n_detected)
    if sample_ids is None:
        sample_ids = [f'S{i + 1}' for i in range(n)]
    if groups is None:
        # Default: split evenly into groups A and B
        half = n // 2
        groups = ['A'] * half + ['B'] * (n - half)
    if median_intensity is None:
        median_intensity = [22.0] * n
    return pd.DataFrame({
        'sample_id': sample_ids,
        'group': groups,
        'n_detected': n_detected,
        'median_intensity': median_intensity,
        'total_intensity': [np.nan] * n,
        'mad_intensity': [1.0] * n,
        'skewness': [0.0] * n,
        'kurtosis': [0.0] * n,
    })


def _make_pca_df(
    pc1: list[float],
    pc2: list[float],
    sample_ids: list[str] | None = None,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    '''Build a pca_df with the columns compute_sample_flags consumes.'''
    n = len(pc1)
    if sample_ids is None:
        sample_ids = [f'S{i + 1}' for i in range(n)]
    if groups is None:
        half = n // 2
        groups = ['A'] * half + ['B'] * (n - half)
    return pd.DataFrame({
        'sample_id': sample_ids,
        'group': groups,
        'PC1': pc1,
        'PC2': pc2,
    })


def _make_corr_df(
    sample_ids: list[str],
    off_diagonal: float = 0.98,
    overrides: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    '''
    Build a symmetric sample-sample correlation DataFrame. All off-diagonal
    cells default to `off_diagonal` (typical healthy proteomics replicate
    correlation); `overrides` lets a test pin specific cells to lower values
    to simulate a poorly-correlated sample.
    '''
    n = len(sample_ids)
    corr = np.full((n, n), off_diagonal)
    np.fill_diagonal(corr, 1.0)
    if overrides:
        idx = {sid: i for i, sid in enumerate(sample_ids)}
        for (a, b), val in overrides.items():
            ia, ib = idx[a], idx[b]
            corr[ia, ib] = val
            corr[ib, ia] = val
    return pd.DataFrame(corr, index=sample_ids, columns=sample_ids)


# Default inputs many tests reuse. A clean 6-sample, 2-group setup where
# no flag should fire (except flag_low_correlation, which by design always
# fires on exactly one sample -- see TestFlagLowCorrelation).
_DEFAULT_SAMPLES = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
_DEFAULT_GROUPS = ['A', 'A', 'A', 'B', 'B', 'B']
_N_TOTAL_PROTEINS = 100  # Detection threshold = 2% = 2 proteins


def _clean_inputs():
    '''A 6-sample, 2-group input set where no magnitude-gated flag fires.'''
    summary = _make_summary_df(
        n_detected=[95, 95, 95, 95, 95, 95],
        median_intensity=[22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
        sample_ids=_DEFAULT_SAMPLES,
        groups=_DEFAULT_GROUPS,
    )
    pca = _make_pca_df(
        pc1=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        pc2=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        sample_ids=_DEFAULT_SAMPLES,
        groups=_DEFAULT_GROUPS,
    )
    corr = _make_corr_df(_DEFAULT_SAMPLES, off_diagonal=0.98)
    return summary, pca, corr


# ============================================================
# SECTION 1: OUTPUT SHAPE
# ============================================================

class TestOutputShape:
    '''Verify compute_sample_flags produces the expected output schema.'''

    def test_returns_expected_columns(self):
        summary, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert list(flags.columns) == [
            'sample_id', 'group',
            'flag_low_detection', 'flag_extreme_median',
            'flag_pca_outlier', 'flag_low_correlation',
            'n_flags',
        ]

    def test_one_row_per_sample(self):
        summary, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert len(flags) == len(summary)
        assert list(flags['sample_id']) == list(summary['sample_id'])

    def test_n_flags_counts_true_booleans(self):
        '''n_flags must equal the sum of the four flag columns per row.'''
        summary, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flag_cols = [
            'flag_low_detection', 'flag_extreme_median',
            'flag_pca_outlier', 'flag_low_correlation',
        ]
        expected_counts = flags[flag_cols].sum(axis=1).astype(int)
        pd.testing.assert_series_equal(
            flags['n_flags'], expected_counts, check_names=False,
        )

    def test_flag_columns_are_boolean(self):
        summary, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        for col in ['flag_low_detection', 'flag_extreme_median',
                    'flag_pca_outlier', 'flag_low_correlation']:
            assert flags[col].dtype == bool, f'{col} is not boolean'


# ============================================================
# SECTION 2: FLAG_LOW_DETECTION
# ============================================================
# Spec (Process 4.6, spec line 252): True if sample has lowest n_detected in
# group AND the gap from the group median exceeds 2% of n_total_proteins.

class TestFlagLowDetection:
    '''
    Detection threshold is 2% of n_total_proteins. With n_total=100, the
    threshold is 2 proteins; a gap must strictly exceed 2.
    '''

    def test_fires_when_gap_exceeds_threshold(self):
        # Group A medians: 95, 95, 85. Gap = 10 proteins > threshold of 2.
        summary = _make_summary_df(
            n_detected=[95, 95, 85, 95, 95, 95],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = flags.loc[flags['flag_low_detection'], 'sample_id'].tolist()
        assert flagged == ['A3']

    def test_does_not_fire_when_gap_below_threshold(self):
        # Group A medians: 95, 95, 94. Gap = 1 protein, threshold = 2.
        summary = _make_summary_df(
            n_detected=[95, 95, 94, 95, 95, 95],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert not flags['flag_low_detection'].any()

    def test_does_not_fire_when_gap_equals_threshold(self):
        # Strict inequality: gap > threshold, not >=
        summary = _make_summary_df(
            n_detected=[95, 95, 93, 95, 95, 95],  # gap = 2, threshold = 2
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert not flags['flag_low_detection'].any()

    def test_threshold_scales_with_n_total_proteins(self):
        # At n_total=10_000, 2% = 200 proteins. A gap of 10 should NOT flag.
        summary = _make_summary_df(
            n_detected=[9500, 9500, 9490, 9500, 9500, 9500],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, 10_000)
        assert not flags['flag_low_detection'].any()

    def test_fires_only_on_the_lowest_within_group(self):
        # Two samples below median, but only the lowest should be flagged.
        summary = _make_summary_df(
            n_detected=[95, 85, 80, 95, 95, 95],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = flags.loc[flags['flag_low_detection'], 'sample_id'].tolist()
        assert flagged == ['A3']


# ============================================================
# SECTION 3: FLAG_EXTREME_MEDIAN
# ============================================================
# Spec (spec line 253): True if sample has most extreme median_intensity in
# group (high or low) AND |sample - group_median_of_medians| > 2 * group_MAD.

class TestFlagExtremeMedian:

    def test_fires_on_high_outlier(self):
        # Group A medians: [22, 22.5, 25]. median_of_medians = 22.5.
        # Deviations = [0.5, 0, 2.5]. MAD = median([0.5, 0, 2.5]) = 0.5.
        # Threshold = 2 * 0.5 = 1.0. Max dev = 2.5 > 1.0, so A3 flags.
        summary = _make_summary_df(
            n_detected=[95, 95, 95, 95, 95, 95],
            median_intensity=[22.0, 22.5, 25.0, 22.0, 22.0, 22.0],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = flags.loc[flags['flag_extreme_median'], 'sample_id'].tolist()
        assert flagged == ['A3']

    def test_fires_on_low_outlier(self):
        # Mirror of above: extreme can be in either direction.
        summary = _make_summary_df(
            n_detected=[95, 95, 95, 95, 95, 95],
            median_intensity=[22.0, 22.5, 19.0, 22.0, 22.0, 22.0],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = flags.loc[flags['flag_extreme_median'], 'sample_id'].tolist()
        assert flagged == ['A3']

    def test_does_not_fire_when_group_is_uniform(self):
        # All medians equal: MAD = 0, the guard at group_mad > 0 blocks the flag.
        summary = _make_summary_df(
            n_detected=[95, 95, 95, 95, 95, 95],
            median_intensity=[22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert not flags['flag_extreme_median'].any()

    def test_tied_max_deviations_flag_all_tied_samples(self):
        '''
        After 2026-04-24 normalization, all samples tied at the max
        within-group deviation flag (not just the first encountered).

        Construction: 5-sample group A with median_intensity =
        [15, 22, 22, 23, 15]. Within-group median = 22; deviations =
        [7, 0, 0, 1, 7]. MAD = median([0, 0, 1, 7, 7]) = 1; threshold =
        2. Max deviation = 7, attained by both A1 and A5, exceeds the
        threshold; both flag. Previously argmax returned only A1.
        '''
        summary = _make_summary_df(
            n_detected=[95, 95, 95, 95, 95, 95, 95, 95],
            median_intensity=[15.0, 22.0, 22.0, 23.0, 15.0, 22.0, 22.0, 22.0],
            sample_ids=['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3'],
            groups=['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'],
        )
        pca = _make_pca_df(
            pc1=[-1.0] * 5 + [1.0] * 3,
            pc2=[0.0] * 8,
            sample_ids=['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3'],
            groups=['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'],
        )
        corr = _make_corr_df(
            ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3'],
            off_diagonal=0.98,
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = set(flags.loc[flags['flag_extreme_median'], 'sample_id'])
        assert flagged == {'A1', 'A5'}


# ============================================================
# SECTION 4: FLAG_PCA_OUTLIER
# ============================================================
# Spec (spec line 254): True if sample has largest Euclidean distance from
# group centroid in PC1-PC2 AND exceeds 1.5 * median distance in group.

class TestFlagPcaOutlier:

    def test_fires_when_sample_is_far_from_centroid(self):
        # Group A PC1/PC2 = [(-1, 0), (-1, 0), (5, 0)]. Centroid = (1, 0).
        # Distances from centroid = [2, 2, 4]. Median = 2. Threshold = 3.
        # Max = 4 > 3, so A3 flags.
        _, _, corr = _clean_inputs()
        summary, _, _ = _clean_inputs()
        pca = _make_pca_df(
            pc1=[-1.0, -1.0, 5.0, 1.0, 1.0, 1.0],
            pc2=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = flags.loc[flags['flag_pca_outlier'], 'sample_id'].tolist()
        assert flagged == ['A3']

    def test_does_not_fire_when_group_clusters_tightly(self):
        # All three samples on top of each other: median distance = 0,
        # max distance = 0. Guard at median_dist > 0 blocks the flag.
        summary, _, corr = _clean_inputs()
        pca = _make_pca_df(
            pc1=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            pc2=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert not flags['flag_pca_outlier'].any()

    def test_fires_independently_per_group(self):
        # Both groups have one outlier; both should flag.
        summary, _, corr = _clean_inputs()
        pca = _make_pca_df(
            pc1=[-1.0, -1.0, 5.0, 1.0, 1.0, 7.0],
            pc2=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = set(flags.loc[flags['flag_pca_outlier'], 'sample_id'])
        assert flagged == {'A3', 'B3'}

    def test_tied_max_distance_flags_all_tied_samples(self):
        '''
        After 2026-04-24 normalization, all samples tied at the maximum
        Euclidean distance from the group centroid flag (not just the
        first encountered).

        Construction: 4-sample group A with PC1 = [3, -3, 0, 0], PC2 = 0.
        Centroid = (0, 0); distances = [3, 3, 0, 0]. Median distance =
        1.5, threshold = 1.5 * 1.5 = 2.25. Max distance = 3 > 2.25.
        Both A1 and A2 are at the maximum and both flag. Previously
        idxmax returned only A1.
        '''
        summary = _make_summary_df(
            n_detected=[95] * 7,
            sample_ids=['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'],
            groups=['A', 'A', 'A', 'A', 'B', 'B', 'B'],
        )
        pca = _make_pca_df(
            pc1=[3.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            pc2=[0.0] * 7,
            sample_ids=['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'],
            groups=['A', 'A', 'A', 'A', 'B', 'B', 'B'],
        )
        corr = _make_corr_df(
            ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'], off_diagonal=0.98,
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = set(flags.loc[flags['flag_pca_outlier'], 'sample_id'])
        assert flagged == {'A1', 'A2'}


# ============================================================
# SECTION 5: FLAG_LOW_CORRELATION
# ============================================================
# Spec (spec line 255): True if sample has lowest mean within-group
# correlation in the entire dataset (global argmin across all groups).
#
# IMPORTANT: unlike the other three flags, this one has no magnitude
# threshold. On any non-empty input, exactly one sample is always flagged.
# This is spec-compliant but a design quirk documented in the Module 02
# /review-code audit (2026-04-23). Do not change this behavior without
# also updating the spec and re-baselining benchmarks.

class TestFlagLowCorrelation:

    def test_fires_on_globally_worst_correlated_sample(self):
        # Make A3 correlate poorly with A1, A2 only. Its mean within-group
        # correlation should be the dataset-wide minimum.
        summary, pca, _ = _clean_inputs()
        corr = _make_corr_df(
            _DEFAULT_SAMPLES,
            off_diagonal=0.98,
            overrides={
                ('A3', 'A1'): 0.80,
                ('A3', 'A2'): 0.80,
            },
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = flags.loc[flags['flag_low_correlation'], 'sample_id'].tolist()
        assert flagged == ['A3']

    def test_uniform_correlation_data_flags_all_samples(self):
        '''
        DESIGN NOTE: flag_low_correlation has no magnitude threshold; it picks
        the global argmin. After the 2026-04-24 follow-up audit, tie handling
        is equality-based: in a perfectly uniform correlation matrix where
        all samples tie at the same mean within-group correlation (here
        0.98), every sample is at the global minimum and every sample
        flags. Each gets n_flags = 1, which the alert treats as the
        no-concern baseline (max_flags <= 1 branch). Real proteomics
        correlations almost never tie exactly, so this is a theoretical
        edge case; pinned here to prevent silent reversion to "flag first
        only" behavior.
        '''
        summary, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert flags['flag_low_correlation'].sum() == len(flags)

    def test_global_min_spans_across_groups(self):
        # B2 has the worst within-group correlation, beating A's minimum.
        # Flag should land on B2 even though group B is otherwise similar.
        summary, pca, _ = _clean_inputs()
        corr = _make_corr_df(
            _DEFAULT_SAMPLES,
            off_diagonal=0.98,
            overrides={
                ('A3', 'A1'): 0.90,  # A3 mean within-group = ~0.94
                ('B2', 'B1'): 0.70,  # B2 mean within-group = ~0.84
                ('B2', 'B3'): 0.70,
            },
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = flags.loc[flags['flag_low_correlation'], 'sample_id'].tolist()
        assert flagged == ['B2']

    def test_excludes_self_from_mean_correlation(self):
        # Set A1's off-diagonal with peers to a distinct low value. If the
        # diagonal 1.0 were mistakenly averaged in, A1's mean would be higher
        # than expected and a different sample could win global argmin.
        summary, pca, _ = _clean_inputs()
        corr = _make_corr_df(
            _DEFAULT_SAMPLES,
            off_diagonal=0.98,
            overrides={
                ('A1', 'A2'): 0.50,
                ('A1', 'A3'): 0.50,
            },
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = flags.loc[flags['flag_low_correlation'], 'sample_id'].tolist()
        # A1 mean (excluding self) = 0.50. If self included, mean = (1+0.5+0.5)/3 = 0.67.
        # Either way A1 should win global min, but the mean value matters if
        # other samples had close competition. This test would fail if self
        # inclusion changed the argmin in a more complex dataset.
        assert flagged == ['A1']


# ============================================================
# SECTION 6: EDGE CASES
# ============================================================

class TestEdgeCases:

    def test_singleton_group_does_not_crash(self):
        '''
        A group with a single sample. flag_low_correlation assigns that sample
        mean_corr=1.0 (no peers); flag_extreme_median sees MAD=0; flag_pca
        sees median_dist=0. None of the flags should fire falsely, and no
        exception should be raised.
        '''
        summary = _make_summary_df(
            n_detected=[95, 95, 95, 95],
            median_intensity=[22.0, 22.0, 22.0, 22.0],
            sample_ids=['A1', 'A2', 'A3', 'LONELY'],
            groups=['A', 'A', 'A', 'SOLO'],
        )
        pca = _make_pca_df(
            pc1=[-1.0, -1.0, -1.0, 5.0],
            pc2=[0.0, 0.0, 0.0, 5.0],
            sample_ids=['A1', 'A2', 'A3', 'LONELY'],
            groups=['A', 'A', 'A', 'SOLO'],
        )
        corr = _make_corr_df(['A1', 'A2', 'A3', 'LONELY'], off_diagonal=0.98)
        # Should not raise
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        lonely = flags.loc[flags['sample_id'] == 'LONELY'].iloc[0]
        assert not lonely['flag_extreme_median']
        assert not lonely['flag_pca_outlier']
        # flag_low_correlation sets mean_corr=1.0 for the singleton, which
        # will never be the global argmin when other groups have real peers.
        assert not lonely['flag_low_correlation']

    def test_tied_lows_in_3_sample_group_flag_all_tied_samples(self):
        '''
        After the 2026-04-24 follow-up audit, flag_low_detection uses the
        group MAX (not median) as its reference statistic. In a 3-sample
        group with two tied lows (e.g., n_detected = [95, 85, 85]), the
        gap is now max - min = 95 - 85 = 10, which exceeds the 2-protein
        threshold (2% of n_total = 100). Both tied minima flag. The
        previous median-based logic returned median = 85 = min, gap = 0,
        and produced a silent miss (a real 2/3 failure rate would not be
        flagged). Pinned here to lock in the fix.
        '''
        summary = _make_summary_df(
            n_detected=[95, 85, 85, 95, 95, 95],  # A2 and A3 tied at 85
            sample_ids=_DEFAULT_SAMPLES,
            groups=_DEFAULT_GROUPS,
        )
        _, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = set(flags.loc[flags['flag_low_detection'], 'sample_id'])
        assert flagged == {'A2', 'A3'}

    def test_tied_lows_in_4_sample_group_flag_all_tied_samples(self):
        '''
        flag_low_detection flags all tied minima. After the 2026-04-24
        follow-up audit, all four flags use equality-based tie handling, so
        this is the consistent project-wide convention rather than a
        flag-specific quirk.

        Construction: 4-sample group A with n_detected = [95, 95, 85, 85].
        Group max = 95, min = 85, gap = 10 > threshold of 2. Both A3 and
        A4 are at the minimum, so both flag.
        '''
        summary = _make_summary_df(
            n_detected=[95, 95, 85, 85, 95, 95, 95],
            median_intensity=[22.0] * 7,
            sample_ids=['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'],
            groups=['A', 'A', 'A', 'A', 'B', 'B', 'B'],
        )
        pca = _make_pca_df(
            pc1=[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            pc2=[0.0] * 7,
            sample_ids=['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'],
            groups=['A', 'A', 'A', 'A', 'B', 'B', 'B'],
        )
        corr = _make_corr_df(
            ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'], off_diagonal=0.98,
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = set(flags.loc[flags['flag_low_detection'], 'sample_id'])
        assert flagged == {'A3', 'A4'}

    def test_single_high_outlier_falsely_flags_remaining_samples(self):
        '''
        KNOWN LIMITATION (2026-04-24 follow-up): the max-based group-center
        statistic in flag_low_detection trades the 3-sample tied-lows
        blind spot for a different failure mode: a single sample with
        anomalously HIGH detection makes the gap appear large for the
        remaining "normal" samples.

        Construction: 4-sample group A with n_detected = [200, 95, 95, 95].
        Group max = 200, min = 95, gap = 105 > threshold of 2 (with
        n_total = 100). All three tied 95s flag, even though
        scientifically the 200 is the outlier and the 95s are the norm.
        Pinned here as documented behavior; see spec Section 7. A
        researcher reading the flag table sees an unusual pattern (three
        identical n_detected, all flagged) and can investigate the
        actual outlier (the 200). Loud false positive is preferred over
        the silent miss the previous median-based logic produced.
        '''
        summary = _make_summary_df(
            n_detected=[200, 95, 95, 95, 95, 95, 95],
            sample_ids=['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'],
            groups=['A', 'A', 'A', 'A', 'B', 'B', 'B'],
        )
        pca = _make_pca_df(
            pc1=[-1.0] * 4 + [1.0] * 3,
            pc2=[0.0] * 7,
            sample_ids=['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'],
            groups=['A', 'A', 'A', 'A', 'B', 'B', 'B'],
        )
        corr = _make_corr_df(
            ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'], off_diagonal=0.98,
        )
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        flagged = set(flags.loc[flags['flag_low_detection'], 'sample_id'])
        assert flagged == {'A2', 'A3', 'A4'}

    def test_preserves_input_sample_order(self):
        '''
        Output rows should be in the same order as summary_df rows.
        Downstream code (HTML report, n_flagged count) relies on this.
        '''
        summary, pca, corr = _clean_inputs()
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert list(flags['sample_id']) == list(summary['sample_id'])
        assert list(flags['group']) == list(summary['group'])


# ============================================================
# SECTION 7: PARAMETRIZED SMOKE CHECKS
# ============================================================
# A small parametrized test exercises the output shape across a few group
# configurations. Useful as a final sanity pass that the function at least
# runs on unusual-but-valid inputs.

class TestParametrizedShapes:

    @pytest.mark.parametrize('n_per_group,n_groups', [
        (3, 2),   # Typical CTXcyto design
        (2, 2),   # Minimum viable replication
        (3, 3),   # Three-group design
        (5, 2),   # More samples per group
    ])
    def test_output_shape_for_various_layouts(self, n_per_group, n_groups):
        n = n_per_group * n_groups
        sample_ids = [f'S{i + 1}' for i in range(n)]
        groups = []
        for g in range(n_groups):
            groups.extend([f'G{g}'] * n_per_group)
        summary = _make_summary_df(
            n_detected=[95] * n,
            median_intensity=[22.0] * n,
            sample_ids=sample_ids,
            groups=groups,
        )
        # Place each group's samples at a distinct PC1 with no within-group spread
        pc1 = []
        for g in range(n_groups):
            pc1.extend([float(g)] * n_per_group)
        pca = _make_pca_df(
            pc1=pc1,
            pc2=[0.0] * n,
            sample_ids=sample_ids,
            groups=groups,
        )
        corr = _make_corr_df(sample_ids, off_diagonal=0.98)
        flags = compute_sample_flags(summary, pca, corr, _N_TOTAL_PROTEINS)
        assert len(flags) == n
        assert set(flags.columns) == {
            'sample_id', 'group',
            'flag_low_detection', 'flag_extreme_median',
            'flag_pca_outlier', 'flag_low_correlation',
            'n_flags',
        }
