#!/usr/bin/env python3
# title: test_filter_proteins.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-21
# last modified: 2026-07-21
#
# purpose:
#   Regression tests for Module 01 FILTER_PROTEINS (bin/filter_proteins.py),
#   Process 4.4. Pins the intended retain/remove decision for every per-group
#   detection pattern, with emphasis on the anchor gate added 2026-07-21 that
#   fixes the heavily-imputed SINGLE-GROUP presence/absence issue (the
#   '4/6-imputed' bug). Each detection pattern (a/3, b/3) at n=3/group is
#   asserted against its expected filter_status so the fix cannot silently
#   regress.
#
# inputs:
#   None (constructs detection-count frames in-memory).
#
# outputs:
#   Test results (stdout via pytest).
#
# usage example:
#   pytest tests/test_filter_proteins.py -v
#
#   copy/paste: pytest tests/test_filter_proteins.py -v

import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

# Add bin/ to path so we can import filter_proteins directly (mirrors other
# test files).
_BIN_DIR = Path(__file__).resolve().parent.parent / 'bin'
sys.path.insert(0, str(_BIN_DIR))

from filter_proteins import (  # noqa: E402  (needs bin/ on sys.path, set above)
    classify_proteins,
    count_detections_per_group,
    validate_min_present_detections,
)

FILTER_SCRIPT = _BIN_DIR / 'filter_proteins.py'


# ============================================================
# Helpers
# ============================================================

# Two groups, 3 replicates each -- the design that surfaced the bug.
GROUP_SIZES_3x3 = {'WT': 3, 'KO': 3}

# Every distinct (WT detections, KO detections) pattern for n=3/group, mapped
# to the filter_status it MUST receive under the default filter
# (min_detections_per_group=2, min_detections_present_group=None -> full
# anchor). Retained statuses: PASSED, PARTIAL, SINGLE-GROUP. Removed statuses:
# WEAK-ANCHOR, SPARSE, ABSENT.
#
# The two bug-relevant rows are marked:
#   (2, 0) -> WEAK-ANCHOR  (was SINGLE-GROUP: 4/6 imputed on 2 real values)
#   (2, 1) -> WEAK-ANCHOR  (was PARTIAL: 3/6 imputed on 3 real values)
PATTERN_EXPECTATIONS = {
    (3, 3): 'PASSED',        # 0 imputed, fully quantitative
    (3, 2): 'PASSED',        # 1 MAR, has full anchor
    (2, 3): 'PASSED',        # symmetric of (3,2)
    (2, 2): 'PASSED',        # 2 MAR, symmetric; kept (see 2026-07-21 decision)
    (3, 1): 'PARTIAL',       # 2 MNAR, anchored on the 3/3 group
    (1, 3): 'PARTIAL',       # symmetric of (3,1)
    (3, 0): 'SINGLE-GROUP',  # clean presence/absence, 3/3 anchor
    (0, 3): 'SINGLE-GROUP',  # symmetric of (3,0)
    (2, 1): 'WEAK-ANCHOR',   # BUG ROW: no full anchor -> removed
    (1, 2): 'WEAK-ANCHOR',   # symmetric of (2,1)
    (2, 0): 'WEAK-ANCHOR',   # BUG ROW: the 4/6 case -> removed
    (0, 2): 'WEAK-ANCHOR',   # symmetric of (2,0)
    (1, 1): 'SPARSE',        # no group meets min_detections
    (1, 0): 'SPARSE',        # single real value
    (0, 1): 'SPARSE',        # single real value
    (0, 0): 'ABSENT',        # nothing detected
}

RETAINED_STATUSES = {'PASSED', 'PARTIAL', 'SINGLE-GROUP'}


def _counts_frame(patterns) -> pd.DataFrame:
    '''Build a detection_counts DataFrame (index=protein_id, cols=groups).'''
    rows = {f'{wt}_{ko}': {'WT': wt, 'KO': ko} for (wt, ko) in patterns}
    return pd.DataFrame.from_dict(rows, orient='index')[['WT', 'KO']]


# ============================================================
# Section 1: per-pattern classification (default anchor = full)
# ============================================================

class TestClassifyPatternsDefault:
    '''Every n=3/group detection pattern gets its pinned filter_status.'''

    @pytest.mark.parametrize('pattern,expected', list(PATTERN_EXPECTATIONS.items()))
    def test_pattern_status(self, pattern, expected):
        counts = _counts_frame([pattern])
        status = classify_proteins(
            counts, min_detections=2, group_sizes=GROUP_SIZES_3x3,
            min_present_detections=None,
        )
        key = f'{pattern[0]}_{pattern[1]}'
        assert status.loc[key] == expected, (
            f'pattern {pattern} expected {expected}, got {status.loc[key]}'
        )

    def test_all_patterns_together(self):
        '''Classifying the full pattern set at once yields the same result
        (guards against any cross-protein leakage in vectorized logic).'''
        counts = _counts_frame(PATTERN_EXPECTATIONS.keys())
        status = classify_proteins(
            counts, min_detections=2, group_sizes=GROUP_SIZES_3x3,
            min_present_detections=None,
        )
        for (wt, ko), expected in PATTERN_EXPECTATIONS.items():
            assert status.loc[f'{wt}_{ko}'] == expected


# ============================================================
# Section 2: the bug is fixed -- heavily-imputed proteins removed
# ============================================================

class TestBugFix:
    '''The 4/6-imputed SINGLE-GROUP proteins no longer reach downstream.'''

    def test_2v0_removed(self):
        '''(2,0): 2/3 in one group, 0/3 in the other. Under the old filter
        this was SINGLE-GROUP (retained) and got 4/6 values imputed on 2 real
        measurements. It must now be WEAK-ANCHOR (removed).'''
        counts = _counts_frame([(2, 0)])
        status = classify_proteins(
            counts, 2, GROUP_SIZES_3x3, None,
        )
        assert status.iloc[0] == 'WEAK-ANCHOR'
        assert status.iloc[0] not in RETAINED_STATUSES

    def test_3v0_still_retained(self):
        '''(3,0): clean presence/absence with a fully-detected anchor group.
        Must remain retained as SINGLE-GROUP.'''
        counts = _counts_frame([(3, 0)])
        status = classify_proteins(counts, 2, GROUP_SIZES_3x3, None)
        assert status.iloc[0] == 'SINGLE-GROUP'
        assert status.iloc[0] in RETAINED_STATUSES

    def test_2v2_still_retained(self):
        '''(2,2): symmetric MAR, no MNAR imputation. The 2026-07-21 decision
        keeps it as PASSED (controlled by min_detections_per_group, not the
        anchor gate).'''
        counts = _counts_frame([(2, 2)])
        status = classify_proteins(counts, 2, GROUP_SIZES_3x3, None)
        assert status.iloc[0] == 'PASSED'


# ============================================================
# Section 3: anchor parameter behavior
# ============================================================

class TestAnchorParameter:
    '''min_detections_present_group tunes the anchor threshold.'''

    def test_explicit_anchor_relaxes_to_two(self):
        '''Setting min_detections_present_group=2 restores the old lenient
        behavior: (2,0) becomes SINGLE-GROUP and (2,1) becomes PARTIAL again,
        because a 2-detection group now counts as an anchor.'''
        counts = _counts_frame([(2, 0), (2, 1)])
        status = classify_proteins(
            counts, min_detections=2, group_sizes=GROUP_SIZES_3x3,
            min_present_detections=2,
        )
        assert status.loc['2_0'] == 'SINGLE-GROUP'
        assert status.loc['2_1'] == 'PARTIAL'

    def test_default_none_requires_full_detection(self):
        '''Default (None) requires the anchor group to be fully detected, so
        (2,0)/(2,1) are removed even though one group meets min_detections.'''
        counts = _counts_frame([(2, 0), (2, 1)])
        status = classify_proteins(
            counts, 2, GROUP_SIZES_3x3, min_present_detections=None,
        )
        assert status.loc['2_0'] == 'WEAK-ANCHOR'
        assert status.loc['2_1'] == 'WEAK-ANCHOR'


# ============================================================
# Section 4: generalization to other replicate counts
# ============================================================

class TestGeneralization:
    '''The full-detection anchor scales with group size (not hardcoded 3).'''

    def test_n5_full_anchor(self):
        '''n=5/group. Default anchor = 5/5. A (4,0) protein passes
        min_detections but has no fully-detected group -> WEAK-ANCHOR. A (5,0)
        protein has a full anchor -> SINGLE-GROUP.'''
        sizes = {'WT': 5, 'KO': 5}
        counts = _counts_frame([(5, 0), (4, 0)])
        status = classify_proteins(
            counts, min_detections=3, group_sizes=sizes,
            min_present_detections=None,
        )
        assert status.loc['5_0'] == 'SINGLE-GROUP'
        assert status.loc['4_0'] == 'WEAK-ANCHOR'

    def test_unequal_group_sizes(self):
        '''Groups may differ in size; each group's anchor threshold is its own
        replicate count. WT has 4 samples, KO has 3. (4,0): WT fully detected
        -> SINGLE-GROUP retained.'''
        sizes = {'WT': 4, 'KO': 3}
        counts = _counts_frame([(4, 0), (3, 0)])
        status = classify_proteins(
            counts, min_detections=2, group_sizes=sizes,
            min_present_detections=None,
        )
        # WT=4 is a full anchor for WT (size 4)
        assert status.loc['4_0'] == 'SINGLE-GROUP'
        # WT=3 is NOT full for a size-4 group, KO=0 -> no anchor -> removed
        assert status.loc['3_0'] == 'WEAK-ANCHOR'


# ============================================================
# Section 5: count_detections_per_group returns group sizes
# ============================================================

class TestCountDetections:
    '''count_detections_per_group reports per-group max detections.'''

    def test_group_sizes_match_columns(self):
        matrix = pd.DataFrame({
            'protein_id': ['P1', 'P2'],
            'WT-1': [1.0, None],
            'WT-2': [2.0, None],
            'WT-3': [3.0, 1.0],
            'KO-1': [1.0, None],
            'KO-2': [None, None],
            'KO-3': [None, None],
        })
        metadata = pd.DataFrame({
            'sample_id': ['WT-1', 'WT-2', 'WT-3', 'KO-1', 'KO-2', 'KO-3'],
            'group': ['WT', 'WT', 'WT', 'KO', 'KO', 'KO'],
        })
        abund_cols = ['WT-1', 'WT-2', 'WT-3', 'KO-1', 'KO-2', 'KO-3']
        counts, sizes = count_detections_per_group(
            matrix, metadata, id_col='protein_id', abund_cols=abund_cols,
            abund_prefix='', group_col='group',
        )
        assert sizes == {'WT': 3, 'KO': 3}
        # P1: WT 3/3, KO 1/3 ; P2: WT 1/3, KO 0/3
        assert counts.loc['P1', 'WT'] == 3
        assert counts.loc['P1', 'KO'] == 1
        assert counts.loc['P2', 'WT'] == 1
        assert counts.loc['P2', 'KO'] == 0

        # And the classification of those two proteins under the default filter:
        status = classify_proteins(counts, 2, sizes, None)
        # P1 (3,1): PARTIAL (full anchor in WT)
        assert status.loc['P1'] == 'PARTIAL'
        # P2 (1,0): SPARSE (no group meets min_detections)
        assert status.loc['P2'] == 'SPARSE'


# ============================================================
# Section 6: end-to-end CLI (real bin/filter_proteins.py run)
# ============================================================

class TestEndToEndCLI:
    '''Run the actual script and confirm the anchor gate reaches the outputs.

    The unit tests above pin classify_proteins in isolation; this exercises
    main() end-to-end (Parquet in -> filtered matrix + filter table + report
    out) so the fix is guarded through the real CLI, not just the function.
    '''

    _SAMPLES: ClassVar[list] = ['WT-1', 'WT-2', 'WT-3', 'KO-1', 'KO-2', 'KO-3']
    _GROUPS: ClassVar[list] = ['WT', 'WT', 'WT', 'KO', 'KO', 'KO']

    def _row(self, n_wt, n_ko, rng):
        '''One abundance row detected in n_wt WT reps and n_ko KO reps.'''
        vals = [np.nan] * 6
        for i in rng.choice([0, 1, 2], n_wt, replace=False):
            vals[i] = float(rng.normal(22, 1))
        for i in rng.choice([3, 4, 5], n_ko, replace=False):
            vals[i] = float(rng.normal(22, 1))
        return vals

    def _run(self, tmp_path, present_param=None):
        '''Build a synthetic run and invoke bin/filter_proteins.py; assert it
        succeeds and return (filter table, retained protein IDs).'''
        proc = self._invoke(tmp_path, present_param)
        assert proc.returncode == 0, f'filter_proteins failed:\n{proc.stderr}'
        table = pd.read_csv(tmp_path / 'run.detection_filter_table.csv') \
            .set_index('protein_id')
        filtered = pd.read_parquet(tmp_path / 'run.filtered_matrix.parquet')
        return table, set(filtered['protein_id'])

    def _invoke(self, tmp_path, present_param=None):
        '''Build inputs and run the CLI; return the completed process (no
        success assertion) so failure paths can be tested too.'''
        import yaml
        rng = np.random.default_rng(7)

        # 55 clean PASSED proteins to clear the >=50-retained hard stop, plus
        # one protein of each diagnostic pattern under named IDs.
        ids, data = [], {s: [] for s in self._SAMPLES}
        for i in range(55):
            ids.append(f'PASS_{i:03d}')
            for si, s in enumerate(self._SAMPLES):
                data[s].append(self._row(3, 3, rng)[si])
        named = {'WEAK_20': (2, 0), 'WEAK_21': (2, 1),
                 'SINGLE_30': (3, 0), 'PASS_22': (2, 2), 'PARTIAL_31': (3, 1)}
        for pid, (a, b) in named.items():
            ids.append(pid)
            row = self._row(a, b, rng)
            for si, s in enumerate(self._SAMPLES):
                data[s].append(row[si])

        matrix = pd.DataFrame({'protein_id': ids, **data})
        metadata = pd.DataFrame({'sample_id': self._SAMPLES, 'group': self._GROUPS})

        mpath = tmp_path / 'run.validated_matrix.parquet'
        mdpath = tmp_path / 'run.validated_metadata.parquet'
        matrix.to_parquet(mpath, index=False)
        metadata.to_parquet(mdpath, index=False)

        params = {
            'input': {'protein_id_column': 'protein_id', 'abundance_prefix': ''},
            'design': {'group_column': 'group'},
            'qc': {'min_detections_per_group': 2,
                   'min_detections_present_group': present_param},
        }
        ppath = tmp_path / 'run_params.yml'
        ppath.write_text(yaml.safe_dump(params))

        return subprocess.run(
            [sys.executable, str(FILTER_SCRIPT),
             '--matrix', str(mpath), '--metadata', str(mdpath),
             '--params', str(ppath), '--run-id', 'run',
             '--outdir', str(tmp_path)],
            capture_output=True, text=True,
        )

    def test_weak_anchor_dropped_from_matrix(self, tmp_path):
        '''The bug rows (2,0) and (2,1) are labeled WEAK-ANCHOR and do NOT
        appear in the filtered matrix that flows downstream.'''
        table, retained_ids = self._run(tmp_path)
        assert table.loc['WEAK_20', 'filter_status'] == 'WEAK-ANCHOR'
        assert table.loc['WEAK_21', 'filter_status'] == 'WEAK-ANCHOR'
        assert 'WEAK_20' not in retained_ids
        assert 'WEAK_21' not in retained_ids

    def test_clean_cases_retained_in_matrix(self, tmp_path):
        '''Clean presence/absence (3,0), symmetric MAR (2,2), and anchored
        partial (3,1) survive to the filtered matrix.'''
        table, retained_ids = self._run(tmp_path)
        assert table.loc['SINGLE_30', 'filter_status'] == 'SINGLE-GROUP'
        assert table.loc['PASS_22', 'filter_status'] == 'PASSED'
        assert table.loc['PARTIAL_31', 'filter_status'] == 'PARTIAL'
        assert {'SINGLE_30', 'PASS_22', 'PARTIAL_31'} <= retained_ids

    def test_out_of_range_anchor_param_fails_loudly(self, tmp_path):
        '''An anchor threshold above the group size (4 with n=3/group) makes
        the CLI exit non-zero with a clear message, instead of silently
        removing every presence/absence protein.'''
        proc = self._invoke(tmp_path, present_param=4)
        assert proc.returncode != 0
        assert 'min_detections_present_group' in proc.stderr

    def test_explicit_valid_anchor_param_runs(self, tmp_path):
        '''An explicit in-range anchor (2) is accepted and, being more lenient
        than full detection, retains the 2/3-vs-0/3 protein as SINGLE-GROUP.'''
        table, retained_ids = self._run(tmp_path, present_param=2)
        assert table.loc['WEAK_20', 'filter_status'] == 'SINGLE-GROUP'
        assert 'WEAK_20' in retained_ids


# ============================================================
# Section 7: anchor-parameter validation
# ============================================================

class TestValidateMinPresentDetections:
    '''validate_min_present_detections rejects footgun values loudly.

    Guards the code-review finding that an out-of-range or wrong-typed
    qc.min_detections_present_group silently corrupts filtering (removes all
    presence/absence proteins, or disables the gate).
    '''

    SIZES: ClassVar[dict] = {'WT': 3, 'KO': 3}

    def test_none_is_valid(self):
        '''None (default) means "require full detection" -- always valid.'''
        assert validate_min_present_detections(None, self.SIZES) is None

    def test_in_range_integer_is_valid(self):
        for v in (1, 2, 3):
            assert validate_min_present_detections(v, self.SIZES) is None

    def test_zero_and_negative_rejected(self):
        '''0 or negative would make every group an anchor (gate disabled).'''
        assert validate_min_present_detections(0, self.SIZES) is not None
        assert validate_min_present_detections(-1, self.SIZES) is not None

    def test_larger_than_every_group_rejected(self):
        '''A threshold above the largest group can never be met -> all
        presence/absence proteins would silently vanish.'''
        assert validate_min_present_detections(4, self.SIZES) is not None

    def test_unequal_groups_bound_is_the_max(self):
        '''With unequal groups the bound is the LARGER group; a value that only
        the larger group can meet is still valid (that group can anchor).'''
        sizes = {'WT': 4, 'KO': 3}
        assert validate_min_present_detections(4, sizes) is None   # WT can hit 4
        assert validate_min_present_detections(5, sizes) is not None

    def test_non_integer_types_rejected(self):
        '''Floats, strings, and bool must be rejected (bool is an int
        subclass, so it is checked explicitly).'''
        for bad in (2.5, '2', True, False, [2]):
            assert validate_min_present_detections(bad, self.SIZES) is not None


# ============================================================
# Section 8: more than two groups
# ============================================================

class TestMultiGroup:
    '''The anchor gate generalizes beyond two groups (the pipeline supports
    >= 2 groups). Per-group anchor thresholds are independent.'''

    SIZES_3: ClassVar[dict] = {'A': 3, 'B': 3, 'C': 3}

    def _counts(self, a, b, c):
        return pd.DataFrame({'A': [a], 'B': [b], 'C': [c]}, index=['prot'])

    def test_full_anchor_one_group_retains_single_group(self):
        '''Detected fully in exactly one group, zero elsewhere -> SINGLE-GROUP
        (has a full anchor).'''
        status = classify_proteins(
            self._counts(3, 0, 0), min_detections=2,
            group_sizes=self.SIZES_3, min_present_detections=None,
        )
        assert status.loc['prot'] == 'SINGLE-GROUP'

    def test_partial_with_full_anchor_retained(self):
        '''Full anchor in A, sub-threshold non-zero in B, zero in C ->
        PARTIAL (retained).'''
        status = classify_proteins(
            self._counts(3, 1, 0), min_detections=2,
            group_sizes=self.SIZES_3, min_present_detections=None,
        )
        assert status.loc['prot'] == 'PARTIAL'

    def test_single_group_without_full_anchor_is_weak(self):
        '''Meets min_detections in exactly one group (2/3) but that group is
        not fully detected, zero elsewhere -> WEAK-ANCHOR (removed).'''
        status = classify_proteins(
            self._counts(2, 0, 0), min_detections=2,
            group_sizes=self.SIZES_3, min_present_detections=None,
        )
        assert status.loc['prot'] == 'WEAK-ANCHOR'

    def test_two_of_three_groups_full_is_passed(self):
        '''Fully detected in two of three groups and meeting threshold in the
        third stays PASSED; the anchor gate does not touch it.'''
        status = classify_proteins(
            self._counts(3, 3, 2), min_detections=2,
            group_sizes=self.SIZES_3, min_present_detections=None,
        )
        assert status.loc['prot'] == 'PASSED'
