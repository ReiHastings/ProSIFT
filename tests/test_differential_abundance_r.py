#!/usr/bin/env python3
# title: test_differential_abundance_r.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-09
# last modified: 2026-10-01
#
# purpose:
#   CLUSTER integration test for Module 04's R statistical fit
#   (_run_one_contrast_r: limma + DEqMS via rpy2). This exercises the real fit
#   end to end on synthetic data with known spiked proteins, checking scientific
#   correctness (fold-change direction, signal detection) rather than exact
#   numbers. TestContrastReversal (added 2026-10-01) is the end-to-end sign
#   check for the contrast-orientation work: it fits KO_vs_WT and WT_vs_KO on
#   the same data through main()'s own label -> R-string path and requires an
#   exact mirror image, anchored to the known spike direction.
#
#   WHERE THIS RUNS: only where rpy2 + an embedded R with limma and DEqMS are
#   installed (the anthill cluster). On a dev machine without a working R,
#   importing rpy2.robjects starts embedded R and SEGFAULTS -- which a normal
#   `pytest.importorskip` cannot survive because the crash is native, not a
#   Python exception. So this module PROBES importability in a SUBPROCESS (a
#   segfault there cannot take down pytest) and skips the whole module if the
#   probe fails. It is marked @pytest.mark.requires_r as well.
#
#   NOT VERIFIED ON THE DEV MACHINE. It was written to mirror main()'s exact
#   call convention (differential_abundance.main); first real execution is on
#   the cluster. If it fails there, that is the test doing its job -- inspect
#   the fit, do not silence it.
#
# inputs:
#   None (synthetic, seeded, in-memory).
#
# outputs:
#   Test results (stdout via pytest). Skips entirely without rpy2 + R + limma + DEqMS.
#
# usage example (on the cluster, inside the prosift env):
#   pytest tests/test_differential_abundance_r.py -v
#
#   copy/paste: pytest tests/test_differential_abundance_r.py -v

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ============================================================
# Module-level gate: probe R availability in a subprocess
# ============================================================

def _r_stack_available() -> bool:
    '''
    True only if rpy2 + embedded R + limma + DEqMS all import cleanly. Run in a
    SUBPROCESS so that a native crash (embedded-R segfault on a misconfigured
    machine) cannot take down the pytest process -- we just get a nonzero return
    code and skip.
    '''
    probe = (
        'import rpy2.robjects; '
        'from rpy2.robjects.packages import importr; '
        'importr("limma"); importr("DEqMS")'
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', probe],
            capture_output=True, timeout=180,
        )
        return result.returncode == 0
    except Exception:
        return False


if not _r_stack_available():
    pytest.skip(
        'rpy2 + R + limma + DEqMS not available (embedded R not importable); '
        'this is a cluster-only integration test.',
        allow_module_level=True,
    )

# Safe to import now: the probe confirmed embedded R starts without crashing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from differential_abundance import (  # noqa: E402
    _run_one_contrast_r,
    assemble_results,
    parse_and_validate_contrasts,
)

pytestmark = pytest.mark.requires_r


# ============================================================
# Synthetic dataset with known spikes
# ============================================================

_SAMPLES = ['WT-1', 'WT-2', 'WT-3', 'KO-1', 'KO-2', 'KO-3']
_GROUPS = ['WT', 'WT', 'WT', 'KO', 'KO', 'KO']
_UNIQUE_GROUPS = sorted(set(_GROUPS))          # ['KO', 'WT'], matching main()

_N_UP, _N_DOWN, _N_NULL = 10, 10, 30
_SPIKE = 2.0                                    # |log2 FC| for spiked proteins
_NOISE = 0.3                                    # within-group SD (log2)


def _up_ids():
    return [f'UP{i:02d}' for i in range(_N_UP)]


def _down_ids():
    return [f'DN{i:02d}' for i in range(_N_DOWN)]


def _null_ids():
    return [f'NULL{i:02d}' for i in range(_N_NULL)]


def _synthetic_fit_inputs():
    '''
    Build a complete (no-NaN) log2 abundance matrix with spiked proteins, plus a
    per-protein peptide-count Series, exactly as main() would hand them to
    _run_one_contrast_r. Deterministic (seeded).
      UP*   : higher in KO (log2 FC = +2)  -> logFC(KO - WT) > 0
      DN*   : lower in KO  (log2 FC = -2)  -> logFC(KO - WT) < 0
      NULL* : no genotype effect
    '''
    rng = np.random.default_rng(0)
    protein_ids = _up_ids() + _down_ids() + _null_ids()

    rows = []
    for pid in protein_ids:
        base = rng.uniform(18.0, 24.0)
        fc = _SPIKE if pid.startswith('UP') else (-_SPIKE if pid.startswith('DN') else 0.0)
        wt = base + rng.normal(0.0, _NOISE, size=3)
        ko = base + fc + rng.normal(0.0, _NOISE, size=3)
        rows.append(list(wt) + list(ko))

    abund = pd.DataFrame(
        rows, index=pd.Index(protein_ids, name='protein_id'), columns=_SAMPLES,
    )
    # Spread of peptide counts so DEqMS can fit its count-variance trend.
    pep_counts = pd.Series(
        rng.integers(2, 15, size=len(protein_ids)).astype('int64'),
        index=abund.index, name='n_peptides',
    )
    return abund, pep_counts


def _params(fdr=0.05, fc=1.0):
    return {'differential_abundance': {'significance': {'fdr_threshold': fdr,
                                                        'fc_threshold': fc}}}


def _id_mapping(protein_ids):
    return pd.DataFrame({'protein_id': protein_ids,
                         'gene_symbol': [f'G_{p}' for p in protein_ids]})


# ============================================================
# Tests
# ============================================================

class TestDEqMSFit:

    def test_deqms_runs_and_recovers_direction(self):
        '''
        DEqMS fit for KO - WT: method label, expected columns, and -- the key
        scientific check -- fold-change SIGN matches the spike for every spiked
        protein (deterministic: |FC|=2 >> noise 0.3).
        '''
        abund, pep_counts = _synthetic_fit_inputs()
        results, method = _run_one_contrast_r(
            abund, _GROUPS, _UNIQUE_GROUPS, pep_counts, 'KO - WT', use_deqms=True,
            quarantine_ids=[], robust_ebayes=False,
        )
        raw = results['full']
        assert method == 'DEqMS'
        assert len(raw) == len(abund)
        assert {'protein_id', 'logFC', 'sca.adj.pval', 'count'}.issubset(raw.columns)

        raw_idx = raw.set_index('protein_id')
        assert (raw_idx.loc[_up_ids(), 'logFC'] > 0).all()
        assert (raw_idx.loc[_down_ids(), 'logFC'] < 0).all()

    def test_deqms_separates_signal_from_null(self):
        '''Spiked proteins have much smaller DEqMS adjusted p-values than nulls.'''
        abund, pep_counts = _synthetic_fit_inputs()
        results, _ = _run_one_contrast_r(
            abund, _GROUPS, _UNIQUE_GROUPS, pep_counts, 'KO - WT', use_deqms=True,
            quarantine_ids=[], robust_ebayes=False,
        )
        raw = results['full']
        raw_idx = raw.set_index('protein_id')
        spiked = _up_ids() + _down_ids()
        spiked_med = raw_idx.loc[spiked, 'sca.adj.pval'].median()
        null_med = raw_idx.loc[_null_ids(), 'sca.adj.pval'].median()
        assert spiked_med < 0.05
        assert spiked_med < null_med

    def test_full_pipeline_calls_spiked_significant(self):
        '''
        Through assemble_results: spiked proteins are called significant with the
        correct direction; null proteins are mostly not significant. Generous
        margins (this cannot be tuned locally).
        '''
        abund, pep_counts = _synthetic_fit_inputs()
        results, method = _run_one_contrast_r(
            abund, _GROUPS, _UNIQUE_GROUPS, pep_counts, 'KO - WT', use_deqms=True,
            quarantine_ids=[], robust_ebayes=False,
        )
        raw = results['full']
        result = assemble_results(
            raw, _id_mapping(abund.index.tolist()), _params(), method, 'KO_vs_WT',
        ).set_index('protein_id')

        # Every spiked protein called significant with the right direction.
        assert (result.loc[_up_ids(), 'significant']).all()
        assert (result.loc[_up_ids(), 'direction'] == 'up').all()
        assert (result.loc[_down_ids(), 'significant']).all()
        assert (result.loc[_down_ids(), 'direction'] == 'down').all()

        # Nulls: allow a few false positives, but they must not dominate.
        n_null_sig = int(result.loc[_null_ids(), 'significant'].sum())
        assert n_null_sig <= max(2, int(0.15 * _N_NULL))


class TestLimmaOnlyFit:

    def test_limma_only_runs_and_recovers_direction(self):
        '''limma-only path (use_deqms=False, pep_counts=None): no DEqMS columns,
        correct fold-change direction.'''
        abund, _ = _synthetic_fit_inputs()
        results, method = _run_one_contrast_r(
            abund, _GROUPS, _UNIQUE_GROUPS, None, 'KO - WT', use_deqms=False,
            quarantine_ids=[], robust_ebayes=False,
        )
        raw = results['full']
        assert method == 'limma'
        assert 'sca.adj.pval' not in raw.columns          # no DEqMS output
        assert {'protein_id', 'logFC', 'adj.P.Val'}.issubset(raw.columns)

        raw_idx = raw.set_index('protein_id')
        assert (raw_idx.loc[_up_ids(), 'logFC'] > 0).all()
        assert (raw_idx.loc[_down_ids(), 'logFC'] < 0).all()


class TestQuarantineMeanShift:
    '''Mean-shift quarantine (spec Section 4.9): a discordant WT-3 is set aside
    via an indicator column; the quarantined (Q2) analysis recovers signal that
    the full (Q1) analysis dilutes.'''

    def _discordant_inputs(self):
        # Standard spiked fixture, then make WT-3 KO-like on the spiked proteins
        # so it opposes its own group (mimics the real HIP WT-3 situation).
        abund, pep_counts = _synthetic_fit_inputs()
        for pid in _up_ids() + _down_ids():
            abund.loc[pid, 'WT-3'] = abund.loc[pid, ['KO-1', 'KO-2', 'KO-3']].mean()
        return abund, pep_counts

    def test_dual_analysis_returned(self):
        abund, pep_counts = self._discordant_inputs()
        results, method = _run_one_contrast_r(
            abund, _GROUPS, _UNIQUE_GROUPS, pep_counts, 'KO - WT', use_deqms=True,
            quarantine_ids=['WT-3'], robust_ebayes=False,
        )
        assert method == 'DEqMS'
        assert set(results.keys()) == {'full', 'quarantined'}
        for key in ('full', 'quarantined'):
            assert len(results[key]) == len(abund)
            assert {'protein_id', 'logFC', 'sca.adj.pval'}.issubset(results[key].columns)

    def test_quarantined_recovers_direction_and_signal(self):
        abund, pep_counts = self._discordant_inputs()
        results, _ = _run_one_contrast_r(
            abund, _GROUPS, _UNIQUE_GROUPS, pep_counts, 'KO - WT', use_deqms=True,
            quarantine_ids=['WT-3'], robust_ebayes=False,
        )
        q = results['quarantined'].set_index('protein_id')
        f = results['full'].set_index('protein_id')
        spiked = _up_ids() + _down_ids()
        # Direction correct in the quarantined analysis.
        assert (q.loc[_up_ids(), 'logFC'] > 0).all()
        assert (q.loc[_down_ids(), 'logFC'] < 0).all()
        # Quarantining the discordant sample sharpens the spiked signal.
        assert (q.loc[spiked, 'sca.adj.pval'].median()
                <= f.loc[spiked, 'sca.adj.pval'].median())

    def test_empty_quarantine_returns_full_only(self):
        abund, pep_counts = _synthetic_fit_inputs()
        results, _ = _run_one_contrast_r(
            abund, _GROUPS, _UNIQUE_GROUPS, pep_counts, 'KO - WT', use_deqms=True,
            quarantine_ids=[], robust_ebayes=False,
        )
        assert set(results.keys()) == {'full'}


# ============================================================
# Contrast reversal (metamorphic): KO_vs_WT vs WT_vs_KO
# ============================================================

# Columns that must flip sign / stay identical when the contrast is reversed.
# Moderated variances (limma eBayes, DEqMS spectraCounteBayes) do not depend on
# the contrast's sign, so the mirror is exact up to floating-point error.
_NEGATED_RAW = {'DEqMS': ['logFC', 't', 'sca.t'], 'limma': ['logFC', 't']}
_INVARIANT_RAW = {
    'DEqMS': ['AveExpr', 'P.Value', 'adj.P.Val', 'sca.P.Value', 'sca.adj.pval'],
    'limma': ['AveExpr', 'P.Value', 'adj.P.Val'],
}
_SWAP = {'up': 'down', 'down': 'up', 'ns': 'ns'}


def _fit_via_label(label: str, use_deqms: bool):
    '''
    Fit one contrast exactly as main() does: label -> parse_and_validate_contrasts
    -> R contrast string -> _run_one_contrast_r -> assemble_results. Routing
    through the parser (not a hand-written 'KO - WT') is what lets this test see
    a label/sign mix-up.
    '''
    abund, pep_counts = _synthetic_fit_inputs()
    params = {**_params(),
              'design': {'group_column': 'genotype', 'contrasts': [label]}}
    (contrast_user, _, _, r_str), = parse_and_validate_contrasts(params, _UNIQUE_GROUPS)
    results, method = _run_one_contrast_r(
        abund, _GROUPS, _UNIQUE_GROUPS, pep_counts if use_deqms else None,
        r_str, use_deqms=use_deqms, quarantine_ids=[], robust_ebayes=False,
    )
    raw = results['full'].set_index('protein_id').sort_index()
    out = assemble_results(
        results['full'], _id_mapping(abund.index.tolist()), params, method,
        contrast_user,
    ).set_index('protein_id').sort_index()
    return method, raw, out


def _reversal_violations(method, fwd_raw, fwd_out, rev_raw, rev_out) -> list[str]:
    '''
    Every way the reversed fit fails to mirror the forward fit (empty = exact
    mirror). Pure pandas, so the negative control can feed it a non-reversed pair.
    '''
    bad = []
    if not fwd_raw.index.equals(rev_raw.index):
        return ['protein sets differ between the two fits']
    for col in _NEGATED_RAW[method]:
        if not np.allclose(rev_raw[col], -fwd_raw[col], rtol=1e-6, atol=1e-9):
            bad.append(f'{col} not negated')
    for col in _INVARIANT_RAW[method]:
        if not np.allclose(rev_raw[col], fwd_raw[col], rtol=1e-6, atol=1e-12):
            bad.append(f'{col} changed')
    if not (rev_out['significant'] == fwd_out['significant']).all():
        bad.append('significant calls changed')
    if not (rev_out['direction'] == fwd_out['direction'].map(_SWAP)).all():
        bad.append('direction not swapped up<->down')
    if not ((rev_out['numerator'] == fwd_out['denominator']).all()
            and (rev_out['denominator'] == fwd_out['numerator']).all()):
        bad.append('numerator/denominator not swapped')
    return bad


class TestContrastReversal:
    '''
    Metamorphic sign gate. Reversing the contrast label must mirror every
    signed statistic exactly and leave every unsigned one unchanged; and, so a
    globally flipped sign cannot pass by symmetry, the forward fit is anchored
    to the construction (UP* proteins are higher in KO by design).
    '''

    @pytest.mark.parametrize('use_deqms', [True, False], ids=['DEqMS', 'limma'])
    def test_reversed_contrast_is_exact_mirror(self, use_deqms):
        method, fwd_raw, fwd_out = _fit_via_label('KO_vs_WT', use_deqms)
        rev_method, rev_raw, rev_out = _fit_via_label('WT_vs_KO', use_deqms)
        assert method == rev_method == ('DEqMS' if use_deqms else 'limma')
        assert _reversal_violations(method, fwd_raw, fwd_out, rev_raw, rev_out) == []

    def test_forward_fit_anchored_to_known_direction(self):
        # Breaks the symmetry: a label/sign mix-up that flipped BOTH fits would
        # still mirror, but cannot also put UP* (higher in KO) at log2_fc > 0
        # under KO_vs_WT.
        _, _, fwd = _fit_via_label('KO_vs_WT', use_deqms=True)
        assert (fwd['numerator'] == 'KO').all() and (fwd['denominator'] == 'WT').all()
        assert (fwd.loc[_up_ids(), 'log2_fc'] > 0).all()
        assert (fwd.loc[_up_ids(), 'direction'] == 'up').all()
        assert (fwd.loc[_down_ids(), 'direction'] == 'down').all()

    def test_negative_control_unreversed_pair_is_caught(self):
        # Teeth: feeding the same (forward) fit as both halves must be reported,
        # otherwise the mirror check above could pass vacuously.
        method, fwd_raw, fwd_out = _fit_via_label('KO_vs_WT', use_deqms=True)
        bad = _reversal_violations(method, fwd_raw, fwd_out, fwd_raw, fwd_out)
        assert 'logFC not negated' in bad
        assert 'direction not swapped up<->down' in bad
        assert 'numerator/denominator not swapped' in bad
