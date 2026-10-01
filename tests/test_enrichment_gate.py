#!/usr/bin/env python3
# title: test_enrichment_gate.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-24
# last modified: 2026-09-30
#
# purpose:
#   Permanent empirical-gate tests for Module 05 ENRICHMENT (bin/enrichment.py).
#   Encodes the APPROVED Piece C review invariants M05-1 .. M05-5, plus gates
#   M05-6 .. M05-11 added 2026-09-29 (GSEA gene_set_size, KNOWN_ISSUES E-9, and
#   its follow-ups), M05-12 added 2026-09-30 (gene_set_version per row,
#   paired with a negative control), and M05-13 added 2026-10-01 (the shipped
#   rrvgo R block run through Rscript), as runnable pytest checks. M05-4 was
#   rewritten 2026-10-01 for the cluster_status contract.
#
#   Teeth: M05-1 .. M05-5 pair each check with a negative control (a known-bad
#   input the check must reject). M05-11 pairs its size-range and sort checks
#   with negative controls; its exact-size and inclusive-bounds checks have none
#   and their teeth were shown against the pre-change code and by mutation.
#   M05-6's teeth were shown by running it against the pre-fix code. M05-7
#   carries a second contract check, and M05-8, M05-9 and M05-10 carry fixture
#   checks, named as such; their teeth were shown against the pre-change code
#   (M05-7, M05-8, M05-9's collision test, M05-10) or by mutation (M05-9's
#   case-variant test, which is green on the pre-change code because case does
#   not change string length).
#
#   Scope split: this file pins empirical properties of real gseapy output and
#   of the pipeline run through main(). Error branches and stubbed-gseapy
#   behaviour live in tests/test_enrichment.py instead:
#     - TestCountMatchedGenes       : every count_matched_genes branch (str/list
#                                     blank rule, empty set, type, Tag % absent,
#                                     unparseable and mismatched)
#     - TestParseOraOverlap         : strict ORA 'Overlap' parsing
#     - TestRunOraSizeFilter        : ORA size filter (incl. bounds), BH family,
#                                     sorted overlap genes
#     - TestRunGseaResultsStructure : run_gsea results-structure raises and Tag %
#                                     forwarding
#     - TestGseapyExceptionHandling : narrowed gseapy exception handling and
#                                     short ranked lists
#     - TestLoadParamsValidation    : gsea_permutations >= 1, min/max size
#
#   Live dependencies: this file requires gseapy to be installed. It imports
#   enrichment at module level and enrichment imports gseapy at the top, so
#   without gseapy collection fails for the whole file (the per-test
#   importorskip calls never run). gseapy is a pinned hard dependency
#   (environment.yml). M05-2 and M05-6 .. M05-11 call gseapy; M05-1, M05-3 and
#   M05-5 do not, but still need it importable. Only M05-13 needs R (Rscript
#   with rrvgo, GO.db, org.Mm.eg.db; marked slow, skipped without Rscript).
#   M05-4 and M05-11 block rpy2 so rrvgo takes its pure-Python degradation path
#   (same monkeypatch trick as test_enrichment.py's degradation test).
#
#   Gate map:
#     M05-1  ORA background identity (invariant)          -> TestOraBackgroundIdentity
#     M05-2  GSEA NES sign (metamorphic, needs gseapy)    -> TestGseaNesSign
#     M05-3  in_significant_set decoupling (neg-control)  -> TestInSignificantSetDecoupling
#     M05-4  rrvgo cluster_status contract (invariant)    -> TestClusterContract
#     M05-5  empty-run writes four outputs (boundary)     -> TestEmptyRunOutputs
#     M05-6  GSEA gene_set_size known answer (needs gseapy) -> TestGseaGeneSetSize
#     M05-7  ORA size filter + BH family (needs gseapy)   -> TestOraSizeFilter
#     M05-8  permutation_num=0 boundary (needs gseapy)    -> TestGseaNoPermutationBoundary
#     M05-9  GSEA symbol-case metamorphic (needs gseapy)  -> TestGseaSymbolCase
#     M05-10 ORA overlap_genes order stable (needs gseapy) -> TestOraOverlapGenesDeterministic
#     M05-11 main() size invariant + stable row order    -> TestMainSizeInvariant
#     M05-13 shipped rrvgo R block via Rscript (slow, R) -> TestClusterLiveRBlock
#
# inputs:
#   None (tests build inputs in-memory / in tmp_path).
#
# outputs:
#   Test results (stdout via pytest).
#
# usage example:
#   pytest tests/test_enrichment_gate.py -v
#
#   copy/paste: pytest tests/test_enrichment_gate.py -v

import sys
import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import enrichment
from enrichment import (
    CLUSTER_STATUSES,
    _annotate_cluster_unit,
    build_protein_term_mapping,
    cluster_go_terms,
    prepare_gene_symbols,
    run_ora,
)

# ============================================================
# M05-1: ORA background identity (invariant, local)
# ============================================================

def _assert_background_is_detected_proteome(
    background: object, da_df: pd.DataFrame, contrast: str,
) -> None:
    '''
    The property under test: the ORA background is exactly the set of unique
    measured gene symbols for the contrast (the detected proteome), never None
    and never a genome/default set. Raises AssertionError otherwise.
    '''
    assert background is not None, 'background must not be omitted (None -> gseapy default)'
    expected = set(
        da_df[da_df['contrast'] == contrast]['gene_symbol'].dropna().unique()
    )
    assert len(background) > 0, 'background must be a concrete non-empty list'
    assert set(background) == expected, (
        'background is not the detected proteome for this contrast'
    )


class TestOraBackgroundIdentity:
    '''
    main() assembles background_genes as
    prepare_gene_symbols(da_df, contrast)[0]['gene_symbol'].tolist() and forwards
    it to gseapy.enrich(background=...). The background must be the detected
    proteome (unique measured gene symbols), never the genome default.
    '''

    def _da(self) -> pd.DataFrame:
        return pd.DataFrame({
            'protein_id':       ['P1', 'P2', 'P3', 'P4', 'P5'],
            'gene_symbol':      ['GeneA', 'GeneA', 'GeneB', 'GeneC', None],
            'contrast':         ['KO_vs_WT'] * 4 + ['X_vs_Y'],
            'significant':      [True, False, False, False, False],
            'deqms_adj_pvalue': [0.01, 0.20, 0.30, 0.40, 0.05],
            'limma_adj_pvalue': [0.02, 0.30, 0.40, 0.50, 0.06],
        })

    def test_background_is_detected_proteome(self):
        da = self._da()
        contrast_df, _ = prepare_gene_symbols(da, 'KO_vs_WT')
        background_genes = contrast_df['gene_symbol'].tolist()
        # Detected proteome for KO_vs_WT is {GeneA, GeneB, GeneC}; the null-symbol
        # protein and the X_vs_Y contrast are excluded.
        _assert_background_is_detected_proteome(background_genes, da, 'KO_vs_WT')
        assert set(background_genes) == {'GeneA', 'GeneB', 'GeneC'}

    def test_run_ora_forwards_background_to_gseapy(self, monkeypatch):
        '''
        Strong local check: spy on gseapy.enrich and assert run_ora forwards the
        assembled background (never None). Returns an empty res2d so run_ora exits
        after the call without needing real gene sets.
        '''
        captured: dict = {}

        class _FakeResult:
            res2d = pd.DataFrame()

        def _fake_enrich(gene_list, gene_sets, background, **kwargs):
            captured['background'] = background
            return _FakeResult()

        monkeypatch.setattr(enrichment.gseapy, 'enrich', _fake_enrich)

        da = self._da()
        contrast_df, _ = prepare_gene_symbols(da, 'KO_vs_WT')
        background_genes = contrast_df['gene_symbol'].tolist()
        sig_genes = contrast_df[contrast_df['significant'].eq(True)]['gene_symbol'].tolist()
        params = {'enrichment': {'fdr_threshold': 0.05}}

        run_ora(
            sig_genes=sig_genes,
            background_genes=background_genes,
            gmt_path='unused.gmt',
            library_name='GO_BP',
            contrast='KO_vs_WT',
            params=params,
        )
        assert captured['background'] is not None
        assert set(captured['background']) == {'GeneA', 'GeneB', 'GeneC'}

    def test_negative_control_omitted_background_is_caught(self):
        '''
        NEGATIVE CONTROL (teeth): a variant that omits the background (passes None,
        which sends gseapy to its genome/default) must be rejected by the identity
        check. A genome-like set that is not the detected proteome must also fail.
        '''
        da = self._da()
        with pytest.raises(AssertionError):
            _assert_background_is_detected_proteome(None, da, 'KO_vs_WT')
        genome_like = [f'Gene{i}' for i in range(20000)]
        with pytest.raises(AssertionError):
            _assert_background_is_detected_proteome(genome_like, da, 'KO_vs_WT')


# ============================================================
# M05-2: GSEA NES sign (metamorphic, needs gseapy)
# ============================================================

class TestGseaNesSign:
    '''
    A gene set built from proteins with large positive deqms_t yields NES > 0
    with ascending=False (the value main() uses); flipping the sign of all
    ranking scores flips the NES sign. The negative control shows ascending=True
    inverts the NES sign, proving the ascending=False choice is load-bearing.
    Runs gseapy.prerank; skipped where gseapy is unavailable (cluster-deferred).
    '''

    _N_GENES = 60
    _SET_SIZE = 20

    def _ranked(self, ascending_signal: bool = False) -> pd.Series:
        # Strictly monotone ranking metric; the top _SET_SIZE genes form the set.
        genes = [f'G{i:02d}' for i in range(self._N_GENES)]
        scores = np.linspace(5.0, -5.0, self._N_GENES)
        rnk = pd.Series(scores, index=genes)
        return -rnk if ascending_signal else rnk

    def _gmt(self, tmp_path: Path) -> str:
        # Gene set = the top-ranked (largest positive t) genes.
        top = [f'G{i:02d}' for i in range(self._SET_SIZE)]
        gmt = tmp_path / 'lib.gmt'
        gmt.write_text('TERM_UP\tdescription\t' + '\t'.join(top) + '\n')
        return str(gmt)

    def _params(self) -> dict:
        return {'enrichment': {
            'fdr_threshold': 0.05,
            'min_gene_set_size': 5,
            'max_gene_set_size': 100,
            'gsea_permutations': 100,
            'gsea_seed': 42,
        }}

    def _nes(self, gsea_df: pd.DataFrame) -> float:
        row = gsea_df.set_index('term_id').loc['TERM_UP']
        return float(row['enrichment_score'])

    def test_nes_sign_tracks_ranking_sign(self, tmp_path):
        pytest.importorskip('gseapy')
        from enrichment import run_gsea

        gmt_path = self._gmt(tmp_path)
        params = self._params()

        with warnings.catch_warnings():
            # gseapy emits assorted UserWarnings/RuntimeWarnings; the project's
            # pytest config escalates those to errors. They are not the subject of
            # this gate, so silence them locally around the third-party call only.
            warnings.simplefilter('ignore')
            df_pos, _ = run_gsea(self._ranked(ascending_signal=False),
                                 gmt_path, 'GO_BP', 'KO_vs_WT', params)
            df_neg, _ = run_gsea(self._ranked(ascending_signal=True),
                                 gmt_path, 'GO_BP', 'KO_vs_WT', params)

        assert not df_pos.empty and not df_neg.empty
        nes_pos = self._nes(df_pos)
        nes_neg = self._nes(df_neg)
        assert nes_pos > 0, f'expected NES > 0 for a top-ranked set, got {nes_pos}'
        assert nes_neg < 0, f'expected NES < 0 after sign flip, got {nes_neg}'

    def test_negative_control_ascending_true_inverts(self, tmp_path):
        '''
        NEGATIVE CONTROL (teeth): with the SAME ranking, gseapy.prerank called
        with ascending=True (the wrong orientation) inverts the NES sign relative
        to ascending=False. This proves the ascending=False choice determines the
        NES sign; if the two agreed, the orientation would carry no meaning.
        '''
        gseapy = pytest.importorskip('gseapy')
        gmt_path = self._gmt(tmp_path)
        rnk = self._ranked(ascending_signal=False)

        def _prerank(ascending: bool):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return gseapy.prerank(
                    rnk=rnk, gene_sets=gmt_path, min_size=5, max_size=100,
                    permutation_num=100, ascending=ascending, no_plot=True,
                    verbose=False, seed=42, threads=1,
                )

        nes_false = float(_prerank(False).res2d.set_index('Term').loc['TERM_UP', 'NES'])
        nes_true = float(_prerank(True).res2d.set_index('Term').loc['TERM_UP', 'NES'])
        assert np.sign(nes_false) != np.sign(nes_true), (
            f'ascending flag did not invert NES sign: {nes_false} vs {nes_true}'
        )


# ============================================================
# M05-6: GSEA gene_set_size is a gene count (known answer, needs gseapy)
# ============================================================

class TestGseaGeneSetSize:
    '''
    GSEA gene_set_size must be the number of the term's genes present in the
    ranked list (spec 05 Section 4.7), not the character length of gseapy's
    ';'-joined matched_genes string (KNOWN_ISSUES E-9).

    Symbols are mixed-case and multi-character (mouse-style), matching the real
    input shape, so a character count can never coincide with the gene count.
    Expected sizes are recomputed here from the GMT file and the ranked Series
    (set intersection), independently of gseapy.

    Teeth: the known-answer test was run against the pre-fix run_gsea
    (2026-09-29) and failed with gene_set_size == 69 instead of 10.
    Runs gseapy.prerank; skipped where gseapy is unavailable.
    '''

    _N_GENES = 60
    _MIN_SIZE = 5
    _MAX_SIZE = 100

    # term -> member symbols. 'Absnt*' symbols are not in the ranked list.
    _SETS: ClassVar[dict] = {
        # 12 genes in the GMT, 10 in the ranked list -> expected size 10
        'TERM_PARTIAL': [f'Prot{i:02d}' for i in range(10)] + ['Absnt01', 'Absnt02'],
        # 20 genes, all ranked -> 20
        'TERM_FULL': [f'Prot{i:02d}' for i in range(20, 40)],
        # 8 genes, 6 ranked -> 6
        'TERM_SMALL': [f'Prot{i:02d}' for i in range(50, 56)] + ['Absnt03', 'Absnt04'],
        # 6 genes in the GMT but only 3 ranked -> below min_size after
        # intersection, so gseapy must exclude it
        'TERM_BELOW_MIN': ['Prot45', 'Prot46', 'Prot47', 'Absnt05', 'Absnt06', 'Absnt07'],
    }

    def _ranked(self) -> pd.Series:
        genes = [f'Prot{i:02d}' for i in range(self._N_GENES)]
        return pd.Series(np.linspace(5.0, -5.0, self._N_GENES), index=genes)

    def _gmt(self, tmp_path: Path) -> str:
        gmt = tmp_path / 'lib.gmt'
        gmt.write_text(''.join(
            f'{term}\tdescription\t' + '\t'.join(genes) + '\n'
            for term, genes in self._SETS.items()
        ))
        return str(gmt)

    def _params(self) -> dict:
        return {'enrichment': {
            'fdr_threshold': 0.05,
            'min_gene_set_size': self._MIN_SIZE,
            'max_gene_set_size': self._MAX_SIZE,
            'gsea_permutations': 100,
            'gsea_seed': 42,
        }}

    def _run(self, tmp_path: Path):
        pytest.importorskip('gseapy')
        from enrichment import run_gsea
        with warnings.catch_warnings():
            # Same local silencing as M05-2: gseapy warnings are not the subject.
            warnings.simplefilter('ignore')
            df, pre_res = run_gsea(self._ranked(), self._gmt(tmp_path),
                                   'GO_BP', 'KO_vs_WT', self._params())
        assert not df.empty
        return df.set_index('term_id'), pre_res

    def test_known_answer_partial_set(self, tmp_path):
        df, _ = self._run(tmp_path)
        assert int(df.loc['TERM_PARTIAL', 'gene_set_size']) == 10

    def test_matches_independent_recomputation(self, tmp_path):
        df, _ = self._run(tmp_path)
        ranked = set(self._ranked().index)
        for term, row in df.iterrows():
            expected = len(set(self._SETS[term]) & ranked)
            assert int(row['gene_set_size']) == expected, (
                f'{term}: gene_set_size {row["gene_set_size"]} != {expected}'
            )

    def test_size_filter_applies_to_intersected_size(self, tmp_path):
        df, _ = self._run(tmp_path)
        assert 'TERM_BELOW_MIN' not in df.index
        assert set(df.index) == {'TERM_PARTIAL', 'TERM_FULL', 'TERM_SMALL'}

    def test_row_invariants(self, tmp_path):
        df, pre_res = self._run(tmp_path)
        tag_denominator = (
            pre_res.res2d.set_index('Term')['Tag %']
            .map(lambda s: int(str(s).split('/')[1]))
        )
        for term, row in df.iterrows():
            size = int(row['gene_set_size'])
            assert size == tag_denominator[term], f'{term}: disagrees with Tag %'
            assert int(row['overlap_size']) <= size, f'{term}: leading edge > set'
            assert self._MIN_SIZE <= size <= self._MAX_SIZE, f'{term}: outside size filter'


# ============================================================
# M05-7: ORA size filter and BH family (invariant, needs gseapy)
# ============================================================

def _bh(pvalues) -> np.ndarray:
    '''Benjamini-Hochberg q-values, written out here independently of gseapy.'''
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


class TestOraSizeFilter:
    '''
    Spec 05 Section 4.5: ORA tests only terms whose size after intersection with
    the background lies in [min_gene_set_size, max_gene_set_size], and BH runs
    over those terms only. Sizes are checked against GMT & background computed
    here, and q-values against an independent BH.

    Second contract check (not a negative control): gseapy.enrich's own output
    on the same input contains the out-of-range terms and a different q-value
    for a kept term, so the filter and the BH recomputation both change the
    result. It fails against the pre-change code, where run_ora returned
    gseapy's output unfiltered.
    '''

    _BACKGROUND: ClassVar[list] = [f'Prot{i:02d}' for i in range(60)]
    _SIG: ClassVar[list] = [f'Prot{i:02d}' for i in range(10)]
    _MIN, _MAX = 5, 40
    _SETS: ClassVar[dict] = {
        # 24 in the GMT but only 4 in the background -> excluded (below min).
        # Proves the filter uses the intersected size, not the GMT size.
        'ORA_SMALL': [f'Prot{i:02d}' for i in range(4)] + [f'Absnt{i:02d}' for i in range(20)],
        'ORA_OK': [f'Prot{i:02d}' for i in list(range(8)) + list(range(30, 42))],  # 20
        'ORA_OK2': [f'Prot{i:02d}' for i in range(5, 15)],                          # 10
        'ORA_BIG': [f'Prot{i:02d}' for i in range(50)],                             # 50 > max
    }

    def _gmt(self, tmp_path: Path) -> str:
        gmt = tmp_path / 'ora.gmt'
        gmt.write_text(''.join(f'{t}\tdesc\t' + '\t'.join(g) + '\n' for t, g in self._SETS.items()))
        return str(gmt)

    def _params(self, min_size: int, max_size: int) -> dict:
        return {'enrichment': {'fdr_threshold': 0.05, 'min_gene_set_size': min_size,
                               'max_gene_set_size': max_size}}

    def _run_ora(self, tmp_path: Path, min_size: int, max_size: int) -> pd.DataFrame:
        pytest.importorskip('gseapy')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = run_ora(self._SIG, self._BACKGROUND, self._gmt(tmp_path), 'GO_BP',
                          'KO_vs_WT', self._params(min_size, max_size))
        return out.set_index('term_id')

    def test_only_in_range_terms_tested(self, tmp_path):
        out = self._run_ora(tmp_path, self._MIN, self._MAX)
        assert set(out.index) == {'ORA_OK', 'ORA_OK2'}
        background = set(self._BACKGROUND)
        for term, row in out.iterrows():
            assert int(row['gene_set_size']) == len(set(self._SETS[term]) & background)

    def test_bh_over_filtered_family(self, tmp_path):
        out = self._run_ora(tmp_path, self._MIN, self._MAX)
        np.testing.assert_allclose(out['adj_pvalue'].to_numpy(), _bh(out['pvalue']))

    def test_no_exclusion_matches_gseapy(self, tmp_path):
        # Metamorphic: with a filter that excludes nothing, run_ora's q-values
        # equal gseapy's own Adjusted P-value.
        gseapy = pytest.importorskip('gseapy')
        out = self._run_ora(tmp_path, 1, 10_000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = gseapy.enrich(gene_list=self._SIG, gene_sets=self._gmt(tmp_path),
                                background=self._BACKGROUND, no_plot=True, verbose=False,
                                cutoff=1.0).res2d.set_index('Term')
        assert set(out.index) == set(raw.index)
        np.testing.assert_allclose(out['adj_pvalue'], raw.loc[out.index, 'Adjusted P-value'])

    def test_raw_pvalues_unchanged_by_filter(self, tmp_path):
        # The filter changes only the BH family: each kept term's hypergeometric
        # p-value equals gseapy's unfiltered P-value for the same term.
        gseapy = pytest.importorskip('gseapy')
        out = self._run_ora(tmp_path, self._MIN, self._MAX)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = gseapy.enrich(gene_list=self._SIG, gene_sets=self._gmt(tmp_path),
                                background=self._BACKGROUND, no_plot=True, verbose=False,
                                cutoff=1.0).res2d.set_index('Term')
        np.testing.assert_array_equal(out['pvalue'].to_numpy(),
                                      raw.loc[out.index, 'P-value'].to_numpy())

    def test_contract_filter_changes_gseapy_result(self, tmp_path):
        gseapy = pytest.importorskip('gseapy')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = gseapy.enrich(gene_list=self._SIG, gene_sets=self._gmt(tmp_path),
                                background=self._BACKGROUND, no_plot=True, verbose=False,
                                cutoff=1.0).res2d.set_index('Term')
        assert {'ORA_SMALL', 'ORA_BIG'} <= set(raw.index)
        out = self._run_ora(tmp_path, self._MIN, self._MAX)
        kept_q = out['adj_pvalue']
        assert not np.allclose(kept_q, raw.loc[kept_q.index, 'Adjusted P-value'])


# ============================================================
# M05-8: permutation_num=0 boundary (needs gseapy)
# ============================================================

class TestGseaNoPermutationBoundary:
    '''
    With permutation_num=0 gseapy omits 'Tag %' from res2d, so the Tag % cross-
    check in count_matched_genes is skipped and the gene split is the only
    safeguard. ProSIFT rejects gsea_permutations < 1 in load_params, so this
    calls gseapy.prerank directly: count_matched_genes must still return the
    GMT & ranked-list size for every term.

    Fixture check (not a negative control): on this fixture the pre-fix len()
    of the raw string differs from the gene count, so a regression to len()
    would fail the main check. The main check's teeth were shown by mutation
    (len of the raw string), since count_matched_genes did not exist pre-fix.
    '''

    def _prerank(self, tmp_path: Path):
        gseapy = pytest.importorskip('gseapy')
        fixture = TestGseaGeneSetSize()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pre_res = gseapy.prerank(
                rnk=fixture._ranked(), gene_sets=fixture._gmt(tmp_path), min_size=5,
                max_size=100, permutation_num=0, ascending=False, no_plot=True,
                verbose=False, seed=42, threads=1,
            )
        return fixture, pre_res

    def test_tag_pct_absent_and_count_still_correct(self, tmp_path):
        fixture, pre_res = self._prerank(tmp_path)
        assert 'Tag %' not in pre_res.res2d.columns
        ranked = set(fixture._ranked().index)
        assert pre_res.results
        for term, record in pre_res.results.items():
            expected = len(set(fixture._SETS[term]) & ranked)
            assert enrichment.count_matched_genes(term, record, tag_pct=None) == expected

    def test_fixture_raw_string_length_differs_from_gene_count(self, tmp_path):
        fixture, pre_res = self._prerank(tmp_path)
        ranked = set(fixture._ranked().index)
        record = pre_res.results['TERM_PARTIAL']
        assert len(record['matched_genes']) != len(set(fixture._SETS['TERM_PARTIAL']) & ranked)


# ============================================================
# M05-9: GSEA symbol case (metamorphic, needs gseapy)
# ============================================================

class TestGseaSymbolCase:
    '''
    Symbol case must not change GSEA results when gseapy can match it: (a)
    mixed-case ranked list + mixed-case GMT, (b) both upper-cased, and (c)
    mixed-case ranked list + upper-case GMT (gseapy upper-cases the list) give
    identical gene_set_size and NES per term. A ranked list holding two symbols
    that differ only by case against an upper-case GMT must not produce a wrong
    count: gseapy 1.1.13 matches the symbol once, so gene_set_size equals the
    case-insensitive intersection.

    Fixture check (not a negative control): a lower-cased ranked list against
    the mixed-case GMT (no upper-casing applies) matches nothing, so gseapy
    raises its "No gene sets passed" LookupError and run_gsea returns no rows
    through that specific handled path. This shows case genuinely matters on
    this fixture, so case_variants_agree is not trivially true.
    '''

    def _run(self, tmp_path: Path, ranked: pd.Series, sets: dict, name: str) -> pd.DataFrame:
        pytest.importorskip('gseapy')
        from enrichment import run_gsea
        gmt = tmp_path / f'{name}.gmt'
        gmt.write_text(''.join(f'{t}\tdesc\t' + '\t'.join(g) + '\n' for t, g in sets.items()))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df, _ = run_gsea(ranked, str(gmt), 'GO_BP', 'KO_vs_WT', TestGseaGeneSetSize()._params())
        return df if df.empty else df.set_index('term_id')

    def _fixture(self):
        fixture = TestGseaGeneSetSize()
        upper_sets = {t: [g.upper() for g in genes] for t, genes in fixture._SETS.items()}
        return fixture._ranked(), fixture._SETS, upper_sets

    def test_case_variants_agree(self, tmp_path):
        ranked, sets, upper_sets = self._fixture()
        base = self._run(tmp_path, ranked, sets, 'mixed')
        upper = self._run(tmp_path, ranked.rename(str.upper), upper_sets, 'upper')
        converted = self._run(tmp_path, ranked, upper_sets, 'converted')
        assert not base.empty, 'no GSEA rows: the comparisons below would be vacuous'
        for other in (upper, converted):
            assert list(other.index) == list(base.index)
            assert other['gene_set_size'].tolist() == base['gene_set_size'].tolist()
            np.testing.assert_allclose(other['enrichment_score'], base['enrichment_score'])

    def test_case_collision_counts_symbol_once(self, tmp_path):
        ranked, _, upper_sets = self._fixture()
        # 'PROT05' and 'Prot05' both ranked; the GMT holds PROT05 once.
        collided = pd.concat([ranked, pd.Series([4.9], index=['PROT05'])]).sort_values(ascending=False)
        out = self._run(tmp_path, collided, upper_sets, 'collision')
        assert not out.empty, 'no GSEA rows: the loop below would assert nothing'
        upper_ranked = {g.upper() for g in collided.index}
        for term, row in out.iterrows():
            assert int(row['gene_set_size']) == len(set(upper_sets[term]) & upper_ranked)

    def test_fixture_unmatched_case_yields_no_terms(self, tmp_path):
        ranked, sets, _ = self._fixture()
        lowered = self._run(tmp_path, ranked.rename(str.lower), sets, 'lowered')
        assert lowered.empty


# ============================================================
# M05-10: ORA overlap_genes order is reproducible (needs gseapy)
# ============================================================

_ORA_ORDER_SCRIPT = r'''
import sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.simplefilter('ignore')
import gseapy, enrichment
genes = [f'Prot{i:02d}' for i in range(60)]
gmt = sys.argv[2]
if sys.argv[3] == 'raw':
    res = gseapy.enrich(gene_list=genes[:12], gene_sets=gmt, background=genes,
                        no_plot=True, verbose=False, cutoff=1.0).res2d
    print(res.sort_values('Term')['Genes'].tolist())
else:
    params = {'enrichment': {'fdr_threshold': 0.05, 'min_gene_set_size': 5,
                             'max_gene_set_size': 100}}
    out = enrichment.run_ora(genes[:12], genes, gmt, 'GO_BP', 'KO_vs_WT', params)
    print(out.sort_values('term_id')['overlap_genes'].tolist())
'''


class TestOraOverlapGenesDeterministic:
    '''
    gseapy joins ORA overlap genes from a Python set, whose order depends on the
    per-process string hash seed. run_ora sorts them, so two processes with
    different PYTHONHASHSEED values write identical overlap_genes.

    Fixture check (not a negative control): gseapy's raw Genes column differs between the same two
    seeds, so the check would catch a missing sort.
    '''

    _SEEDS = ('1', '2')

    def _gmt(self, tmp_path: Path) -> str:
        genes = [f'Prot{i:02d}' for i in range(60)]
        gmt = tmp_path / 'order.gmt'
        gmt.write_text('ORDER_A\tdesc\t' + '\t'.join(genes[:20]) + '\n'
                       + 'ORDER_B\tdesc\t' + '\t'.join(genes[4:30]) + '\n')
        return str(gmt)

    def _outputs(self, tmp_path: Path, mode: str) -> list:
        import os
        import subprocess
        pytest.importorskip('gseapy')
        bin_dir = str(Path(__file__).resolve().parent.parent / 'bin')
        gmt = self._gmt(tmp_path)
        outputs = []
        for seed in self._SEEDS:
            env = {**os.environ, 'PYTHONHASHSEED': seed}
            proc = subprocess.run([sys.executable, '-c', _ORA_ORDER_SCRIPT, bin_dir, gmt, mode],
                                  capture_output=True, text=True, env=env, check=True)
            outputs.append(proc.stdout.strip().splitlines()[-1])
        return outputs

    def test_run_ora_output_identical_across_hash_seeds(self, tmp_path):
        first, second = self._outputs(tmp_path, 'prosift')
        assert first == second

    def test_fixture_raw_gseapy_order_varies(self, tmp_path):
        first, second = self._outputs(tmp_path, 'raw')
        assert first != second


# ============================================================
# M05-11: main() size invariant across ORA and GSEA (needs gseapy)
# ============================================================

def _assert_size_invariants(res: pd.DataFrame, min_size: int, max_size: int) -> None:
    '''
    The property under test: every enrichment row, ORA and GSEA alike, has
    min_size <= gene_set_size <= max_size and overlap_size <= gene_set_size.
    Raises AssertionError naming the first offending row.
    '''
    for _, row in res.iterrows():
        size, overlap = int(row['gene_set_size']), int(row['overlap_size'])
        label = f"{row['analysis_type']} {row['term_id']}"
        assert min_size <= size <= max_size, f'{label}: gene_set_size {size} outside [{min_size}, {max_size}]'
        assert overlap <= size, f'{label}: overlap_size {overlap} > gene_set_size {size}'


_ROW_KEY = ['contrast', 'library', 'analysis_type', 'term_id']


def _assert_sorted_on_key(res: pd.DataFrame) -> None:
    '''Rows are unique on _ROW_KEY and in ascending _ROW_KEY order.'''
    keys = list(res[_ROW_KEY].itertuples(index=False, name=None))
    assert len(set(keys)) == len(keys), 'row key is not unique'
    assert keys == sorted(keys), 'rows are not sorted on the row key'


class TestMainSizeInvariant:
    '''
    End to end through main(): with both ORA and GSEA enabled on a synthetic
    Module 04 table, every row of the written enrichment_results.parquet obeys
    the size invariant, both analyses are present, the out-of-range terms are
    absent from both, and each size equals GMT & detected genes computed here.
    This covers the wiring between run_ora/run_gsea and the written table.
    Terms sitting exactly on min and max are kept by both analyses, and rows
    are written sorted on _ROW_KEY. rpy2 is blocked so rrvgo takes its
    null-column path (as in M05-4).

    Negative controls: the size-range check rejects the written table with one
    row's size pushed out of range, and the sort check rejects the rows
    reversed. The exact-size and inclusive-bounds checks have no paired
    control; they fail against the pre-change code and under mutation.
    '''

    _MIN, _MAX = 5, 30
    _N = 80

    def _genes(self) -> list:
        return [f'Gm{i:03d}' for i in range(self._N)]

    def _sets(self) -> dict:
        g = self._genes()
        return {
            'SET_SMALL': g[:3] + [f'Absnt{i:02d}' for i in range(10)],  # 3 detected -> out
            # Named so the alphabetical order (MID < TOP) is the reverse of
            # gseapy's NES order (TOP, the top-ranked genes, has the higher NES).
            # The sort test can then tell whether term_id is in the sort key.
            'SET_TOP': g[:12],                                          # 12
            'SET_MID': g[10:30],                                        # 20
            'SET_BIG': g[:40],                                          # 40 -> out
        }

    def _write_inputs(self, tmp_path: Path, min_size: int, max_size: int) -> tuple:
        genes = self._genes()
        t_stat = np.linspace(6.0, -6.0, self._N)
        da_df = pd.DataFrame({
            'protein_id':       [f'P{i:03d}' for i in range(self._N)],
            'gene_symbol':      genes,
            'contrast':         'KO_vs_WT',
            'significant':      [i < 15 for i in range(self._N)],
            'log2_fc':          t_stat / 3,
            'deqms_t':          t_stat,
            'limma_t':          t_stat,
            'deqms_pvalue':     np.linspace(1e-6, 0.9, self._N),
            'limma_pvalue':     np.linspace(1e-6, 0.9, self._N),
            'deqms_adj_pvalue': np.linspace(1e-4, 0.95, self._N),
            'limma_adj_pvalue': np.linspace(1e-4, 0.95, self._N),
        })
        results_path = tmp_path / 'results.parquet'
        da_df.to_parquet(results_path, index=False)
        gmt_dir = tmp_path / 'gmt'
        gmt_dir.mkdir()
        (gmt_dir / 'm2.cp.reactome.v2026.1.Mm.symbols.gmt').write_text(''.join(
            f'{t}\tdesc\t' + '\t'.join(g) + '\n' for t, g in self._sets().items()
        ))
        params = {'enrichment': {
            'gene_set_libraries': ['gmt/m2.cp.reactome.v2026.1.Mm.symbols.gmt'],
            'gsea_ranking': 't_statistic', 'run_ora': True, 'run_gsea': True,
            'fdr_threshold': 0.05, 'min_gene_set_size': min_size,
            'max_gene_set_size': max_size, 'plot_top_n': 5, 'plot_top_gsea_traces': 1,
            'gsea_permutations': 100, 'gsea_seed': 42,
        }}
        params_path = tmp_path / 'params.yml'
        params_path.write_text(yaml.safe_dump(params))
        return results_path, params_path

    def _run_main(self, tmp_path: Path, monkeypatch, min_size: int = _MIN,
                  max_size: int = _MAX) -> pd.DataFrame:
        pytest.importorskip('gseapy')
        results_path, params_path = self._write_inputs(tmp_path, min_size, max_size)
        outdir = tmp_path / 'out'
        monkeypatch.setitem(sys.modules, 'rpy2', None)
        monkeypatch.setitem(sys.modules, 'rpy2.robjects', None)
        monkeypatch.setattr(sys, 'argv', [
            'enrichment.py', '--results', str(results_path), '--params', str(params_path),
            '--run-id', 'SIZERUN', '--outdir', str(outdir),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            enrichment.main()
        return pd.read_parquet(outdir / 'SIZERUN.enrichment_results.parquet')

    def test_every_row_within_size_filter(self, tmp_path, monkeypatch):
        res = self._run_main(tmp_path, monkeypatch)
        assert set(res['analysis_type']) == {'ORA', 'GSEA'}
        _assert_size_invariants(res, self._MIN, self._MAX)
        detected = set(self._genes())
        for atype in ('ORA', 'GSEA'):
            sub = res[res['analysis_type'] == atype].set_index('term_id')
            assert set(sub.index) <= {'SET_TOP', 'SET_MID'}, f'{atype}: {set(sub.index)}'
            for term, row in sub.iterrows():
                assert int(row['gene_set_size']) == len(set(self._sets()[term]) & detected)

    def test_size_bounds_inclusive_for_both_analyses(self, tmp_path, monkeypatch):
        # SET_TOP has exactly 12 detected genes and SET_MID exactly 20: with
        # [12, 20] both sit on a bound and must be kept by ORA and by GSEA.
        res = self._run_main(tmp_path, monkeypatch, min_size=12, max_size=20)
        for atype in ('ORA', 'GSEA'):
            kept = set(res.loc[res['analysis_type'] == atype, 'term_id'])
            assert kept == {'SET_TOP', 'SET_MID'}, f'{atype}: {kept}'

    def test_written_rows_sorted_on_unique_key(self, tmp_path, monkeypatch):
        # Byte-reproducible output: rows are written in a stable order on a
        # unique key, so tied-NES terms cannot swap places between runs.
        res = self._run_main(tmp_path, monkeypatch)
        _assert_sorted_on_key(res)
        csv = pd.read_csv(tmp_path / 'out' / 'SIZERUN.enrichment_results.csv')
        assert csv[_ROW_KEY].astype(str).values.tolist() == res[_ROW_KEY].astype(str).values.tolist()

    def test_negative_control_unsorted_rows_are_caught(self, tmp_path, monkeypatch):
        res = self._run_main(tmp_path, monkeypatch)
        with pytest.raises(AssertionError, match='not sorted'):
            _assert_sorted_on_key(res.iloc[::-1].reset_index(drop=True))

    def test_negative_control_out_of_range_row_is_caught(self, tmp_path, monkeypatch):
        res = self._run_main(tmp_path, monkeypatch)
        tampered = res.copy()
        tampered.loc[tampered.index[0], 'gene_set_size'] = self._MAX + 1
        with pytest.raises(AssertionError, match='outside'):
            _assert_size_invariants(tampered, self._MIN, self._MAX)


# ============================================================
# M05-12: gene_set_version per row on a populated run (needs gseapy)
# ============================================================

def _assert_versions_match(res: pd.DataFrame, expected: dict) -> None:
    '''
    The property under test: gene_set_version is a string column, never null,
    and every row carries the release of the GMT its library came from
    (expected maps library short name -> version token).
    '''
    assert str(res['gene_set_version'].dtype) == 'string', res['gene_set_version'].dtype
    assert res['gene_set_version'].notna().all(), 'null gene_set_version'
    for (lib, atype), sub in res.groupby(['library', 'analysis_type']):
        got = set(sub['gene_set_version'])
        assert got == {expected[lib]}, f'{lib} {atype}: gene_set_version {got} != {expected[lib]}'


class TestGeneSetVersionPerRow:
    '''
    End to end through main() with two GMTs from different MSigDB releases:
    every written row, ORA and GSEA, carries its own GMT's release, parsed
    from the filename. rpy2 is blocked so rrvgo takes its null-column path.

    Negative control: the check rejects the written table with the two
    libraries' versions swapped (the wrong-GMT assignment it must catch).
    '''

    _GMTS: ClassVar[dict] = {
        'GO_BP':    'm5.go.bp.v2026.1.Mm.symbols.gmt',
        'REACTOME': 'm2.cp.reactome.v2025.1.Mm.symbols.gmt',
    }
    _EXPECTED: ClassVar[dict] = {'GO_BP': 'v2026.1.Mm', 'REACTOME': 'v2025.1.Mm'}
    _N = 60

    def _run_main(self, tmp_path: Path, monkeypatch) -> pd.DataFrame:
        pytest.importorskip('gseapy')
        genes = [f'Gm{i:03d}' for i in range(self._N)]
        t_stat = np.linspace(5.0, -5.0, self._N)
        pd.DataFrame({
            'protein_id':       [f'P{i:03d}' for i in range(self._N)],
            'gene_symbol':      genes,
            'contrast':         'KO_vs_WT',
            'significant':      [i < 12 for i in range(self._N)],
            'log2_fc':          t_stat / 3,
            'deqms_t':          t_stat,
            'limma_t':          t_stat,
            'deqms_pvalue':     np.linspace(1e-6, 0.9, self._N),
            'limma_pvalue':     np.linspace(1e-6, 0.9, self._N),
            'deqms_adj_pvalue': np.linspace(1e-4, 0.95, self._N),
            'limma_adj_pvalue': np.linspace(1e-4, 0.95, self._N),
        }).to_parquet(tmp_path / 'results.parquet', index=False)
        gmt_dir = tmp_path / 'gmt'
        gmt_dir.mkdir()
        # Distinct term names per library; each set has 10-20 detected genes.
        for lib, name in self._GMTS.items():
            (gmt_dir / name).write_text(''.join(
                f'{lib}_SET{k}\tdesc\t' + '\t'.join(genes[k * 5:k * 5 + 15]) + '\n'
                for k in range(4)
            ))
        (tmp_path / 'params.yml').write_text(yaml.safe_dump({'enrichment': {
            'gene_set_libraries': [f'gmt/{n}' for n in self._GMTS.values()],
            'gsea_ranking': 't_statistic', 'run_ora': True, 'run_gsea': True,
            'fdr_threshold': 0.05, 'min_gene_set_size': 5, 'max_gene_set_size': 30,
            'plot_top_n': 5, 'plot_top_gsea_traces': 1,
            'gsea_permutations': 100, 'gsea_seed': 42,
        }}))
        outdir = tmp_path / 'out'
        monkeypatch.setitem(sys.modules, 'rpy2', None)
        monkeypatch.setitem(sys.modules, 'rpy2.robjects', None)
        monkeypatch.setattr(sys, 'argv', [
            'enrichment.py', '--results', str(tmp_path / 'results.parquet'),
            '--params', str(tmp_path / 'params.yml'), '--run-id', 'VERRUN',
            '--outdir', str(outdir),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            enrichment.main()
        return pd.read_parquet(outdir / 'VERRUN.enrichment_results.parquet')

    def test_every_row_carries_its_gmt_release(self, tmp_path, monkeypatch):
        res = self._run_main(tmp_path, monkeypatch)
        # Both libraries and both analyses must be present for the check to bite.
        assert set(res['library']) == set(self._GMTS)
        assert set(res['analysis_type']) == {'ORA', 'GSEA'}
        _assert_versions_match(res, self._EXPECTED)

    def test_negative_control_swapped_versions_are_caught(self, tmp_path, monkeypatch):
        res = self._run_main(tmp_path, monkeypatch)
        swapped = res.copy()
        swapped['gene_set_version'] = swapped['library'].map(
            {'GO_BP': 'v2025.1.Mm', 'REACTOME': 'v2026.1.Mm'}).astype('string')
        with pytest.raises(AssertionError, match='gene_set_version'):
            _assert_versions_match(swapped, self._EXPECTED)


# ============================================================
# M05-3: in_significant_set decoupling (negative-control, local)
# ============================================================

class TestInSignificantSetDecoupling:
    '''
    build_protein_term_mapping sets in_significant_set from the per-protein DA
    significance flag, independently of whether the term is enrichment-significant.
    For a term whose member genes are all DA-non-significant, every mapping row
    has in_significant_set == False, even though the term appears in (is
    significant in) the enrichment results.
    '''

    def _da(self) -> pd.DataFrame:
        return pd.DataFrame({
            'protein_id':  ['P1', 'P2', 'P3'],
            'gene_symbol': ['GeneA', 'GeneB', 'GeneC'],
            'contrast':    ['KO_vs_WT', 'KO_vs_WT', 'KO_vs_WT'],
            # GeneA/GeneB are DA-non-significant; GeneC is DA-significant.
            'significant': [False, False, True],
        })

    def _gmt(self, tmp_path: Path) -> str:
        gmt = tmp_path / 'lib.gmt'
        # TERM_NS: only DA-non-significant members; TERM_SIG: a DA-significant one.
        gmt.write_text(
            'TERM_NS\tdesc\tGeneA\tGeneB\n'
            'TERM_SIG\tdesc\tGeneC\n'
        )
        return str(gmt)

    def test_non_significant_members_stay_false_though_term_enriched(self, tmp_path):
        gmt_path = self._gmt(tmp_path)
        # Both terms are enrichment-significant (they appear in the results table).
        enr = pd.DataFrame({'term_id': ['TERM_NS', 'TERM_SIG']})
        out = build_protein_term_mapping(
            self._da(), [gmt_path], ['GO_BP'], enr, {},
        )
        ns_rows = out[out['term_id'] == 'TERM_NS']
        assert not ns_rows.empty
        # Decoupling: term is enrichment-significant, members are DA-non-significant.
        assert ns_rows['in_significant_set'].eq(False).all()

    def test_negative_control_significant_member_yields_true(self, tmp_path):
        '''
        NEGATIVE CONTROL (teeth): a term with a DA-significant member yields
        in_significant_set == True for that member, regardless of term enrichment.
        This shows the flag is not hard-wired to False -- so the False above is a
        genuine reflection of per-protein DA status.
        '''
        gmt_path = self._gmt(tmp_path)
        enr = pd.DataFrame({'term_id': ['TERM_NS', 'TERM_SIG']})
        out = build_protein_term_mapping(
            self._da(), [gmt_path], ['GO_BP'], enr, {},
        ).set_index('gene_symbol')
        assert bool(out.loc['GeneC', 'in_significant_set']) is True


# ============================================================
# M05-4: rrvgo cluster_status contract (invariant, local)
# ============================================================

def _assert_rowcount_preserved(inp: pd.DataFrame, out: pd.DataFrame) -> None:
    '''
    The property under test: clustering never drops (or adds) rows. Raises
    AssertionError otherwise.
    '''
    assert len(out) == len(inp), (
        f'row count not preserved: in={len(inp)} out={len(out)}'
    )


def _assert_cluster_contract(out: pd.DataFrame) -> None:
    '''
    The rrvgo output contract (spec Section 4.13). Raises AssertionError if any
    clause fails:
      1. cluster_status is non-null and in the vocabulary; non-GO rows are
         'not_go_library'.
      2. cluster_id is non-null exactly for 'clustered' rows.
      3. is_representative is non-null, so a representatives-only filter can
         never silently drop a row through NA.
      4. Unclustered rows are their own representative (True, no parent_term).
      5. Per (library, analysis_type, contrast, cluster_id): exactly one
         representative, it has the lowest adj_pvalue, and every other member's
         parent_term is its term_id.
      6. GSEA clusters never mix NES signs.
    '''
    status = out['cluster_status']
    assert status.notna().all(), 'cluster_status has nulls'
    assert set(status) <= set(CLUSTER_STATUSES), f'unknown statuses: {set(status) - set(CLUSTER_STATUSES)}'
    non_go = ~out['library'].isin(['GO_BP', 'GO_MF', 'GO_CC'])
    assert (status[non_go] == 'not_go_library').all(), 'non-GO row with a GO status'

    clustered = status == 'clustered'
    assert (out['cluster_id'].notna() == clustered).all(), 'cluster_id set iff clustered violated'
    assert out['is_representative'].notna().all(), 'is_representative has nulls'

    unclustered = out[~clustered]
    assert unclustered['is_representative'].astype(bool).all(), 'unclustered row not its own representative'
    assert unclustered['parent_term'].isna().all(), 'unclustered row has a parent_term'

    keys = ['library', 'analysis_type', 'contrast', 'cluster_id']
    for key, members in out[clustered].groupby(keys):
        reps = members[members['is_representative'].astype(bool)]
        assert len(reps) == 1, f'cluster {key}: {len(reps)} representatives'
        rep = reps.iloc[0]
        if members['adj_pvalue'].notna().any():   # all-NaN cluster: any member may represent it
            assert rep['adj_pvalue'] <= members['adj_pvalue'].min(), f'cluster {key}: representative not most significant'
        others = members[~members['is_representative'].astype(bool)]
        assert (others['parent_term'] == rep['term_id']).all(), f'cluster {key}: parent_term mismatch'
        if key[1] == 'GSEA':
            signs = set(np.sign(members['enrichment_score'].astype(float)))
            assert len(signs) == 1, f'cluster {key}: mixes NES signs {signs}'


def _contract_frame() -> pd.DataFrame:
    '''A valid post-clustering table: one ORA cluster, one up and one down GSEA cluster, unclustered rows.'''
    return pd.DataFrame({
        'library':        ['GO_BP', 'GO_BP', 'GO_BP', 'GO_BP', 'GO_BP', 'GO_BP', 'GO_BP', 'REACTOME'],
        'analysis_type':  ['ORA', 'ORA', 'ORA', 'GSEA', 'GSEA', 'GSEA', 'GSEA', 'GSEA'],
        'contrast':       ['KO_vs_WT'] * 8,
        'term_id':        ['GOBP_A', 'GOBP_B', 'GOBP_C', 'GOBP_U1', 'GOBP_U2', 'GOBP_D1', 'GOBP_D2', 'R-1'],
        'adj_pvalue':     [0.01, 0.02, 0.03, 0.01, 0.02, 0.01, 0.04, 0.001],
        'enrichment_score': [np.nan, np.nan, np.nan, 2.0, 1.5, -2.0, -1.2, 2.5],
        'cluster_status': ['clustered', 'clustered', 'unresolved_name',
                           'clustered', 'clustered', 'clustered', 'clustered', 'not_go_library'],
        'cluster_id':     pd.array([1, 1, None, 1, 1, 2, 2, None], dtype='Int64'),
        'is_representative': pd.array([True, False, True, True, False, True, False, True], dtype='boolean'),
        'parent_term':    [None, 'GOBP_A', None, None, 'GOBP_U1', None, 'GOBP_D1', None],
    })


class TestClusterContract:
    '''
    cluster_go_terms must preserve row count and satisfy _assert_cluster_contract
    on every path. The degradation path (rpy2 blocked) is run locally; the live
    R path is covered by TestClusterLiveRBlock (Rscript) and, through rpy2, only
    on the cluster. Each clause has a negative control proving it has teeth.
    '''

    def _enr(self) -> pd.DataFrame:
        return pd.DataFrame({
            'library':       ['GO_BP', 'REACTOME'],
            'term_id':       ['GOBP_UNRESOLVABLE_XYZ', 'R-HSA-000000'],
            'analysis_type': ['ORA', 'ORA'],
            'contrast':      ['KO_vs_WT', 'KO_vs_WT'],
            'adj_pvalue':    [0.01, 0.02],
            'enrichment_score': [np.nan, np.nan],
        })

    def test_degradation_path_satisfies_contract(self, monkeypatch):
        # Block rpy2 so cluster_go_terms' lazy `import rpy2.robjects` raises
        # ImportError (instead of starting embedded R, which segfaults where R is
        # not linked), taking the degradation path.
        monkeypatch.setitem(sys.modules, 'rpy2', None)
        enr = self._enr()
        out = cluster_go_terms(enr)
        _assert_rowcount_preserved(enr, out)
        _assert_cluster_contract(out)
        # Both terms survive a representatives-only view.
        assert set(out.loc[out['is_representative'], 'term_id']) == set(enr['term_id'])

    def test_valid_frame_passes(self):
        _assert_cluster_contract(_contract_frame())

    @pytest.mark.parametrize('corrupt', [
        # NA is_representative: the original bug (silently dropped by == True).
        lambda f: f.assign(is_representative=f['is_representative'].where(f['library'] == 'GO_BP')),
        # Two representatives in one cluster.
        lambda f: f.assign(is_representative=f['is_representative'].mask(f['term_id'] == 'GOBP_B', True)),
        # Representative is not the most significant member.
        lambda f: f.assign(adj_pvalue=f['adj_pvalue'].mask(f['term_id'] == 'GOBP_B', 0.001)),
        # A GSEA cluster mixing NES signs.
        lambda f: f.assign(cluster_id=f['cluster_id'].mask(f['term_id'] == 'GOBP_D2', 1),
                           parent_term=f['parent_term'].mask(f['term_id'] == 'GOBP_D2', 'GOBP_U1')),
        # Unclustered row not its own representative.
        lambda f: f.assign(is_representative=f['is_representative'].mask(f['term_id'] == 'GOBP_C', False)),
        # cluster_id on an unclustered row.
        lambda f: f.assign(cluster_id=f['cluster_id'].mask(f['term_id'] == 'GOBP_C', 9)),
        # Null / unknown status.
        lambda f: f.assign(cluster_status=f['cluster_status'].mask(f['term_id'] == 'GOBP_C', None)),
        lambda f: f.assign(cluster_status=f['cluster_status'].mask(f['term_id'] == 'R-1', 'clustered')),
    ])
    def test_negative_control_contract_violation_fails(self, corrupt):
        '''NEGATIVE CONTROL (teeth): each clause must reject a frame that breaks it.'''
        with pytest.raises(AssertionError):
            _assert_cluster_contract(corrupt(_contract_frame()))

    def test_negative_control_dropped_row_fails_rowcount(self):
        '''
        NEGATIVE CONTROL (teeth): a clusterer that drops a row must fail the
        row-count-preserved assertion. If it did not, the contract could silently
        lose terms.
        '''
        enr = self._enr()
        dropped = enr.iloc[:-1].copy()          # simulate a lost Reactome row
        with pytest.raises(AssertionError):
            _assert_rowcount_preserved(enr, dropped)


# ============================================================
# M05-13: shipped rrvgo R block, run through Rscript (slow, needs R)
# ============================================================
# Embedded R (rpy2) segfaults on the dev Mac, so the exact R code shipped in
# _RRVGO_R_BLOCK is run here through Rscript with the same inputs rpy2 would set,
# and its outputs feed the real _annotate_cluster_unit. This does NOT exercise
# the rpy2 marshalling in _run_rrvgo_unit (verify that on the cluster).

def _rscript() -> str | None:
    cand = Path(sys.executable).parent / 'Rscript'
    return str(cand) if cand.exists() else None


def _r_chr(values) -> str:
    return 'c(' + ', '.join('"' + str(v).replace('"', '\\"') + '"' for v in values) + ')'


def _run_r_block_via_rscript(unit: pd.DataFrame, tmp_path: Path, timeout: int = 600) -> dict:
    '''Run enrichment._RRVGO_R_BLOCK for one unit in Rscript; return r_out.

    Full-size units (thousands of GO terms) take tens of minutes; raise timeout for those.
    '''
    import subprocess
    ids = unit['term_id'].tolist()
    outputs = ['prosift_stage', 'prosift_error', 'prosift_resolved_msig', 'prosift_resolved_go',
               'prosift_sim_go', 'prosift_red_go', 'prosift_red_cluster']
    script = '\n'.join([
        f'prosift_msigdb_ids <- {_r_chr(ids)}',
        f'prosift_lookup <- {_r_chr(enrichment._msigdb_name_to_go_lookup_phrase(t) for t in ids)}',
        'prosift_scores <- c(' + ', '.join(repr(float(s)) for s in enrichment._rrvgo_scores(unit['adj_pvalue'])) + ')',
        'prosift_ont <- "BP"', 'prosift_orgdb <- "org.Mm.eg.db"',
        f'prosift_threshold <- {enrichment._RRVGO_THRESHOLD}',
        enrichment._RRVGO_R_BLOCK,
        f'for (v in {_r_chr(outputs)}) writeLines(as.character(get(v)), file.path("{tmp_path}", v))',
    ])
    (tmp_path / 'run.R').write_text(script)
    proc = subprocess.run([_rscript(), '--vanilla', str(tmp_path / 'run.R')],
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-2000:]

    def vec(name):
        return (tmp_path / name).read_text().split('\n')[:-1]

    return {
        'stage': vec('prosift_stage')[0],
        'error': ' '.join(vec('prosift_error')),
        'resolved': dict(zip(vec('prosift_resolved_msig'), vec('prosift_resolved_go'), strict=True)),
        'sim_go': set(vec('prosift_sim_go')),
        'red': dict(zip(vec('prosift_red_go'), map(int, vec('prosift_red_cluster')), strict=True)),
    }


@pytest.mark.slow
@pytest.mark.skipif(_rscript() is None, reason='Rscript not in the prosift env')
class TestClusterLiveRBlock:

    def test_real_go_terms_cluster_by_theme(self, tmp_path):
        unit = pd.DataFrame({
            'term_id': [
                'GOBP_MRNA_PROCESSING', 'GOBP_RNA_SPLICING',                       # splicing theme
                'GOBP_SYNAPTIC_VESICLE_CYCLE', 'GOBP_SYNAPTIC_VESICLE_EXOCYTOSIS',  # vesicle theme
                'GOBP_PROTEIN_RNA_COMPLEX_ORGANIZATION',  # GO name has a hyphen: backward lookup misses
                'GOBP_NOT_A_REAL_TERM_XYZ',
            ],
            'adj_pvalue':   [1e-5, 1e-4, 1e-6, 1e-6, 1e-2, 0.5],
            'overlap_size': [10, 8, 6, 6, 4, 2],
        })
        r_out = _run_r_block_via_rscript(unit, tmp_path)
        assert r_out['stage'] == 'ok', r_out
        res = _annotate_cluster_unit(unit, 'ORA', r_out)
        status = dict(zip(unit['term_id'], res['cluster_status'], strict=True))
        assert status['GOBP_NOT_A_REAL_TERM_XYZ'] == 'unresolved_name'
        assert status['GOBP_PROTEIN_RNA_COMPLEX_ORGANIZATION'] == 'unresolved_name'
        cid = dict(zip(unit['term_id'], res['cluster_local'], strict=True))
        assert cid['GOBP_MRNA_PROCESSING'] == cid['GOBP_RNA_SPLICING']
        assert cid['GOBP_SYNAPTIC_VESICLE_CYCLE'] == cid['GOBP_SYNAPTIC_VESICLE_EXOCYTOSIS']
        assert cid['GOBP_MRNA_PROCESSING'] != cid['GOBP_SYNAPTIC_VESICLE_CYCLE']
        # Tied adj_pvalue (1e-6) and tied overlap: term_id breaks the tie.
        reps = set(unit.loc[res['is_representative'] & (res['cluster_status'] == 'clustered'), 'term_id'])
        assert reps == {'GOBP_MRNA_PROCESSING', 'GOBP_SYNAPTIC_VESICLE_CYCLE'}

        framed = unit.assign(library='GO_BP', analysis_type='ORA', contrast='KO_vs_WT',
                             enrichment_score=np.nan,
                             cluster_status=res['cluster_status'], cluster_id=res['cluster_local'],
                             is_representative=res['is_representative'], parent_term=res['parent_term'])
        _assert_cluster_contract(framed)


# ============================================================
# M05-5: Empty-run writes four outputs (boundary, local)
# ============================================================

_ENRICHMENT_SCHEMA = [
    'term_id', 'term_name', 'library', 'analysis_type', 'contrast',
    'pvalue', 'adj_pvalue', 'enrichment_score', 'odds_ratio',
    'combined_score', 'gene_set_size', 'overlap_size', 'overlap_genes',
    'gene_set_version', 'cluster_status', 'cluster_id', 'is_representative',
    'parent_term',
]


def _assert_four_outputs(outdir: Path, run_id: str) -> None:
    '''
    The property under test: all four required Module 05 outputs exist and the
    enrichment results table carries the full column schema. Raises
    AssertionError on a missing output or a truncated schema.
    '''
    required = [
        outdir / f'{run_id}.enrichment_results.parquet',
        outdir / f'{run_id}.enrichment_results.csv',
        outdir / f'{run_id}.protein_term_mapping.parquet',
        outdir / f'{run_id}.enrichment_summary.txt',
    ]
    for path in required:
        assert path.exists(), f'missing required output: {path.name}'
    cols = set(pd.read_parquet(required[0]).columns)
    missing = set(_ENRICHMENT_SCHEMA) - cols
    assert not missing, f'enrichment results schema truncated, missing: {missing}'


class TestEmptyRunOutputs:
    '''
    A run with zero significant genes and zero returned terms must still write all
    four required outputs (enrichment_results.parquet/.csv,
    protein_term_mapping.parquet, enrichment_summary.txt) with the full column
    schema. We drive main() with ORA enabled but no significant genes (ORA
    auto-skips) and GSEA disabled, so no gseapy or R call is needed.
    '''

    def _write_inputs(self, tmp_path: Path) -> tuple[Path, Path, str]:
        run_id = 'EMPTYRUN'
        # Module 04 results: 3 proteins, none significant, all mapped.
        da_df = pd.DataFrame({
            'protein_id':       ['P1', 'P2', 'P3'],
            'gene_symbol':      ['GeneA', 'GeneB', 'GeneC'],
            'contrast':         ['KO_vs_WT', 'KO_vs_WT', 'KO_vs_WT'],
            'significant':      [False, False, False],
            'log2_fc':          [0.1, -0.2, 0.05],
            'deqms_t':          [0.5, -0.4, 0.3],
            'limma_t':          [0.4, -0.3, 0.2],
            'deqms_pvalue':     [0.5, 0.6, 0.7],
            'limma_pvalue':     [0.5, 0.6, 0.7],
            'deqms_adj_pvalue': [0.8, 0.9, 0.95],
            'limma_adj_pvalue': [0.8, 0.9, 0.95],
        })
        results_path = tmp_path / 'results.parquet'
        da_df.to_parquet(results_path, index=False)

        # load_params validates that every gene_set_library exists on disk, so
        # write a real (minimal) GMT. It is never opened during the empty run
        # (build_protein_term_mapping returns early on empty enrichment results),
        # but its presence and 'go.bp' substring drive load_params + the GO_BP
        # short-name mapping. Path resolves relative to the params.yml directory.
        gmt_dir = tmp_path / 'gmt'
        gmt_dir.mkdir()
        gmt_file = gmt_dir / 'm5.go.bp.v2026.1.Mm.symbols.gmt'
        gmt_file.write_text('TERM_A\tdesc\tGeneA\tGeneB\n')

        params = {'enrichment': {
            'gene_set_libraries': ['gmt/m5.go.bp.v2026.1.Mm.symbols.gmt'],
            'gsea_ranking':       't_statistic',
            'run_ora':            True,    # enabled, but no sig genes -> auto-skips
            'run_gsea':           False,   # disabled -> no gseapy/gmt read needed
            'fdr_threshold':      0.05,
            'min_gene_set_size':  5,
            'max_gene_set_size':  500,
            'gsea_permutations':  100,
            'gsea_seed':          42,
        }}
        params_path = tmp_path / 'params.yml'
        params_path.write_text(yaml.safe_dump(params))
        return results_path, params_path, run_id

    def test_empty_run_writes_all_four_outputs(self, tmp_path, monkeypatch):
        results_path, params_path, run_id = self._write_inputs(tmp_path)
        outdir = tmp_path / 'out'
        monkeypatch.setattr(sys, 'argv', [
            'enrichment.py',
            '--results', str(results_path),
            '--params', str(params_path),
            '--run-id', run_id,
            '--outdir', str(outdir),
        ])
        enrichment.main()
        _assert_four_outputs(outdir, run_id)

        # The enrichment results table is genuinely empty (zero terms) yet typed.
        res = pd.read_parquet(outdir / f'{run_id}.enrichment_results.parquet')
        assert len(res) == 0
        # Same gene_set_version dtype as a populated run (stable Parquet schema).
        assert str(res['gene_set_version'].dtype) == 'string'

    def test_negative_control_missing_output_is_caught(self, tmp_path):
        '''
        NEGATIVE CONTROL (teeth): an output directory that is missing a required
        file (here, an empty directory) must fail the four-output contract. If it
        did not, the writer contract could silently emit a partial result set.
        '''
        empty = tmp_path / 'nothing'
        empty.mkdir()
        with pytest.raises(AssertionError):
            _assert_four_outputs(empty, 'EMPTYRUN')

    def test_one_resolvable_term_ends_too_few(self, tmp_path):
        # Exercises the R block's own 'too_few' stage and its output reset: one
        # resolvable and one unresolvable name never reach calculateSimMatrix.
        unit = pd.DataFrame({'term_id': ['GOBP_MRNA_PROCESSING', 'GOBP_NOT_A_REAL_TERM_XYZ'],
                             'adj_pvalue': [1e-5, 0.5], 'overlap_size': [10, 2]})
        r_out = _run_r_block_via_rscript(unit, tmp_path)
        assert r_out['stage'] == 'too_few', r_out
        assert r_out['sim_go'] == set() and r_out['red'] == {}
        res = _annotate_cluster_unit(unit, 'ORA', r_out)
        assert list(res['cluster_status']) == ['too_few_terms', 'unresolved_name']
        assert res['is_representative'].all()
