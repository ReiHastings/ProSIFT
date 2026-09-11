#!/usr/bin/env python3
# title: test_enrichment_gate.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-24
# last modified: 2026-07-24
#
# purpose:
#   Permanent empirical-gate tests for Module 05 ENRICHMENT (bin/enrichment.py).
#   Encodes the APPROVED Piece C review invariants M05-1 .. M05-5 as runnable
#   pytest checks, each paired with a negative control that proves the check has
#   teeth. Most gates are pure Python; one exercises a live gseapy.prerank
#   (guarded so it skips where gseapy is unavailable, in the style of
#   tests/test_differential_abundance_r.py's dependency probe).
#
#   Gate map:
#     M05-1  ORA background identity (invariant)          -> TestOraBackgroundIdentity
#     M05-2  GSEA NES sign (metamorphic, needs gseapy)    -> TestGseaNesSign
#     M05-3  in_significant_set decoupling (neg-control)  -> TestInSignificantSetDecoupling
#     M05-4  rrvgo null-column contract (boundary)        -> TestClusterNullColumnContract
#     M05-5  empty-run writes four outputs (boundary)     -> TestEmptyRunOutputs
#
#   The live rrvgo/R clustering call (M05-4) and any live gseapy call are not
#   required: M05-4 forces the pure-Python null-init path by blocking rpy2 (same
#   monkeypatch trick as test_enrichment.py's degradation test), and M05-1 /
#   M05-5 avoid gseapy entirely. Only M05-2 runs gseapy.prerank and is skipped
#   if gseapy is unavailable.
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
#   copy/paste: pytest tests/test_enrichment_gate.py -q

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import enrichment  # noqa: E402
from enrichment import (  # noqa: E402
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
        sig_genes = contrast_df[contrast_df['significant'] == True]['gene_symbol'].tolist()
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
        assert (ns_rows['in_significant_set'] == False).all()

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
# M05-4: rrvgo null-column contract (boundary, local)
# ============================================================

def _assert_rowcount_preserved(inp: pd.DataFrame, out: pd.DataFrame) -> None:
    '''
    The property under test: clustering never drops (or adds) rows. Raises
    AssertionError otherwise.
    '''
    assert len(out) == len(inp), (
        f'row count not preserved: in={len(inp)} out={len(out)}'
    )


class TestClusterNullColumnContract:
    '''
    cluster_go_terms must preserve row count and add cluster_id /
    is_representative / parent_term. An unresolvable GO term AND any Reactome
    term end with all three columns NA and still appear in the results. We
    exercise the pure-Python framing/null-init path by blocking rpy2 so the lazy
    import fails (graceful degradation), rather than invoking live rrvgo/R.
    '''

    def _enr(self) -> pd.DataFrame:
        return pd.DataFrame({
            'library':       ['GO_BP', 'REACTOME'],
            'term_id':       ['GOBP_UNRESOLVABLE_XYZ', 'R-HSA-000000'],
            'analysis_type': ['ORA', 'ORA'],
            'contrast':      ['KO_vs_WT', 'KO_vs_WT'],
            'adj_pvalue':    [0.01, 0.02],
        })

    def test_null_columns_and_rowcount_preserved(self, monkeypatch):
        # Block rpy2 so cluster_go_terms' lazy `import rpy2.robjects` raises
        # ImportError (instead of starting embedded R, which segfaults where R is
        # not linked), taking the null-init degradation path.
        monkeypatch.setitem(sys.modules, 'rpy2', None)
        enr = self._enr()
        out = cluster_go_terms(enr)

        _assert_rowcount_preserved(enr, out)
        for col in ('cluster_id', 'is_representative', 'parent_term'):
            assert col in out.columns
            assert out[col].isna().all(), f'{col} should be NA on the null-init path'
        # Both the unresolvable GO term and the Reactome term survive.
        assert set(out['term_id']) == {'GOBP_UNRESOLVABLE_XYZ', 'R-HSA-000000'}

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
# M05-5: Empty-run writes four outputs (boundary, local)
# ============================================================

_ENRICHMENT_SCHEMA = [
    'term_id', 'term_name', 'library', 'analysis_type', 'contrast',
    'pvalue', 'adj_pvalue', 'enrichment_score', 'odds_ratio',
    'combined_score', 'gene_set_size', 'overlap_size', 'overlap_genes',
    'cluster_id', 'is_representative', 'parent_term',
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
