#!/usr/bin/env python3
# title: test_enrichment.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-07-09
# last modified: 2026-09-30
#
# purpose:
#   Unit tests for Module 05 ENRICHMENT (bin/enrichment.py). The ORA/GSEA calls
#   (gseapy) and the GO-term redundancy reduction (rrvgo via R) are integration
#   concerns; the module's pure-Python data logic is covered here:
#     - _library_short_name             : GMT path -> short library id
#     - prepare_gene_symbols            : per-contrast filter, drop unmapped,
#                                         dedup by gene (keep most significant)
#     - build_ranked_series             : GSEA ranking metric (3 modes)
#     - build_protein_term_mapping      : GMT parse -> many-to-many protein/term
#     - _truncate_label / _msigdb_name_to_go_lookup_phrase : string helpers
#     - cluster_go_terms (no-rpy2 path) : graceful degradation to null columns
#     - count_matched_genes             : GSEA gene_set_size from gseapy's
#                                         results record; every
#                                         GseapyResultSchemaError branch
#     - _parse_ora_overlap              : strict ORA 'Overlap' parsing
#     - run_ora (gseapy.enrich stubbed) : size filter, BH over the kept family,
#                                         sorted overlap genes, no error masking
#     - run_gsea (gseapy.prerank stubbed): results-structure raises, Tag %
#                                         forwarding, narrowed exception
#                                         handling, short ranked lists
#     - load_params                     : size filter and permutation validation
#     - _gmt_version / _file_sha256     : gene-set release provenance
#     - write_summary (provenance)      : version, checksum, FDR-scope disclosure
#
#   Not tested here: live gseapy calls, the plot_* functions, main, the
#   per-contrast body of write_summary, and the real rrvgo clustering (needs R +
#   rrvgo + GO.db + org db -> a cluster integration test, like Module 04's R
#   fit). Live gseapy (run_ora, run_gsea) and main() are covered by
#   tests/test_enrichment_gate.py (M05-2, M05-5 .. M05-11).
#
#   Requires gseapy installed: enrichment imports gseapy at module level, so this
#   file fails at collection without it (gseapy is pinned in environment.yml).
#   gseapy's functions are stubbed wherever a test needs controlled output.
#
#   enrichment imports cleanly: rpy2 is loaded lazily inside cluster_go_terms, so
#   importing the module does not start embedded R. The cluster_go_terms no-rpy2
#   degradation test blocks rpy2 for the duration of the call (via monkeypatch)
#   so the lazy import fails with ImportError rather than starting R.
#
# inputs:
#   None (tests build inputs in-memory / in tmp_path).
#
# outputs:
#   Test results (stdout via pytest).
#
# usage example:
#   pytest tests/test_enrichment.py -v
#
#   copy/paste: pytest tests/test_enrichment.py -v

import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from enrichment import (
    CLUSTER_STATUS_CLUSTERED,
    CLUSTER_STATUS_GROUP_ERROR,
    CLUSTER_STATUS_NO_DIRECTION,
    CLUSTER_STATUS_NOT_GO,
    CLUSTER_STATUS_RRVGO_DROPPED,
    CLUSTER_STATUS_TOO_FEW,
    CLUSTER_STATUS_UNAVAILABLE,
    CLUSTER_STATUS_UNRESOLVED,
    GMT_VERSION_UNKNOWN,
    GseapyResultSchemaError,
    _annotate_cluster_unit,
    _clustering_summary_lines,
    _clustering_units,
    _file_sha256,
    _gmt_version,
    _library_short_name,
    _msigdb_name_to_go_lookup_phrase,
    _truncate_label,
    build_protein_term_mapping,
    build_ranked_series,
    cluster_go_terms,
    count_matched_genes,
    gsea_axis_label,
    gsea_direction_statement,
    prepare_gene_symbols,
    warn_unclustered_go_groups,
    write_summary,
)

# ============================================================
# Section 1: _library_short_name
# ============================================================

class TestLibraryShortName:

    def test_go_bp(self):
        assert _library_short_name('gmt/m5.go.bp.v2026.1.Mm.symbols.gmt') == 'GO_BP'

    def test_reactome(self):
        assert _library_short_name('gmt/m2.cp.reactome.v2026.1.Mm.symbols.gmt') == 'REACTOME'

    def test_fallback_uppercases_stem(self):
        # No known substring -> uppercase stem, truncated to 20 chars.
        assert _library_short_name('path/mystery_lib.gmt') == 'MYSTERY_LIB'


# ============================================================
# Section 2: prepare_gene_symbols
# ============================================================

class TestPrepareGeneSymbols:

    def _da(self):
        return pd.DataFrame({
            'contrast':         ['KO_vs_WT', 'KO_vs_WT', 'KO_vs_WT', 'KO_vs_WT', 'X_vs_Y'],
            'gene_symbol':      ['GeneA', 'GeneA', 'GeneB', None, 'GeneC'],
            'deqms_adj_pvalue': [0.01, 0.20, 0.05, 0.04, 0.03],
            'limma_adj_pvalue': [0.02, 0.30, 0.06, 0.05, 0.04],
        })

    def test_filters_by_contrast_and_drops_unmapped(self):
        df, stats = prepare_gene_symbols(self._da(), 'KO_vs_WT')
        assert set(df['gene_symbol']) == {'GeneA', 'GeneB'}   # X_vs_Y + null excluded
        assert stats['n_unmapped'] == 1
        assert stats['n_unique'] == 2

    def test_dedup_keeps_most_significant_per_gene(self):
        df, stats = prepare_gene_symbols(self._da(), 'KO_vs_WT')
        # GeneA appears twice (0.01, 0.20); the 0.01 row is kept.
        gene_a = df[df['gene_symbol'] == 'GeneA']
        assert len(gene_a) == 1
        assert gene_a['deqms_adj_pvalue'].iloc[0] == 0.01
        assert stats['n_collapsed'] == 1

    def test_uses_deqms_pval_when_present(self):
        _df, stats = prepare_gene_symbols(self._da(), 'KO_vs_WT')
        assert stats['pval_col'] == 'deqms_adj_pvalue'

    def test_falls_back_to_limma_when_deqms_all_null(self):
        da = self._da()
        da['deqms_adj_pvalue'] = np.nan
        _df, stats = prepare_gene_symbols(da, 'KO_vs_WT')
        assert stats['pval_col'] == 'limma_adj_pvalue'


# ============================================================
# Section 3: build_ranked_series (GSEA ranking metric)
# ============================================================

class TestBuildRankedSeries:

    def _df(self):
        return pd.DataFrame({
            'gene_symbol':   ['GeneA', 'GeneB'],
            'log2_fc':       [2.0, -1.0],
            'deqms_t':       [5.0, -3.0],
            'limma_t':       [4.0, -2.0],
            'deqms_pvalue':  [1e-3, 1e-2],
            'limma_pvalue':  [1e-2, 1e-1],
        })

    def test_t_statistic_prefers_deqms_t(self):
        rnk = build_ranked_series(self._df(), 't_statistic')
        assert rnk['GeneA'] == 5.0
        assert rnk['GeneB'] == -3.0

    def test_t_statistic_falls_back_to_limma_t(self):
        df = self._df()
        df['deqms_t'] = np.nan
        rnk = build_ranked_series(df, 't_statistic')
        assert rnk['GeneA'] == 4.0

    def test_log2fc_ranking(self):
        rnk = build_ranked_series(self._df(), 'log2fc')
        assert rnk['GeneA'] == 2.0
        assert rnk['GeneB'] == -1.0

    def test_signed_log10p_sign_follows_fold_change(self):
        # GeneA: +fc, p=1e-3 -> +3;  GeneB: -fc, p=1e-2 -> -2
        rnk = build_ranked_series(self._df(), 'signed_log10p')
        assert rnk['GeneA'] == pytest.approx(3.0)
        assert rnk['GeneB'] == pytest.approx(-2.0)

    def test_unknown_ranking_exits(self):
        with pytest.raises(SystemExit):
            build_ranked_series(self._df(), 'bogus')


# ============================================================
# Section 4: build_protein_term_mapping
# ============================================================

class TestBuildProteinTermMapping:

    def _da(self):
        return pd.DataFrame({
            'protein_id':  ['P1', 'P2', 'P3'],
            'gene_symbol': ['GeneA', 'GeneB', 'GeneC'],
            'contrast':    ['KO_vs_WT', 'KO_vs_WT', 'KO_vs_WT'],
            'significant': [True, False, True],
        })

    def test_empty_enrichment_returns_empty_schema(self):
        out = build_protein_term_mapping(
            self._da(), [], [], pd.DataFrame(columns=['term_id']), {},
        )
        assert out.empty
        assert 'in_significant_set' in out.columns
        assert 'is_leading_edge' in out.columns

    def test_maps_annotated_proteins_to_tested_terms(self, tmp_path):
        gmt = tmp_path / 'lib.gmt'
        # Only TERM1 is tested; TERM2 must be ignored.
        gmt.write_text('TERM1\tdesc\tGeneA\tGeneB\nTERM2\tdesc\tGeneZ\n')
        enr = pd.DataFrame({'term_id': ['TERM1']})

        out = build_protein_term_mapping(
            self._da(), [str(gmt)], ['GO_BP'], enr, {},
        ).set_index('gene_symbol')

        assert set(out.index) == {'GeneA', 'GeneB'}          # GeneC/TERM2 excluded
        assert out.loc['GeneA', 'protein_id'] == 'P1'
        assert out.loc['GeneA', 'term_id'] == 'TERM1'
        assert out.loc['GeneA', 'library'] == 'GO_BP'
        # GeneA is significant, GeneB is not.
        assert bool(out.loc['GeneA', 'in_significant_set']) is True
        assert bool(out.loc['GeneB', 'in_significant_set']) is False
        # No GSEA supplied -> not leading edge.
        assert bool(out.loc['GeneA', 'is_leading_edge']) is False


# ============================================================
# Section 5: string helpers
# ============================================================

class TestStringHelpers:

    def test_truncate_label_short_unchanged(self):
        assert _truncate_label('short label') == 'short label'

    def test_truncate_label_long_truncated(self):
        s = 'x' * 80
        out = _truncate_label(s, maxlen=55)
        assert len(out) == 55
        assert out.endswith('...')

    def test_msigdb_gobp_to_phrase(self):
        assert _msigdb_name_to_go_lookup_phrase('GOBP_APOPTOTIC_PROCESS') == 'apoptotic process'

    def test_msigdb_gomf_to_phrase(self):
        assert _msigdb_name_to_go_lookup_phrase('GOMF_DNA_BINDING') == 'dna binding'

    def test_msigdb_no_go_prefix(self):
        # Non-GO term: no prefix stripped, underscores -> spaces, lowercased.
        assert _msigdb_name_to_go_lookup_phrase('REACTOME_SIGNALING') == 'reactome signaling'


# ============================================================
# Section 6: cluster_go_terms -- contract without R (degradation paths)
# ============================================================
# The real rrvgo clustering (needs R) is not run here. Empty / non-GO inputs
# return before the rpy2 import; the GO-library case blocks rpy2 so the lazy
# `import rpy2.robjects` fails cleanly instead of starting R. Every path must
# leave cluster_status set and is_representative non-null (True when unclustered).

class TestClusterGoTermsDegradation:

    def test_empty_input_gets_contract_columns(self):
        out = cluster_go_terms(pd.DataFrame(columns=['library', 'term_id']))
        for col in ('cluster_status', 'cluster_id', 'is_representative', 'parent_term'):
            assert col in out.columns
        assert out.empty

    def test_non_go_library_is_its_own_representative(self):
        enr = pd.DataFrame({'library': ['REACTOME', 'REACTOME'],
                            'term_id': ['R1', 'R2'], 'adj_pvalue': [0.01, 0.02]})
        out = cluster_go_terms(enr)
        assert len(out) == 2                                  # rows preserved
        assert (out['cluster_status'] == CLUSTER_STATUS_NOT_GO).all()
        assert out['is_representative'].notna().all() and out['is_representative'].all()
        assert out['cluster_id'].isna().all()
        assert out['parent_term'].isna().all()

    def test_go_library_without_rpy2_is_unavailable_and_warns(self, monkeypatch, caplog):
        # Block rpy2 for this call so cluster_go_terms' lazy `import rpy2.robjects`
        # raises ImportError (instead of starting embedded R, which would segfault
        # where R is not linked), taking the graceful-degradation path.
        monkeypatch.setitem(sys.modules, 'rpy2', None)
        enr = pd.DataFrame({'library': ['GO_BP', 'GO_BP', 'REACTOME'],
                            'analysis_type': ['ORA'] * 3, 'contrast': ['A_vs_B'] * 3,
                            'term_id': ['GOBP_A', 'GOBP_B', 'R1'], 'adj_pvalue': [0.01, 0.02, 0.03]})
        with caplog.at_level('WARNING'):
            out = cluster_go_terms(enr)
        assert len(out) == 3                                  # input preserved
        assert list(out['cluster_status']) == [CLUSTER_STATUS_UNAVAILABLE] * 2 + [CLUSTER_STATUS_NOT_GO]
        assert out['is_representative'].all()
        assert out['cluster_id'].isna().all()
        # The failure check fires: zero clusters must be announced, not silent.
        assert 'produced NO clusters for GO_BP/ORA/A_vs_B' in caplog.text


# ============================================================
# Section 6b: _annotate_cluster_unit (pure-Python clustering core)
# ============================================================
# r_out mimics the R block's outputs, so status assignment, representative
# choice and parent_term are tested without R.

def _unit(term_ids, adj_p, nes=None, overlap=None) -> pd.DataFrame:
    df = pd.DataFrame({'term_id': term_ids, 'adj_pvalue': adj_p})
    if nes is not None:
        df['enrichment_score'] = nes
    if overlap is not None:
        df['overlap_size'] = overlap
    return df


def _r_out(stage='ok', resolved=None, sim_go=None, red=None) -> dict:
    return {'stage': stage, 'error': '', 'resolved': resolved or {},
            'sim_go': set(sim_go or ()), 'red': red or {}}


class TestAnnotateClusterUnit:

    def test_each_status_from_ok_stage(self):
        unit = _unit(['A', 'B', 'C', 'D', 'E'], [0.01, 0.02, 0.03, 0.04, 0.05], overlap=[5] * 5)
        r_out = _r_out(
            resolved={'A': 'GO:1', 'B': 'GO:2', 'C': 'GO:3', 'D': 'GO:4'},   # E unresolved
            sim_go={'GO:1', 'GO:2', 'GO:3'},                                   # GO:4 dropped by rrvgo
            red={'GO:1': 1, 'GO:2': 1, 'GO:3': 2},
        )
        res = _annotate_cluster_unit(unit, 'ORA', r_out)
        assert list(res['cluster_status']) == [
            CLUSTER_STATUS_CLUSTERED, CLUSTER_STATUS_CLUSTERED, CLUSTER_STATUS_CLUSTERED,
            CLUSTER_STATUS_RRVGO_DROPPED, CLUSTER_STATUS_UNRESOLVED,
        ]
        assert res['cluster_local'].tolist()[:3] == [1, 1, 2]
        assert res['cluster_local'].iloc[3:].isna().all()
        # A is the cluster-1 representative (lowest p); C is a singleton cluster.
        assert list(res['is_representative']) == [True, False, True, True, True]
        assert res.at[1, 'parent_term'] == 'A'
        assert res['parent_term'].isna().sum() == 4

    @pytest.mark.parametrize(('stage', 'status_a', 'status_b'), [
        # A: resolved and kept by calculateSimMatrix; B: resolved but dropped by it.
        ('too_few',       CLUSTER_STATUS_TOO_FEW,     CLUSTER_STATUS_TOO_FEW),
        ('sim_error',     CLUSTER_STATUS_GROUP_ERROR, CLUSTER_STATUS_GROUP_ERROR),
        ('sim_too_small', CLUSTER_STATUS_TOO_FEW,     CLUSTER_STATUS_RRVGO_DROPPED),
        ('reduce_error',  CLUSTER_STATUS_GROUP_ERROR, CLUSTER_STATUS_RRVGO_DROPPED),
    ])
    def test_failure_stages(self, stage, status_a, status_b):
        unit = _unit(['A', 'B', 'C'], [0.01, 0.02, 0.03])            # C unresolved
        sim_go = {'GO:1'} if stage in ('sim_too_small', 'reduce_error') else set()
        r_out = _r_out(stage, resolved={'A': 'GO:1', 'B': 'GO:2'}, sim_go=sim_go)
        res = _annotate_cluster_unit(unit, 'ORA', r_out)
        assert list(res['cluster_status']) == [status_a, status_b, CLUSTER_STATUS_UNRESOLVED]
        assert res['is_representative'].all()
        assert res['cluster_local'].isna().all()

    def test_r_error_marks_whole_unit_group_error(self):
        res = _annotate_cluster_unit(_unit(['A', 'B'], [0.01, 0.02]), 'ORA', None)
        assert (res['cluster_status'] == CLUSTER_STATUS_GROUP_ERROR).all()
        assert res['is_representative'].all()

    @pytest.mark.parametrize(('atype', 'unit', 'expected_rep'), [
        # Tied adj_pvalue: GSEA breaks the tie on |NES| ...
        ('GSEA', _unit(['A', 'B', 'C'], [0.0, 0.0, 0.0], nes=[1.5, -2.5, 2.0]), 'B'),
        # ... ORA on overlap_size ...
        ('ORA',  _unit(['A', 'B', 'C'], [0.01, 0.01, 0.01], overlap=[3, 9, 9]), 'B'),
        # ... and full ties fall back to term_id.
        ('ORA',  _unit(['Z', 'M', 'B'], [0.01, 0.01, 0.01], overlap=[4, 4, 4]), 'B'),
        # Missing adj_pvalue sorts last.
        ('ORA',  _unit(['A', 'B'], [np.nan, 0.5], overlap=[9, 1]), 'B'),
    ])
    def test_representative_tie_break(self, atype, unit, expected_rep):
        gos = {t: f'GO:{i}' for i, t in enumerate(unit['term_id'])}
        r_out = _r_out(resolved=gos, sim_go=gos.values(), red=dict.fromkeys(gos.values(), 7))
        for ordered in (unit, unit.iloc[::-1]):              # row order must not matter
            res = _annotate_cluster_unit(ordered, atype, r_out)
            assert ordered.loc[res['is_representative'], 'term_id'].tolist() == [expected_rep]
            assert (res.loc[~res['is_representative'], 'parent_term'] == expected_rep).all()


class TestClusteringUnits:

    def test_ora_is_one_unit(self):
        g = pd.DataFrame({'term_id': ['A', 'B'], 'enrichment_score': [np.nan, np.nan]})
        units, no_dir = _clustering_units(g, 'ORA')
        assert [lbl for lbl, _ in units] == ['all']
        assert len(units[0][1]) == 2
        assert len(no_dir) == 0

    def test_gsea_splits_by_nes_sign(self):
        g = pd.DataFrame({'term_id': list('ABCDE'),
                          'enrichment_score': [2.0, -1.5, 0.0, np.nan, 1.1]})
        units, no_dir = _clustering_units(g, 'GSEA')
        assert {lbl: set(u['term_id']) for lbl, u in units} == {'NES>0': {'A', 'E'}, 'NES<0': {'B'}}
        assert set(g.loc[no_dir, 'term_id']) == {'C', 'D'}


class TestClusterGoTermsOrchestration:
    """
    Drive cluster_go_terms end to end with a fake rpy2 and a stubbed R call, to
    test GSEA direction splitting, one cluster_id sequence per group, and the
    no-direction / too-few paths, without R.
    """

    def _fake_rpy2(self, monkeypatch, red_by_unit):
        import types

        import enrichment
        rpy2 = types.ModuleType('rpy2')
        robjects = types.ModuleType('rpy2.robjects')
        packages = types.ModuleType('rpy2.robjects.packages')
        packages.importr = lambda name: None
        monkeypatch.setitem(sys.modules, 'rpy2', rpy2)
        monkeypatch.setitem(sys.modules, 'rpy2.robjects', robjects)
        monkeypatch.setitem(sys.modules, 'rpy2.robjects.packages', packages)
        calls = []

        def fake_run(ro, unit, ont, orgdb, threshold):
            # Mimics the R block: names starting with 'X' do not resolve, and
            # fewer than 2 resolved GO IDs ends at stage 'too_few'.
            calls.append(list(unit['term_id']))
            gos = {t: 'GO:' + t for t in unit['term_id'] if not t.startswith('X')}
            if len(gos) < 2:
                return _r_out('too_few', resolved=gos)
            red = {'GO:' + t: c for t, c in red_by_unit(unit).items() if t in gos}
            return _r_out(resolved=gos, sim_go=gos.values(), red=red)

        monkeypatch.setattr(enrichment, '_run_rrvgo_unit', fake_run)
        return calls

    def test_gsea_direction_units_and_cluster_numbering(self, monkeypatch):
        enr = pd.DataFrame({
            'library': ['GO_BP'] * 6, 'analysis_type': ['GSEA'] * 6, 'contrast': ['A_vs_B'] * 6,
            'term_id': ['U1', 'U2', 'U3', 'D1', 'D2', 'Z'],
            'adj_pvalue': [0.01, 0.02, 0.03, 0.01, 0.02, 0.5],
            'enrichment_score': [2.0, 1.8, 1.2, -2.0, -1.9, 0.0],
        })
        # rrvgo numbers clusters from 1 in every unit: up {U1,U2}=1, {U3}=2; down {D1,D2}=1.
        calls = self._fake_rpy2(
            monkeypatch, lambda u: {t: (2 if t == 'U3' else 1) for t in u['term_id']})
        out = cluster_go_terms(enr).set_index('term_id')
        assert calls == [['U1', 'U2', 'U3'], ['D1', 'D2']]   # Z (NES 0) never reaches R
        assert out.at['Z', 'cluster_status'] == CLUSTER_STATUS_NO_DIRECTION
        # Up and down clusters share one sequence: 1, 2 (up), 3 (down). No collision.
        assert out.loc[['U1', 'U2', 'U3', 'D1', 'D2'], 'cluster_id'].tolist() == [1, 1, 2, 3, 3]
        assert out.loc[['U2', 'D2'], 'parent_term'].tolist() == ['U1', 'D1']
        assert out['is_representative'].tolist() == [True, False, True, True, False, True]

    def test_single_term_unit_resolves_name_then_too_few(self, monkeypatch):
        # Single-row units still go through R so an unresolvable name is reported
        # as unresolved_name, not hidden behind too_few_terms.
        # Each down unit holds a single row: D1 (resolvable) in contrast A_vs_B,
        # X_BAD (unresolvable) in C_vs_D.
        enr = pd.DataFrame({
            'library': ['GO_BP'] * 6, 'analysis_type': ['GSEA'] * 6,
            'contrast': ['A_vs_B'] * 3 + ['C_vs_D'] * 3,
            'term_id': ['U1', 'U2', 'D1', 'U1', 'U2', 'X_BAD'],
            'adj_pvalue': [0.01, 0.02, 0.03, 0.01, 0.02, 0.04],
            'enrichment_score': [2.0, 1.5, -1.0, 2.0, 1.5, -2.0],
        })
        self._fake_rpy2(monkeypatch, lambda u: dict.fromkeys(u['term_id'], 1))
        out = cluster_go_terms(enr)
        status = dict(zip(out['contrast'] + '/' + out['term_id'], out['cluster_status'], strict=True))
        assert status['A_vs_B/D1'] == CLUSTER_STATUS_TOO_FEW
        assert status['C_vs_D/X_BAD'] == CLUSTER_STATUS_UNRESOLVED
        assert out.loc[out['term_id'].isin(['D1', 'X_BAD']), 'is_representative'].all()

    def test_nan_group_key_still_clustered(self, monkeypatch):
        # A GO row with a missing contrast must not keep the pre-set
        # 'clustering_unavailable' status when R ran (groupby dropna=False).
        enr = pd.DataFrame({
            'library': ['GO_BP'] * 2, 'analysis_type': ['ORA'] * 2, 'contrast': [np.nan] * 2,
            'term_id': ['A', 'B'], 'adj_pvalue': [0.01, 0.02],
        })
        self._fake_rpy2(monkeypatch, lambda u: dict.fromkeys(u['term_id'], 1))
        out = cluster_go_terms(enr)
        assert (out['cluster_status'] == CLUSTER_STATUS_CLUSTERED).all()


class TestClusteringSummaryAndWarning:

    def _cdf(self) -> pd.DataFrame:
        return pd.DataFrame({
            'library':        ['GO_BP'] * 6 + ['REACTOME'],
            'analysis_type':  ['ORA'] * 4 + ['GSEA'] * 2 + ['ORA'],
            'contrast':       ['A_vs_B'] * 7,
            'cluster_status': [CLUSTER_STATUS_CLUSTERED, CLUSTER_STATUS_CLUSTERED,
                               CLUSTER_STATUS_UNRESOLVED, CLUSTER_STATUS_RRVGO_DROPPED,
                               CLUSTER_STATUS_GROUP_ERROR, CLUSTER_STATUS_GROUP_ERROR,
                               CLUSTER_STATUS_NOT_GO],
            'cluster_id':     pd.array([1, 1, None, None, None, None, None], dtype='Int64'),
        })

    def test_summary_counts_all_terms_and_flags_unclustered(self):
        text = '\n'.join(_clustering_summary_lines(self._cdf()))
        # All 4 ORA terms counted (the old summary counted only the 2 clustered ones).
        assert '4 terms: 2 clustered (50.0%) -> 1 clusters; 75.0% names resolved' in text
        assert 'unresolved_name=1' in text
        assert 'dropped_by_rrvgo=1' in text
        assert 'WARNING: no GO_BP GSEA terms were clustered' in text
        # GSEA rows are group_error: the rate cannot be read from status, so none is printed.
        assert 'GSEA       2 terms: 0 clustered (0.0%) -> 0 clusters; name resolution rate unavailable' in text
        assert 'not clustered (not a GO library)' in text
        assert 'threshold=0.7' in text

    def test_warn_skips_frames_without_group_columns(self):
        assert warn_unclustered_go_groups(pd.DataFrame({
            'library': ['GO_BP'], 'cluster_status': [CLUSTER_STATUS_UNAVAILABLE]})) == []

    def test_warn_unclustered_go_groups(self, caplog):
        with caplog.at_level('WARNING'):
            groups = warn_unclustered_go_groups(self._cdf())
        assert groups == [('GO_BP', 'GSEA', 'A_vs_B')]       # ORA had clusters; REACTOME is not GO
        assert 'NOT collapsed' in caplog.text


# ============================================================
# Section: GSEA direction statement
# ============================================================

class TestGseaDirectionStatement:

    def test_names_numerator_as_positive_nes(self):
        txt = gsea_direction_statement('KO_vs_WT')
        assert 'NES > 0 = enriched among genes higher in KO' in txt
        assert 'NES < 0 = higher in WT' in txt

    def test_reversed_label_flips(self):
        assert 'higher in WT' in gsea_direction_statement('WT_vs_KO').split(';')[0]

    @pytest.mark.parametrize('bad', ['KOWT', '_vs_WT', 'KO_vs_'])
    def test_malformed_label_falls_back(self, bad):
        assert gsea_direction_statement(bad) == (
            'NES > 0 = enriched among genes with positive log2 FC')


    def test_axis_label_is_short_and_names_numerator(self):
        assert gsea_axis_label('CTXcyto_KO_vs_CTXcyto_WT') == 'NES (> 0 = higher in CTXcyto_KO)'
        assert gsea_axis_label('KOWT') == 'NES (> 0 = positive log2 FC)'


# ============================================================
# Section: count_matched_genes (GSEA gene_set_size, KNOWN_ISSUES E-9)
# ============================================================

class TestCountMatchedGenes:
    '''
    count_matched_genes turns gseapy's per-term results record into a gene
    count. gseapy 1.1.13 stores matched_genes as a ';'-joined str, so len() on
    the raw value counts characters; every structural surprise must raise
    GseapyResultSchemaError naming the term and the installed gseapy version.
    '''

    _GENES = 'Snap25;Syt1;Gfap;Aqp4'

    def _raises(self, term_data, tag_pct=None, match='TERM_X'):
        import gseapy
        with pytest.raises(GseapyResultSchemaError) as exc:
            count_matched_genes('TERM_X', term_data, tag_pct)
        msg = str(exc.value)
        assert match in msg
        assert gseapy.__version__ in msg
        return msg

    # --- Valid inputs ---

    def test_str_counts_genes_not_characters(self):
        assert count_matched_genes('TERM_X', {'matched_genes': self._GENES}) == 4

    def test_str_ignores_empty_and_whitespace_tokens(self):
        assert count_matched_genes('TERM_X', {'matched_genes': 'Snap25;; ;Syt1;'}) == 2

    def test_list_ignores_blank_entries_like_str(self):
        # The list branch applies the same rule as the str branch.
        assert count_matched_genes('TERM_X', {'matched_genes': ['Snap25', '', ' ', 'Syt1']}) == 2
        assert count_matched_genes('TERM_X', {'matched_genes': np.array(['Snap25', ''])}) == 1

    def test_list_and_ndarray_are_counted(self):
        genes = self._GENES.split(';')
        assert count_matched_genes('TERM_X', {'matched_genes': genes}) == 4
        assert count_matched_genes('TERM_X', {'matched_genes': np.array(genes)}) == 4

    @pytest.mark.parametrize('tag_pct', ['2/4', ' 2 / 4 ', '2/4\n'])
    def test_matching_tag_pct_passes(self, tag_pct):
        # Surrounding whitespace is allowed around each number.
        assert count_matched_genes('TERM_X', {'matched_genes': self._GENES}, tag_pct) == 4

    @pytest.mark.parametrize('tag_pct', [None, '', '   ', float('nan'), np.float32('nan'),
                                         np.float64('nan'), pd.NA, pd.NaT])
    def test_absent_tag_pct_is_skipped(self, tag_pct):
        # Absent Tag %: run_gsea passes None when gseapy omits the column
        # (permutation_num == 0); NaN and blank are treated the same way
        assert count_matched_genes('TERM_X', {'matched_genes': self._GENES}, tag_pct) == 4

    # --- Structural errors ---

    def test_non_dict_record_raises(self):
        msg = self._raises(['Snap25'])
        assert 'expected dict' in msg

    @pytest.mark.parametrize('empty', ['', ';;', ' ; ', [], ['', ' ']])
    def test_empty_matched_set_raises(self, empty):
        # gseapy never reports a term with 0 matched genes (min_size filter)
        msg = self._raises({'matched_genes': empty})
        assert 'no gene symbols' in msg

    @pytest.mark.parametrize('tag_pct', ['0.25', '2 of 5', '25%', '2/5/7', '2/4.0', '/4', '2/\u2074',
                                         '\u0662/\u0664', '\uff12/\uff14'])
    def test_unparseable_tag_pct_raises(self, tag_pct):
        # A present but non-'a/b' Tag % means the format changed; it must not
        # silently disable the cross-check.
        msg = self._raises({'matched_genes': self._GENES}, tag_pct=tag_pct)
        assert 'expected' in msg and tag_pct in msg

    def test_missing_matched_genes_raises_and_lists_fields(self):
        msg = self._raises({'lead_genes': 'Snap25', 'hits': [0]})
        assert "no 'matched_genes'" in msg
        assert 'lead_genes' in msg

    @pytest.mark.parametrize('bad', [4, {'Snap25': 1}, None])
    def test_unsupported_type_raises_naming_type(self, bad):
        msg = self._raises({'matched_genes': bad})
        assert type(bad).__name__ in msg

    def test_tag_pct_mismatch_raises_naming_both_counts(self):
        msg = self._raises({'matched_genes': self._GENES}, tag_pct='2/5')
        assert 'gives 4 genes' in msg and 'gives 5' in msg

    def test_fixture_len_of_str_differs_from_gene_count(self):
        # Fixture check: on this record the pre-fix logic (len of the raw
        # value) gives a different number, so a regression to it would fail
        # the counting tests above.
        assert len(self._GENES) != count_matched_genes('TERM_X', {'matched_genes': self._GENES})


class TestRunGseaResultsStructure:
    '''
    run_gsea's own GseapyResultSchemaError paths, around the per-term lookup:
    .results not a dict, and a res2d term missing from .results (with a hint
    when .results is nested by ranking column, as gseapy does for multi-column
    input). Also checks that res2d 'Tag %' is forwarded to count_matched_genes
    (a disagreeing Tag % must raise), plus a fixture check that the well-formed
    stub passes. gseapy.prerank is replaced by a stub returning a crafted result.
    '''

    _TERM = 'TERM_X'
    _RECORD: ClassVar[dict] = {'matched_genes': 'Snap25;Syt1;Gfap;Aqp4'}

    def _run(self, monkeypatch, results, tag_pct='2/4'):
        import types

        import enrichment
        res2d = pd.DataFrame({
            'Term': [self._TERM], 'NES': [1.2], 'NOM p-val': [0.01],
            'FDR q-val': [0.02], 'Tag %': [tag_pct], 'Lead_genes': ['Snap25;Syt1'],
        })
        stub = types.SimpleNamespace(res2d=res2d, results=results)
        monkeypatch.setattr(enrichment.gseapy, 'prerank', lambda **kwargs: stub)
        params = {'enrichment': {
            'fdr_threshold': 0.05, 'min_gene_set_size': 1, 'max_gene_set_size': 500,
            'gsea_permutations': 10, 'gsea_seed': 42,
        }}
        ranked = pd.Series([2.0, 1.0, -1.0], index=['Snap25', 'Syt1', 'Gfap'])
        return enrichment.run_gsea(ranked, 'unused.gmt', 'GO_BP', 'KO_vs_WT', params)

    def _raises(self, monkeypatch, results) -> str:
        import gseapy
        with pytest.raises(GseapyResultSchemaError) as exc:
            self._run(monkeypatch, results)
        msg = str(exc.value)
        assert gseapy.__version__ in msg
        return msg

    def test_well_formed_stub_passes(self, monkeypatch):
        # Fixture check: the stub itself is valid, so the raises below are
        # caused by the structure under test, not by the stub.
        df, _ = self._run(monkeypatch, {self._TERM: self._RECORD})
        assert int(df.loc[0, 'gene_set_size']) == 4

    def test_tag_pct_is_forwarded_to_cross_check(self, monkeypatch):
        # res2d says 5 matched genes, the record has 4: run_gsea must pass
        # Tag % through to count_matched_genes so the mismatch raises.
        with pytest.raises(GseapyResultSchemaError, match='gives 5'):
            self._run(monkeypatch, {self._TERM: self._RECORD}, tag_pct='2/5')

    def test_results_not_dict_raises(self, monkeypatch):
        msg = self._raises(monkeypatch, [self._RECORD])
        assert '.results is list, expected dict' in msg

    def test_missing_term_raises_without_nested_hint(self, monkeypatch):
        msg = self._raises(monkeypatch, {'OTHER_TERM': self._RECORD})
        assert self._TERM in msg and 'missing from .results' in msg
        assert 'nested' not in msg

    def test_missing_term_in_nested_results_gives_hint(self, monkeypatch):
        nested = {'rank_a': {self._TERM: self._RECORD}, 'rank_b': {self._TERM: self._RECORD}}
        msg = self._raises(monkeypatch, nested)
        assert self._TERM in msg and 'nested by ranking column' in msg

    def test_empty_results_raises_without_nested_hint(self, monkeypatch):
        msg = self._raises(monkeypatch, {})
        assert 'missing from .results' in msg and 'nested' not in msg


# ============================================================
# Section: ORA size filter, Overlap parsing, sorted overlap genes
# ============================================================

class TestParseOraOverlap:
    '''_parse_ora_overlap reads gseapy.enrich's '<hits>/<set size>' strictly.'''

    @pytest.mark.parametrize('overlap', ['3/20', ' 3 / 20 ', '3/20\n'])
    def test_parses_hits_and_size(self, overlap):
        # Surrounding whitespace is allowed around each number.
        from enrichment import _parse_ora_overlap
        assert _parse_ora_overlap('TERM_X', overlap) == (3, 20)

    # Non-ASCII digits: superscript two, Arabic-Indic and fullwidth decimals.
    @pytest.mark.parametrize('bad', ['3', '3/', '3/20/1', 'a/20', '3/2.5', None, float('nan'), '3/\u00b2',
                                     '\u0663/\u0662\u0660', '\uff13/\uff12\uff10'])
    def test_unparseable_raises(self, bad):
        from enrichment import _parse_ora_overlap
        with pytest.raises(GseapyResultSchemaError, match='TERM_X'):
            _parse_ora_overlap('TERM_X', bad)


class TestRunOraSizeFilter:
    '''
    run_ora keeps only terms whose background-intersected size (the Overlap
    denominator) lies in [min_gene_set_size, max_gene_set_size], and recomputes
    BH over the kept terms. gseapy.enrich is stubbed so sizes and p-values are
    exact. Real-gseapy coverage is gate M05-7.
    '''

    # term -> (Overlap, raw p-value, gseapy's own adjusted p, genes)
    # The stub's own adjusted p (0.9) is deliberately not a BH value, so any
    # test that finds a BH q in the output shows run_ora recomputed it.
    _ROWS: ClassVar[dict] = {
        'TOO_SMALL': ('2/10', 0.001, 0.9, 'Gfap;Aqp4'),
        'IN_RANGE_A': ('5/20', 0.010, 0.9, 'Syt1;Snap25;Aqp4;Gfap;Actb'),
        'IN_RANGE_B': ('4/400', 0.030, 0.9, 'Vamp2;Actb;Syt1;Gapdh'),
        'TOO_LARGE': ('9/600', 0.040, 0.9, 'Actb;Gapdh'),
    }

    def _run(self, monkeypatch, min_size=15, max_size=500):
        import types

        import enrichment
        res2d = pd.DataFrame({
            'Term': list(self._ROWS),
            'Overlap': [r[0] for r in self._ROWS.values()],
            'P-value': [r[1] for r in self._ROWS.values()],
            'Adjusted P-value': [r[2] for r in self._ROWS.values()],
            'Odds Ratio': 2.0, 'Combined Score': 5.0,
            'Genes': [r[3] for r in self._ROWS.values()],
        })
        monkeypatch.setattr(enrichment.gseapy, 'enrich',
                            lambda **kwargs: types.SimpleNamespace(res2d=res2d))
        params = {'enrichment': {'fdr_threshold': 0.05, 'min_gene_set_size': min_size,
                                 'max_gene_set_size': max_size}}
        out = enrichment.run_ora(['Syt1'], ['Syt1', 'Actb'], 'unused.gmt',
                                 'GO_BP', 'KO_vs_WT', params)
        return out if out.empty else out.set_index('term_id')

    def test_only_in_range_terms_kept(self, monkeypatch):
        out = self._run(monkeypatch)
        assert list(out.index) == ['IN_RANGE_A', 'IN_RANGE_B']
        assert list(out['gene_set_size']) == [20, 400]
        assert list(out['overlap_size']) == [5, 4]

    def test_bh_recomputed_over_kept_family(self, monkeypatch):
        # BH over p = [0.01, 0.03] (m = 2): q = [0.02, 0.03]
        out = self._run(monkeypatch)
        assert out['adj_pvalue'].tolist() == pytest.approx([0.02, 0.03])

    def test_fixture_unfiltered_family_gives_different_q(self, monkeypatch):
        # Fixture check: with no term excluded, BH over all 4 p-values gives
        # IN_RANGE_B q = 0.04 instead of 0.03, so on this fixture the family
        # matters and the filtered-family test above is not a no-op. The stub's
        # adjusted p is 0.9, so 0.04 can only come from recomputation.
        out = self._run(monkeypatch, min_size=1, max_size=10_000)
        assert len(out) == 4
        assert out.loc['IN_RANGE_B', 'adj_pvalue'] == pytest.approx(0.04)

    def test_bounds_are_inclusive(self, monkeypatch):
        # Terms sized exactly min (20) and max (400) are kept.
        out = self._run(monkeypatch, min_size=20, max_size=400)
        assert list(out.index) == ['IN_RANGE_A', 'IN_RANGE_B']

    def test_all_terms_filtered_returns_empty(self, monkeypatch):
        assert self._run(monkeypatch, min_size=700, max_size=800).empty

    def test_overlap_genes_sorted(self, monkeypatch):
        out = self._run(monkeypatch)
        assert out.loc['IN_RANGE_A', 'overlap_genes'] == 'Actb;Aqp4;Gfap;Snap25;Syt1'


class TestLoadParamsValidation:
    '''load_params rejects size-filter and permutation values that break a run.'''

    def _write(self, tmp_path, **enrichment_overrides) -> str:
        import yaml
        gmt = tmp_path / 'lib.gmt'
        gmt.write_text('TERM\tdesc\tSyt1\n')
        enr = {'gene_set_libraries': [str(gmt)], **enrichment_overrides}
        path = tmp_path / 'params.yml'
        path.write_text(yaml.safe_dump({'enrichment': enr}))
        return str(path)

    def test_defaults_accepted(self, tmp_path):
        from enrichment import load_params
        enr = load_params(self._write(tmp_path))['enrichment']
        assert enr['gsea_permutations'] == 1000

    @pytest.mark.parametrize('bad', [0, -1, 1.5, True, '100'])
    def test_bad_permutations_exit(self, tmp_path, bad):
        from enrichment import load_params
        with pytest.raises(SystemExit):
            load_params(self._write(tmp_path, gsea_permutations=bad))

    def test_permutations_ignored_when_gsea_off(self, tmp_path):
        from enrichment import load_params
        load_params(self._write(tmp_path, gsea_permutations=0, run_gsea=False))

    def test_one_permutation_accepted(self, tmp_path):
        from enrichment import load_params
        load_params(self._write(tmp_path, gsea_permutations=1))

    @pytest.mark.parametrize('sizes', [(0, 500), (20, 10), (15.0, 500), (15, None)])
    def test_bad_size_filter_exit(self, tmp_path, sizes):
        from enrichment import load_params
        with pytest.raises(SystemExit):
            load_params(self._write(tmp_path, min_gene_set_size=sizes[0],
                                    max_gene_set_size=sizes[1]))



class TestLoadParamsStagedLibraries:
    '''
    GMT resolution (KNOWN_ISSUES E-4). Standalone: relative entries resolve
    against the params.yml directory. Nextflow: --gene-set-libraries supplies
    the staged files, which replace the params.yml paths after a one-to-one
    (count, order, filename) check.
    '''

    def _params(self, tmp_path, libraries) -> str:
        import yaml
        path = tmp_path / 'params.yml'
        path.write_text(yaml.safe_dump({'enrichment': {'gene_set_libraries': libraries}}))
        return str(path)

    def _gmt(self, directory, name) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        gmt = directory / name
        gmt.write_text('TERM\tdesc\tSyt1\n')
        return gmt

    def test_relative_entry_resolves_against_params_dir(self, tmp_path, monkeypatch):
        from enrichment import load_params
        gmt = self._gmt(tmp_path, 'example_gene_sets.gmt')
        monkeypatch.chdir(tmp_path.parent)   # cwd must not matter
        enr = load_params(self._params(tmp_path, ['example_gene_sets.gmt']))['enrichment']
        assert enr['gene_set_libraries'] == [str(gmt.resolve())]

    def test_staged_files_replace_unreachable_params_paths(self, tmp_path, monkeypatch):
        # Inside a Nextflow task the params.yml path does not exist; only the
        # staged copy does.
        from enrichment import load_params
        work = tmp_path / 'work'
        staged = self._gmt(work / 'gmt' / '1', 'm5.go.bp.v2026.1.Mm.symbols.gmt')
        params = self._params(tmp_path, ['/nonexistent/m5.go.bp.v2026.1.Mm.symbols.gmt'])
        monkeypatch.chdir(work)
        enr = load_params(params, staged_libraries=['gmt/1/m5.go.bp.v2026.1.Mm.symbols.gmt'])
        assert enr['enrichment']['gene_set_libraries'] == [str(staged.absolute())]

    def test_staged_order_preserved(self, tmp_path):
        from enrichment import load_params
        a = self._gmt(tmp_path / 'gmt' / '1', 'a.gmt')
        b = self._gmt(tmp_path / 'gmt' / '2', 'b.gmt')
        enr = load_params(self._params(tmp_path, ['x/a.gmt', 'y/b.gmt']),
                          staged_libraries=[str(a), str(b)])['enrichment']
        assert [Path(p).name for p in enr['gene_set_libraries']] == ['a.gmt', 'b.gmt']

    def test_staged_count_mismatch_exits(self, tmp_path):
        from enrichment import load_params
        a = self._gmt(tmp_path / 'gmt' / '1', 'a.gmt')
        with pytest.raises(SystemExit):
            load_params(self._params(tmp_path, ['a.gmt', 'b.gmt']), staged_libraries=[str(a)])

    def test_staged_order_swapped_exits(self, tmp_path):
        # NEGATIVE CONTROL: a misordered list would silently swap library labels.
        from enrichment import load_params
        a = self._gmt(tmp_path / 'gmt' / '1', 'a.gmt')
        b = self._gmt(tmp_path / 'gmt' / '2', 'b.gmt')
        with pytest.raises(SystemExit):
            load_params(self._params(tmp_path, ['a.gmt', 'b.gmt']),
                        staged_libraries=[str(b), str(a)])

    def test_padded_entry_matches_staged_file(self, tmp_path):
        # workflows/prosift.nf trims each entry before staging; load_params must
        # compare against the trimmed name too, or a quoted ' a.gmt ' exits.
        from enrichment import load_params
        a = self._gmt(tmp_path / 'gmt' / '1', 'a.gmt')
        enr = load_params(self._params(tmp_path, [' a.gmt ']),
                          staged_libraries=[str(a)])['enrichment']
        assert enr['gene_set_libraries'] == [str(a.absolute())]

    def test_padded_entry_resolves_standalone(self, tmp_path):
        from enrichment import load_params
        gmt = self._gmt(tmp_path, 'a.gmt')
        enr = load_params(self._params(tmp_path, ['  a.gmt\t']))['enrichment']
        assert enr['gene_set_libraries'] == [str(gmt.resolve())]

    @pytest.mark.parametrize('blank', ['', '   ', '\t'])
    def test_blank_entry_exits(self, tmp_path, blank):
        # A blank entry trims to '' and resolves to the params directory itself;
        # it must fail as 'not a file', not reach gseapy.
        from enrichment import load_params
        with pytest.raises(SystemExit):
            load_params(self._params(tmp_path, [blank]))

    def test_trim_matches_java_string_trim(self, tmp_path):
        # Parity with Groovy trim() in workflows/prosift.nf: chars <= U+0020
        # (incl. control chars) are stripped, Unicode whitespace (NBSP) is not.
        from enrichment import _java_trim, load_params
        assert _java_trim('\x01 a.gmt\r\n') == 'a.gmt'
        assert _java_trim('\u00a0a.gmt') == '\u00a0a.gmt'
        gmt = self._gmt(tmp_path, 'a.gmt')
        enr = load_params(self._params(tmp_path, ['\x01a.gmt']))['enrichment']
        assert enr['gene_set_libraries'] == [str(gmt.resolve())]
        with pytest.raises(SystemExit):   # NBSP kept -> '\u00a0a.gmt' does not exist
            load_params(self._params(tmp_path, ['\u00a0a.gmt']))

    def test_staged_missing_file_exits(self, tmp_path):
        from enrichment import load_params
        with pytest.raises(SystemExit):
            load_params(self._params(tmp_path, ['a.gmt']),
                        staged_libraries=[str(tmp_path / 'gmt' / '1' / 'a.gmt')])

    def test_shipped_example_params_has_no_absolute_gmt_path(self):
        # E-4 regression guard: the example must not pin a machine-specific path.
        import yaml
        example = Path(__file__).resolve().parent.parent / 'assets' / 'examples' / 'minimal'
        libs = yaml.safe_load((example / 'params.yml').read_text())['enrichment']['gene_set_libraries']
        for entry in libs:
            assert not Path(entry).is_absolute(), entry
            assert (example / entry).exists(), entry

class TestGseapyExceptionHandling:
    '''
    run_ora has no try/except around gseapy.enrich (empty results come back as
    res2d = None). run_gsea handles only gseapy's plain LookupError "No gene sets
    passed through filtering condition"; every other exception, including
    LookupError subclasses (KeyError, IndexError) and a LookupError with another
    message, propagates. A ranked list shorter than 2 genes is skipped without
    calling gseapy.
    '''

    _PARAMS: ClassVar[dict] = {'enrichment': {
        'fdr_threshold': 0.05, 'min_gene_set_size': 5, 'max_gene_set_size': 500,
        'gsea_permutations': 10, 'gsea_seed': 42,
    }}
    _RANKED = pd.Series([2.0, 1.0, -1.0], index=['Snap25', 'Syt1', 'Gfap'])

    def _patch(self, monkeypatch, name, exc):
        import enrichment

        def _raise(**kwargs):
            raise exc
        monkeypatch.setattr(enrichment.gseapy, name, _raise)
        return enrichment

    def test_ora_gseapy_error_propagates(self, monkeypatch):
        enrichment = self._patch(monkeypatch, 'enrich', TypeError('unexpected keyword'))
        with pytest.raises(TypeError):
            enrichment.run_ora(['Syt1'], ['Syt1', 'Actb'], 'unused.gmt', 'GO_BP',
                               'KO_vs_WT', self._PARAMS)

    def test_gsea_no_sets_lookuperror_is_empty_result(self, monkeypatch):
        enrichment = self._patch(monkeypatch, 'prerank', LookupError(
            'No gene sets passed through filtering condition !!! Hint 1: ...'))
        df, pre_res = enrichment.run_gsea(self._RANKED, 'unused.gmt', 'GO_BP',
                                          'KO_vs_WT', self._PARAMS)
        assert df.empty and pre_res is None

    @pytest.mark.parametrize('exc', [
        LookupError('some other lookup failure'),
        KeyError('No gene sets passed through filtering condition'),
        IndexError('No gene sets passed through filtering condition'),
        RuntimeError('rust backend failed'),
        TypeError('unexpected keyword'),
    ])
    def test_gsea_other_errors_propagate(self, monkeypatch, exc):
        enrichment = self._patch(monkeypatch, 'prerank', exc)
        with pytest.raises(type(exc)):
            enrichment.run_gsea(self._RANKED, 'unused.gmt', 'GO_BP', 'KO_vs_WT', self._PARAMS)

    @pytest.mark.parametrize('n', [0, 1])
    def test_gsea_short_ranked_list_skipped_without_gseapy(self, monkeypatch, n):
        enrichment = self._patch(monkeypatch, 'prerank', AssertionError('prerank must not be called'))
        df, pre_res = enrichment.run_gsea(self._RANKED.iloc[:n], 'unused.gmt', 'GO_BP',
                                          'KO_vs_WT', self._PARAMS)
        assert df.empty and pre_res is None


# ============================================================
# Section: reproducibility provenance (handoff Piece C item 5)
# ============================================================

class TestGmtVersion:

    @pytest.mark.parametrize('name, expected', [
        ('m5.go.bp.v2026.1.Mm.symbols.gmt',        'v2026.1.Mm'),
        ('m2.cp.reactome.v2026.1.Mm.symbols.gmt',  'v2026.1.Mm'),
        ('c5.go.bp.v2024.1.Hs.symbols.gmt',        'v2024.1.Hs'),
        ('c5.go.bp.v7.5.1.symbols.gmt',            'v7.5.1'),   # pre-2023 scheme
    ])
    def test_msigdb_filenames(self, name, expected):
        assert _gmt_version(f'/some/dir/{name}') == expected

    def test_version_parsed_from_filename_not_directory(self):
        # A versioned-looking directory must not leak into a custom GMT's version.
        assert _gmt_version('/msigdb.v2026.1.Mm.x/custom.gmt') == GMT_VERSION_UNKNOWN

    @pytest.mark.parametrize('name', [
        'example_gene_sets.gmt', 'my.v2.gmt.bak.gmt', 'go_bp_v2026.gmt',
    ])
    def test_non_msigdb_names_are_unknown(self, name):
        # 'my.v2.gmt...' has no dotted multi-part version; 'go_bp_v2026' has no dots.
        assert _gmt_version(name) == GMT_VERSION_UNKNOWN


class TestFileSha256:

    def test_known_digest(self, tmp_path):
        f = tmp_path / 'x.gmt'
        f.write_bytes(b'abc')
        # FIPS 180-2 test vector for SHA-256('abc').
        assert _file_sha256(str(f)) == (
            'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_content_change_changes_digest(self, tmp_path):
        a, b = tmp_path / 'a.gmt', tmp_path / 'b.gmt'
        a.write_text('SET1\tna\tGeneA\tGeneB\n')
        b.write_text('SET1\tna\tGeneA\tGeneC\n')
        assert _file_sha256(str(a)) != _file_sha256(str(b))


class TestWriteSummaryProvenance:

    def _write(self, tmp_path, gmt_name='m5.go.bp.v2026.1.Mm.symbols.gmt'):
        gmt = tmp_path / gmt_name
        gmt.write_text('GOBP_X\tna\tGeneA\tGeneB\n')
        da_df = pd.DataFrame({
            'contrast': ['KO_vs_WT'] * 2, 'gene_symbol': ['GeneA', 'GeneB'],
            'significant': [False, False],
        })
        params = {'enrichment': {
            'min_gene_set_size': 15, 'max_gene_set_size': 500,
            'run_ora': True, 'run_gsea': True, 'gsea_ranking': 't_statistic',
            'gsea_permutations': 1000, 'fdr_threshold': 0.05,
        }}
        write_summary('RUN', tmp_path, da_df, params, [str(gmt)], ['GO_BP'],
                      {}, pd.DataFrame(), '2026-09-29 00:00:00')
        return (tmp_path / 'RUN.enrichment_summary.txt').read_text(), gmt

    @staticmethod
    def _library_fields(text, gmt_name) -> list:
        '''Whitespace fields of the library line: [library, version, filename, ...].'''
        line = next(ln for ln in text.splitlines()
                    if gmt_name in ln and not ln.strip().startswith('sha256:'))
        return line.split()

    def test_records_version_and_checksum(self, tmp_path):
        # Read the Version column itself: the token also appears inside the
        # filename column, so a whole-text search passes even with no Version
        # column (it did on the pre-change summary).
        text, gmt = self._write(tmp_path)
        fields = self._library_fields(text, gmt.name)
        assert fields[:3] == ['GO_BP', 'v2026.1.Mm', gmt.name]
        assert f'sha256: {_file_sha256(str(gmt))}' in text

    def test_custom_gmt_records_unknown_version(self, tmp_path):
        text, _ = self._write(tmp_path, gmt_name='example_gene_sets.gmt')
        fields = self._library_fields(text, 'example_gene_sets.gmt')
        assert fields[1] == GMT_VERSION_UNKNOWN

    def test_discloses_per_library_fdr_scope(self, tmp_path):
        text, _ = self._write(tmp_path)
        assert 'per library' in text
        assert 'do NOT hold a joint FDR' in text

    def test_gsea_correction_not_labelled_bh(self, tmp_path):
        # gseapy.prerank's FDR q-val is the permutation GSEA FDR, not BH.
        # NEGATIVE CONTROL: the pre-fix summary had a single line claiming
        # 'Benjamini-Hochberg, per-library' for both analyses.
        text, _ = self._write(tmp_path)
        gsea_line = next(line for line in text.splitlines() if line.startswith('Correction (GSEA)'))
        assert 'Benjamini' not in gsea_line and 'permutation' in gsea_line
        assert 'Benjamini-Hochberg, per-library' not in text


class TestDuplicateLibraryNames:
    '''
    main() refuses two GMTs that map to the same library short name (e.g. two
    GO_BP releases): per-library outputs are keyed by that name, so the second
    library would silently overwrite the first's GSEA results and plots.
    '''

    def _run(self, tmp_path, monkeypatch, gmt_rel_paths):
        import sys

        import enrichment
        import yaml
        for rel in gmt_rel_paths:
            gmt = tmp_path / rel
            gmt.parent.mkdir(parents=True, exist_ok=True)
            gmt.write_text('TERM\tdesc\tGeneA\tGeneB\n')
        pd.DataFrame({
            'protein_id': ['P1', 'P2'], 'gene_symbol': ['GeneA', 'GeneB'],
            'contrast': ['KO_vs_WT'] * 2, 'significant': [False, False],
            'log2_fc': [0.1, -0.1], 'limma_t': [0.5, -0.5],
            'limma_pvalue': [0.6, 0.7], 'limma_adj_pvalue': [0.8, 0.8],
        }).to_parquet(tmp_path / 'da.parquet', index=False)
        (tmp_path / 'params.yml').write_text(yaml.safe_dump(
            {'enrichment': {'gene_set_libraries': list(gmt_rel_paths), 'run_gsea': False}}))
        monkeypatch.setattr(sys, 'argv', [
            'enrichment.py', '--results', str(tmp_path / 'da.parquet'),
            '--params', str(tmp_path / 'params.yml'), '--run-id', 'DUP',
            '--outdir', str(tmp_path / 'out'),
        ])
        enrichment.main()

    def test_two_releases_of_one_collection_exit(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit):
            self._run(tmp_path, monkeypatch, [
                'a/m5.go.bp.v2025.1.Mm.symbols.gmt', 'b/m5.go.bp.v2026.1.Mm.symbols.gmt'])
        assert not (tmp_path / 'out' / 'DUP.enrichment_results.parquet').exists()

    def test_distinct_names_run(self, tmp_path, monkeypatch):
        # Counterpart: distinct short names are accepted (no significant genes,
        # so ORA auto-skips and the run writes an empty table).
        self._run(tmp_path, monkeypatch, [
            'a/m5.go.bp.v2026.1.Mm.symbols.gmt', 'b/m2.cp.reactome.v2026.1.Mm.symbols.gmt'])
        assert (tmp_path / 'out' / 'DUP.enrichment_results.parquet').exists()
