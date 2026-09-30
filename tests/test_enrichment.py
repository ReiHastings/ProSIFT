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
    GMT_VERSION_UNKNOWN,
    GseapyResultSchemaError,
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
# Section 6: cluster_go_terms -- graceful degradation (no rpy2)
# ============================================================
# The real rrvgo clustering (needs R) is not run here. These cover the fallbacks:
# the three cluster columns are added and left null, and input rows preserved.
# Empty / non-GO inputs return before the rpy2 import; the GO-library case blocks
# rpy2 so the lazy `import rpy2.robjects` fails cleanly instead of starting R.

class TestClusterGoTermsDegradation:

    def test_empty_input_gets_null_columns(self):
        out = cluster_go_terms(pd.DataFrame(columns=['library', 'term_id']))
        for col in ('cluster_id', 'is_representative', 'parent_term'):
            assert col in out.columns
        assert out.empty

    def test_non_go_library_skips_with_null_columns(self):
        enr = pd.DataFrame({'library': ['REACTOME', 'REACTOME'],
                            'term_id': ['R1', 'R2'], 'adj_pvalue': [0.01, 0.02]})
        out = cluster_go_terms(enr)
        assert len(out) == 2                                  # rows preserved
        assert out['cluster_id'].isna().all()
        assert out['parent_term'].isna().all()

    def test_go_library_without_rpy2_returns_null_columns(self, monkeypatch):
        # Block rpy2 for this call so cluster_go_terms' lazy `import rpy2.robjects`
        # raises ImportError (instead of starting embedded R, which would segfault
        # where R is not linked), taking the graceful-degradation path.
        monkeypatch.setitem(sys.modules, 'rpy2', None)
        enr = pd.DataFrame({'library': ['GO_BP', 'GO_BP'],
                            'term_id': ['GOBP_A', 'GOBP_B'], 'adj_pvalue': [0.01, 0.02]})
        out = cluster_go_terms(enr)
        assert len(out) == 2                                  # input preserved
        assert out['cluster_id'].isna().all()
        assert out['is_representative'].isna().all()
        assert out['parent_term'].isna().all()



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

    def test_records_version_and_checksum(self, tmp_path):
        text, gmt = self._write(tmp_path)
        assert 'v2026.1.Mm' in text
        assert f'sha256: {_file_sha256(str(gmt))}' in text

    def test_custom_gmt_records_unknown_version(self, tmp_path):
        text, _ = self._write(tmp_path, gmt_name='example_gene_sets.gmt')
        lib_line = next(line for line in text.splitlines() if 'example_gene_sets.gmt' in line)
        assert GMT_VERSION_UNKNOWN in lib_line

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
