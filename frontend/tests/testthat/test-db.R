# =============================================================================
# Title:         test-db.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-15
# Last Modified: 2026-07-15
# Purpose:       Group A unit invariants for the Module 08 data-access layer
#                (frontend/R/db.R). Runs the pure, deterministic query functions
#                against the in-code synthetic SQLite fixture (helper-fixture.R)
#                and pins the invariants that matter: success-only aggregation,
#                no join fan-out, the DEqMS/limma COALESCE fallback, the ORA/GSEA
#                pivot and its strict significance boundary, observed-only QC
#                flags, and null-gene ordering.
# Inputs:        The fixture database (fixture_con()); db.R functions (sourced).
# Outputs:       testthat assertions.
# Usage:         Rscript frontend/run_tests.R    (or testthat::test_file(...))
# =============================================================================

# --- A1: placeholder query-status rows are not counted as real records -------

test_that('A1: db_protein_table excludes placeholder status rows from counts', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tab <- db_protein_table(con, 'KO_vs_WT')

  # P2 has only a no_associations disease row, a no_interactions drug row and a
  # no_hits pubmed row (score 9.9). All must be excluded -> NA aggregates.
  p2 <- tab[tab$protein_id == 'P2', ]
  expect_true(is.na(p2$disease_count))
  expect_true(is.na(p2$drug_count))
  expect_true(is.na(p2$top_pmi))

  # P3 has only a no_ortholog disease placeholder -> NA.
  p3 <- tab[tab$protein_id == 'P3', ]
  expect_true(is.na(p3$disease_count))
})


# --- A2: one row per protein, no LEFT JOIN fan-out ---------------------------

test_that('A2: db_protein_table keeps one row per protein despite multi-record joins', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tab <- db_protein_table(con, 'KO_vs_WT')

  # Every protein appears exactly once even though P1 has 2 disease, 2 drug and
  # 2 pubmed success rows.
  expect_equal(nrow(tab), 6L)
  expect_equal(anyDuplicated(tab$protein_id), 0L)
  expect_equal(tab$disease_count[tab$protein_id == 'P1'], 2L)
})


# --- A3: adj_pvalue = COALESCE(deqms, limma) fallback ------------------------

test_that('A3: adj_pvalue prefers DEqMS and falls back to limma', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tab <- db_protein_table(con, 'KO_vs_WT')

  # P1 has DEqMS present -> use it (0.008), not limma (0.01).
  expect_equal(tab$adj_pvalue[tab$protein_id == 'P1'], 0.008)
  # P2 has DEqMS NULL -> fall back to limma (0.9).
  expect_equal(tab$adj_pvalue[tab$protein_id == 'P2'], 0.9)
})


# --- A4: aggregates count success rows / take MAX pmi ------------------------

test_that('A4: disease_count counts success rows and top_pmi is their MAX', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tab <- db_protein_table(con, 'KO_vs_WT')
  p1 <- tab[tab$protein_id == 'P1', ]

  expect_equal(p1$disease_count, 2L)
  expect_equal(p1$drug_count, 2L)
  expect_equal(p1$top_pmi, 2.5)   # MAX(1.3, 2.5) over success rows
})


# --- A5: db_term_list pivots ORA + GSEA with correct significance flags ------

test_that('A5: db_term_list gives one row per term with side-by-side sig flags', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tl <- db_term_list(con, 'KO_vs_WT')

  expect_equal(anyDuplicated(tl$term_id), 0L)
  gx <- tl[tl$term_id == 'GOBP_X', ]
  expect_equal(gx$ora_sig, 1L)
  expect_equal(gx$gsea_sig, 1L)
  expect_equal(gx$ora_adj_p, 0.01)
  expect_equal(gx$gsea_adj_p, 0.02)
  expect_equal(gx$nes, 1.5)
  expect_equal(gx$odds_ratio, 3.0)
})


# --- A6: significance boundary is strictly < 0.05 ----------------------------

test_that('A6: adj_pvalue exactly 0.05 is NOT significant', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tl <- db_term_list(con, 'KO_vs_WT')
  gy <- tl[tl$term_id == 'GOBP_Y', ]

  # GOBP_Y ORA adj_pvalue is exactly 0.05 -> sig 0 (strict less-than).
  expect_equal(gy$ora_adj_p, 0.05)
  expect_equal(gy$ora_sig, 0L)
  expect_equal(gy$gsea_sig, 0L)   # GSEA adj_pvalue 0.20
})


# --- A7: single-analysis terms still appear ----------------------------------

test_that('A7: terms with only ORA or only GSEA still appear once', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tl <- db_term_list(con, 'KO_vs_WT')

  expect_equal(nrow(tl), 4L)   # GOBP_X, GOBP_Y, GOBP_Z, REACTOME_W

  gz <- tl[tl$term_id == 'GOBP_Z', ]        # ORA only
  expect_equal(gz$ora_sig, 1L)
  expect_true(is.na(gz$gsea_adj_p))
  expect_equal(gz$gsea_sig, 0L)             # COALESCE(gsea.sig, 0)

  rw <- tl[tl$term_id == 'REACTOME_W', ]    # GSEA only
  expect_equal(rw$gsea_sig, 1L)
  expect_true(is.na(rw$ora_adj_p))
  expect_equal(rw$ora_sig, 0L)              # COALESCE(ora.sig, 0)
})


# --- A8: QC flags come only from observed (non-imputed) samples --------------

test_that('A8: db_protein_qc_flags excludes samples where the protein is imputed', {
  con <- fixture_con(); on.exit(close_results_db(con))
  qc <- db_protein_qc_flags(con, 'P1')

  # P1 is imputed in S_KO-1, so that row (and its pca_outlier flag) is excluded.
  expect_equal(nrow(qc), 3L)
  expect_false('S_KO-1' %in% qc$sample_id)
  expect_equal(sum(qc$flag_pca_outlier), 0L)      # the excluded sample's flag
  expect_equal(sum(qc$flag_extreme_median), 1L)   # S_WT-1, an observed sample
})


# --- A9: null / empty gene symbols sort last ---------------------------------

test_that('A9: db_protein_table orders null-gene proteins last', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tab <- db_protein_table(con, 'KO_vs_WT')

  # P6 has a NULL gene symbol; it must be the final row.
  expect_equal(tab$protein_id[nrow(tab)], 'P6')
  expect_true(is.na(tab$gene_symbol[nrow(tab)]))
  # The non-null genes ascend ahead of it.
  expect_equal(tab$protein_id[1], 'P1')
})


# --- A10: enriched terms are significant-only, collapsed, MAX leading edge ----

test_that('A10: db_protein_enriched_terms filters, collapses, and MAXes flags', {
  con <- fixture_con(); on.exit(close_results_db(con))
  et <- db_protein_enriched_terms(con, 'P1')

  # P1 is in GOBP_X (2 contrasts, in_significant_set 1) and GOBP_Y (not in the
  # significant set). Only GOBP_X survives, collapsed to one row.
  expect_equal(nrow(et), 1L)
  expect_equal(et$term_id, 'GOBP_X')
  # is_leading_edge is TRUE (1) in at least one contrast -> MAX = 1.
  expect_equal(as.integer(et$is_leading_edge), 1L)
})


# --- A11: LEFT JOIN keeps proteins absent from the contrast; contrast filter -

test_that('A11: proteins with no DA row for the contrast still appear (NA stats)', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tab <- db_protein_table(con, 'KO_vs_WT')
  p6 <- tab[tab$protein_id == 'P6', ]

  expect_equal(nrow(p6), 1L)
  expect_true(is.na(p6$log2_fc))
  expect_true(is.na(p6$adj_pvalue))

  # The contrast param filters DA: P1 differs between contrasts (0.008 vs 0.02).
  other <- db_protein_table(con, 'AKO_vs_WT')
  expect_equal(other$adj_pvalue[other$protein_id == 'P1'], 0.02)
  # Only P1 has a row in AKO_vs_WT, so every other protein has NA stats there.
  expect_true(is.na(other$log2_fc[other$protein_id == 'P3']))
})


# --- A12: profile section queries are success-only and correctly ordered -----

test_that('A12: disease / drug / chemical profile queries filter and order', {
  con <- fixture_con(); on.exit(close_results_db(con))

  dis <- db_protein_diseases(con, 'P1')
  expect_equal(nrow(dis), 2L)                       # success rows only
  expect_equal(dis$disease_name, c('DiseaseB', 'DiseaseA'))  # gda_score desc

  # P2's no_associations placeholder yields no disease rows.
  expect_equal(nrow(db_protein_diseases(con, 'P2')), 0L)

  dr <- db_protein_drugs(con, 'P1')
  expect_equal(dr$drug_name, c('DrugA', 'DrugX'))   # ordered by drug_name

  chem <- db_protein_chemicals(con, 'P1')
  expect_equal(chem$chemical_name, c('ChemY', 'ChemX'))  # n_publications desc
})


# --- A13: db_term_members membership + left join + fallback + ordering --------

test_that('A13: db_term_members joins DA per contrast with fallback and ordering', {
  con <- fixture_con(); on.exit(close_results_db(con))
  m <- db_term_members(con, 'GOBP_X', 'KO_vs_WT')

  # Members of GOBP_X in KO_vs_WT: P1, P2, P3, P6 (empty gene -> last).
  expect_equal(nrow(m), 4L)
  expect_equal(m$protein_id[nrow(m)], 'P6')

  # P2's DA has DEqMS NULL -> COALESCE fallback to limma (0.9).
  expect_equal(m$adj_pvalue[m$protein_id == 'P2'], 0.9)
  # P6 has no DA row for this contrast -> NA stats (LEFT JOIN).
  expect_true(is.na(m$log2_fc[m$protein_id == 'P6']))
})


# --- A14: core queries return one row (known) / zero rows (unknown) -----------

test_that('A14: db_protein_core and db_term_core are one-row / zero-row', {
  con <- fixture_con(); on.exit(close_results_db(con))

  expect_equal(nrow(db_protein_core(con, 'P1')), 1L)
  expect_equal(nrow(db_protein_core(con, 'NOPE')), 0L)

  expect_equal(nrow(db_term_core(con, 'GOBP_X')), 1L)
  expect_equal(db_term_core(con, 'GOBP_X')$size, 20L)
  expect_equal(nrow(db_term_core(con, 'NOPE')), 0L)
})


# --- A15: db_contrasts is distinct and sorted ascending ----------------------

test_that('A15: db_contrasts returns distinct, sorted contrasts', {
  con <- fixture_con(); on.exit(close_results_db(con))
  expect_equal(db_contrasts(con), c('AKO_vs_WT', 'KO_vs_WT'))
})


# --- A16: open_results_db rejects bad paths clearly --------------------------

test_that('A16: open_results_db errors on empty / missing paths', {
  expect_error(open_results_db(''), 'non-empty database path')
  expect_error(open_results_db(NA_character_), 'non-empty database path')
  expect_error(open_results_db(tempfile('does_not_exist_')), 'not found')
})


# --- Fix 1: enrichment significance respects the alpha argument ---------------

test_that('Fix 1: db_term_list ora_sig respects the alpha threshold', {
  con <- fixture_con(); on.exit(close_results_db(con))
  # GOBP_Y ORA adj_pvalue is exactly 0.05: not significant at 0.05 (strict <),
  # significant at 0.10.
  y05 <- db_term_list(con, 'KO_vs_WT', alpha = 0.05)
  y10 <- db_term_list(con, 'KO_vs_WT', alpha = 0.10)
  expect_equal(y05$ora_sig[y05$term_id == 'GOBP_Y'], 0)
  expect_equal(y10$ora_sig[y10$term_id == 'GOBP_Y'], 1)
  # Default arg is 0.05.
  expect_equal(db_term_list(con, 'KO_vs_WT')$ora_sig[y05$term_id == 'GOBP_Y'], 0)
})

test_that('Fix 1: count(ora_sig) is monotonic non-decreasing in alpha', {
  con <- fixture_con(); on.exit(close_results_db(con))
  counts <- vapply(c(0.001, 0.05, 0.10, 0.5), function(a) {
    sum(db_term_list(con, 'KO_vs_WT', alpha = a)$ora_sig)
  }, numeric(1))
  expect_false(is.unsorted(counts))
})

test_that('Fix 1: db_term_list is order-independent (metamorphic on row order)', {
  base <- fixture_con(); on.exit(close_results_db(base))
  ref <- db_term_list(base, 'KO_vs_WT')
  # Build a DB whose enrichment_results rows are reversed; result must match.
  p <- tempfile('prosift_fixture_shuffled_', fileext = '.db')
  build_fixture_db(p)                       # standard tables, then overwrite one
  con_w <- DBI::dbConnect(RSQLite::SQLite(), p)
  er <- .fx_enrichment_results()
  DBI::dbWriteTable(con_w, 'enrichment_results', er[rev(seq_len(nrow(er))), ],
                    overwrite = TRUE)
  DBI::dbDisconnect(con_w)
  shuf <- open_results_db(p); on.exit(close_results_db(shuf), add = TRUE)
  expect_equal(db_term_list(shuf, 'KO_vs_WT'), ref)
})

test_that('Fix 1: db_term_list / db_protein_table on an absent contrast are 0-row', {
  con <- fixture_con(); on.exit(close_results_db(con))
  expect_equal(nrow(db_term_list(con, 'NO_SUCH_CONTRAST')), 0L)
  # proteins still LEFT-JOIN through (6 rows) but with NA stats -> exercises the
  # empty-DA-for-contrast path without error.
  expect_equal(nrow(db_protein_table(con, 'NO_SUCH_CONTRAST')), 6L)
})


# --- Fix 1/2: db_param robustness + run-parameter accessors -------------------

test_that('db_param returns the default on a missing run_parameters table', {
  con <- fixture_con('no_params'); on.exit(close_results_db(con))
  expect_false(DBI::dbExistsTable(con, 'run_parameters'))
  expect_equal(db_param(con, 'enrichment.fdr_threshold', '0.05'), '0.05')
  expect_null(db_enabled_dbs(con))              # NULL -> fail-open
})

test_that('db_param returns the default on a missing key', {
  con <- fixture_con(); on.exit(close_results_db(con))
  expect_equal(db_param(con, 'no.such.key', 'DEF'), 'DEF')
  expect_equal(db_param(con, 'enrichment.fdr_threshold', '0.05'), '0.05')
})

test_that('databases.enabled is stored/read in the real JSON wire format', {
  con <- fixture_con(); on.exit(close_results_db(con))
  raw <- db_param(con, 'databases.enabled')
  expect_true(is.character(raw) && grepl('^\\[', raw))     # a JSON string array
  parsed <- db_enabled_dbs(con)
  expect_setequal(parsed, c('uniprot', 'pubmed', 'disgenet', 'dgidb', 'ctd'))
  expect_true(db_enabled('dgidb', parsed))
})

test_that('the disabled variant omits dgidb from databases.enabled', {
  con <- fixture_con('disabled'); on.exit(close_results_db(con))
  enabled <- db_enabled_dbs(con)
  expect_false(db_enabled('dgidb', enabled))
  expect_true(db_enabled('disgenet', enabled))
})

test_that("db_enabled_dbs: empty array is all-disabled; JSON null fails open", {
  set_enabled <- function(val) {
    p <- tempfile('fx_dbenabled_', fileext = '.db')
    build_fixture_db(p)
    cw <- DBI::dbConnect(RSQLite::SQLite(), p)
    DBI::dbExecute(
      cw, "UPDATE run_parameters SET value = ? WHERE key = 'databases.enabled'",
      params = list(val))
    DBI::dbDisconnect(cw)
    p
  }
  c_empty <- open_results_db(set_enabled('[]'))
  c_null  <- open_results_db(set_enabled('null'))
  on.exit({ close_results_db(c_empty); close_results_db(c_null) })

  # '[]' -> a KNOWN empty list: every database is off (not fail-open).
  expect_identical(db_enabled_dbs(c_empty), character(0))
  expect_false(db_enabled('dgidb', db_enabled_dbs(c_empty)))

  # 'null' -> unknown -> fail-open (NULL), so every database reads as enabled.
  expect_null(db_enabled_dbs(c_null))
  expect_true(db_enabled('dgidb', db_enabled_dbs(c_null)))
})


# --- Contrast orientation (numerator/denominator) ------------------------------

test_that('split_contrast_label splits on the first _vs_ and NAs malformed labels', {
  g <- split_contrast_label(c('KO_vs_WT', 'A_vs_B_vs_C', 'KOWT', '_vs_WT', 'KO_vs_'))
  expect_identical(g$numerator,   c('KO', 'A', NA, NA, NA))
  expect_identical(g$denominator, c('WT', 'B_vs_C', NA, NA, NA))
})

test_that('db_contrast_groups falls back to the label on pre-orientation DBs', {
  # The default fixture has no numerator/denominator columns (older schema).
  con <- fixture_con(); on.exit(close_results_db(con))
  expect_identical(db_contrast_groups(con, 'KO_vs_WT'),
                   list(numerator = 'KO', denominator = 'WT'))
})

test_that('db_contrast_groups prefers the stored columns over the label', {
  # Precedence probe: store values that differ from the label split, so the
  # test can only pass if the columns are actually read.
  p <- tempfile('fx_orient_', fileext = '.db')
  build_fixture_db(p)
  cw <- DBI::dbConnect(RSQLite::SQLite(), p)
  DBI::dbExecute(cw, 'ALTER TABLE differential_abundance ADD COLUMN numerator TEXT')
  DBI::dbExecute(cw, 'ALTER TABLE differential_abundance ADD COLUMN denominator TEXT')
  DBI::dbExecute(cw, "UPDATE differential_abundance SET numerator = 'KOcol', denominator = 'WTcol'")
  DBI::dbDisconnect(cw)
  con <- open_results_db(p); on.exit(close_results_db(con))

  expect_identical(db_contrast_groups(con, 'KO_vs_WT'),
                   list(numerator = 'KOcol', denominator = 'WTcol'))
  ct <- db_protein_contrasts(con, 'P1')
  expect_identical(ct$numerator, c('KOcol', 'KOcol'))
})

test_that('db_protein_contrasts attaches orientation per contrast row', {
  con <- fixture_con(); on.exit(close_results_db(con))
  ct <- db_protein_contrasts(con, 'P1')          # AKO_vs_WT and KO_vs_WT rows
  expect_identical(ct$numerator,   c('AKO', 'KO'))
  expect_identical(ct$denominator, c('WT', 'WT'))
  # A protein with no DA rows still returns a well-formed empty frame.
  empty <- db_protein_contrasts(con, 'P6')
  expect_equal(nrow(empty), 0)
  expect_true(all(c('numerator', 'denominator') %in% names(empty)))
})


test_that('single-contrast queries carry the contrast orientation', {
  con <- fixture_con(); on.exit(close_results_db(con))
  tab <- db_protein_table(con, 'KO_vs_WT')
  expect_true(all(tab$numerator == 'KO') && all(tab$denominator == 'WT'))
  mem <- db_term_members(con, 'GOBP_X', 'KO_vs_WT')
  expect_true(nrow(mem) > 0)
  expect_true(all(mem$numerator == 'KO') && all(mem$denominator == 'WT'))
})
