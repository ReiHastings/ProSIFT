# =============================================================================
# Title:         test-ui_helpers.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-15
# Last Modified: 2026-07-15
# Purpose:       Group B unit checks for the pure Module 08 formatters. Covers
#                ui_helpers.R (fmt_fc, fmt_p, na_dash, clean_term_name) and the
#                two presentation helpers that live in the view modules:
#                prettify_ctd_actions (mod_protein_profile.R) and sig_dots
#                (mod_bio_process.R). These are string-in / string-out, so they
#                need no fixture database.
# Inputs:        Formatter functions (sourced by helper-source.R).
# Outputs:       testthat assertions.
# Usage:         Rscript frontend/run_tests.R    (or testthat::test_file(...))
# =============================================================================

# --- B1: prettify_ctd_actions (defined in mod_protein_profile.R) --------------

test_that('B1: prettify_ctd_actions expands the CTD action grammar', {
  expect_equal(
    prettify_ctd_actions('decreases^expression|increases^abundance'),
    'decreases expression; increases abundance'
  )
  expect_equal(prettify_ctd_actions(NA_character_), '-')
  # Vectorised over a column, NA per element.
  expect_equal(
    prettify_ctd_actions(c('increases^expression', NA, 'a^b|c^d')),
    c('increases expression', '-', 'a b; c d')
  )
})


# --- B2: clean_term_name -----------------------------------------------------

test_that('B2: clean_term_name strips known prefixes and lowercases', {
  expect_equal(clean_term_name('GOBP_ADAPTIVE_THERMOGENESIS'),
               'adaptive thermogenesis')
  # Each supported library prefix is stripped.
  expect_equal(clean_term_name('GOCC_X'), 'x')
  expect_equal(clean_term_name('GOMF_X'), 'x')
  expect_equal(clean_term_name('REACTOME_SIGNALING'), 'signaling')
  expect_equal(clean_term_name('KEGG_METABOLISM'), 'metabolism')
  expect_equal(clean_term_name('WP_PATHWAY'), 'pathway')
  expect_equal(clean_term_name('HP_PHENOTYPE'), 'phenotype')
  expect_equal(clean_term_name(NA_character_), '-')
})


# --- B3: fmt_fc / fmt_p ------------------------------------------------------

test_that('B3: fmt_fc is signed 2dp and fmt_p is 2 sig figs; NA -> dash', {
  expect_equal(fmt_fc(2), '+2.00')
  expect_equal(fmt_fc(-2.5), '-2.50')
  expect_equal(fmt_fc(0), '+0.00')
  expect_equal(fmt_fc(NA_real_), '-')

  expect_equal(fmt_p(0.012345), '0.012')
  expect_equal(fmt_p(0.00008), '8e-05')
  expect_equal(fmt_p(NA_real_), '-')
})


# --- B4: na_dash -------------------------------------------------------------

test_that('B4: na_dash maps empty-ish values to a dash, passes real values', {
  expect_equal(na_dash(NULL), '-')
  expect_equal(na_dash(character(0)), '-')
  expect_equal(na_dash(NA), '-')
  expect_equal(na_dash(''), '-')
  expect_equal(na_dash('brain'), 'brain')
  expect_equal(na_dash(3), '3')
})


# --- B5: sig_dots (defined in mod_bio_process.R) -----------------------------

test_that('B5: sig_dots fills the ORA / GSEA dot only when significant', {
  both <- sig_dots(1, 1)
  expect_true(grepl('dot-ora', both))
  expect_true(grepl('dot-gsea', both))
  expect_false(grepl('dot-empty', both))

  neither <- sig_dots(0, 0)
  expect_false(grepl('dot-ora', neither))
  expect_false(grepl('dot-gsea', neither))
  expect_equal(lengths(regmatches(neither, gregexpr('dot-empty', neither))), 2L)

  ora_only <- sig_dots(1, 0)
  expect_true(grepl('dot-ora', ora_only))
  expect_true(grepl('dot-empty', ora_only))
  expect_false(grepl('dot-gsea', ora_only))
})


# --- Direction labels follow the contrast's numerator -------------------------

test_that('dir_badge names the group a protein is higher in', {
  lab <- function(x) as.character(x$children[[1]])
  expect_identical(lab(dir_badge('up', 'KO', 'WT')), 'higher in KO')
  expect_identical(lab(dir_badge('down', 'KO', 'WT')), 'higher in WT')
  expect_identical(lab(dir_badge('ns', 'KO', 'WT')), 'ns')
  # Without groups the bare value is kept (backward-compatible call sites).
  expect_identical(lab(dir_badge('up')), 'up')
  expect_identical(lab(dir_badge(NA)), 'ns')
})

test_that('contrast_direction_text states the sign convention', {
  expect_identical(
    contrast_direction_text(list(numerator = 'KO', denominator = 'WT')),
    'log2 FC > 0 and NES > 0 = higher in KO than WT')
  expect_match(contrast_direction_text(list(numerator = NA, denominator = NA)),
               'first-named group')
})


test_that('direction_label is vectorised and tolerates missing groups', {
  expect_identical(direction_label(c('up', 'down', 'ns', NA), 'KO', 'WT'),
                   c('higher in KO', 'higher in WT', 'ns', 'ns'))
  expect_identical(direction_label(c('up', NA)), c('up', 'ns'))
  # NULL / empty groups (e.g. a frame without the columns) fall back, no crash.
  expect_identical(direction_label('down', NULL, NULL), 'down')
  expect_identical(direction_label('down', character(0), 'WT'), 'down')
  lab <- function(x) as.character(x$children[[1]])
  expect_identical(lab(dir_badge('up', NULL, NULL)), 'up')
  expect_identical(lab(dir_badge(NULL)), 'ns')
})
