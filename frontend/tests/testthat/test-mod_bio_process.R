# =============================================================================
# Title:         test-mod_bio_process.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-15
# Last Modified: 2026-07-15
# Purpose:       Group C reactive-logic checks for the Biological Process View
#                module (frontend/R/mod_bio_process.R) using shiny::testServer.
#                Covers the significance filter and term_clicked return (C5),
#                the library-filter sync and subsetting (C6), and the member-
#                selection isolate guard that stops a term change from firing a
#                spurious protein_clicked navigation (C2).
# Inputs:        The synthetic fixture (helper-fixture.R); mod server (sourced).
# Outputs:       testthat assertions.
# Usage:         Rscript frontend/run_tests.R    (or testthat::test_file(...))
# =============================================================================

library(shiny)

# --- C5: significance filter + term_clicked ----------------------------------

test_that('C5: sig_filter subsets the term list and term_clicked returns the id', {
  con <- fixture_con()
  on.exit(close_results_db(con))

  testServer(mod_bio_process_server,
             args = list(con = reactive(con),
                         contrast = reactive('KO_vs_WT'),
                         selected_term = reactive(NULL)), {
    session$setInputs(sig_filter = 'all', lib_filter = 'all')
    expect_equal(nrow(term_filtered()), 4L)

    session$setInputs(sig_filter = 'both')   # only GOBP_X is sig in both
    expect_equal(nrow(term_filtered()), 1L)

    session$setInputs(sig_filter = 'ora')    # GOBP_X, GOBP_Z
    expect_equal(nrow(term_filtered()), 2L)

    session$setInputs(sig_filter = 'gsea')   # GOBP_X, REACTOME_W
    expect_equal(nrow(term_filtered()), 2L)
    expect_equal(output$term_count, '2 terms')

    # term_clicked returns the clicked term_id (rows ordered by term_name).
    session$setInputs(sig_filter = 'all', term_table_rows_selected = 1L)
    expect_equal(session$returned$term_clicked(), 'GOBP_X')
  })
})


# --- C6: library filter stays in sync and subsets correctly ------------------

test_that('C6: lib_filter subsets the term list by library', {
  con <- fixture_con()
  on.exit(close_results_db(con))

  testServer(mod_bio_process_server,
             args = list(con = reactive(con),
                         contrast = reactive('KO_vs_WT'),
                         selected_term = reactive(NULL)), {
    session$setInputs(sig_filter = 'all', lib_filter = 'all')
    session$flushReact()
    expect_setequal(unique(term_data()$library), c('GO_BP', 'REACTOME'))

    session$setInputs(lib_filter = 'GO_BP')   # GOBP_X, GOBP_Y, GOBP_Z
    expect_equal(nrow(term_filtered()), 3L)

    session$setInputs(lib_filter = 'REACTOME')  # REACTOME_W
    expect_equal(nrow(term_filtered()), 1L)
  })
})


# --- C2: member-selection isolate guard (no spurious protein_clicked) --------

test_that('C2: a term change does not fire a spurious protein_clicked', {
  con <- fixture_con()
  on.exit(close_results_db(con))
  sel_term <- reactiveVal('GOBP_X')

  testServer(mod_bio_process_server,
             args = list(con = reactive(con),
                         contrast = reactive('KO_vs_WT'),
                         selected_term = sel_term), {
    # GOBP_X / KO_vs_WT members: P1, P2, P3, P6 (ordered gene-ascending).
    expect_equal(nrow(member_display()), 4L)
    session$setInputs(member_table_rows_selected = 1L)
    expect_equal(session$returned$protein_clicked(), 'P1')

    fires <- 0L
    observe({ session$returned$protein_clicked(); fires <<- fires + 1L })
    session$flushReact()
    base <- fires

    # Changing the term changes the member data, but protein_clicked reads it
    # under isolate() and depends only on the row-selection event, so it must
    # NOT invalidate -> no spurious jump to a protein profile.
    sel_term('REACTOME_W')
    session$flushReact()
    expect_equal(fires, base)
  })
})


# --- Fix 1 wiring: the run's FDR threshold reaches the significance flags -----
# Proves the chain db_param('enrichment.fdr_threshold') -> fdr() -> alpha, not
# just the db-layer argument. GOBP_Y ORA adj_pvalue is exactly 0.05.

test_that('Wiring: a 0.05-run does not flag GOBP_Y as ORA-significant', {
  con <- fixture_con('default')     # run_parameters fdr = 0.05
  on.exit(close_results_db(con))
  testServer(mod_bio_process_server,
             args = list(con = reactive(con), contrast = reactive('KO_vs_WT'),
                         selected_term = reactive(NULL)), {
    session$setInputs(sig_filter = 'ora', lib_filter = 'all')
    expect_false('GOBP_Y' %in% term_filtered()$term_id)
  })
})

test_that('Wiring: a 0.10-run (from run_parameters) flags GOBP_Y as ORA-significant', {
  con <- fixture_con('threshold')   # run_parameters fdr = 0.10
  on.exit(close_results_db(con))
  testServer(mod_bio_process_server,
             args = list(con = reactive(con), contrast = reactive('KO_vs_WT'),
                         selected_term = reactive(NULL)), {
    session$setInputs(sig_filter = 'ora', lib_filter = 'all')
    expect_true('GOBP_Y' %in% term_filtered()$term_id)
  })
})


# --- Member table Direction is labelled by group -------------------------------

test_that('member table Direction reads "higher in <group>"', {
  con <- fixture_con()
  on.exit(close_results_db(con))
  testServer(mod_bio_process_server,
             args = list(con = reactive(con),
                         contrast = reactive('KO_vs_WT'),
                         selected_term = reactive('GOBP_X')), {
    d <- member_display()
    expect_true(nrow(d) > 0)
    expect_false(any(d$Direction %in% c('up', 'down')))
    expect_true(any(grepl('^higher in (KO|WT)$', d$Direction)))
  })
})
