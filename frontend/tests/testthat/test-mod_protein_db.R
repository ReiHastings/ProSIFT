# =============================================================================
# Title:         test-mod_protein_db.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-15
# Last Modified: 2026-07-15
# Purpose:       Group C reactive-logic checks for the Protein Database View
#                module (frontend/R/mod_protein_db.R) using shiny::testServer.
#                Covers the significance-only filter (C3), click-through row ->
#                protein_id mapping plus the gene fallback (C4), and the
#                isolate() regression that stops a filter change from firing a
#                spurious click-through (C1).
# Inputs:        The synthetic fixture (helper-fixture.R); mod server (sourced).
# Outputs:       testthat assertions.
# Usage:         Rscript frontend/run_tests.R    (or testthat::test_file(...))
# =============================================================================

library(shiny)

# --- Shared fixture data (the db_protein_table for KO_vs_WT) -----------------

.pdb_table <- local({
  con <- fixture_con()
  on.exit(close_results_db(con))
  db_protein_table(con, 'KO_vs_WT')
})


# --- C1: the isolate regression (filter change must not fire a click) --------

test_that('C1: toggling the filter with a row selected does not invalidate the click reactive', {
  testServer(mod_protein_db_server,
             args = list(protein_data = reactive(.pdb_table),
                         enabled_dbs = reactive(NULL)), {
    session$setInputs(sig_only = FALSE, table_rows_selected = 1L)

    fires <- 0L
    observe({ session$returned(); fires <<- fires + 1L })
    session$flushReact()
    base <- fires

    # A data change (the significance filter) with the SAME selection must not
    # invalidate the clicked-protein reactive -> no spurious navigation.
    session$setInputs(sig_only = TRUE)
    session$flushReact()
    expect_equal(fires, base)

    # A genuine new selection event DOES invalidate it.
    session$setInputs(table_rows_selected = 2L)
    session$flushReact()
    expect_gt(fires, base)
  })
})


# --- C3: significance-only filter --------------------------------------------

test_that('C3: sig_only filters to significant proteins and updates the count', {
  testServer(mod_protein_db_server,
             args = list(protein_data = reactive(.pdb_table),
                         enabled_dbs = reactive(NULL)), {
    session$setInputs(sig_only = FALSE)
    expect_equal(nrow(filtered_data()), 6L)
    expect_equal(output$row_count, '6 proteins')

    # Significant in KO_vs_WT: P1, P3, P4.
    session$setInputs(sig_only = TRUE)
    expect_equal(nrow(filtered_data()), 3L)
    expect_true(all(filtered_data()$significant == 1L))
    expect_equal(output$row_count, '3 proteins')
  })
})


# --- C4: click-through maps to the right protein; gene fallback --------------

test_that('C4: clicked reactive returns the selected protein_id, NULL when none', {
  testServer(mod_protein_db_server,
             args = list(protein_data = reactive(.pdb_table),
                         enabled_dbs = reactive(NULL)), {
    # No selection yet -> NULL.
    expect_null(session$returned())

    session$setInputs(sig_only = FALSE, table_rows_selected = 1L)
    # Rows are ordered gene-ascending, so row 1 is P1 (Gene1).
    expect_equal(session$returned(), 'P1')

    # Gene fallback: the null-gene protein P6 displays its accession, not blank.
    dd <- display_data()
    expect_equal(dd$Gene[dd$protein_id == 'P6'], 'P6')
  })
})


# --- Fix 2b: disabled-database richness column renders 'n/q', not '--' --------

test_that("Fix 2b: a disabled database's column renders 'n/q', distinct from '--'", {
  # dgidb absent from enabled_dbs -> the whole Drugs column is 'not queried'.
  testServer(mod_protein_db_server,
             args = list(protein_data = reactive(.pdb_table),
                         enabled_dbs = reactive(c('disgenet', 'ctd', 'pubmed'))), {
    session$setInputs(sig_only = FALSE)
    dd <- display_data()
    expect_true(all(dd$Drugs == 'n/q'))
    expect_false(any(dd$Drugs %in% c('Yes', '--')))
    expect_match(as.character(output$nq_legend), 'not enabled', all = FALSE)
  })
})

test_that('Fix 2b: fail-open (enabled_dbs NULL) leaves the Drugs column normal', {
  testServer(mod_protein_db_server,
             args = list(protein_data = reactive(.pdb_table),
                         enabled_dbs = reactive(NULL)), {
    session$setInputs(sig_only = FALSE)
    dd <- display_data()
    expect_false(any(dd$Drugs == 'n/q'))
    expect_true(all(dd$Drugs %in% c('Yes', '--')))
  })
})

test_that('Fix 2b: disabled Diseases/PubMed columns use the n/q render (AC3)', {
  # The rendered DT serializes its column renderers; the 'n/q' JS appears only
  # for a disabled column. Diseases/PubMed disable at the render layer (their
  # display_data stays numeric), Drugs at the R layer.
  render_blob <- function(enabled) {
    blob <- NULL
    testServer(mod_protein_db_server,
               args = list(protein_data = reactive(.pdb_table),
                           enabled_dbs = reactive(enabled)), {
      session$setInputs(sig_only = FALSE)
      blob <<- paste(as.character(output$table), collapse = ' ')
    })
    blob
  }
  # dgidb-only enabled -> disgenet + pubmed off -> their columns render 'n/q'
  # (Drugs is normal because dgidb is enabled here, so any 'n/q' is from the DT
  # renderer, not the R-layer Drugs column).
  expect_true(grepl('n/q', render_blob('dgidb'), fixed = TRUE))
  # all enabled -> no column renders 'n/q'.
  expect_false(grepl('n/q', render_blob(c('disgenet', 'dgidb', 'pubmed')),
                     fixed = TRUE))
})


# --- Direction column is labelled by group, not bare up/down -----------------

test_that('Direction column reads "higher in <group>"', {
  testServer(mod_protein_db_server,
             args = list(protein_data = reactive(.pdb_table),
                         enabled_dbs = reactive(NULL)), {
    dd <- display_data()
    expect_true(all(dd$Direction %in% c('higher in KO', 'higher in WT', 'ns')))
    # Fixture KO_vs_WT: P1 up -> higher in KO; P4 down -> higher in WT.
    expect_identical(dd$Direction[dd$protein_id == 'P1'], 'higher in KO')
    expect_identical(dd$Direction[dd$protein_id == 'P4'], 'higher in WT')
  })
})
