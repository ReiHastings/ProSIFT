# =============================================================================
# Title:         launch.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-16
# Last Modified: 2026-07-16
# Purpose:       Bootstrap and launch the ProSIFT read-only viewer from inside a
#                self-contained bundle. Ensures the six CRAN packages the Shiny
#                app needs are installed, points the app at the bundled data
#                folder via PROSIFT_RESULTS_DIR, and opens the app in a browser.
#                Intended to be invoked by the double-click .command launcher,
#                not run by hand.
# Inputs:        Run with the bundle root as the working directory. Expects a
#                sibling 'app/' folder (a copy of frontend/) and a 'data/' folder
#                containing one subfolder per run with a prosift_results.db file
#                (and an enrichment/ folder of GSEA PNGs).
# Outputs:       A running Shiny app served locally; opens the default browser.
#                May install missing CRAN packages on first run (needs internet
#                that one time).
# Usage:         Rscript launch.R    (normally called by the .command launcher)
# =============================================================================

# --- 1. CRAN mirror ---------------------------------------------------------
# A fresh R install may have no mirror set (repos = '@CRAN@'); pin one so
# install.packages() works non-interactively.
repos <- getOption('repos')
if (is.null(repos[['CRAN']]) || is.na(repos[['CRAN']]) || repos[['CRAN']] == '@CRAN@') {
  options(repos = c(CRAN = 'https://cloud.r-project.org'))
}

# --- 2. Ensure required packages --------------------------------------------
# The viewer depends only on these six CRAN packages (no Bioconductor, no rpy2).
# RSQLite ships as a precompiled binary on macOS, so no compiler is needed.
required <- c('shiny', 'DT', 'DBI', 'RSQLite', 'bslib', 'jsonlite')
have <- vapply(required, requireNamespace, logical(1), quietly = TRUE)
missing <- required[!have]

if (length(missing) > 0) {
  message('First-time setup: installing ', length(missing),
          ' R package(s): ', paste(missing, collapse = ', '))
  message('This can take a few minutes and needs an internet connection.')
  install.packages(missing)
}

# Re-check; stop with a clear message if anything failed to install.
still_missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(still_missing) > 0) {
  stop('Could not install required package(s): ',
       paste(still_missing, collapse = ', '),
       '. Check your internet connection and try again.', call. = FALSE)
}

# --- 3. Point the app at the bundled data -----------------------------------
# The launcher cd's into the bundle root before calling this script, so the
# working directory is the bundle root. config.R reads PROSIFT_RESULTS_DIR and
# scans it recursively for prosift_results.db files.
bundle_dir <- getwd()
data_dir <- file.path(bundle_dir, 'data')
app_dir <- file.path(bundle_dir, 'app')

if (!dir.exists(app_dir)) {
  stop("Cannot find the 'app' folder next to this launcher. The bundle may be ",
       'incomplete; re-copy the whole ProSIFT-viewer folder.', call. = FALSE)
}
if (!dir.exists(data_dir)) {
  stop("Cannot find the 'data' folder next to this launcher. The bundle may be ",
       'incomplete; re-copy the whole ProSIFT-viewer folder.', call. = FALSE)
}

Sys.setenv(PROSIFT_RESULTS_DIR = normalizePath(data_dir))
message('Serving ProSIFT viewer.')
message('  Data folder: ', Sys.getenv('PROSIFT_RESULTS_DIR'))
message('  A browser tab will open shortly. Keep the Terminal window open ',
        'while you use the viewer; close it to stop.')

# --- 4. Launch --------------------------------------------------------------
shiny::runApp(app_dir, launch.browser = TRUE)
