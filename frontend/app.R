# =============================================================================
# Title:         app.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-13
# Last Modified: 2026-09-29
# Purpose:       Module 08 interactive frontend entry point. Builds the app
#                shell (persistent top bar with run + contrast selectors, three
#                tab views) and wires the Protein Database View to a read-only
#                Module 07 SQLite database. Helper and view-module files in R/
#                are auto-sourced by Shiny. Slice 1 of the Module 08 build:
#                skeleton + Protein Database View; profile and process views are
#                placeholders pending later slices.
# Inputs:        A Module 07 results database discovered under the results roots
#                (see config.R). Defaults to the local dev DB in .devdata/.
# Outputs:       A running Shiny application (no data outputs).
# Usage:         From the project root:  R -e "shiny::runApp('frontend')"
# =============================================================================

library(shiny)
library(DT)
library(DBI)
library(RSQLite)
library(bslib)

# --- Theme ------------------------------------------------------------------

# Teal primary; default Bootstrap system font stack (no Google Fonts fetch) to
# keep the frontend offline-first, consistent with ProSIFT's offline principle.
prosift_theme <- bslib::bs_theme(version = 5, primary = PROSIFT_TEAL)


# --- UI ---------------------------------------------------------------------

ui <- shiny::fluidPage(
  theme = prosift_theme,
  shiny::includeCSS('www/prosift.css'),

  # Persistent top bar: brand + run selector (spec Section 3.2 / 4.4.1).
  shiny::div(
    class = 'prosift-topbar',
    shiny::div(class = 'prosift-brand', 'ProSIFT'),
    shiny::div(
      class = 'prosift-run',
      shiny::selectInput('run', label = NULL, choices = NULL, width = '320px')
    )
  ),

  shiny::tabsetPanel(
    id = 'main_tabs',

    # View 1: Protein database.
    shiny::tabPanel(
      'Protein database', value = 'db',
      shiny::div(
        class = 'contrast-bar',
        shiny::tags$label('Contrast:', `for` = 'contrast'),
        shiny::selectInput('contrast', label = NULL, choices = NULL, width = '240px'),
        shiny::span(class = 'contrast-direction',
                    shiny::textOutput('contrast_direction', inline = TRUE))
      ),
      mod_protein_db_ui('db')
    ),

    # View 2: Protein profile.
    shiny::tabPanel(
      'Protein profile', value = 'profile',
      mod_protein_profile_ui('profile')
    ),

    # View 3: Biological process.
    shiny::tabPanel(
      'Biological process', value = 'process',
      shiny::div(class = 'contrast-bar contrast-direction',
                 shiny::textOutput('contrast_direction_process', inline = TRUE)),
      mod_bio_process_ui('process')
    )
  )
)


# --- Server -----------------------------------------------------------------

server <- function(input, output, session) {

  # Step 1: discover available runs and populate the run selector.
  runs <- discover_runs()
  if (nrow(runs) == 0) {
    shiny::showNotification(
      paste('No prosift_results.db found. Set PROSIFT_RESULTS_DIR or stage a',
            'database in frontend/.devdata/.'),
      type = 'error', duration = NULL
    )
  } else {
    shiny::updateSelectInput(
      session, 'run',
      choices = stats::setNames(runs$path, runs$run_id)
    )
  }

  # Step 2: hold one read-only connection per selected run. Switching runs
  # closes the old connection and opens the new one (spec Section 3.3 / 4.4.1).
  con <- shiny::reactiveVal(NULL)

  shiny::observeEvent(input$run, {
    shiny::req(input$run, nzchar(input$run))
    # Open the new connection first; only swap in (and close the old one) on
    # success, so a corrupt/locked/partial database surfaces a clean message
    # without dropping a working connection or crashing the session.
    new_con <- tryCatch(
      open_results_db(input$run),
      error = function(e) {
        shiny::showNotification(
          paste0('Could not open the results database: ', conditionMessage(e)),
          type = 'error', duration = NULL
        )
        NULL
      }
    )
    if (!is.null(new_con)) {
      close_results_db(con())
      con(new_con)
    }
  }, ignoreInit = TRUE)

  # Close the connection cleanly when the session ends.
  session$onSessionEnded(function() close_results_db(shiny::isolate(con())))

  # Step 3: populate the contrast selector from the active connection.
  shiny::observeEvent(con(), {
    shiny::req(con())
    contrasts <- db_contrasts(con())
    shiny::updateSelectInput(session, 'contrast', choices = contrasts)
  })

  # Step 3b: state the sign convention of the selected contrast wherever its
  # statistics are shown, so "up" and positive NES are never read backwards.
  contrast_direction <- shiny::reactive({
    shiny::req(con(), input$contrast)
    contrast_direction_text(db_contrast_groups(con(), input$contrast))
  })
  output$contrast_direction <- shiny::renderText(contrast_direction())
  output$contrast_direction_process <- shiny::renderText(
    sprintf('Contrast %s: %s', input$contrast, contrast_direction())
  )

  # Step 4: app-owned navigation state. The currently viewed protein and term
  # are held here (not inside a view module) so any view can drive navigation --
  # the DB table, a term's member row, and a profile's term link all feed these.
  # Reset on run switch (spec Section 4.4.1).
  current_protein <- shiny::reactiveVal(NULL)
  current_term <- shiny::reactiveVal(NULL)
  shiny::observeEvent(con(), {
    current_protein(NULL)
    current_term(NULL)
  })

  # Step 5: the protein-overview data for the current run + contrast.
  protein_data <- shiny::reactive({
    shiny::req(con(), input$contrast)
    db_protein_table(con(), input$contrast)
  })

  # The Module 06 databases enabled for this run (NULL = unknown -> fail-open).
  enabled_dbs <- shiny::reactive({
    if (is.null(con())) NULL else db_enabled_dbs(con())
  })

  # Step 6: Protein Database View -> clicked protein sets current_protein.
  db_clicked <- mod_protein_db_server('db', protein_data, enabled_dbs)
  shiny::observeEvent(db_clicked(), {
    current_protein(db_clicked())
    shiny::updateTabsetPanel(session, 'main_tabs', selected = 'profile')
  })

  # Step 7: Protein Profile View. `back` -> database tab; `term_selected` (a
  # clicked enriched-term link) -> set current_term and open the process tab.
  profile <- mod_protein_profile_server('profile', con, current_protein)
  shiny::observeEvent(profile$back(), {
    shiny::updateTabsetPanel(session, 'main_tabs', selected = 'db')
  })
  shiny::observeEvent(profile$term_selected(), {
    shiny::req(profile$term_selected())
    current_term(profile$term_selected())
    shiny::updateTabsetPanel(session, 'main_tabs', selected = 'process')
  })

  # Step 8: Biological Process View. Reads current_term (NULL -> term list).
  # `term_clicked` opens a term; `back_to_list` clears it; `protein_clicked`
  # (a member row) sets current_protein and opens the profile tab -- completing
  # the protein <-> term navigation triangle.
  bio <- mod_bio_process_server('process', con,
                                shiny::reactive(input$contrast), current_term)
  shiny::observeEvent(bio$term_clicked(), current_term(bio$term_clicked()))
  shiny::observeEvent(bio$back_to_list(), current_term(NULL))
  shiny::observeEvent(bio$protein_clicked(), {
    current_protein(bio$protein_clicked())
    shiny::updateTabsetPanel(session, 'main_tabs', selected = 'profile')
  })
}

shiny::shinyApp(ui, server)
