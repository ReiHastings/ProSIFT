# =============================================================================
# Title:         mod_bio_process.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-14
# Last Modified: 2026-09-29
# Purpose:       Module 08 Biological Process View (spec Section 4.3). Two
#                states in one tab: a sortable/filterable GO/Reactome term list
#                (ORA + GSEA per contrast), and a term profile (identity,
#                enrichment stats, and a DT member-protein table). Presentation
#                only; SQL in db.R. Shared formatters come from ui_helpers.R.
# Inputs:        con           - reactive DBI connection.
#                contrast      - reactive giving the active contrast.
#                selected_term - reactive (app-owned) giving the term_id to
#                                show (NULL shows the term list).
# Outputs:       UI + server. The server returns a list of reactives:
#                  term_clicked    - a clicked term_id (from the list)
#                  back_to_list    - fires on the 'Back to term list' link
#                  protein_clicked - a clicked member protein_id
# Usage:         mod_bio_process_ui('process')
#                b <- mod_bio_process_server('process', con, contrast, sel_term)
# =============================================================================

# Two coloured dots: purple = ORA, teal = GSEA; filled if significant. Returns
# an HTML string (rendered with escape = FALSE in the term table).
sig_dots <- function(ora_sig, gsea_sig) {
  o <- if (isTRUE(ora_sig == 1)) 'dot-ora' else 'dot-empty'
  g <- if (isTRUE(gsea_sig == 1)) 'dot-gsea' else 'dot-empty'
  sprintf(paste0('<span class="dot %s" title="ORA"></span>',
                 '<span class="dot %s" title="GSEA"></span>'), o, g)
}


# Resolve the published GSEA running-score PNG for one term, or NA_character_ if
# it is not on disk. Module 05 (ENRICHMENT) publishes one PNG per top-N enriched
# term per library to results/<run>/enrichment/ (a sibling of the SQLite DB),
# named {run_id}.{contrast}.{library}.gsea_running_score.{term_id}.png with
# '/', ' ', ':' -> '_' in contrast and term_id (see bin/enrichment.py). Most
# terms fall outside the plotted top-N and legitimately have no file.
gsea_running_score_png <- function(con, contrast, library, term_id) {
  if (length(library) != 1 || is.na(library) ||
      length(term_id) != 1 || is.na(term_id)) return(NA_character_)
  dbfile <- tryCatch(DBI::dbGetInfo(con)$dbname, error = function(e) NA_character_)
  run_id <- tryCatch(
    DBI::dbGetQuery(con, 'SELECT run_id FROM run_metadata LIMIT 1')$run_id,
    error = function(e) NA_character_)
  if (length(dbfile) != 1 || is.na(dbfile) || !nzchar(dbfile) ||
      length(run_id) != 1 || is.na(run_id)) return(NA_character_)
  safe <- function(x) gsub('[/ :]', '_', x)
  fname <- sprintf('%s.%s.%s.gsea_running_score.%s.png',
                   run_id, safe(contrast), library, safe(term_id))
  path <- file.path(dirname(dbfile), 'enrichment', fname)
  if (file.exists(path)) path else NA_character_
}


# --- UI ---------------------------------------------------------------------

mod_bio_process_ui <- function(id) {
  ns <- shiny::NS(id)
  shiny::uiOutput(ns('view'))
}


# --- Server -----------------------------------------------------------------

mod_bio_process_server <- function(id, con, contrast, selected_term) {
  shiny::moduleServer(id, function(input, output, session) {
    ns <- session$ns

    # Which state to show: term list (no term selected) or term profile.
    output$view <- shiny::renderUI({
      if (is.null(selected_term())) {
        shiny::tagList(
          shiny::div(class = 'view-controls',
            shiny::selectInput(ns('sig_filter'), 'Significant in',
              choices = c('All' = 'all', 'Both ORA + GSEA' = 'both',
                          'ORA only' = 'ora', 'GSEA only' = 'gsea'),
              width = '190px'),
            shiny::selectInput(ns('lib_filter'), 'Library',
              choices = c('All' = 'all'), width = '160px'),
            shiny::span(class = 'row-count',
                        shiny::textOutput(ns('term_count'), inline = TRUE))),
          DT::DTOutput(ns('term_table')))
      } else {
        shiny::uiOutput(ns('term_profile'))
      }
    })

    # ---- Term list state ----------------------------------------------------

    # The run's configured enrichment FDR threshold (default 0.05); drives the
    # ORA/GSEA significance dots and the significance filter, so a non-default
    # run is not silently scored at 0.05.
    fdr <- shiny::reactive({
      v <- suppressWarnings(as.numeric(
        db_param(con(), 'enrichment.fdr_threshold', '0.05')))
      if (is.na(v)) 0.05 else v
    })

    term_data <- shiny::reactive({
      shiny::req(con(), contrast())
      db_term_list(con(), contrast(), alpha = fdr())
    })

    # Keep the library filter choices in sync with the data.
    shiny::observeEvent(term_data(), {
      libs <- sort(unique(term_data()$library))
      shiny::updateSelectInput(session, 'lib_filter',
        choices = c('All' = 'all', stats::setNames(libs, libs)),
        selected = shiny::isolate(input$lib_filter) %||% 'all')
    })

    term_filtered <- shiny::reactive({
      d <- term_data()
      sig <- input$sig_filter %||% 'all'
      if (sig == 'both') d <- d[d$ora_sig == 1 & d$gsea_sig == 1, , drop = FALSE]
      else if (sig == 'ora') d <- d[d$ora_sig == 1, , drop = FALSE]
      else if (sig == 'gsea') d <- d[d$gsea_sig == 1, , drop = FALSE]
      lib <- input$lib_filter %||% 'all'
      if (!identical(lib, 'all')) d <- d[d$library == lib, , drop = FALSE]
      d
    })

    output$term_count <- shiny::renderText(sprintf('%d terms', nrow(term_filtered())))

    term_display <- shiny::reactive({
      d <- term_filtered()
      data.frame(
        term_id     = d$term_id,
        Term        = vapply(d$term_name, clean_term_name, character(1)),
        Library     = d$library,
        Size        = d$size,
        `ORA p`     = d$ora_adj_p,
        `Odds ratio` = d$odds_ratio,
        Overlap     = d$overlap_size,
        NES         = d$nes,
        `GSEA p`    = d$gsea_adj_p,
        Significant = mapply(sig_dots, d$ora_sig, d$gsea_sig),
        check.names = FALSE, stringsAsFactors = FALSE)
    })

    output$term_table <- DT::renderDT({
      shiny::req(is.null(selected_term()))
      d <- term_display()
      hide <- which(names(d) == 'term_id') - 1L
      dt <- DT::datatable(d, rownames = FALSE, filter = 'top',
        selection = 'single', class = 'stripe hover row-border',
        escape = FALSE,  # the Significant column is HTML dots
        options = list(pageLength = PROSIFT_PAGE_LENGTH, scrollX = TRUE,
          stateSave = TRUE,  # keep sort/search/page across list<->profile (3.2)
          columnDefs = list(list(visible = FALSE, targets = hide)),
          order = list()))
      dt <- DT::formatSignif(dt, c('ORA p', 'GSEA p'), digits = 2)
      dt <- DT::formatRound(dt, c('Odds ratio', 'NES'), digits = 2)
      dt <- DT::formatStyle(dt, 'NES', fontWeight = 'bold',
        color = DT::styleInterval(0, c('#A32D2D', '#0F6E56')))
      dt
    }, server = TRUE)

    # Depend only on the selection event; read the data with isolate() so a data
    # change (filter/contrast) with a stale row index does not fire a spurious
    # click. A genuine row click changes the selection input and fires normally.
    term_clicked <- shiny::reactive({
      sel <- input$term_table_rows_selected
      if (length(sel) == 0) return(NULL)
      shiny::isolate(term_display())$term_id[sel]
    })

    # ---- Term profile state -------------------------------------------------

    output$term_profile <- shiny::renderUI({
      tid <- selected_term()
      shiny::req(tid, con())
      core <- db_term_core(con(), tid)
      if (nrow(core) == 0) {
        return(shiny::div(class = 'placeholder',
          sprintf('Term %s not found in this run.', tid)))
      }
      st <- db_term_stats(con(), tid)
      ora <- st[st$analysis_type == 'ORA', , drop = FALSE]
      gsea <- st[st$analysis_type == 'GSEA', , drop = FALSE]

      # Running-score plot element for the GSEA section: the published PNG when
      # this term is in the plotted top-N, otherwise a short note. Only built
      # when the term has a GSEA result at all.
      gsea_plot_ui <- NULL
      if (nrow(gsea) > 0) {
        png <- gsea_running_score_png(con(), contrast(), core$library, tid)
        gsea_plot_ui <- if (!is.na(png))
          shiny::div(class = 'gsea-plot', style = 'margin-top:10px;',
            shiny::imageOutput(ns('gsea_running_plot'), height = 'auto'))
        else
          shiny::div(class = 'count-note', style = 'margin-top:10px;',
            paste('Running-score plot not generated for this term',
                  '(only the top enriched terms per library are plotted).'))
      }

      stat_body <- shiny::tagList(
        shiny::div(class = 'subsection-title', 'Over-representation (ORA)'),
        if (nrow(ora) == 0) empty_note('No ORA result for this term.') else
          shiny::div(class = 'section-grid',
            stat_card('adj. p-value', fmt_p(ora$adj_pvalue[1])),
            stat_card('odds ratio', formatC(ora$odds_ratio[1], format = 'g', digits = 3)),
            stat_card('overlap', sprintf('%s / %s', na_dash(ora$overlap_size[1]),
                                         na_dash(core$size)))),
        shiny::div(class = 'subsection-title', style = 'margin-top:12px;',
                   'Gene set enrichment (GSEA)'),
        if (nrow(gsea) == 0) empty_note('No GSEA result for this term.') else
          shiny::tagList(
            shiny::div(class = 'section-grid',
              stat_card('adj. p-value', fmt_p(gsea$adj_pvalue[1])),
              stat_card('NES', shiny::span(
                class = if (!is.na(gsea$nes[1]) && gsea$nes[1] < 0) 'val-down' else 'val-up',
                fmt_fc(gsea$nes[1])))),
            gsea_plot_ui))

      shiny::div(class = 'profile',
        shiny::actionLink(ns('back_to_list'), '← Back to term list',
                          class = 'back-link'),
        shiny::div(class = 'profile-header',
          shiny::h3(clean_term_name(core$term_name)),
          shiny::span(class = 'gene-accession', core$term_id)),
        shiny::div(class = 'profile-sub',
          sprintf('%s · %s proteins detected in this run',
                  na_dash(core$library), na_dash(core$size))),
        section('Enrichment statistics', stat_body),
        section('Member proteins',
          shiny::p(class = 'count-note', paste0(
            'Proteins annotated to this term, with their differential ',
            'abundance. Click a row to open its profile.')),
          DT::DTOutput(ns('member_table'))))
    })

    # GSEA running-score image for the selected term. Served only when the file
    # exists on disk; the profile UI omits the imageOutput for terms outside the
    # plotted top-N, so req() here simply no-ops in that case.
    output$gsea_running_plot <- shiny::renderImage({
      tid <- selected_term()
      shiny::req(tid, con(), contrast())
      core <- db_term_core(con(), tid)
      shiny::req(nrow(core) == 1)
      path <- gsea_running_score_png(con(), contrast(), core$library, tid)
      shiny::req(!is.na(path))
      list(src = path, contentType = 'image/png', width = '100%',
           alt = sprintf('GSEA running enrichment score plot for %s', tid))
    }, deleteFile = FALSE)

    member_display <- shiny::reactive({
      shiny::req(selected_term(), con(), contrast())
      m <- db_term_members(con(), selected_term(), contrast())
      yn <- function(v) ifelse(!is.na(v) & v == 1, 'Yes', '')
      data.frame(
        protein_id = m$protein_id,
        Gene       = m$gene_symbol,
        `log2 FC`  = m$log2_fc,
        `Adj p`    = m$adj_pvalue,
        Direction  = direction_label(m$direction, m$numerator[1], m$denominator[1]),
        `In ORA set` = yn(m$in_significant_set),
        `Leading edge` = yn(m$is_leading_edge),
        check.names = FALSE, stringsAsFactors = FALSE)
    })

    output$member_table <- DT::renderDT({
      shiny::req(!is.null(selected_term()))
      d <- member_display()
      hide <- which(names(d) == 'protein_id') - 1L
      dt <- DT::datatable(d, rownames = FALSE, filter = 'top',
        selection = 'single', class = 'stripe hover row-border',
        options = list(pageLength = 15, scrollX = TRUE, stateSave = TRUE,
          columnDefs = list(list(visible = FALSE, targets = hide)),
          order = list()))
      dt <- DT::formatSignif(dt, 'Adj p', digits = 2)
      dt <- DT::formatStyle(dt, 'log2 FC',
        color = DT::styleInterval(0, c('#A32D2D', '#0F6E56')))
      g <- db_contrast_groups(con(), contrast())
      dt <- DT::formatStyle(dt, 'Direction', fontWeight = 'bold',
        color = direction_style(g$numerator, g$denominator))
      dt
    }, server = TRUE)

    # Clear any stale member-row selection when the term changes. Without this,
    # re-rendering the member table for a new term while the server still holds
    # the previous row index fires a spurious protein_clicked, bouncing the user
    # to a protein profile (caught by the slice-3 smoke test 2026-07-14).
    member_proxy <- DT::dataTableProxy('member_table')
    shiny::observeEvent(selected_term(), {
      DT::selectRows(member_proxy, NULL)
    }, ignoreNULL = FALSE, ignoreInit = TRUE)

    protein_clicked <- shiny::reactive({
      sel <- input$member_table_rows_selected
      if (length(sel) == 0) return(NULL)
      shiny::isolate(member_display())$protein_id[sel]
    })

    list(
      term_clicked = term_clicked,
      back_to_list = shiny::reactive(input$back_to_list),
      protein_clicked = protein_clicked)
  })
}
