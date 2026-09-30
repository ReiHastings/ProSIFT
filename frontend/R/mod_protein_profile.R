# =============================================================================
# Title:         mod_protein_profile.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-14
# Last Modified: 2026-09-29
# Purpose:       Module 08 Protein Profile View (spec Section 4.2). The
#                per-protein 'baseball card': nine sections consolidating
#                differential abundance, data-quality/QC-flag notes, UniProt
#                annotation, enriched terms, disease/drug/chemical associations,
#                and literature co-occurrence. Presentation only; SQL in db.R.
# Inputs:        con              - reactive holding the active DBI connection.
#                selected_protein - reactive giving the protein_id to show
#                                   (NULL shows a 'select a protein' prompt).
# Outputs:       UI + server. The server returns a list with `back`, a reactive
#                that fires when the 'Back to database' link is clicked, so the
#                parent can switch back to the database tab.
# Usage:         mod_protein_profile_ui('profile')
#                p <- mod_protein_profile_server('profile', con, selected)
# =============================================================================

# --- Constants / small formatters -------------------------------------------

# Plain-language names for the four Module 02 sample-level QC flags. These are
# heuristic QC signals, not verdicts (Module 08 spec Section 4.2.1, note 1).
FLAG_LABELS <- c(
  flag_low_detection   = 'low protein detection',
  flag_extreme_median  = 'extreme median abundance',
  flag_pca_outlier     = 'PCA outlier',
  flag_low_correlation = 'low inter-sample correlation'
)

# Rows to show before truncating a long section (with a running total note).
# Shared formatters/badges/section wrappers live in R/ui_helpers.R.
PROFILE_SECTION_CAP <- 15L

# Build an HTML table from a data.frame slice, capped, with a total-count note.
# `cells` is a function(row_df_1) returning a list of <td> tags.
capped_table <- function(df, headers, cells, cap = PROFILE_SECTION_CAP) {
  n <- nrow(df)
  shown <- utils::head(df, cap)
  tbl <- shiny::tags$table(class = 'mini-table',
    shiny::tags$thead(shiny::tags$tr(lapply(headers, shiny::tags$th))),
    shiny::tags$tbody(lapply(seq_len(nrow(shown)), function(i) shiny::tags$tr(cells(shown[i, ])))))
  if (n > cap) {
    shiny::tagList(tbl, shiny::div(class = 'count-note',
      sprintf('Showing %d of %d.', cap, n)))
  } else {
    tbl
  }
}


# --- QC flag notes (Section 3) ----------------------------------------------

# Build the QC-flag notes block from the observed-sample flags data.frame.
qc_notes_block <- function(qc, detection_category) {
  items <- list()
  if (nrow(qc) > 0) {
    for (i in seq_len(nrow(qc))) {
      row <- qc[i, ]
      fired <- names(FLAG_LABELS)[vapply(names(FLAG_LABELS),
        function(fc) isTRUE(as.logical(row[[fc]])), logical(1))]
      if (length(fired) == 0) next
      reasons <- paste(FLAG_LABELS[fired], collapse = ', ')
      items[[length(items) + 1]] <- shiny::tags$li(
        shiny::tags$strong(sprintf('%s (%s)', row$sample_id, na_dash(row$grp))),
        sprintf(': %s', reasons))
    }
  }
  body <- if (length(items) == 0) {
    empty_note('No QC flags on the samples this protein was observed in.')
  } else {
    shiny::tagList(
      shiny::tags$ul(class = 'qc-notes', items),
      shiny::div(class = 'count-note', paste(
        'Heuristic sample-level QC signals, not verdicts. Only samples where',
        'this protein was observed are considered.')))
  }
  single_group <- identical(as.character(detection_category), 'SINGLE-GROUP')
  if (single_group) {
    body <- shiny::tagList(body, shiny::div(class = 'qc-warn', paste(
      'SINGLE-GROUP: this protein is detected in only one condition, so the',
      'other group\'s mean is entirely imputed. Interpret the fold change as a',
      'floor, not a point estimate.')))
  }
  body
}


# --- UI ---------------------------------------------------------------------

mod_protein_profile_ui <- function(id) {
  ns <- shiny::NS(id)
  shiny::uiOutput(ns('content'))
}


# --- Server -----------------------------------------------------------------

mod_protein_profile_server <- function(id, con, selected_protein) {
  shiny::moduleServer(id, function(input, output, session) {
    ns <- session$ns

    output$content <- shiny::renderUI({
      pid <- selected_protein()
      if (is.null(pid)) {
        return(shiny::div(class = 'placeholder',
          'Select a protein from the database view.'))
      }
      shiny::req(con())
      core <- db_protein_core(con(), pid)
      if (nrow(core) == 0) {
        return(shiny::div(class = 'placeholder',
          sprintf('Protein %s not found in this run.', pid)))
      }

      # Null-gene fallback: use the accession as the display name (Module 08
      # follow-up 2026-07-14 -- proteins without a gene symbol showed "NA").
      gene <- if (is.na(core$gene_symbol) || !nzchar(core$gene_symbol)) {
        pid
      } else {
        core$gene_symbol
      }
      ortholog <- na_dash(core$ortholog_mapping_status)
      has_ortholog <- !identical(ortholog, 'no_ortholog') && ortholog != '-'

      # Which Module 06 databases ran this run (NULL = unknown -> fail-open, all
      # assumed enabled). Lets the annotation sections distinguish 'database not
      # enabled' from 'queried, found nothing' (spec 4.2.2 / 4.4).
      enabled_dbs <- db_enabled_dbs(con())

      # External gene-page links for the disease/drug sections (spec 4.4.2),
      # keyed on the human ortholog and shown only when one exists. DisGeNET's
      # public URL path survived its move to disgenet.com (verified 2026-07-14).
      has_entrez <- has_ortholog && !is.na(core$human_ortholog_entrez) &&
        nzchar(core$human_ortholog_entrez)
      has_symbol <- has_ortholog && !is.na(core$human_ortholog_symbol) &&
        nzchar(core$human_ortholog_symbol)
      disgenet_link <- if (has_entrez) shiny::a(
        class = 'section-link', target = '_blank',
        href = sprintf('https://disgenet.com/browser/0/1/0/%s/', core$human_ortholog_entrez),
        'DisGeNET ↗')
      dgidb_link <- if (has_symbol) shiny::a(
        class = 'section-link', target = '_blank',
        href = sprintf('https://www.dgidb.org/genes/%s', core$human_ortholog_symbol),
        'DGIdb ↗')

      # Section 1: header.
      header <- shiny::div(class = 'profile-header-wrap',
        shiny::actionLink(ns('back_to_db'), '← Back to database',
                          class = 'back-link'),
        shiny::div(class = 'profile-header',
          shiny::h3(gene),
          shiny::span(class = 'gene-accession', core$protein_id)),
        shiny::div(class = 'profile-sub',
          na_dash(core$protein_name), ' · ',
          shiny::a(href = sprintf('https://www.uniprot.org/uniprot/%s', core$protein_id),
                   target = '_blank', 'UniProt')))

      # Section 2: differential abundance (stat cards for 1 contrast, else table).
      ct <- db_protein_contrasts(con(), pid)
      da_body <- if (nrow(ct) == 0) {
        empty_note('No differential abundance result for this protein.')
      } else if (nrow(ct) == 1) {
        shiny::div(class = 'section-grid',
          stat_card('log2 fold change', shiny::span(
            class = if (!is.na(ct$log2_fc) && ct$log2_fc < 0) 'val-down' else 'val-up',
            fmt_fc(ct$log2_fc))),
          stat_card('adj. p-value (DEqMS)', fmt_p(ct$adj_pvalue)),
          stat_card('direction', dir_badge(ct$direction, ct$numerator, ct$denominator)),
          stat_card('contrast', ct$contrast))
      } else {
        capped_table(ct, c('Contrast', 'log2 FC', 'Adj p', 'Direction'),
          function(r) list(
            shiny::tags$td(r$contrast), shiny::tags$td(fmt_fc(r$log2_fc)),
            shiny::tags$td(fmt_p(r$adj_pvalue)),
            shiny::tags$td(dir_badge(r$direction, r$numerator, r$denominator))),
          cap = 50L)
      }

      # Section 3: data quality + QC flag notes.
      qc <- db_protein_qc_flags(con(), pid)
      dq_body <- shiny::tagList(
        shiny::div(class = 'section-grid',
          stat_card('detection', det_badge(core$detection_category)),
          stat_card('% imputed', if (is.na(core$imputation_fraction)) '-' else
                    sprintf('%.1f%%', 100 * core$imputation_fraction))),
        qc_notes_block(qc, core$detection_category))

      # Section 4: UniProt annotation.
      up <- db_protein_uniprot(con(), pid)
      anno_rows <- list(
        c('Function', if (nrow(up)) up$function_description else NA),
        c('Subcellular location', if (nrow(up)) up$subcellular_location else NA),
        c('Tissue expression', if (nrow(up)) up$tissue_expression else NA),
        c('Keywords', if (nrow(up)) up$keywords else NA))
      anno_blank <- function(r) is.na(r[2]) || !nzchar(as.character(r[2]))
      anno_body <- if (all(vapply(anno_rows, anno_blank, logical(1)))) {
        empty_note('No UniProt annotation available.')
      } else {
        shiny::tags$table(class = 'anno-table', shiny::tags$tbody(
          lapply(anno_rows, function(r) shiny::tags$tr(
            shiny::tags$th(r[1]), shiny::tags$td(na_dash(r[2]))))))
      }

      # Section 5: enriched terms containing this protein.
      et <- db_protein_enriched_terms(con(), pid)
      et_body <- if (nrow(et) == 0) {
        empty_note('No enriched terms contain this protein.')
      } else {
        # Each term name is a link into the biological-process term profile: the
        # onclick pushes the term_id to the module input `term_link`, which the
        # server returns as `term_selected` for the app to navigate on.
        capped_table(et, c('Term', 'Library', 'Leading edge'),
          function(r) list(
            shiny::tags$td(shiny::tags$a(
              clean_term_name(r$term_name), class = 'term-link',
              style = 'cursor: pointer;',
              onclick = sprintf("Shiny.setInputValue('%s', '%s', {priority: 'event'})",
                                ns('term_link'), r$term_id))),
            shiny::tags$td(r$library),
            shiny::tags$td(if (isTRUE(as.logical(r$is_leading_edge))) 'yes' else '')),
          cap = 25L)
      }

      # Section 6: diseases (database-gated, then ortholog-gated).
      dis_body <- if (!db_enabled('disgenet', enabled_dbs)) {
        empty_note('DisGeNET query was not enabled for this run.')
      } else if (!has_ortholog) {
        empty_note('No human ortholog available -- DisGeNET query not performed.')
      } else {
        dis <- db_protein_diseases(con(), pid)
        if (nrow(dis) == 0) {
          empty_note('No disease associations found.')
        } else {
          shiny::tagList(
            shiny::div(class = 'provenance', 'Queried via human ortholog.'),
            capped_table(dis, c('Disease', 'GDA score', 'Evidence', 'Pubs'),
              function(r) list(
                shiny::tags$td(r$disease_name),
                shiny::tags$td(formatC(r$gda_score, format = 'g', digits = 2)),
                shiny::tags$td(na_dash(r$evidence_index)),
                shiny::tags$td(na_dash(r$n_publications)))))
        }
      }

      # Section 7: drugs (database-gated, then ortholog-gated).
      drug_body <- if (!db_enabled('dgidb', enabled_dbs)) {
        empty_note('DGIdb query was not enabled for this run.')
      } else if (!has_ortholog) {
        empty_note('No human ortholog available -- DGIdb query not performed.')
      } else {
        dr <- db_protein_drugs(con(), pid)
        if (nrow(dr) == 0) {
          empty_note('No drug interactions found.')
        } else {
          capped_table(dr, c('Drug', 'Interaction', 'Approval', 'Sources'),
            function(r) list(
              shiny::tags$td(r$drug_name), shiny::tags$td(na_dash(r$interaction_type)),
              shiny::tags$td(na_dash(r$approval_status)), shiny::tags$td(na_dash(r$sources))),
            cap = 25L)
        }
      }

      # Section 8: chemical interactions (CTD, often long).
      # CTD interactions can run to thousands of rows, so this section is a
      # paged/searchable DT (output$chem_table) rather than a capped table.
      chem_body <- if (!db_enabled('ctd', enabled_dbs)) {
        empty_note('CTD query was not enabled for this run.')
      } else if (nrow(chem_data()) == 0) {
        empty_note('No chemical-gene interactions found.')
      } else {
        DT::DTOutput(ns('chem_table'))
      }

      # Section 9: PubMed co-occurrence (external search links).
      pm <- if (db_enabled('pubmed', enabled_dbs)) {
        db_protein_pubmed(con(), pid)
      } else {
        NULL
      }
      pm_body <- if (!db_enabled('pubmed', enabled_dbs)) {
        empty_note('PubMed query was not enabled for this run.')
      } else if (nrow(pm) == 0) {
        empty_note('No literature co-occurrence data.')
      } else {
        shiny::tags$table(class = 'mini-table',
          shiny::tags$thead(shiny::tags$tr(
            shiny::tags$th('Search term'), shiny::tags$th('Hits'),
            shiny::tags$th('PMI score'), shiny::tags$th(''))),
          shiny::tags$tbody(lapply(seq_len(nrow(pm)), function(i) {
            r <- pm[i, ]
            sym <- if (!is.na(r$human_symbol_used) && nzchar(r$human_symbol_used)) {
              r$human_symbol_used
            } else {
              na_dash(r$mouse_symbol_used)
            }
            url <- sprintf('https://pubmed.ncbi.nlm.nih.gov/?term=%s+AND+%s',
                           utils::URLencode(sym, reserved = TRUE),
                           utils::URLencode(r$search_term, reserved = TRUE))
            shiny::tags$tr(
              shiny::tags$td(r$search_term),
              shiny::tags$td(na_dash(r$hit_count)),
              shiny::tags$td(fmt_p(r$normalized_score)),
              shiny::tags$td(shiny::a(href = url, target = '_blank', 'PubMed')))
          })))
      }

      shiny::div(class = 'profile',
        header,
        section('Differential abundance', da_body),
        section('Data quality', dq_body),
        section('Functional annotation (UniProt)', anno_body),
        section('Enriched terms containing this protein', et_body),
        section(shiny::tagList('Disease associations (DisGeNET)', disgenet_link), dis_body),
        section(shiny::tagList('Druggability (DGIdb)', dgidb_link), drug_body),
        section('Chemical interactions (CTD)', chem_body),
        section('Literature co-occurrence (PubMed)', pm_body))
    })

    # Chemical interactions (Section 8) as a paged/searchable DT. Queried once
    # here and shared with the renderUI empty-vs-table decision above.
    chem_data <- shiny::reactive({
      shiny::req(selected_protein(), con())
      db_protein_chemicals(con(), selected_protein())
    })

    output$chem_table <- DT::renderDT({
      d <- chem_data()
      shiny::req(nrow(d) > 0)
      disp <- data.frame(
        Chemical = d$chemical_name,
        Actions  = prettify_ctd_actions(d$interaction_actions),
        Pubs     = d$n_publications,
        check.names = FALSE, stringsAsFactors = FALSE)
      DT::datatable(disp, rownames = FALSE, selection = 'none',
        class = 'stripe hover row-border compact',
        options = list(
          pageLength = 10,
          lengthMenu = list(c(10, 25, 50, -1), c('10', '25', '50', 'All')),
          scrollX = TRUE, order = list()))
    }, server = TRUE)

    list(
      back = shiny::reactive(input$back_to_db),
      term_selected = shiny::reactive(input$term_link))
  })
}


# --- Helpers ----------------------------------------------------------------

# CTD interaction_actions use a "qualifier^type" grammar joined by "|", e.g.
# "decreases^expression|increases^abundance" -> "decreases expression;
# increases abundance". Vectorised; NA -> "-".
prettify_ctd_actions <- function(x) {
  out <- gsub('\\|', '; ', gsub('\\^', ' ', x))
  out[is.na(out)] <- '-'
  out
}
