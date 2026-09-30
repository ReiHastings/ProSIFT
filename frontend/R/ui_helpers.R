# =============================================================================
# Title:         ui_helpers.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-14
# Last Modified: 2026-09-29
# Purpose:       Small shared UI formatters and building blocks used by more
#                than one Module 08 view module (profile card, biological
#                process view): value formatters, coloured badges, stat cards,
#                and section wrappers. Styling lives in www/prosift.css.
# Inputs:        Scalars from query results (formatters expect length-1 values).
# Outputs:       Character strings or shiny tag objects; sourced by Shiny.
# =============================================================================

# --- Value formatters (expect scalars) --------------------------------------

fmt_fc <- function(x) if (is.na(x)) '-' else sprintf('%+.2f', x)
fmt_p  <- function(x) if (is.na(x)) '-' else formatC(x, format = 'g', digits = 2)

na_dash <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x) || !nzchar(as.character(x))) {
    '-'
  } else {
    as.character(x)
  }
}

# MSigDB-style term names -> readable ("GOBP_ADAPTIVE_THERMOGENESIS" ->
# "adaptive thermogenesis").
clean_term_name <- function(x) {
  if (is.na(x)) return('-')
  x <- sub('^(GOBP|GOCC|GOMF|REACTOME|KEGG|WP|HP)_', '', x)
  tolower(gsub('_', ' ', x))
}


# --- Coloured badges (styling in prosift.css) -------------------------------

# Relabel direction values by the group a protein is higher in ('up' ->
# 'higher in KO', 'down' -> 'higher in WT'), so they read correctly whatever the
# contrast order. Vectorised over `d`; NA direction -> 'ns'. Groups are scalars
# (one contrast); if either is missing (NULL/NA/empty) the bare value is kept.
direction_label <- function(d, numerator = NA, denominator = NA) {
  d <- ifelse(is.na(d), 'ns', d)
  known <- function(x) length(x) == 1 && !is.na(x) && nzchar(x)
  if (!known(numerator) || !known(denominator)) return(d)
  ifelse(d == 'up', paste('higher in', numerator),
         ifelse(d == 'down', paste('higher in', denominator), d))
}

# DT styleEqual colours for direction_label() output: green / red / grey for
# up / down / ns, keyed by whatever labels this contrast produces.
direction_style <- function(numerator = NA, denominator = NA) {
  DT::styleEqual(direction_label(c('up', 'down', 'ns'), numerator, denominator),
                 c('#0F6E56', '#A32D2D', '#8A8F98'))
}

dir_badge <- function(d, numerator = NA, denominator = NA) {
  d <- if (length(d) == 0 || is.na(d)) 'ns' else d
  cls <- switch(d, up = 'badge-up', down = 'badge-down', 'badge-ns')
  shiny::span(class = paste('badge', cls), direction_label(d, numerator, denominator))
}

# Plain-language reading of a positive log2 FC / NES for one contrast, from the
# list(numerator, denominator) returned by db_contrast_groups().
contrast_direction_text <- function(groups) {
  if (is.null(groups) || is.na(groups$numerator) || is.na(groups$denominator)) {
    return('log2 FC > 0 and NES > 0 = higher in the first-named group')
  }
  sprintf('log2 FC > 0 and NES > 0 = higher in %s than %s',
          groups$numerator, groups$denominator)
}

det_badge <- function(cat) {
  shiny::span(class = 'badge det-badge', na_dash(cat))
}


# --- Layout building blocks -------------------------------------------------

stat_card <- function(label, value) {
  shiny::div(class = 'stat-card',
    shiny::div(class = 'stat-label', label),
    shiny::div(class = 'stat-value', value))
}

section <- function(title, ...) {
  shiny::div(class = 'section',
    shiny::div(class = 'section-title', title),
    shiny::tagList(...))
}

empty_note <- function(msg) shiny::div(class = 'empty-note', msg)
