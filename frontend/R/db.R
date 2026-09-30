# =============================================================================
# Title:         db.R
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-13
# Last Modified: 2026-09-29
# Purpose:       Module 08 frontend data-access layer. Opens read-only SQLite
#                connections to a Module 07 results database and provides the
#                query functions each view needs. All SQL lives here so the view
#                modules stay presentation-only (Module 08 spec Section 3.3).
# Inputs:        A path to a prosift_results.db file (Module 07 output).
# Outputs:       Query functions returning data.frames; sourced by app.R.
# Usage:         con <- open_results_db(path); db_protein_table(con, 'KO_vs_WT')
# =============================================================================

# --- Connection lifecycle ---------------------------------------------------

#' Open a read-only connection to a results database.
#'
#' Read-only (SQLITE_RO) because the frontend never writes: it is a pure
#' presentation layer over Module 07's snapshot database.
#'
#' The path is validated first: SQLite silently treats an empty filename as a
#' private temporary database, so an empty/blank path would otherwise open a
#' blank connection instead of failing. A missing file is also caught here to
#' give a clearer message than the driver's generic "unable to open" error.
open_results_db <- function(path) {
  if (length(path) != 1 || is.na(path) || !nzchar(trimws(path))) {
    stop('open_results_db: a non-empty database path is required', call. = FALSE)
  }
  if (!file.exists(path)) {
    stop(sprintf('open_results_db: database file not found: %s', path),
         call. = FALSE)
  }
  DBI::dbConnect(RSQLite::SQLite(), path, flags = RSQLite::SQLITE_RO)
}

#' Close a connection if it is non-null and still valid (safe to call twice).
close_results_db <- function(con) {
  if (!is.null(con) && DBI::dbIsValid(con)) DBI::dbDisconnect(con)
  invisible(NULL)
}


# --- Metadata queries -------------------------------------------------------

#' Distinct contrasts present in the differential_abundance table, ordered.
db_contrasts <- function(con) {
  DBI::dbGetQuery(
    con, 'SELECT DISTINCT contrast FROM differential_abundance ORDER BY contrast'
  )$contrast
}

#' Split 'numerator_vs_denominator' contrast labels on their FIRST '_vs_'.
#'
#' Same rule Module 04 uses to build the fit ('numerator - denominator'), so a
#' positive log2 FC means higher in the numerator. Vectorised; a label without
#' a usable '_vs_' yields NA for both parts. Used only as a fallback for results
#' databases built before Module 04 wrote explicit numerator/denominator columns.
split_contrast_label <- function(contrast) {
  idx <- regexpr('_vs_', contrast, fixed = TRUE)
  ok  <- !is.na(idx) & idx > 1 & (idx + 4L) <= nchar(contrast)
  data.frame(
    numerator   = ifelse(ok, substr(contrast, 1L, idx - 1L), NA_character_),
    denominator = ifelse(ok, substr(contrast, idx + 4L, nchar(contrast)), NA_character_),
    stringsAsFactors = FALSE
  )
}

# TRUE when differential_abundance carries the explicit orientation columns.
.da_has_orientation <- function(con) {
  all(c('numerator', 'denominator') %in% DBI::dbListFields(con, 'differential_abundance'))
}

#' Orientation of one contrast: list(numerator, denominator).
#'
#' Prefers the Module 04 numerator/denominator columns (authoritative); falls
#' back to splitting the label for older databases that lack them.
db_contrast_groups <- function(con, contrast) {
  if (.da_has_orientation(con)) {
    g <- DBI::dbGetQuery(con, '
      SELECT DISTINCT numerator, denominator FROM differential_abundance
      WHERE contrast = ?', params = list(contrast))
    if (nrow(g) == 1) return(list(numerator = g$numerator, denominator = g$denominator))
  }
  g <- split_contrast_label(contrast)
  list(numerator = g$numerator, denominator = g$denominator)
}

# Append the contrast's numerator/denominator to a single-contrast result so a
# view can label 'up'/'down' by group. Constant per call (one contrast).
.with_orientation <- function(con, df, contrast) {
  g <- db_contrast_groups(con, contrast)
  df$numerator   <- rep(g$numerator,   nrow(df))
  df$denominator <- rep(g$denominator, nrow(df))
  df
}


# --- Run parameters (Module 07 run_parameters key/value table) --------------
# The frontend reads a few run-level params (the enrichment FDR threshold, the
# list of enabled databases) to avoid hardcoding defaults or misreading a
# disabled-database empty table as a biological zero. Everything here is
# fail-open: a database that predates the run_parameters table (or lacks a key)
# yields the caller's default, never an error.

#' Return the raw string `value` for one run_parameters key, or `default` if the
#' table or key is absent (or the query errors). Robust to a missing table so an
#' older/partial DB degrades rather than crashing on every render.
db_param <- function(con, key, default = NULL) {
  if (!DBI::dbExistsTable(con, 'run_parameters')) return(default)
  tryCatch({
    v <- DBI::dbGetQuery(
      con, 'SELECT value FROM run_parameters WHERE key = ? LIMIT 1',
      params = list(key))$value
    if (length(v) == 1 && !is.na(v)) v else default
  }, error = function(e) default)
}

#' The list of Module 06 databases that ran, parsed from the JSON-string
#' `databases.enabled` param. Returns a character vector, or NULL when the param
#' is absent/unparseable (NULL = 'unknown', which callers treat as fail-open:
#' every database is assumed enabled, never falsely reported 'not enabled').
db_enabled_dbs <- function(con) {
  raw <- db_param(con, 'databases.enabled', NULL)
  if (is.null(raw)) return(NULL)
  parsed <- tryCatch(jsonlite::fromJSON(raw), error = function(e) NULL)
  # JSON null or a parse failure is 'unknown' -> fail-open (NULL). An empty
  # array parses to a 0-length value -> character(0), i.e. all databases off.
  if (is.null(parsed)) return(NULL)
  as.character(parsed)
}

#' Was database `name` enabled for this run? TRUE when it is listed, FALSE when
#' the list is known and `name` is absent, TRUE when the list is unknown (NULL,
#' fail-open).
db_enabled <- function(name, enabled_dbs) {
  if (is.null(enabled_dbs)) return(TRUE)
  name %in% enabled_dbs
}


# --- Protein Database View (Module 08 Section 4.1) --------------------------

#' Assemble the protein-overview table for one contrast.
#'
#' One row per protein (the analytical spine). Statistical columns are joined
#' from differential_abundance for the requested contrast only; a protein with
#' no row for that contrast still appears (LEFT JOIN), with NA statistics. The
#' annotation-richness columns are aggregated from the Module 06 tables:
#'   - disease_count: number of DisGeNET associations
#'   - drug_count:    number of DGIdb interactions
#'   - top_pmi:       highest PubMed PMI (normalized_score) across search terms
#' Each Module 06 table carries a per-protein status row even when there is no
#' real record (no_ortholog / no_associations / no_interactions / etc), so the
#' aggregates filter to `..._query_status = 'success'` to count genuine records
#' only; a protein with no real record gets NULL (rendered as "--"). Per the
#' 2026-07-13 decision there is NO per-protein QC-flag column here; QC triage
#' uses detection_category and imputation_fraction (Section 4.1.1).
#'
#' The adjusted p-value is DEqMS (primary) with a limma fallback, matching the
#' Module 04 contract (Section 4.1.1).
db_protein_table <- function(con, contrast) {
  # Raw string (r'(...)') so the SQL "success" literals keep their single quotes.
  sql <- r'(
    SELECT
      p.protein_id                                         AS protein_id,
      p.gene_symbol                                        AS gene_symbol,
      u.protein_name                                       AS protein_name,
      d.log2_fc                                            AS log2_fc,
      COALESCE(d.deqms_adj_pvalue, d.limma_adj_pvalue)     AS adj_pvalue,
      d.direction                                          AS direction,
      d.significant                                        AS significant,
      dis.disease_count                                    AS disease_count,
      drg.drug_count                                       AS drug_count,
      pm.top_pmi                                           AS top_pmi,
      p.detection_category                                 AS detection_category,
      p.imputation_fraction                                AS imputation_fraction
    FROM proteins p
    LEFT JOIN differential_abundance d
      ON d.protein_id = p.protein_id AND d.contrast = ?
    LEFT JOIN uniprot_annotations u
      ON u.protein_id = p.protein_id
    LEFT JOIN (
      SELECT protein_id, COUNT(*) AS disease_count
      FROM disease_associations
      WHERE disgenet_query_status = 'success'
      GROUP BY protein_id
    ) dis ON dis.protein_id = p.protein_id
    LEFT JOIN (
      SELECT protein_id, COUNT(*) AS drug_count
      FROM drug_interactions
      WHERE dgidb_query_status = 'success'
      GROUP BY protein_id
    ) drg ON drg.protein_id = p.protein_id
    LEFT JOIN (
      SELECT protein_id, MAX(normalized_score) AS top_pmi
      FROM pubmed_cooccurrence
      WHERE pubmed_query_status = 'success'
      GROUP BY protein_id
    ) pm ON pm.protein_id = p.protein_id
    ORDER BY (p.gene_symbol IS NULL OR p.gene_symbol = ''), p.gene_symbol
  )'
  .with_orientation(con, DBI::dbGetQuery(con, sql, params = list(contrast)), contrast)
}


# --- Protein Profile View (Module 08 Section 4.2) ---------------------------
# One query per profile section. All are parameterised on protein_id, so the
# view module stays presentation-only. Each returns a data.frame (0 rows when
# the section is empty); the module decides the empty-state message.

#' Section 1 + 3 core: identity, ortholog status, and the two per-protein QC
#' signals (detection_category, imputation_fraction). protein_name is joined
#' from UniProt. One row (0 rows if the protein_id is unknown).
db_protein_core <- function(con, pid) {
  sql <- '
    SELECT p.protein_id, p.gene_symbol, u.protein_name,
           p.human_ortholog_symbol, p.human_ortholog_entrez,
           p.ortholog_mapping_status, p.detection_category, p.imputation_fraction
    FROM proteins p
    LEFT JOIN uniprot_annotations u ON u.protein_id = p.protein_id
    WHERE p.protein_id = ?
  '
  DBI::dbGetQuery(con, sql, params = list(pid))
}

#' Section 2: differential abundance across every contrast in the run. DEqMS
#' adjusted p-value (primary) with a limma fallback.
db_protein_contrasts <- function(con, pid) {
  sql <- '
    SELECT contrast, log2_fc,
           COALESCE(deqms_adj_pvalue, limma_adj_pvalue) AS adj_pvalue,
           direction, significant
    FROM differential_abundance
    WHERE protein_id = ?
    ORDER BY contrast
  '
  out <- DBI::dbGetQuery(con, sql, params = list(pid))
  # Attach each contrast's orientation so the view can say "higher in KO"
  # instead of a bare "up" (whose meaning depends on the contrast order).
  groups <- do.call(rbind, lapply(out$contrast, function(ct) {
    g <- db_contrast_groups(con, ct)
    data.frame(numerator = g$numerator, denominator = g$denominator,
               stringsAsFactors = FALSE)
  }))
  if (is.null(groups)) {
    groups <- data.frame(numerator = character(0), denominator = character(0))
  }
  cbind(out, groups)
}

#' Section 3 QC notes: for each sample the protein was OBSERVED in (non-imputed),
#' the sample-level flags that fired. A flag on a sample where this protein was
#' imputed is not evidence about this protein, so imputed cells are excluded
#' (Module 08 spec Section 4.2.1, implementation note 3). "group" is a SQLite
#' reserved word, hence the quoting.
db_protein_qc_flags <- function(con, pid) {
  sql <- r'(
    SELECT sa.sample_id, sa."group" AS grp,
           f.flag_low_detection, f.flag_extreme_median,
           f.flag_pca_outlier, f.flag_low_correlation, f.n_flags
    FROM sample_abundances sa
    JOIN sample_qc_flags f ON f.sample_id = sa.sample_id
    WHERE sa.protein_id = ? AND sa.imputation_status = 'observed'
    ORDER BY sa.sample_id
  )'
  DBI::dbGetQuery(con, sql, params = list(pid))
}

#' Section 4: UniProt functional annotation (one row, possibly all-NA).
db_protein_uniprot <- function(con, pid) {
  sql <- '
    SELECT function_description, subcellular_location, tissue_expression,
           keywords, uniprot_query_status
    FROM uniprot_annotations
    WHERE protein_id = ?
  '
  DBI::dbGetQuery(con, sql, params = list(pid))
}

#' Section 5: GO/Reactome terms that contain this protein AND were significantly
#' enriched (in_significant_set) in at least one contrast. Collapsed across
#' contrasts; is_leading_edge is TRUE if leading-edge in any contrast.
db_protein_enriched_terms <- function(con, pid) {
  sql <- '
    SELECT term_id, term_name, library, MAX(is_leading_edge) AS is_leading_edge
    FROM protein_term_mapping
    WHERE protein_id = ? AND in_significant_set = 1
    GROUP BY term_id, term_name, library
    ORDER BY library, term_name
  '
  DBI::dbGetQuery(con, sql, params = list(pid))
}

#' Section 6: DisGeNET disease associations (real records only), best first.
db_protein_diseases <- function(con, pid) {
  sql <- r'(
    SELECT disease_name, gda_score, disease_type, evidence_index, n_publications
    FROM disease_associations
    WHERE protein_id = ? AND disgenet_query_status = 'success'
    ORDER BY gda_score DESC
  )'
  DBI::dbGetQuery(con, sql, params = list(pid))
}

#' Section 7: DGIdb drug interactions (real records only).
db_protein_drugs <- function(con, pid) {
  sql <- r'(
    SELECT drug_name, interaction_type, approval_status, sources
    FROM drug_interactions
    WHERE protein_id = ? AND dgidb_query_status = 'success'
    ORDER BY drug_name
  )'
  DBI::dbGetQuery(con, sql, params = list(pid))
}

#' Section 8: CTD chemical interactions (real records only). Can be long for
#' well-studied proteins; the module shows the first rows plus a total count.
db_protein_chemicals <- function(con, pid) {
  sql <- r'(
    SELECT chemical_name, interaction_actions, interaction_text, n_publications
    FROM chemical_interactions
    WHERE protein_id = ? AND ctd_query_status = 'success'
    ORDER BY n_publications DESC, chemical_name
  )'
  DBI::dbGetQuery(con, sql, params = list(pid))
}

#' Section 9: PubMed literature co-occurrence, one row per configured search
#' term. hit_count sums the mouse + human symbol hits; symbols are returned so
#' the module can build the external PubMed search links.
db_protein_pubmed <- function(con, pid) {
  sql <- '
    SELECT search_term,
           COALESCE(mouse_hit_count, 0) + COALESCE(human_hit_count, 0) AS hit_count,
           normalized_score, mouse_symbol_used, human_symbol_used
    FROM pubmed_cooccurrence
    WHERE protein_id = ?
    ORDER BY normalized_score DESC
  '
  DBI::dbGetQuery(con, sql, params = list(pid))
}


# --- Biological Process View (Module 08 Section 4.3) ------------------------
# Enrichment is stored one row per (term, analysis_type, contrast). The term
# list pivots the ORA and GSEA rows of a contrast onto one row per term. A term
# is 'significant' at adj_pvalue < alpha, where alpha is the run's configured
# enrichment.fdr_threshold (see db_param; Module 05 convention). Named SQL
# parameters (:contrast, :alpha) avoid positional-binding hazards.

#' Term list for one contrast: one row per GO/Reactome term with the ORA and
#' GSEA statistics side by side, plus per-analysis significance flags. Terms
#' with only an ORA or only a GSEA result still appear (LEFT JOINs). `alpha` is
#' the FDR cutoff for the significance flags (default 0.05).
db_term_list <- function(con, contrast, alpha = 0.05) {
  sql <- r'(
    SELECT t.term_id, t.term_name, t.library, t.size,
           ora.adj_pvalue AS ora_adj_p, ora.odds_ratio, ora.overlap_size,
           gsea.adj_pvalue AS gsea_adj_p, gsea.nes,
           COALESCE(ora.sig, 0) AS ora_sig, COALESCE(gsea.sig, 0) AS gsea_sig
    FROM (
      SELECT term_id, MAX(term_name) AS term_name, MAX(library) AS library,
             MAX(gene_set_size) AS size
      FROM enrichment_results WHERE contrast = :contrast GROUP BY term_id
    ) t
    LEFT JOIN (
      SELECT term_id, adj_pvalue, odds_ratio, overlap_size,
             CASE WHEN adj_pvalue < :alpha THEN 1 ELSE 0 END AS sig
      FROM enrichment_results WHERE contrast = :contrast AND analysis_type = 'ORA'
    ) ora ON ora.term_id = t.term_id
    LEFT JOIN (
      SELECT term_id, adj_pvalue, enrichment_score AS nes,
             CASE WHEN adj_pvalue < :alpha THEN 1 ELSE 0 END AS sig
      FROM enrichment_results WHERE contrast = :contrast AND analysis_type = 'GSEA'
    ) gsea ON gsea.term_id = t.term_id
    ORDER BY t.term_name
  )'
  DBI::dbGetQuery(con, sql, params = list(contrast = contrast, alpha = alpha))
}

#' Term identity (name, id, library, size). One row.
db_term_core <- function(con, term_id) {
  sql <- '
    SELECT term_id, MAX(term_name) AS term_name, MAX(library) AS library,
           MAX(gene_set_size) AS size
    FROM enrichment_results WHERE term_id = ? GROUP BY term_id
  '
  DBI::dbGetQuery(con, sql, params = list(term_id))
}

#' Per-term ORA + GSEA statistics across every contrast (one row per
#' contrast x analysis_type). The module lays them out side by side.
db_term_stats <- function(con, term_id) {
  sql <- '
    SELECT contrast, analysis_type, adj_pvalue, odds_ratio, overlap_size,
           gene_set_size, enrichment_score AS nes
    FROM enrichment_results
    WHERE term_id = ?
    ORDER BY contrast, analysis_type
  '
  DBI::dbGetQuery(con, sql, params = list(term_id))
}

#' Member proteins of a term for one contrast: the mini protein-database table
#' (gene + DA stats + ORA-set / leading-edge flags). Membership and the flags
#' come from protein_term_mapping; the stats from differential_abundance.
db_term_members <- function(con, term_id, contrast) {
  sql <- r'(
    SELECT ptm.protein_id, ptm.gene_symbol,
           d.log2_fc,
           COALESCE(d.deqms_adj_pvalue, d.limma_adj_pvalue) AS adj_pvalue,
           d.direction, ptm.in_significant_set, ptm.is_leading_edge
    FROM protein_term_mapping ptm
    LEFT JOIN differential_abundance d
      ON d.protein_id = ptm.protein_id AND d.contrast = ?
    WHERE ptm.term_id = ? AND ptm.contrast = ?
    ORDER BY (ptm.gene_symbol IS NULL OR ptm.gene_symbol = ''), ptm.gene_symbol
  )'
  .with_orientation(
    con, DBI::dbGetQuery(con, sql, params = list(contrast, term_id, contrast)),
    contrast)
}
