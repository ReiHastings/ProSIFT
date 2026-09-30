#!/usr/bin/env python3
"""
Title:         differential_abundance.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-03-30
Last Modified: 2026-09-29
Purpose:       Module 04 DIFFERENTIAL_ABUNDANCE process. Parses and validates
               contrasts from params.yml, fits a linear model per protein using
               limma empirical Bayes (trend=TRUE) via rpy2, applies DEqMS
               peptide-count-aware variance correction, and produces a per-protein
               results table (17 columns), a plain-text summary, and volcano / MA
               diagnostic plots (static PNG + interactive HTML) for each contrast.
               Supports optional mean-shift sample quarantine with dual
               (primary/sensitivity) reporting (Section 4.9 of the module spec).
Inputs:
  --matrix       {run_id}.imputed_matrix.parquet    (Module 03 IMPUTE)
  --metadata     {run_id}.validated_metadata.parquet (Module 01 VALIDATE_INPUTS)
  --id-mapping   {run_id}.id_mapping.parquet         (Module 01 UNIPROT_MAPPING)
  --params       {run_id}_params.yml
Outputs:
  {run_id}.diff_abundance_results.parquet          (PRIMARY; consumed downstream)
  {run_id}.diff_abundance_results.csv
  {run_id}.diff_abundance_results.sensitivity.*    (only when quarantine active)
  {run_id}.diff_abundance_summary.txt
  {run_id}.analysis_provenance.txt                 (always; records quarantine decision)
  {run_id}.{contrast}.volcano_plot.png/.html  (one pair per contrast; primary)
  {run_id}.{contrast}.ma_plot.png/.html       (one pair per contrast; primary)
  {run_id}.{contrast}.{volcano,ma}_plot.sensitivity.*  (only when quarantine active)
Usage:
  differential_abundance.py \
    --matrix     CTXcyto_WT_vs_CTXcyto_KO.imputed_matrix.parquet \
    --metadata   CTXcyto_WT_vs_CTXcyto_KO.validated_metadata.parquet \
    --id-mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \
    --params     CTXcyto_WT_vs_CTXcyto_KO_params.yml \
    --run-id     CTXcyto_WT_vs_CTXcyto_KO \
    --outdir     .
"""

import argparse
import datetime
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

# rpy2 (embedded R) is imported lazily inside _run_one_contrast_r, NOT at module
# top. Importing rpy2.robjects starts embedded R, which segfaults where R is not
# linked -- an uncatchable native crash that would make this module un-importable
# for the pure-Python tests, --help, or CI. Lazy loading keeps the module import
# side-effect-free; the R stack is only touched when a fit is actually run.

from prosift_plot_utils import save_plot


# ============================================================
# CONSTANTS
# ============================================================

# Colors for volcano / MA plots -- fixed per direction, not group-based
_COLOR_UP = "#d62728"  # D3 red: significant, positive FC
_COLOR_DN = "#1f77b4"  # D3 blue: significant, negative FC
_COLOR_NS = "#cccccc"  # gray: not significant

_PEPTIDE_PREFIX = "peptide_count_"


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="differential_abundance.py",
        description=(
            "ProSIFT Module 04 DIFFERENTIAL_ABUNDANCE: "
            "limma + DEqMS differential abundance analysis"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  differential_abundance.py \\\n"
            "    --matrix     run.imputed_matrix.parquet \\\n"
            "    --metadata   run.validated_metadata.parquet \\\n"
            "    --id-mapping run.id_mapping.parquet \\\n"
            "    --params     run_params.yml \\\n"
            "    --run-id     run \\\n"
            "    --outdir     ."
        ),
    )
    parser.add_argument("--matrix",     required=True,
                        help="Imputed abundance matrix (Parquet)")
    parser.add_argument("--metadata",   required=True,
                        help="Validated metadata (Parquet)")
    parser.add_argument("--id-mapping", required=True, dest="id_mapping",
                        help="ID mapping table (Parquet, from UNIPROT_MAPPING)")
    parser.add_argument("--params",     required=True,
                        help="Run params.yml")
    parser.add_argument("--run-id",     required=True, dest="run_id",
                        help="Run identifier (used as output file prefix)")
    parser.add_argument("--outdir",     required=True,
                        help="Output directory")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_params(params_path: Path) -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def extract_abundance_and_peptide_cols(
    matrix_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Split the imputed matrix into abundance and peptide count DataFrames.

    Returns:
        abund_df   -- index=protein_id, columns=bare sample IDs (prefix stripped)
        pep_df     -- index=protein_id, columns=peptide_count_{sample_id} originals
                      (empty DataFrame if no peptide count columns present)
        sample_ids -- ordered list of bare sample IDs matching abund_df columns
    """
    abundance_prefix = params["input"].get("abundance_prefix", "")

    all_cols     = matrix_df.columns.tolist()
    peptide_cols = [c for c in all_cols if c.startswith(_PEPTIDE_PREFIX)]
    abund_cols   = [
        c for c in all_cols
        if c != "protein_id" and not c.startswith(_PEPTIDE_PREFIX)
    ]

    if not abund_cols:
        raise ValueError("No abundance columns found in imputed matrix.")

    sample_ids = (
        [c[len(abundance_prefix):] for c in abund_cols]
        if abundance_prefix
        else list(abund_cols)
    )

    abund_df = matrix_df.set_index("protein_id")[abund_cols].copy()
    abund_df.columns = sample_ids

    pep_df = (
        matrix_df.set_index("protein_id")[peptide_cols].copy()
        if peptide_cols
        else pd.DataFrame(index=matrix_df.set_index("protein_id").index)
    )

    return abund_df, pep_df, sample_ids


def build_group_map(metadata_df: pd.DataFrame, params: dict) -> dict[str, str]:
    """Return {sample_id: group_label} from validated metadata."""
    group_col = params["design"]["group_column"]
    if group_col not in metadata_df.columns:
        raise ValueError(
            f"design.group_column '{group_col}' not found in metadata. "
            f"Available columns: {metadata_df.columns.tolist()}"
        )
    return dict(
        zip(
            metadata_df["sample_id"].astype(str),
            metadata_df[group_col].astype(str),
        )
    )


# ============================================================
# CONTRAST PARSING AND VALIDATION
# ============================================================

# R reserved words. make.names() appends a dot to each of these, so a design-
# matrix column literally named one of them is non-syntactic and makeContrasts()
# rejects it.
_R_RESERVED_WORDS = frozenset({
    "if", "else", "repeat", "while", "function", "for", "in", "next", "break",
    "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA",
    "NA_integer_", "NA_real_", "NA_complex_", "NA_character_",
})

# A syntactically valid R name: letters, digits, '.', '_'; starts with a letter,
# or with a dot that is NOT followed by a digit. This mirrors 'make.names(x) == x',
# which is exactly the validity check limma's makeContrasts() applies to every
# design-matrix column name.
#
# DELIBERATELY ASCII-ONLY -- do not 'fix' this to a Unicode letter class.
# R's make.names() defines 'letter' by the locale of the running R session, so
# whether a label like 'cafe' with an accent is a valid R name depends on where
# the fit runs (UTF-8 locale: accepted; C/POSIX locale: rejected). ASCII-only is
# over-strict in a UTF-8 locale and exactly right in a C locale, so it fails safe
# in both: worst case a clear Python error telling you to rename a group. A
# Unicode-permissive class would instead fail UNSAFE in a C locale -- passing
# here and then dying inside makeContrasts() on the cluster, which is the exact
# failure this guard exists to prevent, and which would reproduce in only one of
# the two environments. If non-ASCII group labels are ever genuinely needed, the
# correct fix is not a better regex but an rpy2 round-trip that asks the R
# session itself whether make.names(x) == x.
_R_SYNTACTIC_NAME = re.compile(r"^(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*$")


def is_syntactic_r_name(name: str) -> bool:
    """
    Return True iff `name` is a syntactically valid R name (equivalently,
    make.names(name) == name in R).

    Group labels must satisfy this because the R fit assigns each label as a
    design-matrix column name, and limma's makeContrasts() errors on any
    non-syntactic level name. That check runs over the FULL levels vector (every
    group with samples in the run), not only the two groups named in a given
    contrast, so every such label is validated, not just the contrasted ones.

    ASCII-only by design; see the note on _R_SYNTACTIC_NAME above.
    """
    return bool(_R_SYNTACTIC_NAME.match(name)) and name not in _R_RESERVED_WORDS


def split_contrast(contrast_user: str) -> tuple[str, str]:
    """
    Split a 'numerator_vs_denominator' contrast label into its two group names.

    Splits on the FIRST '_vs_' (so a denominator may itself contain '_vs_').
    The fitted contrast is 'numerator - denominator', so a positive log2 fold
    change always means higher abundance in the numerator group. This is the
    single source of truth for that orientation; every direction label written
    by this module is derived from it.
    """
    idx = contrast_user.find("_vs_")
    if idx == -1:
        raise ValueError(
            f"Contrast '{contrast_user}' does not contain '_vs_' delimiter. "
            "Use format 'numerator_vs_denominator' (e.g., 'KO_vs_WT')."
        )
    numerator   = contrast_user[:idx]
    denominator = contrast_user[idx + 4:]
    if not numerator or not denominator:
        raise ValueError(
            f"Contrast '{contrast_user}': numerator or denominator is empty after "
            "splitting on '_vs_'."
        )
    return numerator, denominator


def direction_statement(numerator: str, denominator: str) -> str:
    """Plain-language reading of a positive log2 fold change for one contrast."""
    return f"log2 FC > 0 = higher in {numerator} than {denominator}"


def parse_and_validate_contrasts(
    params: dict,
    available_groups: list[str],
) -> list[tuple[str, str, str, str]]:
    """
    Parse and validate contrast strings from params.yml design.contrasts.

    Each contrast string must use the 'numerator_vs_denominator' format
    (split on first occurrence of '_vs_'). Both group names must be present in
    `available_groups`.

    `available_groups` MUST be the groups whose samples are actually present in
    the abundance matrix (main()'s `unique_groups`), NOT every value in the
    metadata group column. That list is what becomes the R factor's levels
    vector and hence the design-matrix column names, so it is exactly the set R
    validates and exactly the set a contrast can legally name. Passing the wider
    metadata column instead has two failure modes: a non-syntactic group with no
    samples in the run is rejected although R would never see it, and a contrast
    naming a metadata-only group passes here and then dies in makeContrasts() on
    an undefined variable.

    Returns a list of 4-tuples:
        (contrast_user, numerator, denominator, r_contrast_str)
        e.g. ("KO_vs_WT", "KO", "WT", "KO - WT")

    Raises ValueError with a specific message on any validation failure.
    """
    group_col = params["design"]["group_column"]
    available_groups = sorted(set(available_groups))

    # --- Validate group labels are valid R names ---
    # Every group label becomes a design-matrix column name in the R fit, and
    # limma's makeContrasts() rejects any non-syntactic level name, checking the
    # ENTIRE levels vector (not only the two groups named in a contrast). A label
    # like 'WT-A' or '5xFAD' would otherwise abort the fit with a cryptic R error.
    # Validate all groups up front against 'make.names(x) == x' and fail clearly.
    bad_groups = [g for g in available_groups if not is_syntactic_r_name(g)]
    if bad_groups:
        raise ValueError(
            f"Group label(s) {bad_groups} in metadata column '{group_col}' are "
            "not valid R names and would break the differential-abundance model "
            "(group labels become design-matrix column names). Use only ASCII "
            "letters, digits, '.', and '_', start with a letter, and avoid R "
            "reserved words (e.g. 'TRUE', 'NA'). Note that accented and "
            "non-Latin letters are rejected deliberately, even though R may "
            "accept them in some locales. Rename the groups before running."
        )

    raw_contrasts = params.get("design", {}).get("contrasts", [])
    if not raw_contrasts:
        raise ValueError(
            "No contrasts defined in params.yml under design.contrasts. "
            "Add at least one contrast in 'numerator_vs_denominator' format "
            "(e.g., 'KO_vs_WT')."
        )

    parsed: list[tuple[str, str, str, str]] = []
    for contrast_user in raw_contrasts:
        # Split on first _vs_ to handle (unlikely) group names containing '_vs_'
        numerator, denominator = split_contrast(contrast_user)

        # Membership is checked against the samples-present groups, so a group
        # that exists in the metadata but contributes no samples to this run is
        # correctly rejected here rather than in R (it has no design-matrix
        # column, so makeContrasts() would fail on an undefined variable).
        for name, role in [(numerator, "numerator"), (denominator, "denominator")]:
            if name not in available_groups:
                raise ValueError(
                    f"Contrast '{contrast_user}': {role} group '{name}' not found "
                    f"among the groups with samples in this run (metadata column "
                    f"'{group_col}'). Available groups: {available_groups}"
                )

        r_contrast_str = f"{numerator} - {denominator}"
        parsed.append((contrast_user, numerator, denominator, r_contrast_str))

    return parsed


# ============================================================
# SAMPLE QUARANTINE (mean-shift indicator) VALIDATION + PROVENANCE
# ============================================================

def normalize_quarantine_samples(raw: object) -> list[str]:
    """
    Coerce the params value to a de-duplicated, order-preserving list of
    sample-id strings.

    Accepts None (-> []), a bare string (-> single-element list, so a YAML
    scalar like `quarantine_samples: HIPcyto_WT-3` does not explode into
    characters), or a list/tuple. Any other type is a configuration error.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "differential_abundance.quarantine_samples must be a list of sample "
            f"IDs (or a single string), got {type(raw).__name__}."
        )
    # dict.fromkeys preserves first-seen order while removing duplicates, which
    # would otherwise create collinear indicator columns in the design matrix.
    return list(dict.fromkeys(str(s) for s in raw))


def parse_bool_param(value: object, default: bool = False) -> bool:
    """
    Parse a boolean parameter robustly. Guards against a quoted YAML scalar
    (e.g. `robust_ebayes: "false"`) becoming a truthy non-empty string.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "on", "1")
    return bool(value)


def validate_quarantine(
    quarantine_samples: list[str],
    primary_analysis: str,
    sample_ids: list[str],
    group_map: dict[str, str],
    contrasts: list[tuple[str, str, str, str]],
    min_samples_per_group: int,
) -> None:
    """
    Validate the mean-shift quarantine parameters (module spec Section 4.9).

    Raises ValueError on: bad primary_analysis value; quarantine id not present
    in the run; primary_analysis='quarantined' with no quarantined samples; or a
    contrast group left with fewer than min_samples_per_group after quarantine.
    """
    # 1) primary_analysis must be a recognized value
    if primary_analysis not in ("full", "quarantined"):
        raise ValueError(
            f"differential_abundance.primary_analysis must be 'full' or "
            f"'quarantined', got '{primary_analysis}'."
        )

    # 2) every quarantined id must exist among this run's samples
    unknown = [s for s in quarantine_samples if s not in set(sample_ids)]
    if unknown:
        raise ValueError(
            f"quarantine_samples not found in this run's samples: {unknown}. "
            f"Available: {sample_ids}"
        )

    # 3) 'quarantined' primary requires something to quarantine
    if primary_analysis == "quarantined" and not quarantine_samples:
        raise ValueError(
            "primary_analysis='quarantined' requires a non-empty "
            "quarantine_samples list."
        )

    # 4) each contrast group must retain enough samples after quarantine
    if quarantine_samples:
        qset = set(quarantine_samples)
        for contrast_user, numerator, denominator, _ in contrasts:
            for grp in (numerator, denominator):
                remaining = [
                    s for s in sample_ids
                    if group_map.get(s) == grp and s not in qset
                ]
                if len(remaining) < min_samples_per_group:
                    raise ValueError(
                        f"Contrast '{contrast_user}': group '{grp}' has "
                        f"{len(remaining)} sample(s) after quarantine "
                        f"(minimum {min_samples_per_group}). Cannot quarantine a "
                        f"group below estimability."
                    )


def write_provenance(
    run_id: str,
    outdir: Path,
    quarantine_samples: list[str],
    primary_key: str,
    sample_ids: list[str],
    groups: list[str],
    method_used: str,
    robust_ebayes: bool,
) -> None:
    """
    Always-written analysis provenance record (module spec Section 4.9). Records
    the quarantine decision so the exclusion travels with the results even when
    no sample is quarantined ('no samples quarantined').
    """
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    qset = set(quarantine_samples)

    def _counts(exclude: set) -> str:
        counts: dict[str, int] = {}
        for s, g in zip(sample_ids, groups):
            if s not in exclude:
                counts[g] = counts.get(g, 0) + 1
        return ", ".join(f"{g}={counts[g]}" for g in sorted(counts))

    lines = [
        "========================================",
        "ANALYSIS PROVENANCE",
        "========================================",
        "",
        f"Run:                 {run_id}",
        f"Date:                {now}",
        f"Method:              {method_used}  "
        f"(eBayes trend=TRUE, robust={'TRUE' if robust_ebayes else 'FALSE'})",
        "Multiple testing:    Benjamini-Hochberg (pinned explicitly), applied "
        "within each contrast",
        "",
    ]
    if quarantine_samples:
        lines += [
            f"Quarantined samples: {', '.join(quarantine_samples)}",
            f"Primary analysis:    {primary_key}",
            "Method note:         mean-shift indicator (per-sample 0/1 design "
            "column); equivalent to case-deletion for the contrast (She & Owen 2011).",
            "",
            "n per group:",
            f"  full:              {_counts(set())}",
            f"  quarantined:       {_counts(qset)}",
        ]
    else:
        lines += [
            "Quarantined samples: none",
            "Primary analysis:    full (single analysis)",
            "",
            "n per group:",
            f"  full:              {_counts(set())}",
        ]

    path = outdir / f"{run_id}.analysis_provenance.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"  Saved: {path.name}")


# ============================================================
# PEPTIDE COUNT SUMMARIZATION
# ============================================================

def summarize_peptide_counts(pep_df: pd.DataFrame) -> pd.Series:
    """
    Per-protein peptide count: minimum of nonzero values across all samples.

    Counts of 0 correspond to imputed positions (no DIA-NN detection); these
    are excluded from the minimum. Every protein is guaranteed to have at least
    one nonzero value because Module 01's detection filter removes ABSENT
    proteins.

    Returns pd.Series with index=protein_id, dtype int64.
    """
    pep_values = pep_df.values.astype(float)
    pep_values[pep_values == 0] = np.nan        # treat zeros as "no measurement"
    min_nonzero = np.nanmin(pep_values, axis=1)  # shape: (n_proteins,)

    return pd.Series(
        min_nonzero.astype(np.int64),
        index=pep_df.index,
        name="n_peptides",
    )


# ============================================================
# STATISTICAL ANALYSIS -- rpy2 bridge
# ============================================================

def _run_one_contrast_r(
    abund_df: pd.DataFrame,
    groups: list[str],
    unique_groups: list[str],
    pep_counts: "pd.Series | None",
    r_contrast_str: str,
    use_deqms: bool,
    quarantine_ids: list[str],
    robust_ebayes: bool,
) -> tuple[dict[str, pd.DataFrame], str]:
    """
    Run limma + DEqMS (or limma only) for one contrast via rpy2.

    All data is passed to R via the global environment with 'prosift_' prefix.
    R code is executed as string blocks. Results are converted back to pandas
    via pandas2ri.

    Parameters
    ----------
    abund_df      : index=protein_id, columns=sample_ids. No NaN.
    groups        : group label per sample, aligned to abund_df.columns.
    unique_groups : unique group levels (controls factor level order in design).
    pep_counts    : per-protein min-nonzero peptide count. None if limma-only.
    r_contrast_str: R-side contrast expression, e.g. "KO - WT".
    use_deqms     : True to run spectraCounteBayes after eBayes.

    quarantine_ids: bare sample_ids to model out via a mean-shift indicator (Q2).
    robust_ebayes : pass robust=TRUE to eBayes (protein-level; default False).

    Returns
    -------
    results     : dict mapping analysis name -> raw R-column DataFrame. Always has
                  key "full"; also "quarantined" when quarantine_ids is non-empty.
                  Columns: protein_id, logFC, AveExpr, t, P.Value, adj.P.Val, B
                  (plus sca.t, sca.P.Value, sca.adj.pval, count if DEqMS).
    method_used : "DEqMS" or "limma".
    """
    # --- Import rpy2 lazily (see module header) ---
    # Deferred to call time so that merely importing this module never starts
    # embedded R. rpy2 objects (ro, importr, ...) are local to this function.
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise RuntimeError(
            "rpy2 is not installed. Cannot run statistical analysis. "
            "Install rpy2 and ensure R (with limma and DEqMS) is accessible."
        ) from exc

    # --- Load R packages (fail fast with informative message) ---
    try:
        importr("limma")
        if use_deqms:
            importr("DEqMS")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load R packages: {exc}. "
            "Ensure limma and DEqMS are installed in the R library."
        ) from exc

    n_proteins = len(abund_df)
    n_samples  = len(abund_df.columns)
    run_deqms  = bool(use_deqms and pep_counts is not None)

    # --- Pass data to R global environment ---
    # Flatten in row-major (C) order; R matrix built with byrow=TRUE below.
    mat_flat = abund_df.values.astype(float).flatten(order="C")

    ro.globalenv["prosift_mat_values"]     = ro.FloatVector(mat_flat.tolist())
    ro.globalenv["prosift_protein_ids"]    = ro.StrVector(abund_df.index.tolist())
    ro.globalenv["prosift_sample_ids"]     = ro.StrVector(list(abund_df.columns))
    ro.globalenv["prosift_groups"]         = ro.StrVector(groups)
    ro.globalenv["prosift_unique_groups"]  = ro.StrVector(unique_groups)
    ro.globalenv["prosift_contrast_str"]   = ro.StrVector([r_contrast_str])
    ro.globalenv["prosift_use_deqms"]      = ro.BoolVector([run_deqms])
    ro.globalenv["prosift_robust"]         = ro.BoolVector([bool(robust_ebayes)])
    ro.globalenv["prosift_quarantine_ids"] = ro.StrVector(list(quarantine_ids))

    # Peptide counts (DEqMS path only), aligned to protein order
    if run_deqms:
        pep_aligned = pep_counts.reindex(abund_df.index).fillna(1).astype(int)
        ro.globalenv["prosift_pep_counts"] = ro.IntVector(pep_aligned.tolist())
        method_used = "DEqMS"
    else:
        method_used = "limma"

    # --- Build matrix, base design, and a reusable per-analysis fit function ---
    # trend=TRUE (limma-trend) is the pipeline default as of 2026-07-13; robust is
    # opt-in via robust_ebayes. The quarantined analysis (Q2) appends one 0/1
    # mean-shift indicator column per quarantined sample (module spec Section 4.9).
    ro.r(f"""
        # Reconstruct protein x sample matrix (row-major values, byrow=TRUE)
        prosift_mat <- matrix(
            prosift_mat_values,
            nrow  = {n_proteins},
            ncol  = {n_samples},
            byrow = TRUE
        )
        rownames(prosift_mat) <- prosift_protein_ids
        colnames(prosift_mat) <- prosift_sample_ids

        # Means model: one coefficient per group, no intercept
        group_f     <- factor(prosift_groups, levels = prosift_unique_groups)
        design_full <- model.matrix(~ 0 + group_f)
        colnames(design_full) <- prosift_unique_groups

        # One analysis (Q1 or Q2) given a design matrix. makeContrasts references
        # only the group columns, so nuisance (indicator) columns are weighted 0.
        prosift_run_analysis <- function(dmat) {{
            fit  <- limma::lmFit(prosift_mat, dmat)
            cmat <- limma::makeContrasts(
                        contrasts = prosift_contrast_str[1],
                        levels    = dmat
                    )
            fit2 <- limma::contrasts.fit(fit, cmat)
            fit3 <- limma::eBayes(fit2, trend = TRUE, robust = prosift_robust[1])
            if (prosift_use_deqms[1]) {{
                fit3$count <- as.integer(prosift_pep_counts)
                fit4 <- DEqMS::spectraCounteBayes(fit3)
                res  <- DEqMS::outputResult(fit4, coef_col = 1)
                # Pin BH explicitly. outputResult() has no adjust.method argument:
                # adj.P.Val inherits the topTable default and sca.adj.pval is
                # hardcoded inside DEqMS. Recomputing here makes the method
                # independent of either package's defaults (bit-identical to
                # DEqMS 1.28.0 output; p.adjust excludes NA p-values from n).
                res$adj.P.Val    <- stats::p.adjust(res$P.Value,     method = "BH")
                res$sca.adj.pval <- stats::p.adjust(res$sca.P.Value, method = "BH")
            }} else {{
                # sort.by="none" preserves protein order (rowname alignment)
                res  <- limma::topTable(fit3, number = Inf, sort.by = "none", coef = 1,
                                        adjust.method = "BH")
            }}
            res$protein_id <- rownames(res)
            res
        }}

        # Q1: full (all samples, full weight)
        prosift_results_full <- prosift_run_analysis(design_full)

        # Q2: quarantined (mean-shift indicator), only if any ids given
        if (length(prosift_quarantine_ids) > 0) {{
            ind <- vapply(
                prosift_quarantine_ids,
                function(s) as.numeric(prosift_sample_ids == s),
                numeric(length(prosift_sample_ids))
            )
            ind <- matrix(ind, nrow = length(prosift_sample_ids))
            colnames(ind) <- paste0("q_", make.names(prosift_quarantine_ids))
            design_quar <- cbind(design_full, ind)
            prosift_results_quar <- prosift_run_analysis(design_quar)
        }}
    """)

    # --- Convert R data frame(s) to pandas ---
    def _fetch(name: str) -> pd.DataFrame:
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.globalenv[name])
        return df.reset_index(drop=True)

    results = {"full": _fetch("prosift_results_full")}
    if len(quarantine_ids) > 0:
        results["quarantined"] = _fetch("prosift_results_quar")

    return results, method_used


# ============================================================
# RESULT ASSEMBLY
# ============================================================

def assemble_results(
    raw_df: pd.DataFrame,
    id_mapping_df: pd.DataFrame,
    params: dict,
    method_used: str,
    contrast_user: str,
) -> pd.DataFrame:
    """
    Map R column names to the ProSIFT output schema, add gene_symbol,
    compute significance and direction, and attach the contrast label.

    Output schema (17 columns):
        protein_id, gene_symbol, log2_fc, avg_abundance,
        limma_t, limma_pvalue, limma_adj_pvalue,
        deqms_t, deqms_pvalue, deqms_adj_pvalue,
        n_peptides, significant, pvalue_undetermined, direction, contrast,
        numerator, denominator

    `numerator` and `denominator` make the sign self-describing: log2_fc and
    direction='up' mean higher in `numerator`. They are carried per row so the
    orientation survives any downstream export that drops the contrast label's
    naming convention (for example a run_id whose group order is reversed).
    """
    da_cfg      = params.get("differential_abundance", {})
    sig_cfg     = da_cfg.get("significance", {})
    fdr_thresh  = float(sig_cfg.get("fdr_threshold", 0.05))
    fc_thresh   = float(sig_cfg.get("fc_threshold", 1.0))

    # --- Rename R columns to ProSIFT schema names ---
    rename_map = {
        "logFC":       "log2_fc",
        "AveExpr":     "avg_abundance",
        "t":           "limma_t",
        "P.Value":     "limma_pvalue",
        "adj.P.Val":   "limma_adj_pvalue",
    }
    if method_used == "DEqMS":
        rename_map.update({
            "sca.t":       "deqms_t",
            "sca.P.Value": "deqms_pvalue",
            "sca.adj.pval":"deqms_adj_pvalue",
            "count":       "n_peptides",
        })

    out = raw_df.rename(columns=rename_map).copy()

    # Drop extra R columns (B statistic, etc.) not in the output schema
    keep = {"protein_id", "log2_fc", "avg_abundance",
            "limma_t", "limma_pvalue", "limma_adj_pvalue"}
    if method_used == "DEqMS":
        keep.update({"deqms_t", "deqms_pvalue", "deqms_adj_pvalue", "n_peptides"})
    out = out[[c for c in out.columns if c in keep]].copy()

    # --- Fill DEqMS columns with NA if limma-only ---
    for col in ("deqms_t", "deqms_pvalue", "deqms_adj_pvalue", "n_peptides"):
        if col not in out.columns:
            out[col] = pd.NA

    # --- Enforce schema dtypes ---
    # R returns integer counts as int32; cast to int64 per schema spec.
    if out["n_peptides"].notna().any():
        out["n_peptides"] = out["n_peptides"].astype("Int64")

    # --- Add gene symbol from ID mapping ---
    gene_col = None
    for candidate in ("gene_symbol_mouse", "gene_symbol"):
        if candidate in id_mapping_df.columns:
            gene_col = candidate
            break

    if gene_col is not None:
        sym_map = id_mapping_df.set_index('protein_id')[gene_col]
        out["gene_symbol"] = out["protein_id"].map(sym_map)
    else:
        logging.warning(
            "ID mapping table has no 'gene_symbol_mouse' or 'gene_symbol' column. "
            "gene_symbol will be null for all proteins."
        )
        out["gene_symbol"] = pd.NA

    # --- Determine primary p-value column for significance ---
    primary_adj_pval = (
        "deqms_adj_pvalue" if method_used == "DEqMS" else "limma_adj_pvalue"
    )

    # --- Significance call ---
    adj_pval = out[primary_adj_pval].astype(float)
    fc       = out["log2_fc"].astype(float)

    # A NaN primary adjusted p-value is undefined, not non-significant. 'NaN < t'
    # evaluates to False, so such a protein would silently fold into the 'ns'
    # class. Surface it in a dedicated boolean column (plus a logged count) so it
    # is never mistaken for a genuine non-significant call; `direction` stays in
    # {up, down, ns} and its contract is unchanged.
    pvalue_undetermined = adj_pval.isna()
    n_undetermined = int(pvalue_undetermined.sum())
    if n_undetermined > 0:
        logging.warning(
            "%d protein(s) have a NaN %s; flagged pvalue_undetermined=True and "
            "excluded from the significant set (direction stays 'ns').",
            n_undetermined, primary_adj_pval,
        )
    out["pvalue_undetermined"] = pvalue_undetermined

    passes_fdr = adj_pval < fdr_thresh
    passes_fc  = (fc.abs() > fc_thresh) if fc_thresh > 0 else pd.Series(True, index=out.index)
    out["significant"] = passes_fdr & passes_fc

    # --- Direction ---
    def _direction(row: pd.Series) -> str:
        if not row["significant"]:
            return "ns"
        return "up" if row["log2_fc"] > 0 else "down"

    out["direction"] = out.apply(_direction, axis=1)

    # --- Contrast label + explicit orientation ---
    out["contrast"] = contrast_user
    out["numerator"], out["denominator"] = split_contrast(contrast_user)

    # --- Canonical column order ---
    col_order = [
        "protein_id", "gene_symbol",
        "log2_fc", "avg_abundance",
        "limma_t", "limma_pvalue", "limma_adj_pvalue",
        "deqms_t", "deqms_pvalue", "deqms_adj_pvalue",
        "n_peptides",
        "significant", "pvalue_undetermined", "direction", "contrast",
        "numerator", "denominator",
    ]
    out = out[col_order]

    return out


# ============================================================
# SUMMARY TEXT
# ============================================================

def _summary_row(label: str, value: str, width: int = 26) -> str:
    """
    Left-align `label` to the summary's value column (26 chars, matching the
    fixed-width rows around it), always leaving at least one space so a long
    group name never runs into its value.
    """
    return f"{label:<{width - 1}} {value}"


def write_summary_txt(
    contrast_results: list[tuple[str, str, str, pd.DataFrame]],
    run_id: str,
    method_used: str,
    params: dict,
    outdir: Path,
    sensitivity_results: "list[tuple[str, str, str, pd.DataFrame]] | None" = None,
    primary_key: str = "full",
    quarantine_samples: "list[str] | None" = None,
) -> None:
    """
    Write a plain-text summary following the template in Section 4.6 of the
    Module 04 spec.

    contrast_results holds the PRIMARY analysis (contrast_user, numerator,
    denominator, df) per contrast, in order. When quarantine is active,
    sensitivity_results holds the other analysis and a compact SENSITIVITY block
    is appended per contrast (module spec Section 4.9).
    """
    now     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    da_cfg  = params.get("differential_abundance", {})
    sig_cfg = da_cfg.get("significance", {})
    fdr_threshold = sig_cfg.get("fdr_threshold", 0.05)
    fc_threshold  = sig_cfg.get("fc_threshold", 1.0)

    primary_pval_col = "deqms_adj_pvalue" if method_used == "DEqMS" else "limma_adj_pvalue"
    primary_pval_label = "deqms_adj_pvalue" if method_used == "DEqMS" else "limma_adj_pvalue"

    lines = [
        "========================================",
        "DIFFERENTIAL ABUNDANCE SUMMARY",
        "========================================",
        "",
        f"Run:              {run_id}",
        f"Date:             {now}",
        f"Method:           {method_used}",
        "",
    ]

    # Analysis-mode header (mean-shift quarantine / dual reporting, Section 4.9)
    if quarantine_samples:
        other_key = "full" if primary_key == "quarantined" else "quarantined"
        lines += [
            f"Analysis:         PRIMARY = {primary_key} (see below); "
            f"SENSITIVITY = {other_key}",
            f"Quarantined:      {', '.join(quarantine_samples)} "
            f"(mean-shift indicator; see {run_id}.analysis_provenance.txt)",
            "",
        ]

    for contrast_user, numerator, denominator, df in contrast_results:
        n_proteins   = len(df)
        n_numerator  = int((df["direction"] != "ns").any())  # placeholder; compute below
        group_col    = params["design"]["group_column"]

        # Count samples per group
        n_sig  = int(df["significant"].sum())
        n_up   = int((df["direction"] == "up").sum())
        n_down = int((df["direction"] == "down").sum())
        n_ns   = n_proteins - n_sig
        n_undet = (
            int(df["pvalue_undetermined"].sum())
            if "pvalue_undetermined" in df.columns else 0
        )

        pct_sig  = f"{100 * n_sig / n_proteins:.1f}" if n_proteins > 0 else "0.0"
        pct_up   = f"{100 * n_up / n_proteins:.1f}"  if n_proteins > 0 else "0.0"
        pct_down = f"{100 * n_down / n_proteins:.1f}" if n_proteins > 0 else "0.0"

        lines += [
            "----------------------------------------",
            f"CONTRAST: {contrast_user}  ({numerator} - {denominator})",
            f"Direction:  {direction_statement(numerator, denominator)}; "
            f"'up' = higher in {numerator}, 'down' = higher in {denominator}",
            "----------------------------------------",
            "",
            "INPUT",
            "----------------------------------------",
            f"Proteins tested:          {n_proteins}",
            "",
        ]

        # Peptide count section (DEqMS only)
        if method_used == "DEqMS" and "n_peptides" in df.columns:
            pep = df["n_peptides"].dropna().astype(float)
            lines += [
                "PEPTIDE COUNTS (DEqMS)",
                "----------------------------------------",
                f"Summary method:           min of nonzero values per protein",
                f"Median count:             {pep.median():.1f}",
                f"Range:                    {int(pep.min())} - {int(pep.max())}",
                "",
            ]
        else:
            if method_used == "limma":
                lines += [
                    "NOTE: Method = limma (no peptide count correction applied).",
                    "",
                ]

        lines += [
            "SIGNIFICANCE THRESHOLDS",
            "----------------------------------------",
            f"FDR threshold:            {fdr_threshold} (BH-corrected)",
            f"log2 FC threshold:        {fc_threshold} (absolute)",
            f"Primary p-value:          {primary_pval_label}",
            "",
            "RESULTS",
            "----------------------------------------",
            f"Significant proteins:     {n_sig} / {n_proteins} ({pct_sig}%)",
            _summary_row(f"  Up (higher in {numerator}):", f"{n_up} ({pct_up}%)"),
            _summary_row(f"  Down (higher in {denominator}):", f"{n_down} ({pct_down}%)"),
            f"Not significant:          {n_ns}",
            f"  p-value undetermined:   {n_undet} (NaN adjusted p-value; not significant)",
            "",
        ]

        # Top 10 by adjusted p-value
        top10 = (
            df.sort_values(primary_pval_col, ascending=True)
              .head(10)[["protein_id", "gene_symbol", "log2_fc", primary_pval_col]]
        )
        lines += [
            "TOP 10 BY SIGNIFICANCE",
            "----------------------------------------",
            f"  {'protein_id':<24} {'gene_symbol':<16} {'log2_fc':>8}  {'adj_pvalue':>12}",
        ]
        for _, row in top10.iterrows():
            gene = str(row["gene_symbol"]) if pd.notna(row["gene_symbol"]) else "NA"
            pval = row[primary_pval_col]
            pval_str = f"{pval:.3e}" if pd.notna(pval) else "NA"
            lines.append(
                f"  {str(row['protein_id']):<24} {gene:<16} "
                f"{row['log2_fc']:>8.3f}  {pval_str:>12}"
            )

        lines.append("")

    # --- Compact SENSITIVITY section (dual reporting, Section 4.9) ---
    if sensitivity_results:
        other_key = "full" if primary_key == "quarantined" else "quarantined"
        lines += [
            "----------------------------------------",
            f"SENSITIVITY ANALYSIS ({other_key})",
            "----------------------------------------",
            "",
        ]
        for contrast_user, numerator, denominator, df in sensitivity_results:
            n_proteins = len(df)
            n_sig  = int(df["significant"].sum())
            n_up   = int((df["direction"] == "up").sum())
            n_down = int((df["direction"] == "down").sum())
            n_undet = (
                int(df["pvalue_undetermined"].sum())
                if "pvalue_undetermined" in df.columns else 0
            )
            pct    = f"{100 * n_sig / n_proteins:.1f}" if n_proteins > 0 else "0.0"
            undet_note = f"  ({n_undet} p-value undetermined)" if n_undet else ""
            lines += [
                f"CONTRAST: {contrast_user}  ({numerator} - {denominator})",
                f"  Significant proteins:   {n_sig} / {n_proteins} ({pct}%)  "
                f"[{n_up} up in {numerator}, {n_down} up in {denominator}]{undet_note}",
                "",
            ]

    lines.append("========================================")

    path = outdir / f"{run_id}.diff_abundance_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"  Saved: {path.name}")


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def plot_volcano(
    df: pd.DataFrame,
    contrast_user: str,
    run_id: str,
    params: dict,
) -> go.Figure:
    """
    Volcano plot: x = log2_fc, y = -log10(primary adj p-value).
    Points colored by direction (up/down/ns). Threshold lines drawn for
    both FDR and FC cutoffs. Hover text shows protein_id, gene_symbol,
    log2_fc, and adj_pvalue.
    """
    da_cfg      = params.get("differential_abundance", {})
    sig_cfg     = da_cfg.get("significance", {})
    fdr_thresh  = float(sig_cfg.get("fdr_threshold", 0.05))
    fc_thresh   = float(sig_cfg.get("fc_threshold", 1.0))

    # Primary adjusted p-value column
    primary_col = (
        "deqms_adj_pvalue"
        if "deqms_adj_pvalue" in df.columns and df["deqms_adj_pvalue"].notna().any()
        else "limma_adj_pvalue"
    )

    neg_log_p = -np.log10(df[primary_col].clip(lower=1e-300).astype(float))
    log2_fc   = df["log2_fc"].astype(float)

    color_map = {"up": _COLOR_UP, "down": _COLOR_DN, "ns": _COLOR_NS}
    traces: dict[str, dict] = {"up": {"x": [], "y": [], "text": []},
                                "down": {"x": [], "y": [], "text": []},
                                "ns":  {"x": [], "y": [], "text": []}}

    for _, row in df.iterrows():
        direction = str(row["direction"])
        gene      = str(row["gene_symbol"]) if pd.notna(row["gene_symbol"]) else "NA"
        pval      = row[primary_col]
        pval_str  = f"{pval:.3e}" if pd.notna(pval) else "NA"
        hover     = (
            f"{row['protein_id']}<br>"
            f"Gene: {gene}<br>"
            f"log2FC: {row['log2_fc']:.2f}<br>"
            f"adj.p: {pval_str}"
        )
        traces[direction]["x"].append(float(row["log2_fc"]))
        traces[direction]["y"].append(float(neg_log_p[row.name]))
        traces[direction]["text"].append(hover)

    numerator, denominator = split_contrast(contrast_user)
    label_map = {"up": f"Higher in {numerator}", "down": f"Higher in {denominator}",
                 "ns": "Not significant"}
    fig = go.Figure()
    for direction in ("ns", "down", "up"):   # ns drawn first (background)
        d = traces[direction]
        if not d["x"]:
            continue
        opacity = 0.4 if direction == "ns" else 0.8
        fig.add_trace(
            go.Scatter(
                x=d["x"],
                y=d["y"],
                mode="markers",
                name=label_map[direction],
                marker=dict(color=color_map[direction], size=5, opacity=opacity),
                text=d["text"],
                hoverinfo="text",
            )
        )

    # Threshold lines
    y_fdr = -np.log10(fdr_thresh)
    fig.add_hline(y=y_fdr, line_dash="dash", line_color="#888888", line_width=1)
    if fc_thresh > 0:
        fig.add_vline(x=fc_thresh,  line_dash="dash", line_color="#888888", line_width=1)
        fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="#888888", line_width=1)

    fig.update_layout(
        # Short axis title (fits at 700 px with long group names); the full
        # orientation statement lives in the subtitle line.
        title=(f"Volcano Plot: {contrast_user}  ({run_id})<br>"
               f"<sup>{direction_statement(numerator, denominator)}</sup>"),
        xaxis_title=f"log2 FC (> 0 = higher in {numerator})",
        yaxis_title=f"-log10({primary_col})",
        height=560,
        width=700,
        legend_title_text="Direction",
    )
    return fig


def plot_ma(
    df: pd.DataFrame,
    contrast_user: str,
    run_id: str,
    params: dict,
) -> go.Figure:
    """
    MA plot: x = avg_abundance (AveExpr), y = log2_fc.
    Same coloring scheme and threshold lines as the volcano plot.
    Hover text identical to volcano plot.
    """
    da_cfg      = params.get("differential_abundance", {})
    sig_cfg     = da_cfg.get("significance", {})
    fc_thresh   = float(sig_cfg.get("fc_threshold", 1.0))

    primary_col = (
        "deqms_adj_pvalue"
        if "deqms_adj_pvalue" in df.columns and df["deqms_adj_pvalue"].notna().any()
        else "limma_adj_pvalue"
    )

    color_map  = {"up": _COLOR_UP, "down": _COLOR_DN, "ns": _COLOR_NS}
    traces: dict[str, dict] = {"up": {"x": [], "y": [], "text": []},
                                "down": {"x": [], "y": [], "text": []},
                                "ns":  {"x": [], "y": [], "text": []}}

    for _, row in df.iterrows():
        direction = str(row["direction"])
        gene      = str(row["gene_symbol"]) if pd.notna(row["gene_symbol"]) else "NA"
        pval      = row[primary_col]
        pval_str  = f"{pval:.3e}" if pd.notna(pval) else "NA"
        hover     = (
            f"{row['protein_id']}<br>"
            f"Gene: {gene}<br>"
            f"log2FC: {row['log2_fc']:.2f}<br>"
            f"adj.p: {pval_str}"
        )
        traces[direction]["x"].append(float(row["avg_abundance"]))
        traces[direction]["y"].append(float(row["log2_fc"]))
        traces[direction]["text"].append(hover)

    numerator, denominator = split_contrast(contrast_user)
    label_map = {"up": f"Higher in {numerator}", "down": f"Higher in {denominator}",
                 "ns": "Not significant"}
    fig = go.Figure()
    for direction in ("ns", "down", "up"):
        d = traces[direction]
        if not d["x"]:
            continue
        opacity = 0.4 if direction == "ns" else 0.8
        fig.add_trace(
            go.Scatter(
                x=d["x"],
                y=d["y"],
                mode="markers",
                name=label_map[direction],
                marker=dict(color=color_map[direction], size=5, opacity=opacity),
                text=d["text"],
                hoverinfo="text",
            )
        )

    if fc_thresh > 0:
        fig.add_hline(y=fc_thresh,  line_dash="dash", line_color="#888888", line_width=1)
        fig.add_hline(y=-fc_thresh, line_dash="dash", line_color="#888888", line_width=1)

    fig.update_layout(
        title=(f"MA Plot: {contrast_user}  ({run_id})<br>"
               f"<sup>{direction_statement(numerator, denominator)}</sup>"),
        xaxis_title="Average log2 Abundance (AveExpr)",
        yaxis_title=f"log2 FC (> 0 = higher in {numerator})",
        height=520,
        width=700,
        legend_title_text="Direction",
    )
    return fig


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    setup_logging()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id

    logging.info(f"Module 04 DIFFERENTIAL_ABUNDANCE -- {run_id}")

    # --- Load params ---
    params = load_params(Path(args.params))

    # --- Load inputs ---
    logging.info("Loading input files...")
    matrix_df  = pd.read_parquet(args.matrix)
    metadata_df = pd.read_parquet(args.metadata)
    mapping_df  = pd.read_parquet(args.id_mapping)

    # --- Split abundance and peptide count columns ---
    abund_df, pep_df, sample_ids = extract_abundance_and_peptide_cols(matrix_df, params)
    n_proteins = len(abund_df)
    n_samples  = len(sample_ids)
    logging.info(f"  {n_proteins} proteins, {n_samples} samples")

    # --- Build group map ---
    group_map = build_group_map(metadata_df, params)
    missing_samples = [s for s in sample_ids if s not in group_map]
    if missing_samples:
        raise ValueError(
            f"Samples in matrix not found in metadata: {missing_samples}"
        )

    groups        = [group_map[s] for s in sample_ids]
    unique_groups = sorted(set(groups))
    logging.info(f"  Groups: {unique_groups}")

    # --- Validate contrasts ---
    # Validate against `unique_groups` (groups with samples in this run), not the
    # full metadata column: `unique_groups` is what is handed to R as the factor
    # levels / design-matrix column names, so it is exactly the set R validates.
    contrasts = parse_and_validate_contrasts(params, unique_groups)
    logging.info(f"  Contrasts: {[c[0] for c in contrasts]}")

    # --- Determine method and peptide count availability ---
    da_cfg      = params.get("differential_abundance", {})
    method      = da_cfg.get("method", "deqms").lower()
    has_pep     = not pep_df.empty

    if method == "deqms" and not has_pep:
        logging.warning(
            "DEqMS requested but no peptide count columns found. "
            "Falling back to limma."
        )
        use_deqms   = False
        method_used = "limma"
    elif method == "deqms":
        use_deqms   = True
        method_used = "DEqMS"
    else:
        use_deqms   = False
        method_used = "limma"

    logging.info(f"  Statistical method: {method_used}")

    # --- Sample quarantine / dual-reporting parameters (spec Section 4.9) ---
    quarantine_samples = normalize_quarantine_samples(da_cfg.get("quarantine_samples"))
    primary_analysis   = str(da_cfg.get("primary_analysis", "full")).lower()
    robust_ebayes      = parse_bool_param(da_cfg.get("robust_ebayes", False))
    min_per_group      = int(params.get("qc", {}).get("min_samples_per_group", 2))

    validate_quarantine(
        quarantine_samples, primary_analysis, sample_ids,
        group_map, contrasts, min_per_group,
    )

    has_quar    = len(quarantine_samples) > 0
    primary_key = primary_analysis if has_quar else "full"
    if has_quar:
        logging.info(
            f"  Quarantine (mean-shift): {quarantine_samples} | "
            f"primary analysis: {primary_key}"
        )
    if robust_ebayes:
        logging.info("  eBayes robust=TRUE enabled")

    # --- Summarize peptide counts (DEqMS path only) ---
    pep_counts: "pd.Series | None" = None
    if use_deqms:
        logging.info("Summarizing peptide counts (min of nonzero per protein)...")
        pep_counts = summarize_peptide_counts(pep_df)
        logging.info(
            f"  Peptide count range: {int(pep_counts.min())} - {int(pep_counts.max())}, "
            f"median: {pep_counts.median():.1f}"
        )

    # --- Run statistical analysis per contrast (Q1 full + Q2 quarantined) ---
    primary_results: list[tuple[str, str, str, pd.DataFrame]] = []
    sensitivity_results: list[tuple[str, str, str, pd.DataFrame]] = []
    sensitivity_key = "full" if primary_key == "quarantined" else "quarantined"

    for contrast_user, numerator, denominator, r_contrast_str in contrasts:
        logging.info(f"Running contrast: {contrast_user}  ({r_contrast_str})...")

        raw_by_analysis, actual_method = _run_one_contrast_r(
            abund_df, groups, unique_groups,
            pep_counts, r_contrast_str, use_deqms,
            quarantine_samples, robust_ebayes,
        )

        assembled = {
            name: assemble_results(raw, mapping_df, params, actual_method, contrast_user)
            for name, raw in raw_by_analysis.items()
        }

        primary_df = assembled[primary_key]
        primary_results.append((contrast_user, numerator, denominator, primary_df))

        n_sig  = int(primary_df["significant"].sum())
        n_up   = int((primary_df["direction"] == "up").sum())
        n_down = int((primary_df["direction"] == "down").sum())
        logging.info(
            f"  [{primary_key}] {n_sig}/{n_proteins} significant "
            f"({n_up} up, {n_down} down)"
        )

        # --- Diagnostic plots (primary; existing filenames) ---
        logging.info(f"  Generating plots for {contrast_user}...")
        save_plot(plot_volcano(primary_df, contrast_user, run_id, params),
                  outdir / f"{run_id}.{contrast_user}.volcano_plot")
        save_plot(plot_ma(primary_df, contrast_user, run_id, params),
                  outdir / f"{run_id}.{contrast_user}.ma_plot")

        if has_quar:
            sens_df = assembled[sensitivity_key]
            sensitivity_results.append((contrast_user, numerator, denominator, sens_df))
            # Sensitivity plots, `.sensitivity` suffix so they never match the
            # primary globs (module spec Section 2.2 filename constraint).
            save_plot(plot_volcano(sens_df, contrast_user, run_id, params),
                      outdir / f"{run_id}.{contrast_user}.volcano_plot.sensitivity")
            save_plot(plot_ma(sens_df, contrast_user, run_id, params),
                      outdir / f"{run_id}.{contrast_user}.ma_plot.sensitivity")

    # --- Write PRIMARY results table (existing name/schema; consumed downstream) ---
    logging.info("Writing outputs...")
    combined_primary = pd.concat(
        [df for _, _, _, df in primary_results], ignore_index=True
    )
    pq_path  = outdir / f"{run_id}.diff_abundance_results.parquet"
    csv_path = outdir / f"{run_id}.diff_abundance_results.csv"
    combined_primary.to_parquet(pq_path, index=False)
    combined_primary.to_csv(csv_path, index=False)
    logging.info(f"  Saved: {pq_path.name}")
    logging.info(f"  Saved: {csv_path.name}")

    # --- Write SENSITIVITY results table (conditional; NOT consumed downstream) ---
    if has_quar:
        combined_sens = pd.concat(
            [df for _, _, _, df in sensitivity_results], ignore_index=True
        )
        s_pq  = outdir / f"{run_id}.diff_abundance_results.sensitivity.parquet"
        s_csv = outdir / f"{run_id}.diff_abundance_results.sensitivity.csv"
        combined_sens.to_parquet(s_pq, index=False)
        combined_sens.to_csv(s_csv, index=False)
        logging.info(f"  Saved: {s_pq.name}")
        logging.info(f"  Saved: {s_csv.name}")

    # --- Write provenance (always) ---
    write_provenance(
        run_id, outdir, quarantine_samples, primary_key,
        sample_ids, groups, method_used, robust_ebayes,
    )

    # --- Write text summary (dual-aware) ---
    write_summary_txt(
        primary_results, run_id, method_used, params, outdir,
        sensitivity_results=(sensitivity_results if has_quar else None),
        primary_key=primary_key,
        quarantine_samples=(quarantine_samples if has_quar else None),
    )

    # --- Final log ---
    total_sig = int(combined_primary["significant"].sum())
    logging.info(
        f"Module 04 DIFFERENTIAL_ABUNDANCE complete. "
        f"{len(contrasts)} contrast(s), "
        f"{total_sig}/{n_proteins} proteins significant "
        f"(primary: {primary_key}, method: {method_used})."
    )


if __name__ == "__main__":
    main()
