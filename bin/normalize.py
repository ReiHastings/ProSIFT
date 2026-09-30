#!/usr/bin/env python3
"""
Title:         normalize.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-03-30
Last Modified: 2026-04-06 (added: post-norm density plot; correlation heatmap -> clustermap via shared utils)
Purpose:       Module 03 NORMALIZE process. Applies method-aware log2 transformation
               and normalization (median, quantile, VSN, or none) to the post-filter
               abundance matrix. Produces post-normalization diagnostic plots, a CV
               summary table (back-transformed to linear scale), and a normalization
               summary. Missing values remain NaN; peptide count columns pass through
               unchanged.
Inputs:
  --matrix    {run_id}.filtered_matrix.parquet  (Module 01 FILTER_PROTEINS)
  --metadata  {run_id}.validated_metadata.parquet  (Module 01 VALIDATE_INPUTS)
  --params    {run_id}_params.yml
Outputs:
  {run_id}.normalized_matrix.parquet
  {run_id}.normalization_summary.txt
  {run_id}.cv_summary.parquet/.csv
  {run_id}.postnorm_pca_results.parquet
  {run_id}.postnorm_correlation_matrix.parquet
  {run_id}.postnorm_boxplots.png/.html
  {run_id}.postnorm_density.png/.html
  {run_id}.postnorm_pca_scatter.png/.html
  {run_id}.postnorm_clustermap.png/.html
Usage:
  normalize.py --matrix CTXcyto_WT_vs_CTXcyto_KO.filtered_matrix.parquet \
               --metadata CTXcyto_WT_vs_CTXcyto_KO.validated_metadata.parquet \
               --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \
               --run-id CTXcyto_WT_vs_CTXcyto_KO \
               --outdir .
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prosift_plot_utils import (
    compute_and_plot_correlation,
    compute_and_plot_pca,
    make_color_map,
    plot_density,
    plot_intensity_boxplots,
    save_plot,
)

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="normalize.py",
        description="ProSIFT Module 03 NORMALIZE: log2 transformation, normalization, CV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  normalize.py \\\n"
            "    --matrix  run.filtered_matrix.parquet \\\n"
            "    --metadata run.validated_metadata.parquet \\\n"
            "    --params  run_params.yml \\\n"
            "    --run-id  run \\\n"
            "    --outdir  ."
        ),
    )
    parser.add_argument("--matrix",   required=True, help="Filtered abundance matrix (Parquet)")
    parser.add_argument("--metadata", required=True, help="Validated metadata (Parquet)")
    parser.add_argument("--params",   required=True, help="Run params.yml")
    parser.add_argument("--run-id",   required=True, dest="run_id",
                        help="Run identifier (used as output file prefix)")
    parser.add_argument("--outdir",   required=True, help="Output directory")
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


def prepare_abundance(
    matrix_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Extract raw abundance values and peptide count columns from the filtered matrix.

    Returns:
        raw_df      -- raw abundance values; index=protein_id, columns=bare sample_ids
        peptide_df  -- peptide count columns from the original matrix (protein_id index)
        sample_ids  -- ordered list of bare sample IDs
    """
    abundance_prefix = params["input"].get("abundance_prefix", "")
    peptide_prefix   = params["input"].get("peptide_count_prefix", "peptide_count_")

    all_cols = matrix_df.columns.tolist()
    peptide_cols = [c for c in all_cols if c.startswith(peptide_prefix)]
    abund_cols   = [c for c in all_cols if c != "protein_id"
                    and not c.startswith(peptide_prefix)]

    if not abund_cols:
        raise ValueError("No abundance columns found in filtered matrix.")

    # Parse bare sample IDs by stripping the abundance prefix
    sample_ids = (
        [c[len(abundance_prefix):] for c in abund_cols]
        if abundance_prefix
        else list(abund_cols)
    )

    raw_df = matrix_df.set_index("protein_id")[abund_cols].copy()
    raw_df.columns = sample_ids

    peptide_df = (
        matrix_df.set_index("protein_id")[peptide_cols].copy()
        if peptide_cols
        else pd.DataFrame(index=matrix_df.set_index("protein_id").index)
    )

    return raw_df, peptide_df, sample_ids


def build_group_map(metadata_df: pd.DataFrame, params: dict) -> dict[str, str]:
    """Return {sample_id: group_label} from validated metadata."""
    group_column = params["design"]["group_column"]
    if group_column not in metadata_df.columns:
        raise ValueError(f"Group column '{group_column}' not found in metadata.")
    return dict(zip(metadata_df["sample_id"].astype(str),
                    metadata_df[group_column].astype(str), strict=True))


# ============================================================
# LOG2 TRANSFORMATION
# ============================================================

def apply_log2(
    raw_df: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """
    Apply log2(x) transformation. Non-positive values are converted to NaN with
    a warning. Non-positive raw values are removed upstream at validation
    (Module 01); reaching this branch means validation was bypassed, so we
    convert defensively rather than emit -Inf (for 0) or NaN-with-RuntimeWarning
    (for negatives).
    """
    nonpos = raw_df <= 0
    n_nonpos = int(nonpos.to_numpy().sum())
    if n_nonpos > 0:
        msg = (
            f"Found {n_nonpos} non-positive value(s) in the abundance matrix "
            "(expected none after Module 01 validation). "
            "Converted to NaN before log2 transformation."
        )
        logging.warning(msg)
        warnings.append(msg)
        raw_df = raw_df.mask(nonpos)

    return np.log2(raw_df)


# ============================================================
# NORMALIZATION METHODS
# ============================================================

def normalize_median(log2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift each sample's log2 distribution so all samples share the same median.
    Shift = global_median - sample_median. NaN values remain NaN.
    """
    sample_medians = log2_df.median(skipna=True)
    global_median  = float(sample_medians.median())
    shifts = global_median - sample_medians
    return log2_df.add(shifts, axis="columns")


def normalize_quantile(log2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Force all sample distributions to be identical.

    Algorithm:
      1. Rank proteins within each sample (ties averaged; NaN -> NaN).
      2. Sort each sample's values; average across samples at each rank position
         (skipna=True handles differing NaN counts across samples).
      3. Replace each observed value with the reference value at its rank.
    NaN values remain NaN.
    """
    # Step 1: ranks (1-based floats; NaN stays NaN)
    ranks = log2_df.rank(axis=0, method="average", na_option="keep")

    # Step 2: reference distribution -- sort each column, average row-wise
    sorted_cols = pd.DataFrame(
        {col: np.sort(log2_df[col].values) for col in log2_df.columns},
        index=log2_df.index,
    )
    reference = sorted_cols.mean(axis=1, skipna=True).values  # shape: (n_proteins,)
    n_ref = len(reference)

    # Step 3: replace observed values with reference at their rank
    result = log2_df.copy()
    for col in log2_df.columns:
        mask = log2_df[col].notna()
        if mask.sum() == 0:
            continue
        # Convert 1-based rank to 0-based index; clip to valid range
        rank_idx = (ranks.loc[mask, col].round().astype(int) - 1).clip(0, n_ref - 1)
        result.loc[mask, col] = reference[rank_idx.values]

    return result


def normalize_vsn(
    raw_df: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """
    Variance-stabilizing normalization via R's Bioconductor vsn package.

    Uses vsn2() to fit the model (handles NA natively by excluding missing
    values from parameter estimation) and predict() to apply the fitted
    transformation to the full matrix. Output is on generalized log2 scale
    (arcsinh-based, approximately log2). NaN values remain NaN.

    Requires: rpy2, R, and bioconductor-vsn installed in the active environment.
    VSN is installed via BiocManager::install("vsn") -- not available via bioconda.
    """
    try:
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as numpy2ri
        from rpy2.robjects.packages import importr
    except ImportError as e:
        raise RuntimeError(
            "rpy2 is not available. VSN requires rpy2, R, and the Bioconductor vsn "
            "package. Install via: BiocManager::install('vsn') within R."
        ) from e

    numpy2ri.activate()

    try:
        vsn_pkg = importr("vsn")
    except Exception as e:
        raise RuntimeError(
            "Bioconductor vsn package not found. Install via: "
            "BiocManager::install('vsn') within R in the prosift conda environment."
        ) from e

    n_proteins, n_samples = raw_df.shape
    values = raw_df.values.astype(float)  # shape: (n_proteins, n_samples)

    # Build R matrix (column-major fill to match R's storage convention)
    r_matrix = ro.r["matrix"](
        values.flatten(order="F"),
        nrow=n_proteins,
        ncol=n_samples,
    )

    # Fit VSN model -- vsn2() excludes NA values from parameter estimation
    logging.info("  VSN: fitting model via vsn2()...")
    vsn_fit = vsn_pkg.vsn2(r_matrix)

    # Apply fitted transformation to the full matrix
    r_result = ro.r["predict"](vsn_fit, newdata=r_matrix)
    result_values = np.array(r_result).reshape(n_proteins, n_samples, order="F")

    norm_df = pd.DataFrame(result_values, index=raw_df.index, columns=raw_df.columns)

    # Restore NaN positions (vsn2 may fill them; we want NaN preserved)
    norm_df[raw_df.isna()] = np.nan

    return norm_df


# ============================================================
# NORMALIZATION DISPATCHER
# ============================================================

def run_normalization(
    raw_df: pd.DataFrame,
    params: dict,
    warnings: list[str],
) -> tuple[pd.DataFrame, str, bool]:
    """
    Apply method-aware log2 transformation and normalization.

    Returns:
        norm_df         -- normalized (approximately log2) DataFrame
        method_used     -- string description for the summary
        log2_applied    -- whether log2 transformation was applied
    """
    method         = params["normalization"]["method"].lower()
    abundance_type = params["input"]["abundance_type"].lower()

    valid_methods = {"median", "quantile", "vsn", "none"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown normalization.method: '{method}'. "
            f"Valid options: {sorted(valid_methods)}"
        )

    # --- VSN path ---
    if method == "vsn":
        if abundance_type != "raw":
            raise ValueError(
                f"VSN requires raw intensity data. Your data is declared as "
                f"'{abundance_type}'. Use normalization.method: median or quantile instead."
            )
        logging.info("  Method: VSN (via R Bioconductor vsn package)")
        norm_df = normalize_vsn(raw_df, warnings)
        return norm_df, "vsn", False  # VSN handles its own transformation

    # --- Median / Quantile / None paths ---
    if abundance_type == "raw":
        logging.info("  Applying log2(x) transformation...")
        log2_df = apply_log2(raw_df, warnings)
        log2_applied = True
    elif abundance_type == "log2":
        log2_df = raw_df.copy()
        log2_applied = False
    elif abundance_type == "normalized":
        msg = (
            "Data is declared as 'normalized' (abundance_type: normalized). "
            "Skipping normalization and passing through unchanged."
        )
        logging.warning(msg)
        warnings.append(msg)
        return raw_df.copy(), "none (pass-through: data already normalized)", False
    else:
        raise ValueError(
            f"Unknown abundance_type: '{abundance_type}'. "
            "Valid options: raw, log2, normalized."
        )

    if method == "median":
        logging.info("  Applying median normalization...")
        norm_df = normalize_median(log2_df)
        return norm_df, "median", log2_applied

    if method == "quantile":
        logging.info("  Applying quantile normalization...")
        norm_df = normalize_quantile(log2_df)
        return norm_df, "quantile", log2_applied

    if method == "none":
        logging.info("  Method: none (log2 transformation only, no normalization).")
        return log2_df, "none", log2_applied

    # Should be unreachable
    raise ValueError(f"Unhandled method: {method}")


# ============================================================
# CV COMPUTATION
# ============================================================

def compute_cv_summary(
    norm_df: pd.DataFrame,
    group_map: dict[str, str],
    sample_ids: list[str],
) -> pd.DataFrame:
    """
    Per-protein, per-group CV on back-transformed linear scale (2^norm_value).
    CV = SD / mean using observed (non-NaN) values only.
    Proteins with fewer than 2 observed values in a group get NaN for that group.

    Output columns: protein_id, cv_{group1}, cv_{group2}, n_observed_{group1},
    n_observed_{group2}. Group names are taken from the metadata.
    """
    groups = sorted(set(group_map.values()))
    group_samples: dict[str, list[str]] = {
        grp: [s for s in sample_ids if group_map.get(s) == grp]
        for grp in groups
    }

    rows = []
    for protein_id, prot_row in norm_df.iterrows():
        row: dict = {"protein_id": protein_id}
        for grp in groups:
            grp_s = group_samples[grp]
            vals = prot_row[grp_s].dropna().values
            n_obs = len(vals)
            if n_obs >= 2:
                linear_vals = 2.0 ** vals
                cv = float(np.std(linear_vals, ddof=1) / np.mean(linear_vals))
            else:
                cv = np.nan
            row[f"cv_{grp}"]         = cv
            row[f"n_observed_{grp}"] = n_obs
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# OUTPUT WRITERS
# ============================================================

def write_normalized_matrix(
    norm_df: pd.DataFrame,
    peptide_df: pd.DataFrame,
    abundance_prefix: str,
    outdir: Path,
    run_id: str,
) -> Path:
    """
    Reconstruct the original matrix column structure with normalized abundance
    values. Abundance prefix is restored; peptide count columns are appended
    unchanged. Writes Parquet.
    """
    # Restore prefix on abundance columns
    out = norm_df.copy()
    out.columns = [f"{abundance_prefix}{c}" for c in out.columns]

    # Concatenate peptide count columns
    if not peptide_df.empty:
        out = pd.concat([out, peptide_df], axis=1)

    # Restore protein_id as a column (not index)
    out = out.reset_index()

    path = outdir / f"{run_id}.normalized_matrix.parquet"
    out.to_parquet(path, index=False)
    logging.info(f"  Saved: {path.name}")
    return path


def write_normalization_summary(
    run_id: str,
    method_used: str,
    log2_applied: bool,
    n_proteins: int,
    n_samples: int,
    warnings: list[str],
    outdir: Path,
) -> Path:
    """Write a plain-text normalization summary."""
    path = outdir / f"{run_id}.normalization_summary.txt"
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"ProSIFT Normalization Summary -- {run_id}",
        f"Generated: {now}",
        "",
        f"Method:               {method_used}",
        f"Log2 transformation:  {'applied (log2(x))' if log2_applied else 'not applied'}",
        f"Proteins:             {n_proteins}",
        f"Samples:              {n_samples}",
        "",
    ]

    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
    else:
        lines.append("Warnings: none")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"  Saved: {path.name}")
    return path


def write_cv_summary(
    cv_df: pd.DataFrame,
    outdir: Path,
    run_id: str,
) -> None:
    cv_df.to_parquet(outdir / f"{run_id}.cv_summary.parquet", index=False)
    cv_df.to_csv(outdir / f"{run_id}.cv_summary.csv", index=False)
    logging.info(f"  Saved: {run_id}.cv_summary.parquet/.csv")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    setup_logging()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id

    logging.info(f"Module 03 NORMALIZE -- {run_id}")

    # --- Load params ---
    params = load_params(Path(args.params))
    method         = params["normalization"]["method"]
    abundance_type = params["input"]["abundance_type"]
    abundance_prefix = params["input"].get("abundance_prefix", "")
    logging.info(f"  normalization.method={method}, abundance_type={abundance_type}")

    # --- Load input files ---
    logging.info("Loading input files...")
    matrix_df   = pd.read_parquet(args.matrix)
    metadata_df = pd.read_parquet(args.metadata)

    # --- Prepare abundance data ---
    raw_df, peptide_df, sample_ids = prepare_abundance(matrix_df, params)
    n_proteins = len(raw_df)
    n_samples  = len(sample_ids)
    logging.info(f"  {n_proteins} proteins, {n_samples} samples")

    # --- Build group and color maps ---
    group_map = build_group_map(metadata_df, params)
    missing = [s for s in sample_ids if s not in group_map]
    if missing:
        raise ValueError(
            f"Samples in abundance matrix not found in metadata: {missing}"
        )
    groups    = [group_map[s] for s in sample_ids]
    color_map = make_color_map(groups)

    # Minimal summary_df for shared plot functions (sample_id + group only)
    summary_df = pd.DataFrame({"sample_id": sample_ids, "group": groups})

    # --- Normalization ---
    warnings: list[str] = []
    logging.info(f"Normalizing ({method})...")
    norm_df, method_used, log2_applied = run_normalization(raw_df, params, warnings)
    logging.info(
        f"  Normalization complete. log2 applied: {log2_applied}. "
        f"Warnings: {len(warnings)}"
    )

    # --- Write normalized matrix ---
    logging.info("Writing normalized matrix...")
    write_normalized_matrix(norm_df, peptide_df, abundance_prefix, outdir, run_id)

    # --- Write normalization summary ---
    write_normalization_summary(
        run_id, method_used, log2_applied,
        n_proteins, n_samples, warnings, outdir,
    )

    # --- Post-normalization diagnostic plots ---
    logging.info("Generating post-normalization diagnostic plots...")

    box_fig = plot_intensity_boxplots(
        norm_df, summary_df, color_map, run_id,
        label="Post-Normalization",
    )
    save_plot(box_fig, outdir / f"{run_id}.postnorm_boxplots")

    density_fig = plot_density(
        norm_df, summary_df, color_map, run_id,
        label="Post-Normalization",
    )
    save_plot(density_fig, outdir / f"{run_id}.postnorm_density")

    pca_df, pca_fig = compute_and_plot_pca(
        norm_df, summary_df, color_map, run_id,
        label="Post-Normalization",
    )
    pca_df.to_parquet(outdir / f"{run_id}.postnorm_pca_results.parquet", index=False)
    logging.info(f"  Saved: {run_id}.postnorm_pca_results.parquet")
    save_plot(pca_fig, outdir / f"{run_id}.postnorm_pca_scatter")

    corr_df, corr_fig = compute_and_plot_correlation(
        norm_df, summary_df, color_map, run_id,
        label="Post-Normalization",
    )
    corr_df.to_parquet(outdir / f"{run_id}.postnorm_correlation_matrix.parquet")
    logging.info(f"  Saved: {run_id}.postnorm_correlation_matrix.parquet")
    save_plot(corr_fig, outdir / f"{run_id}.postnorm_clustermap")

    # --- CV computation ---
    logging.info("Computing per-protein CV (back-transformed linear scale)...")
    cv_df = compute_cv_summary(norm_df, group_map, sample_ids)
    write_cv_summary(cv_df, outdir, run_id)

    # Log CV summary statistics
    groups_unique = sorted(set(group_map.values()))
    for grp in groups_unique:
        col = f"cv_{grp}"
        if col in cv_df.columns:
            median_cv = cv_df[col].median()
            pct_over50 = (cv_df[col] > 0.5).mean() * 100
            logging.info(
                f"  CV {grp}: median={median_cv:.3f} "
                f"({pct_over50:.1f}% proteins with CV > 50%)"
            )

    logging.info(
        f"Module 03 NORMALIZE complete. "
        f"{n_proteins} proteins, {n_samples} samples, method={method_used}."
    )


if __name__ == "__main__":
    main()
