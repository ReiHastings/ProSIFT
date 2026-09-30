#!/usr/bin/env python3
"""
Title:         impute.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-03-30
Last Modified: 2026-04-07
Purpose:       Module 03 IMPUTE process. Classifies missing values as MNAR or MAR
               using Module 01's detection filter table (DEP-style: SINGLE-GROUP
               proteins = MNAR in their absent group, PASSED proteins with sporadic
               NaN = MAR). Applies MinProb for MNAR values and protein-wise KNN for
               MAR values. Produces two diagnostic plots and a per-protein imputation
               summary. No NaN values remain in the output matrix. Also produces
               a per-cell imputation mask (observed/mnar/mar) for downstream
               re-rendering by Module 03b (QC Report Assembly).
Inputs:
  --matrix       {run_id}.normalized_matrix.parquet  (Module 03 NORMALIZE)
  --metadata     {run_id}.validated_metadata.parquet  (Module 01 VALIDATE_INPUTS)
  --filter-table {run_id}.detection_filter_table.csv  (Module 01 FILTER_PROTEINS)
  --params       {run_id}_params.yml
Outputs:
  {run_id}.imputed_matrix.parquet
  {run_id}.imputation_mask.parquet
  {run_id}.imputation_summary.parquet/.csv
  {run_id}.imputation_summary.txt
  {run_id}.imputation_distributions.png/.html
  {run_id}.imputation_fractions.png/.html
Usage:
  impute.py --matrix CTXcyto_WT_vs_CTXcyto_KO.normalized_matrix.parquet \
            --metadata CTXcyto_WT_vs_CTXcyto_KO.validated_metadata.parquet \
            --filter-table CTXcyto_WT_vs_CTXcyto_KO.detection_filter_table.csv \
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
import plotly.graph_objects as go
import yaml
from prosift_plot_utils import make_color_map, save_plot
from scipy.stats import gaussian_kde
from sklearn.impute import KNNImputer

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="impute.py",
        description="ProSIFT Module 03 IMPUTE: MNAR/MAR classification and imputation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  impute.py \\\n"
            "    --matrix       run.normalized_matrix.parquet \\\n"
            "    --metadata     run.validated_metadata.parquet \\\n"
            "    --filter-table run.detection_filter_table.csv \\\n"
            "    --params       run_params.yml \\\n"
            "    --run-id       run \\\n"
            "    --outdir       ."
        ),
    )
    parser.add_argument("--matrix",       required=True,
                        help="Normalized abundance matrix (Parquet)")
    parser.add_argument("--metadata",     required=True,
                        help="Validated metadata (Parquet)")
    parser.add_argument("--filter-table", required=True, dest="filter_table",
                        help="Detection filter table (CSV, from FILTER_PROTEINS)")
    parser.add_argument("--params",       required=True,
                        help="Run params.yml")
    parser.add_argument("--run-id",       required=True, dest="run_id",
                        help="Run identifier (used as output file prefix)")
    parser.add_argument("--outdir",       required=True,
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


def prepare_normalized_abundance(
    matrix_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """
    Extract log2 normalized abundance values and peptide count columns.

    Returns:
        norm_df          -- log2 abundance; index=protein_id, columns=bare sample_ids
        peptide_df       -- peptide count columns (protein_id index)
        sample_ids       -- ordered list of bare sample IDs
        abundance_prefix -- prefix string (e.g. "abundance_"), for output reconstruction
    """
    abundance_prefix = params["input"].get("abundance_prefix", "")
    peptide_prefix   = params["input"].get("peptide_count_prefix", "peptide_count_")

    all_cols     = matrix_df.columns.tolist()
    peptide_cols = [c for c in all_cols if c.startswith(peptide_prefix)]
    abund_cols   = [c for c in all_cols
                    if c != "protein_id" and not c.startswith(peptide_prefix)]

    if not abund_cols:
        raise ValueError("No abundance columns found in normalized matrix.")

    sample_ids = (
        [c[len(abundance_prefix):] for c in abund_cols]
        if abundance_prefix
        else list(abund_cols)
    )

    norm_df = matrix_df.set_index("protein_id")[abund_cols].copy()
    norm_df.columns = sample_ids

    peptide_df = (
        matrix_df.set_index("protein_id")[peptide_cols].copy()
        if peptide_cols
        else pd.DataFrame(index=matrix_df.set_index("protein_id").index)
    )

    return norm_df, peptide_df, sample_ids, abundance_prefix


def build_group_map(metadata_df: pd.DataFrame, params: dict) -> dict[str, str]:
    """Return {sample_id: group_label} from validated metadata."""
    group_column = params["design"]["group_column"]
    if group_column not in metadata_df.columns:
        raise ValueError(f"Group column '{group_column}' not found in metadata.")
    return dict(zip(metadata_df["sample_id"].astype(str),
                    metadata_df[group_column].astype(str), strict=True))


# ============================================================
# MNAR / MAR CLASSIFICATION
# ============================================================

def classify_missingness(
    norm_df: pd.DataFrame,
    filter_df: pd.DataFrame,
    group_map: dict[str, str],
    sample_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Assign MNAR or MAR classification to each missing value using the
    detection filter table (DEP-style rule):

      SINGLE-GROUP proteins:
        - Missing values in the ABSENT group (0 detections) -> MNAR
        - Any sporadic missing values in the DETECTED group -> MAR

      PARTIAL proteins:
        - Missing values in the sub-threshold group -> MNAR (likely below
          detection limit, similar to SINGLE-GROUP but with some signal)
        - Any sporadic missing values in the passing group -> MAR

      PASSED proteins:
        - Any missing values -> MAR (sporadic technical dropout)

      SPARSE / ABSENT proteins should not appear in the normalized matrix.
      If they do (unexpected), their missing values are left unclassified.

    Returns:
        mnar_mask    -- bool DataFrame, same shape as norm_df; True = impute MNAR
        mar_mask     -- bool DataFrame, same shape as norm_df; True = impute MAR
        protein_class -- per-protein Series: "MNAR", "MAR", "mixed", or "none"
    """
    groups  = sorted(set(group_map.values()))
    grp_samples: dict[str, list[str]] = {
        g: [s for s in sample_ids if group_map[s] == g] for g in groups
    }

    filter_lookup = filter_df.set_index("protein_id")["filter_status"]
    missing_mask  = norm_df.isna()

    mnar_mask     = pd.DataFrame(False, index=norm_df.index, columns=norm_df.columns)
    mar_mask      = pd.DataFrame(False, index=norm_df.index, columns=norm_df.columns)
    protein_class = pd.Series("none", index=norm_df.index, dtype=str)

    for protein_id in norm_df.index:
        # Proteins not in the filter table are skipped (should not happen)
        if protein_id not in filter_lookup.index:
            continue

        prot_missing = missing_mask.loc[protein_id]
        if not prot_missing.any():
            # No missing values -- no imputation needed
            continue

        status = filter_lookup.loc[protein_id]

        if status == "SINGLE-GROUP":
            # Identify the absent group: all replicates are NaN
            absent_grp = None
            for g in groups:
                if prot_missing[grp_samples[g]].all():
                    absent_grp = g
                    break

            if absent_grp is not None:
                # Absent group missing values -> MNAR
                for s in grp_samples[absent_grp]:
                    if prot_missing[s]:
                        mnar_mask.loc[protein_id, s] = True
                # Detected group sporadic missing values -> MAR
                for g in groups:
                    if g == absent_grp:
                        continue
                    for s in grp_samples[g]:
                        if prot_missing[s]:
                            mar_mask.loc[protein_id, s] = True
            else:
                # SINGLE-GROUP but no group is completely absent -- treat all as MNAR
                mnar_mask.loc[protein_id] = prot_missing
            protein_class.loc[protein_id] = "MNAR"

        elif status == "PARTIAL":
            # Sub-threshold group has some detections but below min_detections.
            # The missing values in that group are likely below detection limit
            # (MNAR), while sporadic gaps in the passing group are MAR.
            # Identify the sub-threshold group: has missing values but is NOT
            # completely absent (that would be SINGLE-GROUP).
            has_mnar = False
            has_mar = False
            for g in groups:
                g_missing = prot_missing[grp_samples[g]]
                g_all_missing = g_missing.all()
                g_any_missing = g_missing.any()

                if g_any_missing and not g_all_missing:
                    # Group has a mix of detected and missing values.
                    # Count how many are detected in this group.
                    n_detected = (~g_missing).sum()
                    n_total = len(grp_samples[g])
                    # For the sub-threshold group (minority detections),
                    # missing values are MNAR. For the passing group
                    # (majority detections), missing values are MAR.
                    # Heuristic: if fewer than half are detected, treat
                    # as the sub-threshold group (MNAR); otherwise MAR.
                    if n_detected < n_total / 2:
                        for s in grp_samples[g]:
                            if prot_missing[s]:
                                mnar_mask.loc[protein_id, s] = True
                        has_mnar = True
                    else:
                        for s in grp_samples[g]:
                            if prot_missing[s]:
                                mar_mask.loc[protein_id, s] = True
                        has_mar = True
                elif g_all_missing:
                    # Entire group missing -- MNAR (should not happen for
                    # PARTIAL, but handle defensively)
                    for s in grp_samples[g]:
                        mnar_mask.loc[protein_id, s] = True
                    has_mnar = True
                # else: no missing values in this group, nothing to classify

            protein_class.loc[protein_id] = "mixed" if (has_mnar and has_mar) else (
                "MNAR" if has_mnar else "MAR"
            )

        elif status == "PASSED":
            # Sporadic missingness -> all MAR
            mar_mask.loc[protein_id] = prot_missing
            protein_class.loc[protein_id] = "MAR"

        # SPARSE / ABSENT: should not be in the normalized matrix; leave unclassified

    return mnar_mask, mar_mask, protein_class


# ============================================================
# IMPUTATION METHODS
# ============================================================

def impute_minprob(
    norm_df: pd.DataFrame,
    target_mask: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    MinProb imputation for MNAR (or all missing in single mode).

    Per sample: draw imputed values from Normal(q-th quantile, scale * SD),
    where quantile and SD are computed from the sample's observed values.
    Draws use the provided Generator (seeded externally for reproducibility).

    Parameters read from params:
      imputation.minprob_quantile  -- quantile of observed distribution to center on
      imputation.minprob_scale     -- multiplier on sample SD for imputation width
    """
    quantile_q = params["imputation"]["minprob_quantile"]
    scale      = params["imputation"]["minprob_scale"]

    result = norm_df.copy()

    for col in norm_df.columns:
        positions = target_mask[col]
        n_impute  = int(positions.sum())
        if n_impute == 0:
            continue

        observed = norm_df[col].dropna().values
        if len(observed) == 0:
            logging.warning(f"  MinProb: sample '{col}' has no observed values -- cannot impute.")
            continue

        center = float(np.quantile(observed, quantile_q))
        width  = float(np.std(observed, ddof=1)) * scale

        imputed_vals = rng.normal(loc=center, scale=width, size=n_impute)
        result.loc[positions, col] = imputed_vals

    return result


def impute_knn(
    norm_df: pd.DataFrame,
    target_mask: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Protein-wise KNN imputation for MAR (or all missing in single mode).

    sklearn's KNNImputer on the (n_proteins x n_samples) matrix treats each
    protein as a "sample" and each expression value as a "feature". For a
    protein with a missing value at sample S, it finds k proteins with similar
    abundance profiles (across samples where both have data) and imputes using
    a distance-weighted average of those neighbors' values at sample S.

    Only positions flagged in target_mask are considered missing for
    imputation. Positions already filled (e.g. by MinProb) are treated as
    observed and used as neighbors.
    """
    k = params["imputation"]["knn_k"]

    if not target_mask.any().any():
        return norm_df.copy()

    imputer = KNNImputer(n_neighbors=k, weights="distance")

    # Only MAR positions should remain as NaN at this point (MNAR already filled).
    # KNNImputer imputes all NaN in the matrix, which is exactly what we want.
    imputed_values = imputer.fit_transform(norm_df.values)

    return pd.DataFrame(imputed_values, index=norm_df.index, columns=norm_df.columns)


def impute_left_censored(
    norm_df: pd.DataFrame,
    target_mask: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Left-censored (Perseus-style) imputation. Available in single mode only.

    Per sample: draw from Normal(mean - downshift * SD, width * SD), where
    mean and SD are computed from the sample's observed values.

    Parameters read from params:
      imputation.left_censored_downshift -- SDs below sample mean for center
      imputation.left_censored_width     -- multiplier on sample SD for width
    """
    downshift = params["imputation"]["left_censored_downshift"]
    width     = params["imputation"]["left_censored_width"]

    result = norm_df.copy()

    for col in norm_df.columns:
        positions = target_mask[col]
        n_impute  = int(positions.sum())
        if n_impute == 0:
            continue

        observed = norm_df[col].dropna().values
        if len(observed) == 0:
            logging.warning(
                f"  Left-censored: sample '{col}' has no observed values -- cannot impute."
            )
            continue

        mean_obs = float(np.mean(observed))
        sd_obs   = float(np.std(observed, ddof=1))
        center   = mean_obs - downshift * sd_obs

        imputed_vals = rng.normal(loc=center, scale=width * sd_obs, size=n_impute)
        result.loc[positions, col] = imputed_vals

    return result


# ============================================================
# IMPUTATION DISPATCHER
# ============================================================

def run_imputation(
    norm_df: pd.DataFrame,
    filter_df: pd.DataFrame,
    group_map: dict[str, str],
    sample_ids: list[str],
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """
    Dispatch to mixed or single-method imputation based on params.

    Returns:
        imputed_df    -- fully imputed DataFrame (no NaN)
        mnar_mask     -- bool DataFrame; positions imputed by MNAR method
        mar_mask      -- bool DataFrame; positions imputed by MAR method
        protein_class -- per-protein class string
        method_str    -- human-readable method description for summary
    """
    mode = params["imputation"]["mode"].lower()

    if mode == "mixed":
        mnar_method = params["imputation"]["mnar_method"].lower()
        mar_method  = params["imputation"]["mar_method"].lower()

        logging.info("Classifying missing values (MNAR/MAR)...")
        mnar_mask, mar_mask, protein_class = classify_missingness(
            norm_df, filter_df, group_map, sample_ids
        )

        n_mnar = int(mnar_mask.sum().sum())
        n_mar  = int(mar_mask.sum().sum())
        logging.info(f"  MNAR positions: {n_mnar}  MAR positions: {n_mar}")

        # --- Apply MNAR method ---
        if mnar_method != "minprob":
            raise ValueError(
                f"Unknown imputation.mnar_method: '{mnar_method}'. "
                "Currently only 'minprob' is supported."
            )
        logging.info("Imputing MNAR values (MinProb)...")
        imputed_df = impute_minprob(norm_df, mnar_mask, params, rng)

        # --- Apply MAR method to remaining NaN ---
        if mar_method != "knn":
            raise ValueError(
                f"Unknown imputation.mar_method: '{mar_method}'. "
                "Currently only 'knn' is supported."
            )
        if n_mar > 0:
            logging.info("Imputing MAR values (protein-wise KNN)...")
            imputed_df = impute_knn(imputed_df, mar_mask, params)

        method_str = f"mixed (MNAR: {mnar_method}, MAR: {mar_method})"

    elif mode == "single":
        single_method = params["imputation"]["single_method"].lower()
        logging.info(f"Single-method imputation: {single_method}")

        all_missing = norm_df.isna()
        protein_class = pd.Series("none", index=norm_df.index, dtype=str)
        protein_class[all_missing.any(axis=1)] = f"single:{single_method}"

        # In single mode: all missing → MNAR method mask (minprob/left_censored)
        # or all missing → MAR method mask (knn), for summary bookkeeping.
        mnar_mask = pd.DataFrame(False, index=norm_df.index, columns=norm_df.columns)
        mar_mask  = pd.DataFrame(False, index=norm_df.index, columns=norm_df.columns)

        if single_method == "minprob":
            mnar_mask = all_missing.copy()
            imputed_df = impute_minprob(norm_df, all_missing, params, rng)
        elif single_method == "knn":
            mar_mask = all_missing.copy()
            imputed_df = impute_knn(norm_df, all_missing, params)
        elif single_method == "left_censored":
            mnar_mask = all_missing.copy()
            imputed_df = impute_left_censored(norm_df, all_missing, params, rng)
        else:
            raise ValueError(
                f"Unknown imputation.single_method: '{single_method}'. "
                "Valid options: minprob, knn, left_censored."
            )

        method_str = f"single:{single_method}"

    else:
        raise ValueError(
            f"Unknown imputation.mode: '{mode}'. Valid options: mixed, single."
        )

    return imputed_df, mnar_mask, mar_mask, protein_class, method_str


# ============================================================
# IMPUTATION SUMMARY TABLE
# ============================================================

def build_imputation_summary(
    norm_df: pd.DataFrame,
    filter_df: pd.DataFrame,
    protein_class: pd.Series,
    mnar_mask: pd.DataFrame,
    mar_mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-protein imputation summary table.

    Columns:
      protein_id, filter_status, imputation_class,
      n_mnar_imputed, n_mar_imputed, n_imputed_total
    """
    filter_lookup = filter_df.set_index("protein_id")["filter_status"]

    rows = []
    for protein_id in norm_df.index:
        filter_status = (
            filter_lookup.loc[protein_id]
            if protein_id in filter_lookup.index
            else "unknown"
        )
        n_mnar = int(mnar_mask.loc[protein_id].sum())
        n_mar  = int(mar_mask.loc[protein_id].sum())
        rows.append({
            "protein_id":        protein_id,
            "filter_status":     filter_status,
            "imputation_class":  protein_class.loc[protein_id],
            "n_mnar_imputed":    n_mnar,
            "n_mar_imputed":     n_mar,
            "n_imputed_total":   n_mnar + n_mar,
        })

    return pd.DataFrame(rows)


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def _kde_trace(
    values: np.ndarray,
    label: str,
    color: str,
    dash: str = "solid",
) -> go.Scatter:
    """Compute a KDE curve and return a Plotly Scatter trace."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return go.Scatter(x=[], y=[], name=label)

    kde = gaussian_kde(values, bw_method="scott")
    x_range = np.linspace(values.min(), values.max(), 400)
    y_vals  = kde(x_range)

    return go.Scatter(
        x=x_range,
        y=y_vals,
        mode="lines",
        name=label,
        line=dict(color=color, dash=dash, width=2),
    )


def plot_imputation_distributions(
    norm_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    mnar_mask: pd.DataFrame,
    mar_mask: pd.DataFrame,
    run_id: str,
    mode: str,
) -> go.Figure:
    """
    Overlaid KDE density curves:
      (1) Observed values (all non-NaN values in norm_df)
      (2) MNAR-imputed values (positions where mnar_mask is True)
      (3) MAR-imputed values (positions where mar_mask is True, mixed mode only)

    The MNAR curve should appear as a left-shifted shoulder. The MAR curve
    should overlap the main distribution.
    """
    observed_vals = norm_df.values.flatten()
    observed_vals = observed_vals[~np.isnan(observed_vals)]

    mnar_vals = imputed_df.values[mnar_mask.values]
    mar_vals  = imputed_df.values[mar_mask.values]

    fig = go.Figure()
    fig.add_trace(_kde_trace(observed_vals, "Observed", color="#1f77b4"))

    if mnar_vals.size > 0:
        label = "MNAR-imputed (MinProb)" if mode == "mixed" else "Imputed"
        fig.add_trace(_kde_trace(mnar_vals, label, color="#d62728", dash="dash"))

    if mar_vals.size > 0 and mode == "mixed":
        fig.add_trace(_kde_trace(mar_vals, "MAR-imputed (KNN)", color="#2ca02c", dash="dot"))

    fig.update_layout(
        title=f"Observed vs. Imputed Distributions: {run_id}",
        xaxis_title="log2 Abundance",
        yaxis_title="Density",
        height=480,
        width=700,
        legend_title_text="Value type",
    )
    return fig


def plot_imputation_fractions(
    missing_mask: pd.DataFrame,
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    run_id: str,
) -> go.Figure:
    """
    Per-sample imputation fraction bar chart. Bar height = fraction of protein
    values imputed in that sample. Samples ordered by group. Colored by group.

    A sample with a dramatically higher fraction than its replicates corroborates
    outlier flags from Module 02.
    """
    sample_order = (
        summary_df.sort_values(["group", "sample_id"])["sample_id"].tolist()
    )
    group_of = dict(zip(summary_df["sample_id"], summary_df["group"].astype(str), strict=True))

    n_proteins = len(missing_mask)
    shown_groups: set[str] = set()

    fig = go.Figure()
    for sid in sample_order:
        grp = group_of[sid]
        show_legend = grp not in shown_groups
        shown_groups.add(grp)

        frac = float(missing_mask[sid].sum()) / n_proteins if n_proteins > 0 else 0.0

        fig.add_trace(
            go.Bar(
                x=[sid],
                y=[frac],
                name=grp,
                marker_color=color_map[grp],
                legendgroup=grp,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>{sid}</b><br>"
                    f"Group: {grp}<br>"
                    f"Imputation fraction: %{{y:.3f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Per-Sample Imputation Fraction: {run_id}",
        xaxis_title="Sample",
        yaxis_title="Fraction imputed",
        yaxis=dict(range=[0, 1]),
        height=460,
        width=700,
        showlegend=True,
        barmode="group",
    )
    return fig


# ============================================================
# OUTPUT WRITERS
# ============================================================

def write_imputed_matrix(
    imputed_df: pd.DataFrame,
    peptide_df: pd.DataFrame,
    abundance_prefix: str,
    outdir: Path,
    run_id: str,
) -> None:
    """Restore abundance prefix and write imputed matrix as Parquet."""
    out = imputed_df.copy()
    out.columns = [f"{abundance_prefix}{c}" for c in out.columns]

    if not peptide_df.empty:
        out = pd.concat([out, peptide_df], axis=1)

    out = out.reset_index()  # restore protein_id as column
    path = outdir / f"{run_id}.imputed_matrix.parquet"
    out.to_parquet(path, index=False)
    logging.info(f"  Saved: {path.name}")


def build_imputation_mask(
    norm_df: pd.DataFrame,
    mnar_mask: pd.DataFrame,
    mar_mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a per-cell categorical imputation mask.

    Same shape as the abundance matrix (index = protein_id, columns = sample_ids).
    Each cell is one of: 'observed', 'mnar', 'mar'.

    Parameters
    ----------
    norm_df : pd.DataFrame
        Pre-imputation normalized matrix (used to identify observed values).
    mnar_mask : pd.DataFrame
        Boolean mask of positions imputed via MNAR method.
    mar_mask : pd.DataFrame
        Boolean mask of positions imputed via MAR method.
    """
    mask = pd.DataFrame('observed', index=norm_df.index, columns=norm_df.columns)
    mask[mnar_mask] = 'mnar'
    mask[mar_mask] = 'mar'
    return mask


def write_imputation_mask(
    mask_df: pd.DataFrame,
    outdir: Path,
    run_id: str,
) -> None:
    """Write the per-cell imputation mask as Parquet."""
    out = mask_df.reset_index()  # restore protein_id as column
    path = outdir / f'{run_id}.imputation_mask.parquet'
    out.to_parquet(path, index=False)
    logging.info(f'  Saved: {path.name}')


def write_imputation_summary_table(
    summary_df: pd.DataFrame,
    outdir: Path,
    run_id: str,
) -> None:
    summary_df.to_parquet(
        outdir / f"{run_id}.imputation_summary.parquet", index=False
    )
    summary_df.to_csv(
        outdir / f"{run_id}.imputation_summary.csv", index=False
    )
    logging.info(f"  Saved: {run_id}.imputation_summary.parquet/.csv")


def write_imputation_summary_txt(
    summary_df: pd.DataFrame,
    run_id: str,
    method_str: str,
    params: dict,
    outdir: Path,
) -> None:
    """Write a plain-text imputation summary."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_mnar_proteins  = int((summary_df["imputation_class"] == "MNAR").sum())
    n_mar_proteins   = int((summary_df["imputation_class"] == "MAR").sum())
    n_mnar_values    = int(summary_df["n_mnar_imputed"].sum())
    n_mar_values     = int(summary_df["n_mar_imputed"].sum())
    n_total_imputed  = int(summary_df["n_imputed_total"].sum())
    n_total_proteins = len(summary_df)

    imp_params = params.get("imputation", {})

    lines = [
        f"ProSIFT Imputation Summary -- {run_id}",
        f"Generated: {now}",
        "",
        f"Method:                  {method_str}",
        f"Proteins (total):        {n_total_proteins}",
        f"Proteins with MNAR imputation: {n_mnar_proteins}",
        f"Proteins with MAR imputation:  {n_mar_proteins}",
        "",
        f"Values imputed (MNAR):   {n_mnar_values}",
        f"Values imputed (MAR):    {n_mar_values}",
        f"Values imputed (total):  {n_total_imputed}",
        "",
        "Parameters:",
        f"  imputation.mode:             {imp_params.get('mode', 'mixed')}",
        f"  imputation.minprob_quantile: {imp_params.get('minprob_quantile', 0.01)}",
        f"  imputation.minprob_scale:    {imp_params.get('minprob_scale', 0.3)}",
        f"  imputation.knn_k:            {imp_params.get('knn_k', 10)}",
        f"  imputation.random_seed:      {imp_params.get('random_seed', 42)}",
    ]

    path = outdir / f"{run_id}.imputation_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"  Saved: {path.name}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    setup_logging()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id

    logging.info(f"Module 03 IMPUTE -- {run_id}")

    # --- Load params ---
    params = load_params(Path(args.params))
    mode   = params["imputation"]["mode"].lower()
    seed   = params["imputation"]["random_seed"]
    logging.info(f"  imputation.mode={mode}, random_seed={seed}")

    # Set random seed before any draws
    rng = np.random.default_rng(seed)

    # --- Load inputs ---
    logging.info("Loading input files...")
    norm_matrix_df = pd.read_parquet(args.matrix)
    metadata_df    = pd.read_parquet(args.metadata)
    filter_df      = pd.read_csv(args.filter_table)

    # --- Prepare data ---
    norm_df, peptide_df, sample_ids, abundance_prefix = prepare_normalized_abundance(
        norm_matrix_df, params
    )
    group_map = build_group_map(metadata_df, params)
    missing   = [s for s in sample_ids if s not in group_map]
    if missing:
        raise ValueError(
            f"Samples in normalized matrix not found in metadata: {missing}"
        )

    groups    = [group_map[s] for s in sample_ids]
    color_map = make_color_map(groups)
    summary_df_plot = pd.DataFrame({"sample_id": sample_ids, "group": groups})

    n_proteins     = len(norm_df)
    n_samples      = len(sample_ids)
    n_missing_init = int(norm_df.isna().sum().sum())
    missing_mask   = norm_df.isna()  # snapshot before imputation for diagnostics
    logging.info(
        f"  {n_proteins} proteins, {n_samples} samples, "
        f"{n_missing_init} missing values before imputation"
    )

    # --- Impute ---
    imputed_df, mnar_mask, mar_mask, protein_class, method_str = run_imputation(
        norm_df, filter_df, group_map, sample_ids, params, rng
    )

    # --- Verify completeness ---
    n_remaining_nan = int(imputed_df.isna().sum().sum())
    if n_remaining_nan > 0:
        logging.warning(
            f"  {n_remaining_nan} NaN values remain after imputation. "
            "Check that all MNAR/MAR positions were covered."
        )
    else:
        logging.info("  All missing values imputed. No NaN remain.")

    # --- Build summary table ---
    imputation_summary = build_imputation_summary(
        norm_df, filter_df, protein_class, mnar_mask, mar_mask
    )

    # --- Build imputation mask ---
    imp_mask = build_imputation_mask(norm_df, mnar_mask, mar_mask)

    # --- Write outputs ---
    logging.info("Writing outputs...")
    write_imputed_matrix(imputed_df, peptide_df, abundance_prefix, outdir, run_id)
    write_imputation_mask(imp_mask, outdir, run_id)
    write_imputation_summary_table(imputation_summary, outdir, run_id)
    write_imputation_summary_txt(imputation_summary, run_id, method_str, params, outdir)

    # --- Diagnostic plots ---
    logging.info("Generating imputation diagnostic plots...")

    dist_fig = plot_imputation_distributions(
        norm_df, imputed_df, mnar_mask, mar_mask, run_id, mode
    )
    save_plot(dist_fig, outdir / f"{run_id}.imputation_distributions")

    frac_fig = plot_imputation_fractions(
        missing_mask, summary_df_plot, color_map, run_id
    )
    save_plot(frac_fig, outdir / f"{run_id}.imputation_fractions")

    # --- Log summary ---
    n_mnar_proteins = int((imputation_summary["imputation_class"] == "MNAR").sum())
    n_mar_proteins  = int((imputation_summary["imputation_class"] == "MAR").sum())
    n_mnar_vals     = int(imputation_summary["n_mnar_imputed"].sum())
    n_mar_vals      = int(imputation_summary["n_mar_imputed"].sum())
    logging.info(
        f"  MNAR: {n_mnar_proteins} proteins ({n_mnar_vals} values imputed)"
    )
    logging.info(
        f"  MAR:  {n_mar_proteins} proteins ({n_mar_vals} values imputed)"
    )
    logging.info(
        f"Module 03 IMPUTE complete. "
        f"{n_proteins} proteins, {n_samples} samples, mode={mode}."
    )


if __name__ == "__main__":
    main()
