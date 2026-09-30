#!/usr/bin/env python3
"""
Title:         prenorm_qc.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-03-27
Last Modified: 2026-04-24 (audit remediation; followed by audit follow-ups:
               flag_low_detection switched from median to max group center to
               eliminate the n=3 tied-lows blind spot; the other three flags
               normalized to equality-based tie handling)
Purpose:       Module 02 pre-normalization QC/EDA. Computes per-sample intensity summaries,
               intensity distribution box plots, Q-Q plots, PCA, and sample-to-sample
               correlations on the post-filter, pre-normalization abundance data. Synthesizes
               results into per-sample outlier flags and a consolidated HTML QC report.
               This module does not modify any data.
Inputs:
  --matrix    {run_id}.filtered_matrix.parquet  (Module 01 FILTER_PROTEINS)
  --metadata  {run_id}.validated_metadata.parquet  (Module 01 VALIDATE_INPUTS)
  --params    {run_id}_params.yml
Outputs:
  {run_id}.sample_summary.parquet/.csv
  {run_id}.sample_flags.parquet/.csv
  {run_id}.correlation_matrix.parquet
  {run_id}.pca_results.parquet
  {run_id}.sample_intensity_summary.png/.html
  {run_id}.intensity_boxplots.png/.html
  {run_id}.density_plot.png/.html
  {run_id}.qq_plot.png/.html
  {run_id}.pca_scatter.png/.html
  {run_id}.correlation_heatmap.png/.html
  {run_id}.prenorm_qc_report.html
Usage:
  prenorm_qc.py --matrix CTXcyto_WT_vs_CTXcyto_KO.filtered_matrix.parquet \
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
import plotly.graph_objects as go
import scipy.stats
import yaml
from plotly.subplots import make_subplots
from prosift_plot_utils import (
    compute_and_plot_correlation,
    compute_and_plot_pca,
    make_color_map,
    plot_density,
    plot_intensity_boxplots,
    save_plot,
)

# ============================================================
# CONSTANTS: Outlier flag thresholds (see Module 02 spec, Section 4.6)
# ============================================================

# flag_low_detection: gap from group median must exceed this fraction of total proteins
DETECTION_FLAG_PCT = 0.02

# flag_extreme_median: most extreme sample's deviation must exceed this many group MADs
MEDIAN_FLAG_MAD_MULT = 2.0

# flag_pca_outlier: max within-group PCA distance must exceed this multiplier of median distance
PCA_OUTLIER_DIST_MULT = 1.5


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prenorm_qc.py",
        description="ProSIFT Module 02: Pre-normalization QC/EDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  prenorm_qc.py \\\n"
            "    --matrix run.filtered_matrix.parquet \\\n"
            "    --metadata run.validated_metadata.parquet \\\n"
            "    --params run_params.yml \\\n"
            "    --run-id run \\\n"
            "    --outdir ."
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
# DATA LOADING AND PREPARATION (Section 4.0)
# ============================================================

def load_params(params_path: Path) -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def prepare_abundance(
    matrix_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Identify abundance columns, parse sample IDs, and build raw and log2 DataFrames.

    The log2 working copy uses log2(x) when abundance_type is 'raw', and the
    data as-is when it is already 'log2' or 'normalized'. Non-positive raw
    values are converted to NaN upstream at validation (Module 01), so no
    pseudocount is needed; plain log2(x) keeps this QC transform consistent
    with the NORMALIZE process (which also uses log2(x)).

    Returns:
        raw_df   -- raw abundance values; index=protein_id, columns=sample_ids
        log2_df  -- log2 working copy; same index and columns
        sample_ids -- ordered list of sample IDs
    """
    abundance_prefix = params["input"].get("abundance_prefix", "")
    abundance_type = params["input"]["abundance_type"]

    # Identify abundance columns: exclude protein_id and peptide_count_* columns
    all_cols = matrix_df.columns.tolist()
    exclude = {"protein_id"} | {c for c in all_cols if c.startswith("peptide_count_")}
    abund_cols = [c for c in all_cols if c not in exclude]

    if not abund_cols:
        raise ValueError("No abundance columns found in filtered matrix.")

    # Parse sample IDs by stripping the abundance prefix.
    # Validate prefix consistency before stripping. Without this check, a
    # column that does not start with the prefix would be silently truncated
    # by N characters, producing a malformed sample ID that fails downstream
    # with an opaque "sample not in metadata" error. Fail loudly instead.
    if abundance_prefix:
        non_prefixed = [c for c in abund_cols if not c.startswith(abundance_prefix)]
        if non_prefixed:
            raise ValueError(
                f"Abundance columns do not all start with the configured "
                f"abundance_prefix='{abundance_prefix}': {non_prefixed}"
            )
        sample_ids = [c[len(abundance_prefix):] for c in abund_cols]
    else:
        sample_ids = list(abund_cols)

    # Build raw_df indexed by protein_id, with bare sample IDs as column names
    raw_df = matrix_df.set_index("protein_id")[abund_cols].copy()
    raw_df.columns = sample_ids

    # Non-positive raw values are removed at validation (Module 01), so by the
    # time data reaches QC there should be none. Guard defensively: a <= 0 value
    # here means validation was bypassed, and log2(x) would yield -Inf.
    if abundance_type == "raw":
        min_val = float(raw_df.min(skipna=True).min())
        if min_val <= 0:
            raise ValueError(
                f"Non-positive abundance value(s) found (min={min_val:.4f}) with "
                "abundance_type='raw'. These should be converted to NaN at "
                "validation (Module 01); re-run VALIDATE_INPUTS on this input."
            )
        log2_df = np.log2(raw_df)
    else:
        log2_df = raw_df.copy()

    return raw_df, log2_df, sample_ids


def build_group_map(metadata_df: pd.DataFrame, params: dict) -> dict[str, str]:
    """Return {sample_id: group_label} from validated metadata."""
    group_column = params["design"]["group_column"]
    if group_column not in metadata_df.columns:
        raise ValueError(f"Group column '{group_column}' not found in metadata.")
    return dict(zip(metadata_df["sample_id"].astype(str),
                    metadata_df[group_column].astype(str), strict=True))


# ============================================================
# SECTION 4.1: PER-SAMPLE INTENSITY SUMMARIES
# ============================================================

def compute_sample_summaries(
    raw_df: pd.DataFrame,
    log2_df: pd.DataFrame,
    group_map: dict[str, str],
    sample_ids: list[str],
    abundance_type: str,
) -> pd.DataFrame:
    """
    Compute per-sample descriptive statistics on the log2 working copy.
    total_intensity is computed on the raw linear scale only for 'raw' input.
    """
    rows = []
    for sid in sample_ids:
        log2_vals = log2_df[sid].dropna().values
        raw_vals = raw_df[sid].dropna().values

        n_detected = int(log2_df[sid].notna().sum())
        median_int = float(np.median(log2_vals)) if len(log2_vals) > 0 else np.nan
        total_int  = float(np.sum(raw_vals)) if abundance_type == "raw" else None
        mad_int    = (float(np.median(np.abs(log2_vals - np.median(log2_vals))))
                      if len(log2_vals) > 0 else np.nan)
        skew = float(scipy.stats.skew(log2_vals, bias=True))    if len(log2_vals) > 1 else np.nan
        kurt = float(scipy.stats.kurtosis(log2_vals, bias=True)) if len(log2_vals) > 1 else np.nan

        rows.append({
            "sample_id":        sid,
            "group":            group_map.get(sid, "unknown"),
            "n_detected":       n_detected,
            "median_intensity": median_int,
            "total_intensity":  total_int,
            "mad_intensity":    mad_int,
            "skewness":         skew,
            "kurtosis":         kurt,
        })

    return pd.DataFrame(rows)


def plot_sample_intensity_summary(
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    run_id: str,
    abundance_type: str,
) -> go.Figure:
    """
    Multipanel scatter plot: one panel per summary metric, samples on x-axis,
    colored by group. total_intensity panel included only for 'raw' input.
    """
    metrics = [
        ("n_detected",       "Proteins detected",       False),
        ("median_intensity",  "Median intensity (log2)", False),
        ("total_intensity",   "Total intensity (raw)",   True),   # raw-only panel
        ("mad_intensity",     "MAD intensity (log2)",    False),
        ("skewness",          "Skewness (log2)",         False),
        ("kurtosis",          "Excess kurtosis (log2)",  False),
    ]
    # Drop raw-only panel when input is not raw
    if abundance_type != "raw":
        metrics = [(m, lbl, raw_only) for m, lbl, raw_only in metrics if not raw_only]

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[lbl for _, lbl, _ in metrics],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    shown_groups: set[str] = set()
    for i, (metric, _, _) in enumerate(metrics):
        row = i // n_cols + 1
        col = i % n_cols + 1
        for grp, grp_df in summary_df.groupby("group"):
            show_legend = str(grp) not in shown_groups
            shown_groups.add(str(grp))
            fig.add_trace(
                go.Scatter(
                    x=grp_df["sample_id"],
                    y=grp_df[metric],
                    mode="markers",
                    marker=dict(color=color_map[str(grp)], size=12, line=dict(width=1, color="white")),
                    name=str(grp),
                    legendgroup=str(grp),
                    showlegend=show_legend,
                    hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>",
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=f"Per-Sample Intensity Summary: {run_id}",
        height=280 * n_rows + 80,
        width=950,
        legend_title_text="Group",
    )
    return fig


# ============================================================
# SECTION 4.2b: PER-SAMPLE DENSITY PLOTS
# (implemented in prosift_plot_utils.plot_density)
# ============================================================

# ============================================================
# SECTION 4.3: Q-Q PLOT
# ============================================================

def plot_qq(
    log2_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    run_id: str,
) -> go.Figure:
    """
    Overlaid Q-Q plot: each sample's quantiles vs. theoretical standard normal.
    Values are standardized (subtract mean, divide by SD) before plotting.
    Points on the y=x reference line indicate perfect normality.
    """
    fig = go.Figure()
    shown_groups: set[str] = set()
    all_theoretical: list[float] = []

    for sid in log2_df.columns:
        grp = str(summary_df.loc[summary_df["sample_id"] == sid, "group"].iloc[0])
        show_legend = grp not in shown_groups
        shown_groups.add(grp)

        values = log2_df[sid].dropna().values
        n = len(values)
        if n < 2:
            continue
        # Guard against zero-variance samples. Standardization would divide
        # by zero and silently produce NaN, which Plotly drops from the plot
        # without warning -- the sample would just disappear from the figure.
        if values.std(ddof=1) == 0:
            logging.warning(
                f"plot_qq: skipping {sid} (zero variance, cannot standardize)"
            )
            continue

        # Standardize and sort observed values
        std_vals = (values - values.mean()) / values.std(ddof=1)
        observed = np.sort(std_vals)

        # Theoretical quantiles of N(0,1) for n observations (Blom plotting positions)
        probs = np.linspace(1 / (n + 1), n / (n + 1), n)
        theoretical = scipy.stats.norm.ppf(probs)
        all_theoretical.extend(theoretical.tolist())

        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=observed,
                mode="markers",
                marker=dict(color=color_map[grp], size=3, opacity=0.5),
                name=grp,
                legendgroup=grp,
                legendgrouptitle_text=grp if show_legend else None,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>{sid}</b><br>"
                    "Theoretical: %{x:.3f}<br>"
                    "Observed: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Reference line y = x across the range of theoretical quantiles
    if all_theoretical:
        t_min, t_max = min(all_theoretical), max(all_theoretical)
        fig.add_trace(
            go.Scatter(
                x=[t_min, t_max],
                y=[t_min, t_max],
                mode="lines",
                line=dict(color="black", dash="dash", width=1.5),
                name="Normal reference (y = x)",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=f"Q-Q Plot (Normal Reference): {run_id}",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Observed quantiles (standardized)",
        height=520,
        width=700,
        legend_title_text="Group",
    )
    return fig


# ============================================================
# SECTION 4.4: PRINCIPAL COMPONENT ANALYSIS
# (implemented in prosift_plot_utils.compute_and_plot_pca)
# ============================================================

# ============================================================
# SECTION 4.5: SAMPLE-TO-SAMPLE CORRELATION HEATMAP
# (implemented in prosift_plot_utils.compute_and_plot_correlation)
# ============================================================

# ============================================================
# SECTION 4.6: OUTLIER DETECTION AND SAMPLE FLAGGING
# ============================================================

def compute_sample_flags(
    summary_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    n_total_proteins: int,
) -> pd.DataFrame:
    """
    Compute four per-sample outlier flags using convergent evidence across
    independent metrics. With only ~3 replicates per group, no single metric
    is reliable alone; agreement across multiple metrics indicates real problems.

    Flags:
      flag_low_detection   -- lowest n_detected in group AND gap > 2% of total proteins
      flag_extreme_median  -- most extreme median in group AND gap > 2 group MADs
      flag_pca_outlier     -- max PCA distance from centroid AND > 1.5x median distance
      flag_low_correlation -- lowest mean within-group correlation across all samples

    n_flags: integer count of True flags per sample (0-4).
    """
    flags = summary_df[["sample_id", "group"]].copy().reset_index(drop=True)

    # --- flag_low_detection ---
    # Reference statistic is the group MAX (best-detecting sample), not the
    # median. With n=3 and two samples tied as the lowest (e.g.,
    # [95, 85, 85]), the median collapses onto the tied minimum, the gap
    # becomes 0, and the flag silently misses a real 2/3 failure rate.
    # Using max keeps the gap meaningful in that case. Tradeoff: a single
    # anomalously-HIGH sample can pull the gap up and flag the remaining
    # "normal" samples (see spec Section 7 known limitation). Loud false
    # positive is preferred over silent miss for a QC diagnostic.
    detection_threshold = DETECTION_FLAG_PCT * n_total_proteins
    low_det: dict[str, bool] = {}
    for grp in summary_df["group"].unique():
        grp_df = summary_df[summary_df["group"] == grp]
        group_max = grp_df["n_detected"].max()
        min_val = grp_df["n_detected"].min()
        gap = group_max - min_val
        for _, row in grp_df.iterrows():
            sid = row["sample_id"]
            is_minimum = row["n_detected"] == min_val
            low_det[sid] = bool(is_minimum and gap > detection_threshold)

    flags["flag_low_detection"] = flags["sample_id"].map(low_det)

    # --- flag_extreme_median ---
    # Equality-based tie handling: any sample whose deviation equals the
    # within-group max deviation is flagged, not just the first encountered.
    # Consistent with the other three flags after the 2026-04-24 audit
    # follow-up; previously argmax returned only the first tied index.
    extreme_med: dict[str, bool] = {}
    for grp in summary_df["group"].unique():
        grp_df = summary_df[summary_df["group"] == grp]
        grp_medians = grp_df["median_intensity"].values
        med_of_meds = float(np.median(grp_medians))
        deviations = np.abs(grp_medians - med_of_meds)
        group_mad = float(np.median(deviations))
        max_dev = float(deviations.max())
        for i, (_, row) in enumerate(grp_df.iterrows()):
            sid = row["sample_id"]
            is_most_extreme = bool(deviations[i] == max_dev)
            flagged = bool(
                is_most_extreme
                and group_mad > 0
                and max_dev > MEDIAN_FLAG_MAD_MULT * group_mad
            )
            extreme_med[sid] = flagged

    flags["flag_extreme_median"] = flags["sample_id"].map(extreme_med)

    # --- flag_pca_outlier ---
    # Equality-based tie handling: any sample whose distance equals the
    # within-group max distance flags. Previously idxmax returned the first
    # tied index only.
    pca_flag: dict[str, bool] = {}
    for grp in pca_df["group"].unique():
        grp_pca = pca_df[pca_df["group"] == grp].copy()
        centroid_pc1 = grp_pca["PC1"].mean()
        centroid_pc2 = grp_pca["PC2"].mean()
        distances = np.sqrt(
            (grp_pca["PC1"] - centroid_pc1) ** 2
            + (grp_pca["PC2"] - centroid_pc2) ** 2
        )
        median_dist = float(distances.median())
        max_dist = float(distances.max())
        for idx in grp_pca.index:
            sid = grp_pca.loc[idx, "sample_id"]
            is_max = bool(distances[idx] == max_dist)
            flagged = bool(
                is_max
                and median_dist > 0
                and max_dist > PCA_OUTLIER_DIST_MULT * median_dist
            )
            pca_flag[sid] = flagged

    flags["flag_pca_outlier"] = flags["sample_id"].map(pca_flag)

    # --- flag_low_correlation ---
    # Mean Pearson correlation with own-group replicates (excluding self).
    # Equality-based tie handling: any sample whose mean correlation equals
    # the global minimum flags. In perfectly uniform correlation data, all
    # samples tie and all flag (each gets n_flags = 1, which the alert
    # already treats as the no-concern baseline). Previously min(...,
    # key=...) returned only the first tied sample, which was arbitrary
    # since "first" depends on row order rather than any scientific property.
    within_group_corr: dict[str, float] = {}
    for _, row in summary_df.iterrows():
        sid = row["sample_id"]
        grp = row["group"]
        same_group = summary_df[summary_df["group"] == grp]["sample_id"].tolist()
        others = [s for s in same_group if s != sid and s in corr_df.columns]
        if others and sid in corr_df.index:
            mean_corr = float(corr_df.loc[sid, others].mean())
        else:
            mean_corr = 1.0  # no peers (single sample in group)
        within_group_corr[sid] = mean_corr

    if within_group_corr:
        min_corr = min(within_group_corr.values())
        corr_flag: dict[str, bool] = {
            sid: within_group_corr[sid] == min_corr for sid in flags["sample_id"]
        }
    else:
        corr_flag = {sid: False for sid in flags["sample_id"]}

    flags["flag_low_correlation"] = flags["sample_id"].map(corr_flag)

    # --- n_flags: count of True flags ---
    flag_cols = [
        "flag_low_detection", "flag_extreme_median",
        "flag_pca_outlier", "flag_low_correlation",
    ]
    flags["n_flags"] = flags[flag_cols].sum(axis=1).astype(int)

    return flags


# ============================================================
# OUTPUT: CONSOLIDATED HTML REPORT
# ============================================================

def generate_html_report(
    run_id: str,
    summary_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    plots: dict[str, go.Figure],
    outdir: Path,
    n_total_proteins: int = 0,
) -> Path:
    """
    Build a self-contained HTML QC report with embedded plots (plotly.js embedded
    inline for offline viewing), summary tables, and a flagged-sample alert section.
    """
    report_path = outdir / f"{run_id}.prenorm_qc_report.html"
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Summary table HTML ---
    display_summary = summary_df.copy()
    for col in ["median_intensity", "mad_intensity", "skewness", "kurtosis"]:
        if col in display_summary.columns:
            display_summary[col] = display_summary[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
    if "total_intensity" in display_summary.columns:
        display_summary["total_intensity"] = display_summary["total_intensity"].apply(
            lambda x: f"{x:.2e}" if pd.notna(x) else "N/A"
        )
    summary_table_html = display_summary.to_html(index=False, border=0, classes="data-table")

    # --- Flags table HTML (YES/no for readability) ---
    display_flags = flags_df.copy()
    for col in ["flag_low_detection", "flag_extreme_median",
                "flag_pca_outlier", "flag_low_correlation"]:
        display_flags[col] = display_flags[col].apply(lambda x: "YES" if x else "no")
    flags_table_html = display_flags.to_html(index=False, border=0, classes="data-table")

    # --- Flag alert section ---
    # Note on the baseline: flag_low_correlation is a global argmin with no
    # magnitude threshold, so exactly one sample is flagged on every
    # non-empty run. An n_flags == 1 result is therefore the expected "clean"
    # state, not evidence of a problem. See 02_qc_eda.Rmd Section 4.6.
    max_flags = int(flags_df["n_flags"].max()) if len(flags_df) > 0 else 0
    n_flagged_2plus = int((flags_df["n_flags"] >= 2).sum())

    if max_flags <= 1:
        alert_class = "alert-ok"
        alert_text = (
            "All samples have at most 1 flag. No sample is flagged on multiple "
            "independent criteria. This is consistent with normal replicate variation."
        )
    elif max_flags == 2:
        flagged = flags_df.loc[flags_df["n_flags"] >= 2, "sample_id"].tolist()
        alert_class = "alert-warn"
        alert_text = (
            f"Note: {', '.join(flagged)} flagged on 2 independent criteria. "
            "Worth reviewing, but 2 flags alone is not conclusive evidence of a quality problem."
        )
    else:
        # max_flags >= 3: Strong concern samples, and also list any 2-flag
        # samples in the same dataset (spec Section 4.6: 2 flags = "worth noting",
        # 3+ flags = "strong concern"; both are reported when both are present).
        flagged_3plus = flags_df.loc[flags_df["n_flags"] >= 3, "sample_id"].tolist()
        flagged_2 = flags_df.loc[flags_df["n_flags"] == 2, "sample_id"].tolist()
        alert_class = "alert-concern"
        parts = [
            f"Strong concern: {', '.join(flagged_3plus)} flagged on {max_flags} "
            "independent criteria. Recommend investigating and considering "
            "re-running the pipeline without these samples to assess impact "
            "on downstream results."
        ]
        if flagged_2:
            parts.append(
                f"Also worth reviewing: {', '.join(flagged_2)} flagged on 2 criteria."
            )
        alert_text = " ".join(parts)

    # --- Plot HTML divs (embed plotly.js once in the first plot) ---
    plot_order = [
        ("sample_intensity_summary", "Per-Sample Intensity Summary"),
        ("intensity_boxplots",       "Intensity Distributions (Box Plots)"),
        ("density_plot",             "Intensity Density (Per-Sample KDE)"),
        ("qq_plot",                  "Distributional Shape (Q-Q Plot)"),
        ("pca_scatter",              "Principal Component Analysis"),
        ("correlation_heatmap",      "Sample-to-Sample Correlation"),
    ]

    plot_sections_html = ""
    first_plot = True
    for plot_key, section_title in plot_order:
        if plot_key not in plots:
            continue
        div_html = plots[plot_key].to_html(
            full_html=False,
            include_plotlyjs=first_plot,  # True on first call, embeds ~3.5MB plotly.js
            div_id=f"plot_{plot_key}",
        )
        first_plot = False
        plot_sections_html += f"""
        <div class="plot-section">
            <h2>{section_title}</h2>
            {div_html}
        </div>
"""

    # --- Overview stats ---
    n_samples = len(summary_df)
    max_detected = int(summary_df["n_detected"].max()) if len(summary_df) > 0 else 0

    # --- Assemble full HTML document ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ProSIFT Pre-Normalization QC: {run_id}</title>
    <style>
        body {{
            font-family: Arial, Helvetica, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px 44px 60px 44px;
            color: #333;
            background: #fafafa;
            line-height: 1.5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 6px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 5px;
            margin-top: 44px;
        }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
        .scope-note {{
            background: #eaf2ff;
            border-left: 4px solid #3498db;
            padding: 10px 18px;
            border-radius: 4px;
            font-size: 0.91em;
            margin-bottom: 28px;
        }}
        .overview-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 30px;
        }}
        .stat-box {{
            background: #ecf0f1;
            padding: 16px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-box .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .stat-box .label {{ font-size: 0.82em; color: #7f8c8d; margin-top: 4px; }}
        .alert-ok      {{ background: #d5f5e3; border-left: 4px solid #27ae60; padding: 12px 18px; border-radius: 4px; margin: 14px 0; }}
        .alert-warn    {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 12px 18px; border-radius: 4px; margin: 14px 0; }}
        .alert-concern {{ background: #fdedec; border-left: 4px solid #e74c3c; padding: 12px 18px; border-radius: 4px; margin: 14px 0; }}
        table.data-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 14px 0;
            font-size: 0.91em;
        }}
        table.data-table th, table.data-table td {{
            border: 1px solid #d5d8dc;
            padding: 7px 13px;
            text-align: left;
        }}
        table.data-table th {{ background: #eaecee; font-weight: bold; }}
        table.data-table tr:nth-child(even) {{ background: #f8f9fa; }}
        .note {{ color: #7f8c8d; font-style: italic; font-size: 0.88em; margin: 8px 0 0 0; }}
        .plot-section {{ margin: 28px 0 50px 0; }}
    </style>
</head>
<body>

<h1>ProSIFT Pre-Normalization QC Report</h1>
<div class="meta">
    <strong>Run:</strong> {run_id} &nbsp;|&nbsp; <strong>Generated:</strong> {report_date}
</div>

<div class="scope-note">
    <strong>Scope:</strong> Pre-normalization diagnostics only. This report characterizes the
    post-detection-filter data before any normalization or imputation. Post-normalization
    diagnostics are produced by Module 03. Missingness analysis is in the Module 01 validation
    report. CV analysis is in Module 04. A consolidated QC Report (Phase 2) will compile all
    diagnostic outputs side-by-side.
</div>

<h2>Overview</h2>
<div class="overview-grid">
    <div class="stat-box">
        <div class="value">{n_samples}</div>
        <div class="label">Samples</div>
    </div>
    <div class="stat-box">
        <div class="value">{max_detected:,}</div>
        <div class="label">Max proteins detected</div>
    </div>
    <div class="stat-box">
        <div class="value">{n_flagged_2plus}</div>
        <div class="label">Samples with 2+ flags</div>
    </div>
</div>

<h2>Per-Sample Summary Statistics</h2>
{summary_table_html}
<p class="note">
    All intensity metrics computed on log2 working copy (log2(x+1) for raw input; as-is for
    log2/normalized input). total_intensity is summed on raw linear scale and reported only for
    raw input; N/A otherwise.
    <br><strong>Note:</strong> <code>n_detected</code> counts non-missing values in the
    <em>post-filter</em> matrix ({n_total_proteins:,} proteins after detection filtering).
    For pre-filter detection counts per sample, see the Module 01 missingness report.
</p>

<h2>Sample Outlier Flags</h2>
<div class="{alert_class}">{alert_text}</div>
{flags_table_html}
<p class="note">
    Flags are advisory -- no data is excluded by this module. Interpretation:
    0-1 flags: normal variation; 2 flags: worth reviewing; 3-4 flags: strong concern.
    See Module 02 spec Section 4.6 for flag definitions and threshold rationale.
</p>

{plot_sections_html}

</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    logging.info(f"  Report: {report_path.name}")
    return report_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    setup_logging()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id

    logging.info(f"Module 02: Pre-normalization QC/EDA -- {run_id}")

    # --- Load params ---
    params = load_params(Path(args.params))
    abundance_type = params["input"]["abundance_type"]

    # --- Load input files ---
    logging.info("Loading input files...")
    matrix_df  = pd.read_parquet(args.matrix)
    metadata_df = pd.read_parquet(args.metadata)

    # --- Prepare abundance data ---
    logging.info("Preparing abundance data...")
    raw_df, log2_df, sample_ids = prepare_abundance(matrix_df, params)
    n_total_proteins = len(raw_df)
    logging.info(f"  {n_total_proteins} proteins, {len(sample_ids)} samples")

    # --- Build group and color maps ---
    group_map = build_group_map(metadata_df, params)
    missing_samples = [s for s in sample_ids if s not in group_map]
    if missing_samples:
        raise ValueError(
            f"Samples in abundance matrix not found in metadata: {missing_samples}"
        )
    groups = [group_map[s] for s in sample_ids]
    color_map = make_color_map(groups)

    # --- Section 4.1: Per-sample intensity summaries ---
    logging.info("Computing per-sample summaries...")
    summary_df = compute_sample_summaries(raw_df, log2_df, group_map, sample_ids, abundance_type)
    summary_df.to_parquet(outdir / f"{run_id}.sample_summary.parquet", index=False)
    summary_df.to_csv(outdir / f"{run_id}.sample_summary.csv", index=False)
    logging.info(f"  Saved: {run_id}.sample_summary.parquet/.csv")

    summary_fig = plot_sample_intensity_summary(summary_df, color_map, run_id, abundance_type)
    save_plot(summary_fig, outdir / f"{run_id}.sample_intensity_summary")

    # --- Section 4.2: Intensity distribution box plots ---
    logging.info("Generating intensity box plots...")
    box_fig = plot_intensity_boxplots(log2_df, summary_df, color_map, run_id)
    save_plot(box_fig, outdir / f"{run_id}.intensity_boxplots")

    # --- Section 4.2b: Per-sample density plots ---
    logging.info("Generating density plots...")
    density_fig = plot_density(log2_df, summary_df, color_map, run_id)
    save_plot(density_fig, outdir / f"{run_id}.density_plot")

    # --- Section 4.3: Q-Q plot ---
    logging.info("Generating Q-Q plot...")
    qq_fig = plot_qq(log2_df, summary_df, color_map, run_id)
    save_plot(qq_fig, outdir / f"{run_id}.qq_plot")

    # --- Section 4.4: PCA ---
    logging.info("Computing PCA...")
    pca_df, pca_fig = compute_and_plot_pca(
        log2_df, summary_df, color_map, run_id,
        note=(
            "Pre-normalization PCA may show technical variation "
            "(loading differences) dominating biological signal. Expected before normalization."
        ),
    )
    pca_df.to_parquet(outdir / f"{run_id}.pca_results.parquet", index=False)
    logging.info(f"  Saved: {run_id}.pca_results.parquet")
    save_plot(pca_fig, outdir / f"{run_id}.pca_scatter")

    # --- Section 4.5: Sample-to-sample correlation ---
    logging.info("Computing sample correlation matrix...")
    corr_df, corr_fig = compute_and_plot_correlation(log2_df, summary_df, color_map, run_id)
    corr_df.to_parquet(outdir / f"{run_id}.correlation_matrix.parquet")
    logging.info(f"  Saved: {run_id}.correlation_matrix.parquet")
    save_plot(corr_fig, outdir / f"{run_id}.correlation_heatmap")

    # --- Section 4.6: Outlier detection and sample flagging ---
    logging.info("Computing sample outlier flags...")
    flags_df = compute_sample_flags(summary_df, pca_df, corr_df, n_total_proteins)
    flags_df.to_parquet(outdir / f"{run_id}.sample_flags.parquet", index=False)
    flags_df.to_csv(outdir / f"{run_id}.sample_flags.csv", index=False)
    logging.info(f"  Saved: {run_id}.sample_flags.parquet/.csv")

    n_flagged = int((flags_df["n_flags"] >= 2).sum())
    if n_flagged > 0:
        flagged_names = flags_df.loc[flags_df["n_flags"] >= 2, "sample_id"].tolist()
        logging.warning(f"  {n_flagged} sample(s) with 2+ flags: {flagged_names}")
    else:
        logging.info("  No samples flagged on 2+ criteria.")

    # --- Consolidated HTML report ---
    logging.info("Generating QC report...")
    plots = {
        "sample_intensity_summary": summary_fig,
        "intensity_boxplots":       box_fig,
        "density_plot":             density_fig,
        "qq_plot":                  qq_fig,
        "pca_scatter":              pca_fig,
        "correlation_heatmap":      corr_fig,
    }
    generate_html_report(run_id, summary_df, flags_df, plots, outdir, n_total_proteins)

    logging.info(
        f"Module 02 complete. "
        f"{n_total_proteins} proteins, {len(sample_ids)} samples, "
        f"{n_flagged} sample(s) with 2+ flags."
    )


if __name__ == "__main__":
    main()
