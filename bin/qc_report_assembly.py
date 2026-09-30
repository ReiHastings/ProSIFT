#!/usr/bin/env python3
"""
Title:         qc_report_assembly.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-04-07
Last Modified: 2026-04-07
Purpose:       Module 03b QC Report Assembly. Compiles diagnostic outputs from
               Modules 01-03 into a single consolidated HTML QC report. Re-renders
               all plots from upstream data files using shared plotting utilities.
               Generates one new visualization (CV density plot). Serves as the
               formal checkpoint between data preparation and statistical analysis.
Inputs:
  --metadata         {run_id}.validated_metadata.parquet  (Module 01)
  --filter-table     {run_id}.detection_filter_table.csv  (Module 01)
  --filtered-matrix  {run_id}.filtered_matrix.parquet     (Module 01/02)
  --sample-summary   {run_id}.sample_summary.parquet      (Module 02)
  --sample-flags     {run_id}.sample_flags.parquet        (Module 02)
  --norm-matrix      {run_id}.normalized_matrix.parquet   (Module 03 NORMALIZE)
  --cv-summary       {run_id}.cv_summary.parquet          (Module 03 NORMALIZE)
  --imputed-matrix   {run_id}.imputed_matrix.parquet      (Module 03 IMPUTE)
  --imp-mask         {run_id}.imputation_mask.parquet     (Module 03 IMPUTE)
  --imp-summary      {run_id}.imputation_summary.parquet  (Module 03 IMPUTE)
  --params           {run_id}_params.yml
Outputs:
  {run_id}.qc_report.html
Usage:
  qc_report_assembly.py \
      --metadata CTXcyto_WT_vs_CTXcyto_KO.validated_metadata.parquet \
      --filter-table CTXcyto_WT_vs_CTXcyto_KO.detection_filter_table.csv \
      --filtered-matrix CTXcyto_WT_vs_CTXcyto_KO.filtered_matrix.parquet \
      --sample-summary CTXcyto_WT_vs_CTXcyto_KO.sample_summary.parquet \
      --sample-flags CTXcyto_WT_vs_CTXcyto_KO.sample_flags.parquet \
      --norm-matrix CTXcyto_WT_vs_CTXcyto_KO.normalized_matrix.parquet \
      --cv-summary CTXcyto_WT_vs_CTXcyto_KO.cv_summary.parquet \
      --imputed-matrix CTXcyto_WT_vs_CTXcyto_KO.imputed_matrix.parquet \
      --imp-mask CTXcyto_WT_vs_CTXcyto_KO.imputation_mask.parquet \
      --imp-summary CTXcyto_WT_vs_CTXcyto_KO.imputation_summary.parquet \
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
import plotly.io as pio
import scipy.stats
import yaml
from prosift_plot_utils import (
    FILTER_CATEGORY_COLORS,
    FILTER_CATEGORY_ORDER,
    compute_and_plot_correlation,
    compute_and_plot_pca,
    make_color_map,
    plot_density,
    plot_intensity_boxplots,
)

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='qc_report_assembly.py',
        description='ProSIFT Module 03b: QC Report Assembly',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  qc_report_assembly.py \\\n'
            '    --metadata run.validated_metadata.parquet \\\n'
            '    --filter-table run.detection_filter_table.csv \\\n'
            '    --filtered-matrix run.filtered_matrix.parquet \\\n'
            '    --sample-summary run.sample_summary.parquet \\\n'
            '    --sample-flags run.sample_flags.parquet \\\n'
            '    --norm-matrix run.normalized_matrix.parquet \\\n'
            '    --cv-summary run.cv_summary.parquet \\\n'
            '    --imputed-matrix run.imputed_matrix.parquet \\\n'
            '    --imp-mask run.imputation_mask.parquet \\\n'
            '    --imp-summary run.imputation_summary.parquet \\\n'
            '    --params run_params.yml \\\n'
            '    --run-id run \\\n'
            '    --outdir .'
        ),
    )
    parser.add_argument('--metadata', required=True,
                        help='Validated metadata (Parquet)')
    parser.add_argument('--filter-table', required=True, dest='filter_table',
                        help='Detection filter table (CSV)')
    parser.add_argument('--filtered-matrix', required=True, dest='filtered_matrix',
                        help='Filtered abundance matrix (Parquet)')
    parser.add_argument('--sample-summary', required=True, dest='sample_summary',
                        help='Per-sample summary table (Parquet)')
    parser.add_argument('--sample-flags', required=True, dest='sample_flags',
                        help='Per-sample outlier flags (Parquet)')
    parser.add_argument('--norm-matrix', required=True, dest='norm_matrix',
                        help='Normalized abundance matrix (Parquet)')
    parser.add_argument('--cv-summary', required=True, dest='cv_summary',
                        help='Per-protein CV summary (Parquet)')
    parser.add_argument('--imputed-matrix', required=True, dest='imputed_matrix',
                        help='Imputed abundance matrix (Parquet)')
    parser.add_argument('--imp-mask', required=True, dest='imp_mask',
                        help='Imputation mask (Parquet)')
    parser.add_argument('--imp-summary', required=True, dest='imp_summary',
                        help='Imputation summary table (Parquet)')
    parser.add_argument('--params', required=True,
                        help='Run params.yml')
    parser.add_argument('--run-id', required=True, dest='run_id',
                        help='Run identifier')
    parser.add_argument('--outdir', required=True,
                        help='Output directory')
    # Optional: paths to upstream reports for linking
    parser.add_argument('--missingness-report', dest='missingness_report',
                        default=None,
                        help='Path to missingness report HTML (for linking)')
    parser.add_argument('--prenorm-report', dest='prenorm_report',
                        default=None,
                        help='Path to pre-norm QC report HTML (for linking)')
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_params(params_path: Path) -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def extract_abundance(
    matrix_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract log2 abundance values from a matrix DataFrame.

    Returns a DataFrame with protein_id as index, bare sample_id columns,
    and values on the log2 scale.
    """
    abundance_prefix = params['input'].get('abundance_prefix', '')
    peptide_prefix = params['input'].get('peptide_count_prefix', 'peptide_count_')
    abundance_type = params['input'].get('abundance_type', 'raw')

    all_cols = matrix_df.columns.tolist()
    abund_cols = [
        c for c in all_cols
        if c != 'protein_id' and not c.startswith(peptide_prefix)
    ]

    sample_ids = (
        [c[len(abundance_prefix):] for c in abund_cols]
        if abundance_prefix
        else list(abund_cols)
    )

    df = matrix_df.set_index('protein_id')[abund_cols].copy()
    df.columns = sample_ids

    # Convert to log2 scale if raw. Non-positive values (<= 0) are not valid raw
    # intensities (VALIDATE_INPUTS removes them upstream); mask them to NaN here
    # as a defensive backstop so log2 does not emit -Inf (for 0) or a NaN with a
    # RuntimeWarning (for negatives).
    if abundance_type == 'raw':
        df = np.log2(df.mask(df <= 0))

    return df, sample_ids


# ============================================================
# SECTION 1: RUN OVERVIEW
# ============================================================

def build_run_overview(
    run_id: str,
    metadata_df: pd.DataFrame,
    filter_df: pd.DataFrame,
    sample_flags_df: pd.DataFrame,
    cv_summary_df: pd.DataFrame,
    imp_summary_df: pd.DataFrame,
    params: dict,
) -> str:
    """Build HTML for Section 1: Run Overview."""
    group_col = params['design']['group_column']
    groups = metadata_df[group_col].value_counts().sort_index()

    # Detection filter counts
    category_order = FILTER_CATEGORY_ORDER
    filter_counts = filter_df['filter_status'].value_counts()
    total_input = len(filter_df)
    retained_cats = ['PASSED', 'PARTIAL', 'SINGLE-GROUP']
    total_retained = sum(filter_counts.get(c, 0) for c in retained_cats)

    # Flag summary
    flag_counts = sample_flags_df['n_flags'].value_counts().sort_index()
    n_no_flags = int(flag_counts.get(0, 0))
    n_mild = sum(int(flag_counts.get(i, 0)) for i in [1, 2])
    n_severe = sum(int(flag_counts.get(i, 0)) for i in [3, 4])
    severe_samples = sample_flags_df[sample_flags_df['n_flags'] >= 3]['sample_id'].tolist()

    # Median CV per group
    group_names = sorted(groups.index)
    cv_medians = {}
    for g in group_names:
        cv_col = f'cv_{g}'
        if cv_col in cv_summary_df.columns:
            vals = cv_summary_df[cv_col].dropna()
            cv_medians[g] = float(vals.median()) if len(vals) > 0 else float('nan')

    # Imputation counts
    n_mnar_vals = int(imp_summary_df['n_mnar_imputed'].sum())
    n_mar_vals = int(imp_summary_df['n_mar_imputed'].sum())
    n_total_imp = int(imp_summary_df['n_imputed_total'].sum())
    n_proteins_imputed = int((imp_summary_df['n_imputed_total'] > 0).sum())

    # Method info (read from params, not from the upstream summary text files)
    norm_method = params.get('normalization', {}).get('method', 'unknown')
    imp_mode = params.get('imputation', {}).get('mode', 'unknown')
    imp_mnar = params.get('imputation', {}).get('mnar_method', 'minprob')
    imp_mar = params.get('imputation', {}).get('mar_method', 'knn')
    detection_threshold = params.get('qc', {}).get('min_detections_per_group', 'N/A')

    # Build HTML
    rows = []
    rows.append(f'<div class="overview-field"><b>Run ID:</b> {run_id}</div>')

    # Group info
    group_items = ', '.join(f'{g}: {int(n)} samples' for g, n in groups.items())
    rows.append(f'<div class="overview-field"><b>Groups:</b> {group_items}</div>')
    rows.append(
        f'<div class="overview-field"><b>Detection threshold:</b> '
        f'{detection_threshold} detections per group</div>'
    )

    # Filter category counts
    rows.append('<div class="overview-field"><b>Protein counts by filter category:</b></div>')
    rows.append('<ul class="overview-list">')
    for cat in category_order:
        n = int(filter_counts.get(cat, 0))
        pct = 100 * n / total_input if total_input > 0 else 0
        rows.append(f'  <li>{cat}: {n:,} ({pct:.1f}%)</li>')
    rows.append(f'  <li><b>Retained (PASSED + PARTIAL + SINGLE-GROUP): '
                f'{total_retained:,} / {total_input:,}</b></li>')
    rows.append('</ul>')

    # Normalization
    rows.append(
        f'<div class="overview-field"><b>Normalization:</b> {norm_method}</div>'
    )

    # Imputation
    if imp_mode == 'mixed':
        imp_str = f'mixed (MNAR: {imp_mnar}, MAR: {imp_mar})'
    else:
        imp_str = f'single ({params.get("imputation", {}).get("single_method", "unknown")})'
    rows.append(
        f'<div class="overview-field"><b>Imputation:</b> {imp_str}</div>'
    )
    rows.append(
        f'<div class="overview-field"><b>Imputation counts:</b> '
        f'{n_mnar_vals:,} MNAR + {n_mar_vals:,} MAR = {n_total_imp:,} total '
        f'({n_proteins_imputed:,} proteins affected)</div>'
    )

    # Flag summary
    flag_str = f'0 flags: {n_no_flags}, 1-2 flags: {n_mild}, 3-4 flags: {n_severe}'
    rows.append(
        f'<div class="overview-field"><b>Sample outlier flags:</b> {flag_str}</div>'
    )
    if severe_samples:
        rows.append(
            f'<div class="overview-field overview-warning">'
            f'Samples with 3+ flags: {", ".join(severe_samples)}</div>'
        )

    # Median CV
    cv_items = ', '.join(
        f'{g}: {cv_medians[g]:.3f}' if not np.isnan(cv_medians.get(g, float('nan')))
        else f'{g}: N/A'
        for g in group_names
    )
    rows.append(
        f'<div class="overview-field"><b>Median within-group CV:</b> {cv_items}</div>'
    )

    return '\n'.join(rows)


# ============================================================
# SECTION 2: MISSINGNESS AND DETECTION
# ============================================================

def plot_filter_categories(filter_df: pd.DataFrame, run_id: str) -> go.Figure:
    """Re-render the filter category bar chart from detection_filter_table.csv."""
    category_order = FILTER_CATEGORY_ORDER
    category_colors = FILTER_CATEGORY_COLORS

    counts = filter_df['filter_status'].value_counts()
    total = len(filter_df)

    categories = []
    values = []
    colors = []
    annotations = []

    for cat in category_order:
        n = int(counts.get(cat, 0))
        categories.append(cat)
        values.append(n)
        colors.append(category_colors.get(cat, '#999999'))
        pct = 100 * n / total if total > 0 else 0
        annotations.append(f'{n:,} ({pct:.1f}%)')

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=annotations,
            textposition='outside',
            hovertemplate='%{x}: %{y:,}<extra></extra>',
        )
    )
    fig.update_layout(
        title=f'Detection Filter Categories: {run_id}',
        xaxis_title='Filter Category',
        yaxis_title='Number of Proteins',
        height=420,
        width=650,
        showlegend=False,
    )
    return fig


# ============================================================
# SECTION 3: DISTRIBUTIONAL ASSESSMENT (PAIRED)
# ============================================================

def plot_qq(
    log2_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    run_id: str,
) -> go.Figure:
    """
    Overlaid Q-Q plot for all samples. Compares each sample's intensity
    distribution against a theoretical normal distribution.
    """
    sample_order = (
        summary_df.sort_values(['group', 'sample_id'])['sample_id'].tolist()
    )

    fig = go.Figure()
    shown_groups: set[str] = set()

    for sid in sample_order:
        grp = str(summary_df.loc[summary_df['sample_id'] == sid, 'group'].iloc[0])
        show_legend = grp not in shown_groups
        shown_groups.add(grp)

        values = np.sort(log2_df[sid].dropna().values)
        n = len(values)
        if n < 2:
            continue

        # Theoretical quantiles from standard normal
        theoretical = scipy.stats.norm.ppf(
            (np.arange(1, n + 1) - 0.5) / n
        )

        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=values,
                mode='markers',
                marker=dict(color=color_map[grp], size=3, opacity=0.6),
                name=grp,
                legendgroup=grp,
                legendgrouptitle_text=grp if show_legend else None,
                showlegend=show_legend,
                hovertemplate=(
                    f'<b>{sid}</b><br>'
                    'Theoretical: %{x:.2f}<br>'
                    'Observed: %{y:.2f}<extra></extra>'
                ),
            )
        )

    # Add diagonal reference line
    all_vals = log2_df.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) > 0:
        q_lo, q_hi = np.quantile(all_vals, [0.01, 0.99])
        fig.add_trace(
            go.Scatter(
                x=[scipy.stats.norm.ppf(0.01), scipy.stats.norm.ppf(0.99)],
                y=[q_lo, q_hi],
                mode='lines',
                line=dict(color='black', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip',
            )
        )

    fig.update_layout(
        title=f'Q-Q Plot (Pre-Normalization): {run_id}',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Observed log2 Intensity',
        height=480,
        width=600,
        legend_title_text='Group',
    )
    return fig


# ============================================================
# SECTION 5: IMPUTATION DIAGNOSTICS
# ============================================================

def plot_imputation_overlay(
    imputed_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    run_id: str,
) -> go.Figure:
    """
    Overlaid density curves: observed, MNAR-imputed, MAR-imputed.
    Re-rendered from imputed matrix and imputation mask.
    """
    # Ensure both DataFrames have protein_id as index
    mask_vals = mask_df.set_index('protein_id') if 'protein_id' in mask_df.columns else mask_df
    if 'protein_id' in imputed_df.columns:
        imp_vals = imputed_df.set_index('protein_id')
    else:
        imp_vals = imputed_df

    sample_cols = [c for c in mask_vals.columns]
    imp_vals = imp_vals[sample_cols]

    observed = imp_vals.values[mask_vals.values == 'observed'].flatten()
    mnar = imp_vals.values[mask_vals.values == 'mnar'].flatten()
    mar = imp_vals.values[mask_vals.values == 'mar'].flatten()

    fig = go.Figure()

    def _add_kde(values, label, color, dash='solid'):
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return
        kde = scipy.stats.gaussian_kde(values, bw_method='scott')
        x_range = np.linspace(values.min(), values.max(), 400)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde(x_range),
            mode='lines', name=label,
            line=dict(color=color, dash=dash, width=2),
        ))

    _add_kde(observed, 'Observed', '#1f77b4')
    _add_kde(mnar, 'MNAR-imputed (MinProb)', '#d62728', 'dash')
    _add_kde(mar, 'MAR-imputed (KNN)', '#2ca02c', 'dot')

    fig.update_layout(
        title=f'Observed vs. Imputed Distributions: {run_id}',
        xaxis_title='log2 Abundance',
        yaxis_title='Density',
        height=460,
        width=650,
        legend_title_text='Value type',
    )
    return fig


def plot_imputation_fractions(
    mask_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_map: dict[str, str],
    group_col: str,
    run_id: str,
) -> go.Figure:
    """Per-sample imputation fraction bar chart from imputation mask."""
    mask_vals = mask_df.set_index('protein_id') if 'protein_id' in mask_df.columns else mask_df
    sample_cols = list(mask_vals.columns)

    group_of = dict(zip(
        metadata_df['sample_id'].astype(str),
        metadata_df[group_col].astype(str),
        strict=True,
    ))
    sample_order = sorted(sample_cols, key=lambda s: (group_of.get(s, ''), s))

    n_proteins = len(mask_vals)
    shown_groups: set[str] = set()

    fig = go.Figure()
    for sid in sample_order:
        grp = group_of.get(sid, 'unknown')
        show_legend = grp not in shown_groups
        shown_groups.add(grp)

        n_imputed = int((mask_vals[sid] != 'observed').sum())
        frac = n_imputed / n_proteins if n_proteins > 0 else 0.0

        fig.add_trace(go.Bar(
            x=[sid], y=[frac],
            name=grp,
            marker_color=color_map[grp],
            legendgroup=grp,
            showlegend=show_legend,
            hovertemplate=(
                f'<b>{sid}</b><br>'
                f'Group: {grp}<br>'
                f'Imputation fraction: %{{y:.3f}}<extra></extra>'
            ),
        ))

    fig.update_layout(
        title=f'Per-Sample Imputation Fraction: {run_id}',
        xaxis_title='Sample',
        yaxis_title='Fraction imputed',
        yaxis=dict(range=[0, 1]),
        height=420,
        width=650,
        barmode='group',
    )
    return fig


# ============================================================
# SECTION 6: REPLICATE VARIABILITY
# ============================================================

def plot_cv_density(
    cv_summary_df: pd.DataFrame,
    group_names: list[str],
    color_map: dict[str, str],
    run_id: str,
) -> tuple[go.Figure, dict]:
    """
    Per-group CV density plot from cv_summary.parquet.

    Returns the figure and a dict of per-group summary statistics for the
    annotation table below the plot.
    """
    fig = go.Figure()
    stats = {}

    for g in group_names:
        cv_col = f'cv_{g}'
        if cv_col not in cv_summary_df.columns:
            continue

        values = cv_summary_df[cv_col].dropna().values
        n_excluded = int(cv_summary_df[cv_col].isna().sum())

        if len(values) < 2:
            stats[g] = {
                'median': float('nan'),
                'pct_below_20': float('nan'),
                'pct_below_50': float('nan'),
                'n_excluded': n_excluded,
                'n_used': len(values),
            }
            continue

        median_cv = float(np.median(values))
        pct_below_20 = 100 * float((values < 0.20).sum()) / len(values)
        pct_below_50 = 100 * float((values < 0.50).sum()) / len(values)

        stats[g] = {
            'median': median_cv,
            'pct_below_20': pct_below_20,
            'pct_below_50': pct_below_50,
            'n_excluded': n_excluded,
            'n_used': len(values),
        }

        # KDE density curve
        kde = scipy.stats.gaussian_kde(values, bw_method='scott')
        x_max = min(float(np.percentile(values, 99.5)), 2.0)
        x_grid = np.linspace(0, x_max, 512)
        density = kde(x_grid)

        fig.add_trace(go.Scatter(
            x=x_grid, y=density,
            mode='lines',
            line=dict(color=color_map[g], width=2),
            name=g,
            hovertemplate=(
                f'<b>{g}</b><br>'
                'CV: %{x:.3f}<br>'
                'Density: %{y:.3f}<extra></extra>'
            ),
        ))

        # Median vertical line
        fig.add_vline(
            x=median_cv,
            line=dict(color=color_map[g], dash='dash', width=1.5),
            annotation_text=f'{g} median: {median_cv:.3f}',
            annotation_position='top',
        )

    fig.update_layout(
        title=f'Within-Group CV Distribution: {run_id}',
        xaxis_title='Coefficient of Variation (CV)',
        yaxis_title='Density',
        height=460,
        width=700,
        legend_title_text='Group',
    )
    return fig, stats


# ============================================================
# HTML REPORT ASSEMBLY
# ============================================================

def fig_to_div(fig: go.Figure) -> str:
    """Convert a Plotly figure to an HTML div fragment (no plotly.js)."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def build_table_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame as a styled HTML table."""
    truncated = len(df) > max_rows
    display_df = df.head(max_rows) if truncated else df

    rows = ['<table class="data-table">']
    # Header
    rows.append('<thead><tr>')
    for col in display_df.columns:
        rows.append(f'<th>{col}</th>')
    rows.append('</tr></thead>')
    # Body
    rows.append('<tbody>')
    for _, row in display_df.iterrows():
        rows.append('<tr>')
        for val in row:
            if isinstance(val, float):
                rows.append(f'<td>{val:.4f}</td>')
            else:
                rows.append(f'<td>{val}</td>')
        rows.append('</tr>')
    rows.append('</tbody>')
    rows.append('</table>')

    if truncated:
        rows.append(
            f'<p class="table-note">Showing first {max_rows} of '
            f'{len(df)} rows. See linked files for full data.</p>'
        )
    return '\n'.join(rows)


def make_link(path_str: str | None, label: str) -> str:
    """Generate an HTML link, or 'not available' if the path is None or missing."""
    if path_str is None:
        return f'<span class="link-unavailable">{label} (not available)</span>'
    p = Path(path_str)
    if p.exists():
        return f'<a href="{p.name}">{label}</a>'
    return f'<span class="link-unavailable">{label} (not available)</span>'


# ============================================================
# CSS STYLES
# ============================================================

REPORT_CSS = '''
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 Helvetica, Arial, sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px 40px;
    color: #333;
    line-height: 1.5;
}
h1 {
    border-bottom: 3px solid #1f77b4;
    padding-bottom: 10px;
    color: #1f77b4;
}
h2 {
    border-bottom: 1px solid #ccc;
    padding-bottom: 6px;
    margin-top: 40px;
    color: #444;
}
h3 { color: #555; margin-top: 24px; }

/* --- Section 1: Run Overview --- */
.overview-field { margin: 4px 0; font-size: 0.95em; }
.overview-list { margin: 4px 0 8px 20px; }
.overview-warning { color: #d62728; font-weight: bold; }

/* --- Paired layout for Sections 3 and 4 --- */
.paired-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 16px 0;
}
.paired-panel { min-width: 0; }
.paired-panel h4 {
    text-align: center;
    margin: 0 0 4px 0;
    color: #666;
    font-size: 0.9em;
}

/* --- Tables --- */
.data-table {
    border-collapse: collapse;
    font-size: 0.85em;
    margin: 12px 0;
    width: auto;
}
.data-table th, .data-table td {
    border: 1px solid #ddd;
    padding: 5px 10px;
    text-align: right;
}
.data-table th {
    background-color: #f4f4f4;
    font-weight: 600;
    text-align: center;
}
.data-table td:first-child { text-align: left; }
.table-note { font-size: 0.82em; color: #888; font-style: italic; }

/* --- CV summary table --- */
.cv-summary-table { margin: 12px 0; }

/* --- Links --- */
.supplemental-links {
    margin: 12px 0;
    padding: 8px 12px;
    background: #f8f9fa;
    border-left: 3px solid #1f77b4;
    font-size: 0.9em;
}
.supplemental-links a { color: #1f77b4; }
.link-unavailable { color: #999; font-style: italic; }

/* --- Interpretation notes --- */
.interpretation-note {
    margin: 8px 0;
    padding: 8px 12px;
    background: #fffbe6;
    border-left: 3px solid #ff7f0e;
    font-size: 0.88em;
    color: #555;
}

/* --- Footer --- */
footer {
    margin-top: 40px;
    padding-top: 12px;
    border-top: 1px solid #ddd;
    font-size: 0.82em;
    color: #999;
}
'''


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    setup_logging()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id

    logging.info(f'Module 03b QC Report Assembly -- {run_id}')

    # --- Load all inputs ---
    logging.info('Loading input files...')
    params = load_params(Path(args.params))
    group_col = params['design']['group_column']

    metadata_df = pd.read_parquet(args.metadata)
    filter_df = pd.read_csv(args.filter_table)
    filtered_matrix_df = pd.read_parquet(args.filtered_matrix)
    sample_summary_df = pd.read_parquet(args.sample_summary)
    sample_flags_df = pd.read_parquet(args.sample_flags)
    norm_matrix_df = pd.read_parquet(args.norm_matrix)
    cv_summary_df = pd.read_parquet(args.cv_summary)
    imputed_matrix_df = pd.read_parquet(args.imputed_matrix)
    imp_mask_df = pd.read_parquet(args.imp_mask)
    imp_summary_df = pd.read_parquet(args.imp_summary)

    # --- Extract log2 abundance matrices ---
    pre_log2, sample_ids = extract_abundance(filtered_matrix_df, params)
    post_log2, _ = extract_abundance(norm_matrix_df, params)

    # Post-normalization data is already on log2 scale from NORMALIZE,
    # but extract_abundance applies log2 again if abundance_type is 'raw'.
    # NORMALIZE outputs are already log2, so we re-read without transform.
    post_log2_cols = [
        c for c in norm_matrix_df.columns
        if c != 'protein_id'
        and not c.startswith(params['input'].get('peptide_count_prefix', 'peptide_count_'))
    ]
    abundance_prefix = params['input'].get('abundance_prefix', '')
    post_sample_ids = (
        [c[len(abundance_prefix):] for c in post_log2_cols]
        if abundance_prefix
        else list(post_log2_cols)
    )
    post_log2 = norm_matrix_df.set_index('protein_id')[post_log2_cols].copy()
    post_log2.columns = post_sample_ids

    # Imputed matrix sample columns (same prefix stripping)
    imp_abund_cols = [
        c for c in imputed_matrix_df.columns
        if c != 'protein_id'
        and not c.startswith(params['input'].get('peptide_count_prefix', 'peptide_count_'))
    ]
    imp_sample_ids = (
        [c[len(abundance_prefix):] for c in imp_abund_cols]
        if abundance_prefix
        else list(imp_abund_cols)
    )
    imp_log2 = imputed_matrix_df.set_index('protein_id')[imp_abund_cols].copy()
    imp_log2.columns = imp_sample_ids

    # Group info
    groups = [
        str(metadata_df.loc[metadata_df['sample_id'] == s, group_col].iloc[0])
        for s in sample_ids
    ]
    color_map = make_color_map(groups)
    group_names = sorted(set(groups))

    # Summary df for plotting functions
    summary_df = pd.DataFrame({'sample_id': sample_ids, 'group': groups})

    logging.info(
        f'  {len(pre_log2)} proteins, {len(sample_ids)} samples, '
        f'{len(group_names)} groups'
    )

    # ================================================================
    # RENDER ALL PLOT SECTIONS
    # ================================================================
    plot_divs: dict[str, str] = {}

    # --- Section 1: Run Overview (text, no plots) ---
    logging.info('Building Section 1: Run Overview...')
    section1_html = build_run_overview(
        run_id, metadata_df, filter_df, sample_flags_df,
        cv_summary_df, imp_summary_df, params,
    )

    # --- Section 2: Missingness ---
    logging.info('Building Section 2: Missingness...')
    fig_filter = plot_filter_categories(filter_df, run_id)
    plot_divs['filter_categories'] = fig_to_div(fig_filter)

    # --- Section 3: Distributional Assessment (Paired) ---
    logging.info('Building Section 3: Distributional Assessment...')

    # 3A: Box plots (pre vs post)
    fig_box_pre = plot_intensity_boxplots(
        pre_log2, summary_df, color_map, run_id, label='Pre-Normalization',
    )
    fig_box_post = plot_intensity_boxplots(
        post_log2, summary_df, color_map, run_id, label='Post-Normalization',
    )
    # Match y-axis scales
    all_pre_vals = pre_log2.values.flatten()
    all_post_vals = post_log2.values.flatten()
    all_vals = np.concatenate([
        all_pre_vals[~np.isnan(all_pre_vals)],
        all_post_vals[~np.isnan(all_post_vals)],
    ])
    y_min, y_max = float(np.percentile(all_vals, 0.5)), float(np.percentile(all_vals, 99.5))
    y_margin = (y_max - y_min) * 0.05
    shared_yrange = [y_min - y_margin, y_max + y_margin]
    fig_box_pre.update_layout(yaxis=dict(range=shared_yrange))
    fig_box_post.update_layout(yaxis=dict(range=shared_yrange))

    plot_divs['box_pre'] = fig_to_div(fig_box_pre)
    plot_divs['box_post'] = fig_to_div(fig_box_post)

    # 3B: Density plots (pre vs post)
    fig_den_pre = plot_density(
        pre_log2, summary_df, color_map, run_id, label='Pre-Normalization',
    )
    fig_den_post = plot_density(
        post_log2, summary_df, color_map, run_id, label='Post-Normalization',
    )
    # Match x-axis scales for density
    fig_den_pre.update_layout(xaxis=dict(range=shared_yrange))
    fig_den_post.update_layout(xaxis=dict(range=shared_yrange))

    plot_divs['den_pre'] = fig_to_div(fig_den_pre)
    plot_divs['den_post'] = fig_to_div(fig_den_post)

    # Q-Q plot (pre-normalization only)
    fig_qq = plot_qq(pre_log2, summary_df, color_map, run_id)
    plot_divs['qq'] = fig_to_div(fig_qq)

    # Skewness/kurtosis table
    sk_cols = ['sample_id', 'group', 'skewness', 'kurtosis']
    sk_available = [c for c in sk_cols if c in sample_summary_df.columns]
    sk_table_html = build_table_html(sample_summary_df[sk_available])

    # --- Section 4: Sample Relationships (Paired) ---
    logging.info('Building Section 4: Sample Relationships...')

    # 4A: PCA (pre vs post)
    _, fig_pca_pre = compute_and_plot_pca(
        pre_log2, summary_df, color_map, run_id, label='Pre-Normalization',
    )
    _, fig_pca_post = compute_and_plot_pca(
        post_log2, summary_df, color_map, run_id, label='Post-Normalization',
    )
    plot_divs['pca_pre'] = fig_to_div(fig_pca_pre)
    plot_divs['pca_post'] = fig_to_div(fig_pca_post)

    # 4B: Clustermap (pre vs post)
    _, fig_clust_pre = compute_and_plot_correlation(
        pre_log2, summary_df, color_map, run_id, label='Pre-Normalization',
    )
    _, fig_clust_post = compute_and_plot_correlation(
        post_log2, summary_df, color_map, run_id, label='Post-Normalization',
    )
    plot_divs['clust_pre'] = fig_to_div(fig_clust_pre)
    plot_divs['clust_post'] = fig_to_div(fig_clust_post)

    # Flag table (sorted by flag count descending)
    flags_display = sample_flags_df.sort_values('n_flags', ascending=False)
    flags_table_html = build_table_html(flags_display)

    # Per-sample summary table
    summary_table_html = build_table_html(sample_summary_df)

    # --- Section 5: Imputation Diagnostics ---
    logging.info('Building Section 5: Imputation Diagnostics...')
    fig_imp_overlay = plot_imputation_overlay(imp_log2, imp_mask_df, run_id)
    plot_divs['imp_overlay'] = fig_to_div(fig_imp_overlay)

    fig_imp_frac = plot_imputation_fractions(
        imp_mask_df, metadata_df, color_map, group_col, run_id,
    )
    plot_divs['imp_frac'] = fig_to_div(fig_imp_frac)

    # --- Section 6: Replicate Variability ---
    logging.info('Building Section 6: Replicate Variability...')
    fig_cv, cv_stats = plot_cv_density(cv_summary_df, group_names, color_map, run_id)
    plot_divs['cv_density'] = fig_to_div(fig_cv)

    # CV summary annotation table
    cv_table_rows = ['<table class="data-table cv-summary-table">']
    cv_table_rows.append(
        '<thead><tr><th>Group</th><th>Median CV</th>'
        '<th>% proteins CV &lt; 20%</th><th>% proteins CV &lt; 50%</th>'
        '<th>Proteins used</th><th>Proteins excluded (NaN CV)</th></tr></thead>'
    )
    cv_table_rows.append('<tbody>')
    for g in group_names:
        s = cv_stats.get(g, {})
        median_str = f'{s.get("median", float("nan")):.3f}' if not np.isnan(s.get('median', float('nan'))) else 'N/A'
        pct20 = f'{s.get("pct_below_20", float("nan")):.1f}%' if not np.isnan(s.get('pct_below_20', float('nan'))) else 'N/A'
        pct50 = f'{s.get("pct_below_50", float("nan")):.1f}%' if not np.isnan(s.get('pct_below_50', float('nan'))) else 'N/A'
        n_used = s.get('n_used', 0)
        n_excl = s.get('n_excluded', 0)
        cv_table_rows.append(
            f'<tr><td>{g}</td><td>{median_str}</td><td>{pct20}</td>'
            f'<td>{pct50}</td><td>{n_used:,}</td><td>{n_excl:,}</td></tr>'
        )
    cv_table_rows.append('</tbody></table>')
    cv_table_html = '\n'.join(cv_table_rows)

    # ================================================================
    # COMPOSE FINAL HTML
    # ================================================================
    logging.info('Composing HTML report...')
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Supplemental links
    missingness_link = make_link(args.missingness_report, 'View full missingness report')
    prenorm_link = make_link(args.prenorm_report, 'View full pre-normalization QC report')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ProSIFT QC Report: {run_id}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
{REPORT_CSS}
  </style>
</head>
<body>

<h1>ProSIFT QC Report: {run_id}</h1>

<!-- ============================== -->
<!-- Section 1: Run Overview        -->
<!-- ============================== -->
<h2>1. Run Overview</h2>
{section1_html}

<!-- ============================== -->
<!-- Section 2: Missingness         -->
<!-- ============================== -->
<h2>2. Missingness and Detection Overview</h2>
{plot_divs['filter_categories']}
<div class="supplemental-links">
  {missingness_link}
</div>

<!-- ============================== -->
<!-- Section 3: Distributional      -->
<!-- ============================== -->
<h2>3. Distributional Assessment</h2>

<h3>Box Plots: Pre- vs. Post-Normalization</h3>
<div class="paired-container">
  <div class="paired-panel">
    <h4>Pre-Normalization</h4>
    {plot_divs['box_pre']}
  </div>
  <div class="paired-panel">
    <h4>Post-Normalization</h4>
    {plot_divs['box_post']}
  </div>
</div>

<h3>Density Plots: Pre- vs. Post-Normalization</h3>
<div class="paired-container">
  <div class="paired-panel">
    <h4>Pre-Normalization</h4>
    {plot_divs['den_pre']}
  </div>
  <div class="paired-panel">
    <h4>Post-Normalization</h4>
    {plot_divs['den_post']}
  </div>
</div>

<h3>Q-Q Plot (Pre-Normalization)</h3>
{plot_divs['qq']}

<h3>Skewness and Kurtosis</h3>
{sk_table_html}

<div class="supplemental-links">
  {prenorm_link}
</div>

<!-- ============================== -->
<!-- Section 4: Sample Relationships -->
<!-- ============================== -->
<h2>4. Sample Relationships</h2>

<h3>PCA: Pre- vs. Post-Normalization</h3>
<div class="paired-container">
  <div class="paired-panel">
    <h4>Pre-Normalization</h4>
    {plot_divs['pca_pre']}
  </div>
  <div class="paired-panel">
    <h4>Post-Normalization</h4>
    {plot_divs['pca_post']}
  </div>
</div>

<h3>Correlation Clustermap: Pre- vs. Post-Normalization</h3>
<div class="paired-container">
  <div class="paired-panel">
    <h4>Pre-Normalization</h4>
    {plot_divs['clust_pre']}
  </div>
  <div class="paired-panel">
    <h4>Post-Normalization</h4>
    {plot_divs['clust_post']}
  </div>
</div>

<h3>Sample Outlier Flags</h3>
{flags_table_html}

<div class="interpretation-note">
  <b>Flag interpretation:</b> Flags are advisory indicators of potential issues,
  not exclusion criteria. 0 flags = no concern. 1-2 flags = investigate.
  3-4 convergent flags = strong concern, consider whether the sample should be
  included in downstream analysis.
</div>

<h3>Per-Sample Summary</h3>
{summary_table_html}

<!-- ============================== -->
<!-- Section 5: Imputation          -->
<!-- ============================== -->
<h2>5. Imputation Diagnostics</h2>

<h3>Observed vs. Imputed Distributions</h3>
{plot_divs['imp_overlay']}

<h3>Per-Sample Imputation Fraction</h3>
{plot_divs['imp_frac']}

<div class="supplemental-links">
  Per-protein imputation summary table available in upstream module outputs
  (<code>imputation_summary.parquet</code>).
</div>

<!-- ============================== -->
<!-- Section 6: Replicate Variability -->
<!-- ============================== -->
<h2>6. Replicate Variability</h2>

<h3>Within-Group CV Distribution</h3>
{plot_divs['cv_density']}

{cv_table_html}

<div class="supplemental-links">
  Full per-protein CV data available in upstream module outputs
  (<code>cv_summary.csv</code>).
</div>

<footer>
  Generated by ProSIFT Module 03b (QC Report Assembly)<br>
  Date: {now}
</footer>

</body>
</html>'''

    # --- Write output ---
    report_path = outdir / f'{run_id}.qc_report.html'
    report_path.write_text(html, encoding='utf-8')
    report_size_kb = report_path.stat().st_size / 1024
    logging.info(f'  Saved: {report_path.name} ({report_size_kb:.0f} KB)')
    logging.info('Module 03b QC Report Assembly complete.')


if __name__ == '__main__':
    main()
