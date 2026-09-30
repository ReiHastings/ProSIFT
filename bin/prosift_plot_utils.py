#!/usr/bin/env python3
"""
Title:         prosift_plot_utils.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-03-27
Last Modified: 2026-04-24 (docstring sync; dead import removed)
Purpose:       Shared Plotly plotting utilities used by Module 02 (prenorm_qc.py)
               and Module 03 (normalize.py). Provides color palette helpers, a
               save_plot dispatcher (PNG + HTML), and the four diagnostic plots
               that appear in both pre- and post-normalization QC: intensity box
               plots, per-sample density plots, PCA scatter, and
               sample-to-sample correlation heatmap.
Inputs:        (library module -- imported, not invoked directly)
Outputs:       (library module -- no output files)
Usage:
  from prosift_plot_utils import (
      make_color_map, save_plot,
      plot_intensity_boxplots, compute_and_plot_pca, compute_and_plot_correlation,
  )
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import average, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# COLOR PALETTE
# ============================================================

# D3 qualitative palette -- colorblind-friendly, 10 colors.
_PALETTE = px.colors.qualitative.D3


def make_color_map(groups: list[str]) -> dict[str, str]:
    """Assign a consistent D3 color to each unique group label."""
    unique_groups = sorted(set(groups))
    return {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(unique_groups)}


# Detection-filter category display order and colors. Canonical, shared by every
# report that renders filter categories (Module 01 missingness_report.py, Module
# 03b qc_report_assembly.py) so the same category is always the same color and no
# category is silently dropped. Retained: PASSED, PARTIAL, SINGLE-GROUP.
# Removed: WEAK-ANCHOR, SPARSE, ABSENT.
FILTER_CATEGORY_ORDER = [
    'PASSED', 'PARTIAL', 'SINGLE-GROUP', 'WEAK-ANCHOR', 'SPARSE', 'ABSENT',
]

FILTER_CATEGORY_COLORS = {
    'PASSED':       '#2ca02c',   # green  -- retained, quantitative
    'PARTIAL':      '#ff7f0e',   # orange -- retained, flagged (sub-threshold group)
    'SINGLE-GROUP': '#1f77b4',   # blue   -- retained, presence/absence
    'WEAK-ANCHOR':  '#8c564b',   # brown  -- removed, imputation-heavy (no full anchor)
    'SPARSE':       '#9467bd',   # purple -- removed, sub-threshold in all groups
    'ABSENT':       '#d62728',   # red    -- removed, undetected in this run
}


# ============================================================
# PLOT I/O
# ============================================================

def save_plot(fig: go.Figure, base_path: Path) -> None:
    """
    Save a Plotly figure as both a static PNG (via kaleido) and a
    standalone interactive HTML file with embedded plotly.js.

    base_path must NOT have an extension -- the extension is appended as a
    string rather than using Path.with_suffix, because Path.with_suffix
    replaces only the last dot-separated token in the stem (e.g.,
    "run.intensity_boxplots" would lose "intensity_boxplots").
    """
    png_path  = Path(str(base_path) + ".png")
    html_path = Path(str(base_path) + ".html")

    fig.write_image(str(png_path))
    fig.write_html(str(html_path), full_html=True, include_plotlyjs=True)
    logging.info(f"  Saved: {png_path.name}, {html_path.name}")


# ============================================================
# SHARED DIAGNOSTIC PLOTS
# ============================================================

def plot_intensity_boxplots(
    log2_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    run_id: str,
    label: str = "Pre-Normalization",
) -> go.Figure:
    """
    Box plot of log2 intensity distributions. One box per sample, samples
    ordered by group. Boxes show median, IQR, 1.5xIQR whiskers, and outlier
    points. Used by Module 02 (pre-normalization) and Module 03 (post-normalization).

    Parameters
    ----------
    log2_df : pd.DataFrame
        Log2 abundance matrix; index = protein_id, columns = sample_ids.
    summary_df : pd.DataFrame
        Per-sample summary with at minimum 'sample_id' and 'group' columns.
    color_map : dict[str, str]
        {group_label: hex_color} from make_color_map().
    run_id : str
        Run identifier used in the plot title.
    label : str
        Phase label for the plot title, e.g. "Pre-Normalization" or
        "Post-Normalization".
    """
    sample_order = (
        summary_df.sort_values(["group", "sample_id"])["sample_id"].tolist()
    )

    fig = go.Figure()
    shown_groups: set[str] = set()
    for sid in sample_order:
        grp = str(summary_df.loc[summary_df["sample_id"] == sid, "group"].iloc[0])
        show_legend = grp not in shown_groups
        shown_groups.add(grp)
        values = log2_df[sid].dropna().values
        fig.add_trace(
            go.Box(
                y=values,
                name=sid,
                marker_color=color_map[grp],
                legendgroup=grp,
                legendgrouptitle_text=grp if show_legend else None,
                showlegend=show_legend,
                boxpoints="outliers",
                hoverinfo="y+name",
            )
        )

    fig.update_layout(
        title=f"Intensity Distributions ({label}): {run_id}",
        xaxis_title="Sample",
        yaxis_title="log2 Intensity",
        height=520,
        width=800,
        showlegend=True,
    )
    return fig


def plot_density(
    log2_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    run_id: str,
    label: str = 'Pre-Normalization',
) -> go.Figure:
    """
    Overlaid kernel density estimate (KDE) of log2 intensity distributions.
    One line per sample, colored by group. Complements box plots by showing
    full distributional shape (bimodality, skewness, shoulder effects) that
    summary statistics cannot capture.

    Parameters
    ----------
    log2_df : pd.DataFrame
        Log2 abundance matrix; index = protein_id, columns = sample_ids.
    summary_df : pd.DataFrame
        Per-sample summary with 'sample_id' and 'group' columns.
    color_map : dict[str, str]
        {group_label: hex_color} from make_color_map().
    run_id : str
        Run identifier for plot title.
    label : str
        Phase label for the plot title, e.g. 'Pre-Normalization' or
        'Post-Normalization'.
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

        values = log2_df[sid].dropna().values
        if len(values) < 2:
            continue
        # Guard against zero-variance samples (constant values). gaussian_kde
        # would raise LinAlgError on a singular covariance matrix. This is
        # vanishingly unlikely with post-filter proteomics data but the guard
        # is one line and makes the failure mode loud rather than confusing.
        if values.std(ddof=1) == 0:
            logging.warning(
                f'plot_density: skipping {sid} (zero variance, cannot fit KDE)'
            )
            continue

        # Step 1: Compute KDE using scipy gaussian_kde (Scott's rule bandwidth)
        kde = scipy.stats.gaussian_kde(values)

        # Step 2: Evaluate over a regular grid spanning the data range
        x_min, x_max = float(values.min()), float(values.max())
        margin = (x_max - x_min) * 0.05
        x_grid = np.linspace(x_min - margin, x_max + margin, 512)
        density = kde(x_grid)

        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=density,
                mode='lines',
                line=dict(color=color_map[grp], width=1.5),
                name=grp,
                legendgroup=grp,
                legendgrouptitle_text=grp if show_legend else None,
                showlegend=show_legend,
                hovertemplate=(
                    f'<b>{sid}</b><br>'
                    'log2 intensity: %{x:.2f}<br>'
                    'Density: %{y:.4f}<extra></extra>'
                ),
            )
        )

    fig.update_layout(
        title=f'Intensity Density ({label}): {run_id}',
        xaxis_title='log2 Intensity',
        yaxis_title='Density',
        height=520,
        width=800,
        legend_title_text='Group',
    )
    return fig


def compute_and_plot_pca(
    log2_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    run_id: str,
    label: str = "Pre-Normalization",
    note: str | None = None,
) -> tuple[pd.DataFrame, go.Figure]:
    """
    PCA on complete-case log2 data (proteins with any NaN dropped).
    Centers and scales each protein (correlation-matrix PCA) so all proteins
    contribute equally regardless of absolute abundance.

    Parameters
    ----------
    log2_df : pd.DataFrame
        Log2 abundance matrix; index = protein_id, columns = sample_ids.
    summary_df : pd.DataFrame
        Per-sample summary with at minimum 'sample_id' and 'group' columns.
    color_map : dict[str, str]
        {group_label: hex_color} from make_color_map().
    run_id : str
        Run identifier used in the plot title.
    label : str
        Phase label for the plot title, e.g. "Pre-Normalization".
    note : str or None
        Optional italic annotation text displayed below the plot axis. Pass
        None (default) to omit the annotation.

    Returns
    -------
    pca_df : pd.DataFrame
        Per-sample PCA coordinates (PC1-PC3) and variance explained.
    fig : go.Figure
        PC1 vs PC2 scatter plot.
    """
    sample_ids = log2_df.columns.tolist()
    group_of = {
        row["sample_id"]: str(row["group"])
        for _, row in summary_df.iterrows()
    }

    # --- Drop proteins with any missing value (complete cases only) ---
    complete_df = log2_df.dropna(axis=0, how="any")
    n_proteins_used = len(complete_df)
    n_proteins_total = len(log2_df)
    n_dropped = n_proteins_total - n_proteins_used
    logging.info(
        f"  PCA: using {n_proteins_used} complete-case proteins "
        f"({n_dropped} dropped due to missing values)"
    )

    if n_proteins_used < 2:
        raise ValueError(
            f"Only {n_proteins_used} complete-case proteins available for PCA. "
            "Too many missing values to compute PCA."
        )

    # --- Center and scale: rows=samples, columns=proteins ---
    X = complete_df.T.values  # shape: (n_samples, n_proteins)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Fit PCA ---
    n_components = min(len(sample_ids), n_proteins_used)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)  # shape: (n_samples, n_components)

    var_exp = pca.explained_variance_ratio_
    pc1_var = float(var_exp[0]) if len(var_exp) > 0 else np.nan
    pc2_var = float(var_exp[1]) if len(var_exp) > 1 else np.nan
    pc3_var = float(var_exp[2]) if len(var_exp) > 2 else np.nan

    groups = [group_of[s] for s in sample_ids]

    pca_df = pd.DataFrame({
        "sample_id":              sample_ids,
        "group":                  groups,
        "PC1":                    coords[:, 0],
        "PC2":                    coords[:, 1] if coords.shape[1] > 1 else np.nan,
        "PC3":                    coords[:, 2] if coords.shape[1] > 2 else np.nan,
        "variance_explained_PC1": pc1_var,
        "variance_explained_PC2": pc2_var,
        "variance_explained_PC3": pc3_var,
        "n_proteins_used":        n_proteins_used,
    })

    # --- PC1 vs PC2 scatter plot ---
    pc1_label = f"PC1 ({pc1_var * 100:.1f}%)"
    pc2_label = f"PC2 ({pc2_var * 100:.1f}%)"

    fig = go.Figure()
    shown_groups: set[str] = set()
    for grp, grp_df in pca_df.groupby("group"):
        grp = str(grp)
        show_legend = grp not in shown_groups
        shown_groups.add(grp)
        fig.add_trace(
            go.Scatter(
                x=grp_df["PC1"],
                y=grp_df["PC2"],
                mode="markers+text",
                text=grp_df["sample_id"],
                textposition="top center",
                textfont=dict(size=11),
                marker=dict(color=color_map[grp], size=14,
                            line=dict(width=1, color="white")),
                name=grp,
                legendgroup=grp,
                showlegend=show_legend,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<extra></extra>"
                ),
            )
        )

    layout_kwargs: dict = dict(
        title=f"PCA ({label}): {run_id}",
        xaxis_title=pc1_label,
        yaxis_title=pc2_label,
        height=580,
        width=700,
        legend_title_text="Group",
    )
    if note is not None:
        layout_kwargs["annotations"] = [
            dict(
                text=f"<i>{note}</i>",
                xref="paper", yref="paper",
                x=0.0, y=-0.18,
                showarrow=False,
                font=dict(size=11, color="gray"),
                align="left",
            )
        ]

    fig.update_layout(**layout_kwargs)
    return pca_df, fig


def compute_and_plot_correlation(
    log2_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    run_id: str,
    label: str = 'Pre-Normalization',
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Pearson correlation clustermap with dendrogram arms.

    Pairwise-complete Pearson correlation, hierarchically clustered using
    distance = 1 - r and average linkage. The dendrogram determines sample
    ordering (data-driven rather than fixed group order). Color scale anchored
    to [floor(min_off_diag, 0.05), 1.0].

    Parameters
    ----------
    log2_df : pd.DataFrame
        Log2 abundance matrix; index = protein_id, columns = sample_ids.
    summary_df : pd.DataFrame
        Per-sample summary with at minimum 'sample_id' and 'group' columns.
    color_map : dict[str, str]
        {group_label: hex_color} from make_color_map().
    run_id : str
        Run identifier used in the plot title.
    label : str
        Phase label for the plot title, e.g. 'Pre-Normalization'.

    Returns
    -------
    corr_df : pd.DataFrame
        Square Pearson correlation matrix (sample_id index and columns,
        in original order for downstream use).
    fig : go.Figure
        Clustermap with top dendrogram and annotated heatmap.
    """
    sample_ids = log2_df.columns.tolist()

    # Pairwise-complete Pearson correlation (pandas default)
    corr_df = log2_df.corr(method='pearson')

    # --- Hierarchical clustering: distance = 1 - r, average linkage ---
    # Clip correlations to [0, 1] for distance computation (negative
    # correlations are theoretically possible but extremely unlikely in
    # proteomics replicates; clipping prevents negative distances).
    corr_values = corr_df.values.copy()
    np.fill_diagonal(corr_values, 1.0)
    dist_matrix = 1.0 - np.clip(corr_values, 0.0, 1.0)
    # Ensure exact symmetry and zero diagonal for squareform
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix)
    linkage_matrix = average(condensed)

    # Get dendrogram leaf order (data-driven sample ordering)
    dendro_result = dendrogram(linkage_matrix, no_plot=True)
    leaf_order = dendro_result['leaves']
    ordered_samples = [sample_ids[i] for i in leaf_order]

    # Reorder correlation matrix by dendrogram leaves
    corr_ordered = corr_df.loc[ordered_samples, ordered_samples]

    # --- Color scale: anchor to observed range ---
    n = len(ordered_samples)
    mask = ~np.eye(n, dtype=bool)
    min_corr = float(corr_ordered.values[mask].min())
    zmin = max(0.80, np.floor(min_corr * 20) / 20)

    # --- Build clustermap figure with dendrogram on top ---
    group_of = dict(zip(summary_df['sample_id'], summary_df['group'].astype(str), strict=True))
    axis_labels = [f'{group_of[s]}: {s}' for s in ordered_samples]

    # Subplot layout: dendrogram on top (row 1), heatmap below (row 2)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.15, 0.85],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    # Step 1: Draw dendrogram arms
    icoord = dendro_result['icoord']  # x coordinates of dendrogram links
    dcoord = dendro_result['dcoord']  # y coordinates (heights)
    for xs, ys in zip(icoord, dcoord, strict=True):
        # scipy dendrogram uses 5, 15, 25... as leaf positions (step of 10);
        # remap to 0-based sample indices for alignment with heatmap
        mapped_xs = [(x - 5) / 10 for x in xs]
        fig.add_trace(
            go.Scatter(
                x=mapped_xs,
                y=ys,
                mode='lines',
                line=dict(color='#333', width=1.2),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=1, col=1,
        )

    # Step 2: Draw heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr_ordered.values,
            x=list(range(n)),
            y=list(range(n)),
            colorscale='RdBu',
            zmin=zmin,
            zmax=1.0,
            colorbar=dict(title='Pearson r', y=0.4, len=0.7),
            hovertemplate='%{customdata[0]}<br>%{customdata[1]}<br>r = %{z:.4f}<extra></extra>',
            customdata=[
                [
                    [axis_labels[i], axis_labels[j]]
                    for j in range(n)
                ]
                for i in range(n)
            ],
        ),
        row=2, col=1,
    )

    # Step 3: Cell annotations with 3dp
    annotations = []
    for i in range(n):
        for j in range(n):
            val = corr_ordered.values[i, j]
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f'{val:.3f}',
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    xref='x2', yref='y2',
                )
            )

    fig.update_layout(
        title=f'Sample Correlation Clustermap ({label}): {run_id}',
        annotations=annotations,
        height=620,
        width=680,
        # Dendrogram axis (top)
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, n - 0.5],
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        # Heatmap axes
        xaxis2=dict(
            tickvals=list(range(n)),
            ticktext=axis_labels,
            tickangle=40,
            side='bottom',
        ),
        yaxis2=dict(
            tickvals=list(range(n)),
            ticktext=axis_labels,
            autorange='reversed',
        ),
    )

    # Return corr_df in original sample order (not clustered) so downstream
    # consumers (e.g., outlier flagging) are not affected by reordering.
    return corr_df, fig
