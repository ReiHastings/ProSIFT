#!/usr/bin/env python3
"""
Title:         missingness_report.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-03-27
Last Modified: 2026-04-23 (added: --params arg; group column sourced from design.group_column)
Purpose:       Module 01, Process 4.10: Missingness visualization report.
               Reads the detection filter table (from FILTER_PROTEINS),
               validated abundance matrix, and validated metadata (both from
               VALIDATE_INPUTS) to produce four interactive Plotly plots
               characterizing missingness patterns in the pre-filter dataset.
               Advisory only -- no data are modified.
Inputs:
  --filter-table  {run_id}.detection_filter_table.csv  (FILTER_PROTEINS)
  --matrix        {run_id}.validated_matrix.parquet    (VALIDATE_INPUTS)
  --metadata      {run_id}.validated_metadata.parquet  (VALIDATE_INPUTS)
  --params        per-run params.yml (used to resolve design.group_column)
  --run-id        run identifier string
  --outdir        output directory (default: current directory)
Outputs:
  {run_id}.missingness_report.html           standalone HTML with all four plots
  {run_id}.missingness_plots/                directory of static PNGs
    {run_id}.filter_categories.png
    {run_id}.per_sample_detection.png
    {run_id}.missingness_heatmap.png
    {run_id}.missingness_histogram.png
Usage:
  missingness_report.py \\
      --filter-table CTXcyto_WT_vs_CTXcyto_KO.detection_filter_table.csv \\
      --matrix       CTXcyto_WT_vs_CTXcyto_KO.validated_matrix.parquet \\
      --metadata     CTXcyto_WT_vs_CTXcyto_KO.validated_metadata.parquet \\
      --params       CTXcyto_WT_vs_CTXcyto_KO_params.yml \\
      --run-id       CTXcyto_WT_vs_CTXcyto_KO \\
      --outdir       .
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from prosift_plot_utils import FILTER_CATEGORY_COLORS, FILTER_CATEGORY_ORDER

# Color palette (D3, colorblind-friendly)
_PALETTE = px.colors.qualitative.D3

# Detection filter category display order and colors. Sourced from the canonical
# shared definitions in prosift_plot_utils so this Module 01 report and the
# Module 03b QC report always use the same category-to-color mapping.
_CATEGORY_ORDER = FILTER_CATEGORY_ORDER
_CATEGORY_COLORS = FILTER_CATEGORY_COLORS


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="missingness_report.py",
        description="ProSIFT Module 01 Process 4.10: Missingness visualization report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  missingness_report.py \\\n"
            "      --filter-table run.detection_filter_table.csv \\\n"
            "      --matrix run.validated_matrix.parquet \\\n"
            "      --metadata run.validated_metadata.parquet \\\n"
            "      --params run_params.yml \\\n"
            "      --run-id run --outdir ."
        ),
    )
    parser.add_argument("--filter-table", required=True,
                        help="Detection filter table CSV (from FILTER_PROTEINS)")
    parser.add_argument("--matrix", required=True,
                        help="Validated abundance matrix Parquet (from VALIDATE_INPUTS)")
    parser.add_argument("--metadata", required=True,
                        help="Validated metadata Parquet (from VALIDATE_INPUTS)")
    parser.add_argument("--params", required=True,
                        help="Per-run params.yml (used to resolve design.group_column)")
    parser.add_argument("--run-id", required=True,
                        help="Run identifier string (used for output file names)")
    parser.add_argument("--outdir", default=".",
                        help="Output directory (default: current directory)")
    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_inputs(
    filter_table_path: str,
    matrix_path: str,
    metadata_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three input files and return as DataFrames.

    Returns:
        filter_df:   One row per protein; columns include protein_id and filter_status.
        matrix_df:   Abundance matrix with protein_id index; abundance columns.
        metadata_df: Sample metadata with sample_id and group columns.
    """
    logging.info("Loading detection filter table: %s", filter_table_path)
    filter_df = pd.read_csv(filter_table_path)
    _require_columns(filter_df, ["protein_id", "filter_status"], "filter table")

    logging.info("Loading validated matrix: %s", matrix_path)
    matrix_df = pd.read_parquet(matrix_path)
    if "protein_id" not in matrix_df.columns:
        raise ValueError("validated_matrix.parquet is missing 'protein_id' column")

    logging.info("Loading validated metadata: %s", metadata_path)
    metadata_df = pd.read_parquet(metadata_path)
    _require_columns(metadata_df, ["sample_id"], "metadata")

    return filter_df, matrix_df, metadata_df


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


# ============================================================
# PLOT 1: Filter category bar chart
# ============================================================

def plot_filter_categories(filter_df: pd.DataFrame, run_id: str) -> go.Figure:
    """Bar chart: protein count per detection filter category."""
    counts = filter_df["filter_status"].value_counts().reindex(
        _CATEGORY_ORDER, fill_value=0
    ).reset_index()
    counts.columns = ["category", "count"]
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(1)
    counts["label"] = counts.apply(
        lambda r: f"{r['count']:,} ({r['pct']}%)", axis=1
    )

    colors = [_CATEGORY_COLORS.get(c, "#7f7f7f") for c in counts["category"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["category"],
        y=counts["count"],
        marker_color=colors,
        text=counts["label"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:,} proteins<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{run_id} - Detection filter categories", x=0.5),
        xaxis_title="Filter category",
        yaxis_title="Number of proteins",
        yaxis=dict(range=[0, counts["count"].max() * 1.15]),
        showlegend=False,
        template="simple_white",
        margin=dict(t=60, b=60),
    )
    return fig


# ============================================================
# PLOT 2: Per-sample detection bar chart
# ============================================================

def plot_per_sample_detection(
    matrix_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    group_col: str,
    run_id: str,
    n_total_proteins: int | None = None,
) -> go.Figure:
    """Bar chart: number of proteins detected per sample, colored by group.

    When n_total_proteins is provided, the total input protein count is shown
    in the plot subtitle as a denominator for interpreting detection counts.
    """
    # Identify abundance columns in the matrix (may have an abundance_ prefix)
    abund_cols = [c for c in matrix_df.columns if c != "protein_id"
                  and not c.startswith("peptide_count_")]
    # Count non-NaN per sample; strip abundance_ prefix to match metadata sample_ids
    detection = (
        matrix_df[abund_cols].notna().sum().rename("n_detected").reset_index()
    )
    detection.columns = ["col_name", "n_detected"]
    detection["sample_id"] = detection["col_name"].str.removeprefix("abundance_")

    # Attach group labels using the explicitly-passed group column (not
    # inferred from column position, which would misbehave when covariates
    # or a batch column are present in metadata).
    detection = detection.merge(
        metadata_df[["sample_id", group_col]], on="sample_id", how="left"
    )
    detection = detection.rename(columns={group_col: "group"})

    # Assign colors by group
    groups = detection["group"].unique().tolist()
    color_map = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(groups)}
    detection["color"] = detection["group"].map(color_map)

    fig = go.Figure()
    for group in groups:
        sub = detection[detection["group"] == group]
        fig.add_trace(go.Bar(
            x=sub["sample_id"],
            y=sub["n_detected"],
            name=group,
            marker_color=color_map[group],
            hovertemplate="<b>%{x}</b><br>Detected: %{y:,}<extra></extra>",
        ))

    # Build title with denominator subtitle when total protein count is known
    title_text = f"{run_id} - Per-sample detection counts (pre-filter)"
    if n_total_proteins is not None:
        title_text += f"<br><span style='font-size:12px;color:#666'>out of {n_total_proteins:,} total input proteins</span>"

    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        xaxis_title="Sample",
        yaxis_title="Proteins detected",
        legend_title="Group",
        template="simple_white",
        barmode="group",
        margin=dict(t=80, b=80),
        xaxis=dict(tickangle=-30),
    )
    return fig


# ============================================================
# PLOT 3: Missingness heatmap (filtered-out proteins only)
# ============================================================

def plot_missingness_heatmap(
    filter_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    group_col: str,
    run_id: str,
) -> go.Figure:
    """Binary heatmap of filtered-out proteins x samples.

    Rows sorted by filter category then total detection count.
    Columns grouped by condition.
    """
    # Subset to filtered-out proteins
    removed_categories = ["PARTIAL", "SINGLE-GROUP", "WEAK-ANCHOR", "SPARSE", "ABSENT"]
    removed_ids = filter_df.loc[
        filter_df["filter_status"].isin(removed_categories), "protein_id"
    ].tolist()

    if len(removed_ids) == 0:
        logging.warning("No filtered-out proteins found; skipping heatmap.")
        fig = go.Figure()
        fig.update_layout(title=dict(text=f"{run_id} - Missingness heatmap (no filtered proteins)", x=0.5))
        return fig

    # Abundance columns ordered by group.
    # Matrix columns may carry an abundance_ prefix; build a mapping from
    # bare sample_id -> actual column name so we can look them up correctly.
    all_abund_cols = {
        c.removeprefix("abundance_"): c
        for c in matrix_df.columns
        if c != "protein_id" and not c.startswith("peptide_count_")
    }
    sample_order_bare = metadata_df.sort_values(group_col)["sample_id"].tolist()
    abund_cols = [all_abund_cols[s] for s in sample_order_bare if s in all_abund_cols]
    display_labels = [s for s in sample_order_bare if s in all_abund_cols]

    # Build binary presence/absence matrix for filtered proteins
    subset = matrix_df[matrix_df["protein_id"].isin(removed_ids)].copy()
    subset = subset.set_index("protein_id")[abund_cols]
    # Rename columns to bare sample IDs for display
    subset.columns = display_labels
    detected = subset.notna().astype(int)  # 1=detected, 0=missing

    # Add sort keys from filter table.
    # Always reindex by protein_id so the detection count aligns by key rather
    # than by row position; equal lengths do not imply matching row order.
    cat_order_map = {c: i for i, c in enumerate(removed_categories)}
    filter_sub = filter_df[filter_df["protein_id"].isin(removed_ids)].copy()
    filter_sub["_cat_rank"] = filter_sub["filter_status"].map(cat_order_map)
    filter_sub["_n_detected"] = (
        detected.reindex(filter_sub["protein_id"]).sum(axis=1).values
    )
    filter_sub = filter_sub.sort_values(["_cat_rank", "_n_detected"],
                                        ascending=[True, False])
    sorted_proteins = filter_sub["protein_id"].tolist()
    detected = detected.reindex(sorted_proteins)

    # Build y-axis labels: prepend category prefix every N proteins for visual separation
    y_labels = detected.index.tolist()

    # Category boundary annotations
    shapes = []
    cat_midpoints = []
    current_cat = None
    cat_start = 0
    for i, pid in enumerate(sorted_proteins):
        row_cat = filter_df.loc[filter_df["protein_id"] == pid, "filter_status"].values[0]
        if row_cat != current_cat:
            if current_cat is not None:
                # Draw horizontal line at boundary
                shapes.append(dict(
                    type="line",
                    x0=-0.5, x1=len(display_labels) - 0.5,
                    y0=i - 0.5, y1=i - 0.5,
                    line=dict(color="black", width=1.5),
                ))
                cat_midpoints.append((current_cat, (cat_start + i - 1) / 2))
            current_cat = row_cat
            cat_start = i
    if current_cat is not None:
        cat_midpoints.append((current_cat, (cat_start + len(sorted_proteins) - 1) / 2))

    colorscale = [[0, "#f8f8f8"], [1, "#1f77b4"]]

    fig = go.Figure(data=go.Heatmap(
        z=detected.values,
        x=display_labels,
        y=y_labels,
        colorscale=colorscale,
        showscale=False,
        hovertemplate="Sample: %{x}<br>Protein: %{y}<br>%{z}<extra></extra>",
        zmin=0, zmax=1,
    ))

    fig.update_layout(
        title=dict(
            text=f"{run_id} - Missingness heatmap (filtered-out proteins: {len(sorted_proteins)})",
            x=0.5,
        ),
        xaxis=dict(title="Sample", side="bottom", tickangle=-30),
        yaxis=dict(
            title="Protein (filtered out)",
            showticklabels=len(sorted_proteins) <= 200,
            autorange="reversed",
        ),
        shapes=shapes,
        template="simple_white",
        margin=dict(t=80, b=80, l=120),
        height=max(400, min(1200, len(sorted_proteins) * 4 + 150)),
    )

    # Annotations for category labels on left side
    annotations = []
    for cat_name, y_mid in cat_midpoints:
        annotations.append(dict(
            x=-0.12,
            y=y_mid,
            xref="paper",
            yref="y",
            text=f"<b>{cat_name}</b>",
            showarrow=False,
            font=dict(size=10, color=_CATEGORY_COLORS.get(cat_name, "#333")),
            align="right",
        ))
    fig.update_layout(annotations=annotations)

    return fig


# ============================================================
# PLOT 4: Per-protein missingness rate histogram
# ============================================================

def plot_missingness_histogram(
    matrix_df: pd.DataFrame,
    run_id: str,
) -> go.Figure:
    """Histogram of missingness fraction across all proteins (pre-filter)."""
    abund_cols = [c for c in matrix_df.columns if c != "protein_id"
                  and not c.startswith("peptide_count_")]
    n_samples = len(abund_cols)
    if n_samples == 0:
        raise ValueError("No abundance columns found in validated matrix")

    miss_frac = matrix_df[abund_cols].isna().sum(axis=1) / n_samples
    # Use discrete bins at exact fractions (0/n, 1/n, ..., n/n)
    bins = [i / n_samples for i in range(n_samples + 2)]
    counts, edges = np.histogram(miss_frac.values, bins=bins)

    tick_labels = [f"{i}/{n_samples}" for i in range(n_samples + 1)]
    tick_vals = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

    # Color bars by severity
    bar_colors = []
    for i in range(n_samples + 1):
        frac = i / n_samples
        if frac == 0:
            bar_colors.append("#2ca02c")    # fully detected -- green
        elif frac <= 1 / n_samples:
            bar_colors.append("#98df8a")    # one missing -- light green
        elif frac < 0.5:
            bar_colors.append("#ffbb78")    # minority missing -- orange
        elif frac < 1.0:
            bar_colors.append("#ff7f0e")    # majority missing -- dark orange
        else:
            bar_colors.append("#d62728")    # entirely absent -- red

    fig = go.Figure(data=go.Bar(
        x=tick_vals,
        y=counts,
        marker_color=bar_colors,
        hovertemplate="Missing in %{x:.0%} of samples<br>Proteins: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"{run_id} - Per-protein missingness rate (pre-filter, n={len(matrix_df):,} proteins)",
            x=0.5,
        ),
        xaxis=dict(
            title="Fraction of samples missing",
            tickvals=tick_vals,
            ticktext=tick_labels,
            tickangle=-30,
        ),
        yaxis_title="Number of proteins",
        template="simple_white",
        showlegend=False,
        margin=dict(t=60, b=80),
    )
    return fig


# ============================================================
# OUTPUT: Save PNG and consolidated HTML report
# ============================================================

def save_png(fig: go.Figure, png_path: Path) -> None:
    """Write a static PNG using kaleido. CRITICAL: caller must pass a full .png path."""
    fig.write_image(str(png_path))


def generate_html_report(
    run_id: str,
    figs: list[tuple[str, go.Figure]],
    outdir: Path,
) -> Path:
    """Assemble all four plots into one self-contained HTML file.

    figs: list of (plot_title, figure) in display order.
    The first figure embeds plotly.js; subsequent figures reuse it.
    """
    html_path = outdir / (run_id + ".missingness_report.html")

    # Build per-figure HTML blocks
    plot_blocks = []
    for i, (title, fig) in enumerate(figs):
        include_js = (i == 0)
        div_html = fig.to_html(
            full_html=False,
            include_plotlyjs=include_js,
        )
        plot_blocks.append(
            f'<div class="plot-section"><h2>{title}</h2>{div_html}</div>'
        )

    plots_html = "\n".join(plot_blocks)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ProSIFT Missingness Report - {run_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1   {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 8px; }}
  h2   {{ color: #555; margin-top: 40px; }}
  .plot-section {{ margin-bottom: 40px; }}
  .meta {{ color: #777; font-size: 0.9em; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>ProSIFT Missingness Report</h1>
<p class="meta">Run: <strong>{run_id}</strong> &nbsp;|&nbsp; Module 01 Process 4.10 &nbsp;|&nbsp; Advisory report (pre-filter data)</p>
{plots_html}
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    logging.info("HTML report written: %s", html_path)
    return html_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    args = parse_args()
    run_id = args.run_id
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load params ---
    # Resolve the metadata group column explicitly from params.yml. Inferring
    # it as "first non-sample_id column" breaks when design.covariates or
    # design.batch_column are populated in metadata.
    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    group_col = params["design"]["group_column"]

    # --- Load inputs ---
    filter_df, matrix_df, metadata_df = load_inputs(
        args.filter_table, args.matrix, args.metadata
    )
    n_proteins = len(matrix_df)
    n_removed = (filter_df["filter_status"] != "PASSED").sum()
    logging.info("Loaded %d proteins total; %d removed by detection filter", n_proteins, n_removed)

    # --- Produce plots ---
    logging.info("Generating Plot 1: filter category bar chart")
    fig_categories = plot_filter_categories(filter_df, run_id)

    logging.info("Generating Plot 2: per-sample detection bar chart")
    fig_sample_det = plot_per_sample_detection(
        matrix_df, metadata_df, group_col, run_id, n_total_proteins=n_proteins
    )

    logging.info("Generating Plot 3: missingness heatmap (filtered-out proteins)")
    fig_heatmap = plot_missingness_heatmap(
        filter_df, matrix_df, metadata_df, group_col, run_id
    )

    logging.info("Generating Plot 4: per-protein missingness histogram")
    fig_histogram = plot_missingness_histogram(matrix_df, run_id)

    figs: list[tuple[str, go.Figure]] = [
        ("Plot 1: Detection filter categories", fig_categories),
        ("Plot 2: Per-sample detection counts", fig_sample_det),
        ("Plot 3: Missingness heatmap (filtered-out proteins)", fig_heatmap),
        ("Plot 4: Per-protein missingness rate", fig_histogram),
    ]

    # --- Write static PNGs ---
    plots_dir = outdir / (run_id + ".missingness_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_names = [
        "filter_categories",
        "per_sample_detection",
        "missingness_heatmap",
        "missingness_histogram",
    ]
    for (_, fig), name in zip(figs, plot_names, strict=True):
        # String concatenation to avoid Path.with_suffix replacing multi-part stems
        png_path = plots_dir / (run_id + "." + name + ".png")
        logging.info("Writing PNG: %s", png_path)
        save_png(fig, png_path)

    # --- Write consolidated HTML report ---
    generate_html_report(run_id, figs, outdir)

    logging.info("missingness_report.py complete for run: %s", run_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
