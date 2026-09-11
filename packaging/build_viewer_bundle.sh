#!/bin/bash
# =============================================================================
# Title:         build_viewer_bundle.sh
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-16
# Last Modified: 2026-07-16
# Purpose:       Assemble a self-contained, double-click ProSIFT viewer bundle
#                for handoff to a labmate on macOS. Copies the Shiny frontend,
#                the selected run databases plus their GSEA enrichment PNGs, and
#                the launcher + README into one folder that runs with no pipeline,
#                conda, API keys, or cluster access.
# Inputs:        -s  source results dir holding <run>/prosift_results.db
#                    (default: results_cluster)
#                -f  frontend app dir (default: frontend)
#                -o  output bundle dir (default: ./ProSIFT-viewer)
#                -r  comma-separated run folder names to include
#                    (default: every run under -s that has a prosift_results.db)
#                -l  launcher template dir (default: packaging/launcher)
#                -h  usage
# Outputs:       A bundle directory containing:
#                  app/                   copy of the frontend
#                  data/<run>/            prosift_results.db + enrichment/ PNGs
#                  Launch ProSIFT Viewer.command   double-click launcher
#                  launch.R                        R bootstrap
#                  README-for-viewer.txt           instructions for the recipient
# Usage:         packaging/build_viewer_bundle.sh
#                packaging/build_viewer_bundle.sh -r CTXcyto_WT_vs_CTXcyto_KO -o ~/Desktop/ProSIFT-viewer
# =============================================================================

set -euo pipefail

# --- Defaults (resolved relative to the project root = this script's parent) -
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_DIR="$PROJECT_ROOT/results_cluster"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
OUT_DIR="$PROJECT_ROOT/ProSIFT-viewer"
LAUNCHER_DIR="$SCRIPT_DIR/launcher"
RUNS_CSV=''

usage() {
  cat <<'EOF'
Assemble a self-contained ProSIFT viewer bundle for macOS handoff.

Usage: build_viewer_bundle.sh [-s src] [-f frontend] [-o out] [-r runs] [-l launcher] [-h]

  -s  Source results dir holding <run>/prosift_results.db   (default: results_cluster)
  -f  Frontend app dir                                      (default: frontend)
  -o  Output bundle dir                                     (default: ./ProSIFT-viewer)
  -r  Comma-separated run folder names to include           (default: all runs found)
  -l  Launcher template dir                                 (default: packaging/launcher)
  -h  Show this help

Examples:
  build_viewer_bundle.sh
  build_viewer_bundle.sh -r CTXcyto_WT_vs_CTXcyto_KO,HIPcyto_WT_vs_HIPcyto_KO
  build_viewer_bundle.sh -s results_cluster -o ~/Desktop/ProSIFT-viewer
EOF
}

# --- Argument parsing --------------------------------------------------------
while getopts ':s:f:o:r:l:h' opt; do
  case "$opt" in
    s) SRC_DIR="$OPTARG" ;;
    f) FRONTEND_DIR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    r) RUNS_CSV="$OPTARG" ;;
    l) LAUNCHER_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "ERROR: -$OPTARG requires an argument." >&2; usage; exit 2 ;;
    \?) echo "ERROR: unknown option -$OPTARG." >&2; usage; exit 2 ;;
  esac
done

# --- Validate inputs ---------------------------------------------------------
for d in "$SRC_DIR" "$FRONTEND_DIR" "$LAUNCHER_DIR"; do
  if [ ! -d "$d" ]; then
    echo "ERROR: directory not found: $d" >&2
    exit 1
  fi
done
DB_NAME='prosift_results.db'

# --- Determine which runs to include ----------------------------------------
# 1. Build the run list: explicit -r list, or auto-discover every subfolder of
#    the source dir that contains a prosift_results.db.
declare -a RUNS=()
if [ -n "$RUNS_CSV" ]; then
  IFS=',' read -r -a RUNS <<< "$RUNS_CSV"
else
  while IFS= read -r dbpath; do
    RUNS+=("$(basename "$(dirname "$dbpath")")")
  done < <(find "$SRC_DIR" -maxdepth 2 -name "$DB_NAME" | sort)
fi

if [ "${#RUNS[@]}" -eq 0 ]; then
  echo "ERROR: no runs with a $DB_NAME found under $SRC_DIR (and none given via -r)." >&2
  exit 1
fi

# --- Assemble the bundle -----------------------------------------------------
echo "Building ProSIFT viewer bundle"
echo "  source   : $SRC_DIR"
echo "  frontend : $FRONTEND_DIR"
echo "  output   : $OUT_DIR"
echo "  runs     : ${RUNS[*]}"
echo ''

# 2. Fresh output tree.
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/app" "$OUT_DIR/data"

# 3. Copy the frontend app (only what the app needs to run: app.R, R/, www/).
#    Skip the test suite and dev helpers so the recipient gets a clean app.
cp "$FRONTEND_DIR/app.R" "$OUT_DIR/app/"
cp -R "$FRONTEND_DIR/R" "$OUT_DIR/app/"
cp -R "$FRONTEND_DIR/www" "$OUT_DIR/app/"

# 4. Copy each run: the database plus its enrichment/ PNGs (used by the
#    Biological Process view for GSEA running-score plots). Other result
#    subfolders (differential_abundance/, normalization/) are not read by the
#    viewer and are intentionally left out to keep the bundle small.
for run in "${RUNS[@]}"; do
  src_run="$SRC_DIR/$run"
  db="$src_run/$DB_NAME"
  if [ ! -f "$db" ]; then
    echo "ERROR: no $DB_NAME in $src_run" >&2
    exit 1
  fi
  mkdir -p "$OUT_DIR/data/$run"
  echo "  + $run  ($(du -h "$db" | cut -f1))"
  cp "$db" "$OUT_DIR/data/$run/"
  if [ -d "$src_run/enrichment" ]; then
    cp -R "$src_run/enrichment" "$OUT_DIR/data/$run/"
  else
    echo "    (note: no enrichment/ folder; GSEA plots will be omitted for this run)"
  fi
done

# 5. Drop in the launcher, R bootstrap, and recipient README.
cp "$LAUNCHER_DIR/launch.R" "$OUT_DIR/"
cp "$LAUNCHER_DIR/README-for-viewer.txt" "$OUT_DIR/"
cp "$LAUNCHER_DIR/launch_prosift_viewer.command" "$OUT_DIR/Launch ProSIFT Viewer.command"
chmod +x "$OUT_DIR/Launch ProSIFT Viewer.command"

# --- Summary -----------------------------------------------------------------
echo ''
echo "Done. Bundle at: $OUT_DIR"
echo "  total size: $(du -sh "$OUT_DIR" | cut -f1)"
echo ''
echo "Next steps:"
echo "  1. Test locally:  double-click '$OUT_DIR/Launch ProSIFT Viewer.command'"
echo "  2. Zip and send:  the whole ProSIFT-viewer folder (or share via cloud drive)."
