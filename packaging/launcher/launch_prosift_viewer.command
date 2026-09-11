#!/bin/bash
# =============================================================================
# Title:         launch_prosift_viewer.command  (shipped as "Launch ProSIFT Viewer.command")
# Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
# Author:        Reina Hastings (reinahastings13@gmail.com)
# Created:       2026-07-16
# Last Modified: 2026-07-16
# Purpose:       macOS double-click launcher for the ProSIFT read-only viewer.
#                Locates Rscript, changes into the bundle root, and hands off to
#                launch.R (which installs packages if needed and starts the app).
#                If R is not installed, opens the macOS R download page.
# Inputs:        Lives in the bundle root next to launch.R, app/, and data/.
# Outputs:       Starts the Shiny app (opens a browser tab). No files written.
# Usage:         Double-click in Finder. First time: right-click > Open to get
#                past Gatekeeper (downloaded file quarantine).
# =============================================================================

set -euo pipefail

# --- 1. Move into the bundle root (the folder holding this script) ----------
# When double-clicked, the working directory is the user's home, so cd
# explicitly to this script's directory before doing anything relative.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# --- 2. Locate Rscript ------------------------------------------------------
# Check the common macOS install locations plus whatever is on PATH.
RSCRIPT=''
for candidate in \
  /usr/local/bin/Rscript \
  /opt/homebrew/bin/Rscript \
  /Library/Frameworks/R.framework/Resources/bin/Rscript \
  "$(command -v Rscript 2>/dev/null || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    RSCRIPT="$candidate"
    break
  fi
done

# --- 3. If R is missing, send the user to the download page -----------------
if [ -z "$RSCRIPT" ]; then
  echo ''
  echo 'R is not installed on this Mac yet.'
  echo 'Opening the R download page in your browser...'
  open 'https://cloud.r-project.org/bin/macosx/' || true
  echo ''
  echo 'Install R (the .pkg for your Mac), then double-click this launcher again.'
  echo ''
  read -n 1 -s -r -p 'Press any key to close this window.'
  echo ''
  exit 1
fi

# --- 4. Launch the viewer ---------------------------------------------------
echo ''
echo 'Starting the ProSIFT viewer...'
echo 'The first launch installs a few R packages and may take a few minutes'
echo '(it needs an internet connection that one time). After that it is quick.'
echo ''
echo 'A browser tab will open when it is ready.'
echo 'Keep THIS window open while you use the viewer. Close it to stop.'
echo ''

"$RSCRIPT" launch.R
