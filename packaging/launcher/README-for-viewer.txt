===========================================================================
 ProSIFT Viewer - how to open and browse the results
===========================================================================

This folder is a self-contained viewer for a set of ProSIFT proteomics
results. It runs on your own Mac. Nothing is uploaded anywhere; the data
stays in this folder.

You do NOT need to run the analysis pipeline, and you do NOT need any API
keys or special access. You only need R installed once.


---------------------------------------------------------------------------
 ONE-TIME SETUP (about 5 minutes)
---------------------------------------------------------------------------

1. Install R (only if you do not already have it):
   - Go to https://cloud.r-project.org/bin/macosx/
   - Download the .pkg that matches your Mac (Apple silicon vs. Intel) and
     install it like any other app. You do NOT need RStudio.
   - If you are not sure whether you have R, just try step 2 below first.
     The launcher will send you to this page if R is missing.


---------------------------------------------------------------------------
 EVERY TIME YOU WANT TO LOOK AT THE RESULTS
---------------------------------------------------------------------------

1. Double-click:  Launch ProSIFT Viewer.command

   The FIRST time only, macOS may say the file is from an unidentified
   developer and refuse to open it. If so:
     - Right-click (or Control-click) the file
     - Choose "Open"
     - Click "Open" in the dialog
   You only have to do this once.

2. A small black Terminal window opens. The very first launch installs a
   few R packages, which can take a few minutes and needs internet. Later
   launches skip this and are quick.

3. A browser tab opens with the ProSIFT viewer. If it does not open on its
   own, look in the Terminal window for a line like
   "Listening on http://127.0.0.1:XXXX" and paste that address into your
   browser.

4. When you are done, close the Terminal window to stop the viewer.


---------------------------------------------------------------------------
 WHAT YOU CAN DO IN THE VIEWER
---------------------------------------------------------------------------

- Top bar: pick which run (contrast) you are looking at.

- Protein database tab: a sortable, filterable table of all proteins with
  their fold changes, significance, and annotation richness. Click a row to
  open that protein's full profile.

- Protein profile: everything known about one protein - statistics,
  UniProt annotation, disease associations, drug interactions, chemical
  interactions, literature co-occurrence, and the enriched terms it belongs
  to. Click an enriched term to jump to that term.

- Biological process tab: the enriched GO / Reactome terms, with a member
  protein list and GSEA plots. Click a protein to jump back to its profile.

The protein and biological-process views link back and forth, so you can
click through from a protein to its terms and from a term to its proteins.


---------------------------------------------------------------------------
 TROUBLESHOOTING
---------------------------------------------------------------------------

- "R is not installed": the launcher opens the R download page. Install R,
  then double-click the launcher again.

- Nothing happens / the tab does not open: read the Terminal window for the
  http://127.0.0.1:... address and open it in your browser manually.

- It fails to install packages: check that you have an internet connection
  for the first launch, then try again.

- Anything else: send Rei the text shown in the Terminal window.

Questions: Reina Hastings, reinahastings13@gmail.com
===========================================================================
