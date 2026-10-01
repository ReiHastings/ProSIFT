#!/usr/bin/env python3
"""
Title:         enrichment.py
Project:       ProSIFT (PROtein Statistical Integration and Filtering Tool)
Author:        Reina Hastings (reinahastings13@gmail.com)
Created:       2026-03-31
Last Modified: 2026-09-30
Purpose:       Module 05 ENRICHMENT process. Runs overrepresentation analysis (ORA)
               via gseapy.enrich() and preranked GSEA via gseapy.prerank() against
               local MSigDB GMT files. Operates on gene symbols from Module 04's
               differential abundance results table. Produces a unified enrichment
               results table, a protein-term mapping table, lollipop plots (Plotly,
               PNG + HTML) per contrast per library, GSEA running score plots (gseapy
               built-in, PNG) for top significant terms, and a summary text file.
Inputs:
  --results      {run_id}.diff_abundance_results.parquet  (Module 04 DIFFERENTIAL_ABUNDANCE)
  --params       {run_id}_params.yml
  --gene-set-libraries  GMT files staged by Nextflow, in params.yml order
                 (optional; when omitted, enrichment.gene_set_libraries paths
                 are resolved relative to the params.yml directory)
Outputs:
  {run_id}.enrichment_results.parquet
  {run_id}.enrichment_results.csv
  {run_id}.protein_term_mapping.parquet
  {run_id}.enrichment_summary.txt
  {run_id}.{contrast}.{library}.ora_lollipop.png/.html   (per contrast, per library)
  {run_id}.{contrast}.{library}.gsea_lollipop.png/.html  (per contrast, per library)
  {run_id}.{contrast}.{library}.gsea_running_score.{term}.png  (top N terms)
Usage:
  enrichment.py --results CTXcyto_WT_vs_CTXcyto_KO.diff_abundance_results.parquet \
                --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \
                --run-id CTXcyto_WT_vs_CTXcyto_KO \
                --outdir .
  (Nextflow adds: --gene-set-libraries gmt/1/m5.go.bp.v2026.1.Mm.symbols.gmt ...)
"""

import argparse
import datetime
import hashlib
import logging
import re
import sys
from pathlib import Path

import gseapy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from gseapy.stats import multiple_testing_correction

# rpy2 (embedded R, for rrvgo GO-term redundancy reduction, Section 4.13) is
# imported lazily inside cluster_go_terms, NOT at module top. Importing
# rpy2.robjects starts embedded R, which segfaults where R is not linked -- an
# uncatchable native crash that would make this module un-importable for the
# pure-Python tests, --help, or CI. Lazy loading keeps the module import
# side-effect-free; the R stack is only touched when clustering is actually run.

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="enrichment.py",
        description="ProSIFT Module 05 ENRICHMENT: ORA and GSEA enrichment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  enrichment.py \\\n"
            "    --results CTXcyto_WT_vs_CTXcyto_KO.diff_abundance_results.parquet \\\n"
            "    --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \\\n"
            "    --run-id CTXcyto_WT_vs_CTXcyto_KO \\\n"
            "    --outdir .\n"
        ),
    )
    parser.add_argument("--results",  required=True, help="Module 04 diff_abundance_results.parquet")
    parser.add_argument("--params",   required=True, help="Run params.yml")
    parser.add_argument("--run-id",   required=True, dest="run_id", help="Run identifier prefix for output files")
    parser.add_argument("--outdir",   required=True, help="Output directory")
    parser.add_argument(
        "--gene-set-libraries", nargs="+", default=None, dest="gene_set_libraries",
        metavar="GMT",
        help=("GMT files staged by Nextflow, one per enrichment.gene_set_libraries "
              "entry and in the same order. Overrides the params.yml paths, which "
              "are then only used to check the order. Omit for standalone runs."),
    )
    return parser.parse_args()


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ============================================================
# LIBRARY SHORT NAME MAPPING
# ============================================================

# Maps substrings found in GMT filenames to short library identifiers.
# Used for output file naming and the `library` column in results tables.
# Checked in order; first match wins. Falls back to the filename stem.
_LIBRARY_NAME_MAP: list[tuple[str, str]] = [
    ("go.bp",           "GO_BP"),
    ("go.mf",           "GO_MF"),
    ("go.cc",           "GO_CC"),
    ("kegg_medicus",    "KEGG"),
    ("kegg_legacy",     "KEGG_LEGACY"),
    ("reactome",        "REACTOME"),
    ("hallmark",        "HALLMARK"),
    ("mh.all",          "HALLMARK"),
]

def _library_short_name(gmt_path: str) -> str:
    """Derive a short library identifier from a GMT file path."""
    stem = Path(gmt_path).stem.lower()
    for substring, short_name in _LIBRARY_NAME_MAP:
        if substring in stem:
            return short_name
    # Fallback: uppercase the filename stem, truncated
    return Path(gmt_path).stem.upper()[:20]


# MSigDB release token embedded in official GMT filenames, e.g.
#   m5.go.bp.v2026.1.Mm.symbols.gmt   -> v2026.1.Mm   (2023+ scheme, species-tagged)
#   c5.go.bp.v7.5.1.symbols.gmt       -> v7.5.1       (pre-2023 human scheme)
# The GMT content itself carries no release metadata, so the filename is the
# only source for the version; the SHA-256 recorded alongside it in the summary
# identifies the exact file even if it was renamed.
_MSIGDB_VERSION_RE = re.compile(r"\.(v\d+(?:\.\d+)+(?:\.(?:Hs|Mm))?)\.", re.IGNORECASE)

# Recorded when the filename does not follow MSigDB naming (custom GMTs).
GMT_VERSION_UNKNOWN = "unknown"


def _gmt_version(gmt_path: str) -> str:
    """Return the MSigDB release token from a GMT filename, or 'unknown'."""
    match = _MSIGDB_VERSION_RE.search(Path(gmt_path).name)
    return match.group(1) if match else GMT_VERSION_UNKNOWN


def _java_trim(s: str) -> str:
    """Strip leading/trailing chars <= U+0020, matching Java/Groovy String.trim()."""
    start, end = 0, len(s)
    while start < end and s[start] <= " ":
        start += 1
    while end > start and s[end - 1] <= " ":
        end -= 1
    return s[start:end]


def _file_sha256(path: str) -> str:
    """Hex SHA-256 of a file, read in chunks (GMT files can be tens of MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# PARAMETER LOADING
# ============================================================

def load_params(params_path: str, staged_libraries: list[str] | None = None) -> dict:
    """
    Load and validate enrichment parameters from params.yml.

    staged_libraries: GMT paths staged by Nextflow (--gene-set-libraries). When
    given, they replace the params.yml paths, which must match them one-to-one
    by position and filename. When None (standalone use), params.yml paths are
    resolved relative to the params.yml directory.
    """
    params_dir = Path(params_path).parent.resolve()

    with open(params_path) as fh:
        params = yaml.safe_load(fh)

    enr = params.get("enrichment", {})

    # Required: gene_set_libraries must be a list of paths.
    # Standalone: paths are resolved relative to the params.yml file's
    # directory, so relative paths do not depend on the working directory.
    # Nextflow: the workflow applies the same rule to find the files, stages
    # them into the task, and passes them via --gene-set-libraries (the staged
    # params.yml symlink's directory is the work dir, where they do not exist).
    libraries = enr.get("gene_set_libraries", [])
    if not libraries:
        logging.error("params.yml: enrichment.gene_set_libraries is empty or missing. "
                      "Provide at least one GMT file path.")
        sys.exit(1)
    # Trim each entry exactly as workflows/prosift.nf (resolve_gmts) does with
    # Groovy/Java String.trim(): strip leading/trailing chars <= U+0020 (ASCII
    # whitespace and control chars), not Unicode whitespace such as NBSP. Both
    # sides then accept or reject the same entries.
    libraries = [_java_trim(str(p)) for p in libraries]
    resolved = []
    if staged_libraries is not None:
        # Nextflow path: the workflow read this same list and staged each file
        # under gmt/<index>/<original filename>, so the params.yml path need not
        # exist inside the task. Check the pairing so a misordered or
        # mismatched list fails here instead of mislabelling a library.
        if len(staged_libraries) != len(libraries):
            logging.error("--gene-set-libraries has %d file(s) but params.yml lists %d. "
                          "They must correspond one-to-one.",
                          len(staged_libraries), len(libraries))
            sys.exit(1)
        for staged, entry in zip(staged_libraries, libraries, strict=True):
            staged_p = Path(staged).absolute()
            if staged_p.name != Path(entry).name:
                logging.error("Staged GMT '%s' does not match params.yml entry '%s' "
                              "(filenames differ; order must match).", staged_p.name, entry)
                sys.exit(1)
            if not staged_p.is_file():
                logging.error("Staged gene set library not found (or not a file): %s", staged_p)
                sys.exit(1)
            resolved.append(str(staged_p))
    else:
        for p in libraries:
            resolved_p = Path(p) if Path(p).is_absolute() else (params_dir / p).resolve()
            # is_file(): a blank entry resolves to the params directory itself.
            if not resolved_p.is_file():
                logging.error("Gene set library not found (or not a file): %s (resolved from %r)",
                              resolved_p, p)
                sys.exit(1)
            resolved.append(str(resolved_p))
    enr["gene_set_libraries"] = resolved

    # Defaults with explicit type coercion
    enr.setdefault("run_ora",           True)
    enr.setdefault("run_gsea",          True)
    enr.setdefault("background",        "detected")
    enr.setdefault("gsea_ranking",      "t_statistic")
    enr.setdefault("min_gene_set_size", 15)
    enr.setdefault("max_gene_set_size", 500)
    enr.setdefault("fdr_threshold",     0.05)
    enr.setdefault("plot_top_n",        20)
    enr.setdefault("plot_top_gsea_traces", 10)
    enr.setdefault("gsea_permutations", 1000)
    enr.setdefault("gsea_seed",         42)

    # Validate values that change which terms are tested. bool is excluded
    # explicitly because it is a subclass of int in Python.
    def _is_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    min_size, max_size = enr["min_gene_set_size"], enr["max_gene_set_size"]
    if not (_is_int(min_size) and _is_int(max_size) and 1 <= min_size <= max_size):
        logging.error("params.yml: enrichment.min_gene_set_size (%r) and max_gene_set_size (%r) "
                      "must be integers with 1 <= min <= max.", min_size, max_size)
        sys.exit(1)
    # gseapy.prerank with permutation_num == 0 returns no p-values, FDR or Tag %,
    # so GSEA cannot produce the results table.
    permutations = enr["gsea_permutations"]
    if enr["run_gsea"] and not (_is_int(permutations) and permutations >= 1):
        logging.error("params.yml: enrichment.gsea_permutations must be an integer >= 1 "
                      "when run_gsea is true (got %r).", permutations)
        sys.exit(1)

    params["enrichment"] = enr
    return params


# ============================================================
# GENE SYMBOL PREPARATION
# ============================================================

def prepare_gene_symbols(
    da_df: pd.DataFrame,
    contrast: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Filter and deduplicate the Module 04 results for one contrast.

    Returns:
        contrast_df: filtered, deduplicated DataFrame for this contrast
        stats: dict of counts for summary reporting
    """
    df = da_df[da_df["contrast"] == contrast].copy()
    n_total = len(df)

    # --- Drop unmapped proteins (null gene_symbol) ---
    n_unmapped = df["gene_symbol"].isna().sum()
    df = df[df["gene_symbol"].notna()].copy()
    if n_unmapped > 0:
        logging.warning(
            "Contrast %s: dropped %d proteins with null gene_symbol", contrast, n_unmapped
        )

    # --- Determine primary adjusted p-value column ---
    # Use deqms_adj_pvalue if present and non-null, otherwise limma_adj_pvalue
    if "deqms_adj_pvalue" in df.columns and df["deqms_adj_pvalue"].notna().any():
        pval_col = "deqms_adj_pvalue"
    else:
        pval_col = "limma_adj_pvalue"

    # --- Deduplicate: one row per gene_symbol, keep lowest adj_pvalue ---
    n_before_dedup = len(df)
    df = (
        df.sort_values(pval_col, ascending=True)
          .drop_duplicates(subset="gene_symbol", keep="first")
          .reset_index(drop=True)
    )
    n_collapsed = n_before_dedup - len(df)
    if n_collapsed > 0:
        logging.info(
            "Contrast %s: collapsed %d duplicate gene symbols (kept most significant per gene)",
            contrast, n_collapsed,
        )

    stats = {
        "n_total":     n_total,
        "n_unmapped":  n_unmapped,
        "n_mapped":    n_total - n_unmapped,
        "n_collapsed": n_collapsed,
        "n_unique":    len(df),
        "pval_col":    pval_col,
    }
    return df, stats


# ============================================================
# GSEA RANKING METRIC
# ============================================================

def build_ranked_series(df: pd.DataFrame, ranking: str) -> pd.Series:
    """
    Build the gene-symbol-indexed ranked Series for gseapy.prerank().

    ranking options:
      "t_statistic"   -- deqms_t if available, else limma_t
      "signed_log10p" -- sign(log2_fc) * -log10(raw_pvalue), clamped for p=0
      "log2fc"        -- log2_fc directly
    """
    if ranking == "t_statistic":
        t_col = "deqms_t" if ("deqms_t" in df.columns and df["deqms_t"].notna().any()) else "limma_t"
        rnk = df.set_index("gene_symbol")[t_col].astype(float)

    elif ranking == "signed_log10p":
        # Use deqms_pvalue or limma_pvalue (raw, not adjusted -- GSEA does its own FDR)
        raw_p_col = "deqms_pvalue" if ("deqms_pvalue" in df.columns and df["deqms_pvalue"].notna().any()) else "limma_pvalue"
        pvals = df[raw_p_col].astype(float)
        # Clamp p=0 to the smallest nonzero p-value to avoid log(0) = -inf
        min_p = pvals[pvals > 0].min()
        pvals = pvals.clip(lower=min_p)
        signs = np.sign(df["log2_fc"].astype(float))
        scores = signs * (-np.log10(pvals))
        rnk = pd.Series(scores.values, index=df["gene_symbol"])

    elif ranking == "log2fc":
        rnk = df.set_index("gene_symbol")["log2_fc"].astype(float)

    else:
        logging.error("Unknown gsea_ranking: %s. Choose t_statistic, signed_log10p, or log2fc.", ranking)
        sys.exit(1)

    return rnk.dropna()


# ============================================================
# gseapy OUTPUT VALIDATION (shared by ORA and GSEA)
# ============================================================

# gseapy version whose enrich/prerank output structure this module was
# validated against (pinned in environment.yml). An environment built another
# way may differ, so the installed version is reported in every
# GseapyResultSchemaError message.
_GSEAPY_VALIDATED_VERSION = "1.1.13"

# Message gseapy.prerank raises (as a plain LookupError) when no gene set
# survives size filtering and matching to the ranked list.
_GSEAPY_NO_SETS_MSG = "No gene sets passed through filtering condition"


class GseapyResultSchemaError(RuntimeError):
    """gseapy enrich/prerank output no longer matches the structure ProSIFT was validated against."""


def _gseapy_schema_error(term: str, problem: str) -> GseapyResultSchemaError:
    """Build a GseapyResultSchemaError carrying the term and version context."""
    return GseapyResultSchemaError(
        f"gseapy result for term '{term}': {problem} "
        f"(installed gseapy {gseapy.__version__}; ProSIFT validated against "
        f"gseapy {_GSEAPY_VALIDATED_VERSION}). Check whether the gseapy output "
        f"structure changed."
    )


def _parse_ora_overlap(term: str, overlap: object) -> tuple[int, int]:
    """
    Parse gseapy.enrich's 'Overlap' ('<hits>/<set size>') into (hits, set size).

    The set size is the term's size after intersection with the background
    (gseapy.stats.calc_pvalues), which is what the size filter and
    gene_set_size use. Raises GseapyResultSchemaError if unparseable, since a
    silent 0 would be dropped by the size filter without any signal.
    """
    # ASCII digits only: isdigit/isdecimal alone accept superscripts and
    # non-ASCII decimal digits, which gseapy never emits.
    parts = [part.strip() for part in str(overlap).split("/")]
    if len(parts) != 2 or not all(part.isascii() and part.isdecimal() for part in parts):
        raise _gseapy_schema_error(
            term, f"ORA Overlap is '{overlap}', expected '<hits>/<set size>'"
        )
    return int(parts[0]), int(parts[1])


# ============================================================
# ORA (gseapy.enrich)
# ============================================================

def run_ora(
    sig_genes: list[str],
    background_genes: list[str],
    gmt_path: str,
    library_name: str,
    contrast: str,
    params: dict,
) -> pd.DataFrame:
    """
    Run ORA for one library against one contrast's significant gene set.
    Returns a DataFrame in the unified enrichment results schema.
    Returns an empty DataFrame if no significant genes or no significant terms.
    """
    enr = params["enrichment"]
    fdr_threshold = enr["fdr_threshold"]

    if not sig_genes:
        logging.warning(
            "Contrast %s, library %s: no significant genes, ORA skipped", contrast, library_name
        )
        return pd.DataFrame()

    # No try/except: gseapy.enrich signals "nothing to test" (no overlap, no
    # term in the background) by returning res2d = None, handled below. Any
    # exception is a real failure and must fail the process rather than leave
    # a library silently missing from the results.
    result = gseapy.enrich(
        gene_list=sig_genes,
        gene_sets=gmt_path,
        background=background_genes,
        no_plot=True,
        verbose=False,
        cutoff=1.0,  # Return all terms; we filter ourselves for consistent handling
    )

    res = result.res2d
    if res is None or res.empty:
        logging.info("Contrast %s, library %s ORA: no terms returned", contrast, library_name)
        return pd.DataFrame()

    # --- Gene set size filter (spec Section 4.5) ---
    # gseapy.enrich has no size parameter, so it tests every term with >= 1 hit.
    # Step 1: read each term's background-intersected size from 'Overlap'.
    # Step 2: keep terms within [min_gene_set_size, max_gene_set_size].
    # Step 3: recompute BH over the kept terms with gseapy's own correction.
    # This equals excluding out-of-range sets before testing: hypergeometric
    # p-values are per-term and unchanged; only the BH family shrinks.
    overlaps = [_parse_ora_overlap(t, o) for t, o in zip(res["Term"], res["Overlap"], strict=True)]
    res = res.assign(
        _overlap_size=[hits for hits, _ in overlaps],
        _gene_set_size=[size for _, size in overlaps],
    )
    min_size, max_size = enr["min_gene_set_size"], enr["max_gene_set_size"]
    in_range = res["_gene_set_size"].between(min_size, max_size)
    n_excluded = int((~in_range).sum())
    res = res[in_range].copy()
    logging.info(
        "Contrast %s, library %s ORA: %d terms outside size filter %d-%d excluded before BH",
        contrast, library_name, n_excluded, min_size, max_size,
    )
    if res.empty:
        logging.info("Contrast %s, library %s ORA: no terms within size filter", contrast, library_name)
        return pd.DataFrame()
    res["Adjusted P-value"] = multiple_testing_correction(
        ps=res["P-value"].astype(float).to_numpy(), alpha=1.0, method="benjamini-hochberg",
    )[0]

    # --- Map gseapy columns to unified schema ---
    # gseapy enrich columns: Gene_set, Term, Overlap, P-value, Adjusted P-value,
    #                        Odds Ratio, Combined Score, Genes
    # gseapy joins overlap genes from a Python set, so their order varies between
    # processes (string hash randomization); sort for reproducible output.
    out = pd.DataFrame({
        "term_id":         res["Term"],
        "term_name":       res["Term"],
        "library":         library_name,
        "analysis_type":   "ORA",
        "contrast":        contrast,
        "pvalue":          res["P-value"].astype(float),
        "adj_pvalue":      res["Adjusted P-value"].astype(float),
        "enrichment_score": np.nan,
        "odds_ratio":      res["Odds Ratio"].astype(float),
        "combined_score":  res["Combined Score"].astype(float),
        "gene_set_size":   res["_gene_set_size"],
        "overlap_size":    res["_overlap_size"],
        "overlap_genes":   res["Genes"].map(
            lambda s: ";".join(sorted(g for g in str(s).split(";") if g))
        ),
    })

    logging.info(
        "Contrast %s, library %s ORA: %d terms tested, %d significant (FDR < %.2f)",
        contrast, library_name, len(out),
        (out["adj_pvalue"] < fdr_threshold).sum(), fdr_threshold,
    )
    return out.reset_index(drop=True)


# ============================================================
# GSEA (gseapy.prerank)
# ============================================================

def count_matched_genes(term: str, term_data: object, tag_pct: object = None) -> int:
    """
    Number of the term's genes present in the ranked list (GSEA gene_set_size).

    term_data is pre_res.results[term] from gseapy.prerank. In gseapy 1.1.13
    'matched_genes' is a ';'-joined str, so len() on it counts characters
    (KNOWN_ISSUES E-9); the genes must be split out first.

    tag_pct is the res2d 'Tag %' value ('<leading edge>/<matched set size>').
    Its denominator is computed by gseapy independently of matched_genes, so it
    serves as a cross-check. When permutation_num == 0 gseapy omits the 'Tag %'
    column from res2d entirely, so the caller passes None and the check is
    skipped (as it is for NaN or a blank string); any other value must parse
    as 'a/b'.

    Raises GseapyResultSchemaError if the structure is not as expected, the
    matched set is empty, Tag % is present but unparseable, or the two counts
    disagree.
    """
    # Step 1: structure of the per-term record
    if not isinstance(term_data, dict):
        raise _gseapy_schema_error(
            term, f"results entry is {type(term_data).__name__}, expected dict"
        )
    if "matched_genes" not in term_data:
        raise _gseapy_schema_error(
            term, f"results entry has no 'matched_genes' field "
                  f"(fields present: {sorted(map(str, term_data.keys()))})"
        )

    # Step 2: count genes. A str is the validated format; a sequence of symbols
    # is also unambiguous. Anything else cannot be counted safely.
    matched = term_data["matched_genes"]
    if isinstance(matched, str):
        count = len([g for g in matched.split(";") if g.strip()])
    elif isinstance(matched, (list, tuple, np.ndarray)):
        # Same rule as the str branch: blank entries are not genes.
        count = len([g for g in matched if str(g).strip()])
    else:
        raise _gseapy_schema_error(
            term, f"'matched_genes' is {type(matched).__name__}, "
                  f"expected ';'-joined str or a sequence of gene symbols"
        )
    # gseapy drops sets below min_size after intersecting with the ranked list,
    # so a reported term can never have 0 matched genes; 0 means malformed output.
    if count == 0:
        raise _gseapy_schema_error(
            term, f"'matched_genes' contains no gene symbols ({matched!r})"
        )

    # Step 3: cross-check against the Tag % denominator. Only an absent value
    # (None, NaN, empty string) skips the check; a present value in any format
    # other than 'a/b' is itself a sign the gseapy output changed.
    # pd.isna covers None, float NaN of any width (np.float32 is not a Python
    # float), pd.NA and pd.NaT; is_scalar keeps a list or array from reaching
    # pd.isna, which would return an array.
    tag_absent = (
        (pd.api.types.is_scalar(tag_pct) and pd.isna(tag_pct))
        or (isinstance(tag_pct, str) and not tag_pct.strip())
    )
    if not tag_absent:
        parts = [part.strip() for part in str(tag_pct).split("/")]
        if len(parts) != 2 or not all(part.isascii() and part.isdecimal() for part in parts):
            raise _gseapy_schema_error(
                term, f"Tag % is '{tag_pct}', expected '<leading edge>/<set size>'"
            )
        tag_size = int(parts[1])
        if tag_size != count:
            raise _gseapy_schema_error(
                term, f"matched_genes gives {count} genes but Tag % "
                      f"('{tag_pct}') gives {tag_size}"
            )

    return count


def run_gsea(
    ranked_series: pd.Series,
    gmt_path: str,
    library_name: str,
    contrast: str,
    params: dict,
) -> tuple[pd.DataFrame, object | None]:
    """
    Run preranked GSEA for one library against one contrast's full ranked list.
    Returns (results_df, prerank_result_object).
    results_df is in the unified enrichment results schema.
    prerank_result_object is needed for running score plots; None on failure.
    """
    enr = params["enrichment"]
    fdr_threshold = enr["fdr_threshold"]

    # gseapy.prerank needs at least 2 ranked genes (a 1-gene list crashes inside
    # gseapy with an AttributeError), so treat < 2 like an empty list.
    if len(ranked_series) < 2:
        logging.warning("Contrast %s, library %s: %d ranked gene(s), GSEA skipped",
                        contrast, library_name, len(ranked_series))
        return pd.DataFrame(), None

    try:
        pre_res = gseapy.prerank(
            rnk=ranked_series,
            gene_sets=gmt_path,
            min_size=enr["min_gene_set_size"],
            max_size=enr["max_gene_set_size"],
            permutation_num=enr["gsea_permutations"],
            ascending=False,   # highest scores (most upregulated) first
            no_plot=True,
            verbose=False,
            seed=enr["gsea_seed"],
            threads=1,         # deterministic; parallel threads can affect permutation results
        )
    except LookupError as exc:
        # gseapy raises a plain LookupError when no gene set is left after the
        # size filter and ranked-list matching: a legitimate empty result. Match
        # the exact type and message, because KeyError and IndexError subclass
        # LookupError and would otherwise hide real bugs. Everything else
        # propagates and fails the process.
        if type(exc) is not LookupError or _GSEAPY_NO_SETS_MSG not in str(exc):
            raise
        logging.warning("Contrast %s, library %s: no gene sets passed the size filter "
                        "and ranked-list matching, GSEA skipped", contrast, library_name)
        return pd.DataFrame(), None

    res = pre_res.res2d
    if res is None or res.empty:
        logging.info("Contrast %s, library %s GSEA: no terms returned", contrast, library_name)
        return pd.DataFrame(), pre_res

    # --- Map gseapy columns to unified schema ---
    # gseapy prerank res2d columns:
    #   Name, Term, ES, NES, NOM p-val, FDR q-val, FWER p-val,
    #   Tag %, Gene %, Lead_genes
    def _leading_edge_size(lead_genes_str) -> int:
        if pd.isna(lead_genes_str) or lead_genes_str == "":
            return 0
        return len(str(lead_genes_str).split(";"))

    # gene_set_size comes from the per-term results dict. No fallback value:
    # a structural mismatch raises GseapyResultSchemaError and fails the process.
    results_by_term = pre_res.results
    if not isinstance(results_by_term, dict):
        raise GseapyResultSchemaError(
            f"gseapy prerank .results is {type(results_by_term).__name__}, expected dict "
            f"(installed gseapy {gseapy.__version__}; ProSIFT validated against "
            f"gseapy {_GSEAPY_VALIDATED_VERSION})"
        )

    out_rows = []
    for _, row in res.iterrows():
        term = row["Term"]
        lead_genes = row.get("Lead_genes", "")
        lead_size = _leading_edge_size(lead_genes)
        if term not in results_by_term:
            # gseapy nests results by ranking name when it was given more than
            # one ranking column; say so, since that is the likely cause.
            nested = bool(results_by_term) and all(
                isinstance(v, dict) and term in v for v in results_by_term.values()
            )
            raise _gseapy_schema_error(
                term, "term is in res2d but missing from .results"
                      + (" (.results appears nested by ranking column)" if nested else "")
            )
        gene_set_size = count_matched_genes(term, results_by_term[term], row.get("Tag %"))

        out_rows.append({
            "term_id":          term,
            "term_name":        term,
            "library":          library_name,
            "analysis_type":    "GSEA",
            "contrast":         contrast,
            "pvalue":           float(row["NOM p-val"]),
            "adj_pvalue":       float(row["FDR q-val"]),
            "enrichment_score": float(row["NES"]),
            "odds_ratio":       np.nan,
            "combined_score":   np.nan,
            "gene_set_size":    gene_set_size,
            "overlap_size":     lead_size,
            "overlap_genes":    str(lead_genes) if pd.notna(lead_genes) else "",
        })

    out = pd.DataFrame(out_rows)

    logging.info(
        "Contrast %s, library %s GSEA: %d terms tested, %d significant (FDR < %.2f)",
        contrast, library_name, len(out),
        (out["adj_pvalue"] < fdr_threshold).sum(), fdr_threshold,
    )
    return out.reset_index(drop=True), pre_res


# ============================================================
# PROTEIN-TERM MAPPING TABLE
# ============================================================

def build_protein_term_mapping(
    da_df: pd.DataFrame,
    gmt_paths: list[str],
    library_names: list[str],
    enrichment_results: pd.DataFrame,
    gsea_results_by_key: dict[tuple[str, str], object],
) -> pd.DataFrame:
    """
    Build the many-to-many protein-term mapping table.

    Includes all protein-term pairs for terms that were actually tested
    (i.e., appear in the enrichment results table). Annotates each pair
    with significance status and GSEA leading edge membership.
    """
    if enrichment_results.empty:
        return pd.DataFrame(columns=[
            "gene_symbol", "protein_id", "term_id", "term_name",
            "library", "in_significant_set", "is_leading_edge", "contrast",
        ])

    # Build protein->gene_symbol lookup (all proteins, not just significant)
    protein_gene = (
        da_df[["protein_id", "gene_symbol", "contrast", "significant"]]
        .dropna(subset=["gene_symbol"])
        .drop_duplicates()
    )

    # Build set of significant gene symbols per contrast (for in_significant_set)
    sig_genes_by_contrast: dict[str, set] = {}
    for contrast in da_df["contrast"].unique():
        sig = da_df[(da_df["contrast"] == contrast) & (da_df["significant"].eq(True))]
        sig_genes_by_contrast[contrast] = set(sig["gene_symbol"].dropna())

    # Build leading edge sets per (contrast, library, term)
    leading_edge: dict[tuple[str, str, str], set] = {}
    for (contrast, lib_name), pre_res in gsea_results_by_key.items():
        if pre_res is None:
            continue
        for term, term_data in pre_res.results.items():
            lead_str = term_data.get("lead_genes", "")
            if lead_str:
                leading_edge[(contrast, lib_name, term)] = set(str(lead_str).split(";"))

    # Parse GMT files to get gene-to-term mapping for tested terms only
    tested_terms: set = set(enrichment_results["term_id"].unique())

    rows = []
    for gmt_path, lib_name in zip(gmt_paths, library_names, strict=True):
        with open(gmt_path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                term_id = parts[0]
                if term_id not in tested_terms:
                    continue
                genes_in_term = set(parts[2:])

                for contrast in da_df["contrast"].unique():
                    # Proteins in this run that are annotated to this term
                    pg = protein_gene[protein_gene["contrast"] == contrast]
                    annotated = pg[pg["gene_symbol"].isin(genes_in_term)]

                    for _, prow in annotated.iterrows():
                        gsym = prow["gene_symbol"]
                        rows.append({
                            "gene_symbol":      gsym,
                            "protein_id":       prow["protein_id"],
                            "term_id":          term_id,
                            "term_name":        term_id,
                            "library":          lib_name,
                            "in_significant_set": gsym in sig_genes_by_contrast.get(contrast, set()),
                            "is_leading_edge":  gsym in leading_edge.get((contrast, lib_name, term_id), set()),
                            "contrast":         contrast,
                        })

    if not rows:
        return pd.DataFrame(columns=[
            "gene_symbol", "protein_id", "term_id", "term_name",
            "library", "in_significant_set", "is_leading_edge", "contrast",
        ])

    mapping_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    logging.info("Protein-term mapping: %d rows across %d contrasts", len(mapping_df), da_df["contrast"].nunique())
    return mapping_df


# ============================================================
# VISUALIZATION: LOLLIPOP PLOTS
# ============================================================

# Color scales and visual constants
_LOLLIPOP_COLORSCALE = "Blues_r"     # darker = more significant
_LOLLIPOP_STEM_COLOR = "#cccccc"
_MAX_TERM_LABEL_LEN = 55             # truncate long GO term names for static PNG

def gsea_direction_statement(contrast: str) -> str:
    """
    Plain-language reading of a positive NES for one contrast.

    Module 04 fits 'numerator - denominator' for a 'numerator_vs_denominator'
    label (split on the first '_vs_'), and every gsea_ranking option is signed by
    that log2 fold change, so NES > 0 means the set is concentrated among genes
    higher in the numerator. Falls back to a generic wording if the label does
    not follow the convention.
    """
    groups = _split_contrast_label(contrast)
    if groups is None:
        return "NES > 0 = enriched among genes with positive log2 FC"
    numerator, denominator = groups
    return (f"NES > 0 = enriched among genes higher in {numerator}; "
            f"NES < 0 = higher in {denominator}")


def _split_contrast_label(contrast: str) -> tuple[str, str] | None:
    """Split on the first '_vs_' (Module 04's rule); None if not splittable."""
    idx = contrast.find("_vs_")
    if idx <= 0 or idx + 4 >= len(contrast):
        return None
    return contrast[:idx], contrast[idx + 4:]


def gsea_axis_label(contrast: str) -> str:
    """Short NES axis title that fits the static PNG with long group names."""
    groups = _split_contrast_label(contrast)
    if groups is None:
        return "NES (> 0 = positive log2 FC)"
    return f"NES (> 0 = higher in {groups[0]})"


def _truncate_label(s: str, maxlen: int = _MAX_TERM_LABEL_LEN) -> str:
    return s if len(s) <= maxlen else s[:maxlen - 3] + "..."

def _make_lollipop_fig(
    terms: list[str],
    x_vals: list[float],
    dot_colors: list[float],   # values mapped to color scale (adj_pvalue)
    dot_sizes: list[int],      # overlap or leading edge size
    x_label: str,
    title: str,
    hover_texts: list[str],
    colorbar_title: str,
) -> go.Figure:
    """
    Build a Plotly lollipop figure.
    Terms are displayed on the y-axis (top = most significant).
    """
    labels = [_truncate_label(t) for t in terms]

    # Stems: horizontal lines from x=0 to each dot
    shapes = []
    for i, xv in enumerate(x_vals):
        shapes.append(dict(
            type="line",
            x0=0, x1=xv,
            y0=i, y1=i,
            line=dict(color=_LOLLIPOP_STEM_COLOR, width=1.5),
        ))

    # Dots
    scatter = go.Scatter(
        x=x_vals,
        y=list(range(len(terms))),
        mode="markers",
        marker=dict(
            color=dot_colors,
            colorscale=_LOLLIPOP_COLORSCALE,
            reversescale=False,
            size=[max(6, min(20, 6 + s // 5)) for s in dot_sizes],
            colorbar=dict(title=colorbar_title, thickness=12, len=0.6),
            line=dict(width=0.5, color="#555555"),
        ),
        text=hover_texts,
        hoverinfo="text",
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title=x_label, zeroline=True, zerolinewidth=1, zerolinecolor="#999999"),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            autorange="reversed",
        ),
        shapes=shapes,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=300, r=80, t=80, b=60),  # t=80: two-line title
        height=max(300, 30 * len(terms) + 120),
    )
    return fig


def _save_lollipop(fig: go.Figure, base_path: str) -> None:
    """Save a lollipop figure as both PNG (kaleido) and HTML (Plotly)."""
    try:
        fig.write_image(base_path + ".png", scale=2)
    except Exception as exc:
        logging.warning("Failed to write PNG %s: %s", base_path + ".png", exc)
    try:
        fig.write_html(base_path + ".html", include_plotlyjs="cdn")
    except Exception as exc:
        logging.warning("Failed to write HTML %s: %s", base_path + ".html", exc)


def plot_ora_lollipop(
    ora_df: pd.DataFrame,
    contrast: str,
    library_name: str,
    run_id: str,
    outdir: Path,
    params: dict,
) -> None:
    """Generate ORA lollipop plot for one contrast + library."""
    enr = params["enrichment"]
    fdr_threshold = enr["fdr_threshold"]
    top_n = enr["plot_top_n"]

    sig = ora_df[ora_df["adj_pvalue"] < fdr_threshold].copy()
    if sig.empty:
        logging.info("Contrast %s, library %s ORA: no significant terms -- lollipop plot skipped", contrast, library_name)
        return

    sig = sig.nsmallest(top_n, "adj_pvalue")
    sig = sig.sort_values("adj_pvalue", ascending=False)  # best at top of plot

    # x-axis: combined score (Enrichr-style effect size)
    x_vals      = sig["combined_score"].tolist()
    dot_colors  = sig["adj_pvalue"].tolist()
    dot_sizes   = sig["overlap_size"].tolist()

    hover_texts = [
        (f"<b>{row['term_id']}</b><br>"
         f"adj p-value: {row['adj_pvalue']:.3e}<br>"
         f"Odds Ratio: {row['odds_ratio']:.2f}<br>"
         f"Overlap: {row['overlap_size']} genes<br>"
         f"Genes: {row['overlap_genes']}")
        for _, row in sig.iterrows()
    ]

    fig = _make_lollipop_fig(
        terms=sig["term_id"].tolist(),
        x_vals=x_vals,
        dot_colors=dot_colors,
        dot_sizes=dot_sizes,
        x_label="Combined Score",
        title=(f"ORA: {contrast} | {library_name}<br>"
               f"<sup>Pooled up + down significant set (no direction)</sup>"),
        hover_texts=hover_texts,
        colorbar_title="-log10(adj p)",
    )

    # Add vertical line at x=0
    fig.add_vline(x=0, line_width=1, line_color="#888888")

    safe_contrast = contrast.replace("/", "_")
    base = str(outdir / f"{run_id}.{safe_contrast}.{library_name}.ora_lollipop")
    _save_lollipop(fig, base)
    logging.info("ORA lollipop saved: %s.png/.html", base)


def plot_gsea_lollipop(
    gsea_df: pd.DataFrame,
    contrast: str,
    library_name: str,
    run_id: str,
    outdir: Path,
    params: dict,
) -> None:
    """Generate GSEA lollipop plot for one contrast + library."""
    enr = params["enrichment"]
    fdr_threshold = enr["fdr_threshold"]
    top_n = enr["plot_top_n"]

    sig = gsea_df[gsea_df["adj_pvalue"] < fdr_threshold].copy()
    if sig.empty:
        logging.info("Contrast %s, library %s GSEA: no significant terms -- lollipop plot skipped", contrast, library_name)
        return

    # Order by absolute NES, show top_n
    sig = sig.reindex(sig["enrichment_score"].abs().nlargest(top_n).index)
    sig = sig.sort_values("enrichment_score", ascending=True)  # negative NES at bottom

    x_vals     = sig["enrichment_score"].tolist()
    dot_colors = sig["adj_pvalue"].tolist()
    dot_sizes  = sig["overlap_size"].tolist()

    hover_texts = [
        (f"<b>{row['term_id']}</b><br>"
         f"NES: {row['enrichment_score']:.3f}<br>"
         f"FDR: {row['adj_pvalue']:.3e}<br>"
         f"Leading edge: {row['overlap_size']} genes<br>"
         f"Genes: {row['overlap_genes'][:200]}{'...' if len(str(row['overlap_genes'])) > 200 else ''}")
        for _, row in sig.iterrows()
    ]

    fig = _make_lollipop_fig(
        terms=sig["term_id"].tolist(),
        x_vals=x_vals,
        dot_colors=dot_colors,
        dot_sizes=dot_sizes,
        x_label=gsea_axis_label(contrast),
        title=(f"GSEA: {contrast} | {library_name}<br>"
               f"<sup>{gsea_direction_statement(contrast)}</sup>"),
        hover_texts=hover_texts,
        colorbar_title="FDR q-val",
    )

    # Reference line at NES=0
    fig.add_vline(x=0, line_width=1.5, line_color="#444444")

    safe_contrast = contrast.replace("/", "_")
    base = str(outdir / f"{run_id}.{safe_contrast}.{library_name}.gsea_lollipop")
    _save_lollipop(fig, base)
    logging.info("GSEA lollipop saved: %s.png/.html", base)


# ============================================================
# VISUALIZATION: GSEA RUNNING SCORE PLOTS
# ============================================================

def plot_gsea_running_scores(
    gsea_df: pd.DataFrame,
    pre_res: object,
    contrast: str,
    library_name: str,
    run_id: str,
    outdir: Path,
    params: dict,
) -> None:
    """
    Generate GSEA running enrichment score plots for top N significant terms.
    Uses gseapy's built-in gseaplot() (matplotlib, PNG only).
    """
    enr = params["enrichment"]
    fdr_threshold = enr["fdr_threshold"]
    top_n_traces = enr["plot_top_gsea_traces"]

    sig = gsea_df[gsea_df["adj_pvalue"] < fdr_threshold].copy()
    if sig.empty:
        return

    top_terms = sig.nsmallest(top_n_traces, "adj_pvalue")["term_id"].tolist()

    for term in top_terms:
        try:
            term_data = pre_res.results.get(term)
            if term_data is None:
                continue

            # Sanitize term name for use in a filename
            safe_term = term.replace("/", "_").replace(" ", "_").replace(":", "_")
            safe_contrast = contrast.replace("/", "_")
            ofname = str(
                outdir / f"{run_id}.{safe_contrast}.{library_name}.gsea_running_score.{safe_term}.png"
            )

            gseapy.gseaplot(
                term=term,
                hits=term_data["hits"],
                nes=term_data["nes"],
                pval=term_data["pval"],
                fdr=term_data["fdr"],
                RES=term_data["RES"],
                rank_metric=pre_res.ranking,
                ofname=ofname,
                figsize=(8, 5),
            )
            logging.info("GSEA running score plot: %s", ofname)
        except Exception as exc:
            logging.warning("Failed to generate running score plot for term %s: %s", term, exc)


# ============================================================
# SUMMARY TEXT FILE
# ============================================================

def write_summary(
    run_id: str,
    outdir: Path,
    da_df: pd.DataFrame,
    params: dict,
    gmt_paths: list[str],
    library_names: list[str],
    gene_sym_stats: dict[str, dict],
    enrichment_results: pd.DataFrame,
    timestamp: str,
) -> None:
    enr = params["enrichment"]
    contrasts = sorted(da_df["contrast"].unique())
    lines = []

    lines += [
        "=" * 40,
        "ENRICHMENT ANALYSIS SUMMARY",
        "=" * 40,
        "",
        f"Run:              {run_id}",
        f"Date:             {timestamp}",
        "",
    ]

    # --- Input ---
    # Use stats from the first contrast as a proxy for shared protein counts
    first_stats = next(iter(gene_sym_stats.values())) if gene_sym_stats else {}
    n_total    = first_stats.get("n_total", len(da_df) // max(len(contrasts), 1))
    n_unmapped = first_stats.get("n_unmapped", 0)
    n_mapped   = first_stats.get("n_mapped", n_total - n_unmapped)
    n_unique   = first_stats.get("n_unique", n_mapped)

    lines += [
        "-" * 40,
        "INPUT",
        "-" * 40,
        "",
        f"Proteins from Module 04:      {n_total}",
        f"  With gene symbol:           {n_mapped} ({100*n_mapped//max(n_total,1)}%)",
        f"  Unmapped (dropped):         {n_unmapped}",
        f"Unique gene symbols:          {n_unique}",
        f"Background (ORA):             {n_unique} unique gene symbols (detected proteins)",
        "",
    ]

    # --- Libraries ---
    lines += [
        "-" * 40,
        "GENE SET LIBRARIES",
        "-" * 40,
        "",
    ]
    header = f"  {'Library':<15} {'Version':<14} {'GMT file':<55} {'Size filter'}"
    lines.append(header)
    for gmt, lib in zip(gmt_paths, library_names, strict=True):
        lines.append(
            f"  {lib:<15} {_gmt_version(gmt):<14} {Path(gmt).name:<55} "
            f"{enr['min_gene_set_size']}-{enr['max_gene_set_size']} genes"
        )
        # Checksum pins the exact gene-set content; the version token above is
        # parsed from the filename and would survive a rename or a local edit.
        lines.append(f"  {'':<15} sha256: {_file_sha256(gmt)}")
    lines.append("")

    # --- Parameters ---
    lines += [
        "-" * 40,
        "PARAMETERS",
        "-" * 40,
        "",
        f"ORA:              {'enabled' if enr['run_ora'] else 'disabled'}",
        f"GSEA:             {'enabled' if enr['run_gsea'] else 'disabled'}",
        f"GSEA ranking:     {enr['gsea_ranking']}",
        f"GSEA permutations:{enr['gsea_permutations']}",
        f"FDR threshold:    {enr['fdr_threshold']}",
        # ORA and GSEA use different FDR procedures: gseapy.enrich reports BH
        # adjusted p-values, gseapy.prerank reports the permutation-based GSEA
        # FDR q-value (Subramanian et al. 2005). Both are computed per call,
        # and each call is one contrast x one library.
        "Correction (ORA): Benjamini-Hochberg adjusted p-value",
        "Correction (GSEA):GSEA permutation FDR q-value (Subramanian et al. 2005)",
        "FDR scope:        per library, per contrast. Each gene set library is",
        "                  corrected independently, so significant terms pooled",
        "                  across libraries do NOT hold a joint FDR at the",
        "                  threshold above.",
        "",
    ]

    # --- Per-contrast results ---
    for contrast in contrasts:
        n_sig = int(da_df[(da_df["contrast"] == contrast) & (da_df["significant"].eq(True))]["gene_symbol"].notna().sum())
        n_genes = gene_sym_stats.get(contrast, {}).get("n_unique", "?")

        lines += [
            "-" * 40,
            f"RESULTS: {contrast}",
            "-" * 40,
            "",
            f"Direction:  GSEA {gsea_direction_statement(contrast)}.",
            "            ORA uses the pooled up + down significant set (no direction).",
            f"Significant genes (ORA input):  {n_sig} / {n_genes}",
            "",
        ]

        if not enrichment_results.empty:
            cdf = enrichment_results[enrichment_results["contrast"] == contrast]

            if enr["run_ora"]:
                lines.append("ORA results:")
                lines.append(f"  {'Library':<12} {'Sig terms':<12} Top term")
                for lib in library_names:
                    ldf = cdf[(cdf["library"] == lib) & (cdf["analysis_type"] == "ORA")]
                    sig_count = (ldf["adj_pvalue"] < enr["fdr_threshold"]).sum()
                    if sig_count > 0 and not ldf.empty:
                        top = ldf.nsmallest(1, "adj_pvalue").iloc[0]
                        top_str = f"{_truncate_label(top['term_id'], 40)} (adj_p={top['adj_pvalue']:.2e})"
                    else:
                        top_str = "none"
                    lines.append(f"  {lib:<12} {sig_count:<12} {top_str}")
                lines.append("")

            # --- GO clustering summary (all terms counted, not only clustered ones) ---
            if "cluster_status" in cdf.columns:
                lines.extend(_clustering_summary_lines(cdf))

            if enr["run_gsea"]:
                lines.append("GSEA results:")
                lines.append(f"  {'Library':<12} {'Sig terms':<12} Top term (NES)")
                for lib in library_names:
                    ldf = cdf[(cdf["library"] == lib) & (cdf["analysis_type"] == "GSEA")]
                    sig_count = (ldf["adj_pvalue"] < enr["fdr_threshold"]).sum()
                    if sig_count > 0 and not ldf.empty:
                        top = ldf.reindex(ldf["enrichment_score"].abs().nlargest(1).index).iloc[0]
                        top_str = (
                            f"{_truncate_label(top['term_id'], 35)} "
                            f"(NES={top['enrichment_score']:.2f}, FDR={top['adj_pvalue']:.2e})"
                        )
                    else:
                        top_str = "none"
                    lines.append(f"  {lib:<12} {sig_count:<12} {top_str}")
                lines.append("")
        else:
            lines.append("No enrichment results produced.")
            lines.append("")

    out_path = outdir / f"{run_id}.enrichment_summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    logging.info("Summary written: %s", out_path)


# ============================================================
# GO-TERM REDUNDANCY REDUCTION (rrvgo via rpy2)
# ============================================================

# Map MSigDB GO collection prefix -> GO ontology code consumed by rrvgo
_MSIGDB_GO_PREFIXES = {
    'GO_BP': 'BP',
    'GO_MF': 'MF',
    'GO_CC': 'CC',
}

# Semantic similarity threshold for reduceSimMatrix (Rel similarity).
# 0.7 is the rrvgo default. Higher = fewer, larger clusters.
_RRVGO_THRESHOLD = 0.7

# --- cluster_status vocabulary (spec Section 4.13) ---
# Every row gets exactly one value. Only 'clustered' rows carry a cluster_id.
# All other rows are unclustered and therefore represent themselves
# (is_representative=True), so a "representatives only" filter never silently
# drops them; cluster_status says WHY a row was not clustered.
CLUSTER_STATUS_CLUSTERED     = 'clustered'               # member of an rrvgo cluster
CLUSTER_STATUS_NOT_GO        = 'not_go_library'          # REACTOME etc.: no GO DAG, by design
CLUSTER_STATUS_UNRESOLVED    = 'unresolved_name'         # MSigDB name -> GO ID lookup missed
CLUSTER_STATUS_RRVGO_DROPPED = 'dropped_by_rrvgo'        # resolved, but rrvgo dropped it (no IC, no ancestors, all-NA similarity)
CLUSTER_STATUS_TOO_FEW       = 'too_few_terms'           # <2 clusterable terms in its unit
CLUSTER_STATUS_NO_DIRECTION  = 'no_direction'            # GSEA row with NES 0 or missing (cannot be assigned to up/down)
CLUSTER_STATUS_GROUP_ERROR   = 'group_error'             # R raised for this unit
CLUSTER_STATUS_UNAVAILABLE   = 'clustering_unavailable'  # rpy2 or R packages missing for the whole run

CLUSTER_STATUSES = (
    CLUSTER_STATUS_CLUSTERED, CLUSTER_STATUS_NOT_GO, CLUSTER_STATUS_UNRESOLVED,
    CLUSTER_STATUS_RRVGO_DROPPED, CLUSTER_STATUS_TOO_FEW, CLUSTER_STATUS_NO_DIRECTION,
    CLUSTER_STATUS_GROUP_ERROR, CLUSTER_STATUS_UNAVAILABLE,
)

# R-side outcome of one clustering unit (prosift_stage in _RRVGO_R_BLOCK)
_R_STAGE_OK            = 'ok'             # reduceSimMatrix returned clusters
_R_STAGE_TOO_FEW       = 'too_few'        # <2 distinct resolved GO IDs; rrvgo not called
_R_STAGE_SIM_TOO_SMALL = 'sim_too_small'  # calculateSimMatrix kept <2 terms
_R_STAGE_SIM_ERROR     = 'sim_error'      # calculateSimMatrix raised
_R_STAGE_REDUCE_ERROR  = 'reduce_error'   # reduceSimMatrix raised or returned nothing

# R code run once per clustering unit. Kept as a module constant (not inline)
# so the exact shipped R can also be exercised through Rscript where embedded R
# (rpy2) is not available. Inputs (set in globalenv by the caller):
#   prosift_msigdb_ids, prosift_lookup, prosift_scores, prosift_ont,
#   prosift_orgdb, prosift_threshold
# Outputs are plain atomic vectors (no data.frame conversion needed):
#   prosift_stage, prosift_error, prosift_resolved_msig, prosift_resolved_go,
#   prosift_sim_go, prosift_red_go, prosift_red_cluster
_RRVGO_R_BLOCK = r"""
# 0. Reset every output first, so a value can never leak from the previous unit.
prosift_stage         <- "too_few"
prosift_error         <- ""
prosift_resolved_msig <- character()
prosift_resolved_go   <- character()
prosift_sim_go        <- character()
prosift_red_go        <- character()
prosift_red_cluster   <- integer()

suppressMessages({
    library(GO.db)
    library(rrvgo)
    library(AnnotationDbi)
})

# 1. MSigDB name -> GO ID via GO.db TERM table (case-insensitive).
#    MSigDB names are ALL-CAPS while GO.db stores canonical term names with
#    mixed case (e.g., "DNA binding"), so the match must be case-insensitive.
#    Names that do not resolve are reported back as unresolved.
all_terms_df <- AnnotationDbi::select(
    GO.db,
    keys     = AnnotationDbi::keys(GO.db, keytype = "GOID"),
    keytype  = "GOID",
    columns  = c("TERM", "ONTOLOGY")
)
go_df <- all_terms_df[!is.na(all_terms_df$TERM)
                      & all_terms_df$ONTOLOGY == prosift_ont[1], , drop = FALSE]
go_df$TERM_LC <- tolower(go_df$TERM)
go_df <- go_df[!duplicated(go_df$TERM_LC), , drop = FALSE]

match_idx     <- match(tolower(prosift_lookup), go_df$TERM_LC)
resolved_mask <- !is.na(match_idx)
prosift_resolved_msig <- as.character(prosift_msigdb_ids[resolved_mask])
prosift_resolved_go   <- as.character(go_df$GOID[match_idx[resolved_mask]])
resolved_scores       <- prosift_scores[resolved_mask]

if (length(unique(prosift_resolved_go)) >= 2) {
    # 2. One score per GO ID (highest), as rrvgo expects unique GO IDs.
    ord      <- order(-resolved_scores)
    keep_idx <- ord[!duplicated(prosift_resolved_go[ord])]
    uniq_go     <- prosift_resolved_go[keep_idx]
    uniq_scores <- resolved_scores[keep_idx]
    names(uniq_scores) <- uniq_go

    # 3. Semantic similarity. rrvgo silently drops terms without IC, without
    #    ancestors, or with all-NA similarity; prosift_sim_go records survivors.
    sim <- tryCatch(
        rrvgo::calculateSimMatrix(uniq_go, orgdb = prosift_orgdb[1],
                                  ont = prosift_ont[1], method = "Rel"),
        error = function(e) { prosift_error <<- conditionMessage(e); NULL }
    )
    if (is.null(sim)) {
        prosift_stage <- "sim_error"
    } else {
        prosift_sim_go <- as.character(rownames(sim))
        if (nrow(sim) < 2) {
            prosift_stage <- "sim_too_small"
        } else {
            # 4. Hierarchical clustering at the similarity threshold.
            reduced <- tryCatch(
                rrvgo::reduceSimMatrix(sim, scores = uniq_scores,
                                       threshold = prosift_threshold[1],
                                       orgdb = prosift_orgdb[1]),
                error = function(e) { prosift_error <<- conditionMessage(e); NULL }
            )
            if (is.null(reduced) || nrow(reduced) == 0) {
                prosift_stage <- "reduce_error"
                if (prosift_error == "") prosift_error <- "reduceSimMatrix returned no rows"
            } else {
                prosift_red_go      <- as.character(reduced$go)
                prosift_red_cluster <- as.integer(reduced$cluster)
                prosift_stage       <- "ok"
            }
        }
    }
}
"""


def _msigdb_name_to_go_lookup_phrase(term_id: str) -> str:
    """
    Convert an MSigDB GO term_id (e.g. 'GOBP_APOPTOTIC_PROCESS') to the
    lowercase, space-separated phrase used as a GO.db TERM key
    (e.g. 'apoptotic process'). The GOBP_/GOMF_/GOCC_ prefix is stripped
    and underscores become spaces.
    """
    stripped = term_id
    for p in ('GOBP_', 'GOMF_', 'GOCC_'):
        if stripped.startswith(p):
            stripped = stripped[len(p):]
            break
    return stripped.replace('_', ' ').lower()


def _rrvgo_scores(adj_pvalue: pd.Series) -> np.ndarray:
    """
    rrvgo score per term: -log10(adj_pvalue). A missing p-value scores 0; a zero
    p-value (GSEA permutation floor) is clamped to 1e-300 to stay finite.
    """
    raw_p = adj_pvalue.astype(float).to_numpy()
    raw_p = np.where(np.isnan(raw_p), 1.0, raw_p)
    raw_p = np.where(raw_p <= 0, 1e-300, raw_p)
    return -np.log10(raw_p)


def _pick_representatives(members: pd.DataFrame, analysis_type: str) -> dict:
    """
    Return {cluster_local: row index of its representative}.

    The representative is the most significant member. Ties on adj_pvalue are
    common (BH ties, the GSEA permutation floor), so a fixed tie-break makes the
    choice reproducible regardless of input row order:
      1. lowest adj_pvalue (missing last)
      2. GSEA: largest |NES|; ORA: largest overlap_size (missing last)
      3. term_id, alphabetical
    """
    keyed = members.assign(
        _p=members['adj_pvalue'].astype(float),
        _effect=(
            members['enrichment_score'].astype(float).abs()
            if analysis_type == 'GSEA' and 'enrichment_score' in members
            else members['overlap_size'].astype(float)
            if 'overlap_size' in members
            else 0.0
        ),
        _tid=members['term_id'].astype(str),
    )
    ordered = keyed.sort_values(
        ['_p', '_effect', '_tid'], ascending=[True, False, True],
        na_position='last', kind='stable',
    )
    first = ordered.drop_duplicates(subset='cluster_local', keep='first')
    return dict(zip(first['cluster_local'], first.index, strict=True))


def _annotate_cluster_unit(
    unit: pd.DataFrame,
    analysis_type: str,
    r_out: dict | None,
) -> pd.DataFrame:
    """
    Pure-Python core of cluster_go_terms for ONE clustering unit: assign
    cluster_status, a unit-local cluster number, is_representative and
    parent_term to every row, from the R outcome.

    unit:   the unit's rows (term_id, adj_pvalue, enrichment_score/overlap_size).
    r_out:  None when R raised for this unit; otherwise a dict with keys
            stage (one of the _R_STAGE_* values), resolved ({term_id: GO ID}),
            sim_go (set of GO IDs kept by calculateSimMatrix) and
            red ({GO ID: rrvgo cluster number}).

    Returns a DataFrame indexed like `unit` with columns cluster_status,
    cluster_local (Int64, NA unless clustered), is_representative (bool) and
    parent_term (object, NA for representatives and unclustered rows).
    """
    res = pd.DataFrame(index=unit.index)
    res['cluster_status'] = CLUSTER_STATUS_GROUP_ERROR
    res['cluster_local'] = pd.Series(pd.NA, index=unit.index, dtype='Int64')
    res['is_representative'] = True
    res['parent_term'] = pd.Series(pd.NA, index=unit.index, dtype='object')
    if r_out is None:
        return res

    stage = r_out['stage']
    resolved, sim_go, red = r_out['resolved'], r_out['sim_go'], r_out['red']

    # --- 1. Status for every row, from where it fell out of the R pipeline ---
    for idx, term_id in unit['term_id'].items():
        go_id = resolved.get(term_id)
        if go_id is None:
            status = CLUSTER_STATUS_UNRESOLVED
        elif stage == _R_STAGE_TOO_FEW:
            status = CLUSTER_STATUS_TOO_FEW
        elif stage == _R_STAGE_SIM_ERROR:
            status = CLUSTER_STATUS_GROUP_ERROR
        elif go_id not in sim_go:
            status = CLUSTER_STATUS_RRVGO_DROPPED
        elif stage == _R_STAGE_SIM_TOO_SMALL:
            status = CLUSTER_STATUS_TOO_FEW
        elif stage == _R_STAGE_REDUCE_ERROR:
            status = CLUSTER_STATUS_GROUP_ERROR
        elif go_id in red:
            status = CLUSTER_STATUS_CLUSTERED
            res.at[idx, 'cluster_local'] = int(red[go_id])
        else:
            # Kept in the similarity matrix but absent from reduceSimMatrix output.
            status = CLUSTER_STATUS_RRVGO_DROPPED
        res.at[idx, 'cluster_status'] = status

    # --- 2. One representative per cluster; members point to it ---
    clustered = res['cluster_status'] == CLUSTER_STATUS_CLUSTERED
    if clustered.any():
        members = unit.loc[clustered].assign(cluster_local=res.loc[clustered, 'cluster_local'])
        reps = _pick_representatives(members, analysis_type)
        for idx in members.index:
            rep_idx = reps[members.at[idx, 'cluster_local']]
            if idx != rep_idx:
                res.at[idx, 'is_representative'] = False
                res.at[idx, 'parent_term'] = unit.at[rep_idx, 'term_id']
    return res


def _clustering_units(group: pd.DataFrame, analysis_type: str) -> tuple[list, pd.Index]:
    """
    Split one (library, analysis_type, contrast) group into clustering units.

    ORA is a single unit (its gene set is the pooled up + down significant set,
    so terms carry no direction). GSEA is split by NES sign so a cluster never
    mixes enriched-in-numerator and enriched-in-denominator terms; otherwise a
    significant term of one direction could be hidden behind a representative of
    the other. GSEA rows with NES 0 or missing cannot be placed and are returned
    separately.

    Returns (units, no_direction_index): units is a list of (label, DataFrame).
    """
    if analysis_type != 'GSEA':
        return [('all', group)], group.index[:0]
    nes = group['enrichment_score'].astype(float)
    units = [('NES>0', group[nes > 0]), ('NES<0', group[nes < 0])]
    no_direction = group.index[~((nes > 0) | (nes < 0))]
    return units, no_direction


def _run_rrvgo_unit(ro, unit: pd.DataFrame, ont: str, orgdb: str, threshold: float) -> dict:
    """
    Run _RRVGO_R_BLOCK for one unit through rpy2 and return its outputs in the
    r_out shape consumed by _annotate_cluster_unit. Raises on any R error.
    """
    msigdb_ids = unit['term_id'].astype(str).tolist()
    ro.globalenv['prosift_msigdb_ids'] = ro.StrVector(msigdb_ids)
    ro.globalenv['prosift_lookup']     = ro.StrVector(
        [_msigdb_name_to_go_lookup_phrase(t) for t in msigdb_ids])
    ro.globalenv['prosift_scores']     = ro.FloatVector(_rrvgo_scores(unit['adj_pvalue']).tolist())
    ro.globalenv['prosift_ont']        = ro.StrVector([ont])
    ro.globalenv['prosift_orgdb']      = ro.StrVector([orgdb])
    ro.globalenv['prosift_threshold']  = ro.FloatVector([float(threshold)])
    ro.r(_RRVGO_R_BLOCK)

    def _vec(name: str) -> list:
        return list(ro.globalenv[name])

    error = _vec('prosift_error')
    return {
        'stage':    _vec('prosift_stage')[0],
        'error':    error[0] if error else '',
        'resolved': dict(zip(_vec('prosift_resolved_msig'), _vec('prosift_resolved_go'), strict=True)),
        'sim_go':   set(_vec('prosift_sim_go')),
        'red':      dict(zip(_vec('prosift_red_go'), [int(c) for c in _vec('prosift_red_cluster')], strict=True)),
    }


def warn_unclustered_go_groups(enrichment_results: pd.DataFrame) -> list:
    """
    Failure check for the "unclustered rows are their own representative"
    contract: when a GO group has NO clustered rows, every row is
    is_representative=True and a representatives-only view looks valid while
    nothing was collapsed. Log a warning per such (library, analysis_type,
    contrast) group and return the list of groups.
    """
    needed = {'cluster_status', 'library', 'analysis_type', 'contrast'}
    if enrichment_results.empty or not needed <= set(enrichment_results.columns):
        return []
    go = enrichment_results[enrichment_results['library'].isin(_MSIGDB_GO_PREFIXES.keys())]
    unclustered = []
    for key, sub in go.groupby(['library', 'analysis_type', 'contrast'], sort=True, dropna=False):
        if not (sub['cluster_status'] == CLUSTER_STATUS_CLUSTERED).any():
            counts = sub['cluster_status'].value_counts().to_dict()
            logging.warning(
                'GO clustering produced NO clusters for %s/%s/%s (%d terms; %s). '
                'These terms are NOT collapsed: is_representative=True for all of them.',
                *key, len(sub), counts,
            )
            unclustered.append(key)
    return unclustered


def _clustering_summary_lines(cdf: pd.DataFrame) -> list:
    """
    Summary-text block for one contrast's results: per (library, analysis_type),
    ALL tested terms with the clustered count, the MSigDB -> GO ID resolution
    rate (GO only) and a count per cluster_status, so terms left unclustered are
    visible rather than silently omitted. Flags GO groups with no clusters.
    """
    lines = [
        f"GO clustering (rrvgo, Rel similarity, threshold={_RRVGO_THRESHOLD}; "
        "GSEA clustered separately by NES sign):",
    ]
    for (lib, atype), sub in cdf.groupby(["library", "analysis_type"], sort=True):
        n_terms = len(sub)
        status = sub["cluster_status"]
        if lib not in _MSIGDB_GO_PREFIXES:
            lines.append(f"  {lib} {atype:<6} {n_terms:>5} terms: not clustered (not a GO library)")
            continue
        n_clustered = int((status == CLUSTER_STATUS_CLUSTERED).sum())
        n_clusters = int(sub["cluster_id"].nunique())
        # Name resolution is known exactly for rows R finished with (clustered,
        # unresolved, dropped, too_few). It cannot be read from status when R
        # errored (group_error rows may or may not have resolved, while their
        # unit's unresolved rows are still counted), so report it as unavailable
        # rather than print a biased rate. no_direction rows never reach R and
        # are left out of the denominator.
        n_resolution_known = int(status.isin([
            CLUSTER_STATUS_CLUSTERED, CLUSTER_STATUS_UNRESOLVED,
            CLUSTER_STATUS_RRVGO_DROPPED, CLUSTER_STATUS_TOO_FEW,
        ]).sum())
        n_unresolved = int((status == CLUSTER_STATUS_UNRESOLVED).sum())
        if status.isin([CLUSTER_STATUS_GROUP_ERROR, CLUSTER_STATUS_UNAVAILABLE]).any():
            resolved_str = "name resolution rate unavailable (R did not complete)"
        elif n_resolution_known:
            resolved_str = f"{1 - n_unresolved / n_resolution_known:.1%} names resolved to GO IDs"
        else:
            resolved_str = "name resolution not attempted"
        counts = ", ".join(f"{k}={v}" for k, v in status.value_counts().sort_index().items())
        lines.append(
            f"  {lib} {atype:<6} {n_terms:>5} terms: {n_clustered} clustered "
            f"({n_clustered / n_terms:.1%}) -> {n_clusters} clusters; {resolved_str}"
        )
        lines.append(f"  {'':<{len(lib) + 7}} status: {counts}")
        if n_clustered == 0:
            lines.append(
                f"  WARNING: no {lib} {atype} terms were clustered; is_representative=True "
                "for all of them, so a representatives-only view is NOT collapsed."
            )
    lines.append("")
    return lines


def cluster_go_terms(
    enrichment_results: pd.DataFrame,
    threshold: float = _RRVGO_THRESHOLD,
    orgdb: str = 'org.Mm.eg.db',
) -> pd.DataFrame:
    """
    Add cluster_status, cluster_id, is_representative, parent_term columns to the
    enrichment results table using rrvgo's GO semantic similarity clustering.

    Clustering units: one per (library, analysis_type, contrast) for ORA, and one
    per (library, 'GSEA', contrast, NES sign) for GSEA. GO libraries only
    (GO_BP, GO_MF, GO_CC).

    Contract (spec Section 4.13):
      - cluster_status is never null; see CLUSTER_STATUSES.
      - cluster_id is non-null only for 'clustered' rows. It is unique within a
        (library, analysis_type, contrast) group (GSEA up and down clusters are
        numbered in one sequence), not across groups.
      - is_representative is never null. Exactly one row per cluster is True
        (lowest adj_pvalue, fixed tie-break, see _pick_representatives); every
        unclustered row is True because it represents itself. A
        representatives-only view therefore keeps every non-GO and unclustered term.
      - parent_term is the representative's term_id for non-representative
        cluster members; null otherwise.

    Never drops rows. If rpy2 or the R packages are unavailable, every GO row is
    'clustering_unavailable'. Any GO group left with no clusters is warned about
    (warn_unclustered_go_groups).
    """
    # --- Initialize: every row unclustered and its own representative ---
    out = enrichment_results.copy()
    out['cluster_status'] = pd.Series(CLUSTER_STATUS_NOT_GO, index=out.index, dtype='object')
    out['cluster_id'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    out['is_representative'] = pd.Series(True, index=out.index, dtype='boolean')
    out['parent_term'] = pd.Series(pd.NA, index=out.index, dtype='object')

    if out.empty:
        return out

    go_mask = out['library'].isin(_MSIGDB_GO_PREFIXES.keys())
    if not go_mask.any():
        logging.info('No GO-family libraries in results; skipping redundancy reduction.')
        return out
    out.loc[go_mask, 'cluster_status'] = CLUSTER_STATUS_UNAVAILABLE

    # --- Import rpy2 lazily (see module header) ---
    # Deferred to call time so that merely importing this module never starts
    # embedded R. rpy2 objects (ro, importr) are local to this function.
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError:
        logging.warning(
            'rpy2 not available; GO term redundancy reduction skipped. '
            "All GO terms get cluster_status='%s'.", CLUSTER_STATUS_UNAVAILABLE,
        )
        warn_unclustered_go_groups(out)
        return out

    # --- Load rrvgo + GO.db + organism annotation (fail fast) ---
    try:
        importr('rrvgo')
        importr('GO.db')
        importr(orgdb)
    except Exception as exc:
        logging.warning(
            'Failed to load R packages for GO clustering (%s): %s. '
            "All GO terms get cluster_status='%s'.", orgdb, exc, CLUSTER_STATUS_UNAVAILABLE,
        )
        warn_unclustered_go_groups(out)
        return out

    # --- Iterate (library, analysis_type, contrast) groups of GO rows ---
    for (lib, atype, contrast), group in out[go_mask].groupby(
        ['library', 'analysis_type', 'contrast'], sort=False, dropna=False
    ):
        ont = _MSIGDB_GO_PREFIXES[lib]
        units, no_direction = _clustering_units(group, atype)
        out.loc[no_direction, 'cluster_status'] = CLUSTER_STATUS_NO_DIRECTION
        next_cluster_id = 1  # GSEA up and down units share one numbering sequence

        for label, unit in units:
            if unit.empty:
                continue
            # Single-row units still go through R: the R block resolves the name and
            # returns stage 'too_few', so the row gets unresolved_name or too_few_terms.

            # 1. Run R; any R error leaves this unit as group_error.
            try:
                r_out = _run_rrvgo_unit(ro, unit, ont, orgdb, threshold)
            except Exception as exc:
                logging.warning('R-side clustering failed for %s/%s/%s [%s]: %s',
                                lib, atype, contrast, label, exc)
                r_out = None
            if r_out is not None and r_out['error']:
                logging.warning('rrvgo error for %s/%s/%s [%s] (stage %s): %s',
                                lib, atype, contrast, label, r_out['stage'], r_out['error'])

            # 2. Statuses, representatives, parents (pure Python).
            res = _annotate_cluster_unit(unit, atype, r_out)

            # 3. Renumber rrvgo's per-unit cluster numbers into the group's sequence.
            local_ids = sorted(res['cluster_local'].dropna().unique())
            renumber = {old: next_cluster_id + i for i, old in enumerate(local_ids)}
            next_cluster_id += len(local_ids)

            out.loc[unit.index, 'cluster_status'] = res['cluster_status']
            out.loc[unit.index, 'cluster_id'] = res['cluster_local'].map(renumber).astype('Int64')
            out.loc[unit.index, 'is_representative'] = res['is_representative'].astype('boolean')
            out.loc[unit.index, 'parent_term'] = res['parent_term']

            counts = res['cluster_status'].value_counts().to_dict()
            logging.info('GO clustering %s/%s/%s [%s]: %d terms -> %d clusters; statuses %s',
                         lib, atype, contrast, label, len(unit), len(local_ids), counts)

    warn_unclustered_go_groups(out)
    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logging.info("=" * 60)
    logging.info("ProSIFT Module 05 ENRICHMENT: %s", args.run_id)
    logging.info("=" * 60)

    # --- Load inputs ---
    params = load_params(args.params, staged_libraries=args.gene_set_libraries)
    enr = params["enrichment"]

    logging.info("Loading Module 04 results: %s", args.results)
    da_df = pd.read_parquet(args.results)
    contrasts = sorted(da_df["contrast"].unique())
    logging.info("Contrasts: %s | Proteins: %d", contrasts, len(da_df) // len(contrasts))

    gmt_paths    = enr["gene_set_libraries"]
    library_names = [_library_short_name(p) for p in gmt_paths]
    logging.info("Libraries: %s", library_names)
    # Library short names key every per-library output (results rows, plot
    # filenames, the GSEA results used for running-score plots and the
    # protein-term mapping). Two GMTs with the same short name (e.g. two GO_BP
    # releases) would silently overwrite each other, so refuse them.
    duplicates = sorted({n for n in library_names if library_names.count(n) > 1})
    if duplicates:
        logging.error("Duplicate gene set library short name(s) %s from %s. Each GMT "
                      "must map to a distinct library name: run different releases of "
                      "the same collection as separate runs, or rename custom GMTs whose "
                      "names differ only after the first 20 characters.",
                      duplicates, gmt_paths)
        sys.exit(1)

    # --- Per-contrast enrichment ---
    all_enrichment: list[pd.DataFrame] = []
    gene_sym_stats: dict[str, dict] = {}
    gsea_results_by_key: dict[tuple[str, str], object] = {}  # (contrast, lib_name) -> pre_res

    for contrast in contrasts:
        logging.info("--- Contrast: %s ---", contrast)

        # Gene symbol preparation (shared across libraries for this contrast)
        contrast_df, stats = prepare_gene_symbols(da_df, contrast)
        gene_sym_stats[contrast] = stats

        sig_genes     = contrast_df[contrast_df["significant"].eq(True)]["gene_symbol"].tolist()
        background_genes = contrast_df["gene_symbol"].tolist()

        ranked_series = build_ranked_series(
            contrast_df,
            ranking=enr["gsea_ranking"],
        )

        for gmt_path, lib_name in zip(gmt_paths, library_names, strict=True):
            logging.info("Library: %s", lib_name)
            # Release token for the gene_set_version column (provenance); set
            # per GMT file here rather than mapped by library name afterwards.
            gmt_version = _gmt_version(gmt_path)

            # ORA
            if enr["run_ora"]:
                ora_df = run_ora(
                    sig_genes=sig_genes,
                    background_genes=background_genes,
                    gmt_path=gmt_path,
                    library_name=lib_name,
                    contrast=contrast,
                    params=params,
                )
                if not ora_df.empty:
                    ora_df["gene_set_version"] = gmt_version
                    all_enrichment.append(ora_df)
                    plot_ora_lollipop(
                        ora_df=ora_df,
                        contrast=contrast,
                        library_name=lib_name,
                        run_id=args.run_id,
                        outdir=outdir,
                        params=params,
                    )

            # GSEA
            if enr["run_gsea"]:
                gsea_df, pre_res = run_gsea(
                    ranked_series=ranked_series,
                    gmt_path=gmt_path,
                    library_name=lib_name,
                    contrast=contrast,
                    params=params,
                )
                if not gsea_df.empty:
                    gsea_df["gene_set_version"] = gmt_version
                    all_enrichment.append(gsea_df)
                    gsea_results_by_key[(contrast, lib_name)] = pre_res
                    plot_gsea_lollipop(
                        gsea_df=gsea_df,
                        contrast=contrast,
                        library_name=lib_name,
                        run_id=args.run_id,
                        outdir=outdir,
                        params=params,
                    )
                    if pre_res is not None:
                        plot_gsea_running_scores(
                            gsea_df=gsea_df,
                            pre_res=pre_res,
                            contrast=contrast,
                            library_name=lib_name,
                            run_id=args.run_id,
                            outdir=outdir,
                            params=params,
                        )

    # --- Combine enrichment results ---
    if all_enrichment:
        enrichment_results = pd.concat(all_enrichment, ignore_index=True)
        # Enforce schema dtypes
        enrichment_results["gene_set_size"] = enrichment_results["gene_set_size"].astype("Int64")
        enrichment_results["overlap_size"]  = enrichment_results["overlap_size"].astype("Int64")
    else:
        logging.warning("No enrichment results produced for any contrast or library.")
        enrichment_results = pd.DataFrame(columns=[
            "term_id", "term_name", "library", "analysis_type", "contrast",
            "pvalue", "adj_pvalue", "enrichment_score", "odds_ratio",
            "combined_score", "gene_set_size", "overlap_size", "overlap_genes",
            "gene_set_version", "cluster_status", "cluster_id", "is_representative",
            "parent_term",
        ])
    # Same gene_set_version dtype on the empty and populated paths (other
    # columns in an empty table are still untyped in Parquet; pre-existing).
    enrichment_results["gene_set_version"] = enrichment_results["gene_set_version"].astype("string")

    # --- GO-term redundancy reduction (rrvgo via rpy2) ---
    # Adds cluster_status, cluster_id, is_representative, parent_term. Non-GO
    # libraries (REACTOME, KEGG, HALLMARK) are 'not_go_library' and, like every
    # unclustered term, their own representative (spec Section 4.13).
    logging.info("Running GO term redundancy reduction (rrvgo)...")
    enrichment_results = cluster_go_terms(enrichment_results)

    # --- Protein-term mapping ---
    logging.info("Building protein-term mapping table...")
    mapping_df = build_protein_term_mapping(
        da_df=da_df,
        gmt_paths=gmt_paths,
        library_names=library_names,
        enrichment_results=enrichment_results,
        gsea_results_by_key=gsea_results_by_key,
    )

    # --- Stable row order ---
    # gseapy orders GSEA terms by NES, and terms with tied NES can come back in
    # either order between runs. Sort on a unique key so the written files are
    # byte-identical across runs (Module 07 and the frontend order rows in SQL,
    # so no consumer relies on the previous order).
    enrichment_results = enrichment_results.sort_values(
        ["contrast", "library", "analysis_type", "term_id"], kind="mergesort",
    ).reset_index(drop=True)

    # --- Write outputs ---
    results_parquet = outdir / f"{args.run_id}.enrichment_results.parquet"
    results_csv     = outdir / f"{args.run_id}.enrichment_results.csv"
    mapping_parquet = outdir / f"{args.run_id}.protein_term_mapping.parquet"

    enrichment_results.to_parquet(results_parquet, index=False)
    enrichment_results.to_csv(results_csv, index=False)
    mapping_df.to_parquet(mapping_parquet, index=False)
    logging.info("Enrichment results: %s (%d rows)", results_parquet, len(enrichment_results))
    logging.info("Protein-term mapping: %s (%d rows)", mapping_parquet, len(mapping_df))

    write_summary(
        run_id=args.run_id,
        outdir=outdir,
        da_df=da_df,
        params=params,
        gmt_paths=gmt_paths,
        library_names=library_names,
        gene_sym_stats=gene_sym_stats,
        enrichment_results=enrichment_results,
        timestamp=timestamp,
    )

    logging.info("Module 05 ENRICHMENT complete.")


if __name__ == "__main__":
    main()
