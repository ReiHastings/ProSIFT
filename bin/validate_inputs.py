#!/usr/bin/env python3
# title: validate_inputs.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-03-26
# last modified: 2026-09-29
#
# purpose:
#   Module 01, Processes 4.1-4.3. Validates the abundance matrix and metadata
#   file for a single ProSIFT run, then cross-validates sample IDs between them.
#   Produces validated Parquet outputs containing only the matched working set of
#   samples, and a partial validation report (to be concatenated with reports from
#   subsequent processes). Also records the sign convention of every contrast
#   and warns when the run_id names its groups in the opposite order from a
#   contrast (a labeling hazard: the run name then reads backwards).
#
# inputs:
#   --abundance  : master abundance CSV (protein_id + abundance + peptide_count cols)
#   --metadata   : per-run metadata CSV (sample_id + group column)
#   --params     : per-run params.yml
#   --run-id     : string identifier used to name output files
#   --outdir     : output directory (default: current working directory)
#
# outputs:
#   {outdir}/{run_id}.validated_matrix.parquet
#   {outdir}/{run_id}.validated_metadata.parquet
#   {outdir}/{run_id}.validation_report_part1.txt
#
# usage example:
#   python bin/validate_inputs.py \
#       --abundance prosift_inputs/CTX_synaptosome_abundance.csv \
#       --metadata  prosift_inputs/CTXcyto_WT_vs_CTXcyto_KO/CTXcyto_WT_vs_CTXcyto_KO_metadata.csv \
#       --params    prosift_inputs/CTXcyto_WT_vs_CTXcyto_KO/CTXcyto_WT_vs_CTXcyto_KO_params.yml \
#       --run-id    CTXcyto_WT_vs_CTXcyto_KO \
#       --outdir    results/CTXcyto_WT_vs_CTXcyto_KO/validation

import argparse
import os
import re
import sys
from datetime import datetime

import pandas as pd
import yaml

# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ProSIFT Module 01 -- validate abundance matrix and metadata.'
    )
    parser.add_argument('--abundance', required=True,
                        help='Path to master abundance CSV/TSV')
    parser.add_argument('--metadata', required=True,
                        help='Path to per-run metadata CSV/TSV')
    parser.add_argument('--params', required=True,
                        help='Path to per-run params.yml')
    parser.add_argument('--run-id', required=True, dest='run_id',
                        help='Run identifier for output file naming')
    parser.add_argument('--outdir', default='.',
                        help='Output directory (default: current directory)')
    return parser.parse_args()


# ============================================================
# Config loading
# ============================================================

def load_params(params_path: str) -> dict:
    '''Load and return params.yml as a dict.'''
    with open(params_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def infer_separator(filepath: str) -> str:
    '''Return "," for .csv, "\t" for .tsv/.txt, else try to sniff.'''
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        return ','
    if ext in ('.tsv', '.txt'):
        return '\t'
    # sniff first line
    with open(filepath, encoding='utf-8-sig') as f:
        first = f.readline()
    return '\t' if first.count('\t') > first.count(',') else ','


# ============================================================
# Report helpers
# ============================================================

class Report:
    '''Accumulates validation report lines; writes to file at the end.

    If `report_path` is supplied, `error_exit` will flush the accumulated
    lines to that path before raising SystemExit so the user gets a partial
    report on failure.
    '''

    def __init__(self, run_id: str, report_path: str | None = None):
        self.run_id = run_id
        self._report_path = report_path
        self._lines: list[str] = []
        self._warnings: list[str] = []

    def section(self, title: str) -> None:
        self._lines.append('')
        self._lines.append('=' * 64)
        self._lines.append(title)
        self._lines.append('=' * 64)

    def line(self, text: str = '') -> None:
        self._lines.append(text)

    def warn(self, text: str) -> None:
        msg = f'  WARNING: {text}'
        self._lines.append(msg)
        self._warnings.append(text)

    def ok(self, text: str) -> None:
        self._lines.append(f'  OK: {text}')

    def info(self, text: str) -> None:
        self._lines.append(f'  {text}')

    def error_exit(self, text: str) -> None:
        '''Log the error to the report, flush it to disk, then exit non-zero.'''
        self._lines.append(f'  ERROR: {text}')
        self._lines.append('')
        self._lines.append('Validation failed. Pipeline stopped.')
        # Flush partial report so the user has diagnostic context on failure.
        if self._report_path is not None:
            try:
                parent = os.path.dirname(self._report_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self.write(self._report_path)
            except OSError as exc:
                print(
                    f'WARNING: could not write partial report to '
                    f'{self._report_path}: {exc}',
                    file=sys.stderr,
                )
        print(f'ERROR [{self.run_id}]: {text}', file=sys.stderr)
        raise SystemExit(1)

    def write(self, outpath: str) -> None:
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self._lines))
            f.write('\n')

    @property
    def warning_count(self) -> int:
        return len(self._warnings)


# ============================================================
# Process 4.1 -- Validate Matrix
# ============================================================

def validate_matrix(
    abundance_path: str,
    params: dict,
    report: Report
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Read and validate the abundance matrix.

    Returns (matrix, nonpositive_mask):
      matrix           -- DataFrame with the protein_id column plus all
                          abundance and peptide_count columns. The protein_id
                          column contains the representative accession (first
                          token of semicolon-delimited groups).
      nonpositive_mask -- boolean DataFrame (protein_id + one column per
                          abundance column) marking cells that were <= 0 on raw
                          data and converted to NaN. All-False for non-raw data.

    Raises SystemExit on any hard validation failure.
    '''
    report.section('PROCESS 4.1 -- VALIDATE MATRIX')

    # --- Read file ---
    sep = infer_separator(abundance_path)
    try:
        df = pd.read_csv(abundance_path, sep=sep, encoding='utf-8-sig',
                         low_memory=False)
    except Exception as exc:
        report.error_exit(f'Cannot parse abundance file: {exc}')

    report.ok(f'File parsed: {os.path.basename(abundance_path)}')
    report.info(f'{len(df)} rows, {len(df.columns)} columns')

    # --- Protein ID column ---
    id_col: str = params['input']['protein_id_column']
    if id_col not in df.columns:
        report.error_exit(
            f'protein_id_column "{id_col}" not found in abundance file. '
            f'Columns present: {list(df.columns)}'
        )
    report.ok(f'Protein ID column found: "{id_col}"')

    # --- Extract representative accession from semicolon-delimited groups ---
    original_ids = df[id_col].astype(str)
    df[id_col] = original_ids.str.split(';').str[0].str.strip()
    multi_member = (original_ids.str.contains(';', na=False)).sum()
    if multi_member > 0:
        report.info(
            f'{multi_member} protein group rows with semicolon-delimited '
            f'accessions -- first accession used as representative ID.'
        )

    # --- Duplicate protein IDs ---
    dupes = df[id_col].duplicated(keep=False)
    if dupes.any():
        dupe_ids = df.loc[dupes, id_col].unique().tolist()[:10]
        report.error_exit(
            f'{dupes.sum()} duplicate protein IDs after representative '
            f'accession extraction. Examples: {dupe_ids}'
        )
    report.ok(f'{len(df)} unique protein IDs')

    # --- Identify abundance and peptide count columns ---
    abund_prefix: str = params['input'].get('abundance_prefix', '')
    pep_prefix: str | None = params['input'].get('peptide_count_prefix') or None

    non_id_cols = [c for c in df.columns if c != id_col]

    if pep_prefix:
        pep_cols = [c for c in non_id_cols if c.startswith(pep_prefix)]
        abund_cols = [c for c in non_id_cols if c.startswith(abund_prefix)
                      and not c.startswith(pep_prefix)]
    else:
        pep_cols = []
        abund_cols = [c for c in non_id_cols if c.startswith(abund_prefix)]

    if not abund_cols:
        report.error_exit(
            f'No abundance columns found with prefix "{abund_prefix}". '
            f'Non-ID columns present: {non_id_cols[:10]}'
        )
    report.ok(f'{len(abund_cols)} abundance columns identified '
              f'(prefix: "{abund_prefix}")')

    # --- Abundance type (required; must not be guessed) ---
    # abundance_type governs the non-positive handling below: on 'raw' data a
    # value <= 0 is not a valid intensity and is treated as missing, whereas on
    # 'log2'/'normalized' data 0 is a legitimate value. Because that decision is
    # destructive, the type must be declared explicitly -- no silent default.
    abundance_type_raw = params['input'].get('abundance_type')
    if abundance_type_raw is None:
        report.error_exit(
            'input.abundance_type is required (one of: raw, log2, normalized). '
            'Set it explicitly in params.yml -- ProSIFT will not guess the '
            'scale, because non-positive handling depends on it.'
        )
    abundance_type = str(abundance_type_raw).lower()
    if abundance_type not in {'raw', 'log2', 'normalized'}:
        report.error_exit(
            f"Unknown input.abundance_type '{abundance_type_raw}'. "
            f"Valid options: raw, log2, normalized."
        )

    # --- Abundance columns: numeric or NA ---
    bad_abund: list[str] = []
    for col in abund_cols:
        coerced = pd.to_numeric(df[col], errors='coerce')
        # if coercion introduced NAs that weren't already NA, the column has
        # non-numeric, non-empty values
        original_na = df[col].isna()
        new_na = coerced.isna()
        spurious_na = new_na & ~original_na
        if spurious_na.any():
            bad_abund.append(col)
    if bad_abund:
        report.error_exit(
            f'{len(bad_abund)} abundance column(s) contain non-numeric values: '
            f'{bad_abund[:5]}'
        )
    report.ok('All abundance columns are numeric or NA')

    # --- Convert abundance columns to float ---
    df[abund_cols] = df[abund_cols].apply(pd.to_numeric, errors='coerce')

    # --- Completely empty rows (all abundance values NA) ---
    # Hard-stop per Module 01 spec Section 4.1: rows with no measurements
    # carry no information and almost always indicate an upstream extraction
    # or export error.
    all_na_rows = df[abund_cols].isna().all(axis=1)
    if all_na_rows.any():
        report.error_exit(
            f'{int(all_na_rows.sum())} protein row(s) have no abundance '
            f'values across any sample. Remove these rows from the input or '
            f'investigate the upstream export before re-running.'
        )

    # --- Completely empty columns ---
    # Hard-stop per Module 01 spec Section 4.1: an entirely-NA sample column
    # represents a user data-prep error (wrong column included, missing data,
    # corrupted upload) and must be fixed before downstream analysis.
    all_na_cols = [c for c in abund_cols if df[c].isna().all()]
    if all_na_cols:
        report.error_exit(
            f'{len(all_na_cols)} abundance column(s) are entirely NA: '
            f'{all_na_cols}. Remove or correct these columns before re-running.'
        )

    # --- Non-positive abundance handling (raw only) ---
    # Policy decision (2026-07-08): on raw intensity data a value <= 0 means
    # not-quantified (no valid peak area / below detection), so treat it as
    # missing. Preserve zeros for log2/normalized where 0 is a legitimate value.
    # We record a provenance mask so downstream imputation can treat these as
    # MNAR (below-detection) rather than random dropout. Runs AFTER the empty
    # checks above (which catch originally-blank rows/cols) so that emptiness
    # CREATED by this conversion is handled separately below.
    if abundance_type == 'raw':
        # NaN <= 0 evaluates to False, so already-missing cells are not marked.
        nonpos_mask_bool = df[abund_cols] <= 0
        n_nonpos = int(nonpos_mask_bool.to_numpy().sum())
        if n_nonpos > 0:
            df[abund_cols] = df[abund_cols].mask(nonpos_mask_bool)
            report.warn(
                f'{n_nonpos} non-positive abundance value(s) (<= 0) on raw data '
                f'converted to NaN (treated as below-detection / missing).'
            )
            # Per-column detail, capped so a wide matrix does not flood the report.
            per_col = nonpos_mask_bool.sum()
            affected = per_col[per_col > 0]
            for col, n in affected.head(10).items():
                report.info(f'  {int(n)} converted in "{col}"')
            if len(affected) > 10:
                report.info(f'  ... and {len(affected) - 10} more column(s)')

            # Emptiness created by the conversion is handled asymmetrically:
            #   sample column now fully NaN -> failed sample -> hard-stop
            #   protein row now fully NaN   -> below detection everywhere ->
            #                                  warn; detection filter drops it.
            new_all_na_cols = [c for c in abund_cols if df[c].isna().all()]
            if new_all_na_cols:
                report.error_exit(
                    f'{len(new_all_na_cols)} sample column(s) became entirely '
                    f'NaN after non-positive conversion: {new_all_na_cols}. A '
                    f'sample with no usable measurements cannot be analyzed; '
                    f'investigate the run before proceeding.'
                )
            new_all_na_rows = int(df[abund_cols].isna().all(axis=1).sum())
            if new_all_na_rows > 0:
                report.warn(
                    f'{new_all_na_rows} protein(s) became entirely NaN after '
                    f'non-positive conversion; these will be dropped as ABSENT '
                    f'at the detection filter.'
                )
        else:
            report.ok('No non-positive abundance values on raw data.')
    else:
        # log2 / normalized: zeros are real values; nothing is converted.
        nonpos_mask_bool = pd.DataFrame(False, index=df.index, columns=abund_cols)
        report.info(
            f'abundance_type={abundance_type}: zeros preserved as valid values '
            f'(non-positive conversion applies to raw data only).'
        )

    # --- Minimum protein count ---
    if len(df) < 2:
        report.error_exit(
            f'Abundance matrix contains fewer than 2 proteins ({len(df)} rows). '
            f'Cannot proceed.'
        )

    # --- Peptide count columns (if specified) ---
    if pep_prefix:
        if not pep_cols:
            report.error_exit(
                f'peptide_count_prefix "{pep_prefix}" is set but no matching '
                f'columns found in abundance file.'
            )
        report.ok(f'{len(pep_cols)} peptide count columns identified '
                  f'(prefix: "{pep_prefix}")')

        # sample IDs embedded in peptide count column names must match
        # those embedded in abundance column names
        abund_sample_ids = {c[len(abund_prefix):] for c in abund_cols}
        pep_sample_ids = {c[len(pep_prefix):] for c in pep_cols}
        missing_in_pep = abund_sample_ids - pep_sample_ids
        extra_in_pep = pep_sample_ids - abund_sample_ids
        if missing_in_pep:
            report.error_exit(
                f'Peptide count columns missing for sample(s) present in '
                f'abundance columns: {sorted(missing_in_pep)[:10]}'
            )
        if extra_in_pep:
            report.warn(
                f'Peptide count columns exist for sample(s) not present in '
                f'abundance columns (will be ignored): {sorted(extra_in_pep)[:10]}'
            )

        # peptide count columns: numeric (int or float), zero, or NA
        bad_pep: list[str] = []
        for col in pep_cols:
            coerced = pd.to_numeric(df[col], errors='coerce')
            original_na = df[col].isna()
            new_na = coerced.isna()
            spurious_na = new_na & ~original_na
            if spurious_na.any():
                bad_pep.append(col)
        if bad_pep:
            report.error_exit(
                f'{len(bad_pep)} peptide count column(s) contain non-numeric '
                f'values: {bad_pep[:5]}'
            )
        df[pep_cols] = df[pep_cols].apply(pd.to_numeric, errors='coerce')
        report.ok('All peptide count columns are numeric or NA')

        # Keep peptide counts consistent with the abundance conversion above:
        # a cell whose abundance became NaN has, by the pipeline convention
        # (DEqMS summarize_peptide_counts: count 0 <-> no detection), no
        # supporting peptides. Zero those counts so the count-based variance
        # weighting is not fed a peptide count for a value that no longer exists.
        if nonpos_mask_bool.to_numpy().any():
            n_pep_zeroed = 0
            for abund_col in abund_cols:
                sid = abund_col[len(abund_prefix):]
                pep_col = f'{pep_prefix}{sid}'
                if pep_col in df.columns:
                    cells = nonpos_mask_bool[abund_col]
                    n_pep_zeroed += int(cells.sum())
                    df.loc[cells, pep_col] = 0
            if n_pep_zeroed > 0:
                report.info(
                    f'{n_pep_zeroed} peptide count(s) set to 0 to match '
                    f'non-positive abundance conversion.'
                )
    else:
        report.info('No peptide_count_prefix specified -- DEqMS weighting '
                    'will not be available; falling back to limma.')

    # --- Return subset of columns only ---
    keep_cols = [id_col, *abund_cols, *pep_cols]
    df = df[keep_cols].copy()

    # --- Build provenance mask (protein_id + one bool column per abundance
    #     column, using the SAME prefixed names as the matrix so a consumer can
    #     align by direct column intersection). Rows align with df; write_outputs
    #     reindexes to the post-cross-validate matrix. ---
    nonpositive_mask = nonpos_mask_bool.copy()
    nonpositive_mask.insert(0, id_col, df[id_col].values)

    report.line()
    report.info(f'Matrix validation complete: {len(df)} proteins, '
                f'{len(abund_cols)} abundance columns'
                + (f', {len(pep_cols)} peptide count columns' if pep_cols else ''))
    return df, nonpositive_mask


# ============================================================
# Process 4.2 -- Validate Metadata
# ============================================================

def validate_metadata(
    metadata_path: str,
    params: dict,
    report: Report
) -> pd.DataFrame:
    '''
    Read and validate the metadata file.

    Returns a DataFrame with at least sample_id and the group column.
    Raises SystemExit on any hard validation failure.
    '''
    report.section('PROCESS 4.2 -- VALIDATE METADATA')

    sep = infer_separator(metadata_path)
    try:
        meta = pd.read_csv(metadata_path, sep=sep, encoding='utf-8-sig')
    except Exception as exc:
        report.error_exit(f'Cannot parse metadata file: {exc}')

    report.ok(f'File parsed: {os.path.basename(metadata_path)}')
    report.info(f'{len(meta)} rows, {len(meta.columns)} columns')

    # --- sample_id column ---
    # The metadata must have a sample_id column. Its name is always "sample_id"
    # (prepare_prosift_input.py writes it that way).
    if 'sample_id' not in meta.columns:
        report.error_exit(
            f'"sample_id" column not found in metadata. '
            f'Columns present: {list(meta.columns)}'
        )

    # --- Duplicate sample IDs ---
    dupes = meta['sample_id'].duplicated(keep=False)
    if dupes.any():
        report.error_exit(
            f'{dupes.sum()} duplicate sample_id values in metadata: '
            f'{meta.loc[dupes, "sample_id"].tolist()}'
        )
    report.ok(f'{len(meta)} unique sample IDs')

    # --- Group column ---
    group_col: str = params['design']['group_column']
    if group_col not in meta.columns:
        report.error_exit(
            f'group_column "{group_col}" not found in metadata. '
            f'Columns present: {list(meta.columns)}'
        )
    groups = meta[group_col].dropna().unique()
    if len(groups) < 2:
        report.error_exit(
            f'group_column "{group_col}" contains fewer than 2 distinct groups '
            f'(found: {list(groups)}). Cannot run a comparison.'
        )
    report.ok(f'Group column "{group_col}" found: {sorted(str(g) for g in groups)}')

    # --- Covariate columns ---
    covariates: list[str] = params['design'].get('covariates') or []
    for cov in covariates:
        if cov not in meta.columns:
            report.error_exit(
                f'Covariate column "{cov}" specified in params but not found '
                f'in metadata. Columns present: {list(meta.columns)}'
            )
    if covariates:
        report.ok(f'Covariate column(s) found: {covariates}')

    # --- Batch column ---
    batch_col: str | None = params['design'].get('batch_column') or None
    if batch_col:
        if batch_col not in meta.columns:
            report.error_exit(
                f'batch_column "{batch_col}" specified in params but not found '
                f'in metadata. Columns present: {list(meta.columns)}'
            )
        report.ok(f'Batch column "{batch_col}" found')

    report.line()
    report.info(f'Metadata validation complete: {len(meta)} samples, '
                f'{len(groups)} groups')
    return meta


# ============================================================
# Contrast orientation check (labeling hazard, warn only)
# ============================================================

def _split_vs(label: str) -> tuple[str, str] | None:
    '''Split on the first '_vs_' (Module 04's rule); None if not splittable.'''
    idx = label.find('_vs_')
    if idx <= 0 or idx + 4 >= len(label):
        return None
    return label[:idx], label[idx + 4:]


# Name segments inside one '_'-separated run_id token: an upper-case acronym
# directly followed by lower case splits off ('CTXcyto' -> CTX, cyto), a
# Capitalised word stays whole ('Area2' -> Area, 2), digits split off
# ('WT2' -> WT, 2). Keeping 'Area' whole is what stops a one-letter group 'A'
# from matching it.
_NAME_SEGMENT = re.compile(r'[A-Z]{2,}(?=[a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+')


def _side_group(side: str, groups: tuple[str, str]) -> str | None:
    '''
    Return which of the two contrast groups one side of a run_id names.

    Case-insensitive. A group equal to a whole '_' token scores 2, equal to a
    segment of a token scores 1. The best-scoring group wins; a tie (both or
    neither mentioned) returns None. Scoring whole tokens above segments
    resolves nested names: in 'WT2', group 'WT2' (token) beats 'WT' (segment).
    '''
    tokens = [t for t in side.split('_') if t]
    whole = {t.casefold() for t in tokens}
    parts = {seg.casefold() for t in tokens for seg in _NAME_SEGMENT.findall(t)}

    def score(g: str) -> int:
        g = g.casefold()
        return 2 if g in whole else (1 if g in parts else 0)

    a, b = (score(g) for g in groups)
    if a == b:
        return None
    return groups[0] if a > b else groups[1]


def run_id_order_reversed(run_id: str, numerator: str, denominator: str) -> bool:
    '''
    Return True when run_id names the contrast groups in the reverse order.

    run_id is a free-form label, so each side of its first '_vs_' is matched
    to a group by name parts (see _side_group), not raw substrings: 'A' does
    not match 'Area2', but 'CTX' matches 'CTXcyto'. The check fires only when
    the LEFT side unambiguously names the denominator AND the RIGHT side the
    numerator. Anything it cannot interpret returns False, so it is designed
    to miss an odd name rather than raise a false warning.

    Example: run_id 'CTXcyto_WT_vs_CTXcyto_KO' with contrast 'KO_vs_WT' -> True.
    '''
    sides = _split_vs(run_id)
    if sides is None:
        return False
    groups = (numerator, denominator)
    left, right = sides
    return (_side_group(left, groups) == denominator
            and _side_group(right, groups) == numerator)


def check_contrast_orientation(run_id: str, params: dict, report: Report) -> None:
    '''
    Record how each contrast's sign reads, and warn on a reversed run_id.

    Module 04 fits 'numerator - denominator' for each 'numerator_vs_denominator'
    entry in design.contrasts, so positive log2 FC / NES = higher in the
    numerator. The run_id plays no role in the statistics, but readers often take
    its group order as the direction; a reversed order is a warning, not an
    error, because the results are correct and only the label misleads.
    Malformed contrasts are left to Module 04, which rejects them with context.
    '''
    report.section('CONTRAST ORIENTATION')
    # 'design:' present but empty parses to None, so guard both levels.
    contrasts = (params.get('design') or {}).get('contrasts') or []
    for contrast in contrasts:
        parts = _split_vs(str(contrast))
        if parts is None:
            report.info(f'{contrast}: not in numerator_vs_denominator form '
                        '(Module 04 will reject it)')
            continue
        numerator, denominator = parts
        report.info(f'{contrast}: positive log2 FC / NES = higher in '
                    f'{numerator} than {denominator}')
        if run_id_order_reversed(run_id, numerator, denominator):
            msg = (
                f"run_id '{run_id}' names the groups in the opposite order from "
                f"contrast '{contrast}'. Positive log2 FC / NES means higher in "
                f"{numerator}, NOT in the first group of the run name. Results "
                f"are correct; rename the run (numerator first) or check every "
                f"directional claim against the contrast."
            )
            report.warn(msg)
            print(f'WARNING [{run_id}]: {msg}', file=sys.stderr)


# ============================================================
# Process 4.3 -- Cross-Validate Samples
# ============================================================

def cross_validate(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    params: dict,
    report: Report
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Match sample IDs between the abundance matrix and the metadata.

    Returns (matrix_subset, metadata_subset) containing only the working set
    of matched samples. Raises SystemExit if no samples match or if any group
    falls below qc.min_samples_per_group.
    '''
    report.section('PROCESS 4.3 -- CROSS-VALIDATE SAMPLES')

    id_col: str = params['input']['protein_id_column']
    abund_prefix: str = params['input'].get('abundance_prefix', '')
    pep_prefix: str | None = params['input'].get('peptide_count_prefix') or None
    group_col: str = params['design']['group_column']
    min_per_group: int = params['qc'].get('min_samples_per_group', 2)

    # Derive sample IDs from abundance column names by stripping the prefix
    abund_cols = [c for c in matrix.columns if c != id_col
                  and (not pep_prefix or not c.startswith(pep_prefix))]
    matrix_sample_ids: set[str] = {c[len(abund_prefix):] for c in abund_cols}
    meta_sample_ids: set[str] = set(metadata['sample_id'].astype(str))

    matched = matrix_sample_ids & meta_sample_ids
    only_in_matrix = matrix_sample_ids - meta_sample_ids
    only_in_meta = meta_sample_ids - matrix_sample_ids

    report.info(f'Abundance matrix samples: {len(matrix_sample_ids)}')
    report.info(f'Metadata samples:         {len(meta_sample_ids)}')
    report.info(f'Matched (working set):    {len(matched)}')

    if only_in_matrix:
        report.warn(
            f'{len(only_in_matrix)} sample(s) in abundance matrix but not '
            f'in metadata -- dropped from analysis: '
            f'{sorted(only_in_matrix)}'
        )
    if only_in_meta:
        report.warn(
            f'{len(only_in_meta)} sample(s) in metadata but not in '
            f'abundance matrix -- check for typos or missing data: '
            f'{sorted(only_in_meta)}'
        )

    if not matched:
        report.error_exit(
            'No sample IDs match between the abundance matrix and metadata. '
            'Check that sample IDs in the metadata match the column names in '
            'the abundance matrix (after stripping abundance_prefix).'
        )

    # --- Subset matrix to matched samples ---
    keep_abund_cols = [c for c in abund_cols
                       if c[len(abund_prefix):] in matched]
    if pep_prefix:
        pep_cols = [c for c in matrix.columns if c.startswith(pep_prefix)]
        keep_pep_cols = [c for c in pep_cols
                         if c[len(pep_prefix):] in matched]
    else:
        keep_pep_cols = []

    matrix_out = matrix[[id_col, *keep_abund_cols, *keep_pep_cols]].copy()

    # --- Subset metadata to matched samples ---
    meta_out = metadata[metadata['sample_id'].astype(str).isin(matched)].copy()

    # --- Check group sizes ---
    group_counts = meta_out.groupby(group_col)['sample_id'].count()
    report.line()
    report.info('Samples per group (working set):')
    for grp, n in group_counts.items():
        report.info(f'  {grp}: {n}')

    undersized = group_counts[group_counts < min_per_group]
    if not undersized.empty:
        report.error_exit(
            f'The following group(s) have fewer than {min_per_group} samples '
            f'after cross-validation: '
            f'{dict(undersized)}. '
            f'Adjust qc.min_samples_per_group or provide more samples.'
        )

    report.line()
    report.ok(f'Cross-validation complete. Working set: {len(matched)} samples, '
              f'{matrix_out.shape[0]} proteins.')
    return matrix_out, meta_out


# ============================================================
# Output writing
# ============================================================

def write_outputs(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    nonpositive_mask: pd.DataFrame,
    report: Report,
    run_id: str,
    outdir: str,
    report_path: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    matrix_path = os.path.join(outdir, f'{run_id}.validated_matrix.parquet')
    meta_path = os.path.join(outdir, f'{run_id}.validated_metadata.parquet')
    mask_path = os.path.join(outdir, f'{run_id}.nonpositive_mask.parquet')

    matrix.to_parquet(matrix_path, index=False)
    metadata.to_parquet(meta_path, index=False)

    # Align the mask to the final (post-cross-validate) matrix: keep only the
    # proteins and abundance columns that survived, filling any gap with False.
    # id_col is the mask's first column (inserted in validate_matrix).
    id_col = nonpositive_mask.columns[0]
    final_abund_cols = [c for c in nonpositive_mask.columns
                        if c != id_col and c in matrix.columns]
    mask_out = (
        nonpositive_mask.set_index(id_col)
        .reindex(index=matrix[id_col], columns=final_abund_cols)
        .fillna(False)
        .astype(bool)
        .reset_index()
    )
    mask_out.to_parquet(mask_path, index=False)

    # finalise report header before writing
    report.write(report_path)

    print(f'  Validated matrix:   {matrix_path}')
    print(f'  Validated metadata: {meta_path}')
    print(f'  Non-positive mask:  {mask_path}')
    print(f'  Validation report:  {report_path}')


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    run_id = args.run_id

    # Compute report path up front so Report can flush on error_exit.
    report_path = os.path.join(
        args.outdir, f'{run_id}.validation_report_part1.txt'
    )
    report = Report(run_id, report_path=report_path)

    # --- Report header ---
    report.line(f'ProSIFT Validation Report -- {run_id}')
    report.line(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report.line('Part 1 of 2: Input Validation (Processes 4.1-4.3)')
    report.line('  (Part 2: Detection Filter -- see validation_report_part2.txt)')

    print(f'[{run_id}] Validating inputs...')

    # --- Process 4.1: Validate matrix ---
    matrix, nonpositive_mask = validate_matrix(args.abundance, params, report)

    # --- Process 4.2: Validate metadata ---
    metadata = validate_metadata(args.metadata, params, report)

    # --- Process 4.3: Cross-validate ---
    matrix, metadata = cross_validate(matrix, metadata, params, report)

    # --- Contrast orientation (warn-only labeling check) ---
    check_contrast_orientation(run_id, params, report)

    # --- Summary ---
    report.section('VALIDATION SUMMARY')
    report.info(f'Warnings: {report.warning_count}')
    if report.warning_count == 0:
        report.info('All checks passed with no warnings.')
    report.info(f'Output: {matrix.shape[0]} proteins x '
                f'{matrix.shape[1] - 1} columns (working set)')
    report.info('Status: PASSED -- proceed to detection filter (Part 2)')

    # --- Write outputs ---
    write_outputs(matrix, metadata, nonpositive_mask, report, run_id,
                  args.outdir, report_path)
    print(f'[{run_id}] Validation complete. '
          f'{report.warning_count} warning(s). '
          f'{matrix.shape[0]} proteins, {metadata.shape[0]} samples.')


if __name__ == '__main__':
    main()
