#!/usr/bin/env python3
# title: filter_proteins.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-03-26
# last modified: 2026-04-23
#
# purpose:
#   Module 01, Process 4.4. Applies the per-run detection filter to the
#   validated abundance matrix. Classifies each protein as PASSED, PARTIAL,
#   SINGLE-GROUP, SPARSE, or ABSENT based on per-group detection counts,
#   removes proteins with insufficient signal, and produces a missingness
#   overview and detection filter summary for the validation report.
#
# inputs:
#   --matrix    : {run_id}.validated_matrix.parquet (from validate_inputs.py)
#   --metadata  : {run_id}.validated_metadata.parquet (from validate_inputs.py)
#   --params    : per-run params.yml
#   --run-id    : string identifier used to name output files
#   --outdir    : output directory (default: current working directory)
#
# outputs:
#   {outdir}/{run_id}.filtered_matrix.parquet
#   {outdir}/{run_id}.detection_filter_table.csv
#   {outdir}/{run_id}.validation_report_part2.txt
#
# usage example:
#   python bin/filter_proteins.py \
#       --matrix   results/CTXcyto_WT_vs_CTXcyto_KO/CTXcyto_WT_vs_CTXcyto_KO.validated_matrix.parquet \
#       --metadata results/CTXcyto_WT_vs_CTXcyto_KO/CTXcyto_WT_vs_CTXcyto_KO.validated_metadata.parquet \
#       --params   prosift_inputs/CTXcyto_WT_vs_CTXcyto_KO/CTXcyto_WT_vs_CTXcyto_KO_params.yml \
#       --run-id   CTXcyto_WT_vs_CTXcyto_KO \
#       --outdir   results/CTXcyto_WT_vs_CTXcyto_KO

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ProSIFT Module 01 -- per-run detection filter.'
    )
    parser.add_argument('--matrix', required=True,
                        help='Path to validated matrix Parquet')
    parser.add_argument('--metadata', required=True,
                        help='Path to validated metadata Parquet')
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
    with open(params_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================
# Report helper
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
# Detection counting and classification
# ============================================================

def get_abundance_cols(matrix: pd.DataFrame, id_col: str,
                       pep_prefix: str | None) -> list[str]:
    '''Return the abundance column names from the matrix.'''
    return [c for c in matrix.columns
            if c != id_col
            and (not pep_prefix or not c.startswith(pep_prefix))]


def count_detections_per_group(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    id_col: str,
    abund_cols: list[str],
    abund_prefix: str,
    group_col: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    '''
    For each protein, count non-missing abundance values per group.

    Returns:
      detection_counts -- DataFrame indexed by protein_id with one column per
                          group containing the integer detection count.
      group_sizes      -- dict {group: number of abundance columns in that
                          group} = the maximum possible detection count. Used
                          by the anchor gate to define "fully detected".
    '''
    # map sample_id -> group
    sample_to_group: dict[str, str] = dict(
        zip(metadata['sample_id'].astype(str),
            metadata[group_col].astype(str), strict=True)
    )

    # map abundance column -> group
    col_to_group: dict[str, str] = {
        col: sample_to_group[col[len(abund_prefix):]]
        for col in abund_cols
    }

    groups = metadata[group_col].astype(str).unique()
    abund = matrix.set_index(id_col)[abund_cols]

    detection_counts = pd.DataFrame(index=abund.index)
    group_sizes: dict[str, int] = {}
    for grp in groups:
        grp_cols = [c for c, g in col_to_group.items() if g == grp]
        detection_counts[grp] = abund[grp_cols].notna().sum(axis=1)
        group_sizes[grp] = len(grp_cols)

    return detection_counts, group_sizes


def classify_proteins(
    detection_counts: pd.DataFrame,
    min_detections: int,
    group_sizes: dict[str, int],
    min_present_detections: int | None = None
) -> pd.Series:
    '''
    Classify each protein into PASSED, SINGLE-GROUP, PARTIAL, WEAK-ANCHOR,
    SPARSE, or ABSENT.

    Rules (applied in order):
      ABSENT       -- zero detections in ALL groups
      PASSED       -- >= min_detections in ALL groups
      PARTIAL      -- >= min_detections in at least one group AND
                      sub-threshold but non-zero detections in at least
                      one other group (e.g., 2/3 WT, 1/3 KO with min=2)
      SINGLE-GROUP -- >= min_detections in exactly one group AND
                      zero detections in all other groups
      WEAK-ANCHOR  -- would be PARTIAL or SINGLE-GROUP, but NO group is a
                      solid detection anchor (see below). These proteins
                      rely on heavy MNAR imputation resting on few real
                      measurements, so they are REMOVED.
      SPARSE       -- everything else (detections exist but no group meets
                      the threshold)

    Anchor gate (added 2026-07-21):
      PARTIAL and SINGLE-GROUP proteins have missing values imputed as MNAR
      (synthetic low-end MinProb draws) in their absent / sub-threshold
      group. That imputation is only defensible when the *present* group is
      a solid anchor. We require at least one group to be "fully detected":
      detections >= anchor threshold, where the threshold defaults to that
      group's replicate count (full detection) and can be relaxed to a fixed
      integer via qc.min_detections_present_group. Proteins that pass the
      lenient min_detections filter in one group but have no fully-detected
      anchor group (e.g. 2/3-vs-0/3, or 2/3-vs-1/3 at n=3) become WEAK-ANCHOR
      and are removed. This does NOT touch PASSED proteins (all groups meet
      min_detections, missing values are MAR only). See Module 01 spec
      Section 5 and the 2026-07-21 Decision Register entry.

    Assumptions (enforced upstream by validate_inputs.py, so not re-checked
    here): at least 2 distinct groups (validate_inputs hard-stops on fewer),
    and every group has >= qc.min_samples_per_group samples (default 2), so a
    group is never size 1 at default config. Callers outside the pipeline
    (e.g. unit tests) must uphold these for the anchor semantics to hold: with
    a single group, the SINGLE-GROUP assignment can shadow PASSED; with a
    size-1 group, "full detection" is a single measurement.
    '''
    total_detections = detection_counts.sum(axis=1)
    meets_threshold = detection_counts >= min_detections
    zero_in_group = detection_counts == 0

    n_groups = detection_counts.shape[1]
    n_meeting = meets_threshold.sum(axis=1)
    n_zero = zero_in_group.sum(axis=1)

    status = pd.Series('SPARSE', index=detection_counts.index, dtype=str)

    # ABSENT: no detections anywhere
    status[total_detections == 0] = 'ABSENT'

    # PASSED: meets threshold in all groups
    status[n_meeting == n_groups] = 'PASSED'

    # SINGLE-GROUP: meets threshold in exactly one group,
    #               and all other groups have zero detections
    single_group = (n_meeting == 1) & (n_zero == n_groups - 1)
    status[single_group] = 'SINGLE-GROUP'

    # PARTIAL: meets threshold in at least one group, but at least one
    #          other group has sub-threshold non-zero detections.
    #          This covers the gap between SINGLE-GROUP (zero in other
    #          groups) and PASSED (all groups meet threshold).
    partial = (n_meeting >= 1) & (n_meeting < n_groups) & (n_zero < n_groups - n_meeting)
    # Only apply to proteins still labeled SPARSE (not already SINGLE-GROUP)
    status[(status == 'SPARSE') & partial] = 'PARTIAL'

    # --- Anchor gate: demote imputation-heavy PARTIAL/SINGLE-GROUP proteins ---
    # A group is an anchor if its detection count reaches the anchor threshold.
    # Default threshold = group's replicate count (full detection); an explicit
    # integer in min_present_detections overrides this uniformly.
    anchor_reached = pd.DataFrame(index=detection_counts.index)
    for grp in detection_counts.columns:
        threshold = (
            min_present_detections
            if min_present_detections is not None
            else int(group_sizes[grp])
        )
        anchor_reached[grp] = detection_counts[grp] >= threshold
    has_anchor = anchor_reached.any(axis=1)

    needs_anchor = status.isin(['SINGLE-GROUP', 'PARTIAL'])
    status[needs_anchor & ~has_anchor] = 'WEAK-ANCHOR'

    # SPARSE: anything remaining (already set as default)

    return status


def validate_min_present_detections(
    value: object,
    group_sizes: dict[str, int]
) -> str | None:
    '''
    Validate the qc.min_detections_present_group parameter (the anchor
    threshold used by classify_proteins).

    Returns an error message string if the value is invalid, or None if it is
    acceptable. Acceptable values:
      - None                       -- anchor requires FULL detection in some
                                       group (the default behavior).
      - int in [1, max group size] -- a fixed anchor threshold.

    Rejects (each would silently corrupt filtering, so we fail loudly instead):
      - non-integers, including bool and YAML strings -- would raise deep in
        classify_proteins or bypass the intended comparison.
      - values < 1 (0 or negative) -- would make every group an anchor,
        silently disabling the gate.
      - values larger than every group size -- no group could ever anchor, so
        all SINGLE-GROUP/PARTIAL proteins would be silently removed.
    '''
    if value is None:
        return None
    # bool is a subclass of int; reject it explicitly so True/False cannot pass.
    if isinstance(value, bool) or not isinstance(value, int):
        return (
            'qc.min_detections_present_group must be an integer or null '
            f'(got {type(value).__name__}: {value!r}). Use null to require '
            'full detection in the anchor group.'
        )
    if value < 1:
        return (
            'qc.min_detections_present_group must be >= 1 (or null for full '
            f'detection), got {value}.'
        )
    max_size = max(group_sizes.values()) if group_sizes else 0
    if value > max_size:
        return (
            f'qc.min_detections_present_group ({value}) exceeds the largest '
            f'group size ({max_size}); no group could ever be an anchor, so '
            'every presence/absence and partial protein would be removed. '
            'Lower it or set it to null (full detection).'
        )
    return None


# ============================================================
# Report blocks
# ============================================================

def format_pct(n: int, total: int) -> str:
    return f'{n / total * 100:5.1f}%'


def write_detection_filter_block(
    report: Report,
    status: pd.Series,
    min_detections: int,
    n_input: int,
    min_present_detections: int | None = None
) -> None:
    counts = status.value_counts()
    n_passed = counts.get('PASSED', 0)
    n_single = counts.get('SINGLE-GROUP', 0)
    n_partial = counts.get('PARTIAL', 0)
    n_weak = counts.get('WEAK-ANCHOR', 0)
    n_sparse = counts.get('SPARSE', 0)
    n_absent = counts.get('ABSENT', 0)
    n_retained = n_passed + n_single + n_partial
    n_removed = n_weak + n_sparse + n_absent

    anchor_desc = (
        'fully detected (all replicates)'
        if min_present_detections is None
        else f'detected in >= {min_present_detections} replicates'
    )

    report.section('DETECTION FILTER SUMMARY')
    report.line()
    report.line(f'Filter threshold: min_detections_per_group = {min_detections}')
    report.line(f'  A protein passes if detected (non-missing) in at least '
                f'{min_detections} replicates')
    report.line('  in at least one group.')
    report.line('Anchor requirement: presence/absence and partial proteins are '
                'retained only')
    report.line(f'  if at least one group is an anchor ({anchor_desc}). This prevents '
                f'proteins')
    report.line('  whose results would rest on heavy MNAR imputation over few real '
                'values.')
    report.line()
    report.line('Category definitions:')
    report.line(f'  PASSED           - Detected in >= {min_detections} replicates '
                f'in both groups.')
    report.line('                     Sufficient data for reliable statistical testing.')
    report.line('                     Missing values (if any) imputed as MAR only.')
    report.line(f'  PARTIAL          - Detected in >= {min_detections} replicates '
                f'in one group')
    report.line(f'                     but below threshold (1 to {min_detections - 1} '
                f'detections) in the other,')
    report.line('                     WITH a fully-detected anchor group.')
    report.line('                     Missing values imputed as a mix of MNAR and MAR.')
    report.line(f'  SINGLE-GROUP     - Detected in >= {min_detections} replicates '
                f'in one group')
    report.line('                     but completely absent (0 detections) in the other,')
    report.line('                     WITH a fully-detected anchor group.')
    report.line('                     Retained as presence/absence candidates. The absent')
    report.line('                     group is entirely MNAR-imputed; interpret as')
    report.line('                     qualitative, and read direction from the detection')
    report.line('                     pattern, not the imputation-driven fold change.')
    report.line('  WEAK-ANCHOR      - Would be PARTIAL or SINGLE-GROUP, but no group')
    report.line(f'     (removed)        reaches the anchor ({anchor_desc}).')
    report.line('                     Results would be dominated by MNAR imputation over')
    report.line('                     as few as 2 real measurements. Removed.')
    report.line(f'  SPARSE (removed) - Detected in < {min_detections} replicates '
                f'in ALL groups.')
    report.line('                     Insufficient data for reliable statistical testing')
    report.line('                     in any group.')
    report.line('  ABSENT (removed) - Zero detections across all samples in this run.')
    report.line('                     Protein was retained in the source dataset only')
    report.line('                     because it was detected in conditions not included')
    report.line('                     in this run.')
    report.line()
    report.line('Results:')
    # Fixed 22-char label column so counts stay aligned regardless of label
    # length (the WEAK-ANCHOR label is the longest).
    lbl = 22
    report.line(f'  {"Input proteins:":<{lbl}}{n_input:>6,}')
    report.line(f'  {"Passed:":<{lbl}}{n_passed:>6,}  ({format_pct(n_passed, n_input)})')
    report.line(f'  {"Partial:":<{lbl}}{n_partial:>6,}  ({format_pct(n_partial, n_input)})  [flagged]')
    report.line(f'  {"Single-group:":<{lbl}}{n_single:>6,}  ({format_pct(n_single, n_input)})  [flagged]')
    report.line(f'  {"Weak-anchor (removed):":<{lbl}}{n_weak:>6,}  ({format_pct(n_weak, n_input)})')
    report.line(f'  {"Sparse (removed):":<{lbl}}{n_sparse:>6,}  ({format_pct(n_sparse, n_input)})')
    report.line(f'  {"Absent (removed):":<{lbl}}{n_absent:>6,}  ({format_pct(n_absent, n_input)})')
    report.line(f'  {"─" * (lbl + 6)}')
    report.line(f'  {"Retained:":<{lbl}}{n_retained:>6,}  ({format_pct(n_retained, n_input)})')
    report.line(f'  {"Removed:":<{lbl}}{n_removed:>6,}  ({format_pct(n_removed, n_input)})')


def write_missingness_overview_block(
    report: Report,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    id_col: str,
    abund_cols: list[str],
    abund_prefix: str,
    group_col: str
) -> None:
    n_proteins = len(matrix)
    n_samples = len(abund_cols)
    total_values = n_proteins * n_samples
    abund = matrix.set_index(id_col)[abund_cols]
    total_missing = abund.isna().sum().sum()

    sample_to_group: dict[str, str] = dict(
        zip(metadata['sample_id'].astype(str),
            metadata[group_col].astype(str), strict=True)
    )
    col_to_group: dict[str, str] = {
        col: sample_to_group[col[len(abund_prefix):]]
        for col in abund_cols
    }
    groups = metadata[group_col].astype(str).unique()

    report.section('PRE-FILTER MISSINGNESS OVERVIEW')
    report.line()
    report.line(f'Total abundance values:  {total_values:>8,}  '
                f'({n_proteins:,} proteins x {n_samples} samples)')
    report.line(f'Missing values:          {total_missing:>8,}  '
                f'({total_missing / total_values * 100:.1f}%)')
    report.line()
    report.line('Per-group detection rates:')

    for grp in sorted(groups):
        grp_cols = [c for c, g in col_to_group.items() if g == grp]
        # per-sample detection counts (axis=0: sum across proteins for each sample)
        per_sample = abund[grp_cols].notna().sum(axis=0)
        grp_mean_detected = per_sample.mean()
        report.line(
            f'  {grp}:  mean {grp_mean_detected:,.0f} detected '
            f'({grp_mean_detected / n_proteins * 100:.1f}%), '
            f'range {per_sample.min():,}-{per_sample.max():,}'
        )

    report.line()
    report.line('Per-sample detection:')
    for col in abund_cols:
        sample_id = col[len(abund_prefix):]
        n_detected = abund[col].notna().sum()
        n_missing = abund[col].isna().sum()
        report.line(
            f'  {sample_id:<20}  {n_detected:>6,} detected '
            f'({n_detected / n_proteins * 100:.1f}%),  '
            f'{n_missing:>5,} missing '
            f'({n_missing / n_proteins * 100:.1f}%)'
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    run_id = args.run_id

    # Compute report path up front so Report can flush on error_exit.
    report_path = os.path.join(
        args.outdir, f'{run_id}.validation_report_part2.txt'
    )
    report = Report(run_id, report_path=report_path)
    report.line(f'ProSIFT Validation Report -- {run_id}')
    report.line(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report.line('Part 2 of 2: Detection Filter (Process 4.4)')
    report.line('  (Part 1: Input Validation -- see validation_report_part1.txt)')

    print(f'[{run_id}] Applying detection filter...')

    # --- Load inputs ---
    matrix = pd.read_parquet(args.matrix)
    metadata = pd.read_parquet(args.metadata)

    id_col: str = params['input']['protein_id_column']
    abund_prefix: str = params['input'].get('abundance_prefix', '')
    pep_prefix: str | None = params['input'].get('peptide_count_prefix') or None
    group_col: str = params['design']['group_column']
    min_detections: int = params['qc']['min_detections_per_group']
    # Anchor threshold for presence/absence and partial proteins. Default
    # (None / absent / null in YAML) requires the anchor group to be FULLY
    # detected; an explicit integer sets a fixed minimum instead. See the
    # anchor gate in classify_proteins() and Module 01 spec Section 5.
    min_present_detections: int | None = params['qc'].get(
        'min_detections_present_group'
    )

    abund_cols = get_abundance_cols(matrix, id_col, pep_prefix)
    n_input = len(matrix)

    # Guard the empty-matrix case up front (avoids a divide-by-zero in the
    # percentage accounting below, and there is nothing to filter anyway).
    if n_input == 0:
        report.error_exit('Input matrix contains no proteins; nothing to filter.')

    # --- Pre-filter missingness overview (captures state before any removal) ---
    write_missingness_overview_block(
        report, matrix, metadata, id_col, abund_cols, abund_prefix, group_col
    )

    # --- Classify proteins ---
    detection_counts, group_sizes = count_detections_per_group(
        matrix, metadata, id_col, abund_cols, abund_prefix, group_col
    )

    # Validate the anchor parameter now that group sizes are known. An invalid
    # value (out-of-range, wrong type) would silently corrupt filtering, so we
    # fail loudly with a clear message instead.
    param_err = validate_min_present_detections(min_present_detections, group_sizes)
    if param_err is not None:
        report.error_exit(param_err)

    status = classify_proteins(
        detection_counts, min_detections, group_sizes, min_present_detections
    )

    # --- Detection filter summary block ---
    write_detection_filter_block(
        report, status, min_detections, n_input, min_present_detections
    )

    # --- Warnings and hard stops ---
    # NOTE: PASSED, PARTIAL, and SINGLE-GROUP are retained. PARTIAL/SINGLE-GROUP
    # only survive if they have a fully-detected anchor group; anchor failures
    # are relabeled WEAK-ANCHOR (removed). See Section 4.4 of the module spec.
    retain_mask = status.isin(['PASSED', 'PARTIAL', 'SINGLE-GROUP'])
    n_retained = int(retain_mask.sum())
    n_removed = n_input - n_retained
    pct_removed = n_removed / n_input * 100

    if pct_removed > 20:
        report.warn(
            f'{pct_removed:.1f}% of proteins removed by detection filter '
            f'(>{20}% threshold). This may indicate the input dataset is not '
            f'appropriate for this run configuration.'
        )

    if n_retained < 50:
        report.error_exit(
            f'Only {n_retained} proteins remain after detection filtering '
            f'(minimum required: 50). Insufficient data for meaningful analysis.'
        )

    # --- Filter matrix ---
    filtered_matrix = matrix[matrix[id_col].isin(status[retain_mask].index)].copy()

    # --- Build detection filter summary table ---
    filter_table = detection_counts.copy()
    filter_table.columns = [f'detections_{g}' for g in filter_table.columns]
    filter_table['filter_status'] = status
    filter_table = filter_table.reset_index()

    # --- Summary ---
    report.section('FILTER SUMMARY')
    report.info(f'Warnings: {report.warning_count}')
    report.info(f'Retained: {n_retained:,} proteins '
                f'({n_retained / n_input * 100:.1f}%)')
    report.info(f'Removed:  {n_removed:,} proteins '
                f'({pct_removed:.1f}%)')
    report.info('Status: PASSED -- proceed to ID mapping')

    # --- Write outputs ---
    os.makedirs(args.outdir, exist_ok=True)

    matrix_path = os.path.join(args.outdir, f'{run_id}.filtered_matrix.parquet')
    table_path = os.path.join(args.outdir, f'{run_id}.detection_filter_table.csv')

    filtered_matrix.to_parquet(matrix_path, index=False)
    filter_table.to_csv(table_path, index=False)
    report.write(report_path)

    print(f'  Filtered matrix:      {matrix_path}')
    print(f'  Detection filter table: {table_path}')
    print(f'  Validation report:    {report_path}')
    print(f'[{run_id}] Detection filter complete. '
          f'{report.warning_count} warning(s). '
          f'{n_retained:,} proteins retained, {n_removed:,} removed.')


if __name__ == '__main__':
    main()
