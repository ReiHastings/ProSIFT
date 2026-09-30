# title: generate_run_config.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-03-23
# last modified: 2026-03-23
#
# purpose:
#   Interactive setup tool for ProSIFT data preparation. Reads a proteomics
#   search engine output CSV/TSV, guides the user through identifying columns
#   and defining runs, and outputs a YAML run definition file. The YAML serves
#   as the contract between this script and prepare_prosift_input.py.
#
# inputs:
#   - Abundance matrix file (CSV or TSV) with protein_id + sample columns
#   - Metadata file (CSV or TSV) with sample_id column + grouping columns
#
# outputs:
#   - YAML run definition file (default: run_config.yml)
#
# usage example:
#   python generate_run_config.py --input abundance.csv --metadata metadata.csv
#   python generate_run_config.py -i abundance.csv -m metadata.csv -o my_config.yml

import argparse
import csv
import os
import re
import sys
from collections import OrderedDict
from datetime import date

try:
    import yaml
except ImportError:
    print('Error: PyYAML is required. Install with: pip install pyyaml')
    sys.exit(1)


# ============================================================
# YAML formatting helpers
# ============================================================

class QuotedStr(str):
    '''String subclass that forces YAML double-quoting.'''
    pass


def quoted_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


def ordered_dict_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())


yaml.add_representer(QuotedStr, quoted_str_representer)
yaml.add_representer(OrderedDict, ordered_dict_representer)


# ============================================================
# Input helpers
# ============================================================

def prompt(message, default=None):
    '''Prompt the user for input with an optional default value.'''
    if default is not None:
        raw = input(f'{message} [{default}]: ').strip()
        return raw if raw else str(default)
    return input(f'{message}: ').strip()


def prompt_int(message, default=None, min_val=1):
    '''Prompt for an integer with validation.'''
    while True:
        raw = prompt(message, default)
        try:
            val = int(raw)
            if val < min_val:
                print(f'  Please enter a number >= {min_val}.')
                continue
            return val
        except ValueError:
            print('  Please enter a valid integer.')


def prompt_yes_no(message, default='y'):
    '''Prompt for a yes/no answer. Returns True for yes.'''
    while True:
        raw = prompt(message, default).lower()
        if raw in ('y', 'yes'):
            return True
        if raw in ('n', 'no'):
            return False
        print('  Please enter y or n.')


def prompt_selection(message, options, default=None):
    '''Prompt user to select from a numbered list. Returns 0-based index.'''
    while True:
        raw = prompt(message, default)
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return idx - 1
            print(f'  Please enter a number between 1 and {len(options)}.')
        except ValueError:
            print('  Please enter a valid number.')


def prompt_comma_list(message):
    '''Prompt for a comma-separated list of integers. Returns list of ints.'''
    while True:
        raw = input(f'{message}: ').strip()
        if not raw:
            print('  Please enter at least one number.')
            continue
        try:
            nums = [int(x.strip()) for x in raw.split(',')]
            return nums
        except ValueError:
            print('  Please enter comma-separated numbers (e.g., 1,2,3).')


# ============================================================
# Column analysis
# ============================================================

def detect_delimiter(filepath):
    '''Detect whether a file is CSV or TSV by inspecting the first line.'''
    with open(filepath, encoding='utf-8-sig') as f:
        first_line = f.readline()
    tab_count = first_line.count('\t')
    comma_count = first_line.count(',')
    return '\t' if tab_count > comma_count else ','


def load_headers(filepath, delimiter):
    '''Load column headers from a delimited file.'''
    with open(filepath, encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)
    return [h.strip() for h in headers]


def load_data(filepath, delimiter):
    '''Load the full dataset as a list of dicts.'''
    rows = []
    with open(filepath, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)
    return rows


def find_candidate_id_columns(headers, data):
    '''
    Identify candidate protein ID columns: columns where the values
    are predominantly non-numeric. Returns all matching columns.
    '''
    candidates = []
    for col in headers:
        # sample up to 100 values to check if column is numeric
        sample_vals = [row[col] for row in data[:100] if row.get(col, '').strip()]
        if not sample_vals:
            continue
        numeric_count = 0
        for v in sample_vals:
            try:
                float(v)
                numeric_count += 1
            except ValueError:
                pass
        # if more than half the sampled values are non-numeric, it's a candidate
        if numeric_count < len(sample_vals) * 0.5:
            candidates.append(col)
    return candidates



def extract_sample_ids(columns, prefix):
    '''
    Given column names and a prefix, extract sample IDs by stripping
    the prefix from each column name.
    '''
    ids = []
    for col in columns:
        if col.startswith(prefix):
            sample_id = col[len(prefix):]
            if sample_id:
                ids.append(sample_id)
    return ids


def extract_protein_ids(data, protein_id_col):
    '''
    Extract protein IDs from the data, handling semicolon-delimited
    protein groups by taking the first accession.
    '''
    ids = []
    for row in data:
        raw_id = row.get(protein_id_col, '').strip()
        if not raw_id:
            continue
        # take first accession from semicolon-delimited protein groups
        first_id = raw_id.split(';')[0].strip()
        ids.append(first_id)
    return ids


# ============================================================
# Main interactive workflow
# ============================================================

def main():
    # ----------------------------------------------------------
    # Parse arguments
    # ----------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='ProSIFT Run Configuration Generator. '
                    'Interactively define runs from search engine output.'
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Path to the abundance matrix file (CSV or TSV)'
    )
    parser.add_argument(
        '--metadata', '-m', required=True,
        help='Path to the metadata file (CSV or TSV) with sample_id column'
    )
    parser.add_argument(
        '--output', '-o', default='run_config.yml',
        help='Output YAML config file path (default: run_config.yml)'
    )
    args = parser.parse_args()

    input_path = args.input
    metadata_path = args.metadata
    output_path = args.output

    if not os.path.isfile(input_path):
        print(f'Error: file not found: {input_path}')
        sys.exit(1)
    if not os.path.isfile(metadata_path):
        print(f'Error: file not found: {metadata_path}')
        sys.exit(1)

    # ----------------------------------------------------------
    # Load the source files
    # ----------------------------------------------------------
    delimiter = detect_delimiter(input_path)
    headers = load_headers(input_path, delimiter)
    data = load_data(input_path, delimiter)
    n_rows = len(data)

    meta_delimiter = detect_delimiter(metadata_path)
    meta_headers = load_headers(metadata_path, meta_delimiter)
    meta_data = load_data(metadata_path, meta_delimiter)

    # extract sample IDs from metadata
    if 'sample_id' not in meta_headers:
        print('Error: metadata file must contain a "sample_id" column.')
        sys.exit(1)
    meta_sample_ids = [row['sample_id'].strip() for row in meta_data]
    meta_columns = [col for col in meta_headers if col != 'sample_id']

    print()
    print('=' * 60)
    print('ProSIFT Run Configuration Generator')
    print('=' * 60)
    print(f'Abundance file: {os.path.basename(input_path)}')
    print(f'  Columns: {len(headers)}, Rows: {n_rows}')
    print(f'Metadata file: {os.path.basename(metadata_path)}')
    print(f'  Samples: {len(meta_sample_ids)}, Columns: {", ".join(meta_columns)}')
    print()

    # ----------------------------------------------------------
    # Step 1: Protein ID column
    # ----------------------------------------------------------
    print('--- Step 1: Protein ID Column ---')

    # auto-check for "protein_id" column
    if 'protein_id' in headers:
        protein_id_col = 'protein_id'
        protein_ids = extract_protein_ids(data, protein_id_col)
        unique_ids = set(protein_ids)
        n_duplicates = len(protein_ids) - len(unique_ids)
        print(f'Protein ID column found: "protein_id" ({len(unique_ids)} unique proteins)')
        if n_duplicates > 0:
            print(f'  WARNING: {n_duplicates} duplicate IDs detected after extracting first accession.')
            print('  ProSIFT requires unique protein IDs. Check your input data.')
    else:
        # fallback: show all non-numeric columns and ask
        candidates = find_candidate_id_columns(headers, data)
        if not candidates:
            print('Warning: could not auto-detect candidate ID columns.')
            print('All columns:')
            for i, h in enumerate(headers, 1):
                print(f'  [{i}] {h}')
            candidates = list(headers)
        else:
            print('No "protein_id" column found. Non-numeric columns:')
            for i, col in enumerate(candidates, 1):
                print(f'  [{i}] {col}')

        print()
        id_col_idx = prompt_selection('Which column contains protein IDs?', candidates, default=1)
        protein_id_col = candidates[id_col_idx]

        protein_ids = extract_protein_ids(data, protein_id_col)
        unique_ids = set(protein_ids)
        n_duplicates = len(protein_ids) - len(unique_ids)
        print(f'  {len(unique_ids)} unique proteins.')
        if n_duplicates > 0:
            print(f'  WARNING: {n_duplicates} duplicate IDs detected after extracting first accession.')
            print('  ProSIFT requires unique protein IDs. Check your input data.')

    print()

    # ----------------------------------------------------------
    # Step 2: Identify abundance and peptide count columns
    # ----------------------------------------------------------
    print('--- Step 2: Identify Columns ---')

    # --- Abundance columns ---
    # auto-check for "abundance_" prefix first
    abundance_prefix = None
    sample_ids = None
    auto_abund = ['abundance_' + sid for sid in meta_sample_ids]
    if all(col in headers for col in auto_abund):
        abundance_prefix = 'abundance_'
        abundance_cols = auto_abund
        sample_ids = list(meta_sample_ids)
        print(f'Abundance columns found: {len(sample_ids)} (prefix: "abundance_")')
    else:
        # fallback: check for bare sample IDs as column headers
        matched_bare = [sid for sid in meta_sample_ids if sid in headers]
        if len(matched_bare) == len(meta_sample_ids):
            abundance_prefix = ''
            abundance_cols = matched_bare
            sample_ids = matched_bare
            print(f'Abundance columns found: {len(sample_ids)} (bare sample IDs, no prefix)')
        else:
            # last resort: find any prefix that matches all sample IDs
            candidate_prefixes = set()
            for sid in meta_sample_ids:
                for h in headers:
                    if h.endswith(sid) and h != sid:
                        candidate_prefixes.add(h[:-len(sid)])
            valid_prefixes = []
            for prefix in sorted(candidate_prefixes):
                matching = [sid for sid in meta_sample_ids if prefix + sid in headers]
                if len(matching) == len(meta_sample_ids):
                    valid_prefixes.append(prefix)

            if len(valid_prefixes) == 1:
                abundance_prefix = valid_prefixes[0]
                abundance_cols = [abundance_prefix + sid for sid in meta_sample_ids]
                sample_ids = list(meta_sample_ids)
                print(f'Abundance columns found: {len(sample_ids)} (prefix: "{abundance_prefix}")')
            elif len(valid_prefixes) > 1:
                print('Multiple possible abundance prefixes found:')
                for i, prefix in enumerate(valid_prefixes, 1):
                    print(f'  [{i}] "{prefix}" ({len(meta_sample_ids)} columns)')
                print()
                abund_idx = prompt_selection(
                    'Which prefix contains the raw abundance data?',
                    valid_prefixes, default=1
                )
                abundance_prefix = valid_prefixes[abund_idx]
                abundance_cols = [abundance_prefix + sid for sid in meta_sample_ids]
                sample_ids = list(meta_sample_ids)
                print(f'  Selected: "{abundance_prefix}"')
            else:
                print('Error: could not match metadata sample IDs to abundance matrix columns.')
                print(f'  Metadata samples (first 5): {meta_sample_ids[:5]}')
                print(f'  Matrix columns (first 10): {headers[:10]}')
                sys.exit(1)

    # --- Peptide count columns ---
    # auto-check for "peptide_count_" prefix first
    peptide_prefix = None
    auto_pep = ['peptide_count_' + sid for sid in meta_sample_ids]
    if all(col in headers for col in auto_pep):
        peptide_prefix = 'peptide_count_'
        print(f'Peptide count columns found: {len(auto_pep)} (prefix: "peptide_count_")')
    else:
        # fallback: check remaining numeric columns and ask
        selected_cols = set([*abundance_cols, protein_id_col])
        remaining_cols = [h for h in headers if h not in selected_cols]
        numeric_remaining = []
        for col in remaining_cols:
            sample_vals = [row[col] for row in data[:100] if row.get(col, '').strip()]
            if not sample_vals:
                continue
            numeric_count = sum(1 for v in sample_vals
                                if v.replace('.', '', 1).replace('-', '', 1).isdigit())
            if numeric_count > len(sample_vals) * 0.5:
                numeric_remaining.append(col)

        if numeric_remaining:
            print('No "peptide_count_" columns found.')
            print(f'Remaining numeric columns ({len(numeric_remaining)}):')
            show_count = min(8, len(numeric_remaining))
            for col in numeric_remaining[:show_count]:
                print(f'  {col}')
            if len(numeric_remaining) > show_count:
                print(f'  ... and {len(numeric_remaining) - show_count} more')
            print()
            include_peptides = prompt_yes_no('Include peptide count columns?', default='n')
            if include_peptides:
                peptide_prefix = prompt('Enter the peptide count column prefix')
                matched = [h for h in numeric_remaining if h.startswith(peptide_prefix)]
                print(f'  Found {len(matched)} columns with prefix "{peptide_prefix}"')
                if len(matched) == 0:
                    print('  No columns matched. Skipping peptide counts.')
                    peptide_prefix = None
        else:
            print('No peptide count columns found.')

    print()

    # ----------------------------------------------------------
    # Step 3: Groups
    # ----------------------------------------------------------
    print('--- Step 3: Groups ---')

    # build metadata lookup
    meta_lookup = {}
    for row in meta_data:
        meta_lookup[row['sample_id'].strip()] = row

    # derive groups by stripping -{number} suffix from sample IDs
    groups = OrderedDict()  # group_name -> [sample_ids]
    for sid in sample_ids:
        group_name = re.sub(r'-\d+$', '', sid)
        groups.setdefault(group_name, []).append(sid)

    group_names = list(groups.keys())

    def print_group_list():
        '''Print the numbered group list.'''
        for i, gname in enumerate(group_names, 1):
            members = groups[gname]
            print(f'  {i:>2}. {gname} ({len(members)} samples)')
        print()

    print(f'{len(group_names)} groups ({len(sample_ids)} total samples):')
    print()
    print_group_list()

    # ----------------------------------------------------------
    # Step 4: Define runs
    # ----------------------------------------------------------
    print('--- Step 4: Define Runs ---')

    runs = OrderedDict()
    run_num = 0

    while True:
        run_num += 1
        if run_num > 1:
            # reprint group list
            print()
            print_group_list()

        print(f'--- Run {run_num} ---')
        while True:
            group_nums = prompt_comma_list('Select two groups to compare (e.g., 1,2)')
            if len(group_nums) != 2:
                print('  Please select exactly 2 groups.')
                continue
            valid = True
            for num in group_nums:
                if num < 1 or num > len(group_names):
                    print(f'  Invalid group number: {num}. Must be 1-{len(group_names)}.')
                    valid = False
                    break
            if valid:
                break

        g1_name = group_names[group_nums[0] - 1]
        g2_name = group_names[group_nums[1] - 1]
        g1_samples = groups[g1_name]
        g2_samples = groups[g2_name]

        # infer group_column: find the metadata column where the two groups
        # have different values (each group uniform within itself)
        group_col_name = None
        group1_label = None
        group2_label = None
        for col in meta_columns:
            g1_vals = set(meta_lookup[sid][col] for sid in g1_samples)
            g2_vals = set(meta_lookup[sid][col] for sid in g2_samples)
            # each group should have exactly one value, and they should differ
            if len(g1_vals) == 1 and len(g2_vals) == 1 and g1_vals != g2_vals:
                group_col_name = col
                group1_label = g1_vals.pop()
                group2_label = g2_vals.pop()
                break

        if group_col_name is None:
            # fallback: ask the user
            print(f'  Could not auto-detect which column differs between {g1_name} and {g2_name}.')
            meta_col_options = ', '.join(f'"{c}"' for c in meta_columns)
            group_col_name = prompt(f'  Group column name ({meta_col_options})')
            group1_label = prompt(f'  Label for {g1_name}')
            group2_label = prompt(f'  Label for {g2_name}')

        # auto-generate run name
        run_name = f'{g1_name}_vs_{g2_name}'

        # build sample -> group mapping
        sample_group_map = OrderedDict()
        for sid in g1_samples:
            sample_group_map[sid] = group1_label
        for sid in g2_samples:
            sample_group_map[sid] = group2_label

        # show summary
        print()
        print(f'  Run: {run_name}')
        print(f'  Group column: {group_col_name}')
        print(f'  {group1_label}: {", ".join(g1_samples)}')
        print(f'  {group2_label}: {", ".join(g2_samples)}')

        # safety checks
        if len(g1_samples) < 2:
            print(f'  WARNING: "{group1_label}" has only {len(g1_samples)} sample(s).')
        if len(g2_samples) < 2:
            print(f'  WARNING: "{group2_label}" has only {len(g2_samples)} sample(s).')

        if not prompt_yes_no('Correct?', default='y'):
            print('  Run discarded.')
            run_num -= 1
            if not prompt_yes_no('Try again?', default='y'):
                break
            continue

        runs[run_name] = {
            'samples': sample_group_map,
            'group_column': group_col_name,
        }

        print()
        if not prompt_yes_no('Add another run?', default='y'):
            break

    # ----------------------------------------------------------
    # Step 5: Review and write
    # ----------------------------------------------------------
    print()
    print('--- Step 5: Review ---')
    print()

    # run summary
    print('Run summary:')
    for run_name, run_def in runs.items():
        sample_map = run_def['samples']
        group_col = run_def['group_column']
        groups = OrderedDict()
        for s, g in sample_map.items():
            groups.setdefault(g, []).append(s)
        parts = [f'{len(members)} {g}' for g, members in groups.items()]
        total = sum(len(m) for m in groups.values())
        print(f'  {run_name}: {", ".join(parts)}  ({total} samples, group_column: {group_col})')
    print()

    # check for samples assigned to multiple runs (expected for this dataset)
    sample_run_count = {}
    for run_name, run_def in runs.items():
        for s in run_def['samples']:
            sample_run_count.setdefault(s, []).append(run_name)
    multi_run = {s: r for s, r in sample_run_count.items() if len(r) > 1}
    if multi_run:
        print(f'  Note: {len(multi_run)} samples appear in multiple runs (expected for multi-comparison designs).')
    print()

    if not prompt_yes_no('Write config?', default='y'):
        print('Aborted.')
        sys.exit(0)

    # ----------------------------------------------------------
    # Build and write the YAML config
    # ----------------------------------------------------------
    config = OrderedDict()

    # source section
    source = OrderedDict()
    source['abundance_file'] = QuotedStr(os.path.basename(input_path))
    source['metadata_file'] = QuotedStr(os.path.basename(metadata_path))
    source['protein_id_column'] = QuotedStr(protein_id_col)
    source['abundance_prefix'] = QuotedStr(abundance_prefix)
    if peptide_prefix:
        source['peptide_count_prefix'] = QuotedStr(peptide_prefix)
    else:
        source['peptide_count_prefix'] = None
    config['source'] = source

    # runs section
    runs_section = OrderedDict()
    for run_name, run_def in runs.items():
        run_entry = OrderedDict()
        samples = OrderedDict()
        for sample_id, group in run_def['samples'].items():
            samples[QuotedStr(sample_id)] = group
        run_entry['samples'] = samples
        run_entry['group_column'] = run_def['group_column']
        runs_section[run_name] = run_entry
    config['runs'] = runs_section

    # write YAML with a header comment
    today = date.today().isoformat()
    header = (
        f'# ProSIFT run definition\n'
        f'# Generated by generate_run_config.py on {today}\n'
        f'# Abundance: {os.path.basename(input_path)}\n'
        f'# Metadata: {os.path.basename(metadata_path)}\n'
        f'\n'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        yaml.dump(
            dict(config),  # convert OrderedDict for yaml.dump compatibility
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    print()
    print(f'Wrote run configuration to: {output_path}')
    print()
    print(f'Done. {len(runs)} run(s) defined. Review {output_path}, then run:')
    print(f'  python prepare_prosift_input.py --config {output_path}')


if __name__ == '__main__':
    main()
