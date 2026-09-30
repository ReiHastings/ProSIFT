# title: prepare_ctx_data.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-03-23
# last modified: 2026-03-23
#
# purpose:
#   CTX/Synaptosome dataset-specific format conversion script. Reads the
#   DIA-NN Filtered Proteins CSV (8,948 proteins x 178 columns) and produces
#   three clean master files in ProSIFT-ready format:
#     - CTX_abundance.csv       (protein_id + 24 abundance + 24 peptide count columns)
#     - CTX_metadata.csv        (24 samples with genotype, region, fraction columns)
#     - CTX_protein_reference.csv (gene symbols + descriptions for cross-checking)
#
#   These master files are then fed into generate_run_config.py, which
#   defines the 10 pairwise runs and splits the data accordingly.
#
# inputs:
#   - CTX_synaptosome_project_data/Filtered Proteins-Table 1.csv
#
# outputs:
#   - prosift_inputs/CTX_abundance.csv
#   - prosift_inputs/CTX_metadata.csv
#   - prosift_inputs/CTX_protein_reference.csv
#
# usage example:
#   python scripts/prepare_ctx_data.py
#   python scripts/prepare_ctx_data.py --input "CTX_synaptosome_project_data/Filtered Proteins-Table 1.csv" --outdir prosift_inputs

import argparse
import csv
import os
import sys

# ============================================================
# Source file column constants
# ============================================================

ABUNDANCE_PREFIX = 'Abundance '
PEPTIDE_COUNT_PREFIX = 'Stripped Sequence Count '
PROTEIN_ID_COL = 'Protein Group'
GENE_SYMBOL_COL = 'Genes'
DESCRIPTION_COL = 'First Protein Description'

# the 24 sample IDs in the source file, in order
SAMPLE_IDS = [
    'CTXcyto_WT-1', 'CTXcyto_WT-2', 'CTXcyto_WT-3',
    'CTXcyto_KO-1', 'CTXcyto_KO-2', 'CTXcyto_KO-3',
    'CTXsynap_WT-1', 'CTXsynap_WT-2', 'CTXsynap_WT-3',
    'CTXsynap_KO-1', 'CTXsynap_KO-2', 'CTXsynap_KO-3',
    'HIPcyto_WT-1', 'HIPcyto_WT-2', 'HIPcyto_WT-3',
    'HIPcyto_KO-1', 'HIPcyto_KO-2', 'HIPcyto_KO-3',
    'HIPsynap_WT-1', 'HIPsynap_WT-2', 'HIPsynap_WT-3',
    'HIPsynap_KO-1', 'HIPsynap_KO-2', 'HIPsynap_KO-3',
]

# sample metadata: parsed from sample ID naming convention
# format is {Region}{Fraction}_{Genotype}-{Replicate}
SAMPLE_METADATA = {}
for sid in SAMPLE_IDS:
    # e.g., 'CTXcyto_WT-1' -> region=CTX, fraction=cyto, genotype=WT
    condition, replicate_part = sid.rsplit('-', 1)
    region_fraction, genotype = condition.split('_')
    if region_fraction.startswith('CTX'):
        region = 'CTX'
        fraction = region_fraction[3:]  # 'cyto' or 'synap'
    elif region_fraction.startswith('HIP'):
        region = 'HIP'
        fraction = region_fraction[3:]
    SAMPLE_METADATA[sid] = {
        'genotype': genotype,
        'region': region,
        'fraction': fraction,
    }


# ============================================================
# Processing functions
# ============================================================

def load_source(filepath):
    '''Load the source CSV and return headers + list of row dicts.'''
    with open(filepath, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    print(f'Loaded {len(rows)} proteins from {os.path.basename(filepath)}')
    print(f'  Columns: {len(headers)}')
    return headers, rows


def extract_protein_id(raw_id):
    '''Extract first UniProt accession from a semicolon-delimited protein group.'''
    return raw_id.split(';')[0].strip()


def write_abundance_csv(filepath, rows):
    '''
    Write the master abundance matrix CSV.
    Columns: protein_id, then 24 abundance_{sample_id} columns, then
    24 peptide_count_{sample_id} columns.
    '''
    abundance_src_cols = [ABUNDANCE_PREFIX + sid for sid in SAMPLE_IDS]
    peptide_src_cols = [PEPTIDE_COUNT_PREFIX + sid for sid in SAMPLE_IDS]
    out_headers = (
        ['protein_id']
        + ['abundance_' + sid for sid in SAMPLE_IDS]
        + ['peptide_count_' + sid for sid in SAMPLE_IDS]
    )

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(out_headers)
        for row in rows:
            protein_id = extract_protein_id(row[PROTEIN_ID_COL])
            abund_vals = [row.get(col, '') for col in abundance_src_cols]
            pep_vals = [row.get(col, '') for col in peptide_src_cols]
            writer.writerow([protein_id, *abund_vals, *pep_vals])

    print(f'  Wrote {filepath}')
    print(f'    {len(rows)} proteins x {len(SAMPLE_IDS)} samples '
          f'(+ {len(SAMPLE_IDS)} peptide count columns)')


def write_metadata_csv(filepath):
    '''
    Write the master metadata CSV.
    Columns: sample_id, genotype, region, fraction.
    '''
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'genotype', 'region', 'fraction'])
        for sid in SAMPLE_IDS:
            meta = SAMPLE_METADATA[sid]
            writer.writerow([sid, meta['genotype'], meta['region'], meta['fraction']])

    print(f'  Wrote {filepath}')
    print(f'    {len(SAMPLE_IDS)} samples')


def write_protein_reference(filepath, rows):
    '''Write the protein reference CSV (human convenience file for cross-checking).'''
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['protein_id', 'protein_group', 'gene_symbol', 'description'])
        for row in rows:
            protein_id = extract_protein_id(row[PROTEIN_ID_COL])
            protein_group = row.get(PROTEIN_ID_COL, '')
            gene_symbol = row.get(GENE_SYMBOL_COL, '')
            description = row.get(DESCRIPTION_COL, '')
            writer.writerow([protein_id, protein_group, gene_symbol, description])

    print(f'  Wrote {filepath}')
    print(f'    {len(rows)} proteins')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert CTX/Synaptosome DIA-NN output to ProSIFT master input files.'
    )
    parser.add_argument(
        '--input', '-i',
        default='CTX_synaptosome_project_data/Filtered Proteins-Table 1.csv',
        help='Path to the Filtered Proteins CSV '
             '(default: CTX_synaptosome_project_data/Filtered Proteins-Table 1.csv)'
    )
    parser.add_argument(
        '--outdir', '-o', default='prosift_inputs',
        help='Output directory (default: prosift_inputs/)'
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'Error: file not found: {args.input}')
        sys.exit(1)

    # load source data
    headers, rows = load_source(args.input)

    # verify expected columns exist in the source file
    missing_abund = [sid for sid in SAMPLE_IDS if ABUNDANCE_PREFIX + sid not in headers]
    missing_pep = [sid for sid in SAMPLE_IDS if PEPTIDE_COUNT_PREFIX + sid not in headers]
    if missing_abund:
        print(f'Error: missing abundance columns for: {missing_abund}')
        sys.exit(1)
    if missing_pep:
        print(f'Warning: missing peptide count columns for: {missing_pep}')
    if PROTEIN_ID_COL not in headers:
        print(f'Error: protein ID column "{PROTEIN_ID_COL}" not found')
        sys.exit(1)

    # create output directory
    os.makedirs(args.outdir, exist_ok=True)

    print(f'\nWriting master files to {args.outdir}/\n')

    # write the 3 master files
    write_abundance_csv(os.path.join(args.outdir, 'CTX_abundance.csv'), rows)
    write_metadata_csv(os.path.join(args.outdir, 'CTX_metadata.csv'))
    write_protein_reference(os.path.join(args.outdir, 'CTX_protein_reference.csv'), rows)

    print('\nDone. Next step: define runs with generate_run_config.py')


if __name__ == '__main__':
    main()
