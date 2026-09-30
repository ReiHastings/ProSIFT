#!/usr/bin/env python3
# title: query_ctd.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-07
# last modified: 2026-04-07
#
# purpose:
#   Module 06 QUERY_CTD process. Filters the pre-downloaded CTD bulk file
#   (CTD_chem_gene_ixns.tsv.gz) for chemical-gene interactions matching
#   run proteins. Queries by both mouse Entrez IDs and human ortholog Entrez
#   IDs. Tracks query_organism (mouse/human) per hit. Fully offline after
#   the initial manual download.
#
#   Algorithm:
#     1. Load mapping table, extract Entrez IDs (mouse + human orthologs)
#     2. Locate CTD bulk file in cache directory
#     3. Read the TSV (streaming, to handle the ~60M row file efficiently)
#     4. Filter rows by GeneID column matching our Entrez ID sets
#     5. Map matched rows back to protein_id via Entrez ID lookup
#     6. Label each hit with query_organism = "mouse" or "human"
#     7. Assemble output DataFrame, write Parquet
#
#   NOTE: The CTD bulk file must be manually downloaded before first use.
#   Download from: http://ctdbase.org/downloads/
#   File: CTD_chem_gene_ixns.tsv.gz
#   Place in the cache directory specified by --cachedir.
#
# inputs:
#   --mapping   {run_id}.id_mapping.parquet (Module 01 UNIPROT_MAPPING)
#   --params    {run_id}_params.yml
#   --run-id    Run identifier prefix for output files
#   --cachedir  Application-level cache directory (must contain CTD bulk file)
#   --outdir    Output directory
#
# outputs:
#   {run_id}.ctd_interactions.parquet
#     Columns: protein_id, gene_symbol_queried, query_organism, chemical_name,
#              chemical_mesh_id, chemical_cas_rn, interaction_text,
#              interaction_actions, n_publications, pmids, ctd_query_status
#
# usage example:
#   query_ctd.py --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \
#                --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \
#                --run-id CTXcyto_WT_vs_CTXcyto_KO \
#                --cachedir ./prosift_cache/databases/ctd \
#                --outdir .
#
#   copy/paste: query_ctd.py --mapping IN.parquet --params params.yml --run-id RUN --cachedir ./cache/ctd --outdir .

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from prosift_cache import BulkFileCache, is_database_enabled, load_db_params

# ============================================================
# CONSTANTS
# ============================================================

# Expected filename for the CTD chemical-gene interactions bulk download
CTD_FILENAME = 'CTD_chem_gene_ixns.tsv.gz'

# CTD bulk file column names (in order). The file uses # comment headers,
# then the actual data rows are tab-separated with these fields.
# Reference: http://ctdbase.org/downloads/;jsessionid=...
CTD_COLUMNS = [
    'ChemicalName',
    'ChemicalID',       # MeSH ID (prefixed with "MESH:")
    'CasRN',
    'GeneSymbol',
    'GeneID',           # NCBI Entrez Gene ID
    'GeneForms',
    'Organism',
    'OrganismID',       # NCBI Taxonomy ID
    'Interaction',
    'InteractionActions',
    'PubMedIDs',
]

# NCBI Taxonomy IDs for organism identification
TAXID_MOUSE = 10090
TAXID_HUMAN = 9606


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='query_ctd.py',
        description='ProSIFT Module 06 QUERY_CTD: chemical-gene interactions from CTD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  query_ctd.py \\\n'
            '    --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \\\n'
            '    --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \\\n'
            '    --run-id CTXcyto_WT_vs_CTXcyto_KO \\\n'
            '    --cachedir ./prosift_cache/databases/ctd \\\n'
            '    --outdir .\n'
        ),
    )
    parser.add_argument('--mapping',  required=True, help='Module 01 id_mapping.parquet')
    parser.add_argument('--params',   required=True, help='Run params.yml')
    parser.add_argument('--run-id',   required=True, dest='run_id', help='Run identifier')
    parser.add_argument('--cachedir', required=True, help='Cache directory with CTD bulk file')
    parser.add_argument('--outdir',   required=True, help='Output directory')
    return parser.parse_args()


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


# ============================================================
# CTD BULK FILE PROCESSING
# ============================================================

def build_entrez_lookups(
    mapping: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, list[str]], set[str]]:
    """Build Entrez ID -> protein_id lookup dicts for mouse and human.

    Parameters
    ----------
    mapping : pd.DataFrame
        Module 01 mapping table.

    Returns
    -------
    mouse_lookup : dict
        {entrez_id_str: [protein_id, ...]} for mouse genes.
    human_lookup : dict
        {entrez_id_str: [protein_id, ...]} for human orthologs.
    all_entrez : set
        Union of all Entrez IDs (as strings) for fast row filtering.
    """
    mouse_lookup: dict[str, list[str]] = {}
    human_lookup: dict[str, list[str]] = {}
    all_entrez: set[str] = set()

    for _, row in mapping.iterrows():
        pid = row['protein_id']

        # Mouse Entrez ID
        mouse_eid = row.get('entrez_id_mouse')
        if pd.notna(mouse_eid):
            eid_str = str(int(mouse_eid))
            mouse_lookup.setdefault(eid_str, []).append(pid)
            all_entrez.add(eid_str)

        # Human ortholog Entrez ID
        human_eid = row.get('human_ortholog_entrez')
        if pd.notna(human_eid):
            eid_str = str(int(human_eid))
            human_lookup.setdefault(eid_str, []).append(pid)
            all_entrez.add(eid_str)

    return mouse_lookup, human_lookup, all_entrez


def filter_ctd_file(
    ctd_path: Path,
    mouse_lookup: dict[str, list[str]],
    human_lookup: dict[str, list[str]],
    all_entrez: set[str],
) -> list[dict]:
    """Read and filter the CTD bulk file for matching interactions.

    Reads the gzipped TSV in chunks for memory efficiency. Filters rows
    where GeneID matches any of our mouse or human Entrez IDs.

    Parameters
    ----------
    ctd_path : Path
        Path to CTD_chem_gene_ixns.tsv.gz.
    mouse_lookup : dict
        Mouse Entrez ID -> protein_id list.
    human_lookup : dict
        Human Entrez ID -> protein_id list.
    all_entrez : set
        All Entrez IDs for fast pre-filtering.

    Returns
    -------
    list of dict
        Matched interaction rows with protein_id and query_organism added.
    """
    matched_rows = []
    chunk_size = 100_000
    total_rows_read = 0

    logging.info('Reading CTD bulk file: %s', ctd_path.name)

    # Read in chunks. The CTD file has comment lines starting with '#'
    # at the top, which we skip.
    reader = pd.read_csv(
        ctd_path,
        sep='\t',
        comment='#',
        header=None,
        names=CTD_COLUMNS,
        dtype={'GeneID': str, 'OrganismID': str, 'PubMedIDs': str, 'CasRN': str},
        chunksize=chunk_size,
        compression='gzip',
        on_bad_lines='skip',
    )

    for chunk in reader:
        total_rows_read += len(chunk)

        # Fast pre-filter: only keep rows where GeneID is in our set
        mask = chunk['GeneID'].isin(all_entrez)
        hits = chunk[mask]

        for _, row in hits.iterrows():
            gene_id = str(row['GeneID'])
            organism_id = str(row.get('OrganismID', ''))

            # Determine query_organism and look up protein_ids
            # A single CTD row can match multiple proteins if the same
            # Entrez ID maps to multiple protein_ids (rare but possible)
            protein_ids_and_organisms = []

            if gene_id in mouse_lookup and organism_id == str(TAXID_MOUSE):
                for pid in mouse_lookup[gene_id]:
                    protein_ids_and_organisms.append((pid, 'mouse', row.get('GeneSymbol', '')))
            if gene_id in human_lookup and organism_id == str(TAXID_HUMAN):
                for pid in human_lookup[gene_id]:
                    protein_ids_and_organisms.append((pid, 'human', row.get('GeneSymbol', '')))

            # Also match by gene ID alone (cross-species CTD entries without
            # strict organism filtering - only if the organism matches expected)
            # If neither mouse nor human organism matched, check if the gene_id
            # is in either lookup regardless of organism
            if not protein_ids_and_organisms:
                if gene_id in mouse_lookup:
                    for pid in mouse_lookup[gene_id]:
                        protein_ids_and_organisms.append((pid, 'mouse', row.get('GeneSymbol', '')))
                if gene_id in human_lookup:
                    for pid in human_lookup[gene_id]:
                        protein_ids_and_organisms.append((pid, 'human', row.get('GeneSymbol', '')))

            # Count publications from the PubMedIDs field (pipe-delimited)
            pmids_raw = row.get('PubMedIDs', '')
            pmids_str = str(pmids_raw) if pd.notna(pmids_raw) else ''
            pmid_list = [p.strip() for p in pmids_str.split('|') if p.strip()] if pmids_str else []
            n_pubs = len(pmid_list)
            pmids_out = '; '.join(pmid_list) if pmid_list else None

            # Clean up MeSH ID (CTD prefixes with "MESH:")
            mesh_id = str(row.get('ChemicalID', ''))
            if mesh_id.startswith('MESH:'):
                mesh_id = mesh_id[5:]

            # CAS RN
            cas_rn = row.get('CasRN')
            cas_rn = str(cas_rn) if pd.notna(cas_rn) and str(cas_rn).strip() else None

            for pid, organism, gene_sym in protein_ids_and_organisms:
                matched_rows.append({
                    'protein_id': pid,
                    'gene_symbol_queried': gene_sym,
                    'query_organism': organism,
                    'chemical_name': row.get('ChemicalName'),
                    'chemical_mesh_id': mesh_id,
                    'chemical_cas_rn': cas_rn,
                    'interaction_text': row.get('Interaction'),
                    'interaction_actions': row.get('InteractionActions'),
                    'n_publications': n_pubs,
                    'pmids': pmids_out,
                    'ctd_query_status': 'success',
                })

    logging.info('CTD file: %d total rows read, %d interactions matched',
                 total_rows_read, len(matched_rows))

    return matched_rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()

    logging.info('QUERY_CTD -- %s', args.run_id)

    # --- Load inputs ---
    mapping = pd.read_parquet(args.mapping)
    # Compatibility: rename input_id -> protein_id if needed (pre-rename mapping tables)
    if 'input_id' in mapping.columns and 'protein_id' not in mapping.columns:
        mapping = mapping.rename(columns={'input_id': 'protein_id'})
    logging.info('Loaded mapping table: %d proteins', len(mapping))

    with open(args.params) as fh:
        params = yaml.safe_load(fh)

    db_params = load_db_params(params)

    # --- Check if this database is enabled ---
    if not is_database_enabled(db_params, 'ctd'):
        logging.info('CTD queries disabled in databases.enabled - emitting empty output')
        out = pd.DataFrame({
            'protein_id': mapping['protein_id'],
            'gene_symbol_queried': pd.Series([None] * len(mapping), dtype='object'),
            'query_organism': pd.Series([None] * len(mapping), dtype='object'),
            'chemical_name': pd.Series([None] * len(mapping), dtype='object'),
            'chemical_mesh_id': pd.Series([None] * len(mapping), dtype='object'),
            'chemical_cas_rn': pd.Series([None] * len(mapping), dtype='object'),
            'interaction_text': pd.Series([None] * len(mapping), dtype='object'),
            'interaction_actions': pd.Series([None] * len(mapping), dtype='object'),
            'n_publications': pd.Series([None] * len(mapping), dtype='Int64'),
            'pmids': pd.Series([None] * len(mapping), dtype='object'),
            'ctd_query_status': 'disabled',
        })
        outpath = Path(args.outdir) / f'{args.run_id}.ctd_interactions.parquet'
        out.to_parquet(outpath, index=False)
        logging.info('Wrote %s (%d rows)', outpath.name, len(out))
        return

    # --- Check for CTD bulk file ---
    bulk_cache = BulkFileCache(
        cache_dir=args.cachedir,
        cache_days=db_params['cache_days'],
        force_requery=db_params['force_requery'],
        database_name='ctd',
    )

    ctd_path = bulk_cache.file_path(CTD_FILENAME)
    if not ctd_path.exists():
        logging.error(
            'CTD bulk file not found: %s\n'
            'Download it manually from http://ctdbase.org/downloads/\n'
            'File: %s\n'
            'Place it in: %s',
            ctd_path, CTD_FILENAME, args.cachedir,
        )
        # Emit output with all proteins marked as error
        out = pd.DataFrame({
            'protein_id': mapping['protein_id'],
            'gene_symbol_queried': pd.Series([None] * len(mapping), dtype='object'),
            'query_organism': pd.Series([None] * len(mapping), dtype='object'),
            'chemical_name': pd.Series([None] * len(mapping), dtype='object'),
            'chemical_mesh_id': pd.Series([None] * len(mapping), dtype='object'),
            'chemical_cas_rn': pd.Series([None] * len(mapping), dtype='object'),
            'interaction_text': pd.Series([None] * len(mapping), dtype='object'),
            'interaction_actions': pd.Series([None] * len(mapping), dtype='object'),
            'n_publications': pd.Series([None] * len(mapping), dtype='Int64'),
            'pmids': pd.Series([None] * len(mapping), dtype='object'),
            'ctd_query_status': 'error',
        })
        outpath = Path(args.outdir) / f'{args.run_id}.ctd_interactions.parquet'
        out.to_parquet(outpath, index=False)
        logging.info('Wrote %s with error status (%d rows)', outpath.name, len(out))
        return

    # Check freshness (log warning if stale, but still use it)
    if not bulk_cache.is_fresh(CTD_FILENAME):
        logging.warning(
            'CTD bulk file is older than %d days. Consider re-downloading.',
            db_params['cache_days'],
        )

    # --- Build Entrez ID lookups ---
    mouse_lookup, human_lookup, all_entrez = build_entrez_lookups(mapping)
    logging.info('Entrez IDs: %d mouse, %d human, %d unique total',
                 len(mouse_lookup), len(human_lookup), len(all_entrez))

    if not all_entrez:
        logging.warning('No Entrez IDs found in mapping table - no CTD matches possible')

    # --- Filter CTD file ---
    matched_rows = filter_ctd_file(ctd_path, mouse_lookup, human_lookup, all_entrez)

    # --- Build output ---
    # Start with matched interaction rows
    if matched_rows:
        out = pd.DataFrame(matched_rows)
    else:
        out = pd.DataFrame(columns=[
            'protein_id', 'gene_symbol_queried', 'query_organism',
            'chemical_name', 'chemical_mesh_id', 'chemical_cas_rn',
            'interaction_text', 'interaction_actions', 'n_publications',
            'pmids', 'ctd_query_status',
        ])

    # Add rows for proteins with no interactions
    proteins_with_hits = set(out['protein_id'].unique()) if len(out) > 0 else set()
    no_hit_rows = []
    for pid in mapping['protein_id']:
        if pid not in proteins_with_hits:
            no_hit_rows.append({
                'protein_id': pid,
                'gene_symbol_queried': None,
                'query_organism': None,
                'chemical_name': None,
                'chemical_mesh_id': None,
                'chemical_cas_rn': None,
                'interaction_text': None,
                'interaction_actions': None,
                'n_publications': None,
                'pmids': None,
                'ctd_query_status': 'no_interactions',
            })

    if no_hit_rows:
        no_hit_df = pd.DataFrame(no_hit_rows)
        out = pd.concat([out, no_hit_df], ignore_index=True)

    # --- Enforce column types ---
    out['n_publications'] = out['n_publications'].astype('Int64')

    # --- Write output ---
    outpath = Path(args.outdir) / f'{args.run_id}.ctd_interactions.parquet'
    out.to_parquet(outpath, index=False)

    # --- Summary ---
    status_counts = out['ctd_query_status'].value_counts()
    logging.info('Output: %d rows, %d columns', len(out), len(out.columns))
    for status, count in status_counts.items():
        logging.info('  %s: %d', status, count)

    if 'success' in status_counts.index:
        organism_counts = out[out['ctd_query_status'] == 'success']['query_organism'].value_counts()
        for org, count in organism_counts.items():
            logging.info('  Interactions from %s genes: %d', org, count)

    n_proteins_with_hits = len(proteins_with_hits)
    logging.info('  Proteins with at least one interaction: %d / %d',
                 n_proteins_with_hits, len(mapping))

    logging.info('Wrote %s', outpath.name)
    logging.info('QUERY_CTD complete')


if __name__ == '__main__':
    main()
