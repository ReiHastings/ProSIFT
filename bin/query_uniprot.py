#!/usr/bin/env python3
# title: query_uniprot.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-07
# last modified: 2026-04-07
#
# purpose:
#   Module 06 QUERY_UNIPROT process. Queries UniProt REST API for curated
#   protein annotations: name, function description, subcellular location,
#   tissue expression, and keywords. Uses the /uniprotkb/search endpoint
#   with field selection for efficient batch retrieval. Results are cached
#   per-protein as JSON files for cross-run reuse.
#
#   Algorithm:
#     1. Load mapping table, extract unique UniProt accessions
#     2. Check per-protein cache; collect accessions needing fresh queries
#     3. Batch-query UniProt REST API (/uniprotkb/search) with pagination
#     4. Parse JSON response, extract 5 annotation fields per protein
#     5. Cache raw API responses (unfiltered) as per-protein JSON files
#     6. Assemble output DataFrame with correct schema, write Parquet
#
# inputs:
#   --mapping   {run_id}.id_mapping.parquet (Module 01 UNIPROT_MAPPING)
#   --params    {run_id}_params.yml
#   --run-id    Run identifier prefix for output files
#   --cachedir  Application-level cache directory for UniProt responses
#   --outdir    Output directory
#
# outputs:
#   {run_id}.uniprot_annotations.parquet
#     Columns: protein_id, protein_name, function_description,
#              subcellular_location, tissue_expression, keywords,
#              uniprot_query_status
#
# usage example:
#   query_uniprot.py --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \
#                    --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \
#                    --run-id CTXcyto_WT_vs_CTXcyto_KO \
#                    --cachedir ./prosift_cache/databases/uniprot \
#                    --outdir .
#
#   copy/paste: query_uniprot.py --mapping IN.parquet --params params.yml --run-id RUN --cachedir ./cache/uniprot --outdir .

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

# Shared caching utilities
from prosift_cache import ProteinCache, is_database_enabled, load_db_params

# ============================================================
# CONSTANTS
# ============================================================

UNIPROT_SEARCH_URL = 'https://rest.uniprot.org/uniprotkb/search'

# Fields to request from the UniProt REST API.
# These map to the return_fields parameter in the search endpoint.
UNIPROT_FIELDS = 'protein_name,cc_function,cc_subcellular_location,cc_tissue_specificity,keyword'

# Maximum accessions per query batch. Each accession adds ~25 chars to the
# query string; 100 accessions yields ~2.6K chars, well within URL length limits.
# UniProt's search endpoint uses GET, so the full query must fit in the URL.
BATCH_SIZE = 100
PAGE_SIZE = 500

# Retry configuration for API calls
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='query_uniprot.py',
        description='ProSIFT Module 06 QUERY_UNIPROT: UniProt protein annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  query_uniprot.py \\\n'
            '    --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \\\n'
            '    --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \\\n'
            '    --run-id CTXcyto_WT_vs_CTXcyto_KO \\\n'
            '    --cachedir ./prosift_cache/databases/uniprot \\\n'
            '    --outdir .\n'
        ),
    )
    parser.add_argument('--mapping',  required=True, help='Module 01 id_mapping.parquet')
    parser.add_argument('--params',   required=True, help='Run params.yml')
    parser.add_argument('--run-id',   required=True, dest='run_id', help='Run identifier')
    parser.add_argument('--cachedir', required=True, help='Cache directory for UniProt responses')
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
# UNIPROT API QUERY
# ============================================================

def build_accession_query(accessions: list[str]) -> str:
    """Build a UniProt search query string for a batch of accessions.

    Produces: (accession:Q9WTX5) OR (accession:P12345) OR ...
    """
    terms = [f'(accession:{acc})' for acc in accessions]
    return ' OR '.join(terms)


def query_uniprot_batch(accessions: list[str]) -> dict[str, dict]:
    """Query UniProt REST API for a batch of accessions.

    Returns a dict mapping accession -> raw annotation dict.
    Handles pagination automatically.

    Parameters
    ----------
    accessions : list of str
        UniProt accessions to query.

    Returns
    -------
    dict
        {accession: {protein_name, function_description, subcellular_location,
                     tissue_expression, keywords}} for each found accession.
    """
    query_str = build_accession_query(accessions)
    results = {}

    # Paginate through results
    params = {
        'query': query_str,
        'fields': UNIPROT_FIELDS,
        'format': 'json',
        'size': PAGE_SIZE,
    }

    url = UNIPROT_SEARCH_URL
    page = 0

    while url:
        page += 1
        response = _request_with_retry(url, params if page == 1 else None)
        if response is None:
            logging.error('Failed to retrieve UniProt batch after retries')
            break

        data = response.json()

        # Parse results from this page
        for entry in data.get('results', []):
            acc = entry.get('primaryAccession', '')
            parsed = _parse_uniprot_entry(entry)
            results[acc] = parsed

        # Check for next page via Link header
        url = _get_next_link(response)
        params = None  # params already in the URL for subsequent pages

    return results


def _request_with_retry(url: str, params: dict | None = None) -> requests.Response | None:
    """Make an HTTP GET request with exponential backoff retry.

    Parameters
    ----------
    url : str
        Request URL.
    params : dict or None
        Query parameters (only used for the first page).

    Returns
    -------
    requests.Response or None
        The response, or None if all retries failed.
    """
    delay = INITIAL_RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logging.warning('UniProt request failed (attempt %d/%d): %s',
                            attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    return None


def _get_next_link(response: requests.Response) -> str | None:
    """Extract the 'next' pagination URL from the Link header.

    UniProt REST API uses RFC 5988 Link headers for pagination:
    Link: <https://rest.uniprot.org/...?cursor=...>; rel="next"
    """
    link_header = response.headers.get('Link', '')
    if not link_header:
        return None
    # Parse Link header - look for rel="next"
    for part in link_header.split(','):
        if 'rel="next"' in part:
            # Extract URL between < and >
            start = part.index('<') + 1
            end = part.index('>')
            return part[start:end]
    return None


def _parse_uniprot_entry(entry: dict) -> dict:
    """Extract annotation fields from a UniProt JSON entry.

    Parameters
    ----------
    entry : dict
        A single UniProt entry from the search results JSON.

    Returns
    -------
    dict
        Parsed annotation fields.
    """
    # --- Protein name ---
    protein_name = None
    prot_desc = entry.get('proteinDescription', {})
    rec_name = prot_desc.get('recommendedName')
    if rec_name:
        protein_name = rec_name.get('fullName', {}).get('value')
    elif prot_desc.get('submissionNames'):
        protein_name = prot_desc['submissionNames'][0].get('fullName', {}).get('value')

    # --- Comments (function, subcellular location, tissue specificity) ---
    function_description = None
    subcellular_location = None
    tissue_expression = None

    for comment in entry.get('comments', []):
        ctype = comment.get('commentType', '')

        if ctype == 'FUNCTION':
            # Function comments have 'texts' array
            texts = comment.get('texts', [])
            if texts:
                function_description = '; '.join(t.get('value', '') for t in texts)

        elif ctype == 'SUBCELLULAR LOCATION':
            # Subcellular location has 'subcellularLocations' array
            locs = comment.get('subcellularLocations', [])
            loc_names = []
            for loc in locs:
                loc_val = loc.get('location', {}).get('value')
                if loc_val:
                    loc_names.append(loc_val)
            if loc_names:
                subcellular_location = '; '.join(loc_names)

        elif ctype == 'TISSUE SPECIFICITY':
            texts = comment.get('texts', [])
            if texts:
                tissue_expression = '; '.join(t.get('value', '') for t in texts)

    # --- Keywords ---
    keywords = None
    kw_list = entry.get('keywords', [])
    if kw_list:
        keywords = '; '.join(kw.get('name', '') for kw in kw_list)

    return {
        'protein_name': protein_name,
        'function_description': function_description,
        'subcellular_location': subcellular_location,
        'tissue_expression': tissue_expression,
        'keywords': keywords,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()

    logging.info('QUERY_UNIPROT -- %s', args.run_id)

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
    if not is_database_enabled(db_params, 'uniprot'):
        logging.info('UniProt queries disabled in databases.enabled - emitting empty output')
        out = pd.DataFrame({
            'protein_id': mapping['protein_id'],
            'protein_name': pd.Series([None] * len(mapping), dtype='object'),
            'function_description': pd.Series([None] * len(mapping), dtype='object'),
            'subcellular_location': pd.Series([None] * len(mapping), dtype='object'),
            'tissue_expression': pd.Series([None] * len(mapping), dtype='object'),
            'keywords': pd.Series([None] * len(mapping), dtype='object'),
            'uniprot_query_status': 'disabled',
        })
        outpath = Path(args.outdir) / f'{args.run_id}.uniprot_annotations.parquet'
        out.to_parquet(outpath, index=False)
        logging.info('Wrote %s (%d rows)', outpath.name, len(out))
        return

    # --- Initialize cache ---
    cache = ProteinCache(
        cache_dir=args.cachedir,
        cache_days=db_params['cache_days'],
        force_requery=db_params['force_requery'],
        database_name='uniprot',
    )

    # --- Determine which accessions need querying ---
    unique_accessions = mapping['uniprot_accession'].dropna().unique().tolist()

    # Check cache for each accession
    cached_data = {}     # accession -> parsed annotation dict
    to_query = []        # accessions that need fresh API queries

    for acc in unique_accessions:
        hit = cache.get(acc)
        if hit is not None:
            cached_data[acc] = hit
        else:
            to_query.append(acc)

    logging.info('Cache: %d hits, %d to query (of %d unique accessions)',
                 len(cached_data), len(to_query), len(unique_accessions))

    # --- Query UniProt API in batches ---
    api_results = {}
    if to_query:
        n_batches = (len(to_query) + BATCH_SIZE - 1) // BATCH_SIZE
        logging.info('Querying UniProt API: %d accessions in %d batches',
                     len(to_query), n_batches)

        for i in range(0, len(to_query), BATCH_SIZE):
            batch = to_query[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            logging.info('  Batch %d/%d: %d accessions', batch_num, n_batches, len(batch))

            batch_results = query_uniprot_batch(batch)
            api_results.update(batch_results)

            # Cache each result immediately
            for acc, data in batch_results.items():
                cache.put(acc, data)

        # Log accessions that were queried but not found
        queried_not_found = set(to_query) - set(api_results.keys())
        if queried_not_found:
            logging.info('  %d accessions not found in UniProt', len(queried_not_found))
            # Cache the "not found" result so we don't re-query next time
            for acc in queried_not_found:
                cache.put(acc, {'_not_found': True})

    # --- Merge cached + fresh results ---
    all_results = {**cached_data, **api_results}

    # --- Build output DataFrame ---
    rows = []
    for _, prot in mapping.iterrows():
        pid = prot['protein_id']
        acc = prot.get('uniprot_accession')

        if pd.isna(acc) or acc is None:
            # No accession available (shouldn't happen after Module 01, but handle gracefully)
            rows.append({
                'protein_id': pid,
                'protein_name': None,
                'function_description': None,
                'subcellular_location': None,
                'tissue_expression': None,
                'keywords': None,
                'uniprot_query_status': 'no_accession',
            })
            continue

        data = all_results.get(acc)

        if data is None:
            # Query failed (network error after retries)
            rows.append({
                'protein_id': pid,
                'protein_name': None,
                'function_description': None,
                'subcellular_location': None,
                'tissue_expression': None,
                'keywords': None,
                'uniprot_query_status': 'error',
            })
        elif data.get('_not_found'):
            # Accession not in UniProt
            rows.append({
                'protein_id': pid,
                'protein_name': None,
                'function_description': None,
                'subcellular_location': None,
                'tissue_expression': None,
                'keywords': None,
                'uniprot_query_status': 'not_found',
            })
        else:
            rows.append({
                'protein_id': pid,
                'protein_name': data.get('protein_name'),
                'function_description': data.get('function_description'),
                'subcellular_location': data.get('subcellular_location'),
                'tissue_expression': data.get('tissue_expression'),
                'keywords': data.get('keywords'),
                'uniprot_query_status': 'success',
            })

    out = pd.DataFrame(rows)

    # --- Write output ---
    outpath = Path(args.outdir) / f'{args.run_id}.uniprot_annotations.parquet'
    out.to_parquet(outpath, index=False)

    # --- Summary ---
    status_counts = out['uniprot_query_status'].value_counts()
    logging.info('Output: %d rows, %d columns', len(out), len(out.columns))
    for status, count in status_counts.items():
        logging.info('  %s: %d', status, count)
    logging.info(cache.summary())
    logging.info('Wrote %s', outpath.name)
    logging.info('QUERY_UNIPROT complete')


if __name__ == '__main__':
    main()
