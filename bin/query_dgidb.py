#!/usr/bin/env python3
# title: query_dgidb.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-07
# last modified: 2026-04-07
#
# purpose:
#   Module 06 QUERY_DGIDB process. Queries DGIdb GraphQL API (v5.0) for
#   drug-gene interactions via human orthologs. Proteins without orthologs
#   are flagged "no_ortholog" and skipped. No API key required. No documented
#   rate limits.
#
#   Algorithm:
#     1. Load mapping table, apply ortholog gating
#     2. Check per-gene cache; collect symbols needing fresh queries
#     3. Query DGIdb GraphQL API per gene symbol
#     4. Parse response: drug name, interaction type, score, approval, sources, PMIDs
#     5. Cache raw API responses per gene symbol
#     6. Assemble output DataFrame, write Parquet
#
# inputs:
#   --mapping   {run_id}.id_mapping.parquet (Module 01 UNIPROT_MAPPING)
#   --params    {run_id}_params.yml
#   --run-id    Run identifier prefix for output files
#   --cachedir  Application-level cache directory for DGIdb responses
#   --outdir    Output directory
#
# outputs:
#   {run_id}.dgidb_interactions.parquet
#     Columns: protein_id, human_symbol_queried, drug_name, drug_concept_id,
#              interaction_type, interaction_score, approval_status,
#              n_sources, sources, pmids, dgidb_query_status
#
# usage example:
#   query_dgidb.py --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \
#                  --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \
#                  --run-id CTXcyto_WT_vs_CTXcyto_KO \
#                  --cachedir ./prosift_cache/databases/dgidb \
#                  --outdir .
#
#   copy/paste: query_dgidb.py --mapping IN.parquet --params params.yml --run-id RUN --cachedir ./cache/dgidb --outdir .

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from prosift_cache import ProteinCache, is_database_enabled, load_db_params

# ============================================================
# CONSTANTS
# ============================================================

DGIDB_GRAPHQL_URL = 'https://dgidb.org/api/graphql'

# GraphQL query for drug-gene interactions. Accepts a single gene name.
# Returns drug interactions with drug name, concept ID, interaction type,
# score, approval status, sources, and PMIDs.
DGIDB_QUERY = '''
query ($geneName: String!) {
  genes(names: [$geneName]) {
    nodes {
      name
      interactions {
        drug {
          name
          conceptId
          approved
        }
        interactionTypes {
          type
          directionality
        }
        interactionScore
        interactionClaims {
          source {
            fullName
          }
          publications {
            pmid
          }
        }
      }
    }
  }
}
'''

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='query_dgidb.py',
        description='ProSIFT Module 06 QUERY_DGIDB: drug-gene interactions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  query_dgidb.py \\\n'
            '    --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \\\n'
            '    --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \\\n'
            '    --run-id CTXcyto_WT_vs_CTXcyto_KO \\\n'
            '    --cachedir ./prosift_cache/databases/dgidb \\\n'
            '    --outdir .\n'
        ),
    )
    parser.add_argument('--mapping',  required=True, help='Module 01 id_mapping.parquet')
    parser.add_argument('--params',   required=True, help='Run params.yml')
    parser.add_argument('--run-id',   required=True, dest='run_id', help='Run identifier')
    parser.add_argument('--cachedir', required=True, help='Cache directory for DGIdb responses')
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
# DGIdb GraphQL API
# ============================================================

def query_dgidb_gene(gene_symbol: str) -> dict | None:
    """Query DGIdb GraphQL API for a single gene.

    Returns the raw API response as a dict, or None on failure.
    """
    payload = {
        'query': DGIDB_QUERY,
        'variables': {'geneName': gene_symbol},
    }

    delay = INITIAL_RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                DGIDB_GRAPHQL_URL,
                json=payload,
                timeout=30,
                headers={'Content-Type': 'application/json'},
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logging.warning('DGIdb request failed for %s (attempt %d/%d): %s',
                            gene_symbol, attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    return None


def parse_dgidb_response(raw: dict, gene_symbol: str) -> list[dict]:
    """Parse a DGIdb GraphQL response into interaction rows.

    Parameters
    ----------
    raw : dict
        Raw GraphQL response JSON.
    gene_symbol : str
        The gene symbol that was queried.

    Returns
    -------
    list of dict
        One dict per drug interaction, with fields matching the output schema.
        Empty list if the gene has no interactions.
    """
    interactions = []

    data = raw.get('data', {})
    genes = data.get('genes', {}).get('nodes', [])

    if not genes:
        return interactions

    gene_node = genes[0]
    for ixn in gene_node.get('interactions', []):
        drug = ixn.get('drug', {})
        drug_name = drug.get('name')
        drug_concept_id = drug.get('conceptId')

        # Approval status
        approved = drug.get('approved')
        if approved is True:
            approval_status = 'approved'
        elif approved is False:
            approval_status = 'not_approved'
        else:
            approval_status = None

        # Interaction type(s) - take the first if multiple
        ixn_types = ixn.get('interactionTypes', [])
        interaction_type = None
        if ixn_types:
            interaction_type = ixn_types[0].get('type')

        # Interaction score
        interaction_score = ixn.get('interactionScore')

        # Sources and PMIDs from interaction claims
        sources_set = set()
        pmids_set = set()
        for claim in ixn.get('interactionClaims', []):
            src = claim.get('source', {})
            src_name = src.get('fullName')
            if src_name:
                sources_set.add(src_name)
            for pub in claim.get('publications', []):
                pmid = pub.get('pmid')
                if pmid:
                    pmids_set.add(str(pmid))

        interactions.append({
            'drug_name': drug_name,
            'drug_concept_id': drug_concept_id,
            'interaction_type': interaction_type,
            'interaction_score': interaction_score,
            'approval_status': approval_status,
            'n_sources': len(sources_set),
            'sources': '; '.join(sorted(sources_set)) if sources_set else None,
            'pmids': '; '.join(sorted(pmids_set)) if pmids_set else None,
        })

    return interactions


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()

    logging.info('QUERY_DGIDB -- %s', args.run_id)

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
    if not is_database_enabled(db_params, 'dgidb'):
        logging.info('DGIdb queries disabled in databases.enabled - emitting empty output')
        out = pd.DataFrame({
            'protein_id': mapping['protein_id'],
            'human_symbol_queried': pd.Series([None] * len(mapping), dtype='object'),
            'drug_name': pd.Series([None] * len(mapping), dtype='object'),
            'drug_concept_id': pd.Series([None] * len(mapping), dtype='object'),
            'interaction_type': pd.Series([None] * len(mapping), dtype='object'),
            'interaction_score': pd.Series([None] * len(mapping), dtype='Float64'),
            'approval_status': pd.Series([None] * len(mapping), dtype='object'),
            'n_sources': pd.Series([None] * len(mapping), dtype='Int64'),
            'sources': pd.Series([None] * len(mapping), dtype='object'),
            'pmids': pd.Series([None] * len(mapping), dtype='object'),
            'dgidb_query_status': 'disabled',
        })
        outpath = Path(args.outdir) / f'{args.run_id}.dgidb_interactions.parquet'
        out.to_parquet(outpath, index=False)
        logging.info('Wrote %s (%d rows)', outpath.name, len(out))
        return

    # --- Initialize cache ---
    cache = ProteinCache(
        cache_dir=args.cachedir,
        cache_days=db_params['cache_days'],
        force_requery=db_params['force_requery'],
        database_name='dgidb',
    )

    # --- Ortholog gating ---
    # Skip both confirmed 'no_ortholog' and 'ortholog_lookup_failed' (BioMart
    # outage / schema drift); neither carries a usable human_ortholog_symbol.
    skip_statuses = {'no_ortholog', 'ortholog_lookup_failed'}
    has_ortholog = ~mapping['ortholog_mapping_status'].isin(skip_statuses)
    queryable = mapping[has_ortholog]
    skipped = mapping[~has_ortholog]
    logging.info('Queryable (have ortholog): %d, Skipped (no ortholog): %d',
                 len(queryable), len(skipped))

    # --- Build protein_id -> human_symbol lookup ---
    # Multiple protein_ids can map to the same human symbol; we query each
    # unique symbol once and fan out the results.
    symbol_to_pids: dict[str, list[str]] = {}
    pid_to_symbol: dict[str, str] = {}
    for _, row in queryable.iterrows():
        pid = row['protein_id']
        sym = row.get('human_ortholog_symbol')
        if pd.notna(sym) and sym:
            symbol_to_pids.setdefault(sym, []).append(pid)
            pid_to_symbol[pid] = sym

    unique_symbols = list(symbol_to_pids.keys())
    logging.info('Unique human ortholog symbols to query: %d', len(unique_symbols))

    # --- Check cache, collect symbols needing queries ---
    cached_data: dict[str, list[dict]] = {}
    to_query: list[str] = []

    for sym in unique_symbols:
        hit = cache.get(sym)
        if hit is not None:
            cached_data[sym] = hit
        else:
            to_query.append(sym)

    logging.info('Cache: %d hits, %d to query', len(cached_data), len(to_query))

    # --- Query DGIdb API ---
    api_results: dict[str, list[dict]] = {}
    n_errors = 0

    if to_query:
        logging.info('Querying DGIdb API for %d gene symbols...', len(to_query))
        for i, sym in enumerate(to_query):
            if (i + 1) % 100 == 0 or i == 0:
                logging.info('  Progress: %d / %d', i + 1, len(to_query))

            raw = query_dgidb_gene(sym)
            if raw is None:
                n_errors += 1
                api_results[sym] = None
                continue

            parsed = parse_dgidb_response(raw, sym)
            api_results[sym] = parsed
            # Cache the raw response (not parsed) so we can re-parse if schema changes
            cache.put(sym, parsed)

        logging.info('  Done. %d successful, %d errors', len(to_query) - n_errors, n_errors)

    # --- Merge cached + fresh results ---
    all_results = {**cached_data, **api_results}

    # --- Build output DataFrame ---
    rows = []

    # Proteins with orthologs: expand interactions
    for pid, sym in pid_to_symbol.items():
        interactions = all_results.get(sym)

        if interactions is None:
            # API error
            rows.append({
                'protein_id': pid,
                'human_symbol_queried': sym,
                'drug_name': None,
                'drug_concept_id': None,
                'interaction_type': None,
                'interaction_score': None,
                'approval_status': None,
                'n_sources': None,
                'sources': None,
                'pmids': None,
                'dgidb_query_status': 'error',
            })
        elif not interactions:
            # Gene found but no drug interactions
            rows.append({
                'protein_id': pid,
                'human_symbol_queried': sym,
                'drug_name': None,
                'drug_concept_id': None,
                'interaction_type': None,
                'interaction_score': None,
                'approval_status': None,
                'n_sources': None,
                'sources': None,
                'pmids': None,
                'dgidb_query_status': 'no_interactions',
            })
        else:
            for ixn in interactions:
                rows.append({
                    'protein_id': pid,
                    'human_symbol_queried': sym,
                    'dgidb_query_status': 'success',
                    **ixn,
                })

    # Proteins with ortholog but no symbol (shouldn't happen, but handle)
    queryable_no_symbol = queryable[~queryable['protein_id'].isin(pid_to_symbol)]
    for _, row in queryable_no_symbol.iterrows():
        rows.append({
            'protein_id': row['protein_id'],
            'human_symbol_queried': None,
            'drug_name': None,
            'drug_concept_id': None,
            'interaction_type': None,
            'interaction_score': None,
            'approval_status': None,
            'n_sources': None,
            'sources': None,
            'pmids': None,
            'dgidb_query_status': 'no_symbol',
        })

    # Proteins without orthologs
    for _, row in skipped.iterrows():
        rows.append({
            'protein_id': row['protein_id'],
            'human_symbol_queried': None,
            'drug_name': None,
            'drug_concept_id': None,
            'interaction_type': None,
            'interaction_score': None,
            'approval_status': None,
            'n_sources': None,
            'sources': None,
            'pmids': None,
            'dgidb_query_status': 'no_ortholog',
        })

    out = pd.DataFrame(rows)

    # --- Enforce column order and types ---
    col_order = [
        'protein_id', 'human_symbol_queried', 'drug_name', 'drug_concept_id',
        'interaction_type', 'interaction_score', 'approval_status',
        'n_sources', 'sources', 'pmids', 'dgidb_query_status',
    ]
    out = out[col_order]
    out['interaction_score'] = out['interaction_score'].astype('Float64')
    out['n_sources'] = out['n_sources'].astype('Int64')

    # --- Write output ---
    outpath = Path(args.outdir) / f'{args.run_id}.dgidb_interactions.parquet'
    out.to_parquet(outpath, index=False)

    # --- Summary ---
    status_counts = out['dgidb_query_status'].value_counts()
    logging.info('Output: %d rows, %d columns', len(out), len(out.columns))
    for status, count in status_counts.items():
        logging.info('  %s: %d', status, count)
    logging.info(cache.summary())
    logging.info('Wrote %s', outpath.name)
    logging.info('QUERY_DGIDB complete')


if __name__ == '__main__':
    main()
