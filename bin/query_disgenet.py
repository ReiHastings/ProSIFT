#!/usr/bin/env python3
# title: query_disgenet.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-07
# last modified: 2026-04-07
#
# purpose:
#   Module 06 QUERY_DISGENET process. Queries DisGeNET REST API for gene-disease
#   associations via human orthologs. Proteins without orthologs are flagged
#   "no_ortholog" and skipped. Rate limited (50/min on Free Academic plan).
#   Requires API key via environment variable. Output-time min_score filtering
#   (raw responses cached unfiltered).
#
#   Algorithm:
#     1. Load mapping table, apply ortholog gating
#     2. Check API key at startup (fail clearly if missing)
#     3. Check per-gene cache; collect genes needing fresh queries
#     4. Query DisGeNET REST API per human ortholog (rate limited)
#     5. Cache raw API responses (unfiltered) per Entrez ID
#     6. Apply min_score filtering at output time
#     7. Assemble output DataFrame, write Parquet
#
# inputs:
#   --mapping   {run_id}.id_mapping.parquet (Module 01 UNIPROT_MAPPING)
#   --params    {run_id}_params.yml
#   --run-id    Run identifier prefix for output files
#   --cachedir  Application-level cache directory for DisGeNET responses
#   --outdir    Output directory
#
# outputs:
#   {run_id}.disgenet_associations.parquet
#     Columns: protein_id, human_symbol_queried, disease_id, disease_name,
#              disease_type, gda_score, evidence_index, n_publications,
#              source, disgenet_query_status
#
# usage example:
#   query_disgenet.py --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \
#                     --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \
#                     --run-id CTXcyto_WT_vs_CTXcyto_KO \
#                     --cachedir ./prosift_cache/databases/disgenet \
#                     --outdir .
#
#   copy/paste: query_disgenet.py --mapping IN.parquet --params params.yml --run-id RUN --cachedir ./cache/disgenet --outdir .

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from prosift_cache import ProteinCache, is_database_enabled, load_db_params

# ============================================================
# CONSTANTS
# ============================================================

# DisGeNET migrated to api.disgenet.com in 2025. The old www.disgenet.org/api
# endpoint returns HTML. The v1 API uses the raw API key in the Authorization
# header (no "Bearer" prefix) and gene_ncbi_id as a query parameter.
DISGENET_API_URL = 'https://api.disgenet.com/api/v1/gda/summary'

# Rate limit: 50 queries/min on Free Academic plan
RATE_LIMIT_PER_MIN = 50
MIN_INTERVAL = 60.0 / RATE_LIMIT_PER_MIN  # 1.2 seconds

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 10  # longer initial delay for DisGeNET

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='query_disgenet.py',
        description='ProSIFT Module 06 QUERY_DISGENET: gene-disease associations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  query_disgenet.py \\\n'
            '    --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \\\n'
            '    --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \\\n'
            '    --run-id CTXcyto_WT_vs_CTXcyto_KO \\\n'
            '    --cachedir ./prosift_cache/databases/disgenet \\\n'
            '    --outdir .\n'
        ),
    )
    parser.add_argument('--mapping',  required=True, help='Module 01 id_mapping.parquet')
    parser.add_argument('--params',   required=True, help='Run params.yml')
    parser.add_argument('--run-id',   required=True, dest='run_id', help='Run identifier')
    parser.add_argument('--cachedir', required=True, help='Cache directory for DisGeNET responses')
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
# DisGeNET API
# ============================================================

class DisGeNETClient:
    """Rate-limited DisGeNET REST API client."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._last_request_time = 0.0
        self._request_count = 0

    def query_gene(self, gene_id: str) -> list | None:
        """Query DisGeNET for gene-disease associations.

        Parameters
        ----------
        gene_id : str
            NCBI Entrez Gene ID (human).

        Returns
        -------
        list or None
            List of association dicts from the API, or None on failure.
        """
        self._rate_limit_wait()

        headers = {
            'Authorization': self.api_key,
            'accept': 'application/json',
        }

        params = {'gene_ncbi_id': gene_id}

        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    DISGENET_API_URL,
                    headers=headers,
                    params=params,
                    timeout=30,
                )

                # Handle rate limiting (HTTP 429)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', 60))
                    logging.warning('Rate limited by DisGeNET. Waiting %d seconds...', retry_after)
                    time.sleep(retry_after)
                    self._rate_limit_wait()
                    continue

                # Check for auth errors before raise_for_status
                if resp.status_code in (401, 403):
                    try:
                        err_body = resp.json()
                        err_msg = err_body.get('message') or err_body.get('payload', {}).get('message', '')
                    except Exception:
                        err_msg = resp.text[:200]
                    logging.error('DisGeNET auth error (%d): %s', resp.status_code, err_msg)
                    logging.error('Check your API key and account status at https://www.disgenet.com')
                    return None

                resp.raise_for_status()
                self._request_count += 1
                data = resp.json()

                # DisGeNET returns a list of associations or an empty response
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'payload' in data:
                    payload = data['payload']
                    return payload if isinstance(payload, list) else []
                else:
                    return []

            except requests.RequestException as exc:
                logging.warning('DisGeNET request failed for gene %s (attempt %d/%d): %s',
                                gene_id, attempt, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2
                    self._rate_limit_wait()

        return None

    def _rate_limit_wait(self) -> None:
        """Sleep if needed to respect the 50/min rate limit."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        self._last_request_time = time.time()


def parse_disgenet_associations(raw: list) -> list[dict]:
    """Parse raw DisGeNET API v1 response into structured rows.

    The v1 API (api.disgenet.com) uses camelCase field names:
      diseaseUMLSCUI, diseaseName, diseaseType, score, ei, numPMIDs, el

    Parameters
    ----------
    raw : list
        Raw API response (list of association dicts from /gda/summary).

    Returns
    -------
    list of dict
        Parsed associations with fields matching the output schema.
    """
    associations = []

    for assoc in raw:
        # Disease identifier: UMLS CUI is the primary ID in DisGeNET
        disease_id = assoc.get('diseaseUMLSCUI')

        disease_name = assoc.get('diseaseName')

        # Disease type: e.g., "[disease]", "[phenotype]"
        disease_type = assoc.get('diseaseType')

        # GDA score (0-1)
        gda_score = assoc.get('score')

        # Evidence index (0-1)
        evidence_index = assoc.get('ei')

        # Number of supporting publications
        n_pubs = assoc.get('numPMIDs', 0)

        # Evidence level (e.g., "Definitive", "Limited")
        source = assoc.get('el')

        associations.append({
            'disease_id': disease_id,
            'disease_name': disease_name,
            'disease_type': disease_type,
            'gda_score': float(gda_score) if gda_score is not None else None,
            'evidence_index': float(evidence_index) if evidence_index is not None else None,
            'n_publications': int(n_pubs) if n_pubs is not None else 0,
            'source': source,
        })

    return associations


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()

    logging.info('QUERY_DISGENET -- %s', args.run_id)

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
    if not is_database_enabled(db_params, 'disgenet'):
        logging.info('DisGeNET queries disabled in databases.enabled - emitting empty output')
        out = pd.DataFrame({
            'protein_id': mapping['protein_id'],
            'human_symbol_queried': pd.Series([None] * len(mapping), dtype='object'),
            'disease_id': pd.Series([None] * len(mapping), dtype='object'),
            'disease_name': pd.Series([None] * len(mapping), dtype='object'),
            'disease_type': pd.Series([None] * len(mapping), dtype='object'),
            'gda_score': pd.Series([None] * len(mapping), dtype='Float64'),
            'evidence_index': pd.Series([None] * len(mapping), dtype='Float64'),
            'n_publications': pd.Series([None] * len(mapping), dtype='Int64'),
            'source': pd.Series([None] * len(mapping), dtype='object'),
            'disgenet_query_status': 'disabled',
        })
        outpath = Path(args.outdir) / f'{args.run_id}.disgenet_associations.parquet'
        out.to_parquet(outpath, index=False)
        logging.info('Wrote %s (%d rows)', outpath.name, len(out))
        return

    # --- Check API key ---
    api_key_env = db_params['api_keys']['disgenet']
    api_key = os.environ.get(api_key_env)

    if not api_key:
        logging.error(
            'DisGeNET API key not found. Set the %s environment variable.\n'
            'Register at https://www.disgenet.com/signup for a free academic account.\n'
            'Emitting output with "no_api_key" status.',
            api_key_env,
        )
        # Emit output with no_api_key status (don't crash the pipeline)
        out = pd.DataFrame({
            'protein_id': mapping['protein_id'],
            'human_symbol_queried': pd.Series([None] * len(mapping), dtype='object'),
            'disease_id': pd.Series([None] * len(mapping), dtype='object'),
            'disease_name': pd.Series([None] * len(mapping), dtype='object'),
            'disease_type': pd.Series([None] * len(mapping), dtype='object'),
            'gda_score': pd.Series([None] * len(mapping), dtype='Float64'),
            'evidence_index': pd.Series([None] * len(mapping), dtype='Float64'),
            'n_publications': pd.Series([None] * len(mapping), dtype='Int64'),
            'source': pd.Series([None] * len(mapping), dtype='object'),
            'disgenet_query_status': 'no_api_key',
        })
        outpath = Path(args.outdir) / f'{args.run_id}.disgenet_associations.parquet'
        out.to_parquet(outpath, index=False)
        logging.info('Wrote %s (%d rows, no_api_key)', outpath.name, len(out))
        return

    # --- Initialize client and cache ---
    client = DisGeNETClient(api_key=api_key)
    cache = ProteinCache(
        cache_dir=args.cachedir,
        cache_days=db_params['cache_days'],
        force_requery=db_params['force_requery'],
        database_name='disgenet',
    )
    min_score = db_params['disgenet']['min_score']
    logging.info('min_score filter: %.2f (applied at output time)', min_score)

    # --- Ortholog gating ---
    # Skip both confirmed 'no_ortholog' and 'ortholog_lookup_failed' (BioMart
    # outage / schema drift); neither carries a usable human_ortholog_entrez.
    skip_statuses = {'no_ortholog', 'ortholog_lookup_failed'}
    has_ortholog = ~mapping['ortholog_mapping_status'].isin(skip_statuses)
    queryable = mapping[has_ortholog]
    skipped = mapping[~has_ortholog]
    logging.info('Queryable (have ortholog): %d, Skipped (no ortholog): %d',
                 len(queryable), len(skipped))

    # --- Build gene ID lookup ---
    # Use human_ortholog_entrez as the query ID. Fall back to human_ortholog_symbol
    # for genes where Entrez ID is not available (rare).
    gene_to_pids: dict[str, list[str]] = {}
    pid_to_gene: dict[str, str] = {}
    pid_to_symbol: dict[str, str] = {}

    for _, row in queryable.iterrows():
        pid = row['protein_id']
        entrez = row.get('human_ortholog_entrez')
        symbol = row.get('human_ortholog_symbol')

        if pd.notna(entrez):
            gene_id = str(int(entrez))
        elif pd.notna(symbol):
            gene_id = symbol  # use symbol as fallback
        else:
            continue

        gene_to_pids.setdefault(gene_id, []).append(pid)
        pid_to_gene[pid] = gene_id
        if pd.notna(symbol):
            pid_to_symbol[pid] = symbol

    unique_genes = list(gene_to_pids.keys())
    logging.info('Unique gene IDs to query: %d', len(unique_genes))

    # --- Check cache ---
    cached_data: dict[str, list] = {}
    to_query: list[str] = []

    for gene_id in unique_genes:
        hit = cache.get(gene_id)
        if hit is not None:
            cached_data[gene_id] = hit
        else:
            to_query.append(gene_id)

    logging.info('Cache: %d hits, %d to query', len(cached_data), len(to_query))

    # --- Query DisGeNET API ---
    api_results: dict[str, list | None] = {}
    n_errors = 0

    if to_query:
        logging.info('Querying DisGeNET API for %d genes (estimated %.0f min at 50/min)...',
                     len(to_query), len(to_query) / RATE_LIMIT_PER_MIN)
        for i, gene_id in enumerate(to_query):
            if (i + 1) % 50 == 0 or i == 0:
                logging.info('  Progress: %d / %d', i + 1, len(to_query))

            raw = client.query_gene(gene_id)
            if raw is None:
                n_errors += 1
                api_results[gene_id] = None
                continue

            # Cache the raw response (unfiltered)
            cache.put(gene_id, raw)
            api_results[gene_id] = raw

        logging.info('  Done. %d successful, %d errors, %d total API calls',
                     len(to_query) - n_errors, n_errors, client._request_count)

    # --- Merge cached + fresh ---
    all_results = {**cached_data, **api_results}

    # --- Build output DataFrame ---
    rows = []

    # Proteins with orthologs
    for pid, gene_id in pid_to_gene.items():
        symbol = pid_to_symbol.get(pid)
        raw = all_results.get(gene_id)

        if raw is None:
            rows.append({
                'protein_id': pid,
                'human_symbol_queried': symbol,
                'disease_id': None, 'disease_name': None, 'disease_type': None,
                'gda_score': None, 'evidence_index': None, 'n_publications': None,
                'source': None, 'disgenet_query_status': 'error',
            })
        elif not raw:
            rows.append({
                'protein_id': pid,
                'human_symbol_queried': symbol,
                'disease_id': None, 'disease_name': None, 'disease_type': None,
                'gda_score': None, 'evidence_index': None, 'n_publications': None,
                'source': None, 'disgenet_query_status': 'no_associations',
            })
        else:
            parsed = parse_disgenet_associations(raw)
            # Apply min_score filter at output time
            filtered = [a for a in parsed if a.get('gda_score') is not None
                        and a['gda_score'] >= min_score]

            if not filtered:
                rows.append({
                    'protein_id': pid,
                    'human_symbol_queried': symbol,
                    'disease_id': None, 'disease_name': None, 'disease_type': None,
                    'gda_score': None, 'evidence_index': None, 'n_publications': None,
                    'source': None, 'disgenet_query_status': 'no_associations',
                })
            else:
                for assoc in filtered:
                    rows.append({
                        'protein_id': pid,
                        'human_symbol_queried': symbol,
                        'disgenet_query_status': 'success',
                        **assoc,
                    })

    # Proteins with ortholog but not in gene lookup (no entrez or symbol)
    queryable_not_queried = queryable[~queryable['protein_id'].isin(pid_to_gene)]
    for _, row in queryable_not_queried.iterrows():
        rows.append({
            'protein_id': row['protein_id'],
            'human_symbol_queried': None,
            'disease_id': None, 'disease_name': None, 'disease_type': None,
            'gda_score': None, 'evidence_index': None, 'n_publications': None,
            'source': None, 'disgenet_query_status': 'no_symbol',
        })

    # Proteins without orthologs
    for _, row in skipped.iterrows():
        rows.append({
            'protein_id': row['protein_id'],
            'human_symbol_queried': None,
            'disease_id': None, 'disease_name': None, 'disease_type': None,
            'gda_score': None, 'evidence_index': None, 'n_publications': None,
            'source': None, 'disgenet_query_status': 'no_ortholog',
        })

    out = pd.DataFrame(rows)

    # --- Enforce column order and types ---
    col_order = [
        'protein_id', 'human_symbol_queried', 'disease_id', 'disease_name',
        'disease_type', 'gda_score', 'evidence_index', 'n_publications',
        'source', 'disgenet_query_status',
    ]
    out = out[col_order]
    out['gda_score'] = out['gda_score'].astype('Float64')
    out['evidence_index'] = out['evidence_index'].astype('Float64')
    out['n_publications'] = out['n_publications'].astype('Int64')

    # --- Write output ---
    outpath = Path(args.outdir) / f'{args.run_id}.disgenet_associations.parquet'
    out.to_parquet(outpath, index=False)

    # --- Summary ---
    status_counts = out['disgenet_query_status'].value_counts()
    logging.info('Output: %d rows, %d columns', len(out), len(out.columns))
    for status, count in status_counts.items():
        logging.info('  %s: %d', status, count)
    logging.info(cache.summary())
    logging.info('Wrote %s', outpath.name)
    logging.info('QUERY_DISGENET complete')


if __name__ == '__main__':
    main()
