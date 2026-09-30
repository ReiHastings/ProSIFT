#!/usr/bin/env python3
# title: query_pubmed.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-07
# last modified: 2026-07-14  (compound OR/AND search terms honoured in the
#                co-occurrence query, not just the term-total; cache keys made
#                filesystem-safe via cache_token)
#
# purpose:
#   Module 06 QUERY_PUBMED process. Queries NCBI ESearch API for literature
#   co-occurrence of each protein with user-defined search terms. Computes
#   PMI-based normalized scores. Queries both mouse and human gene symbols
#   where available. Rate limited (10/sec with API key, 3/sec without).
#
#   Algorithm:
#     1. Load mapping table, extract gene symbols (mouse + human orthologs)
#     2. Get total PubMed article count (single query, cached per-run)
#     3. For each protein x search term pair:
#        a. Query co-occurrence: "{symbol}"[tiab] AND <term query>, where the
#           term query is "{term}"[tiab] for a plain term or the parenthesised
#           boolean expression for a compound term (see build_term_query)
#        b. Query total pubs for symbol: "{symbol}"[tiab]
#        c. Cache results per {symbol}_<cache_token(term)> key (filesystem-safe)
#     4. Compute PMI = log2(p(protein AND term) / (p(protein) * p(term)))
#        - Skip if total pubs below min_pubs_for_score threshold
#        - Take max(mouse_pmi, human_pmi) as normalized_score
#     5. Assemble output DataFrame, write Parquet
#
# inputs:
#   --mapping   {run_id}.id_mapping.parquet (Module 01 UNIPROT_MAPPING)
#   --params    {run_id}_params.yml
#   --run-id    Run identifier prefix for output files
#   --cachedir  Application-level cache directory for PubMed responses
#   --outdir    Output directory
#
# outputs:
#   {run_id}.pubmed_cooccurrence.parquet
#     Columns: protein_id, search_term, mouse_symbol_used, human_symbol_used,
#              mouse_hit_count, human_hit_count, mouse_total_pubs,
#              human_total_pubs, normalized_score, pubmed_query_status
#
# usage example:
#   query_pubmed.py --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \
#                   --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \
#                   --run-id CTXcyto_WT_vs_CTXcyto_KO \
#                   --cachedir ./prosift_cache/databases/pubmed \
#                   --outdir .
#
#   copy/paste: query_pubmed.py --mapping IN.parquet --params params.yml --run-id RUN --cachedir ./cache/pubmed --outdir .

import argparse
import logging
import math
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from prosift_cache import ProteinCache, is_database_enabled, load_db_params

# ============================================================
# CONSTANTS
# ============================================================

ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'


def build_term_query(term: str) -> str:
    """PubMed query fragment for one search term.

    A *compound* term -- one containing ' AND ' or ' OR ' -- is passed through
    as a parenthesised boolean expression exactly as written, so variant
    spellings can be combined, e.g.
    '"TBI"[tiab] OR "traumatic brain injury"[tiab]'. Add the [tiab] field tag to
    each operand yourself to keep the title/abstract restriction; a bare
    compound like 'TBI OR traumatic brain injury' runs under PubMed's default
    field mapping. A plain term is matched as an exact title/abstract phrase.

    Used for BOTH the per-term total count and the per-protein co-occurrence
    query, so the PMI numerator and denominator share the same term semantics.
    """
    if ' AND ' in term or ' OR ' in term:
        return f'({term})'
    return f'"{term}"[tiab]'


def cache_token(term: str) -> str:
    """Filesystem-safe token for a search term, used only in cache keys.

    The cache stores one JSON file per key, so the key must be a safe filename
    (ProteinCache does not sanitise). A plain single-word term is returned
    unchanged so existing caches stay valid; spaces, quotes, apostrophes, and
    boolean operators in a compound term collapse to underscores. This affects
    the filename only -- the query and the output `search_term` use the raw term.
    """
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', term).strip('_')
    return token or 'term'

# Rate limits: 10/sec with API key, 3/sec without.
# Use slightly conservative values to avoid 429 errors from burst patterns.
RATE_LIMIT_WITH_KEY = 9
RATE_LIMIT_NO_KEY = 2

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='query_pubmed.py',
        description='ProSIFT Module 06 QUERY_PUBMED: PubMed literature co-occurrence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  query_pubmed.py \\\n'
            '    --mapping CTXcyto_WT_vs_CTXcyto_KO.id_mapping.parquet \\\n'
            '    --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \\\n'
            '    --run-id CTXcyto_WT_vs_CTXcyto_KO \\\n'
            '    --cachedir ./prosift_cache/databases/pubmed \\\n'
            '    --outdir .\n'
        ),
    )
    parser.add_argument('--mapping',  required=True, help='Module 01 id_mapping.parquet')
    parser.add_argument('--params',   required=True, help='Run params.yml')
    parser.add_argument('--run-id',   required=True, dest='run_id', help='Run identifier')
    parser.add_argument('--cachedir', required=True, help='Cache directory for PubMed responses')
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
# PubMed ESearch API
# ============================================================

class PubMedClient:
    """Rate-limited PubMed ESearch client.

    Handles API key injection and rate limiting transparently.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.rate_limit = RATE_LIMIT_WITH_KEY if api_key else RATE_LIMIT_NO_KEY
        self.min_interval = 1.0 / self.rate_limit
        self._last_request_time = 0.0
        self._request_count = 0

        if api_key:
            logging.info('PubMed API key found - rate limit: %d/sec', self.rate_limit)
        else:
            logging.warning('No PubMed API key - rate limit: %d/sec (set %s for 10/sec)',
                            self.rate_limit, 'NCBI_API_KEY')

    def esearch_count(self, query: str) -> int | None:
        """Run an ESearch query and return the hit count.

        Parameters
        ----------
        query : str
            PubMed search string (e.g., '"Shank3"[tiab] AND "ketamine"[tiab]').

        Returns
        -------
        int or None
            Number of matching articles, or None on failure.
        """
        self._rate_limit_wait()

        params = {
            'db': 'pubmed',
            'term': query,
            'rettype': 'count',
            'retmode': 'json',
        }
        if self.api_key:
            params['api_key'] = self.api_key

        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(ESEARCH_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                count = int(data['esearchresult']['count'])
                self._request_count += 1
                return count
            except (requests.RequestException, KeyError, ValueError) as exc:
                logging.warning('ESearch failed (attempt %d/%d): %s | query: %s',
                                attempt, MAX_RETRIES, exc, query[:80])
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2
                    self._rate_limit_wait()
        return None

    def _rate_limit_wait(self) -> None:
        """Sleep if needed to respect the rate limit."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()


def compute_pmi(
    cooccurrence_count: int,
    symbol_total: int,
    term_count: int,
    total_articles: int,
) -> float | None:
    """Compute pointwise mutual information (PMI).

    PMI = log2(p(protein AND term) / (p(protein) * p(term)))

    Returns None if any count is zero (log undefined).
    """
    if cooccurrence_count == 0 or symbol_total == 0 or term_count == 0 or total_articles == 0:
        return None

    p_co = cooccurrence_count / total_articles
    p_symbol = symbol_total / total_articles
    p_term = term_count / total_articles

    ratio = p_co / (p_symbol * p_term)
    if ratio <= 0:
        return None

    return math.log2(ratio)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()

    logging.info('QUERY_PUBMED -- %s', args.run_id)

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
    if not is_database_enabled(db_params, 'pubmed'):
        logging.info('PubMed queries disabled in databases.enabled - emitting empty output')
        search_terms = db_params['pubmed']['search_terms'] or ['disabled']
        rows = []
        for _, prot in mapping.iterrows():
            for term in search_terms:
                rows.append({
                    'protein_id': prot['protein_id'],
                    'search_term': term,
                    'mouse_symbol_used': None, 'human_symbol_used': None,
                    'mouse_hit_count': None, 'human_hit_count': None,
                    'mouse_total_pubs': None, 'human_total_pubs': None,
                    'normalized_score': None, 'pubmed_query_status': 'disabled',
                })
        out = pd.DataFrame(rows)
        for c in ['mouse_hit_count', 'human_hit_count', 'mouse_total_pubs', 'human_total_pubs']:
            out[c] = out[c].astype('Int64')
        out['normalized_score'] = out['normalized_score'].astype('Float64')
        outpath = Path(args.outdir) / f'{args.run_id}.pubmed_cooccurrence.parquet'
        out.to_parquet(outpath, index=False)
        logging.info('Wrote %s (%d rows)', outpath.name, len(out))
        return

    # --- Read parameters ---
    search_terms = db_params['pubmed']['search_terms']
    min_pubs_for_score = db_params['pubmed']['min_pubs_for_score']

    if not search_terms:
        logging.error('No search terms configured in databases.pubmed.search_terms')
        raise SystemExit(1)

    logging.info('Search terms: %s', search_terms)
    logging.info('Min pubs for score: %d', min_pubs_for_score)

    # --- Initialize API client ---
    api_key_env = db_params['api_keys']['pubmed']
    api_key = os.environ.get(api_key_env)
    client = PubMedClient(api_key=api_key)

    # --- Initialize cache ---
    cache = ProteinCache(
        cache_dir=args.cachedir,
        cache_days=db_params['cache_days'],
        force_requery=db_params['force_requery'],
        database_name='pubmed',
    )

    # --- Step 1: Get total PubMed article count (cached) ---
    total_key = '_total_pubmed_count'
    cached_total = cache.get(total_key)
    if cached_total is not None:
        total_articles = cached_total['count']
        logging.info('Total PubMed articles (cached): %d', total_articles)
    else:
        # Use 'all[sb]' (all subset) to get total PubMed article count.
        # An empty query string returns an error; this is the standard idiom.
        total_articles = client.esearch_count('all[sb]')
        if total_articles is None:
            logging.error('Failed to retrieve total PubMed article count')
            total_articles = 37_000_000  # conservative fallback
            logging.warning('Using fallback estimate: %d', total_articles)
        else:
            cache.put(total_key, {'count': total_articles})
            logging.info('Total PubMed articles: %d', total_articles)

    # --- Step 2: Get per-term total counts (cached) ---
    term_counts: dict[str, int] = {}
    for term in search_terms:
        term_key = f'_term_count_{cache_token(term)}'
        cached_term = cache.get(term_key)
        if cached_term is not None:
            term_counts[term] = cached_term['count']
        else:
            count = client.esearch_count(build_term_query(term))
            if count is not None:
                term_counts[term] = count
                cache.put(term_key, {'count': count})
            else:
                term_counts[term] = 0
                logging.warning('Failed to get term count for "%s", using 0', term)

    for term, count in term_counts.items():
        logging.info('  Term "%s": %d articles', term, count)

    # --- Step 3: Query per protein x term ---
    # Cache key: {gene_symbol}_{search_term}
    # Each cache entry stores: {cooccurrence: int, total_pubs: int}
    n_proteins = len(mapping)
    n_terms = len(search_terms)
    total_queries = n_proteins * n_terms
    logging.info('Processing %d proteins x %d terms = %d pairs',
                 n_proteins, n_terms, total_queries)

    rows = []
    queries_made = 0
    cache_hits = 0

    for prot_idx, (_, prot) in enumerate(mapping.iterrows()):
        pid = prot['protein_id']
        mouse_sym = prot.get('gene_symbol_mouse')
        human_sym = prot.get('human_ortholog_symbol')

        # Skip proteins with no gene symbol at all
        has_mouse = pd.notna(mouse_sym) and mouse_sym
        has_human = pd.notna(human_sym) and human_sym

        if not has_mouse and not has_human:
            for term in search_terms:
                rows.append({
                    'protein_id': pid,
                    'search_term': term,
                    'mouse_symbol_used': None,
                    'human_symbol_used': None,
                    'mouse_hit_count': None,
                    'human_hit_count': None,
                    'mouse_total_pubs': None,
                    'human_total_pubs': None,
                    'normalized_score': None,
                    'pubmed_query_status': 'no_symbol',
                })
            continue

        for term in search_terms:
            mouse_co = None
            mouse_total = None
            human_co = None
            human_total = None

            # --- Mouse symbol queries ---
            if has_mouse:
                mouse_cache_key = f'{mouse_sym}_{cache_token(term)}'
                cached = cache.get(mouse_cache_key)
                if cached is not None:
                    mouse_co = cached['cooccurrence']
                    mouse_total = cached['total_pubs']
                    cache_hits += 1
                else:
                    # Co-occurrence query
                    co_query = f'"{mouse_sym}"[tiab] AND {build_term_query(term)}'
                    mouse_co = client.esearch_count(co_query)
                    queries_made += 1

                    # Total pubs for this symbol
                    sym_key = f'{mouse_sym}_total'
                    cached_sym = cache.get(sym_key)
                    if cached_sym is not None:
                        mouse_total = cached_sym['total_pubs']
                    else:
                        total_query = f'"{mouse_sym}"[tiab]'
                        mouse_total = client.esearch_count(total_query)
                        queries_made += 1
                        if mouse_total is not None:
                            cache.put(sym_key, {'total_pubs': mouse_total})

                    # Cache the co-occurrence result
                    if mouse_co is not None and mouse_total is not None:
                        cache.put(mouse_cache_key, {
                            'cooccurrence': mouse_co,
                            'total_pubs': mouse_total,
                        })

            # --- Human symbol queries ---
            if has_human:
                human_cache_key = f'{human_sym}_{cache_token(term)}'
                cached = cache.get(human_cache_key)
                if cached is not None:
                    human_co = cached['cooccurrence']
                    human_total = cached['total_pubs']
                    cache_hits += 1
                else:
                    co_query = f'"{human_sym}"[tiab] AND {build_term_query(term)}'
                    human_co = client.esearch_count(co_query)
                    queries_made += 1

                    sym_key = f'{human_sym}_total'
                    cached_sym = cache.get(sym_key)
                    if cached_sym is not None:
                        human_total = cached_sym['total_pubs']
                    else:
                        total_query = f'"{human_sym}"[tiab]'
                        human_total = client.esearch_count(total_query)
                        queries_made += 1
                        if human_total is not None:
                            cache.put(sym_key, {'total_pubs': human_total})

                    if human_co is not None and human_total is not None:
                        cache.put(human_cache_key, {
                            'cooccurrence': human_co,
                            'total_pubs': human_total,
                        })

            # --- Compute PMI ---
            normalized_score = None
            term_count = term_counts.get(term, 0)

            mouse_pmi = None
            human_pmi = None

            # Mouse PMI
            if mouse_total is not None and mouse_total >= min_pubs_for_score and mouse_co is not None:
                mouse_pmi = compute_pmi(mouse_co, mouse_total, term_count, total_articles)

            # Human PMI
            if human_total is not None and human_total >= min_pubs_for_score and human_co is not None:
                human_pmi = compute_pmi(human_co, human_total, term_count, total_articles)

            # Take the max of mouse and human PMI
            if mouse_pmi is not None and human_pmi is not None:
                normalized_score = max(mouse_pmi, human_pmi)
            elif mouse_pmi is not None:
                normalized_score = mouse_pmi
            elif human_pmi is not None:
                normalized_score = human_pmi

            # Determine status
            status = 'success'
            if mouse_co is None and human_co is None:
                status = 'error'

            rows.append({
                'protein_id': pid,
                'search_term': term,
                'mouse_symbol_used': mouse_sym if has_mouse else None,
                'human_symbol_used': human_sym if has_human else None,
                'mouse_hit_count': mouse_co,
                'human_hit_count': human_co,
                'mouse_total_pubs': mouse_total,
                'human_total_pubs': human_total,
                'normalized_score': normalized_score,
                'pubmed_query_status': status,
            })

        # Progress logging
        if (prot_idx + 1) % 500 == 0 or prot_idx == 0:
            logging.info('  Progress: %d / %d proteins (%d API calls, %d cache hits)',
                         prot_idx + 1, n_proteins, queries_made, cache_hits)

    logging.info('  Done: %d API calls, %d cache hits', queries_made, cache_hits)

    # --- Build output DataFrame ---
    out = pd.DataFrame(rows)

    # Enforce column types
    for col in ['mouse_hit_count', 'human_hit_count', 'mouse_total_pubs', 'human_total_pubs']:
        out[col] = out[col].astype('Int64')
    out['normalized_score'] = out['normalized_score'].astype('Float64')

    # --- Write output ---
    outpath = Path(args.outdir) / f'{args.run_id}.pubmed_cooccurrence.parquet'
    out.to_parquet(outpath, index=False)

    # --- Summary ---
    status_counts = out['pubmed_query_status'].value_counts()
    logging.info('Output: %d rows, %d columns', len(out), len(out.columns))
    for status, count in status_counts.items():
        logging.info('  %s: %d', status, count)

    # PMI stats
    scored = out['normalized_score'].dropna()
    if len(scored) > 0:
        logging.info('PMI scores: %d computed, range [%.2f, %.2f], median %.2f',
                     len(scored), scored.min(), scored.max(), scored.median())
    else:
        logging.info('PMI scores: none computed (all below threshold or no data)')

    logging.info(cache.summary())
    logging.info('Wrote %s', outpath.name)
    logging.info('QUERY_PUBMED complete')


if __name__ == '__main__':
    main()
