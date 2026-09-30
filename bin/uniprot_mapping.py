#!/usr/bin/env python3
# title: uniprot_mapping.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-03-26
# last modified: 2026-04-23
#
# purpose:
#   Processes 4.5-4.9 of Module 01 (Input Validation). Takes the
#   detection-filtered protein list, queries the UniProt bulk ID mapping
#   API to retrieve gene symbols, Entrez IDs, and Ensembl gene IDs, handles
#   edge cases (unmapped, isoform-level accessions, multiple mappings), and
#   caches results to avoid redundant API calls across re-runs. When
#   project.organism is 'mouse', also performs Layer 2 ortholog mapping
#   (mouse to human) via Ensembl BioMart to populate the three ortholog
#   columns in the mapping table.
#
# inputs:
#   - {run_id}.filtered_matrix.parquet  (from filter_proteins.py)
#   - {run_id}_params.yml               (ProSIFT run parameters)
#
# outputs:
#   - {run_id}.id_mapping.parquet       (one row per input protein ID)
#   - {run_id}.validation_report_part3.txt
#
# usage example:
#   python bin/uniprot_mapping.py \
#       --matrix CTXcyto_WT_vs_CTXcyto_KO.filtered_matrix.parquet \
#       --params CTXcyto_WT_vs_CTXcyto_KO_params.yml \
#       --run_id CTXcyto_WT_vs_CTXcyto_KO \
#       --cachedir results/cache/id_mapping \
#       --outdir .

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from io import StringIO

try:
    import pandas as pd
    import requests
    import yaml
except ImportError as exc:
    print(f'Error: required package not available -- {exc}')
    print('Activate the prosift conda environment before running.')
    sys.exit(1)


# ============================================================
# Constants
# ============================================================

UNIPROT_BASE   = 'https://rest.uniprot.org'
IDMAP_RUN_URL  = f'{UNIPROT_BASE}/idmapping/run'
IDMAP_STATUS   = f'{UNIPROT_BASE}/idmapping/status'

FROM_TYPE   = 'UniProtKB_AC-ID'
TO_TYPE     = 'UniProtKB'

# Page size for paginated results fetch
RESULTS_PAGE_SIZE = 500

RETRY_DELAYS   = [5, 15, 45]   # seconds between job-submission retries
POLL_INTERVAL  = 5             # seconds between status polls
MAX_POLLS      = 120           # give up after 10 minutes (120 * 5s)

# --- BioMart constants (Layer 2: mouse -> human ortholog mapping) ---
# BioMart is queried via requests POST (no additional library dependency;
# the XML REST endpoint is simple enough to drive directly).
# Mirror list: retry attempts cycle through mirrors in order so a persistent
# outage on one server automatically falls over to the next.
BIOMART_MIRRORS = [
    'https://www.ensembl.org/biomart/martservice',
    'https://useast.ensembl.org/biomart/martservice',
]
BIOMART_RETRY_DELAYS = [10, 30, 90]   # seconds between BioMart retries
BIOMART_CHUNK_SIZE   = 500            # IDs per BioMart request

# BioMart XML attributes for mouse-to-human ortholog query (mmusculus_gene_ensembl).
# Column ORDER in the returned TSV matches attribute order; renamed to
# _MOUSE_ORTHOLOG_COLS after parsing (positional rename, not by display name,
# so it is stable across Ensembl release display-name changes).
_MOUSE_ORTHOLOG_ATTRS = [
    'ensembl_gene_id',
    'external_gene_name',
    'hsapiens_homolog_ensembl_gene',
    'hsapiens_homolog_associated_gene_name',
    'hsapiens_homolog_orthology_type',
    'hsapiens_homolog_perc_id',
    'hsapiens_homolog_orthology_confidence',
]
_MOUSE_ORTHOLOG_COLS = [
    'ensembl_gene_mouse',
    'gene_symbol_mouse_bm',
    'human_ensembl_gene',
    'human_gene_name',
    'homology_type',
    'perc_id',
    'confidence',
]

# BioMart XML attributes for human Entrez ID lookup (hsapiens_gene_ensembl).
# The mmusculus homolog attributes do not include the human Entrez ID directly,
# so a second query is needed to get it.
_HUMAN_ENTREZ_ATTRS = ['ensembl_gene_id', 'entrezgene_id']
_HUMAN_ENTREZ_COLS  = ['human_ensembl_gene', 'entrez_id']

# --- Cache schema version ---
# Increment whenever the mapping table structure changes in a way that makes
# old cached files incompatible. On load, a version mismatch triggers
# regeneration (treat as expired). This resolves the open question logged
# 2026-03-26 about silent stale-cache serving after code changes.
#   1 = original (null ortholog columns, Phase 1-2)
#   2 = BioMart ortholog columns populated (2026-04-02)
#   3 = id column renamed at source from 'input_id' to 'protein_id' (2026-04-23)
CURRENT_SCHEMA_VERSION = 3

# Output column schema (Process 4.8)
SCHEMA_COLS = [
    'protein_id',
    'uniprot_accession',
    'gene_symbol_mouse',
    'entrez_id_mouse',
    'ensembl_gene_mouse',
    'human_ortholog_symbol',
    'human_ortholog_entrez',
    'ortholog_mapping_status',
    'mapping_status',
    'mapping_notes',
]

# Regex pattern for isoform-level UniProt accessions (e.g. Q8K1M6-14)
ISOFORM_RE = re.compile(r'^([A-Z0-9]+)-\d+$', re.IGNORECASE)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Processes 4.5-4.9: UniProt bulk ID mapping for ProSIFT.'
    )
    parser.add_argument(
        '--matrix', '-m', required=True,
        help='Path to {run_id}.filtered_matrix.parquet'
    )
    parser.add_argument(
        '--params', '-p', required=True,
        help='Path to {run_id}_params.yml'
    )
    parser.add_argument(
        '--run_id', '-r', required=True,
        help='Run identifier (used to name output files)'
    )
    parser.add_argument(
        '--cachedir', '-d', default='cache/id_mapping',
        help='Directory for UniProt mapping cache (default: cache/id_mapping)'
    )
    parser.add_argument(
        '--outdir', '-o', default='.',
        help='Output directory (default: current directory)'
    )
    return parser.parse_args()


# ============================================================
# Load inputs
# ============================================================

def load_params(params_path: str) -> dict:
    '''Load and return the ProSIFT params YAML.'''
    with open(params_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_protein_ids(matrix_path: str, id_col: str) -> list[str]:
    '''
    Read filtered_matrix.parquet and return a list of protein IDs in
    the order they appear. The matrix already has first-accession extraction
    applied (from validate_inputs.py), so no further parsing is needed.
    '''
    df = pd.read_parquet(matrix_path, columns=[id_col])
    return df[id_col].tolist()


# ============================================================
# Cache management
# ============================================================

def _cache_paths(cachedir: str, cache_key: str) -> tuple[str, str]:
    parquet = os.path.join(cachedir, f'{cache_key}.parquet')
    meta    = os.path.join(cachedir, f'{cache_key}.json')
    return parquet, meta


def compute_cache_key(protein_ids: list[str], organism: str) -> str:
    '''MD5 of sorted IDs + source type + organism.'''
    payload = '|'.join(sorted(protein_ids)) + f'|{FROM_TYPE}|{organism}'
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def load_cache(cachedir: str, cache_key: str, cache_days: int) -> pd.DataFrame | None:
    '''
    Return cached DataFrame if valid: exists, not expired, and schema version
    matches CURRENT_SCHEMA_VERSION. Returns None on any failure condition so
    the caller falls through to a full regeneration.
    '''
    parquet_path, meta_path = _cache_paths(cachedir, cache_key)
    if not os.path.isfile(parquet_path) or not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    # Schema version check: caches from before ortholog mapping (version 1 or
    # missing) must be regenerated so the ortholog columns are populated.
    cached_version = meta.get('schema_version', 1)
    if cached_version != CURRENT_SCHEMA_VERSION:
        print(f'  Cache schema version mismatch '
              f'(cached: {cached_version}, current: {CURRENT_SCHEMA_VERSION}). '
              f'Regenerating.')
        return None
    age_days = (datetime.now(UTC).timestamp() - meta['timestamp']) / 86400
    if age_days > cache_days:
        print(f'  Cache expired ({age_days:.1f} days old, limit {cache_days} days).')
        return None
    print(f'  Cache hit ({age_days:.1f} days old). Loading from {parquet_path}')
    return pd.read_parquet(parquet_path)


def save_cache(cachedir: str, cache_key: str, df: pd.DataFrame) -> None:
    '''Write mapping table and metadata to cache directory.'''
    os.makedirs(cachedir, exist_ok=True)
    parquet_path, meta_path = _cache_paths(cachedir, cache_key)
    df.to_parquet(parquet_path, index=False)
    meta = {
        'timestamp': datetime.now(UTC).timestamp(),
        'n_proteins': len(df),
        'cache_key': cache_key,
        'schema_version': CURRENT_SCHEMA_VERSION,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  Cached to {parquet_path}')


# ============================================================
# UniProt ID mapping API
# ============================================================

def _submit_job(ids: list[str]) -> str:
    '''Submit bulk ID mapping job. Returns job ID.'''
    resp = requests.post(
        IDMAP_RUN_URL,
        data={'ids': ','.join(ids), 'from': FROM_TYPE, 'to': TO_TYPE},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()['jobId']


def _get_results_url(job_id: str) -> str:
    '''
    Poll the status endpoint until the job finishes, then return the
    results URL.

    The UniProt API v2 redirects the status endpoint to the results endpoint
    once a job is complete. When `to=UniProtKB`, the redirect goes to:
      .../idmapping/uniprotkb/results/{jobId}
    We follow the redirect and detect completion by URL pattern or inline
    results in the response body.
    '''
    for _ in range(MAX_POLLS):
        resp = requests.get(
            f'{IDMAP_STATUS}/{job_id}',
            timeout=30,
            allow_redirects=True,
        )
        resp.raise_for_status()
        # If we were redirected to a results endpoint, the job is done.
        # The redirect URL IS the results URL.
        if '/results/' in resp.url:
            return resp.url
        data = resp.json()
        status = data.get('jobStatus', '')
        if status == 'FINISHED':
            # No redirect yet -- construct the results URL from convention
            return f'{UNIPROT_BASE}/idmapping/uniprotkb/results/{job_id}'
        if status in ('ERROR', 'FAILED'):
            raise RuntimeError(f'UniProt ID mapping job failed: {data}')
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f'UniProt ID mapping job {job_id} did not finish after '
        f'{MAX_POLLS * POLL_INTERVAL}s'
    )


def _fetch_all_results(results_url: str) -> list[dict]:
    '''
    Paginate through all results at the given URL and return a flat list
    of {"from": ..., "to": {...}} dicts. Pagination is driven by the
    Link: <url>; rel="next" header.
    '''
    all_results: list[dict] = []
    next_url: str | None = results_url
    page = 0

    while next_url:
        page += 1
        sep = '&' if '?' in next_url else '?'
        paged_url = f'{next_url}{sep}size={RESULTS_PAGE_SIZE}'
        resp = requests.get(paged_url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get('results', [])
        all_results.extend(batch)
        print(f'    Page {page}: {len(batch)} results (total so far: {len(all_results)})')

        # Follow Link: <...>; rel="next" header
        link_header = resp.headers.get('Link', '')
        next_url = _parse_next_link(link_header)

    return all_results


def _parse_next_link(link_header: str) -> str | None:
    '''Extract the "next" URL from a Link header, or None.'''
    # Format: <https://...>; rel="next"
    for part in link_header.split(','):
        part = part.strip()
        if 'rel="next"' in part:
            # extract URL between < and >
            start = part.find('<')
            end   = part.find('>')
            if start >= 0 and end > start:
                return part[start + 1:end]
    return None


def query_uniprot(ids: list[str]) -> list[dict]:
    '''
    Submit a bulk ID mapping job, poll until done, paginate through all
    results, and return a flat list of raw result dicts.
    Retries the full sequence up to 3 times on network errors.
    '''
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0, *RETRY_DELAYS]):
        if delay:
            print(f'  Retrying in {delay}s (attempt {attempt + 1})...')
            time.sleep(delay)
        try:
            print(f'  Submitting {len(ids)} IDs to UniProt ID mapping API...')
            job_id = _submit_job(ids)
            print(f'  Job ID: {job_id} -- polling for completion...')
            results_url = _get_results_url(job_id)
            print(f'  Job finished. Fetching results from: {results_url}')
            return _fetch_all_results(results_url)
        except (requests.RequestException, RuntimeError, TimeoutError) as exc:
            last_exc = exc
            print(f'  Warning: attempt {attempt + 1} failed -- {exc}')
    raise RuntimeError(
        f'UniProt ID mapping failed after {len(RETRY_DELAYS) + 1} attempts. '
        f'Last error: {last_exc}'
    )


# ============================================================
# Response parsing
# ============================================================

def _is_isoform(accession: str) -> bool:
    return bool(ISOFORM_RE.match(accession))


def _canonical(accession: str) -> str:
    '''Strip isoform suffix if present.'''
    m = ISOFORM_RE.match(accession)
    return m.group(1) if m else accession


def _extract_fields(to_entry: dict) -> tuple[str | None, str | None, str | None, str | None]:
    '''
    Extract (canonical_accession, gene_symbol, entrez_id, ensembl_gene)
    from a UniProt KB entry JSON object (the "to" field in API results).

    Gene symbol: to.genes[0].geneName.value
    Entrez ID:   first cross-reference with database == "GeneID", .id field
    Ensembl gene: first cross-reference with database == "Ensembl", property
                  with key == "GeneId" (note: GeneId not GeneID)
    '''
    accession = to_entry.get('primaryAccession')

    # Gene symbol
    genes = to_entry.get('genes', [])
    gene_symbol: str | None = None
    if genes:
        gene_name = genes[0].get('geneName', {})
        gene_symbol = gene_name.get('value') or None

    # Entrez ID and Ensembl gene from cross-references
    entrez_id:    str | None = None
    ensembl_gene: str | None = None
    for xref in to_entry.get('uniProtKBCrossReferences', []):
        db = xref.get('database', '')
        if db == 'GeneID' and entrez_id is None:
            entrez_id = xref.get('id') or None
        elif db == 'Ensembl' and ensembl_gene is None:
            for prop in xref.get('properties', []):
                if prop.get('key') == 'GeneId':
                    ensembl_gene = prop.get('value') or None
                    break
        if entrez_id and ensembl_gene:
            break  # both found, no need to scan further

    return accession, gene_symbol, entrez_id, ensembl_gene


def parse_json_results(api_results: list[dict], input_ids: list[str]) -> pd.DataFrame:
    '''
    Parse the list of {"from": ..., "to": {...}} dicts returned by the
    UniProt ID mapping API and build the mapping table.

    Steps:
    1. Group API results by input ID.
    2. Classify mapping_status: mapped, unmapped, isoform_collapsed,
       accession_redirected, multiple_mappings.
    3. Leave ortholog columns null (populated later by map_orthologs()).
    4. Build one row per input ID preserving input order.
    '''
    # Build a lookup: input_id -> list of "to" entry dicts
    results_by_id: dict[str, list[dict]] = {}
    for item in api_results:
        from_id  = item.get('from', '')
        to_entry = item.get('to', {})
        results_by_id.setdefault(from_id, []).append(to_entry)

    # --- Build one row per input ID ---
    rows: list[dict] = []

    for input_id in input_ids:
        hits = results_by_id.get(input_id, [])

        if len(hits) == 0:
            # --- Unmapped ---
            rows.append({
                'protein_id':               input_id,
                'uniprot_accession':        None,
                'gene_symbol_mouse':        None,
                'entrez_id_mouse':          None,
                'ensembl_gene_mouse':       None,
                'human_ortholog_symbol':    None,
                'human_ortholog_entrez':    None,
                'ortholog_mapping_status':  None,
                'mapping_status':           'unmapped',
                'mapping_notes':            'ID not found in UniProt response.',
            })

        elif len(hits) == 1:
            accession, gene_sym, entrez, ensembl = _extract_fields(hits[0])
            is_isoform = _is_isoform(input_id) and accession == _canonical(input_id)
            is_redirected = not is_isoform and accession != input_id

            if is_isoform:
                status = 'isoform_collapsed'
                notes  = f'Isoform {input_id} collapsed to canonical {accession}.'
            elif is_redirected:
                # Input was a secondary/merged accession; UniProt returned a
                # different primary. The abundance data is still keyed on
                # protein_id but all annotation columns use the canonical
                # accession.
                status = 'accession_redirected'
                notes  = (
                    f'Input accession {input_id} is a secondary or merged '
                    f'accession; UniProt canonical is {accession}. '
                    f'Abundance data remains keyed on protein_id.'
                )
            else:
                status = 'mapped'
                notes  = ''
            rows.append({
                'protein_id':               input_id,
                'uniprot_accession':        accession,
                'gene_symbol_mouse':        gene_sym,
                'entrez_id_mouse':          entrez,
                'ensembl_gene_mouse':       ensembl,
                'human_ortholog_symbol':    None,
                'human_ortholog_entrez':    None,
                'ortholog_mapping_status':  None,
                'mapping_status':           status,
                'mapping_notes':            notes,
            })

        else:
            # --- Multiple mappings: take first, log the rest ---
            accession, gene_sym, entrez, ensembl = _extract_fields(hits[0])
            other_accessions = '; '.join(
                h.get('primaryAccession', '') for h in hits[1:]
            )
            rows.append({
                'protein_id':               input_id,
                'uniprot_accession':        accession,
                'gene_symbol_mouse':        gene_sym,
                'entrez_id_mouse':          entrez,
                'ensembl_gene_mouse':       ensembl,
                'human_ortholog_symbol':    None,
                'human_ortholog_entrez':    None,
                'ortholog_mapping_status':  None,
                'mapping_status':           'multiple_mappings',
                'mapping_notes':            (
                    f'Input ID mapped to multiple UniProt entries. '
                    f'Using first ({accession}); '
                    f'others: {other_accessions}.'
                ),
            })

    df = pd.DataFrame(rows, columns=SCHEMA_COLS)
    return df


# ============================================================
# BioMart ortholog mapping (Layer 2: mouse -> human)
# ============================================================

def _biomart_post(xml: str) -> pd.DataFrame:
    '''
    POST an XML query to Ensembl BioMart and return results as a DataFrame.

    BioMart returns TSV with a header line. Uses requests directly, consistent
    with the UniProt Layer 1 pattern (no additional library dependency needed).

    BioMart-level errors come back with HTTP 200 but error text in the body
    (typically starting with "ERROR" or "Query ERROR"). These are detected and
    raised as RuntimeError so the retry loop can handle them.

    Retry attempts cycle through BIOMART_MIRRORS in round-robin order so a
    persistent outage on the primary server falls over to the next mirror.
    '''
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0, *BIOMART_RETRY_DELAYS]):
        if delay:
            mirror = BIOMART_MIRRORS[attempt % len(BIOMART_MIRRORS)]
            print(f'  BioMart: retrying in {delay}s on {mirror} '
                  f'(attempt {attempt + 1})...')
            time.sleep(delay)
        else:
            mirror = BIOMART_MIRRORS[0]
        try:
            resp = requests.post(
                mirror,
                data={'query': xml},
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                # Empty body = no records matched the filter (valid, not an error)
                return pd.DataFrame()
            first_line = text.splitlines()[0]
            if first_line.upper().startswith('ERROR') or 'Query ERROR' in text:
                raise RuntimeError(
                    f'BioMart returned error in response body: {text[:300]}'
                )
            df = pd.read_csv(StringIO(text), sep='\t', header=0)
            return df
        except (requests.RequestException, RuntimeError) as exc:
            last_exc = exc
            print(f'  BioMart: attempt {attempt + 1} failed -- {exc}')
    raise RuntimeError(
        f'BioMart query failed after {len(BIOMART_RETRY_DELAYS) + 1} attempts. '
        f'Last error: {last_exc}'
    )


def _build_ortholog_xml(ids: list[str], filter_name: str) -> str:
    '''
    Build a BioMart XML query for mouse-to-human ortholog attributes.
    filter_name: 'ensembl_gene_id'     for primary Ensembl ID lookup
                 'external_gene_name'  for gene symbol fallback
    '''
    attrs_xml = ''.join(
        f'<Attribute name="{attr}"/>' for attr in _MOUSE_ORTHOLOG_ATTRS
    )
    id_values = ','.join(ids)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="1" '
        'uniqueRows="0" count="">'
        '<Dataset name="mmusculus_gene_ensembl" interface="default">'
        f'<Filter name="{filter_name}" value="{id_values}"/>'
        f'{attrs_xml}'
        '</Dataset>'
        '</Query>'
    )


def _query_orthologs_by_ensembl(
    ensembl_ids: list[str],
) -> tuple[pd.DataFrame, set[str]]:
    '''
    Query BioMart for mouse-to-human orthologs using mouse Ensembl gene IDs
    (primary join key from UniProt Layer 1 mapping).

    Returns (results_df, failed_ids). `results_df` has columns
    _MOUSE_ORTHOLOG_COLS; multiple rows per mouse gene are expected when
    one-to-many orthologs exist; rows where human_ensembl_gene is empty/null
    indicate no human ortholog for that gene. `failed_ids` contains the
    Ensembl IDs whose chunk returned a malformed response (wrong column count)
    and were therefore not resolved -- these should be marked with status
    'ortholog_lookup_failed' downstream rather than silently falling through
    to 'no_ortholog'. Processed in chunks of BIOMART_CHUNK_SIZE.
    '''
    if not ensembl_ids:
        return pd.DataFrame(columns=_MOUSE_ORTHOLOG_COLS), set()

    all_chunks: list[pd.DataFrame] = []
    failed_ids: set[str] = set()
    n_chunks = (len(ensembl_ids) + BIOMART_CHUNK_SIZE - 1) // BIOMART_CHUNK_SIZE
    for i in range(0, len(ensembl_ids), BIOMART_CHUNK_SIZE):
        chunk     = ensembl_ids[i:i + BIOMART_CHUNK_SIZE]
        chunk_num = i // BIOMART_CHUNK_SIZE + 1
        print(f'  BioMart (Ensembl ID): chunk {chunk_num}/{n_chunks}, '
              f'{len(chunk)} IDs...')
        xml      = _build_ortholog_xml(chunk, filter_name='ensembl_gene_id')
        chunk_df = _biomart_post(xml)
        if not chunk_df.empty:
            if len(chunk_df.columns) == len(_MOUSE_ORTHOLOG_COLS):
                chunk_df.columns = _MOUSE_ORTHOLOG_COLS
            else:
                # Schema drift or partial outage: response came back but its
                # shape is wrong. Mark this chunk's IDs as lookup failures so
                # downstream code can distinguish "confirmed no ortholog" from
                # "BioMart could not answer."
                print(f'  ERROR: unexpected BioMart column count '
                      f'(expected {len(_MOUSE_ORTHOLOG_COLS)}, '
                      f'got {len(chunk_df.columns)}). Marking {len(chunk)} '
                      f'Ensembl IDs as ortholog_lookup_failed.')
                failed_ids.update(chunk)
                chunk_df = pd.DataFrame(columns=_MOUSE_ORTHOLOG_COLS)
        else:
            chunk_df = pd.DataFrame(columns=_MOUSE_ORTHOLOG_COLS)
        all_chunks.append(chunk_df)

    return pd.concat(all_chunks, ignore_index=True), failed_ids


def _query_orthologs_by_symbol(
    gene_symbols: list[str],
) -> tuple[pd.DataFrame, set[str]]:
    '''
    Fallback: query BioMart using mouse gene symbols for proteins where
    ensembl_gene_mouse is null (~3.4% from the CTXcyto benchmark, typically
    minor isoforms or poorly annotated UniProt entries).

    Returns (results_df, failed_symbols). See _query_orthologs_by_ensembl for
    the semantics of `failed_symbols`. A symbol can map to multiple Ensembl
    gene IDs (gene family members with shared names); _resolve_orthologs()
    groups by gene_symbol_mouse_bm and selects the highest-confidence
    ortholog from any resulting rows.
    '''
    if not gene_symbols:
        return pd.DataFrame(columns=_MOUSE_ORTHOLOG_COLS), set()

    all_chunks: list[pd.DataFrame] = []
    failed_symbols: set[str] = set()
    n_chunks = (len(gene_symbols) + BIOMART_CHUNK_SIZE - 1) // BIOMART_CHUNK_SIZE
    for i in range(0, len(gene_symbols), BIOMART_CHUNK_SIZE):
        chunk     = gene_symbols[i:i + BIOMART_CHUNK_SIZE]
        chunk_num = i // BIOMART_CHUNK_SIZE + 1
        print(f'  BioMart (gene symbol fallback): chunk {chunk_num}/{n_chunks}, '
              f'{len(chunk)} symbols...')
        xml      = _build_ortholog_xml(chunk, filter_name='external_gene_name')
        chunk_df = _biomart_post(xml)
        if not chunk_df.empty:
            if len(chunk_df.columns) == len(_MOUSE_ORTHOLOG_COLS):
                chunk_df.columns = _MOUSE_ORTHOLOG_COLS
            else:
                print(f'  ERROR: unexpected BioMart column count '
                      f'(expected {len(_MOUSE_ORTHOLOG_COLS)}, '
                      f'got {len(chunk_df.columns)}). Marking {len(chunk)} '
                      f'symbols as ortholog_lookup_failed.')
                failed_symbols.update(chunk)
                chunk_df = pd.DataFrame(columns=_MOUSE_ORTHOLOG_COLS)
        else:
            chunk_df = pd.DataFrame(columns=_MOUSE_ORTHOLOG_COLS)
        all_chunks.append(chunk_df)

    return pd.concat(all_chunks, ignore_index=True), failed_symbols


def _query_human_entrez(human_ensembl_ids: list[str]) -> dict[str, str | None]:
    '''
    Query BioMart hsapiens_gene_ensembl to map human Ensembl gene IDs to NCBI
    Entrez gene IDs. Returns dict: human_ensembl_id -> entrez_id (str) or None.

    A second BioMart call is required because the mmusculus_gene_ensembl homolog
    attributes do not include the human Entrez gene ID directly -- only the
    human Ensembl gene ID and gene symbol are available from that dataset.
    BioMart returns Entrez IDs as floats (e.g. 7157.0); these are converted to
    integer strings ('7157').
    '''
    if not human_ensembl_ids:
        return {}

    attrs_xml = ''.join(
        f'<Attribute name="{attr}"/>' for attr in _HUMAN_ENTREZ_ATTRS
    )
    all_chunks: list[pd.DataFrame] = []
    n_chunks = (len(human_ensembl_ids) + BIOMART_CHUNK_SIZE - 1) // BIOMART_CHUNK_SIZE

    for i in range(0, len(human_ensembl_ids), BIOMART_CHUNK_SIZE):
        chunk     = human_ensembl_ids[i:i + BIOMART_CHUNK_SIZE]
        chunk_num = i // BIOMART_CHUNK_SIZE + 1
        print(f'  BioMart (human Entrez): chunk {chunk_num}/{n_chunks}, '
              f'{len(chunk)} IDs...')
        id_values = ','.join(chunk)
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<!DOCTYPE Query>'
            '<Query virtualSchemaName="default" formatter="TSV" header="1" '
            'uniqueRows="1" count="">'
            '<Dataset name="hsapiens_gene_ensembl" interface="default">'
            f'<Filter name="ensembl_gene_id" value="{id_values}"/>'
            f'{attrs_xml}'
            '</Dataset>'
            '</Query>'
        )
        chunk_df = _biomart_post(xml)
        if not chunk_df.empty:
            if len(chunk_df.columns) == len(_HUMAN_ENTREZ_COLS):
                chunk_df.columns = _HUMAN_ENTREZ_COLS
                all_chunks.append(chunk_df)
            else:
                print(f'  Warning: unexpected column count from human Entrez query '
                      f'(expected {len(_HUMAN_ENTREZ_COLS)}, '
                      f'got {len(chunk_df.columns)}). Skipping chunk.')

    if not all_chunks:
        return {}

    df = pd.concat(all_chunks, ignore_index=True)

    result: dict[str, str | None] = {}
    for _, row in df.iterrows():
        eid = row['human_ensembl_gene']
        if pd.isna(eid):
            continue
        eid = str(eid)
        entrez_val = row['entrez_id']
        if pd.notna(entrez_val):
            try:
                # BioMart returns Entrez IDs as floats (e.g. 7157.0)
                result[eid] = str(int(float(entrez_val)))
            except (ValueError, TypeError):
                result[eid] = str(entrez_val)
        else:
            result.setdefault(eid, None)

    return result


def _resolve_orthologs(
    ortholog_df: pd.DataFrame,
    entrez_map: dict[str, str | None],
    join_key: str,
) -> dict[str, dict]:
    '''
    For each unique value of join_key in ortholog_df, select the primary human
    ortholog and build a result record.

    Selection rule for one-to-many cases (multiple human rows per mouse gene):
      1. Prefer confidence == 1 (high confidence per Ensembl's classifier).
      2. Among ties, prefer the cleaner structural relationship:
         ortholog_one2one > ortholog_one2many > ortholog_many2many
         (one2one is a cleaner 1:1 evolutionary correspondence; many2many
         reflects gene family expansion and is less reliable even when
         perc_id is high).
      3. Among remaining ties, prefer highest perc_id (% sequence identity).
      4. The top-ranked ortholog is stored in human_ortholog_symbol/entrez.
      5. All orthologs are listed in the notes field for downstream consumers
         (e.g. PubMed module can optionally query all ortholog symbols).

    ortholog_mapping_status values assigned here:
      one_to_one  -- exactly one non-null human gene row for this mouse gene
      one_to_many -- multiple non-null human gene rows for this mouse gene

    'no_ortholog' is assigned in map_orthologs() for mouse genes present in
    the filter input but returning zero human rows from BioMart.

    Returns: dict mapping str(join_key_value) -> {symbol, entrez, status, notes}
    '''
    result: dict[str, dict] = {}

    # Keep only rows that have an actual human ortholog (non-null, non-empty)
    has_orth = ortholog_df[
        ortholog_df['human_ensembl_gene'].notna() &
        (ortholog_df['human_ensembl_gene'].astype(str).str.strip() != '')
    ].copy()

    if has_orth.empty:
        return result

    # Coerce numeric columns; fill NaN with 0 so sort order is well-defined
    has_orth['confidence'] = (
        pd.to_numeric(has_orth['confidence'], errors='coerce').fillna(0)
    )
    has_orth['perc_id'] = (
        pd.to_numeric(has_orth['perc_id'], errors='coerce').fillna(0.0)
    )
    # Structural-quality rank for homology_type (lower is better).
    # BioMart returns 'ortholog_one2one', 'ortholog_one2many', or
    # 'ortholog_many2many'; unexpected values sort last.
    _HOMOLOGY_TYPE_RANK = {
        'ortholog_one2one':   0,
        'ortholog_one2many':  1,
        'ortholog_many2many': 2,
    }
    has_orth['_homology_rank'] = (
        has_orth['homology_type'].map(_HOMOLOGY_TYPE_RANK).fillna(3).astype(int)
    )

    for key_val, group in has_orth.groupby(join_key):
        # Tie-break order (spec Section 4.7): confidence DESC, then
        # structural-quality rank ASC (one2one > one2many > many2many), then
        # perc_id DESC as final tiebreaker.
        group   = group.sort_values(
            ['confidence', '_homology_rank', 'perc_id'],
            ascending=[False, True, False],
        )
        n       = len(group)
        primary = group.iloc[0]

        primary_symbol  = primary['human_gene_name']
        primary_symbol  = str(primary_symbol) if pd.notna(primary_symbol) else None
        primary_ensembl = str(primary['human_ensembl_gene'])
        primary_entrez  = entrez_map.get(primary_ensembl)

        if n == 1:
            status = 'one_to_one'
            notes  = ''
        else:
            status = 'one_to_many'
            # Enumerate all orthologs so downstream consumers can use them
            orth_info: list[str] = []
            for _, r in group.iterrows():
                sym        = str(r['human_gene_name']) if pd.notna(r['human_gene_name']) else 'unknown'
                h_eid      = str(r['human_ensembl_gene'])
                entrez_str = entrez_map.get(h_eid) or 'NA'
                conf       = int(r['confidence'])
                pct        = f"{r['perc_id']:.1f}"
                orth_info.append(f'{sym} (Entrez:{entrez_str}, {pct}% id, conf={conf})')
            notes = 'All human orthologs: ' + '; '.join(orth_info) + '.'

        result[str(key_val)] = {
            'symbol': primary_symbol,
            'entrez': primary_entrez,
            'status': status,
            'notes':  notes,
        }

    return result


def map_orthologs(mapping_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Layer 2 ortholog mapping: populate human_ortholog_symbol,
    human_ortholog_entrez, and ortholog_mapping_status using Ensembl BioMart.
    Called only when project.organism == 'mouse'.

    Strategy:
      Primary lookup: join on ensembl_gene_mouse (stable, unambiguous ID from
        UniProt Layer 1). Covers ~96.6% of proteins in the CTXcyto benchmark.
      Fallback: for proteins with null ensembl_gene_mouse, query BioMart by
        gene_symbol_mouse. Noted in mapping_notes.
      Truly unmapped proteins (mapping_status == 'unmapped') have no gene
        symbol or Ensembl ID; their ortholog columns remain null rather than
        'no_ortholog', which would be misleading for proteins that couldn't
        even be mapped to UniProt.

    ortholog_mapping_status values:
      one_to_one, one_to_many, many_to_many -- BioMart returned a homolog.
      no_ortholog                            -- BioMart returned no rows for
                                                this gene (confirmed absence).
      ortholog_lookup_failed                 -- BioMart returned a malformed
                                                response (schema drift, partial
                                                outage); the true state is
                                                unknown and the protein should
                                                be rechecked on the next run.
    '''
    df = mapping_df.copy()

    # ----------------------------------------------------------------
    # Step 1: Primary lookup by ensembl_gene_mouse
    # ----------------------------------------------------------------
    # UniProt returns versioned Ensembl IDs (e.g., ENSMUSG00000043154.16).
    # BioMart requires unversioned IDs (ENSMUSG00000043154); the suffix is
    # stripped here. The versioned form is preserved in the mapping table.
    ensembl_ids_versioned = (
        df['ensembl_gene_mouse']
        .dropna()
        .unique()
        .tolist()
    )
    # Build a deduped unversioned list, and a lookup from unversioned -> versioned
    # so BioMart results can be mapped back to the original IDs in the table.
    versioned_to_base: dict[str, str] = {}
    for vid in ensembl_ids_versioned:
        base = vid.split('.')[0]
        versioned_to_base[vid] = base

    ensembl_ids = list({vid.split('.')[0] for vid in ensembl_ids_versioned})
    print(f'  BioMart: primary lookup for {len(ensembl_ids)} unique Ensembl IDs '
          f'(version suffixes stripped)...')
    ensembl_ortholog_df, failed_ensembl_ids = _query_orthologs_by_ensembl(ensembl_ids)

    # ----------------------------------------------------------------
    # Step 2: Fetch human Entrez IDs for all human Ensembl IDs found
    # ----------------------------------------------------------------
    human_ensembl_all = (
        ensembl_ortholog_df['human_ensembl_gene']
        .dropna()
        .astype(str)
        .str.strip()
    )
    human_ensembl_all = human_ensembl_all[human_ensembl_all != ''].unique().tolist()
    print(f'  BioMart: fetching Entrez IDs for {len(human_ensembl_all)} '
          f'unique human Ensembl IDs...')
    entrez_map = _query_human_entrez(human_ensembl_all)

    # ----------------------------------------------------------------
    # Step 3: Resolve primary ortholog per mouse Ensembl gene
    # ----------------------------------------------------------------
    ensembl_orth_map = _resolve_orthologs(
        ensembl_ortholog_df, entrez_map, join_key='ensembl_gene_mouse'
    )

    # ----------------------------------------------------------------
    # Step 4: Apply results to the mapping table
    # ----------------------------------------------------------------
    for idx, row in df.iterrows():
        e_id = row['ensembl_gene_mouse']
        if pd.isna(e_id):
            continue  # no Ensembl ID -- handled in symbol fallback below
        # Strip version suffix to match keys in ensembl_orth_map
        key = str(e_id).split('.')[0]
        if key in ensembl_orth_map:
            orth = ensembl_orth_map[key]
            df.at[idx, 'human_ortholog_symbol']   = orth['symbol']
            df.at[idx, 'human_ortholog_entrez']   = orth['entrez']
            df.at[idx, 'ortholog_mapping_status'] = orth['status']
            if orth['notes']:
                existing = df.at[idx, 'mapping_notes'] or ''
                sep = ' ' if existing else ''
                df.at[idx, 'mapping_notes'] = existing + sep + orth['notes']
        elif key in failed_ensembl_ids:
            # BioMart returned a malformed response for this chunk; we do not
            # actually know whether an ortholog exists.
            df.at[idx, 'ortholog_mapping_status'] = 'ortholog_lookup_failed'
            existing = df.at[idx, 'mapping_notes'] or ''
            sep = ' ' if existing else ''
            df.at[idx, 'mapping_notes'] = existing + sep + (
                'BioMart returned a malformed response for this chunk; '
                'ortholog lookup could not be completed.'
            )
        else:
            # Gene found in UniProt but BioMart returned no rows for it
            df.at[idx, 'ortholog_mapping_status'] = 'no_ortholog'

    # ----------------------------------------------------------------
    # Step 5: Gene symbol fallback for proteins with null ensembl_gene_mouse
    # ----------------------------------------------------------------
    # Excludes unmapped proteins (no gene symbol available for those).
    null_ensembl_mask = (
        df['ensembl_gene_mouse'].isna() &
        df['gene_symbol_mouse'].notna() &
        (df['mapping_status'] != 'unmapped')
    )
    null_ensembl_df = df[null_ensembl_mask]

    if not null_ensembl_df.empty:
        symbols = null_ensembl_df['gene_symbol_mouse'].dropna().unique().tolist()
        print(f'  BioMart (gene symbol fallback): {len(null_ensembl_df)} proteins '
              f'with null Ensembl ID, {len(symbols)} unique symbols...')
        sym_ortholog_df, failed_symbols = _query_orthologs_by_symbol(symbols)

        # Fetch Entrez IDs only for human Ensembl IDs not already in entrez_map
        sym_human_ensembl = (
            sym_ortholog_df['human_ensembl_gene']
            .dropna()
            .astype(str)
            .str.strip()
        )
        sym_human_ensembl = sym_human_ensembl[sym_human_ensembl != ''].unique().tolist()
        new_ids = [x for x in sym_human_ensembl if x not in entrez_map]
        if new_ids:
            new_entrez = _query_human_entrez(new_ids)
            entrez_map.update(new_entrez)

        sym_orth_map = _resolve_orthologs(
            sym_ortholog_df, entrez_map, join_key='gene_symbol_mouse_bm'
        )

        for idx, row in null_ensembl_df.iterrows():
            sym = row['gene_symbol_mouse']
            if sym and sym in sym_orth_map:
                orth = sym_orth_map[sym]
                df.at[idx, 'human_ortholog_symbol']   = orth['symbol']
                df.at[idx, 'human_ortholog_entrez']   = orth['entrez']
                df.at[idx, 'ortholog_mapping_status'] = orth['status']
                fallback_note = (
                    'Ortholog mapped via gene symbol fallback '
                    '(Ensembl ID unavailable from UniProt).'
                )
                if orth['notes']:
                    fallback_note += ' ' + orth['notes']
                existing = df.at[idx, 'mapping_notes'] or ''
                sep = ' ' if existing else ''
                df.at[idx, 'mapping_notes'] = existing + sep + fallback_note
            elif sym and sym in failed_symbols:
                df.at[idx, 'ortholog_mapping_status'] = 'ortholog_lookup_failed'
                existing = df.at[idx, 'mapping_notes'] or ''
                sep = ' ' if existing else ''
                df.at[idx, 'mapping_notes'] = existing + sep + (
                    'BioMart returned a malformed response for this chunk; '
                    'ortholog lookup could not be completed.'
                )
            else:
                df.at[idx, 'ortholog_mapping_status'] = 'no_ortholog'

    # --- Cleanup: any remaining null ortholog statuses for non-unmapped proteins ---
    # Rare edge case: a protein was UniProt-mapped but has neither an Ensembl ID
    # nor a gene symbol (no lookup possible). Set to 'no_ortholog' so downstream
    # consumers see a defined status rather than null.
    can_lookup_mask = (
        (df['mapping_status'] != 'unmapped') &
        df['ortholog_mapping_status'].isna()
    )
    if can_lookup_mask.any():
        df.loc[can_lookup_mask, 'ortholog_mapping_status'] = 'no_ortholog'
        print(f'  {can_lookup_mask.sum()} protein(s) with no usable ID set to no_ortholog.')

    return df


# ============================================================
# Validation report (part 3)
# ============================================================

def write_report(
    filepath: str,
    run_id: str,
    n_input: int,
    df: pd.DataFrame,
    cache_hit: bool,
    cache_key: str,
    cache_days: int,
    organism: str,
) -> None:
    '''Write validation_report_part3.txt summarizing ID mapping results.'''

    status_counts = df['mapping_status'].value_counts()

    def _pct(n: int) -> str:
        return f'{100 * n / n_input:.1f}%'

    n_mapped      = int(status_counts.get('mapped', 0))
    n_isoform     = int(status_counts.get('isoform_collapsed', 0))
    n_redirected  = int(status_counts.get('accession_redirected', 0))
    n_multi       = int(status_counts.get('multiple_mappings', 0))
    n_unmapped    = int(status_counts.get('unmapped', 0))

    # Coverage of individual annotation fields (excluding unmapped rows)
    mapped_df = df[df['mapping_status'] != 'unmapped']
    n_gene_sym  = int(mapped_df['gene_symbol_mouse'].notna().sum())
    n_entrez    = int(mapped_df['entrez_id_mouse'].notna().sum())
    n_ensembl   = int(mapped_df['ensembl_gene_mouse'].notna().sum())

    unmapped_ids = df.loc[df['mapping_status'] == 'unmapped', 'protein_id'].tolist()
    unmapped_preview = ', '.join(unmapped_ids[:10])
    if len(unmapped_ids) > 10:
        unmapped_preview += f' ... ({len(unmapped_ids) - 10} more)'

    lines = [
        '=' * 70,
        '  ProSIFT Validation Report -- Part 3: UniProt ID Mapping',
        f'  Run: {run_id}',
        f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '=' * 70,
        '',
        '--- Mapping Summary ---',
        '',
        f'  Proteins submitted to UniProt:    {n_input}',
        f'  Cache:                            {"HIT" if cache_hit else "MISS"} '
                                               f'(key: {cache_key[:12]}...)',
        '',
        '  Mapping status:',
        f'    Mapped:                         {n_mapped} ({_pct(n_mapped)})',
        f'    Isoform collapsed:              {n_isoform} ({_pct(n_isoform)})',
        f'    Accession redirected:           {n_redirected} ({_pct(n_redirected)})',
        f'    Multiple mappings:              {n_multi} ({_pct(n_multi)})',
        f'    Unmapped:                       {n_unmapped} ({_pct(n_unmapped)})',
        '',
        '--- Annotation Field Coverage ---',
        '  (excludes unmapped proteins)',
        '',
        f'  Gene symbol (mouse):              {n_gene_sym} / {n_input - n_unmapped}',
        f'  Entrez ID (mouse):                {n_entrez} / {n_input - n_unmapped}',
        f'  Ensembl gene (mouse):             {n_ensembl} / {n_input - n_unmapped}',
        f'  (Denominator excludes {n_unmapped} unmapped protein(s) with no annotation.)',
        '',
    ]

    # --- Ortholog mapping section ---
    if organism == 'mouse':
        orth_counts  = df['ortholog_mapping_status'].value_counts()
        n_121        = int(orth_counts.get('one_to_one',  0))
        n_12m        = int(orth_counts.get('one_to_many', 0))
        n_none       = int(orth_counts.get('no_ortholog', 0))
        n_null       = int(df['ortholog_mapping_status'].isna().sum())
        n_mappable   = n_input - n_unmapped
        n_sym_fb     = int(
            df['mapping_notes']
            .fillna('')
            .str.contains('gene symbol fallback', case=False)
            .sum()
        )
        lines += [
            '--- Ortholog Mapping (Mouse -> Human, Ensembl BioMart) ---',
            '',
            f'  One-to-one:                       {n_121} ({_pct(n_121)})',
            f'  One-to-many:                      {n_12m} ({_pct(n_12m)})',
            f'  No ortholog:                      {n_none} ({_pct(n_none)})',
            f'  Null (UniProt-unmapped, skipped): {n_null} ({_pct(n_null)})',
            f'  Total with ortholog:              {n_121 + n_12m} / {n_mappable}',
            f'  Gene symbol fallback used:        {n_sym_fb} proteins',
            '',
        ]
    else:
        lines += [
            '--- Ortholog Mapping ---',
            '',
            f'  Organism: {organism} -- ortholog mapping not applicable.',
            '',
        ]

    if n_unmapped > 0:
        lines += [
            '--- Unmapped Protein IDs ---',
            '',
            f'  {n_unmapped} protein(s) could not be mapped. These are retained in',
            '  the mapping table with null annotation fields.',
            '  Common causes: obsolete accessions, contaminant prefixes,',
            '  non-UniProt IDs.',
            '',
            f'  Unmapped IDs: {unmapped_preview}',
            '',
        ]

    if n_redirected > 0:
        redirected = df.loc[df['mapping_status'] == 'accession_redirected',
                            ['protein_id', 'uniprot_accession']]
        redirect_lines = [
            f'    {row.protein_id}  ->  {row.uniprot_accession}'
            for row in redirected.itertuples()
        ]
        lines += [
            '--- Accession-Redirected Protein IDs ---',
            '',
            f'  {n_redirected} protein(s) submitted as secondary or merged accessions.',
            '  UniProt returned a different canonical accession for each. Abundance',
            '  data remains keyed on protein_id; downstream annotation uses uniprot_accession.',
            '  This typically indicates the search database FASTA was built from an',
            '  older UniProt release.',
            '',
            *redirect_lines,
            '',
        ]

    if n_multi > 0:
        multi_ids = df.loc[
            df['mapping_status'] == 'multiple_mappings', 'protein_id'
        ].tolist()
        lines += [
            '--- Multiple-Mapping Protein IDs ---',
            '',
            f'  {n_multi} protein(s) returned more than one UniProt entry.',
            '  The first entry was used; see mapping_notes for alternatives.',
            f'  IDs: {", ".join(multi_ids[:10])}',
            '',
        ]

    lines += [
        '--- Cache Configuration ---',
        '',
        f'  Cache lifetime:                   {cache_days} days',
        f'  Cache key:                        {cache_key}',
        f'  Schema version:                   {CURRENT_SCHEMA_VERSION}',
        '',
        '=' * 70,
    ]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    run_id    = args.run_id
    outdir    = args.outdir
    cachedir  = args.cachedir

    os.makedirs(outdir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load params and extract settings
    # ----------------------------------------------------------
    print(f'[{run_id}] Loading params...')
    params = load_params(args.params)

    id_col      = params['input']['protein_id_column']
    organism    = params['project']['organism']
    cache_days  = params.get('databases', {}).get('cache_days', 30)

    # ----------------------------------------------------------
    # 2. Extract protein IDs from filtered matrix
    # ----------------------------------------------------------
    print(f'[{run_id}] Reading filtered matrix: {args.matrix}')
    protein_ids = extract_protein_ids(args.matrix, id_col)
    print(f'  {len(protein_ids)} proteins')

    if len(protein_ids) == 0:
        print('Error: filtered matrix contains no proteins.')
        sys.exit(1)

    # ----------------------------------------------------------
    # 3. Check cache
    # ----------------------------------------------------------
    cache_key = compute_cache_key(protein_ids, organism)
    print(f'[{run_id}] Cache key: {cache_key}')

    mapping_df = load_cache(cachedir, cache_key, cache_days)
    cache_hit  = mapping_df is not None

    # ----------------------------------------------------------
    # 4. Query UniProt if cache miss
    # ----------------------------------------------------------
    if not cache_hit:
        print(f'[{run_id}] Cache miss -- querying UniProt ID mapping API...')
        api_results = query_uniprot(protein_ids)
        print(f'  Total results returned: {len(api_results)}')
        mapping_df = parse_json_results(api_results, protein_ids)

        # ----------------------------------------------------------
        # 5. Layer 2 ortholog mapping (mouse only)
        #    Runs after UniProt mapping and before caching so that
        #    the cached Parquet contains the complete table.
        # ----------------------------------------------------------
        if organism == 'mouse':
            print(f'[{run_id}] Running Layer 2 ortholog mapping via Ensembl BioMart...')
            mapping_df = map_orthologs(mapping_df)
        else:
            print(f'[{run_id}] Organism is "{organism}" -- skipping ortholog mapping.')

        # ----------------------------------------------------------
        # 6. Cache the completed mapping table (includes ortholog data)
        # ----------------------------------------------------------
        print(f'[{run_id}] Saving mapping table to cache...')
        save_cache(cachedir, cache_key, mapping_df)
    else:
        # Re-order cached table to match current protein_id order (may differ
        # if proteins were re-ordered between runs with the same set of IDs)
        id_order = {pid: i for i, pid in enumerate(protein_ids)}
        mapping_df = mapping_df.copy()
        mapping_df['_order'] = mapping_df['protein_id'].map(id_order)
        mapping_df = mapping_df.sort_values('_order').drop(columns='_order')

    # ----------------------------------------------------------
    # 7. Write outputs
    # ----------------------------------------------------------
    mapping_out = os.path.join(outdir, f'{run_id}.id_mapping.parquet')
    report_out  = os.path.join(outdir, f'{run_id}.validation_report_part3.txt')

    mapping_df.to_parquet(mapping_out, index=False)
    print(f'[{run_id}] Written: {mapping_out}')

    write_report(
        report_out,
        run_id=run_id,
        n_input=len(protein_ids),
        df=mapping_df,
        cache_hit=cache_hit,
        cache_key=cache_key,
        cache_days=cache_days,
        organism=organism,
    )
    print(f'[{run_id}] Written: {report_out}')

    # Summary
    status_counts = mapping_df['mapping_status'].value_counts()
    print(f'\n[{run_id}] Mapping complete:')
    for status, count in status_counts.items():
        print(f'  {status}: {count} ({100 * count / len(protein_ids):.1f}%)')

    if organism == 'mouse':
        orth_counts = mapping_df['ortholog_mapping_status'].value_counts()
        print(f'\n[{run_id}] Ortholog mapping:')
        for status, count in orth_counts.items():
            print(f'  {status}: {count} ({100 * count / len(protein_ids):.1f}%)')

    print(f'\n[{run_id}] Done.')


if __name__ == '__main__':
    main()
