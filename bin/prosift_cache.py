#!/usr/bin/env python3
# title: prosift_cache.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-07
# last modified: 2026-04-07
#
# purpose:
#   Shared caching utility for Module 06 database query scripts. Provides
#   per-protein JSON file caching with age-based invalidation, force_requery
#   override, and _metadata.json management. Two modes: per-key (API databases)
#   and whole-file (CTD bulk download).
#
#   This module implements the application-level cache layer described in
#   Module 06 spec Section 4.7. It complements Nextflow's process-level
#   caching (-resume) by enabling fine-grained per-protein cache management.
#
# inputs:
#   Imported as a Python module by the five database query scripts.
#   Not intended to be run as a standalone script.
#
# outputs:
#   Cache directory structure:
#     {cache_dir}/
#       {cache_key}.json     -- per-protein cached API responses
#       _metadata.json       -- cache version, last update, source info
#
# usage example:
#   from prosift_cache import ProteinCache
#
#   cache = ProteinCache(
#       cache_dir='./prosift_cache/databases/uniprot',
#       cache_days=30,
#       force_requery=False,
#       database_name='uniprot',
#   )
#   data = cache.get('Q9WTX5')
#   if data is None:
#       data = query_api('Q9WTX5')
#       cache.put('Q9WTX5', data)
#
#   copy/paste: N/A (library module, not a standalone script)

import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache metadata schema version. Bump when the cache format changes
# in a backwards-incompatible way.
CACHE_SCHEMA_VERSION = 1


# ============================================================
# PROTEIN CACHE (per-key JSON files)
# ============================================================

class ProteinCache:
    """Application-level cache for per-protein API responses.

    Each cached entry is a JSON file named {cache_key}.json. Raw API responses
    are stored unfiltered so that changing filter parameters (e.g., DisGeNET
    min_score) does not invalidate the cache.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for this database's cache files.
    cache_days : int
        Maximum age (in days) before a cached entry is considered stale.
    force_requery : bool
        If True, ignore all cached entries and re-query everything.
    database_name : str
        Name of the database (for metadata and logging).
    """

    def __init__(
        self,
        cache_dir: str,
        cache_days: int = 30,
        force_requery: bool = False,
        database_name: str = 'unknown',
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_days = cache_days
        self.force_requery = force_requery
        self.database_name = database_name

        # Counters for summary logging
        self.hits = 0
        self.misses = 0
        self.expired = 0
        self.errors = 0

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or update metadata
        self._init_metadata()

    # --- Public API ---

    def get(self, cache_key: str) -> Any | None:
        """Retrieve a cached entry by key.

        Returns the deserialized JSON data if the cache file exists and is not
        stale. Returns None if the entry is missing, expired, or force_requery
        is True.

        Parameters
        ----------
        cache_key : str
            Cache key (used as the JSON filename, minus extension).
            Must be filesystem-safe (no slashes, colons, etc.).

        Returns
        -------
        dict or list or None
            The cached data, or None if not available.
        """
        if self.force_requery:
            self.misses += 1
            return None

        path = self._key_path(cache_key)
        if not path.exists():
            self.misses += 1
            return None

        # Check age
        age_days = self._file_age_days(path)
        if age_days > self.cache_days:
            self.expired += 1
            logger.debug('Cache expired for %s (%.1f days old, limit %d)',
                         cache_key, age_days, self.cache_days)
            return None

        # Read and deserialize
        try:
            with open(path) as fh:
                data = json.load(fh)
            self.hits += 1
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Cache read error for %s: %s', cache_key, exc)
            self.errors += 1
            return None

    def put(self, cache_key: str, data: Any) -> None:
        """Store data in the cache.

        Parameters
        ----------
        cache_key : str
            Cache key (filesystem-safe string).
        data : any JSON-serializable
            The data to cache.
        """
        path = self._key_path(cache_key)
        try:
            with open(path, 'w') as fh:
                json.dump(data, fh, indent=2, default=str)
        except OSError as exc:
            logger.warning('Cache write error for %s: %s', cache_key, exc)
            self.errors += 1

    def has(self, cache_key: str) -> bool:
        """Check if a valid (non-expired) cache entry exists.

        Returns False if force_requery is True.
        """
        if self.force_requery:
            return False
        path = self._key_path(cache_key)
        if not path.exists():
            return False
        return self._file_age_days(path) <= self.cache_days

    def summary(self) -> str:
        """Return a human-readable summary of cache activity."""
        total = self.hits + self.misses + self.expired
        return (
            f'Cache summary ({self.database_name}): '
            f'{self.hits} hits, {self.misses} misses, '
            f'{self.expired} expired, {self.errors} errors '
            f'({total} total lookups)'
        )

    # --- Metadata management ---

    def _init_metadata(self) -> None:
        """Create or update _metadata.json in the cache directory."""
        meta_path = self.cache_dir / '_metadata.json'
        now = datetime.datetime.now(datetime.UTC).isoformat()

        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
            except (json.JSONDecodeError, OSError):
                meta = {}
        else:
            meta = {
                'database': self.database_name,
                'cache_schema_version': CACHE_SCHEMA_VERSION,
                'created': now,
            }

        meta['last_accessed'] = now
        meta['cache_days'] = self.cache_days
        meta['force_requery'] = self.force_requery

        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2)

    def get_metadata(self) -> dict:
        """Read and return the _metadata.json contents."""
        meta_path = self.cache_dir / '_metadata.json'
        if not meta_path.exists():
            return {}
        with open(meta_path) as fh:
            return json.load(fh)

    # --- Internal helpers ---

    def _key_path(self, cache_key: str) -> Path:
        """Return the filesystem path for a cache key."""
        # Sanitize the key: replace characters that are problematic in filenames
        safe_key = cache_key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return self.cache_dir / f'{safe_key}.json'

    @staticmethod
    def _file_age_days(path: Path) -> float:
        """Return the age of a file in days (based on modification time)."""
        mtime = path.stat().st_mtime
        age_seconds = time.time() - mtime
        return age_seconds / 86400.0


# ============================================================
# BULK FILE CACHE (for CTD)
# ============================================================

class BulkFileCache:
    """Cache manager for bulk download files (e.g., CTD TSV).

    Unlike ProteinCache, this does not store per-protein entries. Instead,
    it manages a single downloaded file and tracks its age for refresh.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for the cached bulk file.
    cache_days : int
        Maximum age before the file should be re-downloaded.
    force_requery : bool
        If True, treat the file as stale regardless of age.
    database_name : str
        Name of the database (for metadata and logging).
    """

    def __init__(
        self,
        cache_dir: str,
        cache_days: int = 30,
        force_requery: bool = False,
        database_name: str = 'unknown',
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_days = cache_days
        self.force_requery = force_requery
        self.database_name = database_name

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_metadata()

    def get_file(self, filename: str) -> Path | None:
        """Return the path to the cached file if it exists and is fresh.

        Parameters
        ----------
        filename : str
            Name of the cached file (e.g., 'CTD_chem_gene_ixns.tsv.gz').

        Returns
        -------
        Path or None
            Path to the file if valid, None if missing or stale.
        """
        path = self.cache_dir / filename
        if not path.exists():
            return None
        if self.force_requery:
            return None
        age = self._file_age_days(path)
        if age > self.cache_days:
            logger.info('Bulk file %s is stale (%.1f days old, limit %d)',
                        filename, age, self.cache_days)
            return None
        return path

    def file_path(self, filename: str) -> Path:
        """Return the expected path for a bulk file (whether or not it exists)."""
        return self.cache_dir / filename

    def is_fresh(self, filename: str) -> bool:
        """Check if the file exists and is within the cache lifetime."""
        return self.get_file(filename) is not None

    def _init_metadata(self) -> None:
        """Create or update _metadata.json."""
        meta_path = self.cache_dir / '_metadata.json'
        now = datetime.datetime.now(datetime.UTC).isoformat()

        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
            except (json.JSONDecodeError, OSError):
                meta = {}
        else:
            meta = {
                'database': self.database_name,
                'cache_schema_version': CACHE_SCHEMA_VERSION,
                'created': now,
            }

        meta['last_accessed'] = now
        meta['cache_days'] = self.cache_days

        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2)

    @staticmethod
    def _file_age_days(path: Path) -> float:
        """Return the age of a file in days."""
        mtime = path.stat().st_mtime
        return (time.time() - mtime) / 86400.0


# ============================================================
# HELPER: Load database parameters from params.yml
# ============================================================

def load_db_params(params: dict) -> dict:
    """Extract the databases section from a parsed params.yml.

    Returns a dict with normalized defaults for all database parameters.

    Parameters
    ----------
    params : dict
        Parsed params.yml contents.

    Returns
    -------
    dict
        Database configuration with defaults applied.
    """
    db = params.get('databases', {})
    return {
        'enabled': db.get('enabled', ['uniprot', 'pubmed', 'disgenet', 'dgidb', 'ctd']),
        'query_scope': db.get('query_scope', 'all'),
        'cache_dir': db.get('cache_dir', './prosift_cache/databases'),
        'cache_days': db.get('cache_days', 30),
        'force_requery': db.get('force_requery', False),
        'pubmed': {
            'search_terms': db.get('pubmed', {}).get('search_terms', []),
            'normalization': db.get('pubmed', {}).get('normalization', 'pmi'),
            'min_pubs_for_score': db.get('pubmed', {}).get('min_pubs_for_score', 5),
        },
        'disgenet': {
            'min_score': db.get('disgenet', {}).get('min_score', 0.1),
        },
        'api_keys': {
            'pubmed': db.get('api_keys', {}).get('pubmed', 'NCBI_API_KEY'),
            'disgenet': db.get('api_keys', {}).get('disgenet', 'DISGENET_API_KEY'),
        },
    }


def is_database_enabled(db_params: dict, database_name: str) -> bool:
    """Check if a specific database is in the enabled list.

    Parameters
    ----------
    db_params : dict
        Output of load_db_params().
    database_name : str
        One of: 'uniprot', 'pubmed', 'disgenet', 'dgidb', 'ctd'.

    Returns
    -------
    bool
    """
    return database_name in db_params['enabled']


def get_api_key(env_var_name: str, required: bool = True) -> str | None:
    """Read an API key from an environment variable.

    Parameters
    ----------
    env_var_name : str
        Name of the environment variable.
    required : bool
        If True and the variable is not set, raise an error.

    Returns
    -------
    str or None
        The API key value, or None if not set and not required.

    Raises
    ------
    SystemExit
        If required is True and the environment variable is not set.
    """
    value = os.environ.get(env_var_name)
    if value is None and required:
        logger.error(
            'Required API key not found. Set the %s environment variable.\n'
            'Example: export %s=your_key_here',
            env_var_name, env_var_name,
        )
        raise SystemExit(1)
    return value
