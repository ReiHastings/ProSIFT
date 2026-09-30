#!/usr/bin/env python3
# title: test_prosift_cache.py
# project: ProSIFT (PROtein Statistical Integration and Filtering Tool)
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-04-07
# last modified: 2026-04-07
#
# purpose:
#   Unit tests for the shared caching utility (bin/prosift_cache.py).
#   Tests cache hit, miss, expiration, force_requery, metadata creation,
#   and the BulkFileCache class.
#
# inputs:
#   None (uses temporary directories)
#
# outputs:
#   Test results (stdout via pytest)
#
# usage example:
#   pytest tests/test_prosift_cache.py -v
#
#   copy/paste: pytest tests/test_prosift_cache.py -v

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add bin/ to path so we can import prosift_cache
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from prosift_cache import (
    BulkFileCache,
    ProteinCache,
    get_api_key,
    is_database_enabled,
    load_db_params,
)

# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def tmp_cache(tmp_path):
    """Create a ProteinCache in a temporary directory."""
    return ProteinCache(
        cache_dir=str(tmp_path / 'test_cache'),
        cache_days=30,
        force_requery=False,
        database_name='test_db',
    )


@pytest.fixture
def tmp_bulk_cache(tmp_path):
    """Create a BulkFileCache in a temporary directory."""
    return BulkFileCache(
        cache_dir=str(tmp_path / 'bulk_cache'),
        cache_days=30,
        force_requery=False,
        database_name='test_bulk',
    )


# ============================================================
# ProteinCache TESTS
# ============================================================

class TestProteinCacheHitMiss:
    """Test basic cache hit and miss behavior."""

    def test_cache_miss_on_empty(self, tmp_cache):
        """get() returns None for a key that was never stored."""
        result = tmp_cache.get('nonexistent_key')
        assert result is None
        assert tmp_cache.misses == 1

    def test_cache_hit_after_put(self, tmp_cache):
        """get() returns stored data after put()."""
        data = {'protein_name': 'Shank3', 'function': 'scaffold protein'}
        tmp_cache.put('Q9WTX5', data)
        result = tmp_cache.get('Q9WTX5')
        assert result == data
        assert tmp_cache.hits == 1

    def test_has_returns_false_for_missing(self, tmp_cache):
        """has() returns False for missing keys."""
        assert tmp_cache.has('nonexistent') is False

    def test_has_returns_true_after_put(self, tmp_cache):
        """has() returns True for stored keys."""
        tmp_cache.put('Q9WTX5', {'data': 'test'})
        assert tmp_cache.has('Q9WTX5') is True

    def test_stores_various_types(self, tmp_cache):
        """Cache handles dicts, lists, strings, and nested structures."""
        test_cases = [
            ('key_dict', {'a': 1, 'b': [2, 3]}),
            ('key_list', [1, 2, 3]),
            ('key_str', 'simple string'),
            ('key_nested', {'results': [{'id': 1}, {'id': 2}]}),
        ]
        for key, data in test_cases:
            tmp_cache.put(key, data)
            assert tmp_cache.get(key) == data


class TestProteinCacheExpiration:
    """Test time-based cache invalidation."""

    def test_expired_entry_returns_none(self, tmp_path):
        """Entries older than cache_days are treated as misses."""
        # Use a very short cache lifetime (0 days = always expired)
        cache = ProteinCache(
            cache_dir=str(tmp_path / 'expire_test'),
            cache_days=0,
            force_requery=False,
            database_name='test_expire',
        )
        cache.put('Q9WTX5', {'data': 'test'})

        # The file was just created, but cache_days=0 means it is immediately stale
        # We need to set the mtime to the past for a reliable test
        path = cache._key_path('Q9WTX5')
        old_time = time.time() - 86400  # 1 day ago
        os.utime(path, (old_time, old_time))

        result = cache.get('Q9WTX5')
        assert result is None
        assert cache.expired == 1

    def test_fresh_entry_returns_data(self, tmp_path):
        """Entries within cache_days are returned."""
        cache = ProteinCache(
            cache_dir=str(tmp_path / 'fresh_test'),
            cache_days=30,
            force_requery=False,
            database_name='test_fresh',
        )
        data = {'data': 'test'}
        cache.put('Q9WTX5', data)
        result = cache.get('Q9WTX5')
        assert result == data
        assert cache.hits == 1


class TestProteinCacheForceRequery:
    """Test force_requery override."""

    def test_force_requery_ignores_cache(self, tmp_path):
        """When force_requery=True, get() always returns None."""
        cache = ProteinCache(
            cache_dir=str(tmp_path / 'force_test'),
            cache_days=30,
            force_requery=True,
            database_name='test_force',
        )
        cache.put('Q9WTX5', {'data': 'test'})
        result = cache.get('Q9WTX5')
        assert result is None
        assert cache.misses == 1

    def test_force_requery_has_returns_false(self, tmp_path):
        """has() returns False when force_requery=True."""
        cache = ProteinCache(
            cache_dir=str(tmp_path / 'force_has_test'),
            cache_days=30,
            force_requery=True,
            database_name='test_force_has',
        )
        cache.put('Q9WTX5', {'data': 'test'})
        assert cache.has('Q9WTX5') is False


class TestProteinCacheMetadata:
    """Test _metadata.json creation and updates."""

    def test_metadata_created_on_init(self, tmp_cache):
        """_metadata.json is created when cache is initialized."""
        meta_path = tmp_cache.cache_dir / '_metadata.json'
        assert meta_path.exists()

    def test_metadata_contains_required_fields(self, tmp_cache):
        """Metadata includes database name, version, and timestamps."""
        meta = tmp_cache.get_metadata()
        assert meta['database'] == 'test_db'
        assert meta['cache_schema_version'] == 1
        assert 'created' in meta
        assert 'last_accessed' in meta
        assert meta['cache_days'] == 30

    def test_metadata_updates_on_reinit(self, tmp_path):
        """Re-initializing updates last_accessed without losing created."""
        cache_dir = str(tmp_path / 'meta_update')
        c1 = ProteinCache(cache_dir=cache_dir, database_name='test')
        meta1 = c1.get_metadata()
        created1 = meta1['created']

        c2 = ProteinCache(cache_dir=cache_dir, database_name='test')
        meta2 = c2.get_metadata()
        assert meta2['created'] == created1  # created unchanged
        assert 'last_accessed' in meta2


class TestProteinCacheSummary:
    """Test the summary reporting."""

    def test_summary_string(self, tmp_cache):
        """summary() returns a readable string with correct counts."""
        tmp_cache.put('A', {'x': 1})
        tmp_cache.get('A')       # hit
        tmp_cache.get('B')       # miss
        tmp_cache.get('C')       # miss
        s = tmp_cache.summary()
        assert '1 hits' in s
        assert '2 misses' in s
        assert 'test_db' in s


class TestProteinCacheKeySanitization:
    """Test that special characters in keys are handled."""

    def test_key_with_special_chars(self, tmp_cache):
        """Keys with slashes and colons are sanitized."""
        tmp_cache.put('gene/symbol:term', {'data': 1})
        result = tmp_cache.get('gene/symbol:term')
        assert result == {'data': 1}


# ============================================================
# BulkFileCache TESTS
# ============================================================

class TestBulkFileCache:

    def test_missing_file_returns_none(self, tmp_bulk_cache):
        """get_file() returns None for a file that does not exist."""
        result = tmp_bulk_cache.get_file('nonexistent.tsv.gz')
        assert result is None

    def test_fresh_file_returns_path(self, tmp_bulk_cache):
        """get_file() returns path for a fresh file."""
        # Create a dummy file
        path = tmp_bulk_cache.file_path('test.tsv.gz')
        path.write_text('dummy data')

        result = tmp_bulk_cache.get_file('test.tsv.gz')
        assert result is not None
        assert result.name == 'test.tsv.gz'

    def test_stale_file_returns_none(self, tmp_path):
        """get_file() returns None for a file older than cache_days."""
        cache = BulkFileCache(
            cache_dir=str(tmp_path / 'stale_bulk'),
            cache_days=1,
            database_name='test_stale',
        )
        path = cache.file_path('old.tsv.gz')
        path.write_text('old data')
        old_time = time.time() - 86400 * 2  # 2 days ago
        os.utime(path, (old_time, old_time))

        assert cache.get_file('old.tsv.gz') is None

    def test_force_requery_ignores_file(self, tmp_path):
        """get_file() returns None when force_requery=True."""
        cache = BulkFileCache(
            cache_dir=str(tmp_path / 'force_bulk'),
            cache_days=30,
            force_requery=True,
            database_name='test_force',
        )
        path = cache.file_path('test.tsv.gz')
        path.write_text('data')

        assert cache.get_file('test.tsv.gz') is None

    def test_is_fresh(self, tmp_bulk_cache):
        """is_fresh() correctly identifies fresh vs missing files."""
        assert tmp_bulk_cache.is_fresh('missing.tsv.gz') is False
        path = tmp_bulk_cache.file_path('present.tsv.gz')
        path.write_text('data')
        assert tmp_bulk_cache.is_fresh('present.tsv.gz') is True

    def test_metadata_created(self, tmp_bulk_cache):
        """Metadata file is created for bulk cache."""
        meta_path = tmp_bulk_cache.cache_dir / '_metadata.json'
        assert meta_path.exists()


# ============================================================
# HELPER FUNCTION TESTS
# ============================================================

class TestLoadDbParams:
    """Test load_db_params() defaults and overrides."""

    def test_defaults_applied(self):
        """Empty params dict gets all defaults."""
        result = load_db_params({})
        assert result['cache_days'] == 30
        assert result['force_requery'] is False
        assert 'uniprot' in result['enabled']
        assert result['pubmed']['min_pubs_for_score'] == 5
        assert result['disgenet']['min_score'] == 0.1

    def test_overrides(self):
        """Explicit values override defaults."""
        params = {
            'databases': {
                'cache_days': 60,
                'force_requery': True,
                'enabled': ['uniprot', 'ctd'],
                'pubmed': {
                    'search_terms': ['ketamine'],
                    'min_pubs_for_score': 10,
                },
            }
        }
        result = load_db_params(params)
        assert result['cache_days'] == 60
        assert result['force_requery'] is True
        assert result['enabled'] == ['uniprot', 'ctd']
        assert result['pubmed']['search_terms'] == ['ketamine']
        assert result['pubmed']['min_pubs_for_score'] == 10


class TestIsDatabaseEnabled:

    def test_enabled_database(self):
        db_params = {'enabled': ['uniprot', 'pubmed', 'ctd']}
        assert is_database_enabled(db_params, 'uniprot') is True

    def test_disabled_database(self):
        db_params = {'enabled': ['uniprot', 'ctd']}
        assert is_database_enabled(db_params, 'pubmed') is False


class TestGetApiKey:

    def test_key_present(self, monkeypatch):
        monkeypatch.setenv('TEST_API_KEY', 'abc123')
        assert get_api_key('TEST_API_KEY') == 'abc123'

    def test_key_missing_required(self, monkeypatch):
        monkeypatch.delenv('MISSING_KEY', raising=False)
        with pytest.raises(SystemExit):
            get_api_key('MISSING_KEY', required=True)

    def test_key_missing_optional(self, monkeypatch):
        monkeypatch.delenv('MISSING_KEY', raising=False)
        assert get_api_key('MISSING_KEY', required=False) is None
