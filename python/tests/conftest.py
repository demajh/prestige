"""Pytest fixtures for prestige tests."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

import prestige


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    path = Path(tempfile.mkdtemp(prefix="prestige_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def db_path(temp_dir):
    """Provide a database path within temp directory."""
    return str(temp_dir / "test_db")


@pytest.fixture
def store(db_path):
    """Create a test store with default options."""
    with prestige.open(db_path) as s:
        yield s


@pytest.fixture
def store_with_ttl(db_path):
    """Create a test store with TTL enabled."""
    with prestige.open(db_path, default_ttl_seconds=60) as s:
        yield s


@pytest.fixture
def store_with_cache_limit(db_path):
    """Create a test store with size limit."""
    with prestige.open(db_path, max_store_bytes=1_000_000) as s:
        yield s


# Skip markers for optional features
skip_if_no_semantic = pytest.mark.skipif(
    not prestige.SEMANTIC_AVAILABLE,
    reason="Semantic deduplication not available"
)

skip_if_no_server = pytest.mark.skipif(
    not prestige.SERVER_AVAILABLE,
    reason="Server module not available"
)
