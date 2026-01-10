"""Tests for prestige Store class."""

import pytest
import prestige


class TestBasicOperations:
    """Test basic KV operations."""

    def test_put_get_bytes(self, store):
        store.put("key1", b"value1")
        assert store.get("key1") == b"value1"

    def test_put_get_str(self, store):
        store.put("key1", "value1")
        assert store.get("key1") == b"value1"  # Returns bytes
        assert store.get("key1", decode=True) == "value1"

    def test_get_not_found(self, store):
        with pytest.raises(prestige.NotFoundError):
            store.get("nonexistent")

    def test_get_with_default(self, store):
        result = store.get("nonexistent", default=b"default")
        assert result == b"default"

    def test_get_with_default_none(self, store):
        result = store.get("nonexistent", default=None)
        assert result is None

    def test_delete(self, store):
        store.put("key1", b"value1")
        store.delete("key1")
        with pytest.raises(prestige.NotFoundError):
            store.get("key1")

    def test_contains(self, store):
        store.put("key1", b"value1")
        assert "key1" in store
        assert "nonexistent" not in store


class TestDeduplication:
    """Test content deduplication."""

    def test_exact_dedup(self, store):
        store.put("key1", b"identical content")
        store.put("key2", b"identical content")
        store.put("key3", b"different content")

        assert store.count_keys() == 3
        assert store.count_unique_values() == 2

    def test_dedup_ratio(self, store):
        for i in range(100):
            store.put(f"key_{i}", b"same value")

        health = store.get_health()
        assert health["total_keys"] == 100
        assert health["total_objects"] == 1
        assert health["dedup_ratio"] == pytest.approx(100.0)


class TestDictLikeInterface:
    """Test dict-like operations."""

    def test_setitem_getitem(self, store):
        store["key1"] = b"value1"
        assert store["key1"] == b"value1"

    def test_delitem(self, store):
        store["key1"] = b"value1"
        del store["key1"]
        assert "key1" not in store

    def test_len(self, store):
        store["key1"] = b"value1"
        store["key2"] = b"value2"
        assert len(store) >= 2  # Approximate count


class TestContextManager:
    """Test context manager behavior."""

    def test_context_manager(self, db_path):
        with prestige.open(db_path) as store:
            store.put("key1", b"value1")
            assert not store.closed

        assert store.closed

    def test_explicit_close(self, db_path):
        store = prestige.Store.open(db_path)
        store.put("key1", b"value1")
        store.close()

        with pytest.raises(ValueError, match="closed"):
            store.put("key2", b"value2")

    def test_persistence(self, db_path):
        # Write and close
        with prestige.open(db_path) as store:
            store.put("key1", b"value1")

        # Reopen and verify
        with prestige.open(db_path) as store:
            assert store.get("key1") == b"value1"


class TestListKeys:
    """Test key listing functionality."""

    def test_list_all_keys(self, store):
        store.put("a", b"1")
        store.put("b", b"2")
        store.put("c", b"3")

        keys = store.list_keys()
        assert set(keys) == {"a", "b", "c"}

    def test_list_with_limit(self, store):
        for i in range(100):
            store.put(f"key_{i:03d}", b"value")

        keys = store.list_keys(limit=10)
        assert len(keys) == 10

    def test_list_with_prefix(self, store):
        store.put("user_1", b"1")
        store.put("user_2", b"2")
        store.put("item_1", b"3")

        keys = store.list_keys(prefix="user_")
        assert set(keys) == {"user_1", "user_2"}


class TestCounting:
    """Test counting operations."""

    def test_count_keys_exact(self, store):
        for i in range(10):
            store.put(f"key_{i}", b"value")

        assert store.count_keys(approximate=False) == 10

    def test_count_keys_approximate(self, store):
        for i in range(10):
            store.put(f"key_{i}", b"value")

        # Approximate count should be close
        count = store.count_keys(approximate=True)
        assert count >= 0  # Just verify it works

    def test_count_unique_values(self, store):
        store.put("key1", b"value_a")
        store.put("key2", b"value_a")  # Duplicate
        store.put("key3", b"value_b")

        assert store.count_unique_values() == 2


class TestCacheManagement:
    """Test cache management operations."""

    def test_sweep_empty_store(self, store):
        deleted = store.sweep()
        assert deleted == 0

    def test_prune_empty(self, store):
        deleted = store.prune(max_age_seconds=0, max_idle_seconds=0)
        assert deleted >= 0

    def test_get_health(self, store):
        store.put("key1", b"value1")
        store.put("key2", b"value2")
        store.put("key3", b"value1")  # Duplicate

        health = store.get_health()

        assert health["total_keys"] == 3
        assert health["total_objects"] == 2
        assert health["dedup_ratio"] == pytest.approx(1.5)
        assert health["total_bytes"] > 0


class TestFlush:
    """Test flush operation."""

    def test_flush(self, store):
        store.put("key1", b"value1")
        store.flush()  # Should not raise


class TestProperties:
    """Test store properties."""

    def test_path(self, db_path):
        with prestige.open(db_path) as store:
            assert store.path == db_path

    def test_closed(self, db_path):
        store = prestige.Store.open(db_path)
        assert not store.closed
        store.close()
        assert store.closed

    def test_total_bytes(self, store):
        store.put("key1", b"value1")
        assert store.total_bytes >= 0


class TestRepr:
    """Test string representation."""

    def test_repr(self, db_path):
        with prestige.open(db_path) as store:
            repr_str = repr(store)
            assert "prestige.Store" in repr_str
            assert db_path in repr_str
            assert "closed=False" in repr_str


class TestOptions:
    """Test Options class."""

    def test_default_options(self):
        options = prestige.Options()
        assert options.block_cache_bytes > 0
        assert options.default_ttl_seconds == 0

    def test_set_options(self):
        options = prestige.Options()
        options.default_ttl_seconds = 3600
        options.max_store_bytes = 10 * 1024 * 1024 * 1024

        assert options.default_ttl_seconds == 3600
        assert options.max_store_bytes == 10 * 1024 * 1024 * 1024

    def test_options_repr(self):
        options = prestige.Options()
        repr_str = repr(options)
        assert "prestige.Options" in repr_str


class TestOpenFunction:
    """Test the open() convenience function."""

    def test_open_with_kwargs(self, db_path):
        with prestige.open(db_path, default_ttl_seconds=3600) as store:
            store.put("key", b"value")

    def test_open_with_options(self, db_path):
        options = prestige.Options()
        options.default_ttl_seconds = 3600
        with prestige.open(db_path, options) as store:
            store.put("key", b"value")

    def test_open_invalid_kwarg(self, db_path):
        with pytest.raises(ValueError, match="Unknown option"):
            prestige.open(db_path, invalid_option=123)


class TestExceptions:
    """Test exception types."""

    def test_not_found_error(self, store):
        with pytest.raises(prestige.NotFoundError):
            store.get("nonexistent")

    def test_exception_hierarchy(self):
        assert issubclass(prestige.NotFoundError, prestige.PrestigeError)
        assert issubclass(prestige.InvalidArgumentError, prestige.PrestigeError)
        assert issubclass(prestige.IOError, prestige.PrestigeError)
        assert issubclass(prestige.CorruptionError, prestige.PrestigeError)
        assert issubclass(prestige.BusyError, prestige.PrestigeError)
        assert issubclass(prestige.TimedOutError, prestige.PrestigeError)


class TestVersion:
    """Test version info."""

    def test_version_string(self):
        assert prestige.__version__
        assert isinstance(prestige.__version__, str)

    def test_feature_flags(self):
        assert isinstance(prestige.SEMANTIC_AVAILABLE, bool)
        assert isinstance(prestige.SERVER_AVAILABLE, bool)


@pytest.mark.skipif(
    not prestige.SEMANTIC_AVAILABLE,
    reason="Semantic deduplication not available"
)
class TestSemanticOptions:
    """Test semantic deduplication options including judge LLM."""

    def test_semantic_mode_option(self):
        options = prestige.Options()
        assert options.dedup_mode == prestige.DedupMode.EXACT
        options.dedup_mode = prestige.DedupMode.SEMANTIC
        assert options.dedup_mode == prestige.DedupMode.SEMANTIC

    def test_semantic_threshold_option(self):
        options = prestige.Options()
        options.semantic_threshold = 0.85
        assert options.semantic_threshold == pytest.approx(0.85)

    def test_judge_enabled_option(self):
        options = prestige.Options()
        assert options.semantic_judge_enabled is False
        options.semantic_judge_enabled = True
        assert options.semantic_judge_enabled is True

    def test_judge_threshold_option(self):
        options = prestige.Options()
        assert options.semantic_judge_threshold == pytest.approx(0.75)
        options.semantic_judge_threshold = 0.80
        assert options.semantic_judge_threshold == pytest.approx(0.80)

    def test_judge_model_path_option(self):
        options = prestige.Options()
        assert options.semantic_judge_model_path == ""
        options.semantic_judge_model_path = "/path/to/prometheus.gguf"
        assert options.semantic_judge_model_path == "/path/to/prometheus.gguf"

    def test_judge_context_size_option(self):
        options = prestige.Options()
        assert options.semantic_judge_context_size == 4096
        options.semantic_judge_context_size = 8192
        assert options.semantic_judge_context_size == 8192

    def test_judge_gpu_layers_option(self):
        options = prestige.Options()
        assert options.semantic_judge_gpu_layers == 0
        options.semantic_judge_gpu_layers = -1  # All layers to GPU
        assert options.semantic_judge_gpu_layers == -1

    def test_judge_num_threads_option(self):
        options = prestige.Options()
        assert options.semantic_judge_num_threads == 0  # 0 = all cores
        options.semantic_judge_num_threads = 4
        assert options.semantic_judge_num_threads == 4

    def test_judge_max_tokens_option(self):
        options = prestige.Options()
        assert options.semantic_judge_max_tokens == 256
        options.semantic_judge_max_tokens = 512
        assert options.semantic_judge_max_tokens == 512

    def test_rnn_enabled_option(self):
        options = prestige.Options()
        assert options.semantic_rnn_enabled is False
        options.semantic_rnn_enabled = True
        assert options.semantic_rnn_enabled is True

    def test_margin_enabled_option(self):
        options = prestige.Options()
        assert options.semantic_margin_enabled is False
        options.semantic_margin_enabled = True
        assert options.semantic_margin_enabled is True

    def test_margin_threshold_option(self):
        options = prestige.Options()
        assert options.semantic_margin_threshold == pytest.approx(0.05)
        options.semantic_margin_threshold = 0.10
        assert options.semantic_margin_threshold == pytest.approx(0.10)
