"""Tests for prestige.dataloaders.dedup_store module."""

import pytest

import prestige
from prestige.dataloaders import (
    DedupConfig,
    DedupMode,
    DedupStore,
    create_dedup_store,
)

# Marker for tests that require semantic mode
skip_if_no_semantic = pytest.mark.skipif(
    not prestige.SEMANTIC_AVAILABLE, reason="Semantic deduplication not available"
)


class TestDedupStoreBasic:
    """Basic tests for DedupStore."""

    def test_open_close(self, exact_config):
        """Test opening and closing store."""
        store = DedupStore(exact_config)
        store.open()

        assert store._is_open
        assert store._store is not None

        store.close()

        assert not store._is_open
        assert store._store is None

    def test_context_manager(self, exact_config):
        """Test context manager protocol."""
        with DedupStore(exact_config) as store:
            assert store._is_open
            store.check_and_add("key", "value")

        assert not store._is_open

    def test_double_open_raises(self, exact_config):
        """Test that double open raises error."""
        store = DedupStore(exact_config)
        store.open()

        with pytest.raises(RuntimeError, match="already open"):
            store.open()

        store.close()

    def test_check_and_add_without_open_raises(self, exact_config):
        """Test that operations without open raise error."""
        store = DedupStore(exact_config)

        with pytest.raises(RuntimeError, match="not open"):
            store.check_and_add("key", "value")


class TestDedupStoreExact:
    """Tests for exact deduplication."""

    def test_unique_items(self, exact_config):
        """Test that unique items are not marked as duplicates."""
        with DedupStore(exact_config) as store:
            is_dup1, obj_id1 = store.check_and_add("k1", "value one")
            is_dup2, obj_id2 = store.check_and_add("k2", "value two")
            is_dup3, obj_id3 = store.check_and_add("k3", "value three")

            assert not is_dup1
            assert not is_dup2
            assert not is_dup3

            # All should have different object IDs
            assert obj_id1 != obj_id2
            assert obj_id2 != obj_id3

    def test_exact_duplicates(self, exact_config):
        """Test that exact duplicates are detected."""
        with DedupStore(exact_config) as store:
            is_dup1, obj_id1 = store.check_and_add("k1", "same value")
            is_dup2, obj_id2 = store.check_and_add("k2", "same value")
            is_dup3, obj_id3 = store.check_and_add("k3", "different value")

            assert not is_dup1  # First occurrence
            assert is_dup2  # Duplicate
            assert not is_dup3  # Different value

            # Duplicates share same object ID
            assert obj_id1 == obj_id2
            assert obj_id1 != obj_id3

    def test_mixed_duplicates(self, exact_config, sample_data):
        """Test processing mixed data with duplicates."""
        with DedupStore(exact_config) as store:
            duplicates = 0
            uniques = 0

            for i, item in enumerate(sample_data):
                is_dup, _ = store.check_and_add(f"item_{i}", item["text"])
                if is_dup:
                    duplicates += 1
                else:
                    uniques += 1

            # sample_data has 8 items, 5 unique
            assert uniques == 5
            assert duplicates == 3


class TestDedupStoreTracking:
    """Tests for tracking functionality."""

    def test_unique_count(self, exact_config):
        """Test unique count tracking."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value one")
            store.check_and_add("k2", "value one")  # Duplicate
            store.check_and_add("k3", "value two")

            assert store.unique_count == 2

    def test_total_keys(self, exact_config):
        """Test total keys tracking."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value one")
            store.check_and_add("k2", "value one")
            store.check_and_add("k3", "value two")

            assert store.total_keys == 3

    def test_dedup_ratio(self, exact_config):
        """Test dedup ratio calculation."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value one")
            store.check_and_add("k2", "value one")  # Duplicate
            store.check_and_add("k3", "value two")
            store.check_and_add("k4", "value two")  # Duplicate

            # 4 keys, 2 unique -> ratio of 2.0
            assert store.dedup_ratio == 2.0

    def test_get_object_id(self, exact_config):
        """Test getting object ID for a key."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value")
            obj_id = store.get_object_id("k1")

            assert obj_id is not None
            assert isinstance(obj_id, bytes)

    def test_is_duplicate_of(self, exact_config):
        """Test checking if two keys are duplicates."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "same value")
            store.check_and_add("k2", "same value")
            store.check_and_add("k3", "different")

            assert store.is_duplicate_of("k1", "k2")
            assert not store.is_duplicate_of("k1", "k3")


class TestDedupStoreMetrics:
    """Tests for metrics collection."""

    def test_metrics_collection(self, exact_config):
        """Test that metrics are collected."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value one")
            store.check_and_add("k2", "value one")
            store.check_and_add("k3", "value two")

            metrics = store.get_metrics()

            assert metrics.storage_bytes >= 0
            assert metrics.store_total_keys >= 0

    def test_metrics_reset(self, exact_config):
        """Test resetting tracking state."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value one")
            store.check_and_add("k2", "value one")

            assert store.unique_count == 1
            assert store.total_keys == 2

            store.reset_tracking()

            assert store.unique_count == 0
            assert store.total_keys == 0


class TestDedupStoreContamination:
    """Tests for contamination checking."""

    def test_check_contamination(self, exact_config):
        """Test checking for contamination."""
        with DedupStore(exact_config) as store:
            # Add some items
            store.check_and_add("k1", "existing value")
            store.check_and_add("k2", "another value")

            # Check contamination
            is_cont1, _ = store.check_contamination("check1", "existing value")
            is_cont2, _ = store.check_contamination("check2", "new value")

            assert is_cont1  # Should be contaminated
            assert not is_cont2  # Should not be contaminated

    def test_check_contamination_does_not_add(self, exact_config):
        """Test that check_contamination doesn't add to store."""
        with DedupStore(exact_config) as store:
            initial_count = store.unique_count

            store.check_contamination("temp", "new value")

            # Count should not have increased
            assert store.unique_count == initial_count


class TestDedupStoreFlush:
    """Tests for flush functionality."""

    def test_flush(self, exact_config):
        """Test that flush doesn't raise."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value")
            store.flush()  # Should not raise

    def test_get_health(self, exact_config):
        """Test getting health stats."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value")
            store.flush()

            health = store.get_health()

            assert isinstance(health, dict)
            assert "total_keys" in health or "total_objects" in health


class TestCreateDedupStore:
    """Tests for convenience factory function."""

    def test_create_exact_store(self, store_path):
        """Test creating exact dedup store."""
        store = create_dedup_store(
            mode="exact",
            store_path=store_path,
        )

        assert store.config.mode == DedupMode.EXACT

    def test_create_semantic_store(self, store_path):
        """Test creating semantic dedup store (config only)."""
        store = create_dedup_store(
            mode="semantic",
            threshold=0.9,
            store_path=store_path,
        )

        # Config should always be created regardless of semantic support
        assert store.config.mode == DedupMode.SEMANTIC
        assert store.config.semantic_threshold == 0.9


class TestSemanticModeAvailability:
    """Tests for semantic mode availability handling."""

    @skip_if_no_semantic
    def test_semantic_store_opens_when_available(self, semantic_config):
        """Test that semantic store opens when semantic support is available."""
        if semantic_config.semantic_model_path is None:
            pytest.skip("Semantic model not downloaded (run: python benchmarks/semantic_dedup/models.py bge-small)")

        with DedupStore(semantic_config) as store:
            assert store._is_open

    @pytest.mark.skipif(
        prestige.SEMANTIC_AVAILABLE,
        reason="Test only runs when semantic mode is NOT available",
    )
    def test_semantic_store_raises_when_not_available(self, semantic_config):
        """Test that semantic store raises error when semantic support is not available."""
        store = DedupStore(semantic_config)

        with pytest.raises(RuntimeError, match="PRESTIGE_ENABLE_SEMANTIC"):
            store.open()


class TestDedupStoreRepr:
    """Tests for string representation."""

    def test_repr_closed(self, exact_config):
        """Test repr when store is closed."""
        store = DedupStore(exact_config)
        r = repr(store)

        assert "DedupStore" in r
        assert "closed" in r

    def test_repr_open(self, exact_config):
        """Test repr when store is open."""
        with DedupStore(exact_config) as store:
            store.check_and_add("k1", "value")
            r = repr(store)

            assert "DedupStore" in r
            assert "open" in r
