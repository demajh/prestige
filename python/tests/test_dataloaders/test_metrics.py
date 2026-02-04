"""Tests for prestige.dataloaders.metrics module."""

import time

import pytest

from prestige.dataloaders import DedupMetrics


class TestDedupMetrics:
    """Tests for DedupMetrics dataclass."""

    def test_initial_values(self):
        """Test initial metric values are zero."""
        metrics = DedupMetrics()
        assert metrics.total_seen == 0
        assert metrics.unique_kept == 0
        assert metrics.duplicates_removed == 0
        assert metrics.contaminated_count == 0

    def test_record_unique_item(self):
        """Test recording a unique item."""
        metrics = DedupMetrics()
        metrics.record_item(is_duplicate=False, index=0)

        assert metrics.total_seen == 1
        assert metrics.unique_kept == 1
        assert metrics.duplicates_removed == 0

    def test_record_duplicate_item(self):
        """Test recording a duplicate item."""
        metrics = DedupMetrics()
        metrics.record_item(is_duplicate=True, index=0)

        assert metrics.total_seen == 1
        assert metrics.unique_kept == 0
        assert metrics.duplicates_removed == 1

    def test_record_multiple_items(self):
        """Test recording multiple items with mixed duplicates."""
        metrics = DedupMetrics()
        metrics.record_item(is_duplicate=False, index=0)
        metrics.record_item(is_duplicate=False, index=1)
        metrics.record_item(is_duplicate=True, index=2)
        metrics.record_item(is_duplicate=False, index=3)
        metrics.record_item(is_duplicate=True, index=4)

        assert metrics.total_seen == 5
        assert metrics.unique_kept == 3
        assert metrics.duplicates_removed == 2

    def test_removed_indices_tracking(self):
        """Test that removed indices are tracked."""
        metrics = DedupMetrics()
        metrics.record_item(is_duplicate=False, index=0)
        metrics.record_item(is_duplicate=True, index=1)
        metrics.record_item(is_duplicate=True, index=3)

        assert metrics.removed_indices == [1, 3]

    def test_removed_keys_tracking(self):
        """Test that removed keys are tracked."""
        metrics = DedupMetrics()
        metrics.record_item(is_duplicate=True, index=0, key="key_0")
        metrics.record_item(is_duplicate=True, index=1, key="key_1")

        assert metrics.removed_keys == ["key_0", "key_1"]


class TestDedupMetricsRates:
    """Tests for computed rate properties."""

    def test_dedup_ratio_no_duplicates(self):
        """Test dedup ratio when all items are unique."""
        metrics = DedupMetrics()
        for i in range(5):
            metrics.record_item(is_duplicate=False, index=i)

        assert metrics.dedup_ratio == 1.0

    def test_dedup_ratio_with_duplicates(self):
        """Test dedup ratio with duplicates."""
        metrics = DedupMetrics()
        # 10 items, 5 unique
        for i in range(5):
            metrics.record_item(is_duplicate=False, index=i)
        for i in range(5, 10):
            metrics.record_item(is_duplicate=True, index=i)

        assert metrics.dedup_ratio == 2.0

    def test_dedup_ratio_empty(self):
        """Test dedup ratio with no items."""
        metrics = DedupMetrics()
        assert metrics.dedup_ratio == 1.0

    def test_removal_rate(self):
        """Test removal rate calculation."""
        metrics = DedupMetrics()
        # 4 items, 1 duplicate
        for i in range(3):
            metrics.record_item(is_duplicate=False, index=i)
        metrics.record_item(is_duplicate=True, index=3)

        assert metrics.removal_rate == 0.25
        assert metrics.keep_rate == 0.75

    def test_removal_rate_empty(self):
        """Test removal rate with no items."""
        metrics = DedupMetrics()
        assert metrics.removal_rate == 0.0
        assert metrics.keep_rate == 1.0


class TestDedupMetricsTiming:
    """Tests for timing functionality."""

    def test_start_finish(self):
        """Test start and finish timing."""
        metrics = DedupMetrics()
        metrics.start()
        time.sleep(0.01)  # Small delay
        metrics.finish()

        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.elapsed_seconds >= 0.01

    def test_elapsed_seconds_not_finished(self):
        """Test elapsed seconds when not finished returns current time."""
        metrics = DedupMetrics()
        metrics.start()
        time.sleep(0.01)

        # Should return elapsed time even without finish()
        assert metrics.elapsed_seconds >= 0.01

    def test_elapsed_seconds_not_started(self):
        """Test elapsed seconds when not started returns zero."""
        metrics = DedupMetrics()
        assert metrics.elapsed_seconds == 0.0

    def test_items_per_second(self):
        """Test throughput calculation."""
        metrics = DedupMetrics()
        metrics.start()

        # Record some items
        for i in range(100):
            metrics.record_item(is_duplicate=False, index=i)

        time.sleep(0.01)
        metrics.finish()

        # Should have processed items per second
        assert metrics.items_per_second > 0

    def test_items_per_second_no_time(self):
        """Test throughput with zero elapsed time."""
        metrics = DedupMetrics()
        assert metrics.items_per_second == 0.0


class TestDedupMetricsContamination:
    """Tests for contamination tracking."""

    def test_record_contamination(self):
        """Test recording contamination."""
        metrics = DedupMetrics()
        metrics.total_seen = 10
        metrics.record_contamination(index=3)
        metrics.record_contamination(index=7)

        assert metrics.contaminated_count == 2
        assert metrics.contaminated_indices == [3, 7]

    def test_contamination_rate(self):
        """Test contamination rate calculation."""
        metrics = DedupMetrics()
        metrics.total_seen = 100
        for i in range(10):
            metrics.record_contamination(index=i)

        assert metrics.contamination_rate == 0.10

    def test_contamination_rate_empty(self):
        """Test contamination rate with no items."""
        metrics = DedupMetrics()
        assert metrics.contamination_rate == 0.0


class TestDedupMetricsMerge:
    """Tests for merging metrics."""

    def test_merge_counts(self):
        """Test merging metrics from multiple sources."""
        metrics1 = DedupMetrics()
        metrics1.total_seen = 100
        metrics1.unique_kept = 80
        metrics1.duplicates_removed = 20

        metrics2 = DedupMetrics()
        metrics2.total_seen = 50
        metrics2.unique_kept = 45
        metrics2.duplicates_removed = 5

        metrics1.merge(metrics2)

        assert metrics1.total_seen == 150
        assert metrics1.unique_kept == 125
        assert metrics1.duplicates_removed == 25

    def test_merge_lists(self):
        """Test merging list fields."""
        metrics1 = DedupMetrics()
        metrics1.removed_indices = [1, 2, 3]
        metrics1.batch_times = [0.1, 0.2]

        metrics2 = DedupMetrics()
        metrics2.removed_indices = [4, 5]
        metrics2.batch_times = [0.3]

        metrics1.merge(metrics2)

        assert metrics1.removed_indices == [1, 2, 3, 4, 5]
        assert metrics1.batch_times == [0.1, 0.2, 0.3]


class TestDedupMetricsSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = DedupMetrics()
        metrics.total_seen = 100
        metrics.unique_kept = 80
        metrics.duplicates_removed = 20
        metrics.contaminated_count = 5
        metrics.storage_bytes = 1024

        d = metrics.to_dict()

        assert d["total_seen"] == 100
        assert d["unique_kept"] == 80
        assert d["duplicates_removed"] == 20
        assert d["contaminated_count"] == 5
        assert d["storage_bytes"] == 1024
        assert "dedup_ratio" in d
        assert "removal_rate" in d
        assert "keep_rate" in d

    def test_summary(self):
        """Test summary generation."""
        metrics = DedupMetrics()
        metrics.total_seen = 100
        metrics.unique_kept = 80
        metrics.duplicates_removed = 20

        summary = metrics.summary()

        assert "100" in summary
        assert "80" in summary
        assert "20" in summary
        assert "Deduplication Results" in summary

    def test_repr(self):
        """Test string representation."""
        metrics = DedupMetrics()
        metrics.total_seen = 100
        metrics.unique_kept = 50
        metrics.duplicates_removed = 50

        r = repr(metrics)

        assert "DedupMetrics" in r
        assert "100" in r
        assert "50" in r


class TestDedupMetricsStoreHealth:
    """Tests for store health integration."""

    def test_update_from_store_health(self):
        """Test updating metrics from store health stats."""
        metrics = DedupMetrics()

        health = {
            "total_bytes": 10240,
            "total_keys": 100,
            "total_objects": 80,
        }

        metrics.update_from_store_health(health)

        assert metrics.storage_bytes == 10240
        assert metrics.store_total_keys == 100
        assert metrics.store_unique_objects == 80

    def test_update_from_empty_health(self):
        """Test updating with empty health dict."""
        metrics = DedupMetrics()
        metrics.update_from_store_health({})

        assert metrics.storage_bytes == 0
        assert metrics.store_total_keys == 0
        assert metrics.store_unique_objects == 0
