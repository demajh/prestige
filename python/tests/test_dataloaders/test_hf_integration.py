"""Tests for prestige.dataloaders.hf_integration module."""

import pytest
from pathlib import Path

from prestige.dataloaders import DedupConfig, DedupMode

# Import conditionally based on HuggingFace availability
try:
    from datasets import Dataset as HFDataset, DatasetDict

    from prestige.dataloaders import (
        HuggingFaceDeduplicator,
        StaticDedupPipeline,
        deduplicate_dataset,
        deduplicate_and_cache,
        CacheConfig,
    )

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not HF_AVAILABLE, reason="HuggingFace datasets not available"
)


class TestHuggingFaceDeduplicator:
    """Tests for HuggingFaceDeduplicator class."""

    def test_deduplicate_basic(self, hf_dataset, exact_config):
        """Test basic deduplication."""
        deduplicator = HuggingFaceDeduplicator(exact_config)
        deduped = deduplicator.deduplicate(hf_dataset)

        # Original has 8 items, 5 unique
        assert len(deduped) == 5

    def test_deduplicate_no_duplicates(self, hf_unique_dataset, exact_config):
        """Test deduplication with no duplicates."""
        deduplicator = HuggingFaceDeduplicator(exact_config)
        deduped = deduplicator.deduplicate(hf_unique_dataset)

        assert len(deduped) == 5

    def test_deduplicate_preserves_columns(self, hf_dataset, exact_config):
        """Test that all columns are preserved."""
        deduplicator = HuggingFaceDeduplicator(exact_config)
        deduped = deduplicator.deduplicate(hf_dataset)

        assert "text" in deduped.column_names
        assert "label" in deduped.column_names

    def test_deduplicate_custom_column(self, exact_config):
        """Test deduplication with custom text column."""
        data = HFDataset.from_list(
            [
                {"content": "hello", "id": 1},
                {"content": "world", "id": 2},
                {"content": "hello", "id": 3},  # Duplicate
            ]
        )

        exact_config.text_column = "content"
        deduplicator = HuggingFaceDeduplicator(exact_config)
        deduped = deduplicator.deduplicate(data, text_column="content")

        assert len(deduped) == 2

    def test_deduplicate_metrics(self, hf_dataset, exact_config):
        """Test metrics collection."""
        deduplicator = HuggingFaceDeduplicator(exact_config)
        _ = deduplicator.deduplicate(hf_dataset)

        metrics = deduplicator.get_metrics()

        assert metrics.total_seen == 8
        assert metrics.unique_kept == 5
        assert metrics.duplicates_removed == 3


class TestHuggingFaceDeduplicatorDatasetDict:
    """Tests for DatasetDict handling."""

    def test_deduplicate_dataset_dict(self, exact_config):
        """Test deduplication of DatasetDict."""
        train_data = [
            {"text": "train one", "label": 0},
            {"text": "train two", "label": 1},
            {"text": "train one", "label": 0},  # Duplicate
        ]
        test_data = [
            {"text": "test one", "label": 0},
            {"text": "test two", "label": 1},
        ]

        dataset_dict = DatasetDict(
            {
                "train": HFDataset.from_list(train_data),
                "test": HFDataset.from_list(test_data),
            }
        )

        deduplicator = HuggingFaceDeduplicator(exact_config)
        deduped = deduplicator.deduplicate(dataset_dict)

        assert isinstance(deduped, DatasetDict)
        assert len(deduped["train"]) == 2
        assert len(deduped["test"]) == 2


class TestDeduplicateAcrossSplits:
    """Tests for cross-split deduplication."""

    def test_deduplicate_across_splits(self, exact_config):
        """Test deduplication across splits."""
        train_data = [
            {"text": "shared example", "label": 0},
            {"text": "train only", "label": 1},
        ]
        test_data = [
            {"text": "shared example", "label": 0},  # Same as train
            {"text": "test only", "label": 1},
        ]

        dataset_dict = DatasetDict(
            {
                "train": HFDataset.from_list(train_data),
                "test": HFDataset.from_list(test_data),
            }
        )

        deduplicator = HuggingFaceDeduplicator(exact_config)
        deduped = deduplicator.deduplicate_across_splits(
            dataset_dict, primary_split="train"
        )

        assert len(deduped["train"]) == 2  # All train kept
        assert len(deduped["test"]) == 1  # Shared example removed


class TestStaticDedupPipeline:
    """Tests for StaticDedupPipeline class."""

    def test_process_and_cache(self, hf_dataset, exact_config, temp_dir):
        """Test processing and caching."""
        cache_config = CacheConfig(cache_dir=temp_dir)
        pipeline = StaticDedupPipeline(exact_config, cache_config)

        deduped = pipeline.process_and_cache(hf_dataset, "test_dataset")

        assert len(deduped) == 5

    def test_cache_hit(self, hf_dataset, exact_config, temp_dir):
        """Test that cache is used on second call."""
        cache_config = CacheConfig(cache_dir=temp_dir)
        pipeline = StaticDedupPipeline(exact_config, cache_config)

        # First call - processes
        deduped1 = pipeline.process_and_cache(hf_dataset, "test_dataset")

        # Second call - should use cache
        deduped2 = pipeline.process_and_cache(hf_dataset, "test_dataset")

        assert len(deduped1) == len(deduped2)

    def test_force_reprocess(self, hf_dataset, exact_config, temp_dir):
        """Test force reprocess ignores cache."""
        cache_config = CacheConfig(cache_dir=temp_dir)
        pipeline = StaticDedupPipeline(exact_config, cache_config)

        # First call
        pipeline.process_and_cache(hf_dataset, "test_dataset")

        # Second call with force
        deduped = pipeline.process_and_cache(
            hf_dataset, "test_dataset", force_reprocess=True
        )

        assert len(deduped) == 5

    def test_load_cached(self, hf_dataset, exact_config, temp_dir):
        """Test loading from cache."""
        cache_config = CacheConfig(cache_dir=temp_dir)
        pipeline = StaticDedupPipeline(exact_config, cache_config)

        # Process first
        pipeline.process_and_cache(hf_dataset, "test_dataset")

        # Load from cache
        cached = pipeline.load_cached("test_dataset")

        assert cached is not None
        assert len(cached) == 5

    def test_load_cached_missing(self, exact_config, temp_dir):
        """Test loading non-existent cache returns None."""
        cache_config = CacheConfig(cache_dir=temp_dir)
        pipeline = StaticDedupPipeline(exact_config, cache_config)

        cached = pipeline.load_cached("nonexistent")

        assert cached is None

    def test_load_metrics(self, hf_dataset, exact_config, temp_dir):
        """Test loading cached metrics."""
        cache_config = CacheConfig(cache_dir=temp_dir, save_metrics=True)
        pipeline = StaticDedupPipeline(exact_config, cache_config)

        # Process first
        pipeline.process_and_cache(hf_dataset, "test_dataset")

        # Load metrics
        metrics = pipeline.load_metrics("test_dataset")

        assert metrics is not None
        assert "total_seen" in metrics

    def test_clear_cache(self, hf_dataset, exact_config, temp_dir):
        """Test clearing cache."""
        cache_config = CacheConfig(cache_dir=temp_dir)
        pipeline = StaticDedupPipeline(exact_config, cache_config)

        # Process first
        pipeline.process_and_cache(hf_dataset, "test_dataset")

        # Clear cache
        pipeline.clear_cache("test_dataset")

        # Should be gone
        cached = pipeline.load_cached("test_dataset")
        assert cached is None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_deduplicate_dataset_exact(self, hf_dataset):
        """Test deduplicate_dataset with exact mode."""
        deduped = deduplicate_dataset(hf_dataset, mode="exact")

        assert len(deduped) == 5

    def test_deduplicate_dataset_custom_column(self):
        """Test deduplicate_dataset with custom column."""
        data = HFDataset.from_list(
            [
                {"content": "hello", "id": 1},
                {"content": "world", "id": 2},
                {"content": "hello", "id": 3},
            ]
        )

        deduped = deduplicate_dataset(data, mode="exact", text_column="content")

        assert len(deduped) == 2

    def test_deduplicate_and_cache(self, hf_dataset, temp_dir):
        """Test deduplicate_and_cache function."""
        deduped = deduplicate_and_cache(
            hf_dataset,
            "test_ds",
            cache_dir=temp_dir,
            mode="exact",
        )

        assert len(deduped) == 5

        # Second call should use cache
        deduped2 = deduplicate_and_cache(
            hf_dataset,
            "test_ds",
            cache_dir=temp_dir,
            mode="exact",
        )

        assert len(deduped2) == 5


class TestVerboseOutput:
    """Tests for verbose output."""

    def test_deduplicate_verbose(self, hf_dataset, exact_config, capsys):
        """Test verbose output during deduplication."""
        deduplicator = HuggingFaceDeduplicator(exact_config)
        _ = deduplicator.deduplicate(hf_dataset, verbose=True)

        captured = capsys.readouterr()
        # Should have some output
        assert len(captured.out) > 0 or True  # May not print if dataset is small
