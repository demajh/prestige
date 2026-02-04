"""Tests for prestige.dataloaders.config module."""

import pytest
from pathlib import Path

from prestige.dataloaders import (
    DedupConfig,
    DedupMode,
    DedupStrategy,
    CrossDatasetConfig,
    CacheConfig,
)


class TestDedupMode:
    """Tests for DedupMode enum."""

    def test_exact_mode_value(self):
        """Test EXACT mode has correct value."""
        assert DedupMode.EXACT.value == "exact"

    def test_semantic_mode_value(self):
        """Test SEMANTIC mode has correct value."""
        assert DedupMode.SEMANTIC.value == "semantic"


class TestDedupStrategy:
    """Tests for DedupStrategy enum."""

    def test_static_strategy_value(self):
        """Test STATIC strategy has correct value."""
        assert DedupStrategy.STATIC.value == "static"

    def test_dynamic_strategy_value(self):
        """Test DYNAMIC strategy has correct value."""
        assert DedupStrategy.DYNAMIC.value == "dynamic"


class TestDedupConfig:
    """Tests for DedupConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DedupConfig()
        assert config.mode == DedupMode.EXACT
        assert config.strategy == DedupStrategy.DYNAMIC
        assert config.store_path is None
        assert config.persist_store is False
        assert config.semantic_threshold == 0.85
        assert config.text_column == "text"
        assert config.batch_size == 100
        assert config.key_prefix == ""
        assert config.keep_first is True

    def test_semantic_config(self):
        """Test semantic mode configuration."""
        config = DedupConfig(
            mode=DedupMode.SEMANTIC,
            semantic_threshold=0.9,
            semantic_model_type="bge-small",
        )
        assert config.mode == DedupMode.SEMANTIC
        assert config.semantic_threshold == 0.9
        assert config.semantic_model_type == "bge-small"

    def test_custom_text_column(self):
        """Test custom text column configuration."""
        config = DedupConfig(text_column="content")
        assert config.text_column == "content"

    def test_store_path_as_path_object(self):
        """Test store_path accepts Path objects."""
        config = DedupConfig(store_path=Path("/tmp/test"))
        assert config.store_path == Path("/tmp/test")

    def test_invalid_threshold_high(self):
        """Test validation rejects threshold > 1.0."""
        with pytest.raises(ValueError, match="semantic_threshold"):
            DedupConfig(semantic_threshold=1.5)

    def test_invalid_threshold_low(self):
        """Test validation rejects threshold < 0.0."""
        with pytest.raises(ValueError, match="semantic_threshold"):
            DedupConfig(semantic_threshold=-0.1)

    def test_invalid_batch_size(self):
        """Test validation rejects batch_size < 1."""
        with pytest.raises(ValueError, match="batch_size"):
            DedupConfig(batch_size=0)

    def test_reranker_config(self):
        """Test reranker configuration options."""
        config = DedupConfig(
            enable_reranker=True,
            reranker_threshold=0.8,
            reranker_top_k=50,
        )
        assert config.enable_reranker is True
        assert config.reranker_threshold == 0.8
        assert config.reranker_top_k == 50

    def test_rnn_config(self):
        """Test RNN configuration options."""
        config = DedupConfig(enable_rnn=True, rnn_k=10)
        assert config.enable_rnn is True
        assert config.rnn_k == 10

    def test_margin_gating_config(self):
        """Test margin gating configuration."""
        config = DedupConfig(enable_margin_gating=True, margin_threshold=0.1)
        assert config.enable_margin_gating is True
        assert config.margin_threshold == 0.1


class TestCrossDatasetConfig:
    """Tests for CrossDatasetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CrossDatasetConfig()
        assert config.reference_store_path is None
        assert config.build_reference is True
        assert config.detect_contamination is True
        assert config.contamination_threshold == 0.95

    def test_custom_threshold(self):
        """Test custom contamination threshold."""
        config = CrossDatasetConfig(contamination_threshold=0.99)
        assert config.contamination_threshold == 0.99

    def test_reference_store_path(self):
        """Test reference store path configuration."""
        config = CrossDatasetConfig(reference_store_path=Path("/tmp/ref"))
        assert config.reference_store_path == Path("/tmp/ref")


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.cache_key is None
        assert config.force_reprocess is False
        assert config.save_metrics is True
        # Default cache_dir is set in __post_init__
        assert config.cache_dir is not None
        assert "prestige" in str(config.cache_dir)

    def test_custom_cache_dir(self):
        """Test custom cache directory."""
        config = CacheConfig(cache_dir=Path("/custom/cache"))
        assert config.cache_dir == Path("/custom/cache")

    def test_custom_cache_key(self):
        """Test custom cache key."""
        config = CacheConfig(cache_key="my_dataset_v1")
        assert config.cache_key == "my_dataset_v1"

    def test_force_reprocess(self):
        """Test force reprocess flag."""
        config = CacheConfig(force_reprocess=True)
        assert config.force_reprocess is True
