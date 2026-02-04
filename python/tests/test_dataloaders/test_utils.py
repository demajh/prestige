"""Tests for prestige.dataloaders.utils module."""

import pytest

from prestige.dataloaders import DedupConfig, DedupMode
from prestige.dataloaders.utils import (
    extract_text,
    make_key,
    hash_text,
    hash_config,
    batched,
    create_text_extractor,
    estimate_memory_usage,
    validate_text_column,
)


class TestExtractText:
    """Tests for extract_text function."""

    def test_extract_from_dict(self):
        """Test extracting text from dictionary."""
        item = {"text": "hello world", "label": 0}
        result = extract_text(item, "text")
        assert result == "hello world"

    def test_extract_from_dict_custom_column(self):
        """Test extracting from custom column."""
        item = {"content": "hello", "text": "other"}
        result = extract_text(item, "content")
        assert result == "hello"

    def test_extract_missing_column(self):
        """Test extracting missing column returns empty string."""
        item = {"other": "value"}
        result = extract_text(item, "text")
        assert result == ""

    def test_extract_from_string(self):
        """Test extracting from string directly."""
        result = extract_text("hello world", "text")
        assert result == "hello world"

    def test_extract_multiple_columns(self):
        """Test extracting from multiple columns."""
        item = {"q1": "hello", "q2": "world"}
        result = extract_text(item, text_columns=["q1", "q2"])
        assert result == "hello world"

    def test_extract_multiple_columns_custom_separator(self):
        """Test extracting with custom separator."""
        item = {"a": "hello", "b": "world"}
        result = extract_text(item, text_columns=["a", "b"], separator=" | ")
        assert result == "hello | world"

    def test_extract_none_value(self):
        """Test extracting None value returns empty string."""
        item = {"text": None}
        result = extract_text(item, "text")
        assert result == ""


class TestMakeKey:
    """Tests for make_key function."""

    def test_basic_key(self):
        """Test basic key generation."""
        key = make_key(42)
        assert key == "item_42"

    def test_key_with_prefix(self):
        """Test key with prefix."""
        key = make_key(0, prefix="train_")
        assert key == "train__item_0"

    def test_key_with_namespace(self):
        """Test key with namespace."""
        key = make_key(0, namespace="dataset1")
        assert key == "dataset1_item_0"

    def test_key_with_both(self):
        """Test key with prefix and namespace."""
        key = make_key(0, prefix="batch_", namespace="ds1")
        assert key == "ds1_batch__item_0"


class TestHashText:
    """Tests for hash_text function."""

    def test_hash_returns_string(self):
        """Test hash returns string."""
        result = hash_text("hello world")
        assert isinstance(result, str)

    def test_hash_length(self):
        """Test hash is 16 characters."""
        result = hash_text("test")
        assert len(result) == 16

    def test_hash_deterministic(self):
        """Test hash is deterministic."""
        h1 = hash_text("hello")
        h2 = hash_text("hello")
        assert h1 == h2

    def test_hash_different_inputs(self):
        """Test different inputs produce different hashes."""
        h1 = hash_text("hello")
        h2 = hash_text("world")
        assert h1 != h2


class TestHashConfig:
    """Tests for hash_config function."""

    def test_hash_config_returns_string(self):
        """Test hash_config returns string."""
        config = DedupConfig()
        result = hash_config(config)
        assert isinstance(result, str)

    def test_hash_config_length(self):
        """Test hash is 8 characters."""
        config = DedupConfig()
        result = hash_config(config)
        assert len(result) == 8

    def test_hash_config_deterministic(self):
        """Test hash is deterministic for same config."""
        config1 = DedupConfig(mode=DedupMode.SEMANTIC, semantic_threshold=0.9)
        config2 = DedupConfig(mode=DedupMode.SEMANTIC, semantic_threshold=0.9)
        assert hash_config(config1) == hash_config(config2)

    def test_hash_config_different(self):
        """Test different configs produce different hashes."""
        config1 = DedupConfig(mode=DedupMode.EXACT)
        config2 = DedupConfig(mode=DedupMode.SEMANTIC)
        assert hash_config(config1) != hash_config(config2)


class TestBatched:
    """Tests for batched function."""

    def test_basic_batching(self):
        """Test basic batching."""
        items = [1, 2, 3, 4, 5]
        batches = list(batched(items, 2))

        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5]

    def test_exact_batch_size(self):
        """Test when items divide evenly."""
        items = [1, 2, 3, 4]
        batches = list(batched(items, 2))

        assert len(batches) == 2
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]

    def test_single_batch(self):
        """Test when all items fit in one batch."""
        items = [1, 2, 3]
        batches = list(batched(items, 10))

        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_empty_input(self):
        """Test empty input."""
        batches = list(batched([], 2))
        assert len(batches) == 0


class TestCreateTextExtractor:
    """Tests for create_text_extractor function."""

    def test_basic_extractor(self):
        """Test basic extractor creation."""
        extractor = create_text_extractor("content")
        item = {"content": "hello", "other": "world"}

        result = extractor(item)
        assert result == "hello"

    def test_extractor_with_preprocess(self):
        """Test extractor with preprocessing."""
        extractor = create_text_extractor("text", preprocess=str.upper)
        item = {"text": "hello"}

        result = extractor(item)
        assert result == "HELLO"

    def test_extractor_multiple_columns(self):
        """Test extractor with multiple columns."""
        extractor = create_text_extractor(text_columns=["a", "b"])
        item = {"a": "hello", "b": "world"}

        result = extractor(item)
        assert result == "hello world"


class TestEstimateMemoryUsage:
    """Tests for estimate_memory_usage function."""

    def test_basic_estimate(self):
        """Test basic memory estimation."""
        estimate = estimate_memory_usage(1000)

        assert "total_bytes" in estimate
        assert "total_mb" in estimate
        assert estimate["total_bytes"] > 0

    def test_exact_mode_estimate(self):
        """Test exact mode estimation."""
        estimate = estimate_memory_usage(1000, mode="exact")

        assert estimate["hash_bytes"] > 0
        assert estimate["embedding_bytes"] == 0

    def test_semantic_mode_estimate(self):
        """Test semantic mode estimation."""
        estimate = estimate_memory_usage(1000, mode="semantic")

        assert estimate["hash_bytes"] == 0
        assert estimate["embedding_bytes"] > 0
        assert estimate["hnsw_bytes"] > 0

    def test_custom_embedding_dim(self):
        """Test custom embedding dimension."""
        estimate_small = estimate_memory_usage(1000, mode="semantic", embedding_dim=384)
        estimate_large = estimate_memory_usage(1000, mode="semantic", embedding_dim=1024)

        assert estimate_large["embedding_bytes"] > estimate_small["embedding_bytes"]


class TestValidateTextColumn:
    """Tests for validate_text_column function."""

    def test_valid_column_dict(self):
        """Test validation with valid column in dict."""
        item = {"text": "hello", "label": 0}
        result = validate_text_column(item, "text")
        assert result is True

    def test_invalid_column_dict(self):
        """Test validation with invalid column in dict."""
        item = {"content": "hello"}
        with pytest.raises(ValueError, match="text"):
            validate_text_column(item, "text")

    def test_valid_multiple_columns(self):
        """Test validation with multiple columns."""
        item = {"q1": "hello", "q2": "world"}
        result = validate_text_column(item, "text", text_columns=["q1", "q2"])
        assert result is True

    def test_invalid_multiple_columns(self):
        """Test validation with missing column in list."""
        item = {"q1": "hello"}
        with pytest.raises(ValueError, match="q2"):
            validate_text_column(item, "text", text_columns=["q1", "q2"])

    def test_shows_available_columns(self):
        """Test error message shows available columns."""
        item = {"content": "hello", "label": 0}
        with pytest.raises(ValueError) as exc_info:
            validate_text_column(item, "text")

        assert "content" in str(exc_info.value)
        assert "label" in str(exc_info.value)
