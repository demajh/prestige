"""Tests for prestige.dataloaders.streaming module."""

import pytest

from prestige.dataloaders import DedupConfig, DedupMode

# Import conditionally based on torch availability
try:
    import torch
    from torch.utils.data import DataLoader

    from prestige.dataloaders import (
        StreamingDedupDataset,
        DynamicDedupIterator,
        ChunkedDedupProcessor,
        create_streaming_dataloader,
        deduplicate_iterator,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDynamicDedupIterator:
    """Tests for DynamicDedupIterator class."""

    def test_basic_iteration(self, sample_data, exact_config):
        """Test basic iteration with deduplication."""
        iterator = DynamicDedupIterator(iter(sample_data), exact_config)

        items = list(iterator)

        # sample_data has 8 items, 5 unique
        assert len(items) == 5

    def test_preserves_data(self, unique_data, exact_config):
        """Test that data is preserved."""
        iterator = DynamicDedupIterator(iter(unique_data), exact_config)

        items = list(iterator)

        assert len(items) == 5
        texts = {item["text"] for item in items}
        original_texts = {item["text"] for item in unique_data}
        assert texts == original_texts

    def test_metrics(self, sample_data, exact_config):
        """Test metrics collection."""
        iterator = DynamicDedupIterator(iter(sample_data), exact_config)

        _ = list(iterator)
        metrics = iterator.get_metrics()

        assert metrics.total_seen == 8
        assert metrics.unique_kept == 5
        assert metrics.duplicates_removed == 3


class TestDeduplicateIterator:
    """Tests for deduplicate_iterator function."""

    def test_basic_usage(self, sample_data):
        """Test basic usage."""
        items = list(deduplicate_iterator(iter(sample_data), mode="exact"))

        assert len(items) == 5

    def test_custom_threshold(self, sample_data):
        """Test with custom threshold."""
        items = list(
            deduplicate_iterator(iter(sample_data), mode="exact", threshold=0.9)
        )

        assert len(items) == 5

    def test_custom_text_column(self):
        """Test with custom text column."""
        data = [
            {"content": "hello", "id": 1},
            {"content": "world", "id": 2},
            {"content": "hello", "id": 3},  # Duplicate
        ]

        items = list(deduplicate_iterator(iter(data), text_column="content"))

        assert len(items) == 2


class TestChunkedDedupProcessor:
    """Tests for ChunkedDedupProcessor class."""

    def test_basic_chunking(self, sample_data, exact_config):
        """Test basic chunked processing."""
        processor = ChunkedDedupProcessor(exact_config, chunk_size=3)

        chunks = list(processor.process_chunks(iter(sample_data)))

        # 5 unique items with chunk_size=3 -> 2 chunks
        assert len(chunks) == 2
        total_items = sum(len(chunk) for chunk in chunks)
        assert total_items == 5

    def test_single_chunk(self, sample_data, exact_config):
        """Test when all items fit in one chunk."""
        processor = ChunkedDedupProcessor(exact_config, chunk_size=10)

        chunks = list(processor.process_chunks(iter(sample_data)))

        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_metrics(self, sample_data, exact_config):
        """Test metrics collection."""
        processor = ChunkedDedupProcessor(exact_config, chunk_size=3)

        _ = list(processor.process_chunks(iter(sample_data)))
        metrics = processor.get_metrics()

        assert metrics.total_seen == 8
        assert metrics.duplicates_removed == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestStreamingDedupDataset:
    """Tests for StreamingDedupDataset class."""

    def test_basic_iteration(self, sample_data, exact_config):
        """Test basic iteration."""

        def data_generator():
            return iter(sample_data)

        dataset = StreamingDedupDataset(data_generator, exact_config)

        items = list(dataset)

        assert len(items) == 5

    def test_with_dataloader(self, sample_data, exact_config):
        """Test with PyTorch DataLoader."""

        def data_generator():
            return iter(sample_data)

        dataset = StreamingDedupDataset(data_generator, exact_config)
        loader = DataLoader(dataset, batch_size=2)

        all_items = []
        for batch in loader:
            # Batch is a list or dict depending on data format
            if isinstance(batch, dict):
                batch_size = len(batch["text"])
            else:
                batch_size = len(batch)
            all_items.extend(range(batch_size))

        assert len(all_items) == 5

    def test_transform(self, sample_data, exact_config):
        """Test with transform."""

        def data_generator():
            return iter(sample_data)

        def uppercase_text(item):
            item = dict(item)
            item["text"] = item["text"].upper()
            return item

        dataset = StreamingDedupDataset(
            data_generator, exact_config, transform=uppercase_text
        )

        items = list(dataset)

        assert all(item["text"].isupper() for item in items)

    def test_metrics(self, sample_data, exact_config):
        """Test metrics collection."""

        def data_generator():
            return iter(sample_data)

        dataset = StreamingDedupDataset(data_generator, exact_config)

        _ = list(dataset)
        metrics = dataset.get_metrics()

        assert metrics.total_seen == 8


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCreateStreamingDataloader:
    """Tests for create_streaming_dataloader function."""

    def test_from_iterator_factory(self, sample_data, exact_config):
        """Test creating dataloader from iterator factory."""

        def data_generator():
            return iter(sample_data)

        loader = create_streaming_dataloader(
            data_generator, config=exact_config, batch_size=2
        )

        batches = list(loader)
        assert len(batches) >= 1

    def test_invalid_source_raises(self, exact_config):
        """Test that invalid source raises TypeError."""
        with pytest.raises(TypeError):
            create_streaming_dataloader(
                "not a callable or dataset",  # Invalid
                config=exact_config,
            )


# HuggingFace streaming tests
try:
    from datasets import IterableDataset as HFIterableDataset

    HF_ITERABLE_AVAILABLE = True
except ImportError:
    HF_ITERABLE_AVAILABLE = False


@pytest.mark.skipif(
    not HF_ITERABLE_AVAILABLE or not TORCH_AVAILABLE,
    reason="HuggingFace IterableDataset or PyTorch not available",
)
class TestHuggingFaceStreaming:
    """Tests for HuggingFace streaming dataset integration."""

    def test_from_hf_iterable(self, sample_data, exact_config):
        """Test with HuggingFace IterableDataset."""

        def gen():
            for item in sample_data:
                yield item

        hf_dataset = HFIterableDataset.from_generator(gen)

        loader = create_streaming_dataloader(
            hf_dataset, config=exact_config, batch_size=2
        )

        batches = list(loader)
        assert len(batches) >= 1


class TestStreamingEdgeCases:
    """Tests for edge cases in streaming."""

    def test_empty_iterator(self, exact_config):
        """Test with empty iterator."""
        iterator = DynamicDedupIterator(iter([]), exact_config)

        items = list(iterator)

        assert len(items) == 0

    def test_single_item(self, exact_config):
        """Test with single item."""
        data = [{"text": "only one", "label": 0}]
        iterator = DynamicDedupIterator(iter(data), exact_config)

        items = list(iterator)

        assert len(items) == 1

    def test_all_duplicates(self, exact_config):
        """Test with all duplicates."""
        data = [{"text": "same", "label": i} for i in range(5)]
        iterator = DynamicDedupIterator(iter(data), exact_config)

        items = list(iterator)

        assert len(items) == 1

    def test_chunked_empty(self, exact_config):
        """Test chunked processor with empty input."""
        processor = ChunkedDedupProcessor(exact_config, chunk_size=3)

        chunks = list(processor.process_chunks(iter([])))

        assert len(chunks) == 0
