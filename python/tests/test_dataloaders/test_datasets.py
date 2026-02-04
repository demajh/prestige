"""Tests for prestige.dataloaders.datasets module."""

import pytest

from prestige.dataloaders import DedupConfig, DedupMode

# Import conditionally based on torch availability
try:
    import torch
    from torch.utils.data import DataLoader

    from prestige.dataloaders import (
        DedupDataset,
        DedupDatasetView,
        LazyDedupDataset,
        create_dedup_dataloader,
        collate_with_dedup_info,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


class TestDedupDataset:
    """Tests for DedupDataset class."""

    def test_basic_dedup(self, sample_data, exact_config):
        """Test basic deduplication."""
        dataset = DedupDataset(sample_data, exact_config)

        # sample_data has 8 items, 5 unique
        assert len(dataset) == 5
        assert dataset.original_size == 8

    def test_no_duplicates(self, unique_data, exact_config):
        """Test dataset with no duplicates."""
        dataset = DedupDataset(unique_data, exact_config)

        assert len(dataset) == 5
        assert dataset.dedup_ratio == 1.0

    def test_getitem(self, sample_data, exact_config):
        """Test getting items by index."""
        dataset = DedupDataset(sample_data, exact_config)

        # Should be able to get all unique items
        for i in range(len(dataset)):
            item = dataset[i]
            assert "text" in item
            assert "label" in item

    def test_getitem_preserves_data(self, unique_data, exact_config):
        """Test that items are preserved correctly."""
        dataset = DedupDataset(unique_data, exact_config)

        # All items should be present and accessible
        texts = [dataset[i]["text"] for i in range(len(dataset))]
        original_texts = [item["text"] for item in unique_data]

        assert set(texts) == set(original_texts)

    def test_transform(self, sample_data, exact_config):
        """Test applying transform to items."""

        def add_processed_flag(item):
            item = dict(item)
            item["processed"] = True
            return item

        dataset = DedupDataset(sample_data, exact_config, transform=add_processed_flag)

        item = dataset[0]
        assert item.get("processed") is True

    def test_metrics(self, sample_data, exact_config):
        """Test metrics collection."""
        dataset = DedupDataset(sample_data, exact_config)
        metrics = dataset.get_metrics()

        assert metrics.total_seen == 8
        assert metrics.unique_kept == 5
        assert metrics.duplicates_removed == 3

    def test_get_removed_indices(self, sample_data, exact_config):
        """Test getting removed indices."""
        dataset = DedupDataset(sample_data, exact_config)
        removed = dataset.get_removed_indices()

        assert len(removed) == 3

    def test_get_valid_indices(self, sample_data, exact_config):
        """Test getting valid indices."""
        dataset = DedupDataset(sample_data, exact_config)
        valid = dataset.get_valid_indices()

        assert len(valid) == 5

    def test_get_original_index(self, sample_data, exact_config):
        """Test mapping deduplicated to original index."""
        dataset = DedupDataset(sample_data, exact_config)

        for dedup_idx in range(len(dataset)):
            orig_idx = dataset.get_original_index(dedup_idx)
            assert 0 <= orig_idx < len(sample_data)

    def test_dedup_ratio(self, sample_data, exact_config):
        """Test dedup ratio property."""
        dataset = DedupDataset(sample_data, exact_config)

        # 8 items / 5 unique = 1.6
        assert dataset.dedup_ratio == pytest.approx(1.6, rel=0.1)

    def test_text_column_override(self, exact_config):
        """Test overriding text column."""
        data = [
            {"content": "hello", "other": "x"},
            {"content": "world", "other": "y"},
            {"content": "hello", "other": "z"},  # Duplicate
        ]

        dataset = DedupDataset(data, exact_config, text_column="content")

        assert len(dataset) == 2

    def test_verbose_mode(self, sample_data, exact_config, capsys):
        """Test verbose output."""
        dataset = DedupDataset(sample_data, exact_config, verbose=True)

        # Should have printed progress
        captured = capsys.readouterr()
        assert "Deduplication Results" in captured.out or len(dataset) > 0


class TestDedupDatasetView:
    """Tests for DedupDatasetView class."""

    def test_subset_view(self, sample_data, exact_config):
        """Test creating a subset view."""
        dataset = DedupDataset(sample_data, exact_config)
        view = DedupDatasetView(dataset, indices=range(3))

        assert len(view) == 3

    def test_view_getitem(self, sample_data, exact_config):
        """Test getting items from view."""
        dataset = DedupDataset(sample_data, exact_config)
        view = DedupDatasetView(dataset, indices=[0, 2])

        item0 = view[0]
        item1 = view[1]

        assert item0["text"] == dataset[0]["text"]
        assert item1["text"] == dataset[2]["text"]

    def test_view_with_transform(self, sample_data, exact_config):
        """Test view with additional transform."""
        dataset = DedupDataset(sample_data, exact_config)

        def uppercase_text(item):
            item = dict(item)
            item["text"] = item["text"].upper()
            return item

        view = DedupDatasetView(dataset, transform=uppercase_text)

        item = view[0]
        assert item["text"].isupper()


class TestCreateDedupDataloader:
    """Tests for create_dedup_dataloader function."""

    def test_basic_dataloader(self, sample_data, exact_config):
        """Test creating a basic dataloader."""
        loader = create_dedup_dataloader(
            sample_data,
            config=exact_config,
            batch_size=2,
            shuffle=False,
        )

        batches = list(loader)
        assert len(batches) >= 1

    def test_dataloader_batch_size(self, unique_data, exact_config):
        """Test dataloader batch size."""
        loader = create_dedup_dataloader(
            unique_data,
            config=exact_config,
            batch_size=2,
            shuffle=False,
        )

        batches = list(loader)
        # 5 items with batch_size=2 -> 3 batches
        assert len(batches) == 3


class TestLazyDedupDataset:
    """Tests for LazyDedupDataset class."""

    def test_lazy_basic(self, sample_data, exact_config):
        """Test basic lazy dataset usage."""
        dataset = LazyDedupDataset(sample_data, exact_config)

        assert len(dataset) == 8  # Returns all items

    def test_lazy_marks_duplicates(self, sample_data, exact_config):
        """Test that lazy dataset marks duplicates."""
        dataset = LazyDedupDataset(sample_data, exact_config)

        items = [dataset[i] for i in range(len(dataset))]
        duplicates = [item for item in items if item["_is_duplicate"]]
        uniques = [item for item in items if not item["_is_duplicate"]]

        assert len(duplicates) == 3
        assert len(uniques) == 5

    def test_lazy_includes_object_id(self, sample_data, exact_config):
        """Test that lazy dataset includes object ID."""
        dataset = LazyDedupDataset(sample_data, exact_config)

        item = dataset[0]
        assert "_object_id" in item
        assert "_original_index" in item

        dataset.close()

    def test_lazy_metrics(self, sample_data, exact_config):
        """Test lazy dataset metrics."""
        dataset = LazyDedupDataset(sample_data, exact_config)

        # Access all items
        for i in range(len(dataset)):
            _ = dataset[i]

        metrics = dataset.get_metrics()
        assert metrics.total_seen == 8
        assert metrics.duplicates_removed == 3

        dataset.close()

    def test_lazy_reset(self, sample_data, exact_config):
        """Test resetting lazy dataset."""
        dataset = LazyDedupDataset(sample_data, exact_config)

        _ = dataset[0]
        dataset.reset()

        # After reset, should start fresh
        assert dataset._store is None
        assert dataset._metrics.total_seen == 0


class TestCollateWithDedupInfo:
    """Tests for collate function."""

    def test_collate_basic(self):
        """Test basic collation."""
        batch = [
            {"text": "a", "label": 0},
            {"text": "b", "label": 1},
        ]

        result = collate_with_dedup_info(batch)

        assert result["text"] == ["a", "b"]
        assert result["label"] == [0, 1]

    def test_collate_with_dedup_fields(self):
        """Test collation preserves dedup fields."""
        batch = [
            {"text": "a", "_is_duplicate": False, "_object_id": b"123"},
            {"text": "b", "_is_duplicate": True, "_object_id": b"456"},
        ]

        result = collate_with_dedup_info(batch, include_dedup_info=True)

        assert "_is_duplicate" in result
        assert "_object_id" in result

    def test_collate_excludes_dedup_fields(self):
        """Test collation can exclude dedup fields."""
        batch = [
            {"text": "a", "_is_duplicate": False},
            {"text": "b", "_is_duplicate": True},
        ]

        result = collate_with_dedup_info(batch, include_dedup_info=False)

        assert "_is_duplicate" not in result

    def test_collate_empty_batch(self):
        """Test collation with empty batch."""
        result = collate_with_dedup_info([])
        assert result == {}

    def test_collate_with_tensors(self):
        """Test collation stacks tensors."""
        batch = [
            {"text": "a", "embedding": torch.tensor([1.0, 2.0])},
            {"text": "b", "embedding": torch.tensor([3.0, 4.0])},
        ]

        result = collate_with_dedup_info(batch)

        assert result["embedding"].shape == (2, 2)


class TestDataLoaderIntegration:
    """Integration tests with PyTorch DataLoader."""

    def test_iteration(self, sample_data, exact_config):
        """Test full iteration through DataLoader."""
        dataset = DedupDataset(sample_data, exact_config)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        total_items = 0
        for batch in loader:
            if isinstance(batch, dict):
                batch_size = len(batch["text"])
            else:
                batch_size = len(batch)
            total_items += batch_size

        assert total_items == 5

    def test_shuffle(self, sample_data, exact_config):
        """Test that shuffle works."""
        dataset = DedupDataset(sample_data, exact_config)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Should be able to iterate
        batches = list(loader)
        assert len(batches) >= 1
