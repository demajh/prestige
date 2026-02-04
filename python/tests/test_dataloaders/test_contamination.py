"""Tests for prestige.dataloaders.contamination module."""

import pytest

from prestige.dataloaders import (
    DedupConfig,
    DedupMode,
    ContaminationDetector,
    CrossDatasetDeduplicator,
    CrossDatasetConfig,
    detect_train_test_leakage,
    filter_train_test_leakage,
)


class TestContaminationDetector:
    """Tests for ContaminationDetector class."""

    def test_build_reference_index(self, test_data, exact_config, temp_dir):
        """Test building reference index."""
        cross_config = CrossDatasetConfig(reference_store_path=temp_dir / "ref")

        detector = ContaminationDetector(exact_config, cross_config)
        count = detector.build_reference_index(test_data, "test")

        assert count == 3
        assert detector.reference_count == 3

        detector.close()

    def test_check_contamination_exact(self, train_data, test_data, exact_config, temp_dir):
        """Test exact contamination detection."""
        cross_config = CrossDatasetConfig(reference_store_path=temp_dir / "ref")

        detector = ContaminationDetector(exact_config, cross_config)
        detector.build_reference_index(test_data, "test")

        contaminated, metrics = detector.check_contamination(train_data)

        # train_data[3] is exact match with test_data[1]
        assert len(contaminated) == 1
        assert 3 in contaminated

        detector.close()

    def test_check_contamination_none(self, unique_data, exact_config, temp_dir):
        """Test no contamination detected."""
        cross_config = CrossDatasetConfig(reference_store_path=temp_dir / "ref")

        detector = ContaminationDetector(exact_config, cross_config)
        detector.build_reference_index(unique_data[:2], "ref")

        contaminated, _ = detector.check_contamination(unique_data[2:])

        assert len(contaminated) == 0

        detector.close()

    def test_check_contamination_without_reference_raises(self, train_data, exact_config):
        """Test that checking without reference raises error."""
        detector = ContaminationDetector(exact_config)

        with pytest.raises(RuntimeError, match="build_reference_index"):
            detector.check_contamination(train_data)

    def test_filter_contaminated(self, train_data, test_data, exact_config, temp_dir):
        """Test filtering contaminated samples."""
        cross_config = CrossDatasetConfig(reference_store_path=temp_dir / "ref")

        detector = ContaminationDetector(exact_config, cross_config)
        detector.build_reference_index(test_data, "test")

        filtered = detector.filter_contaminated(train_data)

        # Original had 5, 1 contaminated
        assert len(filtered) == 4

        detector.close()

    def test_metrics(self, train_data, test_data, exact_config, temp_dir):
        """Test metrics collection."""
        cross_config = CrossDatasetConfig(reference_store_path=temp_dir / "ref")

        detector = ContaminationDetector(exact_config, cross_config)
        detector.build_reference_index(test_data, "test")
        _, metrics = detector.check_contamination(train_data)

        assert metrics.total_seen == 5
        assert metrics.contaminated_count == 1
        assert metrics.contamination_rate == 0.2

        detector.close()

    def test_context_manager(self, test_data, exact_config, temp_dir):
        """Test context manager protocol."""
        cross_config = CrossDatasetConfig(reference_store_path=temp_dir / "ref")

        with ContaminationDetector(exact_config, cross_config) as detector:
            detector.build_reference_index(test_data, "test")
            assert detector.reference_count == 3


class TestDetectTrainTestLeakage:
    """Tests for detect_train_test_leakage function."""

    def test_detect_leakage(self, train_data, test_data):
        """Test detecting train/test leakage."""
        results = detect_train_test_leakage(
            train_data, test_data, mode="exact", threshold=0.95
        )

        assert results["contaminated_count"] == 1
        assert results["reference_count"] == 3
        assert 3 in results["contaminated_train_indices"]

    def test_detect_no_leakage(self, unique_data):
        """Test no leakage detection."""
        # Split unique data - no overlap
        train = unique_data[:3]
        test = unique_data[3:]

        results = detect_train_test_leakage(train, test, mode="exact")

        assert results["contaminated_count"] == 0
        assert results["contamination_rate"] == 0.0

    def test_detect_full_leakage(self):
        """Test full contamination detection."""
        data = [{"text": "same text", "label": i} for i in range(5)]

        results = detect_train_test_leakage(data, data[:1], mode="exact")

        # All training items match the single test item
        assert results["contaminated_count"] == 5


class TestFilterTrainTestLeakage:
    """Tests for filter_train_test_leakage function."""

    def test_filter_leakage(self, train_data, test_data):
        """Test filtering train/test leakage."""
        filtered = filter_train_test_leakage(train_data, test_data, mode="exact")

        assert len(filtered) == 4

    def test_filter_no_change(self, unique_data):
        """Test filtering when no leakage."""
        train = unique_data[:3]
        test = unique_data[3:]

        filtered = filter_train_test_leakage(train, test, mode="exact")

        assert len(filtered) == 3


class TestCrossDatasetDeduplicator:
    """Tests for CrossDatasetDeduplicator class."""

    def test_add_single_dataset(self, unique_data, exact_config):
        """Test adding a single dataset."""
        with CrossDatasetDeduplicator(exact_config) as dedup:
            kept, removed = dedup.add_dataset(unique_data, "dataset1")

            assert kept == 5
            assert removed == 0

    def test_add_dataset_with_duplicates(self, sample_data, exact_config):
        """Test adding dataset with internal duplicates."""
        with CrossDatasetDeduplicator(exact_config) as dedup:
            kept, removed = dedup.add_dataset(sample_data, "dataset1")

            assert kept == 5
            assert removed == 3

    def test_cross_dataset_dedup(self, exact_config):
        """Test deduplication across datasets."""
        dataset1 = [
            {"text": "shared item", "label": 0},
            {"text": "unique to ds1", "label": 1},
        ]
        dataset2 = [
            {"text": "shared item", "label": 0},  # Duplicate of dataset1
            {"text": "unique to ds2", "label": 1},
        ]

        with CrossDatasetDeduplicator(exact_config) as dedup:
            kept1, removed1 = dedup.add_dataset(dataset1, "ds1")
            kept2, removed2 = dedup.add_dataset(dataset2, "ds2")

            assert kept1 == 2
            assert removed1 == 0
            assert kept2 == 1  # shared item removed
            assert removed2 == 1

    def test_get_valid_indices(self, sample_data, exact_config):
        """Test getting valid indices for a dataset."""
        with CrossDatasetDeduplicator(exact_config) as dedup:
            dedup.add_dataset(sample_data, "ds1")
            valid = dedup.get_valid_indices("ds1")

            assert len(valid) == 5

    def test_filter_dataset(self, sample_data, exact_config):
        """Test filtering a dataset."""
        with CrossDatasetDeduplicator(exact_config) as dedup:
            dedup.add_dataset(sample_data, "ds1")
            filtered = dedup.filter_dataset(sample_data, "ds1")

            assert len(filtered) == 5

    def test_metrics(self, sample_data, exact_config):
        """Test metrics collection."""
        with CrossDatasetDeduplicator(exact_config) as dedup:
            dedup.add_dataset(sample_data, "ds1")
            metrics = dedup.get_metrics()

            assert metrics.total_seen == 8
            assert metrics.duplicates_removed == 3


# HuggingFace-specific tests
try:
    from datasets import Dataset as HFDataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@pytest.mark.skipif(not HF_AVAILABLE, reason="HuggingFace datasets not available")
class TestContaminationWithHuggingFace:
    """Tests for contamination detection with HuggingFace datasets."""

    def test_filter_hf_dataset(self, hf_train_dataset, hf_test_dataset, exact_config, temp_dir):
        """Test filtering HuggingFace dataset."""
        cross_config = CrossDatasetConfig(reference_store_path=temp_dir / "ref")

        with ContaminationDetector(exact_config, cross_config) as detector:
            detector.build_reference_index(hf_test_dataset, "test")
            filtered = detector.filter_contaminated(hf_train_dataset)

            # Should return HFDataset
            assert isinstance(filtered, HFDataset)
            assert len(filtered) == 4

    def test_detect_leakage_hf(self, hf_train_dataset, hf_test_dataset):
        """Test leakage detection with HuggingFace datasets."""
        results = detect_train_test_leakage(
            hf_train_dataset, hf_test_dataset, mode="exact"
        )

        assert results["contaminated_count"] == 1

    def test_filter_leakage_hf(self, hf_train_dataset, hf_test_dataset):
        """Test filtering with HuggingFace datasets."""
        filtered = filter_train_test_leakage(
            hf_train_dataset, hf_test_dataset, mode="exact"
        )

        assert isinstance(filtered, HFDataset)
        assert len(filtered) == 4
