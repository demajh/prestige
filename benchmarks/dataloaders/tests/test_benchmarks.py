"""Tests for dataloader benchmarks."""

import pytest
from pathlib import Path

from ..config import BenchmarkConfig, BenchmarkCategory, DedupMode, quick_config
from ..datasets import (
    SyntheticDataset,
    RealDataset,
    generate_classification_dataset,
    generate_imbalanced_dataset,
    generate_contaminated_dataset,
    generate_paraphrase_dataset,
    get_dataset,
    list_datasets,
    list_real_datasets,
    HF_AVAILABLE,
)
from ..metrics import (
    compute_confidence_interval,
    EffectSize,
    paired_t_test,
    ClassDistribution,
    kl_divergence,
    DedupQualityMetrics,
)
from ..models import (
    get_model,
    TfidfLogisticRegression,
    list_models,
)


class TestSyntheticDatasets:
    """Tests for synthetic dataset generation."""

    def test_generate_classification_dataset(self):
        """Test basic classification dataset generation."""
        dataset = generate_classification_dataset(size=100, num_classes=3, seed=42)

        assert len(dataset) == 100
        assert len(dataset.texts) == 100
        assert len(dataset.labels) == 100
        assert all(0 <= label < 3 for label in dataset.labels)

    def test_duplicate_rate(self):
        """Test that duplicate rate is approximately correct."""
        dataset = generate_classification_dataset(
            size=1000, duplicate_rate=0.3, seed=42
        )

        # Should be approximately 30% duplicates (with some tolerance)
        # The actual rate may be slightly higher due to how duplicates are generated
        dup_rate = dataset.duplicate_rate
        assert 0.2 <= dup_rate <= 0.5

    def test_imbalanced_dataset(self):
        """Test imbalanced dataset generation."""
        dataset = generate_imbalanced_dataset(size=1000, seed=42)

        dist = ClassDistribution.from_labels(dataset.labels)

        # First class should dominate
        assert dist.proportions[0] > 0.5

    def test_contaminated_dataset(self):
        """Test contaminated train/test dataset."""
        train, test = generate_contaminated_dataset(
            train_size=800,
            test_size=200,
            contamination_rate=0.1,
            seed=42,
        )

        assert len(train) == 800
        assert len(test) == 200

        # Should have contaminated indices recorded
        assert "contaminated_indices" in train.metadata

    def test_paraphrase_dataset(self):
        """Test paraphrase dataset generation."""
        dataset = generate_paraphrase_dataset(
            size=500, paraphrase_rate=0.4, seed=42
        )

        paraphrase_count = sum(1 for s in dataset.samples if s.is_paraphrase)
        assert paraphrase_count > 0

    def test_get_dataset(self):
        """Test dataset retrieval by name."""
        dataset = get_dataset("synth_small", seed=42)
        assert len(dataset) == 1000

    def test_list_datasets(self):
        """Test listing available datasets."""
        names = list_datasets()
        assert "synth_classification" in names
        assert "synth_small" in names

    def test_list_datasets_includes_real(self):
        """Test that list_datasets includes real datasets."""
        names = list_datasets(include_real=True)
        assert "mrpc" in names
        assert "qqp" in names
        assert "sst2" in names

    def test_list_datasets_synthetic_only(self):
        """Test that list_datasets can exclude real datasets."""
        names = list_datasets(include_real=False)
        assert "synth_classification" in names
        assert "mrpc" not in names


class TestRealDatasets:
    """Tests for real dataset loading."""

    def test_list_real_datasets(self):
        """Test listing real datasets."""
        names = list_real_datasets()
        assert "mrpc" in names
        assert "qqp" in names
        assert "sst2" in names
        assert "ag_news" in names

    @pytest.mark.skipif(not HF_AVAILABLE, reason="HuggingFace datasets not installed")
    def test_load_sst2(self):
        """Test loading SST-2 dataset."""
        dataset = get_dataset("sst2", split="validation", max_samples=100)

        assert isinstance(dataset, RealDataset)
        assert len(dataset) == 100
        assert len(dataset.texts) == 100
        assert len(dataset.labels) == 100
        assert dataset.name == "sst2"

    @pytest.mark.skipif(not HF_AVAILABLE, reason="HuggingFace datasets not installed")
    def test_load_mrpc(self):
        """Test loading MRPC dataset (paraphrase pairs)."""
        dataset = get_dataset("mrpc", split="validation")

        assert isinstance(dataset, RealDataset)
        assert len(dataset) > 0
        assert dataset.text_pairs is not None
        assert dataset.pair_labels is not None
        assert len(dataset.text_pairs) == len(dataset)

    @pytest.mark.skipif(not HF_AVAILABLE, reason="HuggingFace datasets not installed")
    def test_real_dataset_interface_compatibility(self):
        """Test that RealDataset has similar interface to SyntheticDataset."""
        real = get_dataset("sst2", split="validation", max_samples=50)
        synth = get_dataset("synth_small", seed=42)

        # Both should support these operations
        assert len(real) > 0
        assert len(synth) > 0

        assert hasattr(real, 'texts')
        assert hasattr(synth, 'texts')

        assert hasattr(real, 'labels')
        assert hasattr(synth, 'labels')

        assert hasattr(real, 'to_dict_list')
        assert hasattr(synth, 'to_dict_list')

        # to_dict_list should return list of dicts
        real_dicts = real.to_dict_list()
        synth_dicts = synth.to_dict_list()

        assert isinstance(real_dicts, list)
        assert isinstance(synth_dicts, list)
        assert "text" in real_dicts[0] or "label" in real_dicts[0]


class TestStatisticalMetrics:
    """Tests for statistical metric calculations."""

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        values = [0.8, 0.82, 0.79, 0.81, 0.83]
        ci = compute_confidence_interval(values, confidence=0.95)

        assert 0.79 < ci.mean < 0.83
        assert ci.lower < ci.mean < ci.upper
        assert ci.std > 0

    def test_effect_size(self):
        """Test Cohen's d calculation."""
        group1 = [0.70, 0.72, 0.71, 0.69, 0.70]
        group2 = [0.85, 0.87, 0.86, 0.84, 0.86]

        effect = EffectSize.compute(group1, group2)

        assert effect.cohens_d > 0  # Group 2 is better
        assert effect.interpretation in ["small", "medium", "large"]

    def test_paired_t_test(self):
        """Test paired t-test."""
        group1 = [0.70, 0.72, 0.71, 0.69, 0.70]
        group2 = [0.85, 0.87, 0.86, 0.84, 0.86]

        result = paired_t_test(group1, group2)

        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant

    def test_class_distribution(self):
        """Test class distribution calculation."""
        labels = [0, 0, 0, 1, 1, 2]
        dist = ClassDistribution.from_labels(labels)

        assert dist.counts[0] == 3
        assert dist.counts[1] == 2
        assert dist.counts[2] == 1
        assert abs(dist.proportions[0] - 0.5) < 0.01

    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        labels1 = [0, 0, 0, 1, 1, 2]
        labels2 = [0, 0, 0, 1, 1, 2]  # Same

        dist1 = ClassDistribution.from_labels(labels1)
        dist2 = ClassDistribution.from_labels(labels2)

        kl = kl_divergence(dist1, dist2)
        assert kl < 0.01  # Should be very small for identical distributions

    def test_dedup_quality_metrics(self):
        """Test dedup quality metric calculation."""
        metrics = DedupQualityMetrics()

        # Add some predictions
        metrics.update(predicted_duplicate=True, is_true_duplicate=True)  # TP
        metrics.update(predicted_duplicate=True, is_true_duplicate=False)  # FP
        metrics.update(predicted_duplicate=False, is_true_duplicate=True)  # FN
        metrics.update(predicted_duplicate=False, is_true_duplicate=False)  # TN

        assert metrics.true_positives == 1
        assert metrics.false_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_negatives == 1
        assert metrics.precision == 0.5
        assert metrics.recall == 0.5


class TestModels:
    """Tests for baseline models."""

    def test_logistic_regression(self):
        """Test TF-IDF + Logistic Regression model."""
        model = TfidfLogisticRegression(max_features=100, random_state=42)

        texts = [
            "This is a positive example",
            "Another positive sentence",
            "Negative sentiment here",
            "More negative content",
        ]
        labels = [1, 1, 0, 0]

        result = model.fit(texts, labels)
        assert result.train_accuracy > 0.5

        preds = model.predict(texts)
        assert len(preds) == 4

        eval_result = model.evaluate(texts, labels)
        assert "accuracy" in eval_result

    def test_get_model(self):
        """Test model retrieval by type."""
        model = get_model("logistic_regression", random_state=42)
        assert isinstance(model, TfidfLogisticRegression)

    def test_list_models(self):
        """Test listing available models."""
        names = list_models()
        assert "logistic_regression" in names


class TestConfig:
    """Tests for benchmark configuration."""

    def test_quick_config(self):
        """Test quick mode configuration."""
        config = quick_config()
        assert config.quick_mode
        assert config.statistical.num_seeds == 2

    def test_config_defaults(self):
        """Test default configuration."""
        config = BenchmarkConfig()
        assert config.statistical.num_seeds == 5
        assert config.dataset.text_column == "text"

    def test_category_selection(self):
        """Test category selection."""
        config = BenchmarkConfig(
            categories=[BenchmarkCategory.GENERALIZATION]
        )
        assert BenchmarkCategory.GENERALIZATION in config.categories


class TestBenchmarks:
    """Tests for benchmark execution."""

    def test_generalization_benchmark_runs(self):
        """Test that generalization benchmark can execute."""
        from ..benchmarks.generalization import bench_test_accuracy_with_dedup

        config = quick_config()
        config.dataset.name = "synth_small"

        result = bench_test_accuracy_with_dedup(config, seed=42)

        assert result.benchmark_name == "test_accuracy_with_dedup"
        assert result.generalization is not None

    def test_contamination_benchmark_runs(self):
        """Test that contamination benchmark can execute."""
        from ..benchmarks.contamination import bench_contamination_rate

        config = quick_config()

        result = bench_contamination_rate(config, seed=42)

        assert result.benchmark_name == "contamination_rate"
        assert result.contamination is not None

    def test_detection_quality_benchmark_runs(self):
        """Test that detection quality benchmark can execute."""
        from ..benchmarks.detection_quality import bench_label_preservation

        config = quick_config()
        config.dataset.name = "synth_small"

        result = bench_label_preservation(config, seed=42)

        assert result.benchmark_name == "label_preservation"
        assert result.original_distribution is not None
        assert result.deduped_distribution is not None
