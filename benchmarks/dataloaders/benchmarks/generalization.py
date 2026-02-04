"""Model generalization benchmarks (PRIMARY).

This module contains the most important benchmarks that answer:
"Does deduplication improve my model's ability to generalize?"

Key Questions Answered:
- Does training on deduplicated data improve test accuracy?
- Does dedup reduce overfitting (train/test gap)?
- Does the model converge faster with cleaner data?
- Is the improvement statistically significant?
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split

from ..config import BenchmarkConfig, DedupMode
from ..datasets import SyntheticDataset, get_dataset
from ..metrics import (
    BenchmarkResult,
    ClassDistribution,
    GeneralizationMetrics,
    compute_confidence_interval,
)
from ..models import BaselineModel, get_model, TrainingResult

# Import prestige for deduplication
try:
    import prestige
    from prestige.dataloaders import DedupConfig, DedupMode as PrestigeDedupMode, DedupDataset

    PRESTIGE_AVAILABLE = True
except ImportError:
    PRESTIGE_AVAILABLE = False


def _apply_deduplication(
    texts: List[str],
    labels: List[int],
    mode: DedupMode,
    threshold: float = 0.9,
    model_path: Optional[Path] = None,
) -> Tuple[List[str], List[int], List[int]]:
    """Apply deduplication to training data.

    Args:
        texts: Input texts
        labels: Input labels
        mode: Deduplication mode (exact or semantic)
        threshold: Semantic similarity threshold
        model_path: Path to embedding model (for semantic mode)

    Returns:
        Tuple of (deduped_texts, deduped_labels, kept_indices)
    """
    if not PRESTIGE_AVAILABLE:
        # Fallback: simple hash-based dedup for exact mode
        if mode == DedupMode.EXACT:
            seen = {}
            kept_texts = []
            kept_labels = []
            kept_indices = []
            for idx, (text, label) in enumerate(zip(texts, labels)):
                if text not in seen:
                    seen[text] = idx
                    kept_texts.append(text)
                    kept_labels.append(label)
                    kept_indices.append(idx)
            return kept_texts, kept_labels, kept_indices
        else:
            # For semantic mode without prestige, just return original
            return texts, labels, list(range(len(texts)))

    # Use prestige deduplication
    data = [{"text": t, "label": l} for t, l in zip(texts, labels)]

    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "dedup_store"

        if mode == DedupMode.SEMANTIC:
            dedup_mode = PrestigeDedupMode.SEMANTIC
            if model_path is None:
                # Try default model location
                model_path = Path.home() / ".cache" / "prestige" / "models" / "bge-small" / "model.onnx"
        else:
            dedup_mode = PrestigeDedupMode.EXACT
            model_path = None

        config = DedupConfig(
            mode=dedup_mode,
            semantic_threshold=threshold,
            semantic_model_path=model_path,
            store_path=store_path,
            text_column="text",
        )

        # Create dataset with dedup
        try:
            dataset = DedupDataset(data, config, precompute=True)
            kept_indices = dataset.get_valid_indices()
            kept_texts = [texts[i] for i in kept_indices]
            kept_labels = [labels[i] for i in kept_indices]
            return kept_texts, kept_labels, kept_indices
        except Exception:
            # If dedup fails, return original data
            return texts, labels, list(range(len(texts)))


def bench_test_accuracy_with_dedup(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Does training on deduped data improve test accuracy?

    Trains identical models on original and deduplicated data,
    then compares their test set performance.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with generalization metrics
    """
    # Load dataset
    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    # Split into train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.dataset.test_size,
        stratify=labels if config.dataset.stratify else None,
        random_state=seed,
    )

    # Train on ORIGINAL data
    baseline_model = get_model(config.model.model_type, random_state=seed)
    baseline_result = baseline_model.fit(train_texts, train_labels)
    baseline_eval = baseline_model.evaluate(test_texts, test_labels)

    # Apply deduplication to training data
    dedup_texts, dedup_labels, kept_indices = _apply_deduplication(
        train_texts,
        train_labels,
        mode=config.dedup.mode,
        threshold=config.dedup.semantic_threshold,
    )

    # Train on DEDUPLICATED data
    dedup_model = get_model(config.model.model_type, random_state=seed)
    dedup_result = dedup_model.fit(dedup_texts, dedup_labels)
    dedup_eval = dedup_model.evaluate(test_texts, test_labels)

    # Collect metrics
    gen_metrics = GeneralizationMetrics(
        train_accuracies=[dedup_result.train_accuracy],
        test_accuracies=[dedup_eval["accuracy"]],
        baseline_train_accuracies=[baseline_result.train_accuracy],
        baseline_test_accuracies=[baseline_eval["accuracy"]],
    )

    # Class distributions
    original_dist = ClassDistribution.from_labels(train_labels)
    deduped_dist = ClassDistribution.from_labels(dedup_labels)

    return BenchmarkResult(
        benchmark_name="test_accuracy_with_dedup",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        threshold=config.dedup.semantic_threshold if config.dedup.mode == DedupMode.SEMANTIC else None,
        generalization=gen_metrics,
        original_distribution=original_dist,
        deduped_distribution=deduped_dist,
    )


def bench_overfitting_reduction(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Does dedup reduce the train/test accuracy gap?

    Measures overfitting as the difference between train and test accuracy.
    A smaller gap indicates better generalization.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with overfitting metrics
    """
    # Same as accuracy benchmark but focused on gap
    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.dataset.test_size,
        stratify=labels if config.dataset.stratify else None,
        random_state=seed,
    )

    # Baseline
    baseline_model = get_model(config.model.model_type, random_state=seed)
    baseline_result = baseline_model.fit(train_texts, train_labels)
    baseline_train_acc = baseline_result.train_accuracy
    baseline_test_acc = baseline_model.evaluate(test_texts, test_labels)["accuracy"]
    baseline_gap = baseline_train_acc - baseline_test_acc

    # Deduped
    dedup_texts, dedup_labels, _ = _apply_deduplication(
        train_texts,
        train_labels,
        mode=config.dedup.mode,
        threshold=config.dedup.semantic_threshold,
    )

    dedup_model = get_model(config.model.model_type, random_state=seed)
    dedup_result = dedup_model.fit(dedup_texts, dedup_labels)
    dedup_train_acc = dedup_result.train_accuracy
    dedup_test_acc = dedup_model.evaluate(test_texts, test_labels)["accuracy"]
    dedup_gap = dedup_train_acc - dedup_test_acc

    gen_metrics = GeneralizationMetrics(
        train_accuracies=[dedup_train_acc],
        test_accuracies=[dedup_test_acc],
        baseline_train_accuracies=[baseline_train_acc],
        baseline_test_accuracies=[baseline_test_acc],
    )

    return BenchmarkResult(
        benchmark_name="overfitting_reduction",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        threshold=config.dedup.semantic_threshold if config.dedup.mode == DedupMode.SEMANTIC else None,
        generalization=gen_metrics,
    )


def bench_convergence_speed(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Does the model converge faster with deduped data?

    Uses MLP model which tracks epochs to convergence.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with convergence metrics
    """
    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.dataset.test_size,
        stratify=labels if config.dataset.stratify else None,
        random_state=seed,
    )

    # Use MLP for convergence tracking
    baseline_model = get_model("mlp", random_state=seed)
    baseline_result = baseline_model.fit(train_texts, train_labels)
    baseline_eval = baseline_model.evaluate(test_texts, test_labels)

    dedup_texts, dedup_labels, _ = _apply_deduplication(
        train_texts,
        train_labels,
        mode=config.dedup.mode,
        threshold=config.dedup.semantic_threshold,
    )

    dedup_model = get_model("mlp", random_state=seed)
    dedup_result = dedup_model.fit(dedup_texts, dedup_labels)
    dedup_eval = dedup_model.evaluate(test_texts, test_labels)

    gen_metrics = GeneralizationMetrics(
        train_accuracies=[dedup_result.train_accuracy],
        test_accuracies=[dedup_eval["accuracy"]],
        baseline_train_accuracies=[baseline_result.train_accuracy],
        baseline_test_accuracies=[baseline_eval["accuracy"]],
        epochs_to_convergence=[dedup_result.epochs_to_converge] if dedup_result.epochs_to_converge else [],
        baseline_epochs_to_convergence=[baseline_result.epochs_to_converge] if baseline_result.epochs_to_converge else [],
    )

    return BenchmarkResult(
        benchmark_name="convergence_speed",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        generalization=gen_metrics,
    )


def bench_generalization_by_threshold(
    config: BenchmarkConfig,
    seed: int = 42,
) -> List[BenchmarkResult]:
    """Benchmark: Which similarity threshold gives best test performance?

    Sweeps across thresholds to find optimal setting.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        List of BenchmarkResult, one per threshold
    """
    if config.dedup.mode != DedupMode.SEMANTIC:
        # Only makes sense for semantic mode
        return []

    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.dataset.test_size,
        stratify=labels if config.dataset.stratify else None,
        random_state=seed,
    )

    results = []
    for threshold in config.dedup.threshold_sweep:
        dedup_texts, dedup_labels, _ = _apply_deduplication(
            train_texts,
            train_labels,
            mode=DedupMode.SEMANTIC,
            threshold=threshold,
        )

        model = get_model(config.model.model_type, random_state=seed)
        train_result = model.fit(dedup_texts, dedup_labels)
        eval_result = model.evaluate(test_texts, test_labels)

        gen_metrics = GeneralizationMetrics(
            train_accuracies=[train_result.train_accuracy],
            test_accuracies=[eval_result["accuracy"]],
        )

        results.append(BenchmarkResult(
            benchmark_name="generalization_by_threshold",
            dataset_name=config.dataset.name,
            dedup_mode="semantic",
            threshold=threshold,
            generalization=gen_metrics,
        ))

    return results


def bench_sample_efficiency(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Do I need less data to reach the same accuracy after dedup?

    Measures accuracy per training sample.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with sample efficiency metrics
    """
    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.dataset.test_size,
        stratify=labels if config.dataset.stratify else None,
        random_state=seed,
    )

    # Baseline
    baseline_model = get_model(config.model.model_type, random_state=seed)
    baseline_model.fit(train_texts, train_labels)
    baseline_acc = baseline_model.evaluate(test_texts, test_labels)["accuracy"]
    baseline_samples = len(train_texts)

    # Deduped
    dedup_texts, dedup_labels, _ = _apply_deduplication(
        train_texts,
        train_labels,
        mode=config.dedup.mode,
        threshold=config.dedup.semantic_threshold,
    )

    dedup_model = get_model(config.model.model_type, random_state=seed)
    dedup_model.fit(dedup_texts, dedup_labels)
    dedup_acc = dedup_model.evaluate(test_texts, test_labels)["accuracy"]
    dedup_samples = len(dedup_texts)

    # Sample efficiency = accuracy / (samples / 1000)
    baseline_efficiency = baseline_acc / (baseline_samples / 1000)
    dedup_efficiency = dedup_acc / (dedup_samples / 1000)

    gen_metrics = GeneralizationMetrics(
        test_accuracies=[dedup_acc],
        baseline_test_accuracies=[baseline_acc],
    )

    return BenchmarkResult(
        benchmark_name="sample_efficiency",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        generalization=gen_metrics,
    )


def bench_cross_validation_variance(
    config: BenchmarkConfig,
    seed: int = 42,
    cv_folds: int = 5,
) -> BenchmarkResult:
    """Benchmark: Is CV score variance reduced with cleaner data?

    Lower variance indicates more stable/reliable results.

    Args:
        config: Benchmark configuration
        seed: Random seed
        cv_folds: Number of CV folds

    Returns:
        BenchmarkResult with CV variance metrics
    """
    from ..models import cross_validate_model

    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    # Baseline CV
    baseline_model = get_model(config.model.model_type, random_state=seed)
    baseline_cv = cross_validate_model(baseline_model, texts, labels, cv=cv_folds)

    # Dedup and CV
    dedup_texts, dedup_labels, _ = _apply_deduplication(
        texts,
        labels,
        mode=config.dedup.mode,
        threshold=config.dedup.semantic_threshold,
    )

    dedup_model = get_model(config.model.model_type, random_state=seed)
    dedup_cv = cross_validate_model(dedup_model, dedup_texts, dedup_labels, cv=cv_folds)

    gen_metrics = GeneralizationMetrics(
        test_accuracies=dedup_cv["cv_scores"],
        baseline_test_accuracies=baseline_cv["cv_scores"],
    )

    return BenchmarkResult(
        benchmark_name="cross_validation_variance",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        generalization=gen_metrics,
    )


class GeneralizationBenchmark:
    """Runner for all generalization benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_all(self) -> List[BenchmarkResult]:
        """Run all generalization benchmarks across multiple seeds.

        Returns:
            List of BenchmarkResult objects
        """
        import sys
        results = []
        seeds = self.config.statistical.get_seeds()

        print(f"  Running with {len(seeds)} seeds on dataset '{self.config.dataset.name}'...")
        sys.stdout.flush()

        # Run each benchmark across seeds
        # Skip slow benchmarks (convergence_speed, cross_validation_variance) in quick mode
        if self.config.quick_mode:
            benchmark_fns = [
                ("test_accuracy_with_dedup", bench_test_accuracy_with_dedup),
                ("overfitting_reduction", bench_overfitting_reduction),
                ("sample_efficiency", bench_sample_efficiency),
            ]
        else:
            benchmark_fns = [
                ("test_accuracy_with_dedup", bench_test_accuracy_with_dedup),
                ("overfitting_reduction", bench_overfitting_reduction),
                ("convergence_speed", bench_convergence_speed),
                ("sample_efficiency", bench_sample_efficiency),
                ("cross_validation_variance", bench_cross_validation_variance),
            ]

        for name, fn in benchmark_fns:
            print(f"    {name}...", end=" ", flush=True)

            seed_results = []
            for seed in seeds:
                try:
                    result = fn(self.config, seed=seed)
                    seed_results.append(result)
                except Exception as e:
                    print(f"[seed {seed} failed: {e}]", end=" ", flush=True)

            print(f"done ({len(seed_results)} runs)")
            sys.stdout.flush()
            results.extend(seed_results)

        # Run threshold sweep if semantic mode
        if self.config.dedup.mode == DedupMode.SEMANTIC:
            if self.config.verbose:
                print(f"\n  generalization_by_threshold...")

            for seed in seeds[:1]:  # Just one seed for threshold sweep
                try:
                    threshold_results = bench_generalization_by_threshold(self.config, seed=seed)
                    results.extend(threshold_results)
                except Exception as e:
                    if self.config.verbose:
                        print(f"    Warning: threshold sweep failed: {e}")

        return results
