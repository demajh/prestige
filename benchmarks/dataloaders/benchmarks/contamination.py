"""Contamination detection benchmarks.

This module contains benchmarks that answer:
"Is my test set contaminated, making my metrics unreliable?"

Key Questions Answered:
- What percentage of test samples appear in training data?
- How much are my metrics inflated by leakage?
- What's my true accuracy after removing contaminated samples?
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sklearn.model_selection import train_test_split

from ..config import BenchmarkConfig, DedupMode
from ..datasets import SyntheticDataset, get_contaminated_dataset, get_dataset
from ..metrics import (
    BenchmarkResult,
    ContaminationMetrics,
    GeneralizationMetrics,
)
from ..models import get_model

# Import prestige for contamination detection
try:
    import prestige
    from prestige.dataloaders import (
        DedupConfig,
        DedupMode as PrestigeDedupMode,
        ContaminationDetector,
        detect_train_test_leakage,
    )

    PRESTIGE_AVAILABLE = True
except ImportError:
    PRESTIGE_AVAILABLE = False


def _detect_contamination_simple(
    train_texts: List[str],
    test_texts: List[str],
    mode: str = "exact",
) -> Tuple[List[int], float]:
    """Simple contamination detection without prestige.

    Args:
        train_texts: Training texts
        test_texts: Test texts
        mode: "exact" for hash-based

    Returns:
        Tuple of (contaminated_test_indices, contamination_rate)
    """
    # Build set of training text hashes
    train_set = set(train_texts)

    contaminated = []
    for idx, text in enumerate(test_texts):
        if text in train_set:
            contaminated.append(idx)

    rate = len(contaminated) / len(test_texts) if test_texts else 0.0
    return contaminated, rate


def _detect_contamination_prestige(
    train_texts: List[str],
    test_texts: List[str],
    mode: DedupMode,
    threshold: float = 0.95,
) -> Tuple[List[int], float]:
    """Detect contamination using prestige.

    Args:
        train_texts: Training texts
        test_texts: Test texts
        mode: Dedup mode
        threshold: Similarity threshold for semantic mode

    Returns:
        Tuple of (contaminated_test_indices, contamination_rate)
    """
    if not PRESTIGE_AVAILABLE:
        return _detect_contamination_simple(train_texts, test_texts)

    try:
        mode_str = "semantic" if mode == DedupMode.SEMANTIC else "exact"
        train_data = [{"text": t} for t in train_texts]
        test_data = [{"text": t} for t in test_texts]

        results = detect_train_test_leakage(
            train_data=train_data,
            test_data=test_data,
            mode=mode_str,
            threshold=threshold,
            text_column="text",
            verbose=False,
        )

        return results["contaminated_train_indices"], results["contamination_rate"]
    except Exception:
        # Fallback to simple detection
        return _detect_contamination_simple(train_texts, test_texts)


def bench_contamination_rate(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: What percentage of test samples appear in training data?

    Uses the synthetic contaminated dataset with known leakage.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with contamination metrics
    """
    # Use contaminated dataset
    train_dataset, test_dataset = get_contaminated_dataset(seed=seed)

    train_texts = train_dataset.texts
    test_texts = test_dataset.texts

    # Detect contamination
    contaminated_indices, rate = _detect_contamination_prestige(
        train_texts,
        test_texts,
        mode=config.dedup.mode,
        threshold=config.dedup.semantic_threshold,
    )

    # Compare to ground truth (we know which were contaminated)
    true_contaminated = set(train_dataset.metadata.get("contaminated_indices", []))
    detected_set = set(contaminated_indices)

    cont_metrics = ContaminationMetrics(
        contaminated_count=len(contaminated_indices),
        total_test_samples=len(test_texts),
        contaminated_indices=contaminated_indices,
    )

    return BenchmarkResult(
        benchmark_name="contamination_rate",
        dataset_name="synth_contaminated",
        dedup_mode=config.dedup.mode.value,
        threshold=config.dedup.semantic_threshold if config.dedup.mode == DedupMode.SEMANTIC else None,
        contamination=cont_metrics,
    )


def bench_metric_inflation_estimate(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: How much are my metrics inflated by leakage?

    Compares accuracy on full test set vs clean (non-contaminated) test set.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with metric inflation estimates
    """
    train_dataset, test_dataset = get_contaminated_dataset(seed=seed)

    train_texts = train_dataset.texts
    train_labels = train_dataset.labels
    test_texts = test_dataset.texts
    test_labels = test_dataset.labels

    # Detect contamination
    contaminated_indices, _ = _detect_contamination_prestige(
        train_texts,
        test_texts,
        mode=config.dedup.mode,
        threshold=config.dedup.semantic_threshold,
    )
    contaminated_set = set(contaminated_indices)

    # Train model on training data
    model = get_model(config.model.model_type, random_state=seed)
    model.fit(train_texts, train_labels)

    # Evaluate on FULL test set (contaminated)
    full_eval = model.evaluate(test_texts, test_labels)
    full_accuracy = full_eval["accuracy"]

    # Evaluate on CLEAN test set (non-contaminated samples only)
    clean_indices = [i for i in range(len(test_texts)) if i not in contaminated_set]
    clean_texts = [test_texts[i] for i in clean_indices]
    clean_labels = [test_labels[i] for i in clean_indices]

    if clean_texts:
        clean_eval = model.evaluate(clean_texts, clean_labels)
        clean_accuracy = clean_eval["accuracy"]
    else:
        clean_accuracy = full_accuracy

    # Metric inflation = full - clean
    inflation = full_accuracy - clean_accuracy

    gen_metrics = GeneralizationMetrics(
        test_accuracies=[clean_accuracy],
        baseline_test_accuracies=[full_accuracy],
    )

    cont_metrics = ContaminationMetrics(
        contaminated_count=len(contaminated_set),
        total_test_samples=len(test_texts),
        contaminated_indices=list(contaminated_set),
    )

    return BenchmarkResult(
        benchmark_name="metric_inflation_estimate",
        dataset_name="synth_contaminated",
        dedup_mode=config.dedup.mode.value,
        generalization=gen_metrics,
        contamination=cont_metrics,
    )


def bench_contamination_by_threshold(
    config: BenchmarkConfig,
    seed: int = 42,
) -> List[BenchmarkResult]:
    """Benchmark: How sensitive is contamination detection to threshold?

    Sweeps across thresholds for semantic mode.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        List of BenchmarkResult, one per threshold
    """
    if config.dedup.mode != DedupMode.SEMANTIC:
        # Only applicable for semantic mode
        return []

    train_dataset, test_dataset = get_contaminated_dataset(seed=seed)
    train_texts = train_dataset.texts
    test_texts = test_dataset.texts

    results = []
    for threshold in config.dedup.threshold_sweep:
        contaminated_indices, rate = _detect_contamination_prestige(
            train_texts,
            test_texts,
            mode=DedupMode.SEMANTIC,
            threshold=threshold,
        )

        cont_metrics = ContaminationMetrics(
            contaminated_count=len(contaminated_indices),
            total_test_samples=len(test_texts),
            contaminated_indices=contaminated_indices,
        )

        results.append(BenchmarkResult(
            benchmark_name="contamination_by_threshold",
            dataset_name="synth_contaminated",
            dedup_mode="semantic",
            threshold=threshold,
            contamination=cont_metrics,
        ))

    return results


def bench_clean_test_performance(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: What's my true accuracy after removing contaminated samples?

    Reports the clean test accuracy and compares to reported (contaminated).

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with clean test performance
    """
    train_dataset, test_dataset = get_contaminated_dataset(seed=seed)

    train_texts = train_dataset.texts
    train_labels = train_dataset.labels
    test_texts = test_dataset.texts
    test_labels = test_dataset.labels

    # Detect contamination
    contaminated_indices, _ = _detect_contamination_prestige(
        train_texts,
        test_texts,
        mode=config.dedup.mode,
        threshold=config.dedup.semantic_threshold,
    )
    contaminated_set = set(contaminated_indices)

    # Train model
    model = get_model(config.model.model_type, random_state=seed)
    model.fit(train_texts, train_labels)

    # Evaluate on full and clean test sets
    full_accuracy = model.evaluate(test_texts, test_labels)["accuracy"]

    clean_indices = [i for i in range(len(test_texts)) if i not in contaminated_set]
    clean_texts = [test_texts[i] for i in clean_indices]
    clean_labels = [test_labels[i] for i in clean_indices]

    clean_accuracy = model.evaluate(clean_texts, clean_labels)["accuracy"] if clean_texts else full_accuracy

    gen_metrics = GeneralizationMetrics(
        test_accuracies=[clean_accuracy],
        baseline_test_accuracies=[full_accuracy],
    )

    cont_metrics = ContaminationMetrics(
        contaminated_count=len(contaminated_set),
        total_test_samples=len(test_texts),
    )

    return BenchmarkResult(
        benchmark_name="clean_test_performance",
        dataset_name="synth_contaminated",
        dedup_mode=config.dedup.mode.value,
        generalization=gen_metrics,
        contamination=cont_metrics,
    )


def bench_cross_validation_leakage(
    config: BenchmarkConfig,
    seed: int = 42,
    cv_folds: int = 5,
) -> BenchmarkResult:
    """Benchmark: Is there leakage across my CV folds?

    Checks if duplicates span across CV folds, which would inflate scores.

    Args:
        config: Benchmark configuration
        seed: Random seed
        cv_folds: Number of CV folds

    Returns:
        BenchmarkResult with CV leakage metrics
    """
    from sklearn.model_selection import KFold

    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    total_leakage = 0
    total_test_samples = 0

    for train_idx, test_idx in kfold.split(texts):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        # Check for leakage in this fold
        contaminated, _ = _detect_contamination_prestige(
            train_texts,
            test_texts,
            mode=config.dedup.mode,
            threshold=config.dedup.semantic_threshold,
        )

        total_leakage += len(contaminated)
        total_test_samples += len(test_texts)

    overall_rate = total_leakage / total_test_samples if total_test_samples > 0 else 0.0

    cont_metrics = ContaminationMetrics(
        contaminated_count=total_leakage,
        total_test_samples=total_test_samples,
    )

    return BenchmarkResult(
        benchmark_name="cross_validation_leakage",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        contamination=cont_metrics,
    )


class ContaminationBenchmark:
    """Runner for all contamination benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_all(self) -> List[BenchmarkResult]:
        """Run all contamination benchmarks.

        Returns:
            List of BenchmarkResult objects
        """
        results = []

        if self.config.verbose:
            print("Running contamination benchmarks...")

        benchmark_fns = [
            ("contamination_rate", bench_contamination_rate),
            ("metric_inflation_estimate", bench_metric_inflation_estimate),
            ("clean_test_performance", bench_clean_test_performance),
            ("cross_validation_leakage", bench_cross_validation_leakage),
        ]

        seeds = self.config.statistical.get_seeds()

        for name, fn in benchmark_fns:
            if self.config.verbose:
                print(f"\n  {name}...")

            for seed in seeds:
                try:
                    result = fn(self.config, seed=seed)
                    results.append(result)
                except Exception as e:
                    if self.config.verbose:
                        print(f"    Warning: seed {seed} failed: {e}")

        # Threshold sweep (only for semantic mode)
        if self.config.dedup.mode == DedupMode.SEMANTIC:
            if self.config.verbose:
                print(f"\n  contamination_by_threshold...")

            try:
                threshold_results = bench_contamination_by_threshold(
                    self.config, seed=seeds[0]
                )
                results.extend(threshold_results)
            except Exception as e:
                if self.config.verbose:
                    print(f"    Warning: threshold sweep failed: {e}")

        return results
