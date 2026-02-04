"""Duplicate detection quality benchmarks.

This module contains benchmarks that answer:
"Am I removing true duplicates or losing valuable training signal?"

Key Questions Answered:
- What is the precision/recall of duplicate detection?
- Are class distributions preserved after deduplication?
- Does dedup disproportionately remove minority class samples?
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import BenchmarkConfig, DedupMode
from ..datasets import (
    SyntheticDataset,
    get_dataset,
    generate_classification_dataset,
    generate_imbalanced_dataset,
)
from ..metrics import (
    BenchmarkResult,
    DedupQualityMetrics,
    ClassDistribution,
    kl_divergence,
    minority_class_retention,
)

# Import prestige for deduplication
try:
    import prestige
    from prestige.dataloaders import DedupConfig, DedupMode as PrestigeDedupMode, DedupDataset

    PRESTIGE_AVAILABLE = True
except ImportError:
    PRESTIGE_AVAILABLE = False


def _run_deduplication(
    dataset: SyntheticDataset,
    mode: DedupMode,
    threshold: float = 0.9,
) -> Tuple[List[int], List[int], DedupQualityMetrics]:
    """Run deduplication and compute quality metrics against ground truth.

    Args:
        dataset: SyntheticDataset with ground truth duplicate annotations
        mode: Deduplication mode
        threshold: Similarity threshold for semantic mode

    Returns:
        Tuple of (kept_indices, removed_indices, quality_metrics)
    """
    texts = dataset.texts
    labels = dataset.labels
    ground_truth_duplicates = set(
        i for i, sample in enumerate(dataset.samples) if sample.is_duplicate
    )

    # Run deduplication
    if not PRESTIGE_AVAILABLE:
        # Simple hash-based dedup
        seen = {}
        kept_indices = []
        removed_indices = []

        for idx, text in enumerate(texts):
            if text not in seen:
                seen[text] = idx
                kept_indices.append(idx)
            else:
                removed_indices.append(idx)
    else:
        # Use prestige
        data = [{"text": t, "label": l} for t, l in zip(texts, labels)]

        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "dedup_store"

            if mode == DedupMode.SEMANTIC:
                dedup_mode = PrestigeDedupMode.SEMANTIC
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

            try:
                dedup_dataset = DedupDataset(data, config, precompute=True)
                kept_indices = dedup_dataset.get_valid_indices()
                removed_indices = dedup_dataset.get_removed_indices()
            except Exception:
                # Fallback
                seen = {}
                kept_indices = []
                removed_indices = []
                for idx, text in enumerate(texts):
                    if text not in seen:
                        seen[text] = idx
                        kept_indices.append(idx)
                    else:
                        removed_indices.append(idx)

    # Compute quality metrics against ground truth
    removed_set = set(removed_indices)
    metrics = DedupQualityMetrics()

    for idx in range(len(texts)):
        predicted_duplicate = idx in removed_set
        is_true_duplicate = idx in ground_truth_duplicates
        metrics.update(predicted_duplicate, is_true_duplicate)

    return kept_indices, removed_indices, metrics


def bench_precision_recall_curve(
    config: BenchmarkConfig,
    seed: int = 42,
) -> List[BenchmarkResult]:
    """Benchmark: Precision/recall curve across thresholds.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        List of BenchmarkResult with precision/recall at each threshold
    """
    # Only meaningful for semantic mode
    if config.dedup.mode != DedupMode.SEMANTIC:
        # Single result for exact mode
        dataset = get_dataset(config.dataset.name, seed=seed)
        _, _, metrics = _run_deduplication(dataset, DedupMode.EXACT)

        return [BenchmarkResult(
            benchmark_name="precision_recall_curve",
            dataset_name=config.dataset.name,
            dedup_mode="exact",
            dedup_quality=metrics,
        )]

    dataset = get_dataset(config.dataset.name, seed=seed)
    results = []

    for threshold in config.dedup.threshold_sweep:
        _, _, metrics = _run_deduplication(dataset, DedupMode.SEMANTIC, threshold)

        results.append(BenchmarkResult(
            benchmark_name="precision_recall_curve",
            dataset_name=config.dataset.name,
            dedup_mode="semantic",
            threshold=threshold,
            dedup_quality=metrics,
        ))

    return results


def bench_false_positive_analysis(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: What examples are incorrectly marked as duplicates?

    Identifies false positives - unique items that were removed.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with false positive details
    """
    dataset = get_dataset(config.dataset.name, seed=seed)
    kept_indices, removed_indices, metrics = _run_deduplication(
        dataset, config.dedup.mode, config.dedup.semantic_threshold
    )

    # Find false positives (removed but not actually duplicates)
    ground_truth_duplicates = set(
        i for i, sample in enumerate(dataset.samples) if sample.is_duplicate
    )

    false_positives = [
        idx for idx in removed_indices if idx not in ground_truth_duplicates
    ]

    return BenchmarkResult(
        benchmark_name="false_positive_analysis",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        threshold=config.dedup.semantic_threshold if config.dedup.mode == DedupMode.SEMANTIC else None,
        dedup_quality=metrics,
    )


def bench_threshold_sensitivity(
    config: BenchmarkConfig,
    seed: int = 42,
) -> List[BenchmarkResult]:
    """Benchmark: How does threshold affect precision/recall tradeoff?

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        List of BenchmarkResult showing threshold sensitivity
    """
    if config.dedup.mode != DedupMode.SEMANTIC:
        return []

    dataset = get_dataset(config.dataset.name, seed=seed)
    results = []

    for threshold in config.dedup.threshold_sweep:
        _, _, metrics = _run_deduplication(dataset, DedupMode.SEMANTIC, threshold)

        results.append(BenchmarkResult(
            benchmark_name="threshold_sensitivity",
            dataset_name=config.dataset.name,
            dedup_mode="semantic",
            threshold=threshold,
            dedup_quality=metrics,
        ))

    return results


def bench_label_preservation(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Are class distributions preserved after deduplication?

    Measures KL divergence between original and deduped class distributions.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with class distribution metrics
    """
    dataset = get_dataset(config.dataset.name, seed=seed)
    kept_indices, _, metrics = _run_deduplication(
        dataset, config.dedup.mode, config.dedup.semantic_threshold
    )

    # Compute class distributions
    original_labels = dataset.labels
    deduped_labels = [dataset.labels[i] for i in kept_indices]

    original_dist = ClassDistribution.from_labels(original_labels)
    deduped_dist = ClassDistribution.from_labels(deduped_labels)

    # Compute KL divergence
    kl_div = kl_divergence(original_dist, deduped_dist)

    return BenchmarkResult(
        benchmark_name="label_preservation",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        threshold=config.dedup.semantic_threshold if config.dedup.mode == DedupMode.SEMANTIC else None,
        dedup_quality=metrics,
        original_distribution=original_dist,
        deduped_distribution=deduped_dist,
    )


def bench_rare_class_impact(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Does dedup disproportionately remove minority class samples?

    Uses imbalanced dataset to test fairness of deduplication.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with minority class retention metrics
    """
    # Use imbalanced dataset
    dataset = generate_imbalanced_dataset(
        size=config.dataset.max_samples or 10000,
        duplicate_rate=0.3,
        seed=seed,
    )

    kept_indices, _, metrics = _run_deduplication(
        dataset, config.dedup.mode, config.dedup.semantic_threshold
    )

    # Compute class distributions
    original_labels = dataset.labels
    deduped_labels = [dataset.labels[i] for i in kept_indices]

    original_dist = ClassDistribution.from_labels(original_labels)
    deduped_dist = ClassDistribution.from_labels(deduped_labels)

    # Compute minority class retention
    retention = minority_class_retention(original_dist, deduped_dist, minority_threshold=0.1)

    return BenchmarkResult(
        benchmark_name="rare_class_impact",
        dataset_name="synth_imbalanced",
        dedup_mode=config.dedup.mode.value,
        threshold=config.dedup.semantic_threshold if config.dedup.mode == DedupMode.SEMANTIC else None,
        dedup_quality=metrics,
        original_distribution=original_dist,
        deduped_distribution=deduped_dist,
    )


class DetectionQualityBenchmark:
    """Runner for all detection quality benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_all(self) -> List[BenchmarkResult]:
        """Run all detection quality benchmarks.

        Returns:
            List of BenchmarkResult objects
        """
        import sys
        results = []
        seeds = self.config.statistical.get_seeds()
        print(f"  Running with {len(seeds)} seeds...")
        sys.stdout.flush()

        # Precision/recall curve (threshold sweep)
        print("    precision_recall_curve...", end=" ", flush=True)
        try:
            pr_results = bench_precision_recall_curve(self.config, seed=seeds[0])
            results.extend(pr_results)
            print(f"done ({len(pr_results)} thresholds)")
        except Exception as e:
            print(f"failed: {e}")
        sys.stdout.flush()

        # Other benchmarks
        benchmark_fns = [
            ("false_positive_analysis", bench_false_positive_analysis),
            ("label_preservation", bench_label_preservation),
            ("rare_class_impact", bench_rare_class_impact),
        ]

        for name, fn in benchmark_fns:
            print(f"    {name}...", end=" ", flush=True)
            count = 0
            for seed in seeds:
                try:
                    result = fn(self.config, seed=seed)
                    results.append(result)
                    count += 1
                except Exception as e:
                    pass
            print(f"done ({count} runs)")
            sys.stdout.flush()

        # Threshold sensitivity (only for semantic)
        if self.config.dedup.mode == DedupMode.SEMANTIC:
            print("    threshold_sensitivity...", end=" ", flush=True)
            try:
                threshold_results = bench_threshold_sensitivity(self.config, seed=seeds[0])
                results.extend(threshold_results)
                print(f"done ({len(threshold_results)} thresholds)")
            except Exception as e:
                print(f"failed: {e}")
            sys.stdout.flush()

        return results
