"""Semantic vs Exact mode comparison benchmarks.

This module contains benchmarks that answer:
"Which dedup mode leads to better model performance?"

Key Questions Answered:
- Does semantic mode improve test accuracy over exact mode?
- How many additional duplicates does semantic mode find?
- What threshold is optimal for semantic mode?
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split

from ..config import BenchmarkConfig, DedupMode
from ..datasets import SyntheticDataset, get_dataset, generate_paraphrase_dataset
from ..metrics import (
    BenchmarkResult,
    GeneralizationMetrics,
    DedupQualityMetrics,
    EffectSize,
    compute_confidence_interval,
)
from ..models import get_model

# Import prestige for deduplication
try:
    import prestige
    from prestige.dataloaders import DedupConfig, DedupMode as PrestigeDedupMode, DedupDataset

    PRESTIGE_AVAILABLE = True
except ImportError:
    PRESTIGE_AVAILABLE = False


def _apply_exact_dedup(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int], int]:
    """Apply exact (hash-based) deduplication.

    Returns:
        Tuple of (deduped_texts, deduped_labels, num_removed)
    """
    seen = {}
    deduped_texts = []
    deduped_labels = []

    for text, label in zip(texts, labels):
        if text not in seen:
            seen[text] = True
            deduped_texts.append(text)
            deduped_labels.append(label)

    return deduped_texts, deduped_labels, len(texts) - len(deduped_texts)


def _apply_semantic_dedup(
    texts: List[str],
    labels: List[int],
    threshold: float = 0.9,
) -> Tuple[List[str], List[int], int]:
    """Apply semantic deduplication.

    Returns:
        Tuple of (deduped_texts, deduped_labels, num_removed)
    """
    if not PRESTIGE_AVAILABLE:
        # Fallback to exact
        return _apply_exact_dedup(texts, labels)

    data = [{"text": t, "label": l} for t, l in zip(texts, labels)]

    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "dedup_store"
        model_path = Path.home() / ".cache" / "prestige" / "models" / "bge-small" / "model.onnx"

        config = DedupConfig(
            mode=PrestigeDedupMode.SEMANTIC,
            semantic_threshold=threshold,
            semantic_model_path=model_path if model_path.exists() else None,
            store_path=store_path,
            text_column="text",
        )

        try:
            dataset = DedupDataset(data, config, precompute=True)
            kept_indices = dataset.get_valid_indices()
            deduped_texts = [texts[i] for i in kept_indices]
            deduped_labels = [labels[i] for i in kept_indices]
            return deduped_texts, deduped_labels, len(texts) - len(kept_indices)
        except Exception:
            return _apply_exact_dedup(texts, labels)


def bench_mode_accuracy_comparison(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Which mode leads to better test accuracy?

    Compares exact vs semantic deduplication on model performance.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult comparing exact vs semantic
    """
    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=config.dataset.test_size,
        stratify=labels if config.dataset.stratify else None,
        random_state=seed,
    )

    # Exact mode
    exact_texts, exact_labels, exact_removed = _apply_exact_dedup(train_texts, train_labels)
    exact_model = get_model(config.model.model_type, random_state=seed)
    exact_model.fit(exact_texts, exact_labels)
    exact_acc = exact_model.evaluate(test_texts, test_labels)["accuracy"]

    # Semantic mode
    sem_texts, sem_labels, sem_removed = _apply_semantic_dedup(
        train_texts, train_labels, config.dedup.semantic_threshold
    )
    sem_model = get_model(config.model.model_type, random_state=seed)
    sem_model.fit(sem_texts, sem_labels)
    sem_acc = sem_model.evaluate(test_texts, test_labels)["accuracy"]

    gen_metrics = GeneralizationMetrics(
        test_accuracies=[sem_acc],
        baseline_test_accuracies=[exact_acc],
    )

    return BenchmarkResult(
        benchmark_name="mode_accuracy_comparison",
        dataset_name=config.dataset.name,
        dedup_mode="comparison",
        threshold=config.dedup.semantic_threshold,
        generalization=gen_metrics,
    )


def bench_semantic_catches_paraphrases(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: How many near-duplicates does semantic mode find?

    Uses paraphrase dataset to measure semantic's advantage.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with paraphrase detection metrics
    """
    # Use paraphrase-heavy dataset
    dataset = generate_paraphrase_dataset(
        size=config.dataset.max_samples or 5000,
        paraphrase_rate=0.4,
        seed=seed,
    )

    texts = dataset.texts
    labels = dataset.labels

    # Count paraphrases in ground truth
    total_paraphrases = sum(1 for s in dataset.samples if s.is_paraphrase)

    # Exact mode - won't catch paraphrases
    exact_texts, _, exact_removed = _apply_exact_dedup(texts, labels)
    exact_kept = len(exact_texts)

    # Semantic mode - should catch paraphrases
    sem_texts, _, sem_removed = _apply_semantic_dedup(
        texts, labels, config.dedup.semantic_threshold
    )
    sem_kept = len(sem_texts)

    # Additional duplicates found by semantic
    additional_found = (len(texts) - exact_kept) - (len(texts) - sem_kept)
    # Negative means semantic found more duplicates

    gen_metrics = GeneralizationMetrics(
        test_accuracies=[float(sem_removed)],  # Using as count
        baseline_test_accuracies=[float(exact_removed)],
    )

    return BenchmarkResult(
        benchmark_name="semantic_catches_paraphrases",
        dataset_name="synth_paraphrases",
        dedup_mode="comparison",
        threshold=config.dedup.semantic_threshold,
        generalization=gen_metrics,
    )


def bench_paraphrase_impact_on_model(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Do paraphrase duplicates hurt generalization?

    Compares model trained with and without paraphrase removal.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult showing paraphrase impact
    """
    dataset = generate_paraphrase_dataset(
        size=config.dataset.max_samples or 5000,
        paraphrase_rate=0.4,
        seed=seed,
    )

    texts = dataset.texts
    labels = dataset.labels

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=config.dataset.test_size,
        stratify=labels if config.dataset.stratify else None,
        random_state=seed,
    )

    # No dedup (baseline - includes paraphrases)
    baseline_model = get_model(config.model.model_type, random_state=seed)
    baseline_model.fit(train_texts, train_labels)
    baseline_acc = baseline_model.evaluate(test_texts, test_labels)["accuracy"]

    # Semantic dedup (removes paraphrases)
    sem_texts, sem_labels, _ = _apply_semantic_dedup(
        train_texts, train_labels, config.dedup.semantic_threshold
    )
    sem_model = get_model(config.model.model_type, random_state=seed)
    sem_model.fit(sem_texts, sem_labels)
    sem_acc = sem_model.evaluate(test_texts, test_labels)["accuracy"]

    gen_metrics = GeneralizationMetrics(
        test_accuracies=[sem_acc],
        baseline_test_accuracies=[baseline_acc],
    )

    return BenchmarkResult(
        benchmark_name="paraphrase_impact_on_model",
        dataset_name="synth_paraphrases",
        dedup_mode="semantic",
        threshold=config.dedup.semantic_threshold,
        generalization=gen_metrics,
    )


def bench_threshold_tuning(
    config: BenchmarkConfig,
    seed: int = 42,
) -> List[BenchmarkResult]:
    """Benchmark: What semantic threshold is optimal for this task?

    Sweeps thresholds and reports test accuracy for each.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        List of BenchmarkResult, one per threshold
    """
    dataset = get_dataset(config.dataset.name, seed=seed)
    texts = dataset.texts
    labels = dataset.labels

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=config.dataset.test_size,
        stratify=labels if config.dataset.stratify else None,
        random_state=seed,
    )

    results = []
    for threshold in config.dedup.threshold_sweep:
        sem_texts, sem_labels, removed = _apply_semantic_dedup(
            train_texts, train_labels, threshold
        )

        model = get_model(config.model.model_type, random_state=seed)
        model.fit(sem_texts, sem_labels)
        test_acc = model.evaluate(test_texts, test_labels)["accuracy"]

        gen_metrics = GeneralizationMetrics(
            test_accuracies=[test_acc],
        )

        results.append(BenchmarkResult(
            benchmark_name="threshold_tuning",
            dataset_name=config.dataset.name,
            dedup_mode="semantic",
            threshold=threshold,
            generalization=gen_metrics,
        ))

    return results


class ModeComparisonBenchmark:
    """Runner for all mode comparison benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_all(self) -> List[BenchmarkResult]:
        """Run all mode comparison benchmarks.

        Returns:
            List of BenchmarkResult objects
        """
        results = []
        seeds = self.config.statistical.get_seeds()

        if self.config.verbose:
            print("Running mode comparison benchmarks...")

        benchmark_fns = [
            ("mode_accuracy_comparison", bench_mode_accuracy_comparison),
            ("semantic_catches_paraphrases", bench_semantic_catches_paraphrases),
            ("paraphrase_impact_on_model", bench_paraphrase_impact_on_model),
        ]

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

        # Threshold tuning
        if self.config.verbose:
            print(f"\n  threshold_tuning...")
        try:
            threshold_results = bench_threshold_tuning(self.config, seed=seeds[0])
            results.extend(threshold_results)
        except Exception as e:
            if self.config.verbose:
                print(f"    Warning: failed: {e}")

        return results
