"""Statistical significance and reproducibility benchmarks.

This module contains benchmarks that answer:
"Are my results statistically reliable?"

Key Questions Answered:
- Are results stable across random seeds?
- What's the confidence interval for improvement?
- Is the improvement practically significant (effect size)?
"""

from typing import Any, Dict, List, Optional

from sklearn.model_selection import train_test_split

from ..config import BenchmarkConfig, DedupMode
from ..datasets import get_dataset
from ..metrics import (
    BenchmarkResult,
    GeneralizationMetrics,
    compute_confidence_interval,
    EffectSize,
    paired_t_test,
    HypothesisTest,
)
from ..models import get_model

from .generalization import _apply_deduplication


def bench_multi_seed_variance(
    config: BenchmarkConfig,
    seeds: Optional[List[int]] = None,
) -> BenchmarkResult:
    """Benchmark: Are results stable across random seeds?

    Runs the same experiment across multiple seeds and reports variance.

    Args:
        config: Benchmark configuration
        seeds: List of seeds to use (default: from config)

    Returns:
        BenchmarkResult with multi-seed variance metrics
    """
    if seeds is None:
        seeds = config.statistical.get_seeds()

    dataset = get_dataset(config.dataset.name, seed=seeds[0])

    baseline_accs = []
    dedup_accs = []

    for seed in seeds:
        texts = dataset.texts
        labels = dataset.labels

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels,
            test_size=config.dataset.test_size,
            stratify=labels if config.dataset.stratify else None,
            random_state=seed,
        )

        # Baseline (no dedup)
        baseline_model = get_model(config.model.model_type, random_state=seed)
        baseline_model.fit(train_texts, train_labels)
        baseline_acc = baseline_model.evaluate(test_texts, test_labels)["accuracy"]
        baseline_accs.append(baseline_acc)

        # Deduped
        dedup_texts, dedup_labels, _ = _apply_deduplication(
            train_texts, train_labels,
            mode=config.dedup.mode,
            threshold=config.dedup.semantic_threshold,
        )
        dedup_model = get_model(config.model.model_type, random_state=seed)
        dedup_model.fit(dedup_texts, dedup_labels)
        dedup_acc = dedup_model.evaluate(test_texts, test_labels)["accuracy"]
        dedup_accs.append(dedup_acc)

    gen_metrics = GeneralizationMetrics(
        test_accuracies=dedup_accs,
        baseline_test_accuracies=baseline_accs,
    )

    return BenchmarkResult(
        benchmark_name="multi_seed_variance",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        generalization=gen_metrics,
    )


def bench_confidence_intervals(
    config: BenchmarkConfig,
    seeds: Optional[List[int]] = None,
) -> BenchmarkResult:
    """Benchmark: What's the 95% CI for accuracy improvement?

    Args:
        config: Benchmark configuration
        seeds: List of seeds to use

    Returns:
        BenchmarkResult with confidence interval metrics
    """
    if seeds is None:
        seeds = config.statistical.get_seeds()

    # Run the multi-seed benchmark to get data
    result = bench_multi_seed_variance(config, seeds)

    # The CI is already computed in GeneralizationMetrics
    return BenchmarkResult(
        benchmark_name="confidence_intervals",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        generalization=result.generalization,
    )


def bench_effect_size(
    config: BenchmarkConfig,
    seeds: Optional[List[int]] = None,
) -> BenchmarkResult:
    """Benchmark: Is the improvement practically significant?

    Computes Cohen's d effect size.

    Args:
        config: Benchmark configuration
        seeds: List of seeds to use

    Returns:
        BenchmarkResult with effect size metrics
    """
    if seeds is None:
        seeds = config.statistical.get_seeds()

    # Run multi-seed to get data
    result = bench_multi_seed_variance(config, seeds)
    gen = result.generalization

    # Compute effect size
    effect = EffectSize.compute(
        gen.baseline_test_accuracies,
        gen.test_accuracies,
    )

    # Compute hypothesis test
    test = paired_t_test(
        gen.baseline_test_accuracies,
        gen.test_accuracies,
        alpha=config.statistical.alpha,
    )

    return BenchmarkResult(
        benchmark_name="effect_size",
        dataset_name=config.dataset.name,
        dedup_mode=config.dedup.mode.value,
        generalization=gen,
        effect_size=effect,
        hypothesis_test=test,
    )


class StatisticalBenchmark:
    """Runner for all statistical benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_all(self) -> List[BenchmarkResult]:
        """Run all statistical benchmarks.

        Returns:
            List of BenchmarkResult objects
        """
        results = []
        seeds = self.config.statistical.get_seeds()

        if self.config.verbose:
            print("Running statistical benchmarks...")

        # These benchmarks internally use multiple seeds
        benchmark_fns = [
            ("multi_seed_variance", bench_multi_seed_variance),
            ("confidence_intervals", bench_confidence_intervals),
            ("effect_size", bench_effect_size),
        ]

        for name, fn in benchmark_fns:
            if self.config.verbose:
                print(f"\n  {name}...")

            try:
                result = fn(self.config, seeds=seeds)
                results.append(result)
            except Exception as e:
                if self.config.verbose:
                    print(f"    Warning: failed: {e}")

        return results
