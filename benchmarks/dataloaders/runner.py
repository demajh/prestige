"""Benchmark runner for dataloader evaluation.

This module provides the main benchmark execution framework that:
- Runs benchmarks across multiple random seeds
- Collects statistical metrics
- Aggregates results
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .config import BenchmarkCategory, BenchmarkConfig
from .datasets import (
    SyntheticDataset,
    get_contaminated_dataset,
    get_dataset,
)
from .metrics import (
    BenchmarkResult,
    GeneralizationMetrics,
    ContaminationMetrics,
    DedupQualityMetrics,
    ClassDistribution,
    compute_confidence_interval,
    EffectSize,
    paired_t_test,
)
from .models import BaselineModel, get_model


class BenchmarkRunner:
    """Main benchmark execution engine.

    Runs benchmarks with statistical rigor:
    - Multiple random seeds
    - Confidence intervals
    - Effect size calculations
    - Hypothesis testing
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results: List[BenchmarkResult] = []

    def run(self) -> List[BenchmarkResult]:
        """Run all configured benchmarks.

        Returns:
            List of BenchmarkResult objects
        """
        categories = self.config.categories

        if BenchmarkCategory.ALL in categories:
            categories = [
                BenchmarkCategory.GENERALIZATION,
                BenchmarkCategory.CONTAMINATION,
                BenchmarkCategory.DETECTION_QUALITY,
                BenchmarkCategory.MODE_COMPARISON,
                BenchmarkCategory.STATISTICAL,
                BenchmarkCategory.PERFORMANCE,
            ]

        for category in categories:
            # Always show progress for --all runs
            print(f"\n{'='*60}")
            print(f"Running {category.value} benchmarks...")
            print(f"{'='*60}")
            import sys
            sys.stdout.flush()

            if category == BenchmarkCategory.GENERALIZATION:
                self._run_generalization_benchmarks()
            elif category == BenchmarkCategory.CONTAMINATION:
                self._run_contamination_benchmarks()
            elif category == BenchmarkCategory.DETECTION_QUALITY:
                self._run_detection_quality_benchmarks()
            elif category == BenchmarkCategory.MODE_COMPARISON:
                self._run_mode_comparison_benchmarks()
            elif category == BenchmarkCategory.STATISTICAL:
                self._run_statistical_benchmarks()
            elif category == BenchmarkCategory.PERFORMANCE:
                self._run_performance_benchmarks()

        return self.results

    def _run_generalization_benchmarks(self) -> None:
        """Run model generalization benchmarks."""
        from .benchmarks.generalization import GeneralizationBenchmark

        benchmark = GeneralizationBenchmark(self.config)
        results = benchmark.run_all()
        self.results.extend(results)

    def _run_contamination_benchmarks(self) -> None:
        """Run contamination detection benchmarks."""
        from .benchmarks.contamination import ContaminationBenchmark

        benchmark = ContaminationBenchmark(self.config)
        results = benchmark.run_all()
        self.results.extend(results)

    def _run_detection_quality_benchmarks(self) -> None:
        """Run dedup detection quality benchmarks."""
        from .benchmarks.detection_quality import DetectionQualityBenchmark

        benchmark = DetectionQualityBenchmark(self.config)
        results = benchmark.run_all()
        self.results.extend(results)

    def _run_mode_comparison_benchmarks(self) -> None:
        """Run semantic vs exact mode comparison benchmarks."""
        from .benchmarks.mode_comparison import ModeComparisonBenchmark

        benchmark = ModeComparisonBenchmark(self.config)
        results = benchmark.run_all()
        self.results.extend(results)

    def _run_statistical_benchmarks(self) -> None:
        """Run statistical reproducibility benchmarks."""
        from .benchmarks.statistical import StatisticalBenchmark

        benchmark = StatisticalBenchmark(self.config)
        results = benchmark.run_all()
        self.results.extend(results)

    def _run_performance_benchmarks(self) -> None:
        """Run processing performance benchmarks."""
        from .benchmarks.performance import PerformanceBenchmark

        benchmark = PerformanceBenchmark(self.config)
        results = benchmark.run_all()
        self.results.extend(results)

    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Save results to JSON file.

        Args:
            output_path: Output file path (default: auto-generated)

        Returns:
            Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.output_dir / f"benchmark_results_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = self._result_to_dict(result)
            serializable_results.append(result_dict)

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self._config_to_dict(),
            "results": serializable_results,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        if self.config.verbose:
            print(f"\nResults saved to: {output_path}")

        return output_path

    def _to_python_type(self, val: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(val, (np.bool_, np.generic)):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {k: self._to_python_type(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self._to_python_type(v) for v in val]
        return val

    def _result_to_dict(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Convert BenchmarkResult to serializable dictionary."""
        d = {
            "benchmark_name": result.benchmark_name,
            "dataset_name": result.dataset_name,
            "dedup_mode": result.dedup_mode,
            "threshold": result.threshold,
        }

        if result.generalization:
            gen = result.generalization
            d["generalization"] = {
                "train_accuracies": gen.train_accuracies,
                "test_accuracies": gen.test_accuracies,
                "baseline_train_accuracies": gen.baseline_train_accuracies,
                "baseline_test_accuracies": gen.baseline_test_accuracies,
                "epochs_to_convergence": gen.epochs_to_convergence,
            }

            # Add computed metrics
            improvement = gen.test_accuracy_improvement()
            d["generalization"]["test_accuracy_improvement"] = {
                "mean": improvement.mean,
                "ci_lower": improvement.lower,
                "ci_upper": improvement.upper,
                "std": improvement.std,
            }

            effect = gen.effect_size()
            d["generalization"]["effect_size"] = {
                "cohens_d": effect.cohens_d,
                "interpretation": effect.interpretation,
            }

            test = gen.hypothesis_test()
            d["generalization"]["hypothesis_test"] = {
                "p_value": test.p_value,
                "is_significant": test.is_significant,
            }

        if result.contamination:
            cont = result.contamination
            d["contamination"] = {
                "contaminated_count": cont.contaminated_count,
                "total_test_samples": cont.total_test_samples,
                "contamination_rate": cont.contamination_rate,
                "leakage_severity": cont.leakage_severity,
            }

        if result.dedup_quality:
            qual = result.dedup_quality
            d["dedup_quality"] = {
                "precision": qual.precision,
                "recall": qual.recall,
                "f1_score": qual.f1_score,
                "accuracy": qual.accuracy,
                "false_positive_rate": qual.false_positive_rate,
                "tp": qual.true_positives,
                "fp": qual.false_positives,
                "tn": qual.true_negatives,
                "fn": qual.false_negatives,
            }

        if result.original_distribution:
            d["original_distribution"] = {
                "counts": result.original_distribution.counts,
                "proportions": result.original_distribution.proportions,
            }

        if result.deduped_distribution:
            d["deduped_distribution"] = {
                "counts": result.deduped_distribution.counts,
                "proportions": result.deduped_distribution.proportions,
            }

        # Performance metrics
        if result.throughput_samples_per_sec is not None:
            d["throughput_samples_per_sec"] = result.throughput_samples_per_sec
        if result.peak_memory_gb is not None:
            d["peak_memory_gb"] = result.peak_memory_gb
        if result.processing_time_sec is not None:
            d["processing_time_sec"] = result.processing_time_sec

        # Convert all numpy types to Python native types
        return self._to_python_type(d)

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dictionary."""
        return {
            "categories": [c.value for c in self.config.categories],
            "quick_mode": self.config.quick_mode,
            "statistical": {
                "num_seeds": self.config.statistical.num_seeds,
                "base_seed": self.config.statistical.base_seed,
                "confidence_level": self.config.statistical.confidence_level,
            },
            "model": {
                "model_type": self.config.model.model_type,
            },
            "dataset": {
                "name": self.config.dataset.name,
                "text_column": self.config.dataset.text_column,
                "label_column": self.config.dataset.label_column,
            },
            "dedup": {
                "mode": self.config.dedup.mode.value,
                "semantic_threshold": self.config.dedup.semantic_threshold,
            },
        }


def run_single_benchmark(
    benchmark_fn: Callable,
    config: BenchmarkConfig,
    **kwargs,
) -> BenchmarkResult:
    """Run a single benchmark function.

    Args:
        benchmark_fn: Benchmark function to run
        config: Benchmark configuration
        **kwargs: Additional arguments for the benchmark

    Returns:
        BenchmarkResult
    """
    return benchmark_fn(config, **kwargs)


def run_multi_seed_benchmark(
    benchmark_fn: Callable,
    config: BenchmarkConfig,
    seeds: Optional[List[int]] = None,
    **kwargs,
) -> List[BenchmarkResult]:
    """Run a benchmark across multiple random seeds.

    Args:
        benchmark_fn: Benchmark function to run
        config: Benchmark configuration
        seeds: List of random seeds (default: from config)
        **kwargs: Additional arguments for the benchmark

    Returns:
        List of BenchmarkResult objects (one per seed)
    """
    if seeds is None:
        seeds = config.statistical.get_seeds()

    results = []
    for seed in seeds:
        if config.verbose:
            print(f"\nRunning with seed {seed}...")

        result = benchmark_fn(config, seed=seed, **kwargs)
        results.append(result)

    return results


def aggregate_multi_seed_results(
    results: List[BenchmarkResult],
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Aggregate results from multiple seeds into a single result.

    Computes confidence intervals and statistical tests.

    Args:
        results: List of single-seed results
        config: Benchmark configuration

    Returns:
        Aggregated BenchmarkResult with statistical metrics
    """
    if not results:
        raise ValueError("No results to aggregate")

    # Use first result as template
    template = results[0]

    # Aggregate generalization metrics
    gen_metrics = None
    if all(r.generalization for r in results):
        gen_metrics = GeneralizationMetrics()

        for r in results:
            if r.generalization.train_accuracies:
                gen_metrics.train_accuracies.extend(r.generalization.train_accuracies)
            if r.generalization.test_accuracies:
                gen_metrics.test_accuracies.extend(r.generalization.test_accuracies)
            if r.generalization.baseline_train_accuracies:
                gen_metrics.baseline_train_accuracies.extend(
                    r.generalization.baseline_train_accuracies
                )
            if r.generalization.baseline_test_accuracies:
                gen_metrics.baseline_test_accuracies.extend(
                    r.generalization.baseline_test_accuracies
                )

    # Aggregate contamination metrics (typically same across seeds)
    cont_metrics = None
    if results[0].contamination:
        cont_metrics = results[0].contamination

    # Aggregate dedup quality metrics
    quality_metrics = None
    if all(r.dedup_quality for r in results):
        quality_metrics = DedupQualityMetrics()
        for r in results:
            quality_metrics.true_positives += r.dedup_quality.true_positives
            quality_metrics.false_positives += r.dedup_quality.false_positives
            quality_metrics.true_negatives += r.dedup_quality.true_negatives
            quality_metrics.false_negatives += r.dedup_quality.false_negatives

    return BenchmarkResult(
        benchmark_name=template.benchmark_name,
        dataset_name=template.dataset_name,
        dedup_mode=template.dedup_mode,
        threshold=template.threshold,
        generalization=gen_metrics,
        contamination=cont_metrics,
        dedup_quality=quality_metrics,
        original_distribution=template.original_distribution,
        deduped_distribution=template.deduped_distribution,
        timestamp=datetime.now().isoformat(),
    )


# Convenience functions
def quick_benchmark(dataset_name: str = "synth_small") -> List[BenchmarkResult]:
    """Run quick benchmarks on a dataset.

    Args:
        dataset_name: Name of synthetic dataset

    Returns:
        List of benchmark results
    """
    from .config import quick_config

    config = quick_config()
    config.dataset.name = dataset_name

    runner = BenchmarkRunner(config)
    return runner.run()


def full_benchmark(dataset_name: str = "synth_classification") -> List[BenchmarkResult]:
    """Run full benchmark suite on a dataset.

    Args:
        dataset_name: Name of synthetic dataset

    Returns:
        List of benchmark results
    """
    from .config import full_config

    config = full_config()
    config.dataset.name = dataset_name

    runner = BenchmarkRunner(config)
    return runner.run()
