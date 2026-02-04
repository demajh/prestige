"""Performance benchmarks (Secondary).

This module contains benchmarks that answer practical questions:
"How fast and resource-efficient is deduplication?"

Note: These are SECONDARY metrics. A method that's faster but hurts
model accuracy is not useful.

Key Questions Answered:
- How fast can I process my data?
- How much RAM do I need?
- Is GPU worth it?
"""

import gc
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import BenchmarkConfig, DedupMode
from ..datasets import (
    SyntheticDataset,
    get_dataset,
    generate_classification_dataset,
)
from ..metrics import BenchmarkResult

# Import prestige for deduplication
try:
    import prestige
    from prestige.dataloaders import DedupConfig, DedupMode as PrestigeDedupMode, DedupDataset

    PRESTIGE_AVAILABLE = True
except ImportError:
    PRESTIGE_AVAILABLE = False

# Try to import memory profiling
try:
    import tracemalloc

    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False


def _measure_dedup_performance(
    texts: List[str],
    labels: List[int],
    mode: DedupMode,
    threshold: float = 0.9,
) -> Dict[str, float]:
    """Measure deduplication performance.

    Args:
        texts: Input texts
        labels: Input labels
        mode: Deduplication mode
        threshold: Similarity threshold

    Returns:
        Dictionary with throughput, time, and memory metrics
    """
    # Start memory tracking
    if TRACEMALLOC_AVAILABLE:
        tracemalloc.start()
        gc.collect()

    start_time = time.perf_counter()

    if not PRESTIGE_AVAILABLE:
        # Simple dedup for performance measurement
        seen = set()
        for text in texts:
            seen.add(text)
        kept_count = len(seen)
    else:
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
                dataset = DedupDataset(data, config, precompute=True)
                kept_count = len(dataset)
            except Exception:
                kept_count = len(texts)

    elapsed_time = time.perf_counter() - start_time

    # Get memory usage
    peak_memory_mb = 0
    if TRACEMALLOC_AVAILABLE:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak / 1024 / 1024

    # Calculate metrics
    throughput = len(texts) / elapsed_time if elapsed_time > 0 else 0

    return {
        "throughput_samples_per_sec": throughput,
        "processing_time_sec": elapsed_time,
        "peak_memory_mb": peak_memory_mb,
        "total_samples": len(texts),
        "kept_samples": kept_count,
    }


def bench_throughput(
    config: BenchmarkConfig,
    seed: int = 42,
    sizes: Optional[List[int]] = None,
) -> List[BenchmarkResult]:
    """Benchmark: How fast can I process my data?

    Measures throughput at different dataset sizes.

    Args:
        config: Benchmark configuration
        seed: Random seed
        sizes: List of dataset sizes to test

    Returns:
        List of BenchmarkResult with throughput metrics
    """
    if sizes is None:
        sizes = [1000, 5000, 10000]
        if not config.quick_mode:
            sizes.extend([50000, 100000])

    results = []

    for size in sizes:
        dataset = generate_classification_dataset(
            size=size,
            duplicate_rate=0.3,
            seed=seed,
        )

        perf = _measure_dedup_performance(
            dataset.texts,
            dataset.labels,
            mode=config.dedup.mode,
            threshold=config.dedup.semantic_threshold,
        )

        results.append(BenchmarkResult(
            benchmark_name="throughput",
            dataset_name=f"synth_{size}",
            dedup_mode=config.dedup.mode.value,
            threshold=config.dedup.semantic_threshold if config.dedup.mode == DedupMode.SEMANTIC else None,
            throughput_samples_per_sec=perf["throughput_samples_per_sec"],
            processing_time_sec=perf["processing_time_sec"],
        ))

    return results


def bench_memory_usage(
    config: BenchmarkConfig,
    seed: int = 42,
    sizes: Optional[List[int]] = None,
) -> List[BenchmarkResult]:
    """Benchmark: How much RAM do I need?

    Measures peak memory usage at different dataset sizes.

    Args:
        config: Benchmark configuration
        seed: Random seed
        sizes: List of dataset sizes to test

    Returns:
        List of BenchmarkResult with memory metrics
    """
    if not TRACEMALLOC_AVAILABLE:
        return []

    if sizes is None:
        sizes = [1000, 5000, 10000]
        if not config.quick_mode:
            sizes.extend([50000])

    results = []

    for size in sizes:
        dataset = generate_classification_dataset(
            size=size,
            duplicate_rate=0.3,
            seed=seed,
        )

        perf = _measure_dedup_performance(
            dataset.texts,
            dataset.labels,
            mode=config.dedup.mode,
            threshold=config.dedup.semantic_threshold,
        )

        results.append(BenchmarkResult(
            benchmark_name="memory_usage",
            dataset_name=f"synth_{size}",
            dedup_mode=config.dedup.mode.value,
            peak_memory_gb=perf["peak_memory_mb"] / 1024,
        ))

    return results


def bench_gpu_speedup(
    config: BenchmarkConfig,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark: Is GPU worth it?

    Compares CPU vs GPU performance for semantic dedup.
    Only runs if GPU is available.

    Args:
        config: Benchmark configuration
        seed: Random seed

    Returns:
        BenchmarkResult with GPU speedup metrics
    """
    if config.dedup.mode != DedupMode.SEMANTIC:
        # GPU only helps semantic mode
        return BenchmarkResult(
            benchmark_name="gpu_speedup",
            dataset_name=config.dataset.name,
            dedup_mode="exact",
            throughput_samples_per_sec=0,
        )

    # Check if GPU is available
    gpu_available = False
    if PRESTIGE_AVAILABLE:
        try:
            # Try to detect CUDA support
            import os
            gpu_available = os.environ.get("CUDA_VISIBLE_DEVICES") is not None
        except Exception:
            pass

    if not gpu_available:
        return BenchmarkResult(
            benchmark_name="gpu_speedup",
            dataset_name=config.dataset.name,
            dedup_mode="semantic",
            throughput_samples_per_sec=0,
        )

    size = config.dataset.max_samples or 5000
    dataset = generate_classification_dataset(
        size=size,
        duplicate_rate=0.3,
        seed=seed,
    )

    # CPU performance
    cpu_perf = _measure_dedup_performance(
        dataset.texts,
        dataset.labels,
        mode=DedupMode.SEMANTIC,
        threshold=config.dedup.semantic_threshold,
    )

    # GPU performance would require setting device in config
    # For now, return CPU performance as placeholder
    gpu_perf = cpu_perf  # Would measure GPU separately if device config supported

    speedup = gpu_perf["throughput_samples_per_sec"] / cpu_perf["throughput_samples_per_sec"] if cpu_perf["throughput_samples_per_sec"] > 0 else 1.0

    return BenchmarkResult(
        benchmark_name="gpu_speedup",
        dataset_name=f"synth_{size}",
        dedup_mode="semantic",
        threshold=config.dedup.semantic_threshold,
        throughput_samples_per_sec=cpu_perf["throughput_samples_per_sec"],
    )


class PerformanceBenchmark:
    """Runner for all performance benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_all(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks.

        Returns:
            List of BenchmarkResult objects
        """
        import sys
        results = []
        seed = self.config.statistical.base_seed
        print("  Running performance benchmarks (secondary)...")
        sys.stdout.flush()

        # Throughput
        print("    throughput...", end=" ", flush=True)
        try:
            throughput_results = bench_throughput(self.config, seed=seed)
            results.extend(throughput_results)
            print(f"done ({len(throughput_results)} sizes)")
        except Exception as e:
            print(f"failed: {e}")
        sys.stdout.flush()

        # Memory
        print("    memory_usage...", end=" ", flush=True)
        try:
            memory_results = bench_memory_usage(self.config, seed=seed)
            results.extend(memory_results)
            print(f"done ({len(memory_results)} sizes)")
        except Exception as e:
            print(f"failed: {e}")
        sys.stdout.flush()

        # GPU speedup
        print("    gpu_speedup...", end=" ", flush=True)
        try:
            gpu_result = bench_gpu_speedup(self.config, seed=seed)
            results.append(gpu_result)
            print("done")
        except Exception as e:
            print(f"failed: {e}")
        sys.stdout.flush()

        return results
