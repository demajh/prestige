"""Benchmark runner for semantic deduplication evaluation."""

import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

try:
    import prestige
except ImportError:
    raise ImportError(
        "Prestige Python bindings not found. "
        "Please install from python/ directory: pip install -e python/"
    )

from .dataset_loader import DatasetLoader, DatasetConfig
from .metrics import ConfusionMatrix, BenchmarkMetrics, MetricsAggregator


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    dataset_config: DatasetConfig
    thresholds: List[float]
    cache_dir: Path
    store_dir: Optional[Path] = None
    embedding_model: str = "bge-small"
    batch_size: int = 100
    verbose: bool = True
    pooling: str = "mean"  # "mean" or "cls"
    sample_size: Optional[int] = None  # Number of pairs to sample (None = use all)
    device: str = "auto"  # "auto", "gpu", or "cpu"

    # Reranker settings
    enable_reranker: bool = False
    reranker_model: str = "bge-reranker-base"
    reranker_threshold: float = 0.8
    reranker_top_k: int = 100
    reranker_batch_size: int = 8
    reranker_fallback: bool = True

    # Reciprocal Nearest Neighbor (RNN) + Margin Gating settings
    enable_rnn: bool = False
    rnn_k: int = 0  # 0 = use semantic_search_k
    enable_margin: bool = False
    margin_threshold: float = 0.05

    # Judge LLM settings (Prometheus 2 for gray zone evaluation)
    enable_judge: bool = False
    judge_model: str = "prometheus-7b-v2.0"
    judge_threshold: float = 0.75  # Min similarity to trigger judge
    judge_context_size: int = 4096
    judge_gpu_layers: int = 0  # 0 = CPU only, -1 = all layers to GPU


class SemanticDedupBenchmark:
    """Benchmark harness for semantic deduplication evaluation."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.dataset_loader = DatasetLoader(config.cache_dir)
        self.results = MetricsAggregator()

    def run(self) -> MetricsAggregator:
        """Run benchmark across all thresholds.

        Returns:
            Metrics aggregator with results for all thresholds
        """
        # Load dataset
        if self.config.verbose:
            print(f"\nLoading dataset: {self.config.dataset_config.name}")

        df = self.dataset_loader.load_dataset(self.config.dataset_config)

        # Sample if requested
        original_size = len(df)
        if self.config.sample_size is not None and self.config.sample_size < len(df):
            df = df.sample(n=self.config.sample_size, random_state=42)
            if self.config.verbose:
                print(f"Sampled {len(df)} pairs from {original_size} total")

        if self.config.verbose:
            print(f"Dataset size: {len(df)} pairs")
            pos_count = df['is_duplicate'].sum()
            neg_count = len(df) - pos_count
            print(f"  Positive (duplicate): {pos_count} ({100*pos_count/len(df):.1f}%)")
            print(f"  Negative (unique): {neg_count} ({100*neg_count/len(df):.1f}%)")

        # Run for each threshold
        for threshold in self.config.thresholds:
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Running threshold: {threshold}")
                print(f"{'='*60}")

            self._run_threshold(df, threshold)

        return self.results

    def _run_threshold(self, df: pd.DataFrame, threshold: float):
        """Run benchmark for a specific threshold.

        Args:
            df: Dataset dataframe
            threshold: Semantic similarity threshold
        """
        # Create temporary store directory
        if self.config.store_dir:
            store_path = self.config.store_dir / f"threshold_{threshold}"
            store_path.mkdir(parents=True, exist_ok=True)
            cleanup = False
        else:
            temp_dir = tempfile.mkdtemp(prefix=f"prestige_bench_{threshold}_")
            store_path = Path(temp_dir)
            cleanup = True

        try:
            # Open store with semantic deduplication
            options = prestige.Options()
            options.dedup_mode = prestige.DedupMode.SEMANTIC
            options.semantic_threshold = threshold
            options.semantic_model_path = str(self._get_model_path())

            # Set model type based on model name
            model_name = self.config.embedding_model.lower()
            if "e5-large" in model_name or "e5large" in model_name:
                options.semantic_model_type = prestige.SemanticModel.E5_LARGE
            elif "bge-m3" in model_name or "bgem3" in model_name:
                options.semantic_model_type = prestige.SemanticModel.BGE_M3
            elif "nomic" in model_name:
                options.semantic_model_type = prestige.SemanticModel.NOMIC_EMBED
            elif "bge-large" in model_name:
                options.semantic_model_type = prestige.SemanticModel.BGE_LARGE
            elif "bge" in model_name:
                options.semantic_model_type = prestige.SemanticModel.BGE_SMALL
            elif "minilm" in model_name:
                options.semantic_model_type = prestige.SemanticModel.MINILM
            else:
                # Default to MiniLM
                options.semantic_model_type = prestige.SemanticModel.MINILM

            # Set pooling strategy
            if self.config.pooling.lower() == "cls":
                options.semantic_pooling = prestige.SemanticPooling.CLS
            else:
                options.semantic_pooling = prestige.SemanticPooling.MEAN

            # Set device (CPU, GPU, or Auto)
            device_str = self.config.device.lower()
            if device_str == "gpu":
                options.semantic_device = prestige.SemanticDevice.GPU
            elif device_str == "cpu":
                options.semantic_device = prestige.SemanticDevice.CPU
            else:
                options.semantic_device = prestige.SemanticDevice.AUTO

            # Configure reranker if enabled
            if self.config.enable_reranker:
                options.semantic_reranker_enabled = True
                options.semantic_reranker_model_path = str(self._get_reranker_model_path())
                options.semantic_reranker_threshold = self.config.reranker_threshold
                options.semantic_reranker_top_k = self.config.reranker_top_k
                options.semantic_reranker_batch_size = self.config.reranker_batch_size
                options.semantic_reranker_fallback = self.config.reranker_fallback

            # Configure RNN + margin gating if enabled
            if self.config.enable_rnn:
                options.semantic_rnn_enabled = True
                options.semantic_rnn_k = self.config.rnn_k

            if self.config.enable_margin:
                options.semantic_margin_enabled = True
                options.semantic_margin_threshold = self.config.margin_threshold

            # Configure judge LLM if enabled (Prometheus 2 for gray zone evaluation)
            if self.config.enable_judge:
                options.semantic_judge_enabled = True
                options.semantic_judge_model_path = str(self._get_judge_model_path())
                options.semantic_judge_threshold = self.config.judge_threshold
                options.semantic_judge_context_size = self.config.judge_context_size
                options.semantic_judge_gpu_layers = self.config.judge_gpu_layers

            store = prestige.Store.open(str(store_path), options)

            # Initialize metrics
            confusion_matrix = ConfusionMatrix()
            latencies = []
            pair_idx = 0

            # Process dataset
            iterator = self.dataset_loader.get_text_pairs(df)
            if self.config.verbose:
                iterator = tqdm(
                    list(iterator),
                    desc=f"Processing (threshold={threshold})",
                    unit="pairs"
                )

            for text1, text2, ground_truth in iterator:
                # Generate unique keys for this pair
                key1 = f"pair_{pair_idx}_text1"
                key2 = f"pair_{pair_idx}_text2"

                # Measure latency for the pair
                start_time = time.perf_counter()

                try:
                    # Put first text
                    store.put(key1, text1)
                    obj_id1 = store.get_object_id(key1)

                    # Put second text
                    store.put(key2, text2)
                    obj_id2 = store.get_object_id(key2)

                    # Check if they deduplicated to the same object
                    did_dedup = (obj_id1 == obj_id2)

                except Exception as e:
                    if self.config.verbose:
                        print(f"\nError processing pair {pair_idx}: {e}")
                    # Treat errors as non-dedup
                    did_dedup = False

                elapsed = time.perf_counter() - start_time
                latencies.append(elapsed)

                # Update confusion matrix
                confusion_matrix.update(
                    predicted=did_dedup,
                    ground_truth=ground_truth
                )

                pair_idx += 1

                # Periodic flush to avoid memory buildup
                if pair_idx % self.config.batch_size == 0:
                    store.flush()

            # Final flush
            store.flush()

            # Collect final statistics
            health = store.get_health()
            total_keys = health['total_keys']
            unique_objects = health['total_objects']
            storage_bytes = store.total_bytes

            dedup_ratio = total_keys / unique_objects if unique_objects > 0 else 1.0

            # Close store
            store.close()

            # Record results
            self.results.add_threshold_result(
                threshold=threshold,
                confusion_matrix=confusion_matrix,
                latencies=latencies,
                dedup_ratio=dedup_ratio,
                storage_bytes=storage_bytes,
                unique_objects=unique_objects,
                total_keys=total_keys,
            )

            # Print summary
            if self.config.verbose:
                metrics = BenchmarkMetrics.from_confusion_matrix(confusion_matrix)
                status_parts = []
                if self.config.enable_reranker:
                    status_parts.append("reranker")
                if self.config.enable_rnn:
                    status_parts.append(f"RNN-k{self.config.rnn_k or 'auto'}")
                if self.config.enable_margin:
                    status_parts.append(f"margin={self.config.margin_threshold}")
                status_str = f" ({', '.join(status_parts)})" if status_parts else ""
                print(f"\nResults for threshold {threshold}{status_str}:")
                print(f"  Precision: {metrics.precision:.4f}")
                print(f"  Recall:    {metrics.recall:.4f}")
                print(f"  F1 Score:  {metrics.f1_score:.4f}")
                print(f"  Accuracy:  {metrics.accuracy:.4f}")
                print(f"  Dedup Ratio: {dedup_ratio:.2f}x")
                print(f"  Storage: {storage_bytes / 1024 / 1024:.2f} MB")

        finally:
            # Cleanup temporary directory
            if cleanup and store_path.exists():
                shutil.rmtree(store_path)

    def _get_model_path(self) -> Path:
        """Get path to embedding model.

        Returns:
            Path to ONNX model file

        Raises:
            FileNotFoundError: If model not found
        """
        # Check common model locations
        # Model is stored in subdirectory: models/bge-small/model.onnx
        model_file = "model.onnx"
        model_dir_name = self.config.embedding_model

        # Check cache directory
        cache_model = self.config.cache_dir / "models" / model_dir_name / model_file
        if cache_model.exists():
            return cache_model

        # Check user cache (primary location used by models.py)
        user_cache = Path.home() / ".cache" / "prestige" / "models" / model_dir_name / model_file
        if user_cache.exists():
            return user_cache

        # Check models/ directory relative to project root
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models" / model_dir_name / model_file
        if models_dir.exists():
            return models_dir

        raise FileNotFoundError(
            f"Embedding model not found: {model_dir_name}/{model_file}\n"
            f"Searched locations:\n"
            f"  - {cache_model}\n"
            f"  - {user_cache}\n"
            f"  - {models_dir}\n"
            f"Please download the model using:\n"
            f"  python -m benchmarks.semantic_dedup.models {model_dir_name}"
        )

    def _get_reranker_model_path(self) -> Path:
        """Get path to reranker model.

        Returns:
            Path to reranker ONNX model file

        Raises:
            FileNotFoundError: If reranker model not found
        """
        # Check common reranker model locations
        model_file = "model.onnx"
        model_dir_name = self.config.reranker_model

        # Check cache directory
        cache_model = self.config.cache_dir / "models" / model_dir_name / model_file
        if cache_model.exists():
            return cache_model

        # Check user cache
        user_cache = Path.home() / ".cache" / "prestige" / "models" / model_dir_name / model_file
        if user_cache.exists():
            return user_cache

        # Check models/ directory relative to project root
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models" / model_dir_name / model_file
        if models_dir.exists():
            return models_dir

        raise FileNotFoundError(
            f"Reranker model not found: {model_dir_name}/{model_file}\n"
            f"Searched locations:\n"
            f"  - {cache_model}\n"
            f"  - {user_cache}\n"
            f"  - {models_dir}\n"
            f"Please download the BGE reranker from Hugging Face:\n"
            f"  https://huggingface.co/BAAI/bge-reranker-v2-m3"
        )

    def _get_judge_model_path(self) -> Path:
        """Get path to judge LLM model (Prometheus 2).

        Returns:
            Path to GGUF model file

        Raises:
            FileNotFoundError: If judge model not found
        """
        # Look for GGUF model files with various quantization levels
        model_dir_name = self.config.judge_model
        gguf_patterns = [
            "*.gguf",
            "*.Q4_K_M.gguf",
            "*.Q5_K_M.gguf",
            "*.Q8_0.gguf",
            "model.gguf",
        ]

        search_dirs = [
            self.config.cache_dir / "models" / model_dir_name,
            Path.home() / ".cache" / "prestige" / "models" / model_dir_name,
            Path(__file__).parent.parent.parent / "models" / model_dir_name,
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in gguf_patterns:
                    matches = list(search_dir.glob(pattern))
                    if matches:
                        return matches[0]

        raise FileNotFoundError(
            f"Judge LLM model not found: {model_dir_name}\n"
            f"Searched locations:\n"
            + "\n".join(f"  - {d}" for d in search_dirs) +
            f"\n\nPlease download Prometheus 2 from Hugging Face:\n"
            f"  https://huggingface.co/prometheus-eval/prometheus-7b-v2.0\n"
            f"\nRecommended: Download a GGUF quantized version for efficient inference."
        )


def run_quick_benchmark(
    dataset_name: str = "mrpc",
    thresholds: Optional[List[float]] = None,
    cache_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """Quick utility to run a benchmark.

    Args:
        dataset_name: Name of dataset to benchmark
        thresholds: List of thresholds to test (default: [0.85, 0.90, 0.95])
        cache_dir: Cache directory for datasets and models
        verbose: Print progress information

    Returns:
        Dictionary with best F1 results

    Example:
        >>> from benchmarks.semantic_dedup.runner import run_quick_benchmark
        >>> results = run_quick_benchmark("mrpc", thresholds=[0.85, 0.90])
        >>> print(f"Best F1: {results['f1_score']:.3f} at threshold {results['threshold']}")
    """
    from .dataset_loader import get_dataset_config

    if thresholds is None:
        thresholds = [0.85, 0.90, 0.95]

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "prestige" / "benchmarks"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset config
    dataset_config = get_dataset_config(dataset_name)

    # Create benchmark config
    config = BenchmarkConfig(
        dataset_config=dataset_config,
        thresholds=thresholds,
        cache_dir=cache_dir,
        verbose=verbose,
    )

    # Run benchmark
    benchmark = SemanticDedupBenchmark(config)
    results = benchmark.run()

    # Return best F1 result
    return results.get_best_f1()
