"""Configuration classes for deduplicated dataloaders.

This module provides configuration dataclasses for controlling deduplication
behavior in dataloaders, including dedup mode, thresholds, and advanced options.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List


class DedupMode(Enum):
    """Deduplication mode."""

    EXACT = "exact"  # SHA-256 hash-based exact matching
    SEMANTIC = "semantic"  # Embedding-based semantic similarity


class DedupStrategy(Enum):
    """Deduplication strategy."""

    STATIC = "static"  # Pre-process dataset once, cache result
    DYNAMIC = "dynamic"  # Filter during iteration (on-the-fly)


@dataclass
class DedupConfig:
    """Configuration for deduplication behavior.

    Attributes:
        mode: Deduplication mode (EXACT or SEMANTIC)
        strategy: When to deduplicate (STATIC pre-filter or DYNAMIC on-the-fly)
        store_path: Custom path for prestige store (None = temp directory)
        persist_store: Keep store after processing (useful for caching)
        semantic_threshold: Similarity threshold for semantic dedup (0.0-1.0)
        semantic_model_type: Embedding model type ("bge-small", "minilm", etc.)
        semantic_model_path: Custom path to ONNX model file
        semantic_device: Device for inference ("auto", "cpu", "gpu")
        semantic_num_threads: Number of threads for inference (0 = all cores)
        enable_reranker: Use two-stage reranking for higher precision
        reranker_threshold: Reranker similarity threshold
        reranker_model_path: Custom path to reranker ONNX model
        reranker_top_k: Number of candidates to rerank
        enable_rnn: Enable reciprocal nearest neighbor filtering
        rnn_k: RNN k value (0 = use semantic_search_k)
        enable_margin_gating: Enable margin-based false positive reduction
        margin_threshold: Margin gating threshold
        text_column: Column name containing text to deduplicate
        batch_size: Flush batch size for periodic commits
        key_prefix: Prefix for generated keys (namespace)
        keep_first: Keep first occurrence (True) or last (False)
        return_dedup_info: Include deduplication metadata in output items

    Example:
        >>> config = DedupConfig(
        ...     mode=DedupMode.SEMANTIC,
        ...     semantic_threshold=0.9,
        ...     text_column="content",
        ... )
    """

    # Core settings
    mode: DedupMode = DedupMode.EXACT
    strategy: DedupStrategy = DedupStrategy.DYNAMIC

    # Prestige store settings
    store_path: Optional[Path] = None
    persist_store: bool = False

    # Semantic mode settings
    semantic_threshold: float = 0.85
    semantic_model_type: str = "bge-small"
    semantic_model_path: Optional[Path] = None
    semantic_device: str = "auto"  # "auto", "cpu", "gpu"
    semantic_num_threads: int = 0  # 0 = all cores
    semantic_search_k: int = 50  # k for kNN search

    # Reranker options (two-stage retrieval)
    enable_reranker: bool = False
    reranker_threshold: float = 0.8
    reranker_model_path: Optional[Path] = None
    reranker_top_k: int = 100
    reranker_batch_size: int = 8

    # Reciprocal nearest neighbor
    enable_rnn: bool = False
    rnn_k: int = 0  # 0 = use semantic_search_k

    # Margin gating
    enable_margin_gating: bool = False
    margin_threshold: float = 0.05

    # Processing settings
    text_column: str = "text"
    batch_size: int = 100
    key_prefix: str = ""
    keep_first: bool = True
    return_dedup_info: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError(
                f"semantic_threshold must be between 0.0 and 1.0, got {self.semantic_threshold}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class CrossDatasetConfig:
    """Configuration for cross-dataset deduplication and contamination detection.

    Attributes:
        reference_store_path: Path to store containing reference data
        build_reference: Whether to build reference index or use existing
        detect_contamination: Enable contamination detection mode
        contamination_threshold: Threshold for contamination (typically higher than dedup)

    Example:
        >>> config = CrossDatasetConfig(
        ...     reference_store_path=Path("./test_index"),
        ...     contamination_threshold=0.95,
        ... )
    """

    reference_store_path: Optional[Path] = None
    build_reference: bool = True
    detect_contamination: bool = True
    contamination_threshold: float = 0.95  # Higher threshold for contamination


@dataclass
class CacheConfig:
    """Configuration for static deduplication caching.

    Attributes:
        cache_dir: Directory for cached deduplicated datasets
        cache_key: Custom cache key (None = auto-generate from config)
        force_reprocess: Ignore cache and always reprocess
        save_metrics: Save metrics alongside cached data

    Example:
        >>> config = CacheConfig(
        ...     cache_dir=Path("~/.cache/prestige/dedup"),
        ...     force_reprocess=False,
        ... )
    """

    cache_dir: Optional[Path] = None
    cache_key: Optional[str] = None
    force_reprocess: bool = False
    save_metrics: bool = True

    def __post_init__(self):
        """Set default cache directory if not specified."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "prestige" / "dedup"
