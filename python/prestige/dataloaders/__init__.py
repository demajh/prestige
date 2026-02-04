"""Deduplicated dataloaders for ML training with prestige.

This package provides PyTorch and HuggingFace integrations for training on
deduplicated data. It supports both exact (SHA-256) and semantic (embedding)
deduplication, with static (pre-filter) and dynamic (on-the-fly) strategies.

Key Features:
- PyTorch Dataset and DataLoader wrappers with automatic deduplication
- HuggingFace datasets integration
- Train/test contamination detection
- Streaming support for large datasets
- Rich metrics and observability

Quick Start:

    # PyTorch integration
    >>> from prestige.dataloaders import DedupDataset, DedupConfig, DedupMode
    >>> config = DedupConfig(mode=DedupMode.SEMANTIC, semantic_threshold=0.9)
    >>> dataset = DedupDataset(train_data, config)
    >>> loader = DataLoader(dataset, batch_size=32)

    # HuggingFace one-liner
    >>> from prestige.dataloaders import deduplicate_dataset
    >>> deduped = deduplicate_dataset(hf_dataset, mode="semantic", threshold=0.85)

    # Contamination detection
    >>> from prestige.dataloaders import detect_train_test_leakage
    >>> results = detect_train_test_leakage(train_ds, test_ds)
    >>> print(f"Contamination rate: {results['contamination_rate']:.2%}")

See individual modules for more details and examples.
"""

# Configuration
from .config import (
    DedupConfig,
    DedupMode,
    DedupStrategy,
    CrossDatasetConfig,
    CacheConfig,
)

# Metrics
from .metrics import DedupMetrics

# Store wrapper
from .dedup_store import DedupStore, create_dedup_store

# Utilities
from .utils import (
    extract_text,
    make_key,
    hash_text,
    hash_config,
    batched,
    create_text_extractor,
    estimate_memory_usage,
    validate_text_column,
)

# PyTorch integration (optional - requires torch)
try:
    from .datasets import (
        DedupDataset,
        DedupDatasetView,
        LazyDedupDataset,
        create_dedup_dataloader,
        collate_with_dedup_info,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DedupDataset = None
    DedupDatasetView = None
    LazyDedupDataset = None
    create_dedup_dataloader = None
    collate_with_dedup_info = None

# HuggingFace integration (optional - requires datasets)
try:
    from .hf_integration import (
        HuggingFaceDeduplicator,
        StaticDedupPipeline,
        deduplicate_dataset,
        deduplicate_and_cache,
    )

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HuggingFaceDeduplicator = None
    StaticDedupPipeline = None
    deduplicate_dataset = None
    deduplicate_and_cache = None

# Contamination detection
from .contamination import (
    ContaminationDetector,
    CrossDatasetDeduplicator,
    detect_train_test_leakage,
    filter_train_test_leakage,
)

# Streaming support (optional - requires torch)
try:
    from .streaming import (
        StreamingDedupDataset,
        DynamicDedupIterator,
        ChunkedDedupProcessor,
        create_streaming_dataloader,
        deduplicate_iterator,
    )
except ImportError:
    StreamingDedupDataset = None
    DynamicDedupIterator = None
    ChunkedDedupProcessor = None
    create_streaming_dataloader = None
    deduplicate_iterator = None


__all__ = [
    # Configuration
    "DedupConfig",
    "DedupMode",
    "DedupStrategy",
    "CrossDatasetConfig",
    "CacheConfig",
    # Metrics
    "DedupMetrics",
    # Store
    "DedupStore",
    "create_dedup_store",
    # Utilities
    "extract_text",
    "make_key",
    "hash_text",
    "hash_config",
    "batched",
    "create_text_extractor",
    "estimate_memory_usage",
    "validate_text_column",
    # PyTorch (conditional)
    "DedupDataset",
    "DedupDatasetView",
    "LazyDedupDataset",
    "create_dedup_dataloader",
    "collate_with_dedup_info",
    # HuggingFace (conditional)
    "HuggingFaceDeduplicator",
    "StaticDedupPipeline",
    "deduplicate_dataset",
    "deduplicate_and_cache",
    # Contamination
    "ContaminationDetector",
    "CrossDatasetDeduplicator",
    "detect_train_test_leakage",
    "filter_train_test_leakage",
    # Streaming (conditional)
    "StreamingDedupDataset",
    "DynamicDedupIterator",
    "ChunkedDedupProcessor",
    "create_streaming_dataloader",
    "deduplicate_iterator",
    # Feature flags
    "TORCH_AVAILABLE",
    "HF_AVAILABLE",
]
