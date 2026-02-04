# Deduplicated Dataloaders for ML Training

The `prestige.dataloaders` package provides PyTorch and HuggingFace integrations for training on deduplicated data. It leverages Prestige's deduplication capabilities to remove duplicate or near-duplicate examples from training datasets.

## Features

- **PyTorch Integration**: `DedupDataset` wraps any sequence as a `torch.utils.data.Dataset`
- **HuggingFace Integration**: Works with `datasets.Dataset` and `DatasetDict`
- **Multiple Dedup Modes**: Exact (SHA-256) and semantic (embedding-based)
- **Static & Dynamic**: Pre-filter once or deduplicate on-the-fly
- **Contamination Detection**: Detect train/test leakage
- **Streaming Support**: Handle datasets too large for memory
- **Rich Metrics**: Track dedup ratios, throughput, and removed items

## Installation

The dataloaders package requires the base `prestige` package plus optional dependencies:

```bash
# Install prestige
pip install prestige

# Install optional dependencies for dataloaders
pip install torch datasets
```

## Quick Start

### PyTorch Dataset

```python
from prestige.dataloaders import DedupDataset, DedupConfig, DedupMode
from torch.utils.data import DataLoader

# Your training data
train_data = [
    {"text": "Example one", "label": 0},
    {"text": "Example two", "label": 1},
    {"text": "Example one", "label": 0},  # Duplicate!
    # ...
]

# Configure deduplication
config = DedupConfig(
    mode=DedupMode.SEMANTIC,  # or DedupMode.EXACT
    semantic_threshold=0.9,
    text_column="text",
)

# Create deduplicated dataset
dataset = DedupDataset(train_data, config)
print(f"Original: {dataset.original_size}, Deduplicated: {len(dataset)}")

# Use with DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    # Train on deduplicated data
    pass

# Check metrics
metrics = dataset.get_metrics()
print(metrics.summary())
```

### HuggingFace Dataset

```python
from datasets import load_dataset
from prestige.dataloaders import deduplicate_dataset

# Load a HuggingFace dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")

# One-liner deduplication
deduped = deduplicate_dataset(
    ds["train"],
    mode="semantic",
    threshold=0.85,
    text_column="text",
)

print(f"Reduced from {len(ds['train'])} to {len(deduped)} examples")
```

### Train/Test Contamination Detection

```python
from prestige.dataloaders import detect_train_test_leakage

# Check for leakage
results = detect_train_test_leakage(
    train_data,
    test_data,
    mode="semantic",
    threshold=0.95,  # Higher threshold for contamination
    text_column="text",
)

if results["contaminated_count"] > 0:
    print(f"WARNING: {results['contamination_rate']:.2%} contamination detected!")
    print(f"Contaminated indices: {results['contaminated_train_indices']}")
```

## Configuration

### DedupConfig

The main configuration class for deduplication behavior:

```python
from prestige.dataloaders import DedupConfig, DedupMode, DedupStrategy

config = DedupConfig(
    # Core settings
    mode=DedupMode.SEMANTIC,        # EXACT or SEMANTIC
    strategy=DedupStrategy.DYNAMIC,  # STATIC or DYNAMIC

    # Semantic settings
    semantic_threshold=0.85,         # Similarity threshold (0.0-1.0)
    semantic_model_type="bge-small", # Embedding model
    semantic_device="auto",          # "auto", "cpu", "gpu"

    # Processing
    text_column="text",              # Column containing text
    batch_size=100,                  # Flush frequency

    # Advanced semantic options
    enable_reranker=False,           # Two-stage reranking
    reranker_threshold=0.8,
    enable_rnn=False,                # Reciprocal nearest neighbor
    enable_margin_gating=False,      # Margin-based filtering
)
```

### Deduplication Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `EXACT` | SHA-256 hash matching | Byte-identical duplicates |
| `SEMANTIC` | Embedding similarity | Paraphrases, near-duplicates |

### Semantic Model Types

| Model | Dimensions | Notes |
|-------|------------|-------|
| `minilm` | 384 | Fast, general purpose |
| `bge-small` | 384 | Good quality, recommended |
| `bge-large` | 1024 | Higher quality, slower |
| `e5-large` | 1024 | Instruction-tuned |
| `bge-m3` | 1024 | Multilingual |
| `nomic` | 768 | Long context support |

## PyTorch Integration

### DedupDataset

Wraps sequence data with precomputed deduplication:

```python
from prestige.dataloaders import DedupDataset

dataset = DedupDataset(
    data,                    # List, HuggingFace dataset, etc.
    config,                  # DedupConfig
    transform=my_transform,  # Optional transform
    precompute=True,         # Compute dedup index immediately
    verbose=True,            # Print progress
)

# Properties
len(dataset)               # Number of unique items
dataset.original_size      # Original size before dedup
dataset.dedup_ratio        # original / deduplicated

# Methods
dataset.get_metrics()         # DedupMetrics object
dataset.get_removed_indices() # Indices of removed items
dataset.get_valid_indices()   # Indices of kept items
```

### LazyDedupDataset

For on-the-fly deduplication with metadata:

```python
from prestige.dataloaders import LazyDedupDataset

dataset = LazyDedupDataset(data, config)

for item in dataset:
    if not item["_is_duplicate"]:
        # Process unique item
        print(item["text"])

    # Available metadata:
    # item["_is_duplicate"] - bool
    # item["_object_id"] - bytes
    # item["_original_index"] - int

dataset.close()  # Clean up
```

### Convenience Function

```python
from prestige.dataloaders import create_dedup_dataloader

loader = create_dedup_dataloader(
    data,
    config=config,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)
```

## HuggingFace Integration

### HuggingFaceDeduplicator

Full control over deduplication:

```python
from prestige.dataloaders import HuggingFaceDeduplicator

deduplicator = HuggingFaceDeduplicator(config)

# Deduplicate single dataset
deduped = deduplicator.deduplicate(dataset)

# Deduplicate DatasetDict
deduped_dict = deduplicator.deduplicate(dataset_dict)

# Cross-split deduplication (remove from val/test what's in train)
deduped_dict = deduplicator.deduplicate_across_splits(
    dataset_dict,
    primary_split="train",
)

# Get metrics
metrics = deduplicator.get_metrics()
```

### Static Pipeline with Caching

For large datasets, cache the deduplicated result:

```python
from prestige.dataloaders import StaticDedupPipeline, CacheConfig

cache_config = CacheConfig(
    cache_dir=Path("./dedup_cache"),
    save_metrics=True,
)

pipeline = StaticDedupPipeline(config, cache_config)

# First call: processes and caches
deduped = pipeline.process_and_cache(dataset, "my_dataset_v1")

# Second call: loads from cache instantly
deduped = pipeline.process_and_cache(dataset, "my_dataset_v1")

# Force reprocess
deduped = pipeline.process_and_cache(
    dataset, "my_dataset_v1",
    force_reprocess=True,
)

# Load cached metrics
metrics = pipeline.load_metrics("my_dataset_v1")
```

### Convenience Functions

```python
from prestige.dataloaders import deduplicate_dataset, deduplicate_and_cache

# One-liner
deduped = deduplicate_dataset(
    dataset,
    mode="semantic",
    threshold=0.85,
    text_column="text",
)

# With caching
deduped = deduplicate_and_cache(
    dataset,
    "my_dataset_v1",
    cache_dir=Path("./cache"),
    mode="semantic",
)
```

## Contamination Detection

### ContaminationDetector

For fine-grained control:

```python
from prestige.dataloaders import ContaminationDetector, CrossDatasetConfig

cross_config = CrossDatasetConfig(
    contamination_threshold=0.95,  # Higher = stricter
)

detector = ContaminationDetector(config, cross_config)

# Build index from test set
detector.build_reference_index(test_data, "test_set")

# Check training data
contaminated_indices, metrics = detector.check_contamination(train_data)

# Or filter directly
clean_train = detector.filter_contaminated(train_data)

detector.close()
```

### Convenience Functions

```python
from prestige.dataloaders import (
    detect_train_test_leakage,
    filter_train_test_leakage,
)

# Detect
results = detect_train_test_leakage(train_data, test_data)
print(f"Contamination: {results['contamination_rate']:.2%}")

# Filter
clean_train = filter_train_test_leakage(train_data, test_data)
```

### Cross-Dataset Deduplication

Deduplicate across multiple data sources:

```python
from prestige.dataloaders import CrossDatasetDeduplicator

with CrossDatasetDeduplicator(config) as dedup:
    # Add datasets in order - later ones dedupe against earlier
    dedup.add_dataset(wiki_data, "wiki")
    dedup.add_dataset(books_data, "books")  # Removes wiki duplicates
    dedup.add_dataset(web_data, "web")      # Removes wiki+books duplicates

    # Get filtered datasets
    wiki_clean = dedup.filter_dataset(wiki_data, "wiki")
    books_clean = dedup.filter_dataset(books_data, "books")
    web_clean = dedup.filter_dataset(web_data, "web")
```

## Streaming Support

For large datasets that don't fit in memory:

### StreamingDedupDataset (PyTorch)

```python
from prestige.dataloaders import StreamingDedupDataset
from torch.utils.data import DataLoader

def data_generator():
    for line in open("huge_file.txt"):
        yield {"text": line.strip()}

dataset = StreamingDedupDataset(data_generator, config)
loader = DataLoader(dataset, batch_size=32, num_workers=4)

for batch in loader:
    # Process deduplicated batches
    pass
```

### DynamicDedupIterator

Wrap any iterator:

```python
from prestige.dataloaders import DynamicDedupIterator

iterator = DynamicDedupIterator(data_iter, config)

for item in iterator:
    # Process deduplicated items
    pass

print(iterator.get_metrics().summary())
```

### ChunkedDedupProcessor

Process in chunks:

```python
from prestige.dataloaders import ChunkedDedupProcessor

processor = ChunkedDedupProcessor(config, chunk_size=10000)

for chunk in processor.process_chunks(data_iterator):
    # Save/process chunk of deduplicated items
    save_to_disk(chunk)
```

### Convenience Function

```python
from prestige.dataloaders import deduplicate_iterator

for item in deduplicate_iterator(data_iter, mode="semantic"):
    print(item["text"])
```

## Metrics

All operations collect metrics via `DedupMetrics`:

```python
metrics = dataset.get_metrics()

# Counts
metrics.total_seen          # Items processed
metrics.unique_kept         # Unique items
metrics.duplicates_removed  # Removed duplicates

# Rates
metrics.dedup_ratio         # total / unique
metrics.removal_rate        # removed / total
metrics.keep_rate           # 1 - removal_rate

# Timing
metrics.elapsed_seconds
metrics.items_per_second

# Contamination
metrics.contaminated_count
metrics.contamination_rate

# Serialization
metrics.to_dict()           # For JSON/logging
metrics.summary()           # Human-readable string
```

## Advanced: Semantic Options

### Two-Stage Reranking

Use a cross-encoder for higher precision:

```python
config = DedupConfig(
    mode=DedupMode.SEMANTIC,
    semantic_threshold=0.75,  # Lower initial threshold
    enable_reranker=True,
    reranker_threshold=0.85,  # Final threshold
    reranker_top_k=100,       # Candidates to rerank
)
```

### Reciprocal Nearest Neighbor (RNN)

Reduce false positives from "hub" documents:

```python
config = DedupConfig(
    mode=DedupMode.SEMANTIC,
    enable_rnn=True,
    rnn_k=10,  # 0 = auto
)
```

### Margin Gating

Additional false-positive reduction:

```python
config = DedupConfig(
    mode=DedupMode.SEMANTIC,
    enable_margin_gating=True,
    margin_threshold=0.05,
)
```

## Memory Estimation

Estimate memory requirements:

```python
from prestige.dataloaders import estimate_memory_usage

estimate = estimate_memory_usage(
    num_items=1_000_000,
    avg_text_length=500,
    mode="semantic",
    embedding_dim=384,
)

print(f"Estimated memory: {estimate['total_gb']:.1f} GB")
```

## Best Practices

1. **Choose the right mode**:
   - Use `EXACT` for byte-identical duplicates (fast)
   - Use `SEMANTIC` for paraphrases and near-duplicates

2. **Tune thresholds**:
   - Training dedup: 0.85-0.95
   - Contamination detection: 0.95-0.99

3. **Use caching for large datasets**:
   - `StaticDedupPipeline` caches results to disk
   - Subsequent runs load instantly

4. **Stream large datasets**:
   - Use `StreamingDedupDataset` for data that doesn't fit in memory
   - Process in chunks with `ChunkedDedupProcessor`

5. **Check for contamination**:
   - Always run `detect_train_test_leakage` before training
   - Use higher threshold (0.95+) for contamination

## API Reference

### Classes

- `DedupConfig` - Configuration for deduplication
- `DedupMetrics` - Metrics collection
- `DedupStore` - Low-level store wrapper
- `DedupDataset` - PyTorch Dataset wrapper
- `LazyDedupDataset` - On-the-fly deduplication
- `HuggingFaceDeduplicator` - HuggingFace integration
- `StaticDedupPipeline` - Caching pipeline
- `ContaminationDetector` - Leakage detection
- `CrossDatasetDeduplicator` - Multi-dataset dedup
- `StreamingDedupDataset` - Streaming support
- `DynamicDedupIterator` - Iterator wrapper
- `ChunkedDedupProcessor` - Chunked processing

### Functions

- `deduplicate_dataset()` - One-liner HuggingFace dedup
- `deduplicate_and_cache()` - Dedup with caching
- `detect_train_test_leakage()` - Contamination detection
- `filter_train_test_leakage()` - Remove contamination
- `create_dedup_dataloader()` - PyTorch DataLoader factory
- `create_streaming_dataloader()` - Streaming DataLoader
- `deduplicate_iterator()` - Iterator dedup wrapper

### Enums

- `DedupMode` - EXACT, SEMANTIC
- `DedupStrategy` - STATIC, DYNAMIC
