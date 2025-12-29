# Semantic Deduplication Benchmark Harness

A comprehensive benchmark suite for evaluating Prestige's semantic deduplication performance against labeled datasets.

## Overview

This benchmark harness measures how accurately Prestige's semantic deduplication identifies duplicate text pairs across multiple datasets:

- **QQP** (Quora Question Pairs): 400k question pairs
- **MRPC** (Microsoft Research Paraphrase Corpus): 5.8k sentence pairs
- **STS-B** (Semantic Textual Similarity Benchmark): Graded similarity scores
- **PAWS** (Paraphrase Adversaries from Word Scrambling): 108k adversarial pairs

## Quick Start

### 1. Install Dependencies

```bash
# Install Python bindings
cd python
pip install -e .

# Install benchmark dependencies
pip install -r benchmarks/semantic_dedup/requirements.txt
```

### 2. Download Embedding Model

```bash
python -m benchmarks.semantic_dedup.models bge-small
```

### 3. Run a Quick Benchmark

```bash
# Run on MRPC dataset with default thresholds
cd benchmarks/semantic_dedup
python cli.py run --datasets mrpc --output ./results
```

## CLI Commands

### Run Benchmarks

```bash
python cli.py run [OPTIONS]
```

**Options:**
- `--datasets, -d`: Comma-separated dataset names (default: `mrpc`)
- `--thresholds, -t`: Comma-separated thresholds (default: `0.85,0.90,0.95`)
- `--output, -o`: Output directory (default: `./results`)
- `--cache-dir`: Cache directory for datasets/models
- `--model`: Embedding model name (default: `bge-small`)
- `--batch-size`: Batch size for flushes (default: `100`)
- `--quiet, -q`: Suppress progress output
- `--json-only`: Skip HTML report generation

**Examples:**

```bash
# Run multiple datasets
python cli.py run --datasets mrpc,qqp --thresholds 0.85,0.90,0.95

# Quick test with single threshold
python cli.py run --datasets mrpc --thresholds 0.90 --quiet

# Full evaluation
python cli.py run --datasets mrpc,qqp,stsb,paws
```

### Compare Against Baseline

```bash
python cli.py compare CURRENT BASELINE [OPTIONS]
```

**Examples:**

```bash
# Compare results
python cli.py compare results/current.json results/baseline.json

# Save comparison
python cli.py compare current.json baseline.json --output comparison.json

# Custom regression threshold
python cli.py compare current.json baseline.json --threshold 3.0
```

### Generate HTML Report

```bash
python cli.py report INPUT_JSON [OPTIONS]
```

**Examples:**

```bash
# Generate HTML from JSON
python cli.py report results/mrpc_results.json

# Custom output path
python cli.py report results/mrpc_results.json --output report.html
```

### List Available Datasets

```bash
python cli.py list-datasets
```

## Python API

### Quick Benchmark

```python
from benchmarks.semantic_dedup.runner import run_quick_benchmark

# Run benchmark on MRPC
results = run_quick_benchmark(
    dataset_name="mrpc",
    thresholds=[0.85, 0.90, 0.95],
    verbose=True
)

print(f"Best F1: {results['f1_score']:.3f}")
print(f"Best threshold: {results['threshold']:.2f}")
```

### Custom Benchmark

```python
from pathlib import Path
from benchmarks.semantic_dedup.datasets import get_dataset_config
from benchmarks.semantic_dedup.runner import SemanticDedupBenchmark, BenchmarkConfig
from benchmarks.semantic_dedup.report import ReportGenerator

# Configure benchmark
dataset_config = get_dataset_config("mrpc")
config = BenchmarkConfig(
    dataset_config=dataset_config,
    thresholds=[0.85, 0.90, 0.95],
    cache_dir=Path("~/.cache/prestige/benchmarks").expanduser(),
    embedding_model="bge-small",
    verbose=True
)

# Run benchmark
benchmark = SemanticDedupBenchmark(config)
results = benchmark.run()

# Generate reports
report_gen = ReportGenerator("mrpc", results)
report_gen.save_json(Path("results.json"))
report_gen.save_html(Path("report.html"))

# Get best result
best = results.get_best_f1()
print(f"Best F1: {best['f1_score']:.4f} at threshold {best['threshold']:.2f}")
```

## Metrics

The benchmark computes the following metrics:

### Classification Metrics
- **Precision**: TP / (TP + FP) - How many predicted duplicates are actually duplicates
- **Recall**: TP / (TP + FN) - How many actual duplicates are detected
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total - Overall correctness

### Performance Metrics
- **Latency**: P50, P95, P99 percentiles (milliseconds)
- **Throughput**: Documents per second
- **Storage Efficiency**: Total bytes, deduplication ratio

### Aggregate Metrics
- **ROC AUC**: Area under the ROC curve (FPR vs TPR)
- **PR AUC**: Area under the Precision-Recall curve

## Output Format

### JSON Report

```json
{
  "timestamp": "2025-12-28T20:30:00Z",
  "dataset_name": "mrpc",
  "thresholds": [0.85, 0.90, 0.95],
  "results": [
    {
      "threshold": 0.90,
      "precision": 0.9234,
      "recall": 0.8765,
      "f1_score": 0.8993,
      "accuracy": 0.9145,
      "tp": 1234,
      "fp": 102,
      "tn": 3456,
      "fn": 178,
      "dedup_ratio": 1.85,
      "latency_p50_ms": 2.34,
      "latency_p95_ms": 5.67,
      "latency_p99_ms": 8.92,
      "storage_bytes": 12345678,
      "unique_objects": 2345,
      "total_keys": 5678
    }
  ],
  "best_f1": {...},
  "roc_auc": 0.9456,
  "pr_auc": 0.9234,
  "summary": {...}
}
```

### HTML Report

Interactive HTML report with:
- Summary dashboard with key metrics
- Results table by threshold
- Confusion matrix
- Styled for readability

## Datasets

### MRPC (Microsoft Research Paraphrase Corpus)

- **Source**: GLUE benchmark
- **Size**: ~5,800 sentence pairs
- **Domain**: News articles
- **Use case**: Quick iteration, smoke testing
- **Runtime**: ~5 minutes

### QQP (Quora Question Pairs)

- **Source**: GLUE benchmark
- **Size**: ~400,000 question pairs
- **Domain**: Community Q&A
- **Use case**: Large-scale evaluation
- **Runtime**: ~60-90 minutes

### STS-B (Semantic Textual Similarity Benchmark)

- **Source**: GLUE benchmark
- **Size**: ~8,600 sentence pairs
- **Domain**: News, captions, forums
- **Labels**: Graded 0-5 (threshold ≥4.0 for duplicates)
- **Use case**: Continuous similarity evaluation

### PAWS (Paraphrase Adversaries)

- **Source**: Google Research
- **Size**: ~108,000 pairs
- **Domain**: Wikipedia, QQP
- **Challenge**: High lexical overlap, low semantic similarity
- **Use case**: Adversarial robustness testing

## Model Management

### Download Models

```bash
# Download BGE-small (default, ~133 MB)
python -m benchmarks.semantic_dedup.models bge-small

# Download MiniLM (~90 MB)
python -m benchmarks.semantic_dedup.models minilm

# List available models
python -m benchmarks.semantic_dedup.models --list

# Verify cached model
python -m benchmarks.semantic_dedup.models bge-small --verify

# Clear cache
python -m benchmarks.semantic_dedup.models --clear
```

### Model Registry

| Model | Description | Size | Dimensions |
|-------|-------------|------|------------|
| `bge-small` | BGE-small English v1.5 (default) | 133 MB | 384 |
| `minilm` | MiniLM-L6-v2 | 90 MB | 384 |

## CI Integration

The benchmark suite runs automatically in GitHub Actions:

- **On Pull Requests**: MRPC only (~5 min)
- **Weekly**: Full suite (MRPC, QQP, STS-B, PAWS)
- **Manual**: Custom dataset/threshold selection

### Workflow

1. Build Prestige with `PRESTIGE_ENABLE_SEMANTIC=ON`
2. Install Python dependencies
3. Download embedding models
4. Run benchmarks
5. Compare against baseline (PRs only)
6. Comment PR with results
7. Upload artifacts (JSON + HTML)

### Regression Detection

Regressions are flagged if:
- F1 score drops >5%
- ROC AUC drops >5%
- P95 latency increases >5%

## Troubleshooting

### ImportError: prestige module not found

```bash
cd python
pip install -e .
```

### Model not found

```bash
python -m benchmarks.semantic_dedup.models bge-small
```

### ONNX Runtime not found

```bash
# Ubuntu/Debian
sudo apt-get install libonnxruntime-dev

# macOS
brew install onnxruntime
```

### Dataset download fails

Datasets are downloaded from HuggingFace. If download fails:

1. Check internet connection
2. Try again (downloads are cached)
3. Manually download and cache:

```python
from datasets import load_dataset
dataset = load_dataset("glue", "mrpc", split="train")
```

## Performance Tips

### Fast Iteration

Use MRPC with a single threshold:

```bash
python cli.py run --datasets mrpc --thresholds 0.90 --quiet
```

### Parallel Execution

Run multiple datasets in separate processes:

```bash
python cli.py run --datasets mrpc --output results/mrpc &
python cli.py run --datasets qqp --output results/qqp &
wait
```

### Cache Management

Dataset and model caches are stored in:
- Datasets: `~/.cache/prestige/benchmarks/`
- Models: `~/.cache/prestige/models/`

Clear caches to force re-download:

```bash
rm -rf ~/.cache/prestige/benchmarks/
python -m benchmarks.semantic_dedup.models --clear
```

## Architecture

### Components

```
benchmarks/semantic_dedup/
├── __init__.py          # Package initialization
├── cli.py               # Click-based CLI
├── datasets.py          # Dataset loading and caching
├── runner.py            # Benchmark execution engine
├── metrics.py           # Precision/recall/F1 calculation
├── report.py            # JSON/HTML report generation
├── models.py            # Model download and management
└── requirements.txt     # Python dependencies
```

### Workflow

1. **Dataset Loading**: Download from HuggingFace, cache as Parquet
2. **Model Setup**: Download ONNX model if not cached
3. **Store Creation**: Create Prestige store with semantic dedup enabled
4. **Pair Processing**: For each text pair:
   - Put both texts with unique keys
   - Get object IDs to check deduplication
   - Compare with ground truth label
   - Update confusion matrix
5. **Metrics Calculation**: Compute precision/recall/F1/ROC/PR
6. **Report Generation**: Save JSON and HTML reports

## Contributing

### Adding a New Dataset

1. Add configuration to `datasets.py`:

```python
DATASETS["my_dataset"] = DatasetConfig(
    name="my_dataset",
    source="huggingface/repo",
    text_columns=["text1", "text2"],
    label_column="label",
    positive_label=1,
)
```

2. Update documentation
3. Add to CI workflow (optional)

### Adding a New Model

1. Add to `models.py` MODEL_REGISTRY:

```python
"my_model": {
    "model_file": "model.onnx",
    "vocab_file": "vocab.txt",
    "hf_repo": "org/model-name",
    "model_url": "https://...",
    "vocab_url": "https://...",
    "model_size_mb": 100,
    "description": "Model description",
}
```

2. Test download and benchmarking
3. Update documentation

## References

- [Prestige Documentation](../../README.md)
- [Python Bindings Guide](../../docs/python-bindings.md)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [PAWS Dataset](https://github.com/google-research-datasets/paws)
- [BGE Models](https://huggingface.co/BAAI/bge-small-en-v1.5)
