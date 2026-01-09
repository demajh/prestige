# Semantic Deduplication

Semantic mode uses neural network embeddings to deduplicate semantically similar content, even when the exact bytes differ.

## How It Works

1. Each value is converted to a 384-dimensional embedding vector using an ONNX model
2. New values are compared against existing embeddings using cosine similarity
3. If similarity >= threshold, the existing object is reused (semantic match)
4. Otherwise, a new object is created and added to the vector index

## Enabling Semantic Mode

```cpp
prestige::Options opt;
opt.dedup_mode = prestige::DedupMode::kSemantic;
opt.semantic_model_path = "./models/model.onnx";  // Required
opt.semantic_model_type = prestige::SemanticModel::kBGESmall;
opt.semantic_threshold = 0.85f;  // Required: cosine similarity threshold [0.0, 1.0]

std::unique_ptr<prestige::Store> db;
prestige::Store::Open("./my_semantic_db", &db, opt);
```

## Obtaining an ONNX Model

Download a pre-exported model and vocabulary from Hugging Face:

```bash
mkdir -p models && cd models

# Download BGE-small-en-v1.5 (recommended)
curl -LO https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model.onnx
curl -LO https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/vocab.txt

# Or download all-MiniLM-L6-v2
curl -LO https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -LO https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/vocab.txt
```

**Important:** The `vocab.txt` file must be in the same directory as `model.onnx`. Prestige auto-detects the vocabulary file.

## Supported Models

| Model | Enum Value | Dimensions | Notes |
|-------|------------|------------|-------|
| all-MiniLM-L6-v2 | `kMiniLM` | 384 | Fast, good general-purpose |
| BGE-small-en-v1.5 | `kBGESmall` | 384 | Better quality, recommended |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `semantic_model_path` | (required) | Path to ONNX model file |
| `semantic_model_type` | `kMiniLM` | Model type: `kMiniLM` or `kBGESmall` |
| `semantic_threshold` | (required) | Cosine similarity threshold [0.0, 1.0] |
| `semantic_max_text_bytes` | 8192 | Max text bytes to embed (longer texts truncated) |
| `semantic_search_k` | 10 | Number of nearest neighbors to check |
| `semantic_index_save_interval` | 1000 | Auto-save index every N inserts (0 = disabled) |

### HNSW Index Tuning

| Option | Default | Description |
|--------|---------|-------------|
| `hnsw_m` | 16 | Max connections per node (higher = better recall, more memory) |
| `hnsw_ef_construction` | 200 | Build-time search depth (higher = better index quality) |
| `hnsw_ef_search` | 50 | Query-time search depth (higher = better recall, slower) |

## Choosing a Threshold

The `semantic_threshold` controls how similar two values must be to deduplicate:

| Threshold | Behavior |
|-----------|----------|
| 0.95+ | Very strict - only near-identical paraphrases |
| 0.85-0.95 | Balanced - good for most use cases |
| 0.70-0.85 | Loose - groups related content |
| <0.70 | Very loose - high false positive risk |

Start with 0.85 and adjust based on your data.

## Example

```cpp
db->Put("key1", "The quick brown fox jumps over the lazy dog.");
db->Put("key2", "A fast brown fox leaps above a sleepy dog.");  // Semantic match!

// Both keys point to the same object (if similarity >= threshold)
```

## Storage Overhead

Semantic mode adds:
- 1.5KB per unique value for the embedding vector (384 floats)
- External HNSW index file (`db_path.vec_index`)

## Build Requirements

Semantic mode requires building with `PRESTIGE_ENABLE_SEMANTIC=ON`:

```bash
cmake .. -DPRESTIGE_ENABLE_SEMANTIC=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXRUNTIME_LIBRARY=/path/to/libonnxruntime.so
```

## Two-Stage Retrieval with Reranker

For higher accuracy semantic deduplication, prestige supports **two-stage retrieval** with BGE-reranker-v2-m3:

1. **Stage 1**: Fast embedding search retrieves 100-200 candidates
2. **Stage 2**: Cross-encoder reranker scores candidates for precise matching

### Enabling Reranker

```cpp
prestige::Options opt;
opt.dedup_mode = prestige::DedupMode::kSemantic;
opt.semantic_model_path = "./models/bge-small.onnx";
opt.semantic_threshold = 0.75f;  // Lower for initial retrieval

// Enable reranker
opt.semantic_reranker_enabled = true;
opt.semantic_reranker_model_path = "./models/bge-reranker-v2-m3.onnx";
opt.semantic_reranker_threshold = 0.8f;  // Higher for final decision
opt.semantic_reranker_top_k = 100;  // Candidates to rerank

std::unique_ptr<prestige::Store> db;
prestige::Store::Open("./my_semantic_db", &db, opt);
```

### Reranker Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `semantic_reranker_enabled` | false | Enable two-stage retrieval |
| `semantic_reranker_model_path` | (required) | Path to BGE reranker ONNX model |
| `semantic_reranker_threshold` | 0.7 | Reranker score threshold [0.0, 1.0] |
| `semantic_reranker_top_k` | 100 | Number of candidates to rerank |
| `semantic_reranker_batch_size` | 8 | Batch size for efficiency |
| `semantic_reranker_fallback` | true | Fall back to embeddings if reranker fails |

### Performance Characteristics

**Accuracy Improvements** (typical):
- **Precision**: +3-8%  
- **Recall**: +5-12%
- **F1 Score**: +4-10%

**Latency Overhead**:
- **Small datasets**: +50-100%
- **Large datasets**: +30-80%
- **Batch processing reduces per-item overhead**

### Downloading BGE Reranker

```bash
mkdir -p models/bge-reranker-v2-m3
cd models/bge-reranker-v2-m3
curl -LO https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/onnx/model.onnx
curl -LO https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/vocab.txt
```

### When to Use Reranker

**Recommended for:**
- High-accuracy use cases where precision matters
- Paraphrase detection and content similarity
- Offline batch processing
- Small to medium throughput (<10k operations/day)

**Not recommended for:**
- High-throughput, latency-sensitive applications
- Simple exact-match scenarios  
- Very large scale real-time systems
