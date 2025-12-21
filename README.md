# prestige unique value store

**What is a unique value store?**

A unique value store is a queryable collection of unique values.  In order to maintain the 'unique'-ness of the collection, there must be some mechanism to remove duplicates.

prestige implements this collection as a key-value store.  It is effectively a wrapper around RocksDB, performing deduplication of elements after Put() operations.

**Two deduplication modes:**

- **Exact mode** (default): Uses SHA-256 hashing for byte-identical deduplication
- **Semantic mode**: Uses neural network embeddings to deduplicate semantically similar content

**Why the name 'prestige'?**

In the movie, *The Prestige*, Hugh Jackman's character "deduplicates" himself after every time he performs his version of *The Transported Man* trick:
https://www.youtube.com/shorts/rfmHhWYKslU

The **prestige unique value store** is a less gruesome, but no less effective way to **deduplicate your data** and maintain a **queryable store** with a **single physical copy** of each unique value.

---

### prestige vs a vanilla RocksDB KV store

In vanilla RocksDB, each `Put(key, value)` stores `value` under `key` (so identical values are duplicated across keys). Prestige introduces **indirection**:

- User keys map to **object IDs**
- Object IDs map to the **actual value bytes**
- A **dedup index** maps a **content hash** to an object ID so identical values are stored once

---

## High-level design

Prestige implements a content-deduplicated value store using multiple RocksDB Column Families. The store supports two mutually exclusive deduplication modes:

### Exact Mode (default)

Uses SHA-256 content hashing for byte-identical deduplication:

1. **User mapping**: `user_key -> object_id`
2. **Object store**: `object_id -> value_bytes`
3. **Dedup index**: `sha256(value_bytes) -> object_id`
4. **Refcount**: `object_id -> uint64_le`
5. **Object meta**: `object_id -> sha256(value_bytes)` (used for safe GC / index cleanup)

### Semantic Mode

Uses neural network embeddings for semantic similarity deduplication:

1. **User mapping**: `user_key -> object_id`
2. **Object store**: `object_id -> value_bytes`
3. **Embeddings**: `object_id -> embedding_vector` (384 floats)
4. **Refcount**: `object_id -> uint64_le`
5. **Vector index**: External HNSW index file for similarity search

### Column families and schema

**Exact mode:**

| Column family | Purpose | Key | Value |
|---|---|---|---|
| `prestige_user_kv` | User key to object reference | `user_key` | `object_id` (16 bytes) |
| `prestige_object_store` | Actual stored values | `object_id` (16 bytes) | raw `value_bytes` |
| `prestige_dedup_index` | Dedup lookup | `sha256(value_bytes)` (32 bytes) | `object_id` (16 bytes) |
| `prestige_refcount` | Reference counting for GC | `object_id` | `uint64_le` |
| `prestige_object_meta` | Reverse mapping for GC | `object_id` | `sha256(value_bytes)` |

**Semantic mode** (adds/replaces):

| Column family | Purpose | Key | Value |
|---|---|---|---|
| `prestige_embeddings` | Embedding vectors | `object_id` (16 bytes) | 384 floats (1536 bytes) |
| External file: `db_path.vec_index` | HNSW vector index | - | hnswlib format |

### Identifiers

- **Dedup key (exact mode)**: `SHA-256(value_bytes)` (exact / byte-identical dedup).
- **Dedup key (semantic mode)**: Embedding vector from ONNX model (semantic similarity dedup).
- **Object ID**: random 128-bit ID (16 bytes).
  This avoids using the hash directly as the storage key, but still allows dedup via the index.

---

## Operation semantics

### Put(user_key, value_bytes)

#### Exact Mode

1. Compute `digest = sha256(value_bytes)`.
2. Start a RocksDB **transaction** (TransactionDB).
3. **Lock**:
   - the `user_key` row in `prestige_user_kv` (detect overwrite)
   - the `digest` row in `prestige_dedup_index` (prevent double-insert races)
4. If `digest` already exists in `prestige_dedup_index`:
   - reuse its `object_id`
5. Otherwise (first time this value is seen):
   - allocate a new 16-byte `object_id`
   - write:
     - `prestige_object_store[object_id] = value_bytes`
     - `prestige_object_meta[object_id] = digest`
     - `prestige_dedup_index[digest] = object_id`
     - `prestige_refcount[object_id] = 0`
6. Write/overwrite:
   - `prestige_user_kv[user_key] = object_id`
7. Increment refcount for the new `object_id`.
8. If overwriting an existing `user_key`:
   - decrement refcount of the old `object_id`
   - if refcount reaches 0 and `enable_gc=true`, delete the old object and clean up its indices.
9. Commit.

#### Semantic Mode

1. Compute `embedding = onnx_model(value_bytes)` (384-dimensional vector).
2. Search vector index for `k` nearest neighbors.
3. For each candidate, compute cosine similarity from L2 distance.
4. If any candidate has `similarity >= threshold`:
   - reuse its `object_id` (semantic match found)
5. Otherwise (semantically unique value):
   - allocate a new 16-byte `object_id`
   - write:
     - `prestige_object_store[object_id] = value_bytes`
     - `prestige_embeddings[object_id] = embedding`
     - `prestige_refcount[object_id] = 0`
   - add embedding to vector index
6. Start transaction, lock `user_key`, update mappings and refcounts.
7. Commit transaction.
8. Periodically save vector index to disk.

**Atomicity and concurrency:** the whole operation is a single TransactionDB commit. Conflicts return retryable statuses; the implementation retries up to `Options::max_retries`.

### Get(user_key) -> value_bytes

Read path is intentionally simple:

1. `object_id = prestige_user_kv[user_key]`
2. `value_bytes = prestige_object_store[object_id]`

This is a “RocksDB-like” API: users never see object IDs.

### Delete(user_key)

1. Transactionally remove `prestige_user_kv[user_key]`
2. Decrement the referenced object’s refcount
3. If refcount reaches 0 and `enable_gc=true`:
   - delete `prestige_object_store[object_id]`
   - delete `prestige_refcount[object_id]`
   - delete `prestige_object_meta[object_id]`
   - delete `prestige_dedup_index[digest]` **only if** it still points to `object_id`
     (protects against races where a digest was remapped)

---

## Build

### Dependencies

**Required:**
- A RocksDB build that includes **TransactionDB** support (`rocksdb/utilities/transaction_db.h`).

**Optional (for semantic mode):**
- ONNX Runtime (C++ API)
- hnswlib (fetched automatically via CMake)

### Build with CMake

**Basic build (exact mode only):**

```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

**With semantic deduplication:**

```bash
mkdir -p build
cmake -S . -B build -DPRESTIGE_ENABLE_SEMANTIC=ON
cmake --build build -j
```

Outputs:
- `prestige_uvs` (library)
- `prestige_example_basic` (example program)
- `prestige_example_semantic` (semantic example, if enabled)
- `prestige_cli` (CLI)

### Obtaining an ONNX Model

For semantic mode, you need an ONNX-exported sentence transformer model:

```bash
pip install optimum[onnxruntime]

# Export all-MiniLM-L6-v2 (recommended)
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 \
    --task feature-extraction ./minilm_onnx/

# Or export BGE-small-en-v1.5
optimum-cli export onnx --model BAAI/bge-small-en-v1.5 \
    --task feature-extraction ./bge_small_onnx/
```

---

## Usage

### Example programs

**Exact dedup example:**

```bash
./build/prestige_example_basic
```

**Semantic dedup example:**

```bash
./build/prestige_example_semantic ./minilm_onnx/model.onnx
```

### CLI

```bash
# Put values (dedup happens automatically)
./build/prestige_cli ./prestige_db put k1 HELLO
./build/prestige_cli ./prestige_db put k2 HELLO

# Read (returns raw bytes)
./build/prestige_cli ./prestige_db get k2

# Delete keys (GC happens when refcount hits 0)
./build/prestige_cli ./prestige_db del k1
./build/prestige_cli ./prestige_db del k2
```

### Programmatic Usage

**Exact mode (default):**

```cpp
#include <prestige/store.hpp>

prestige::Options opt;
std::unique_ptr<prestige::Store> db;
prestige::Store::Open("./my_db", &db, opt);

db->Put("key1", "value1");
db->Put("key2", "value1");  // Deduplicates with key1
```

**Semantic mode:**

```cpp
#include <prestige/store.hpp>

prestige::Options opt;
opt.dedup_mode = prestige::DedupMode::kSemantic;
opt.semantic_model_path = "./minilm_onnx/model.onnx";
opt.semantic_model_type = prestige::SemanticModel::kMiniLM;
opt.semantic_threshold = 0.85f;  // Cosine similarity threshold

std::unique_ptr<prestige::Store> db;
prestige::Store::Open("./my_semantic_db", &db, opt);

db->Put("key1", "The quick brown fox jumps over the lazy dog.");
db->Put("key2", "A fast brown fox leaps above a sleepy dog.");  // Semantic match!
```

---

## Configuration

`prestige::Options` (see `include/prestige/store.hpp`):

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `block_cache_bytes` | 256 MB | LRU block cache size for RocksDB table blocks |
| `bloom_bits_per_key` | 10 | Bloom filter bits per key (for point-lookups) |
| `lock_timeout_ms` | 2000 | TransactionDB lock timeout |
| `max_retries` | 16 | Max transaction retries on conflicts/busy statuses |
| `enable_gc` | true | Whether to delete objects when refcount reaches 0 |
| `dedup_mode` | `kExact` | Deduplication mode: `kExact` or `kSemantic` |

### Semantic mode options

| Option | Default | Description |
|--------|---------|-------------|
| `semantic_model_path` | (required) | Path to ONNX model file |
| `semantic_model_type` | `kMiniLM` | Model type: `kMiniLM` or `kBGESmall` |
| `semantic_threshold` | (required) | Cosine similarity threshold [0.0, 1.0] |
| `semantic_max_text_bytes` | 8192 | Max text bytes to embed (longer texts truncated) |
| `semantic_search_k` | 10 | Number of nearest neighbors to check |
| `semantic_index_save_interval` | 1000 | Auto-save index every N inserts (0 = disabled) |

### HNSW index options

| Option | Default | Description |
|--------|---------|-------------|
| `hnsw_m` | 16 | Max connections per node |
| `hnsw_ef_construction` | 200 | Build-time search depth |
| `hnsw_ef_search` | 50 | Query-time search depth |

---

### Observability (metrics + tracing)

Prestige can emit lightweight **metrics** and **traces** if you provide hooks via
`prestige::Options`:

- `Options::metrics`: a `std::shared_ptr<prestige::MetricsSink>`
- `Options::tracer`: a `std::shared_ptr<prestige::Tracer>`

When unset (the default), there is essentially no overhead.

#### Metrics emitted

The store emits a small, stable set of counters/histograms (names are strings):

- Get:
  - Counters: `prestige.get.calls`, `prestige.get.ok_total`, `prestige.get.not_found_total`, `prestige.get.error_total`
  - Histograms: `prestige.get.latency_us`, `prestige.get.user_lookup_us`, `prestige.get.object_lookup_us`, `prestige.get.value_bytes`
- Put:
  - Counters: `prestige.put.calls`, `prestige.put.ok_total`, `prestige.put.timed_out_total`, `prestige.put.error_total`,
    `prestige.put.retry_total`, `prestige.put.dedup_hit_total`, `prestige.put.dedup_miss_total`,
    `prestige.put.object_created_total`, `prestige.put.noop_overwrite_total`
  - Histograms: `prestige.put.latency_us`, `prestige.put.sha256_us`, `prestige.put.commit_us`, `prestige.put.value_bytes`, `prestige.put.attempts`
- Delete:
  - Counters: `prestige.delete.calls`, `prestige.delete.ok_total`, `prestige.delete.not_found_total`, `prestige.delete.timed_out_total`, `prestige.delete.error_total`, `prestige.delete.retry_total`
  - Histograms: `prestige.delete.latency_us`, `prestige.delete.commit_us`, `prestige.delete.attempts`
- GC (immediate mode):
  - Counter: `prestige.gc.deleted_objects_total`
- Semantic mode (additional):
  - Counters: `prestige.semantic.hit_total`, `prestige.semantic.miss_total`, `prestige.put.embed_error_total`, `prestige.semantic.index_add_error_total`
  - Histograms: `prestige.put.embed_us`, `prestige.semantic.lookup_us`, `prestige.semantic.candidates_checked`

#### Tracing emitted

If `Options::tracer` is set, the store emits spans:

- `prestige.Get`
- `prestige.Put`
- `prestige.Delete`

Attributes include common fields like `key_bytes`, `value_bytes`, `latency_us`,
`attempts`, `dedup_hit`, and a coarse `status` label.

Events are added for retries (e.g. `retry.commit`) and for GC deletion (e.g.
`gc.delete_object`).

---

## Guarantees and trade-offs

### Guarantees

- **Exact deduplication** (exact mode): byte-identical values dedup to a single stored copy.
- **Semantic deduplication** (semantic mode): semantically similar values (above threshold) share storage.
- **Atomic updates** across all CFs via a single transaction commit.
- **Safe overwrite semantics**: overwriting a key updates refcounts and (optionally) reclaims unreferenced objects.

### Trade-offs

- **Read amplification**: GET is two point-reads (key→id, id→value).
- **Write overhead (exact mode)**: PUT does SHA-256 + multi-CF writes + transactional locking.
- **Write overhead (semantic mode)**: PUT does ONNX inference (~10-50ms) + vector search + multi-CF writes.
- **Immediate GC**: if enabled, an unreferenced object is deleted as soon as the last key is removed; a later identical Put will re-create it.
- **Semantic mode storage**: adds 1.5KB per unique value for embedding vectors + external index file.
- **Semantic mode accuracy**: depends on model quality and threshold tuning; may have false positives/negatives.

---

## Roadmap / TODO

If you want to take this beyond a prototype, the next steps tend to be:

- Batch APIs: `PutMany`, `GetMany`, `DeleteMany` (amortize transaction cost).
- Chunk-level dedup (content-defined chunking) for large blobs.
- Optional value canonicalization before hashing (exact mode).
- Background GC / tombstone handling modes (instead of immediate delete).
- FAISS backend for semantic mode (IVF+PQ for larger scale).
- Proper tokenization for embedding models (currently uses simplified tokenization).
- Hybrid mode: exact dedup with semantic fallback.

---

## License

This code is released on the Apache 2.0 license.  