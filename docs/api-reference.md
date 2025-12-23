# API Reference

## Store Class

### Opening a Store

```cpp
static rocksdb::Status Store::Open(
    const std::string& db_path,
    std::unique_ptr<Store>* out,
    const Options& opt = Options{});
```

Opens or creates a prestige store at `db_path`. Creates required column families automatically.

### Core Methods

```cpp
rocksdb::Status Put(std::string_view user_key, std::string_view value_bytes);
```
Store a value under `user_key`. Deduplicates automatically based on configured mode.

```cpp
rocksdb::Status Get(std::string_view user_key, std::string* value_bytes_out) const;
```
Retrieve the value for `user_key`. Returns `NotFound` if key doesn't exist or object is expired (TTL).

```cpp
rocksdb::Status Delete(std::string_view user_key);
```
Delete a key mapping. If `enable_gc=true` and refcount reaches 0, the object is also deleted.

```cpp
void Close();
```
Close the store and release RocksDB resources. Safe to call multiple times.

### Enumeration Methods

```cpp
rocksdb::Status CountKeys(uint64_t* out_key_count) const;
rocksdb::Status CountUniqueValues(uint64_t* out_unique_value_count) const;
rocksdb::Status ListKeys(std::vector<std::string>* out_keys,
                         uint64_t limit = 0,
                         std::string_view prefix = {}) const;
```

### Cache Management Methods

```cpp
rocksdb::Status Sweep(uint64_t* deleted_count);
```
Delete all expired (past TTL) and orphaned (refcount=0) objects.

```cpp
rocksdb::Status Prune(uint64_t max_age_seconds,
                      uint64_t max_idle_seconds,
                      uint64_t* deleted_count);
```
Delete objects older than `max_age_seconds` OR not accessed for `max_idle_seconds`.

```cpp
rocksdb::Status EvictLRU(uint64_t target_bytes, uint64_t* evicted_count);
```
Evict least-recently-used objects until store size is at or below `target_bytes`.

```cpp
rocksdb::Status GetHealth(HealthStats* stats) const;
```
Get store health statistics. Performs a scan of all objects.

```cpp
uint64_t GetTotalStoreBytes() const;
```
Get current approximate total store size in bytes.

```cpp
void EmitCacheMetrics();
```
Emit current cache statistics to the metrics sink. Call periodically for observability.

---

## Options

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `block_cache_bytes` | 256 MB | LRU block cache size for RocksDB table blocks |
| `bloom_bits_per_key` | 10 | Bloom filter bits per key (for point-lookups) |
| `lock_timeout_ms` | 2000 | TransactionDB lock timeout |
| `max_retries` | 16 | Max transaction retries on conflicts/busy statuses |
| `enable_gc` | true | Whether to delete objects when refcount reaches 0 |
| `dedup_mode` | `kExact` | Deduplication mode: `kExact` or `kSemantic` |

### Cache Behavior Options

| Option | Default | Description |
|--------|---------|-------------|
| `default_ttl_seconds` | 0 | Default TTL for objects in seconds (0 = no expiration) |
| `max_store_bytes` | 0 | Maximum store size in bytes (0 = unlimited) |
| `eviction_target_ratio` | 0.8 | When evicting, reduce to this ratio of max_store_bytes |
| `track_access_time` | true | Track last access time for LRU (slight overhead on Get) |

### Text Normalization Options

| Option | Default | Description |
|--------|---------|-------------|
| `normalization_mode` | `kNone` | Normalization level: `kNone`, `kWhitespace`, `kASCII`, or `kUnicode` |
| `normalization_max_bytes` | 1MB | Skip normalization for values larger than this (0 = unlimited) |

### Semantic Mode Options

| Option | Default | Description |
|--------|---------|-------------|
| `semantic_model_path` | (required) | Path to ONNX model file (vocab.txt auto-detected in same directory) |
| `semantic_model_type` | `kMiniLM` | Model type: `kMiniLM` or `kBGESmall` |
| `semantic_threshold` | (required) | Cosine similarity threshold [0.0, 1.0] |
| `semantic_max_text_bytes` | 8192 | Max text bytes to embed (longer texts truncated) |
| `semantic_search_k` | 10 | Number of nearest neighbors to check |
| `semantic_index_save_interval` | 1000 | Auto-save index every N inserts (0 = disabled) |

### HNSW Index Options

| Option | Default | Description |
|--------|---------|-------------|
| `hnsw_m` | 16 | Max connections per node |
| `hnsw_ef_construction` | 200 | Build-time search depth |
| `hnsw_ef_search` | 50 | Query-time search depth |

### Observability Options

| Option | Default | Description |
|--------|---------|-------------|
| `metrics` | nullptr | `std::shared_ptr<MetricsSink>` for metrics emission |
| `tracer` | nullptr | `std::shared_ptr<Tracer>` for distributed tracing |

---

## Types

### DedupMode

```cpp
enum class DedupMode {
  kExact,     // SHA-256 content hash (default)
  kSemantic   // Embedding similarity via vector index
};
```

### NormalizationMode

```cpp
enum class NormalizationMode {
  kNone,        // No normalization (byte-exact dedup)
  kWhitespace,  // Collapse whitespace runs, trim edges
  kASCII,       // Whitespace + ASCII case folding
  kUnicode      // Whitespace + ASCII + Unicode-lite normalizations
};
```

### SemanticModel

```cpp
enum class SemanticModel {
  kMiniLM,    // all-MiniLM-L6-v2 (384 dimensions)
  kBGESmall   // BGE-small-en-v1.5 (384 dimensions)
};
```

### HealthStats

```cpp
struct HealthStats {
  uint64_t total_keys;          // Number of user keys
  uint64_t total_objects;       // Number of unique deduplicated objects
  uint64_t total_bytes;         // Total size of all objects
  uint64_t expired_objects;     // Objects past TTL (not yet swept)
  uint64_t orphaned_objects;    // Objects with refcount=0 (not yet GC'd)
  uint64_t oldest_object_age_s; // Age of oldest object in seconds
  uint64_t newest_access_age_s; // Time since most recent access
  double dedup_ratio;           // Ratio of keys to objects (higher = more dedup)
};
```

### MetricsSink Interface

```cpp
struct MetricsSink {
  virtual void Counter(std::string_view name, uint64_t delta) = 0;
  virtual void Histogram(std::string_view name, uint64_t value) = 0;
  virtual void Gauge(std::string_view name, double value) {}
};
```

### Tracer Interface

```cpp
struct Tracer {
  virtual std::unique_ptr<TraceSpan> StartSpan(std::string_view name) = 0;
};

struct TraceSpan {
  virtual void SetAttribute(std::string_view key, uint64_t value) = 0;
  virtual void SetAttribute(std::string_view key, std::string_view value) = 0;
  virtual void AddEvent(std::string_view name) = 0;
  virtual void End(const rocksdb::Status& status) = 0;
};
```
