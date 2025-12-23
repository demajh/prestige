# Observability

Prestige can emit lightweight **metrics** and **traces** if you provide hooks via `prestige::Options`:

- `Options::metrics`: a `std::shared_ptr<prestige::MetricsSink>`
- `Options::tracer`: a `std::shared_ptr<prestige::Tracer>`

When unset (the default), there is essentially no overhead.

## Metrics Interface

```cpp
struct MetricsSink {
  virtual void Counter(std::string_view name, uint64_t delta) = 0;
  virtual void Histogram(std::string_view name, uint64_t value) = 0;
  virtual void Gauge(std::string_view name, double value) {}
};
```

## Metrics Emitted

### Get Operations

- **Counters**: `prestige.get.calls`, `prestige.get.ok_total`, `prestige.get.not_found_total`, `prestige.get.error_total`, `prestige.get.expired_total`
- **Histograms**: `prestige.get.latency_us`, `prestige.get.user_lookup_us`, `prestige.get.object_lookup_us`, `prestige.get.value_bytes`

### Put Operations

- **Counters**: `prestige.put.calls`, `prestige.put.ok_total`, `prestige.put.timed_out_total`, `prestige.put.error_total`, `prestige.put.retry_total`, `prestige.put.dedup_hit_total`, `prestige.put.dedup_miss_total`, `prestige.put.object_created_total`, `prestige.put.noop_overwrite_total`
- **Histograms**: `prestige.put.latency_us`, `prestige.put.sha256_us`, `prestige.put.normalize_us`, `prestige.put.commit_us`, `prestige.put.value_bytes`, `prestige.put.attempts`, `prestige.put.batch_writes`

### Delete Operations

- **Counters**: `prestige.delete.calls`, `prestige.delete.ok_total`, `prestige.delete.not_found_total`, `prestige.delete.timed_out_total`, `prestige.delete.error_total`, `prestige.delete.retry_total`
- **Histograms**: `prestige.delete.latency_us`, `prestige.delete.commit_us`, `prestige.delete.attempts`, `prestige.delete.batch_writes`

### Transaction

- **Histogram**: `prestige.txn.wait_us` (time spent waiting due to lock contention/retries)

### GC

- **Counter**: `prestige.gc.deleted_objects_total`

### Semantic Mode (additional)

- **Counters**: `prestige.semantic.hit_total`, `prestige.semantic.miss_total`, `prestige.put.embed_error_total`, `prestige.semantic.index_add_error_total`
- **Histograms**: `prestige.put.embed_us`, `prestige.semantic.lookup_us`, `prestige.semantic.candidates_checked`

### Cache (via `EmitCacheMetrics()`)

Call `EmitCacheMetrics()` periodically (e.g., every few seconds) to emit:

- **Counters**: `prestige.cache.hit_total`, `prestige.cache.miss_total` (deltas since last call)
- **Gauges**: `prestige.cache.fill_ratio` (0.0-1.0), `prestige.cache.usage_bytes`, `prestige.cache.capacity_bytes`
- **Gauges**: `prestige.bloom.useful_total`, `prestige.bloom.checked_total`

### TTL/LRU Management

- **Counters**: `prestige.sweep.deleted_total`, `prestige.prune.deleted_total`, `prestige.evict.deleted_total`
- **Gauges**: `prestige.store.total_bytes` (current store size)

---

## Tracing Interface

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

## Spans Emitted

If `Options::tracer` is set, the store emits spans:

- `prestige.Get`
- `prestige.Put`
- `prestige.Delete`

### Span Attributes

- `key_bytes` - Size of the user key
- `value_bytes` - Size of the value
- `latency_us` - Operation latency in microseconds
- `attempts` - Number of transaction attempts
- `dedup_hit` - Whether deduplication found a match (1 or 0)
- `had_old` - Whether this was an overwrite (1 or 0)
- `noop_overwrite` - Whether the overwrite was a no-op (same value)
- `status` - Coarse status label (ok, not_found, timed_out, etc.)

### Span Events

- `retry.user_key_lock` - Retry due to user key lock contention
- `retry.dedup_lock` - Retry due to dedup index lock contention
- `retry.commit` - Retry due to commit conflict
- `gc.delete_object` - Object was garbage collected

## Example Implementation

```cpp
class MyMetrics : public prestige::MetricsSink {
public:
  void Counter(std::string_view name, uint64_t delta) override {
    // Send to Prometheus, StatsD, etc.
  }
  void Histogram(std::string_view name, uint64_t value) override {
    // Record histogram sample
  }
  void Gauge(std::string_view name, double value) override {
    // Set gauge value
  }
};

prestige::Options opt;
opt.metrics = std::make_shared<MyMetrics>();
```
