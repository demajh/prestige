# prestige unique value store

**What is a unique value store?**

A unique value store is a queryable collection of unique values.  In order to maintain the 'unique'-ness of the collection, there must be some mechanism to remove duplicates.

prestige implements this collection as a key-value store.  It is effectively a wrapper around RocksDB, performing deduplication of elements after Put() operations.  

**Why the name 'prestige'?**

In the movie, *The Prestige*, Hugh Jackman's character “deduplicates” himself after every time he performs his version of *The Transported Man* trick:
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

Prestige implements a content-deduplicated value store using multiple RocksDB Column Families:

1. **User mapping**: `user_key -> object_id`
2. **Object store**: `object_id -> value_bytes`
3. **Dedup index**: `sha256(value_bytes) -> object_id`
4. **Refcount**: `object_id -> uint64_le`
5. **Object meta**: `object_id -> sha256(value_bytes)` (used for safe GC / index cleanup)

### Column families and schema

| Column family | Purpose | Key | Value |
|---|---|---|---|
| `prestige_user_kv` | User key to object reference | `user_key` | `object_id` (16 bytes) |
| `prestige_object_store` | Actual stored values | `object_id` (16 bytes) | raw `value_bytes` |
| `prestige_dedup_index` | Dedup lookup | `sha256(value_bytes)` (32 bytes) | `object_id` (16 bytes) |
| `prestige_refcount` | Reference counting for GC | `object_id` | `uint64_le` |
| `prestige_object_meta` | Reverse mapping for GC | `object_id` | `sha256(value_bytes)` |

### Identifiers

- **Dedup key**: `SHA-256(value_bytes)` (exact / byte-identical dedup).
- **Object ID**: random 128-bit ID (16 bytes).  
  This avoids using the hash directly as the storage key, but still allows dedup via the index.

---

## Operation semantics

### Put(user_key, value_bytes)

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

- A RocksDB build that includes **TransactionDB** support (`rocksdb/utilities/transaction_db.h`).

### Build with CMake

```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

Outputs:
- `prestige_uvs` (library)
- `prestige_example_basic` (example program)
- `prestige_cli` (CLI)

---

## Usage

### Example program

```bash
./build/prestige_example_basic
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

---

## Configuration

`prestige::Options` (see `include/prestige/store.hpp`):

- `block_cache_bytes`: LRU block cache size for RocksDB table blocks.
- `bloom_bits_per_key`: bloom filter bits per key (used for point-lookups).
- `lock_timeout_ms`: TransactionDB lock timeout.
- `max_retries`: max transaction retries on conflicts / busy statuses.
- `enable_gc`: whether to immediately delete objects when refcount reaches 0.

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

## Guarantees and trade-offs (prototype)

### Guarantees

- **Exact deduplication**: byte-identical values dedup to a single stored copy.
- **Atomic updates** across all CFs via a single transaction commit.
- **Safe overwrite semantics**: overwriting a key updates refcounts and (optionally) reclaims unreferenced objects.

### Trade-offs

- **Read amplification**: GET is two point-reads (key→id, id→value).
- **Write overhead**: PUT does SHA-256 + multi-CF writes + transactional locking.
- **Exact-only**: no canonicalization (e.g., JSON normalization), no semantic matching.
- **Immediate GC**: if enabled, an unreferenced object is deleted as soon as the last key is removed; a later identical Put will re-create it.

---

## Roadmap / TODO

If you want to take this beyond a prototype, the next steps tend to be:

- Batch APIs: `PutMany`, `GetMany`, `DeleteMany` (amortize transaction cost).
- Chunk-level dedup (content-defined chunking) for large blobs.
- Optional value canonicalization before hashing.
- Background GC / tombstone handling modes (instead of immediate delete).

---

## License

This code is released on the Apache 2.0 license.  