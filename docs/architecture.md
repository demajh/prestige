# Architecture

Prestige implements a content-deduplicated value store using multiple RocksDB Column Families. The store supports two mutually exclusive deduplication modes.

## Exact Mode (default)

Uses SHA-256 content hashing for byte-identical deduplication:

1. **User mapping**: `user_key -> object_id`
2. **Object store**: `object_id -> value_bytes`
3. **Dedup index**: `sha256(value_bytes) -> object_id`
4. **Refcount**: `object_id -> uint64_le`
5. **Object meta**: `object_id -> sha256(value_bytes)` (used for safe GC / index cleanup)

## Semantic Mode

Uses neural network embeddings for semantic similarity deduplication:

1. **User mapping**: `user_key -> object_id`
2. **Object store**: `object_id -> value_bytes`
3. **Embeddings**: `object_id -> embedding_vector` (384 floats)
4. **Refcount**: `object_id -> uint64_le`
5. **Vector index**: External HNSW index file for similarity search

## Column Families and Schema

### Exact mode

| Column family | Purpose | Key | Value |
|---|---|---|---|
| `prestige_user_kv` | User key to object reference | `user_key` | `object_id` (16 bytes) |
| `prestige_object_store` | Actual stored values | `object_id` (16 bytes) | raw `value_bytes` |
| `prestige_dedup_index` | Dedup lookup | `sha256(value_bytes)` (32 bytes) | `object_id` (16 bytes) |
| `prestige_refcount` | Reference counting for GC | `object_id` | `uint64_le` |
| `prestige_object_meta` | Reverse mapping for GC | `object_id` | `sha256(value_bytes)` |

### Semantic mode (adds/replaces)

| Column family | Purpose | Key | Value |
|---|---|---|---|
| `prestige_embeddings` | Embedding vectors | `object_id` (16 bytes) | 384 floats (1536 bytes) |
| External file: `db_path.vec_index` | HNSW vector index | - | hnswlib format |

## Identifiers

- **Dedup key (exact mode)**: `SHA-256(value_bytes)` (exact / byte-identical dedup).
- **Dedup key (semantic mode)**: Embedding vector from ONNX model (semantic similarity dedup).
- **Object ID**: random 128-bit ID (16 bytes). This avoids using the hash directly as the storage key, but still allows dedup via the index.

---

## Operation Semantics

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

This is a "RocksDB-like" API: users never see object IDs.

### Delete(user_key)

1. Transactionally remove `prestige_user_kv[user_key]`
2. Decrement the referenced object's refcount
3. If refcount reaches 0 and `enable_gc=true`:
   - delete `prestige_object_store[object_id]`
   - delete `prestige_refcount[object_id]`
   - delete `prestige_object_meta[object_id]`
   - delete `prestige_dedup_index[digest]` **only if** it still points to `object_id`
     (protects against races where a digest was remapped)

---

## Guarantees and Trade-offs

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
