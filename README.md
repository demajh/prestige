# prestige unique value store

**Why the name 'prestige'?**

In the movie, *The Prestige*, Hugh Jackman's character “deduplicates” himself after every time he performs his version of *The Transported Man* trick:
https://www.youtube.com/shorts/rfmHhWYKslU

The **prestige unique value store** is a less gruesome, but no less effective way to **deduplicate your data** and maintain a **queryable store** with a **single physical copy** of each unique value.

---

## What this repo contains (prototype implementation)

This repository is a small, buildable C++ prototype that implements the “unique value store” design discussed earlier, packaged as a real project:

- A **library** (`prestige::Store`) that wraps **RocksDB TransactionDB** and performs **deduplication on Put**.
- A small **example program** under `examples/` to validate behavior.
- A minimal **CLI tool** under `tools/` for manual testing.

### What changed vs a vanilla RocksDB KV store

In vanilla RocksDB, each `Put(key, value)` stores `value` under `key` (so identical values are duplicated across keys). Prestige introduces **indirection**:

- User keys map to **object IDs**
- Object IDs map to the **actual value bytes**
- A **dedup index** maps a **content hash** to an object ID so identical values are stored once

### What changed vs the initial single-file sketch

This prototype is intentionally “repo-shaped” rather than “single header”:

- ✅ Split into a **public header** (`include/prestige/store.hpp`) and **implementation** (`src/store.cpp`)
- ✅ Moved hashing + low-level helpers into `include/prestige/internal.hpp`
- ✅ Added a **CLI** (`tools/prestige_cli.cpp`) and **example** (`examples/basic.cpp`)
- ✅ Renamed all Column Families to `prestige_*` to avoid collisions and make on-disk layout self-identifying
- ✅ Added `prestige_object_meta` (object_id → digest) so garbage-collection can safely remove the dedup index entry for an object

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




