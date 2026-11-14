# Python Examples

This directory contains example Python scripts demonstrating how to use the Prestige Python bindings.

## Prerequisites

Install the Prestige Python package:

```bash
cd python
pip install .
```

## Examples

### basic.py

Demonstrates core functionality:
- Opening and closing a store
- Basic put/get/delete operations
- Dict-like interface usage
- Deduplication statistics
- Health monitoring
- Key listing and filtering
- Binary data handling
- Persistence across sessions

Run:
```bash
python examples/basic.py
```

### embedding_cache.py

Shows how to use Prestige as an embedding cache:
- Caching expensive embedding computations
- Automatic deduplication of identical texts
- TTL-based cache expiration
- Binary storage of float vectors
- Hit/miss statistics
- Batch processing optimization

Run:
```bash
python examples/embedding_cache.py
```

This is a practical example for RAG (Retrieval-Augmented Generation) applications where you want to cache embeddings from OpenAI, Cohere, or local models.

### outcomes_demo.py

The Python twin of `examples/outcomes.cpp` (see [docs/outcomes-demo.md](../../docs/outcomes-demo.md)):
- Stores values with `metadata` (task family and ground-truth item id)
- Asks `candidates()` for the nearest stored values of each query and applies the caller's threshold
- Records every verdict with `record_outcome()` under a versioned family id
- Prints `family_report()` for each family: false-accept rate, similarity and rank distributions of the
  false accepts, and the advisory suggested threshold

Run (from the repository root; semantic mode needs a source build with `PRESTIGE_ENABLE_SEMANTIC=ON` and an ONNX
model, otherwise the store runs in exact mode and says so):
```bash
python python/examples/outcomes_demo.py --model models/bge-small-en-v1.5_onnx/model.onnx
```

### canonical_row.py

A worked example for hashing a record's canonical bytes (see [docs/canonical-row.md](../../docs/canonical-row.md)).
Needs only the standard library; the prestige package is used by `--store` alone.
- Encodes one row as a canonical line and prints its hex, byte count and SHA-256
- Prints each field's value, byte span in the line and the SHA-256 of that span
- Edge rows (null, negative integer, integer above 2^53, empty string, non-ASCII text) and a near-miss row
- Control rows: the same row through wrong-but-plausible encoders, with their hashes
- Two decoders (exact integers, doubles) and a chain of heads
- `--check` recomputes everything against `examples/canonical_row_vectors.json`; `--compare <hex>` diagnoses a line
  from your own encoder

Run (from the repository root):
```bash
python python/examples/canonical_row.py
python python/examples/canonical_row.py --check
```

## Creating Your Own Examples

The Python bindings provide a simple, Pythonic interface:

```python
import prestige

# Open a store
with prestige.open("/path/to/db") as store:
    # Store values
    store.put("key", "value")

    # Retrieve values
    value = store.get("key", decode=True)

    # Check deduplication
    print(f"Dedup ratio: {store.count_keys() / store.count_unique_values():.1f}x")
```

See the [Python Bindings documentation](../../docs/python-bindings.md) for complete API reference.
