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
