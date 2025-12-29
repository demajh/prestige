# prestige

[![CI](https://github.com/demajh/prestige/actions/workflows/ci.yml/badge.svg)](https://github.com/demajh/prestige/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/demajh/prestige)](https://github.com/demajh/prestige/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A unique value store built on RocksDB.

## What is prestige?

prestige is a **unique value store** - a queryable collection where duplicate values are automatically eliminated. When you store the same content under multiple keys, prestige keeps only one physical copy and maintains references from each key.

In a vanilla key-value store, `Put("k1", data)` and `Put("k2", data)` stores `data` twice. In prestige, the second Put detects the duplicate and reuses the existing copy - you get two keys pointing to one stored value.

Prestige supports two deduplication strategies:

- **Exact mode** (default): Uses SHA-256 hashing for byte-identical deduplication
- **Semantic mode**: Uses neural network embeddings to deduplicate semantically similar content (e.g., "The quick brown fox" and "A fast brown fox" can share storage)

## Why "prestige"?

In the movie, The Prestige, Hugh Jackman's character "deduplicates" himself after every time he performs his version of The Transported Man trick: https://www.youtube.com/shorts/rfmHhWYKslU

The prestige unique value store is a less gruesome, but no less effective way to deduplicate your data and maintain a queryable store with a single physical copy of each unique value.

## Features

- **Automatic deduplication** - identical values stored once, referenced many times
- **Two dedup modes** - exact (SHA-256) or semantic (neural embeddings)
- **Text normalization** - optional case/whitespace normalization for stable dedup keys
- **Cache semantics** - TTL expiration, LRU eviction, health monitoring
- **Atomic operations** - RocksDB TransactionDB ensures consistency
- **Observability** - pluggable metrics and distributed tracing hooks
- **Simple API** - familiar `Put`/`Get`/`Delete` interface
- **HTTP Server** - REST API for standalone deployment with Prometheus metrics

## Quick Start

### Install

```bash
# macOS
brew install rocksdb
git clone https://github.com/demajh/prestige.git && cd prestige
mkdir build && cd build && cmake .. && make

# Ubuntu/Debian
sudo apt-get install librocksdb-dev
git clone https://github.com/demajh/prestige.git && cd prestige
mkdir build && cd build && cmake .. && make
```

See [docs/building.md](docs/building.md) for detailed build instructions and pre-built binaries.

### Use (C++)

```cpp
#include <prestige/store.hpp>

prestige::Options opt;
std::unique_ptr<prestige::Store> db;
prestige::Store::Open("./my_db", &db, opt);

db->Put("key1", "hello world");
db->Put("key2", "hello world");  // Deduplicates - same content!

std::string value;
db->Get("key1", &value);  // "hello world"
```

### Use (Python)

```bash
# Install from source
cd python
pip install .
```

```python
import prestige

# Open a store (automatically creates if doesn't exist)
with prestige.open("./my_db") as store:
    # Store values
    store.put("key1", "hello world")
    store.put("key2", "hello world")  # Deduplicates - same content!

    # Retrieve values
    value = store.get("key1", decode=True)  # "hello world"

    # Check deduplication stats
    print(f"Keys: {store.count_keys()}")                    # 2
    print(f"Unique values: {store.count_unique_values()}")  # 1
```

### CLI

```bash
./prestige_cli ./mydb put k1 "hello"
./prestige_cli ./mydb put k2 "hello"  # Deduplicates
./prestige_cli ./mydb get k1
./prestige_cli ./mydb count           # 2 keys, 1 unique value
```

### HTTP Server

prestige can run as a standalone HTTP server exposing a REST API:

```bash
# Build with server support (requires Drogon framework)
cmake -B build -DPRESTIGE_BUILD_SERVER=ON
cmake --build build

# Start the server
./build/prestige-server --db-path ./mydb --port 8080
```

Use the REST API:

```bash
# Store a value
curl -X PUT http://localhost:8080/api/v1/kv/mykey -d "hello world"

# Retrieve a value
curl http://localhost:8080/api/v1/kv/mykey

# Delete a key
curl -X DELETE http://localhost:8080/api/v1/kv/mykey

# List keys
curl "http://localhost:8080/api/v1/kv?prefix=user:&limit=100"

# Health check
curl http://localhost:8080/health/ready
```

See [docs/server.md](docs/server.md) for full API reference and configuration options.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | How prestige works internally |
| [API Reference](docs/api-reference.md) | Complete API documentation |
| [Building](docs/building.md) | Build instructions and dependencies |
| [CLI](docs/cli.md) | Command-line tool reference |
| [HTTP Server](docs/server.md) | REST API server reference |
| [Python Bindings](docs/python-bindings.md) | Python API reference and examples |
| [Cache Semantics](docs/cache-semantics.md) | TTL, LRU eviction, health stats |
| [Semantic Dedup](docs/semantic-dedup.md) | Neural embedding-based deduplication |
| [Normalization](docs/normalization.md) | Text normalization options |
| [Observability](docs/observability.md) | Metrics and tracing |

## Roadmap

- End-to-end RAG examples + benchmark harness
- Public embedding-cache functions: `GetEmbedding*`, `PutEmbedding*`, and metadata access (dims/dtype/model fingerprint), ideally via an `EmbeddingCache` wrapper atop the generic store.
- Concurrency: "inflight" reservation/lease support so multiple workers don't double-embed the same missing chunk.
- Batch APIs: `PutMany`, `GetMany`, `DeleteMany` (and batch variants for dedup-key operations) to amortize transaction cost and match embedding-provider batching.
- Model/version-aware cache keys + binary embedding format + metadata (dims/dtype/etc.), and a configurable "bring your own embedder" interface so users can plug in OpenAI/Cohere/local/ONNX/etc.
- Explicit dedup-key operations (pre-embed lookup): e.g., `GetByDedupKey`, `PutByDedupKey`, and (optionally) `Link(alias_key -> dedup_key)` so you dedup *before* computing embeddings.

## License

Apache 2.0
