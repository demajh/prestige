---
name: prestige
version: 0.2.0
description: A key-value store that keeps one physical copy of every unique value. Deduplicate memories, caches and datasets with Put/Get; exact (SHA-256) or semantic (embedding) matching; TTL and LRU eviction; RocksDB underneath.
homepage: https://github.com/demajh/prestige
metadata: {"openclaw":{"category":"storage","install":"pip install prestige-uvs","import":"prestige","language":"python","runtime":"local, no network, no account"}}
---

# prestige — unique value store

prestige is a local key-value store where duplicate values are stored once. `Put("k1", data)` and `Put("k2", data)`
keep one copy of `data` and two keys pointing at it; deleting a key only frees the value when nothing else references
it. It is built on RocksDB (transactions, crash safety) and runs entirely on your machine: no network, no account,
no telemetry.

## Skill files

| File | URL |
|------|-----|
| **SKILL.md** (this file) | `https://raw.githubusercontent.com/demajh/prestige/main/skill.md` |
| **package metadata** | `https://raw.githubusercontent.com/demajh/prestige/main/skill.json` |
| Python API reference | `https://github.com/demajh/prestige/blob/main/docs/python-bindings.md` |
| Cache semantics (TTL, LRU) | `https://github.com/demajh/prestige/blob/main/docs/cache-semantics.md` |
| Semantic dedup (embeddings) | `https://github.com/demajh/prestige/blob/main/docs/semantic-dedup.md` |
| ML dataloaders | `https://github.com/demajh/prestige/blob/main/docs/dataloaders.md` |

**Install locally:**
```bash
mkdir -p ~/.moltbot/skills/prestige
curl -s https://raw.githubusercontent.com/demajh/prestige/main/skill.md > ~/.moltbot/skills/prestige/SKILL.md
curl -s https://raw.githubusercontent.com/demajh/prestige/main/skill.json > ~/.moltbot/skills/prestige/package.json
```

## When to reach for it

- **Agent memory that keeps re-storing the same thing.** Notes, tool outputs, fetched pages and summaries written
  under new keys every session. prestige stores each distinct value once and tells you the dedup ratio.
- **A cache with real semantics.** Per-entry or default TTL, a size cap with LRU eviction, health stats, atomic
  operations. Keys can be anything; values are bytes or text.
- **Near-duplicates, not just byte-identical ones.** Semantic mode embeds values (BGE-small or MiniLM via ONNX) and
  treats anything above a cosine threshold as the same value. Text normalization (case, whitespace) is a cheaper
  middle step.
- **Training or evaluation data.** `prestige.dataloaders` deduplicates Hugging Face / PyTorch datasets and measures
  train/test contamination before you fine-tune or benchmark.

## Install

```bash
pip install prestige-uvs
```

Binary wheels (RocksDB bundled, Python 3.9 to 3.13) for Linux x86_64 and macOS (Intel and Apple silicon). The
import name is `prestige`. Check it works:

```bash
python -c "import prestige; print(prestige.__version__, prestige.SEMANTIC_AVAILABLE)"
```

`SEMANTIC_AVAILABLE` is `False` in the wheel: exact mode, normalization, TTL, LRU and the dataloaders are all there;
semantic mode needs a build with ONNX Runtime (below).

From source (any platform with a C++17 compiler; required for semantic mode):

```bash
# macOS: brew install rocksdb openssl@3      Ubuntu/Debian: sudo apt-get install librocksdb-dev libssl-dev cmake
git clone https://github.com/demajh/prestige.git && cd prestige/python && pip install .
# semantic mode: export PRESTIGE_ENABLE_SEMANTIC=1 ONNXRUNTIME_DIR=/path/to/onnxruntime before `pip install .`
```

## Use it (Python)

```python
import prestige

with prestige.open("./memory_store") as store:           # creates the directory if needed
    store.put("session-42/summary", "The user prefers terse answers.")
    store.put("session-43/summary", "The user prefers terse answers.")   # same bytes: stored once
    store.get("session-43/summary", decode=True)          # 'The user prefers terse answers.'
    store.count_keys(), store.count_unique_values()       # (2, 1)
    store.get_health()["dedup_ratio"]                      # 2.0
    "session-42/summary" in store                          # True
    del store["session-42/summary"]                        # value survives: another key still points at it
```

Dict-style access (`store[k] = v`, `store[k]`, `len(store)`) works too. Values are `bytes`; pass `decode=True` to
get `str`. Missing keys raise `prestige.NotFoundError` unless you pass `default=`.

**Cache semantics**

```python
opts = prestige.Options()
opts.default_ttl_seconds = 24 * 3600          # entries expire after a day (per-put TTL also available)
opts.max_store_bytes = 2 * 1024**3            # cap at 2 GB; least-recently-used values are evicted first
opts.normalization_mode = prestige.NormalizationMode.kCaseWhitespace   # "Hello  World" == "hello world"
with prestige.open("./cache", opts) as cache:
    cache.put("tool:web:https://example.com", page_bytes)
    cache.get_health()                        # size, entries, hit/evict counters, dedup ratio
```

**Semantic mode** (source build with ONNX Runtime; model files from Hugging Face as in docs/semantic-dedup.md):

```python
opts = prestige.Options()
opts.dedup_mode = prestige.DedupMode.kSemantic
opts.semantic_model_path = "./models/model.onnx"   # bge-small-en-v1.5 or all-MiniLM-L6-v2, vocab.txt beside it
opts.semantic_threshold = 0.9                      # cosine similarity above which two values are "the same"
```

**Datasets**

```python
from datasets import load_dataset
from prestige.dataloaders import deduplicate_dataset, detect_train_test_leakage
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
clean = deduplicate_dataset(ds["train"], mode="exact")
leak = detect_train_test_leakage(ds["train"], ds["test"], threshold=0.95)   # {'contamination_rate': ..., ...}
```

## Use it as a service

A standalone HTTP server (REST `PUT/GET/DELETE /api/v1/kv/{key}`, prefix listing, `/health/ready`, Prometheus metrics)
ships in the C++ build: `cmake -B build -DPRESTIGE_BUILD_SERVER=ON && cmake --build build`, then
`./build/prestige-server --db-path ./mydb --port 8080`. See docs/server.md.

## Good to know

- One process at a time per store directory (RocksDB lock). Open the store once and keep it open.
- Everything is local. prestige never sends data anywhere; there is nothing to sign up for.
- Apache 2.0. Issues and pull requests at https://github.com/demajh/prestige.
