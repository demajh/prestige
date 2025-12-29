# Python Bindings

Prestige provides Python bindings that expose the full functionality of the C++ library through a Pythonic interface.

## Installation

### From Source

```bash
cd python
pip install .
```

The installation process uses CMake to build the C++ extension module. The build process automatically:
- Detects your platform (Windows, macOS, Linux)
- Configures RocksDB dependencies
- Compiles the native extension module
- Installs the Python package

### Requirements

- Python 3.8 or later
- CMake 3.15 or later
- C++17 compatible compiler
- RocksDB library (see [building.md](building.md) for platform-specific installation)

### Optional Features

Enable semantic deduplication during installation:

```bash
export PRESTIGE_ENABLE_SEMANTIC=1
export ONNXRUNTIME_DIR=/path/to/onnxruntime
cd python
pip install .
```

### Development Installation

For development with testing dependencies:

```bash
cd python
pip install -e ".[dev]"
```

## Quick Start

### Basic Operations

```python
import prestige

# Open a store (context manager automatically closes)
with prestige.open("./my_db") as store:
    # Put values (accepts bytes or str)
    store.put("key1", "hello world")
    store.put("key2", b"binary data")

    # Get values (returns bytes by default)
    value = store.get("key1")  # b"hello world"

    # Get and decode to string
    value = store.get("key1", decode=True)  # "hello world"

    # Delete a key
    store.delete("key1")

    # Check if key exists
    if "key2" in store:
        print("key2 exists")
```

### Dict-like Interface

The Store class supports Python's dict-like operations:

```python
with prestige.open("./my_db") as store:
    # Set items
    store["key1"] = b"value1"
    store["key2"] = "value2"  # Auto-encodes strings

    # Get items
    value = store["key1"]  # b"value1"

    # Delete items
    del store["key1"]

    # Check membership
    if "key2" in store:
        print("found")

    # Get approximate count
    count = len(store)
```

### Deduplication

Prestige automatically deduplicates identical content:

```python
with prestige.open("./my_db") as store:
    # Store the same content under different keys
    for i in range(100):
        store.put(f"key_{i}", "identical content")

    # Check deduplication efficiency
    print(f"Total keys: {store.count_keys()}")           # 100
    print(f"Unique values: {store.count_unique_values()}")  # 1

    # Get health stats
    health = store.get_health()
    print(f"Dedup ratio: {health['dedup_ratio']:.1f}x")  # 100.0x
```

## API Reference

### Factory Function

#### `prestige.open(path, options=None, **kwargs)`

Open or create a prestige store.

**Parameters:**
- `path` (str): Database directory path (created if doesn't exist)
- `options` (Options, optional): Pre-configured Options object
- `**kwargs`: Options attributes (e.g., `default_ttl_seconds=3600`)

**Returns:** Store instance

**Examples:**

```python
# Simple usage
with prestige.open("/tmp/mydb") as store:
    store.put("key", "value")

# With keyword arguments
with prestige.open("/tmp/mydb", default_ttl_seconds=3600) as store:
    store.put("key", "value")

# With pre-configured options
options = prestige.Options()
options.max_store_bytes = 10 * 1024**3  # 10GB limit
with prestige.open("/tmp/mydb", options) as store:
    store.put("key", "value")
```

### Store Class

#### `Store.open(path, options=None)`

Class method to open a store (alternative to `prestige.open()`).

**Parameters:**
- `path` (str): Database directory path
- `options` (Options, optional): Configuration options

**Returns:** Store instance

#### `put(key, value, ttl_seconds=None)`

Store a value under a key.

**Parameters:**
- `key` (str): The key
- `value` (bytes or str): The value to store
- `ttl_seconds` (int, optional): Time-to-live in seconds (overrides default)

**Returns:** None

**Example:**

```python
store.put("key1", "value")
store.put("key2", b"binary", ttl_seconds=3600)
```

#### `get(key, default=None, decode=False)`

Retrieve a value by key.

**Parameters:**
- `key` (str): The key to look up
- `default` (any, optional): Value to return if key not found
- `decode` (bool): If True, decode bytes to UTF-8 string

**Returns:** bytes (or str if `decode=True`)

**Raises:** `NotFoundError` if key doesn't exist and no default provided

**Examples:**

```python
# Get as bytes
value = store.get("key1")  # b"value"

# Get as string
value = store.get("key1", decode=True)  # "value"

# With default
value = store.get("missing", default=b"not found")
```

#### `delete(key)`

Delete a key.

**Parameters:**
- `key` (str): The key to delete

**Returns:** None

**Raises:** `NotFoundError` if key doesn't exist

#### `list_keys(prefix="", limit=None)`

List keys in the store.

**Parameters:**
- `prefix` (str, optional): Only return keys with this prefix
- `limit` (int, optional): Maximum number of keys to return

**Returns:** list of str

**Examples:**

```python
# List all keys
all_keys = store.list_keys()

# List with prefix
user_keys = store.list_keys(prefix="user:")

# Limit results
first_10 = store.list_keys(limit=10)
```

#### `count_keys(approximate=True)`

Count the number of keys in the store.

**Parameters:**
- `approximate` (bool): If True, use fast approximate count

**Returns:** int

**Example:**

```python
exact_count = store.count_keys(approximate=False)
approx_count = store.count_keys(approximate=True)
```

#### `count_unique_values()`

Count the number of unique values stored.

**Returns:** int

#### `sweep()`

Remove unreferenced values from the store.

**Returns:** int (number of objects deleted)

#### `prune(max_age_seconds=0, max_idle_seconds=0)`

Remove expired or idle entries.

**Parameters:**
- `max_age_seconds` (int): Remove entries older than this
- `max_idle_seconds` (int): Remove entries not accessed in this time

**Returns:** int (number of keys deleted)

**Example:**

```python
# Remove entries older than 7 days
deleted = store.prune(max_age_seconds=7 * 24 * 3600)
```

#### `get_health()`

Get store health statistics.

**Returns:** dict with keys:
- `total_keys` (int): Number of keys
- `total_objects` (int): Number of unique values
- `total_bytes` (int): Total storage used
- `dedup_ratio` (float): Deduplication ratio (keys/objects)

**Example:**

```python
health = store.get_health()
print(f"Storage: {health['total_bytes'] / 1024**2:.1f} MB")
print(f"Dedup: {health['dedup_ratio']:.2f}x")
```

#### `flush()`

Flush pending writes to disk.

**Returns:** None

#### `close()`

Close the store and release resources.

**Returns:** None

**Note:** Automatically called when using context manager.

#### Properties

- `path` (str): Database directory path
- `closed` (bool): Whether the store is closed
- `total_bytes` (int): Total storage used in bytes

### Options Class

Configuration options for opening a store.

#### Attributes

- `dedup_mode` (DedupMode): Deduplication mode (EXACT or SEMANTIC)
- `normalization_mode` (NormalizationMode): Text normalization mode
- `default_ttl_seconds` (int): Default time-to-live (0 = no expiration)
- `max_store_bytes` (int): Maximum storage size (0 = unlimited)
- `block_cache_bytes` (int): RocksDB block cache size
- `write_buffer_bytes` (int): RocksDB write buffer size
- `enable_statistics` (bool): Enable RocksDB statistics

**Example:**

```python
options = prestige.Options()
options.default_ttl_seconds = 3600  # 1 hour default TTL
options.max_store_bytes = 10 * 1024**3  # 10GB limit
options.block_cache_bytes = 256 * 1024**2  # 256MB cache

with prestige.open("/tmp/mydb", options) as store:
    pass
```

### Enums

#### DedupMode

- `DedupMode.EXACT`: SHA-256 hash-based exact deduplication (default)
- `DedupMode.SEMANTIC`: Neural embedding-based semantic deduplication

#### NormalizationMode

- `NormalizationMode.NONE`: No normalization
- `NormalizationMode.LOWERCASE`: Convert to lowercase
- `NormalizationMode.WHITESPACE`: Normalize whitespace
- `NormalizationMode.FULL`: Lowercase + whitespace normalization

**Example:**

```python
options = prestige.Options()
options.dedup_mode = prestige.DedupMode.EXACT
options.normalization_mode = prestige.NormalizationMode.FULL

with prestige.open("/tmp/mydb", options) as store:
    store.put("key", "  HELLO  ")
    # Stored as "hello" due to normalization
```

### Exceptions

All exceptions inherit from `PrestigeError`:

- `NotFoundError`: Key not found
- `InvalidArgumentError`: Invalid argument provided
- `IOError`: I/O error occurred
- `CorruptionError`: Database corruption detected
- `BusyError`: Resource is busy
- `TimedOutError`: Operation timed out

**Example:**

```python
try:
    value = store.get("missing_key")
except prestige.NotFoundError:
    print("Key not found")
except prestige.PrestigeError as e:
    print(f"Error: {e}")
```

### Feature Flags

Check available features at runtime:

```python
import prestige

print(f"Version: {prestige.__version__}")
print(f"Semantic dedup available: {prestige.SEMANTIC_AVAILABLE}")
print(f"Server available: {prestige.SERVER_AVAILABLE}")
```

## Advanced Usage

### Semantic Deduplication

When built with semantic dedup support:

```python
import prestige

options = prestige.Options()
options.dedup_mode = prestige.DedupMode.SEMANTIC

# Configure semantic model (if available)
if prestige.SEMANTIC_AVAILABLE:
    model = prestige.SemanticModel()
    model.model_path = "/path/to/model.onnx"
    # Additional semantic configuration...

with prestige.open("/tmp/semantic_db", options) as store:
    store.put("key1", "The quick brown fox")
    store.put("key2", "A fast brown fox")
    # May deduplicate if semantically similar
```

### Cache Management

```python
import prestige

options = prestige.Options()
options.default_ttl_seconds = 3600  # 1 hour TTL
options.max_store_bytes = 1024**3   # 1GB max

with prestige.open("/tmp/cache_db", options) as store:
    # Store with custom TTL
    store.put("session:123", "user_data", ttl_seconds=7200)

    # Prune old entries
    deleted = store.prune(max_age_seconds=24 * 3600)
    print(f"Deleted {deleted} expired entries")

    # Clean up unreferenced values
    freed = store.sweep()
    print(f"Freed {freed} unreferenced objects")
```

### Batch Operations

For better performance with multiple operations:

```python
with prestige.open("/tmp/mydb") as store:
    # Batch puts
    data = {f"key_{i}": f"value_{i}" for i in range(1000)}
    for key, value in data.items():
        store.put(key, value)

    # Flush to ensure persistence
    store.flush()

    # Batch gets
    values = [store.get(f"key_{i}", decode=True) for i in range(1000)]
```

### Error Handling

```python
import prestige

try:
    with prestige.open("/tmp/mydb") as store:
        store.put("key", "value")
        value = store.get("key", decode=True)

except prestige.NotFoundError:
    print("Key not found")
except prestige.IOError as e:
    print(f"I/O error: {e}")
except prestige.CorruptionError as e:
    print(f"Database corrupted: {e}")
    # Consider restore from backup
except prestige.PrestigeError as e:
    print(f"Prestige error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Persistence

Data persists across sessions:

```python
# Session 1: Write data
with prestige.open("/tmp/persistent_db") as store:
    store.put("user:123", "John Doe")
    store.put("user:456", "Jane Smith")

# Session 2: Read data (works even after restart)
with prestige.open("/tmp/persistent_db") as store:
    user = store.get("user:123", decode=True)
    print(user)  # "John Doe"
```

## Integration Examples

### As an Embedding Cache

```python
import prestige

class EmbeddingCache:
    def __init__(self, db_path):
        self.store = prestige.open(db_path, default_ttl_seconds=7*24*3600)

    def get_embedding(self, text, model_fn):
        """Get embedding from cache or compute and cache it."""
        try:
            # Try cache first
            return self.store.get(f"emb:{text}")
        except prestige.NotFoundError:
            # Compute and cache
            embedding = model_fn(text)
            self.store.put(f"emb:{text}", embedding)
            return embedding

    def close(self):
        self.store.close()

# Usage
cache = EmbeddingCache("/tmp/embeddings_cache")
try:
    embedding = cache.get_embedding("hello world", my_embedding_fn)
finally:
    cache.close()
```

### Session Store

```python
import prestige
import json

class SessionStore:
    def __init__(self, db_path):
        options = prestige.Options()
        options.default_ttl_seconds = 3600  # 1 hour sessions
        self.store = prestige.open(db_path, options)

    def create_session(self, session_id, data):
        """Create a new session."""
        self.store.put(f"session:{session_id}", json.dumps(data))

    def get_session(self, session_id):
        """Get session data."""
        try:
            data = self.store.get(f"session:{session_id}", decode=True)
            return json.loads(data)
        except prestige.NotFoundError:
            return None

    def delete_session(self, session_id):
        """Delete a session."""
        try:
            self.store.delete(f"session:{session_id}")
        except prestige.NotFoundError:
            pass

    def cleanup(self):
        """Remove expired sessions."""
        return self.store.prune()

# Usage
sessions = SessionStore("/tmp/sessions")
sessions.create_session("abc123", {"user_id": 42, "role": "admin"})
data = sessions.get_session("abc123")
```

## Testing

The Python bindings include a comprehensive test suite:

```bash
cd python
pip install -e ".[dev]"
pytest tests/
```

Run with coverage:

```bash
pytest --cov=prestige --cov-report=html tests/
```

## Performance Tips

1. **Use context managers**: Ensures proper resource cleanup
   ```python
   with prestige.open(path) as store:
       # operations
   ```

2. **Batch operations**: Group multiple puts together when possible

3. **Adjust cache sizes**: Tune `block_cache_bytes` for your workload
   ```python
   options = prestige.Options()
   options.block_cache_bytes = 512 * 1024**2  # 512MB
   ```

4. **Use approximate counts**: Much faster than exact counts
   ```python
   count = store.count_keys(approximate=True)
   ```

5. **Periodic maintenance**: Run `sweep()` and `prune()` during off-peak hours
   ```python
   store.sweep()  # Clean unreferenced values
   store.prune(max_age_seconds=30*24*3600)  # Remove 30+ day old entries
   ```

## Troubleshooting

### Installation Issues

If you encounter build errors:

1. Ensure RocksDB is installed (see [building.md](building.md))
2. Check CMake version is 3.15+
3. Verify C++17 compiler is available

### Runtime Errors

**NotFoundError on get()**
- Use `get(key, default=None)` to avoid exceptions
- Or check with `if key in store:` before getting

**IOError during operations**
- Check disk space
- Verify write permissions on database directory
- Ensure database isn't corrupted (restore from backup if needed)

**Memory issues**
- Reduce `block_cache_bytes` in options
- Set `max_store_bytes` limit
- Run `sweep()` to free unreferenced values

## See Also

- [Architecture](architecture.md): Understanding prestige internals
- [API Reference](api-reference.md): Complete C++ API
- [Cache Semantics](cache-semantics.md): TTL and eviction details
- [Semantic Dedup](semantic-dedup.md): Embedding-based deduplication
