# CLI Reference

The `prestige_cli` tool provides command-line access to prestige stores.

## Basic Usage

```bash
prestige_cli <db_path> <command> [args...]
```

## Commands

### Put

Store a value under a key:

```bash
prestige_cli ./db put <key> <value>
```

### Get

Retrieve a value by key (prints raw bytes to stdout):

```bash
prestige_cli ./db get <key>
```

### Delete

Delete a key (and GC the object if unreferenced):

```bash
prestige_cli ./db del <key>
```

### List Keys

List all keys, optionally with a prefix and limit:

```bash
prestige_cli ./db keys                    # List all keys
prestige_cli ./db keys "user:" 100        # List up to 100 keys with prefix "user:"
```

### Count

Count keys and unique values (exact, requires full scan):

```bash
prestige_cli ./db count
# Output: keys=1000 unique_values=500
```

### Count (Approximate)

Get approximate counts using RocksDB's internal estimates (O(1), no scan):

```bash
prestige_cli ./db count-approx
# Output: keys~=1000 unique_values~=500 bytes~=1048576 (approximate)
```

Use this for large stores where full scans take too long. Estimates may be 10-50% off, especially after many deletes.

### Health

Show store health statistics:

```bash
prestige_cli ./db health
```

Output includes:
- Total keys and unique objects
- Total bytes stored
- Expired and orphaned object counts
- Deduplication ratio
- Oldest object age
- Time since most recent access

### Sweep

Delete all expired (past TTL) and orphaned (refcount=0) objects:

```bash
prestige_cli ./db sweep
```

### Prune

Delete objects based on age and/or idle time:

```bash
prestige_cli ./db prune <max_age_seconds> <max_idle_seconds>
```

Examples:
```bash
prestige_cli ./db prune 3600 0      # Delete objects older than 1 hour
prestige_cli ./db prune 0 1800      # Delete objects idle for 30+ minutes
prestige_cli ./db prune 3600 1800   # Delete objects older than 1hr OR idle 30min
```

### Evict

Evict least-recently-used objects until store size is at or below target:

```bash
prestige_cli ./db evict <target_bytes>
```

Example:
```bash
prestige_cli ./db evict 1073741824  # Evict until store is <= 1GB
```

## Examples

```bash
# Basic CRUD
prestige_cli ./mydb put user:1 '{"name": "Alice"}'
prestige_cli ./mydb put user:2 '{"name": "Bob"}'
prestige_cli ./mydb get user:1
prestige_cli ./mydb del user:1

# Deduplication in action
prestige_cli ./mydb put k1 "Hello World"
prestige_cli ./mydb put k2 "Hello World"  # Deduplicates with k1
prestige_cli ./mydb count                  # Shows 2 keys, 1 unique value

# Cache management
prestige_cli ./mydb health
prestige_cli ./mydb sweep
prestige_cli ./mydb prune 86400 3600      # Delete old/idle objects
prestige_cli ./mydb evict 536870912       # Evict to 512MB
```
