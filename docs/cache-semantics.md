# Cache Semantics

Prestige supports optional TTL-based expiration and LRU-based eviction for cache-like use cases. Both are disabled by default.

## TTL Expiration

When `default_ttl_seconds > 0`, objects expire based on their creation time:

- Expired objects return `NotFound` on `Get()` (same behavior as missing keys)
- All keys pointing to an expired object will return `NotFound`
- Expired objects are not automatically deleted; use `Sweep()` or `Prune()` to reclaim space

```cpp
prestige::Options opt;
opt.default_ttl_seconds = 3600;  // 1 hour TTL
```

## LRU Eviction

When `max_store_bytes > 0`, you can trigger LRU eviction to reclaim space:

- Each `Get()` updates the object's last-access timestamp (if `track_access_time=true`)
- `EvictLRU(target_bytes)` removes least-recently-used objects until the store is at or below the target size
- Eviction removes both the objects and all user keys pointing to them

```cpp
prestige::Options opt;
opt.max_store_bytes = 1024 * 1024 * 1024;  // 1 GB limit
opt.eviction_target_ratio = 0.8;           // Evict to 80% when triggered

// Later, trigger eviction manually:
uint64_t evicted = 0;
db->EvictLRU(opt.max_store_bytes * opt.eviction_target_ratio, &evicted);
```

## Cache Management Methods

| Method | Description |
|--------|-------------|
| `Sweep(uint64_t* deleted)` | Delete all expired and orphaned (refcount=0) objects |
| `Prune(max_age_s, max_idle_s, uint64_t* deleted)` | Delete objects older than `max_age_s` or not accessed for `max_idle_s` |
| `EvictLRU(target_bytes, uint64_t* evicted)` | Evict LRU objects until store size <= target_bytes |
| `GetHealth(HealthStats* stats)` | Get store health statistics |
| `GetTotalStoreBytes()` | Get current approximate store size |

## Health Statistics

`GetHealth()` returns a `HealthStats` struct:

```cpp
struct HealthStats {
  uint64_t total_keys;          // Number of user keys
  uint64_t total_objects;       // Number of unique deduplicated objects
  uint64_t total_bytes;         // Total size of all objects
  uint64_t expired_objects;     // Objects past TTL (not yet swept)
  uint64_t orphaned_objects;    // Objects with refcount=0 (not yet GC'd)
  uint64_t oldest_object_age_s; // Age of oldest object in seconds
  uint64_t newest_access_age_s; // Time since most recent access
  double dedup_ratio;           // Ratio of keys to objects (higher = more dedup)
};
```

## CLI Commands

```bash
# Show store health stats
prestige_cli ./db health

# Delete expired/orphaned objects
prestige_cli ./db sweep

# Delete objects older than 1hr OR idle for 30min
prestige_cli ./db prune 3600 1800

# Evict LRU until store is <= 1GB
prestige_cli ./db evict 1073741824
```
