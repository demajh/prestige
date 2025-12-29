# Prestige HTTP Server

The prestige server exposes all Store operations via a REST/JSON API.

## Building

Enable the server build option:

```bash
cmake -B build -DPRESTIGE_BUILD_SERVER=ON
cmake --build build
```

**Prerequisites**: Drogon framework must be installed. See [Drogon installation](https://github.com/drogonframework/drogon/wiki/ENG-02-Installation).

## Running

### Basic Usage

```bash
# Start server with database path
prestige-server --db-path /var/lib/prestige/data --port 8080

# With config file
prestige-server --config /etc/prestige/server.yaml
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config, -c <path>` | Path to YAML config file | none |
| `--host <addr>` | Bind address | `0.0.0.0` |
| `--port, -p <port>` | Listen port | `8080` |
| `--threads <n>` | Worker threads (0=auto) | `0` |
| `--db-path <path>` | Database path | required |
| `--log-level <level>` | Log level (debug, info, warn, error) | `info` |
| `--help, -h` | Show help | - |

## Configuration File

Create a YAML config file:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  threads: 0  # 0 = auto-detect CPU cores
  log_level: info

store:
  path: "/var/lib/prestige/data"
  block_cache_bytes: 268435456  # 256MB
  default_ttl_seconds: 3600
  max_store_bytes: 10737418240  # 10GB

metrics:
  enabled: true
  path: "/metrics"
```

## REST API

### KV Store Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `PUT` | `/api/v1/kv/{key}` | Store a value |
| `GET` | `/api/v1/kv/{key}` | Retrieve a value |
| `DELETE` | `/api/v1/kv/{key}` | Delete a key |
| `GET` | `/api/v1/kv?prefix=&limit=` | List keys |
| `GET` | `/api/v1/kv/_count` | Count keys (exact, O(N)) |
| `GET` | `/api/v1/kv/_count/approx` | Count keys (fast estimate, O(1)) |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/admin/sweep` | Sweep expired/orphaned objects |
| `POST` | `/api/v1/admin/prune` | Prune by age/idle time |
| `POST` | `/api/v1/admin/evict` | LRU eviction to target bytes |
| `POST` | `/api/v1/admin/flush` | Flush to disk |

### Health Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Readiness check with stats |
| `GET` | `/metrics` | Prometheus metrics |

## Examples

### Store and Retrieve Values

```bash
# Store a value (raw body)
curl -X PUT http://localhost:8080/api/v1/kv/mykey \
  -d "hello world"

# Store a value (JSON)
curl -X PUT http://localhost:8080/api/v1/kv/mykey \
  -H "Content-Type: application/json" \
  -d '{"value": "hello world"}'

# Get value (raw)
curl http://localhost:8080/api/v1/kv/mykey

# Get value (JSON)
curl http://localhost:8080/api/v1/kv/mykey \
  -H "Accept: application/json"

# Delete
curl -X DELETE http://localhost:8080/api/v1/kv/mykey

# List keys with prefix
curl "http://localhost:8080/api/v1/kv?prefix=user:&limit=100"

# Count (exact)
curl http://localhost:8080/api/v1/kv/_count

# Count (approximate, fast)
curl http://localhost:8080/api/v1/kv/_count/approx
```

### Admin Operations

```bash
# Sweep expired/orphaned objects
curl -X POST http://localhost:8080/api/v1/admin/sweep

# Prune by age and idle time
curl -X POST http://localhost:8080/api/v1/admin/prune \
  -H "Content-Type: application/json" \
  -d '{"max_age_s": 86400, "max_idle_s": 3600}'

# Evict LRU to target size
curl -X POST http://localhost:8080/api/v1/admin/evict \
  -H "Content-Type: application/json" \
  -d '{"target_bytes": 1073741824}'

# Flush to disk
curl -X POST http://localhost:8080/api/v1/admin/flush
```

### Health Checks

```bash
# Liveness (for k8s liveness probe)
curl http://localhost:8080/health

# Readiness with stats (for k8s readiness probe)
curl http://localhost:8080/health/ready

# Prometheus metrics
curl http://localhost:8080/metrics
```

## Metrics

When metrics are enabled, the `/metrics` endpoint exposes Prometheus-format metrics:

### Store Metrics

- `prestige_keys_total` - Total number of keys
- `prestige_objects_total` - Total unique objects
- `prestige_store_bytes` - Total store size in bytes

### HTTP Metrics

- `prestige_http_requests_total{method,path,status}` - Request counter
- `prestige_http_request_duration_ms` - Request latency histogram

## Graceful Shutdown

The server handles `SIGTERM`, `SIGINT`, and `SIGHUP` signals for graceful shutdown:

1. Stop accepting new connections
2. Drain in-flight requests
3. Flush store to disk
4. Exit cleanly

## Thread Safety

The server is fully thread-safe:
- All KV operations use RocksDB TransactionDB
- Metrics collection is mutex-protected
- Drogon's async model handles concurrent requests efficiently
