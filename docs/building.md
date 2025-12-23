# Building from Source

## Dependencies

### Required

- CMake 3.16+
- C++17 compiler (GCC 8+, Clang 7+, or MSVC 2019+)
- RocksDB with TransactionDB support
- OpenSSL (for SHA-256 hashing)

### Optional (for semantic mode)

- ONNX Runtime (C++ API)
- hnswlib (fetched automatically via CMake)

## Install Dependencies

### Ubuntu/Debian

```bash
sudo apt-get install librocksdb-dev libssl-dev

# For semantic mode
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/
sudo ldconfig
```

### macOS

```bash
brew install rocksdb

# For semantic mode
brew install onnxruntime
```

## Build

```bash
git clone https://github.com/demajh/prestige.git
cd prestige
mkdir build && cd build

# Basic build (exact dedup only)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# With semantic dedup (Linux)
cmake .. -DCMAKE_BUILD_TYPE=Release -DPRESTIGE_ENABLE_SEMANTIC=ON
make -j$(nproc)
sudo make install

# With semantic dedup (macOS with Homebrew)
cmake .. -DCMAKE_BUILD_TYPE=Release -DPRESTIGE_ENABLE_SEMANTIC=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/opt/homebrew/include/onnxruntime \
  -DONNXRUNTIME_LIBRARY=/opt/homebrew/lib/libonnxruntime.dylib
make -j$(sysctl -n hw.ncpu)
sudo make install
```

## Build Outputs

- `libprestige_uvs.a` / `libprestige_uvs.so` (library)
- `prestige_cli` (CLI tool)
- `prestige_example_basic` (example program)
- `prestige_example_observability` (observability example)
- `prestige_example_semantic` (semantic example, if enabled)

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `PRESTIGE_BUILD_TOOLS` | ON | Build CLI tool |
| `PRESTIGE_BUILD_EXAMPLES` | ON | Build example programs |
| `PRESTIGE_ENABLE_SEMANTIC` | OFF | Enable semantic deduplication |
| `PRESTIGE_USE_FAISS` | OFF | Use FAISS backend (requires FAISS) |

## Using in Your Project

### CMake

```cmake
find_package(prestige_uvs REQUIRED)
target_link_libraries(your_target PRIVATE prestige::prestige_uvs)
```

### pkg-config

```bash
g++ -std=c++17 my_app.cpp $(pkg-config --cflags --libs prestige_uvs) -o my_app
```

## Pre-built Binaries

### Ubuntu/Debian (apt)

```bash
curl -fsSL https://demajh.github.io/prestige/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/prestige-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/prestige-archive-keyring.gpg] https://demajh.github.io/prestige stable main" | sudo tee /etc/apt/sources.list.d/prestige.list
sudo apt-get update
sudo apt-get install prestige prestige-dev
```

### Direct Download

```bash
# Linux x64
curl -LO https://github.com/demajh/prestige/releases/latest/download/prestige-VERSION-linux-x64.tar.gz
sudo tar -xzf prestige-VERSION-linux-x64.tar.gz -C /

# macOS ARM64
curl -LO https://github.com/demajh/prestige/releases/latest/download/prestige-VERSION-macos-arm64.tar.gz
sudo tar -xzf prestige-VERSION-macos-arm64.tar.gz -C /
```
