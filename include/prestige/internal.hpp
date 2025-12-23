#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <chrono>
#include <vector>

#include <openssl/evp.h>
#include <rocksdb/status.h>

namespace prestige::internal {

// Monotonic timestamp helper for metrics/tracing (microseconds).
inline uint64_t NowMicros() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
}

// Wall-clock timestamp for TTL calculations (microseconds since epoch).
// Use this for TTL, not NowMicros(), because TTL needs real time.
inline uint64_t WallClockMicros() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
}
  
// SHA-256 wrapper using OpenSSL's EVP API.
// This provides a battle-tested, FIPS-validated implementation.
class Sha256 {
 public:
  static constexpr size_t kDigestBytes = 32;

  static std::array<uint8_t, kDigestBytes> Digest(std::string_view data) {
    std::array<uint8_t, kDigestBytes> out{};
    unsigned int len = 0;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx) {
      if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) &&
          EVP_DigestUpdate(ctx, data.data(), data.size()) &&
          EVP_DigestFinal_ex(ctx, out.data(), &len)) {
        // Success
      }
      EVP_MD_CTX_free(ctx);
    }

    return out;
  }
};

inline std::string ToBytes(const uint8_t* p, size_t n) {
  return std::string(reinterpret_cast<const char*>(p), n);
}

inline std::string EncodeU64LE(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) {
    s[i] = static_cast<char>(v & 0xffu);
    v >>= 8;
  }
  return s;
}

inline bool DecodeU64LE(std::string_view s, uint64_t* out) {
  if (s.size() != 8) return false;
  uint64_t v = 0;
  // little endian decode
  for (int i = 7; i >= 0; --i) {
    v <<= 8;
    v |= static_cast<uint8_t>(s[static_cast<size_t>(i)]);
  }
  *out = v;
  return true;
}

inline std::array<uint8_t, 16> RandomObjectId128() {
  thread_local std::mt19937_64 rng([]{
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    seed ^= static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return seed;
  }());

  std::array<uint8_t, 16> id{};
  uint64_t a = rng();
  uint64_t b = rng();
  std::memcpy(id.data() + 0, &a, 8);
  std::memcpy(id.data() + 8, &b, 8);
  return id;
}

inline bool IsRetryableTxnStatus(const rocksdb::Status& s) {
  return s.IsBusy() || s.IsTimedOut() || s.IsTryAgain() || s.IsAborted();
}

// ---------------------------------------------------------------------------
// Object metadata for TTL and LRU tracking
// ---------------------------------------------------------------------------

// Metadata stored per object in prestige_object_meta CF.
// Serialization format:
//   [digest_key_len:4 LE][digest_key bytes][created_at_us:8 LE][last_accessed_us:8 LE][size_bytes:8 LE]
struct ObjectMeta {
  std::string digest_key;       // SHA-256 hash or semantic marker
  uint64_t created_at_us = 0;   // Wall-clock creation time (microseconds)
  uint64_t last_accessed_us = 0; // Wall-clock last access time (for LRU)
  uint64_t size_bytes = 0;      // Object size in bytes

  std::string Serialize() const {
    std::string out;
    out.reserve(4 + digest_key.size() + 24);

    // digest_key length (4 bytes, little-endian)
    uint32_t len = static_cast<uint32_t>(digest_key.size());
    out.push_back(static_cast<char>(len & 0xff));
    out.push_back(static_cast<char>((len >> 8) & 0xff));
    out.push_back(static_cast<char>((len >> 16) & 0xff));
    out.push_back(static_cast<char>((len >> 24) & 0xff));

    // digest_key bytes
    out.append(digest_key);

    // created_at_us (8 bytes, little-endian)
    out.append(EncodeU64LE(created_at_us));

    // last_accessed_us (8 bytes, little-endian)
    out.append(EncodeU64LE(last_accessed_us));

    // size_bytes (8 bytes, little-endian)
    out.append(EncodeU64LE(size_bytes));

    return out;
  }

  static bool Deserialize(std::string_view data, ObjectMeta* out) {
    if (!out) return false;

    // Minimum size: 4 (len) + 0 (digest) + 24 (timestamps + size)
    if (data.size() < 28) {
      // Try legacy format: just digest_key bytes
      out->digest_key = std::string(data);
      out->created_at_us = 0;
      out->last_accessed_us = 0;
      out->size_bytes = 0;
      return true;
    }

    // Read digest_key length
    uint32_t len = static_cast<uint8_t>(data[0]) |
                   (static_cast<uint8_t>(data[1]) << 8) |
                   (static_cast<uint8_t>(data[2]) << 16) |
                   (static_cast<uint8_t>(data[3]) << 24);

    // Validate total size
    if (data.size() != 4 + len + 24) {
      // Doesn't match new format, treat as legacy
      out->digest_key = std::string(data);
      out->created_at_us = 0;
      out->last_accessed_us = 0;
      out->size_bytes = 0;
      return true;
    }

    // Read digest_key
    out->digest_key = std::string(data.substr(4, len));

    // Read timestamps and size
    size_t offset = 4 + len;
    if (!DecodeU64LE(data.substr(offset, 8), &out->created_at_us)) return false;
    offset += 8;
    if (!DecodeU64LE(data.substr(offset, 8), &out->last_accessed_us)) return false;
    offset += 8;
    if (!DecodeU64LE(data.substr(offset, 8), &out->size_bytes)) return false;

    return true;
  }

  // Check if this is legacy format (no timestamps)
  bool IsLegacy() const {
    return created_at_us == 0 && last_accessed_us == 0 && size_bytes == 0;
  }
};

// ---------------------------------------------------------------------------
// LRU index key helpers
// ---------------------------------------------------------------------------

// Create LRU index key: [timestamp:8 big-endian][object_id]
// Big-endian ensures oldest (smallest) timestamps sort first when iterating.
inline std::string MakeLRUKey(uint64_t timestamp_us, std::string_view obj_id) {
  std::string key;
  key.reserve(8 + obj_id.size());

  // Big-endian timestamp for proper sort order
  for (int i = 7; i >= 0; --i) {
    key.push_back(static_cast<char>((timestamp_us >> (i * 8)) & 0xff));
  }

  key.append(obj_id);
  return key;
}

// Parse LRU index key to extract timestamp and object_id.
inline bool ParseLRUKey(std::string_view key, uint64_t* timestamp_us, std::string* obj_id) {
  if (key.size() < 8) return false;

  // Big-endian decode
  uint64_t ts = 0;
  for (int i = 0; i < 8; ++i) {
    ts = (ts << 8) | static_cast<uint8_t>(key[i]);
  }

  *timestamp_us = ts;
  *obj_id = std::string(key.substr(8));
  return true;
}

// ---------------------------------------------------------------------------
// Embedding utilities for semantic dedup
// ---------------------------------------------------------------------------

// Embedding dimensions for supported models
constexpr size_t kMiniLMDimensions = 384;
constexpr size_t kBGESmallDimensions = 384;
constexpr size_t kDefaultEmbeddingDimensions = 384;

// Serialize embedding vector to bytes (little-endian floats)
inline std::string SerializeEmbedding(const std::vector<float>& embedding) {
  std::string out;
  out.resize(embedding.size() * sizeof(float));
  std::memcpy(out.data(), embedding.data(), out.size());
  return out;
}

// Deserialize bytes to embedding vector
inline bool DeserializeEmbedding(std::string_view bytes, std::vector<float>* out) {
  if (bytes.size() % sizeof(float) != 0) return false;
  size_t count = bytes.size() / sizeof(float);
  out->resize(count);
  std::memcpy(out->data(), bytes.data(), bytes.size());
  return true;
}

// Compute cosine similarity between two embeddings
// Returns value in [-1.0, 1.0]; 1.0 means identical directions
inline float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0f;

  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  if (denom < 1e-12f) return 0.0f;
  return dot / denom;
}

// Convert cosine similarity to L2 distance (for normalized vectors)
// L2^2 = 2 - 2*cos(similarity) for unit vectors
inline float CosineToL2Distance(float cosine_sim) {
  return std::sqrt(2.0f - 2.0f * cosine_sim);
}

// Convert L2 distance to cosine similarity (for normalized vectors)
inline float L2DistanceToCosine(float l2_dist) {
  return 1.0f - (l2_dist * l2_dist) / 2.0f;
}

}  // namespace prestige::internal
