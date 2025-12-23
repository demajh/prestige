#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prestige::internal {

// Search result: object_id + distance
struct SearchResult {
  std::string object_id;  // 16-byte object ID
  float distance;         // L2 distance (lower = more similar)
};

// Statistics about evicted entries
struct EvictionStats {
  size_t entries_evicted = 0;     // Number of entries evicted
  size_t current_size = 0;        // Current number of entries
  size_t max_capacity = 0;        // Maximum capacity
  size_t deleted_count = 0;       // Number of soft-deleted entries
};

// Abstract interface for vector similarity search
class VectorIndex {
 public:
  virtual ~VectorIndex() = default;

  // Add embedding with associated object_id label
  // Returns true on success
  // If max_entries is set and exceeded, evicts LRU entries automatically
  virtual bool Add(const std::vector<float>& embedding,
                   const std::string& object_id) = 0;

  // Search for k nearest neighbors
  // Returns list of (object_id, distance) pairs, sorted by distance ascending
  virtual std::vector<SearchResult> Search(
      const std::vector<float>& query, int k) const = 0;

  // Mark object_id as deleted
  // Note: HNSW doesn't support true deletion, so this marks for filtering
  virtual bool MarkDeleted(const std::string& object_id) = 0;

  // Persistence
  virtual bool Save(const std::string& path) = 0;
  virtual bool Load(const std::string& path) = 0;

  // Stats
  virtual size_t Size() const = 0;
  virtual size_t Dimension() const = 0;
  virtual size_t DeletedCount() const = 0;

  // Set search parameters (e.g., ef_search for HNSW, nprobe for FAISS)
  virtual void SetSearchParam(const std::string& key, int value) = 0;

  // Capacity management
  // Set maximum number of entries (0 = unlimited). When exceeded, LRU eviction occurs.
  virtual void SetMaxEntries(size_t max_entries) = 0;
  virtual size_t GetMaxEntries() const = 0;

  // Get eviction statistics
  virtual EvictionStats GetEvictionStats() const = 0;

  // Compact the index by rebuilding without deleted entries
  // This reclaims memory from soft-deleted vectors
  // Returns true on success
  virtual bool Compact() = 0;
};

// Factory function for HNSW index (hnswlib)
// - dimension: embedding dimension (e.g., 384)
// - max_elements: initial capacity (will grow automatically)
// - m: max connections per node (default 16)
// - ef_construction: build-time search depth (default 200)
// - max_entries: maximum entries before LRU eviction (0 = unlimited)
std::unique_ptr<VectorIndex> CreateHNSWIndex(
    size_t dimension,
    size_t max_elements = 10000,
    int m = 16,
    int ef_construction = 200,
    size_t max_entries = 0);

#ifdef PRESTIGE_USE_FAISS
// Factory function for FAISS IVF+PQ index
// - dimension: embedding dimension (e.g., 384)
// - nlist: number of IVF clusters
// - pq_m: PQ sub-quantizers (must divide dimension)
// - pq_nbits: bits per sub-quantizer
std::unique_ptr<VectorIndex> CreateFAISSIndex(
    size_t dimension,
    int nlist = 100,
    int pq_m = 48,
    int pq_nbits = 8);
#endif

}  // namespace prestige::internal
