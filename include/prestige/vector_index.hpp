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

// Abstract interface for vector similarity search
class VectorIndex {
 public:
  virtual ~VectorIndex() = default;

  // Add embedding with associated object_id label
  // Returns true on success
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

  // Set search parameters (e.g., ef_search for HNSW, nprobe for FAISS)
  virtual void SetSearchParam(const std::string& key, int value) = 0;
};

// Factory function for HNSW index (hnswlib)
// - dimension: embedding dimension (e.g., 384)
// - max_elements: initial capacity (will grow automatically)
// - m: max connections per node (default 16)
// - ef_construction: build-time search depth (default 200)
std::unique_ptr<VectorIndex> CreateHNSWIndex(
    size_t dimension,
    size_t max_elements = 10000,
    int m = 16,
    int ef_construction = 200);

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
