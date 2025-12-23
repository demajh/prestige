#include <prestige/vector_index.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace prestige::internal {

namespace {
// Get current time in microseconds for LRU tracking
inline uint64_t NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
}  // namespace

// HNSW implementation using hnswlib with LRU eviction support
class HNSWIndex : public VectorIndex {
 public:
  HNSWIndex(size_t dimension, size_t max_elements, int m, int ef_construction,
            size_t max_entries)
      : dimension_(dimension),
        max_elements_(max_elements),
        m_(m),
        ef_construction_(ef_construction),
        ef_search_(50),
        max_entries_(max_entries) {
    // Use L2 space (Euclidean distance)
    // For cosine similarity with normalized vectors, L2 and cosine are equivalent
    space_ = std::make_unique<hnswlib::L2Space>(dimension);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(), max_elements, m, ef_construction);
    index_->setEf(ef_search_);
  }

  bool Add(const std::vector<float>& embedding,
           const std::string& object_id) override {
    std::lock_guard<std::mutex> lock(mutex_);

    if (embedding.size() != dimension_) {
      return false;
    }

    // Check if we need to evict before adding
    if (max_entries_ > 0 && LiveCount() >= max_entries_) {
      EvictLRULocked();
    }

    // Grow index if needed
    if (current_count_ >= max_elements_) {
      size_t new_max = max_elements_ * 2;
      index_->resizeIndex(new_max);
      max_elements_ = new_max;
    }

    // Use internal label (sequential ID)
    hnswlib::labeltype label = current_count_;

    try {
      index_->addPoint(embedding.data(), label);
    } catch (const std::exception&) {
      return false;
    }

    // Map label <-> object_id
    label_to_object_id_[label] = object_id;
    object_id_to_label_[object_id] = label;
    current_count_++;

    // Track access time for LRU
    uint64_t now = NowMicros();
    lru_list_.push_back({label, now});
    label_to_lru_iter_[label] = std::prev(lru_list_.end());

    return true;
  }

  std::vector<SearchResult> Search(const std::vector<float>& query,
                                   int k) const override {
    std::lock_guard<std::mutex> lock(mutex_);

    if (query.size() != dimension_ || current_count_ == 0) {
      return {};
    }

    // Request more candidates to account for deleted items
    int search_k = std::min(static_cast<size_t>(k * 2), current_count_);

    std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
    try {
      result = index_->searchKnn(query.data(), search_k);
    } catch (const std::exception&) {
      return {};
    }

    std::vector<SearchResult> results;
    results.reserve(k);

    // Extract results, filtering deleted items
    while (!result.empty() && results.size() < static_cast<size_t>(k)) {
      auto [distance, label] = result.top();
      result.pop();

      // Skip deleted items
      if (deleted_labels_.count(label)) {
        continue;
      }

      auto it = label_to_object_id_.find(label);
      if (it != label_to_object_id_.end()) {
        results.push_back({it->second, distance});
        // Update LRU on access (move to back of list)
        UpdateLRULocked(label);
      }
    }

    // Results come from priority queue (max-heap), so reverse for ascending order
    std::reverse(results.begin(), results.end());
    return results;
  }

  bool MarkDeleted(const std::string& object_id) override {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = object_id_to_label_.find(object_id);
    if (it == object_id_to_label_.end()) {
      return false;
    }

    hnswlib::labeltype label = it->second;
    deleted_labels_.insert(label);

    // Remove from LRU tracking
    auto lru_it = label_to_lru_iter_.find(label);
    if (lru_it != label_to_lru_iter_.end()) {
      lru_list_.erase(lru_it->second);
      label_to_lru_iter_.erase(lru_it);
    }

    // Remove from mappings to free memory
    label_to_object_id_.erase(label);
    object_id_to_label_.erase(it);

    return true;
  }

  bool Save(const std::string& path) override {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
      // Save HNSW index
      index_->saveIndex(path);

      // Save metadata (object_id mappings, deleted set, and LRU data)
      std::string meta_path = path + ".meta";
      std::ofstream ofs(meta_path, std::ios::binary);
      if (!ofs) return false;

      // Write version for forward compatibility
      uint32_t version = 2;  // Version 2 includes LRU data
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

      // Write dimension and counts
      ofs.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
      ofs.write(reinterpret_cast<const char*>(&current_count_), sizeof(current_count_));

      size_t mapping_count = label_to_object_id_.size();
      ofs.write(reinterpret_cast<const char*>(&mapping_count), sizeof(mapping_count));

      size_t deleted_count = deleted_labels_.size();
      ofs.write(reinterpret_cast<const char*>(&deleted_count), sizeof(deleted_count));

      // Write max_entries and eviction stats
      ofs.write(reinterpret_cast<const char*>(&max_entries_), sizeof(max_entries_));
      ofs.write(reinterpret_cast<const char*>(&total_evicted_), sizeof(total_evicted_));

      // Write label -> object_id mappings with LRU timestamps
      for (const auto& [label, obj_id] : label_to_object_id_) {
        ofs.write(reinterpret_cast<const char*>(&label), sizeof(label));
        size_t len = obj_id.size();
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        ofs.write(obj_id.data(), len);

        // Write LRU timestamp for this label
        uint64_t lru_time = 0;
        auto lru_it = label_to_lru_iter_.find(label);
        if (lru_it != label_to_lru_iter_.end()) {
          lru_time = lru_it->second->access_time;
        }
        ofs.write(reinterpret_cast<const char*>(&lru_time), sizeof(lru_time));
      }

      // Write deleted labels
      for (hnswlib::labeltype label : deleted_labels_) {
        ofs.write(reinterpret_cast<const char*>(&label), sizeof(label));
      }

      // Ensure data is flushed to disk
      ofs.flush();
      if (!ofs.good()) return false;

      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

  bool Load(const std::string& path) override {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
      // Check if index file exists
      std::ifstream check(path);
      if (!check.good()) {
        // No existing index, start fresh
        return true;
      }
      check.close();

      // Load HNSW index
      index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
          space_.get(), path);
      index_->setEf(ef_search_);

      // Load metadata
      std::string meta_path = path + ".meta";
      std::ifstream ifs(meta_path, std::ios::binary);
      if (!ifs) {
        // Index exists but no metadata - inconsistent state
        return false;
      }

      // Clear existing state
      label_to_object_id_.clear();
      object_id_to_label_.clear();
      deleted_labels_.clear();
      lru_list_.clear();
      label_to_lru_iter_.clear();

      // Peek at first bytes to detect version
      uint32_t version = 1;  // Default to version 1 (no version field)
      size_t first_size;
      ifs.read(reinterpret_cast<char*>(&first_size), sizeof(first_size));

      // Version 2 starts with a small version number (2)
      // Version 1 starts with dimension (typically 384)
      if (first_size == 2) {
        version = 2;
        // Read dimension
        ifs.read(reinterpret_cast<char*>(&first_size), sizeof(first_size));
      }

      size_t saved_dimension = first_size;
      if (saved_dimension != dimension_) {
        return false;  // Dimension mismatch
      }

      ifs.read(reinterpret_cast<char*>(&current_count_), sizeof(current_count_));

      size_t mapping_count = current_count_;  // Version 1 compatibility
      size_t deleted_count;

      if (version >= 2) {
        ifs.read(reinterpret_cast<char*>(&mapping_count), sizeof(mapping_count));
      }
      ifs.read(reinterpret_cast<char*>(&deleted_count), sizeof(deleted_count));

      if (version >= 2) {
        // Read max_entries and eviction stats
        ifs.read(reinterpret_cast<char*>(&max_entries_), sizeof(max_entries_));
        ifs.read(reinterpret_cast<char*>(&total_evicted_), sizeof(total_evicted_));
      }

      // Read label -> object_id mappings (with optional LRU timestamps for v2)
      std::vector<std::pair<hnswlib::labeltype, uint64_t>> lru_entries;
      for (size_t i = 0; i < mapping_count; ++i) {
        hnswlib::labeltype label;
        ifs.read(reinterpret_cast<char*>(&label), sizeof(label));
        size_t len;
        ifs.read(reinterpret_cast<char*>(&len), sizeof(len));

        // Validate length to prevent allocation attacks
        if (len > 1024 * 1024) {  // 1MB max for object ID
          return false;
        }

        std::string obj_id(len, '\0');
        ifs.read(obj_id.data(), len);

        label_to_object_id_[label] = obj_id;
        object_id_to_label_[obj_id] = label;

        if (version >= 2) {
          uint64_t lru_time;
          ifs.read(reinterpret_cast<char*>(&lru_time), sizeof(lru_time));
          lru_entries.emplace_back(label, lru_time);
        }
      }

      // Read deleted labels
      for (size_t i = 0; i < deleted_count; ++i) {
        hnswlib::labeltype label;
        ifs.read(reinterpret_cast<char*>(&label), sizeof(label));
        deleted_labels_.insert(label);
      }

      // Rebuild LRU list (sorted by access time for v2, current time for v1)
      if (version >= 2) {
        // Sort by access time (oldest first)
        std::sort(lru_entries.begin(), lru_entries.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        for (const auto& [label, access_time] : lru_entries) {
          lru_list_.push_back({label, access_time});
          label_to_lru_iter_[label] = std::prev(lru_list_.end());
        }
      } else {
        // Version 1: assign current time to all entries
        uint64_t now = NowMicros();
        for (const auto& [label, obj_id] : label_to_object_id_) {
          lru_list_.push_back({label, now});
          label_to_lru_iter_[label] = std::prev(lru_list_.end());
        }
      }

      max_elements_ = index_->max_elements_;
      return ifs.good();
    } catch (const std::exception&) {
      return false;
    }
  }

  size_t Size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return LiveCount();
  }

  size_t Dimension() const override {
    return dimension_;
  }

  size_t DeletedCount() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return deleted_labels_.size();
  }

  void SetSearchParam(const std::string& key, int value) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (key == "ef_search" || key == "ef") {
      ef_search_ = value;
      index_->setEf(value);
    }
  }

  void SetMaxEntries(size_t max_entries) override {
    std::lock_guard<std::mutex> lock(mutex_);
    max_entries_ = max_entries;
    // Evict if we're now over capacity
    while (max_entries_ > 0 && LiveCount() > max_entries_) {
      EvictLRULocked();
    }
  }

  size_t GetMaxEntries() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_entries_;
  }

  EvictionStats GetEvictionStats() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    EvictionStats stats;
    stats.entries_evicted = total_evicted_;
    stats.current_size = LiveCount();
    stats.max_capacity = max_entries_;
    stats.deleted_count = deleted_labels_.size();
    return stats;
  }

  bool Compact() override {
    std::lock_guard<std::mutex> lock(mutex_);

    if (deleted_labels_.empty()) {
      return true;  // Nothing to compact
    }

    try {
      // Build a new index with only live entries
      auto new_space = std::make_unique<hnswlib::L2Space>(dimension_);
      size_t live_count = LiveCount();
      size_t new_max = std::max(live_count * 2, static_cast<size_t>(10000));

      auto new_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
          new_space.get(), new_max, m_, ef_construction_);
      new_index->setEf(ef_search_);

      // Re-add all non-deleted entries with new sequential labels
      std::unordered_map<hnswlib::labeltype, std::string> new_label_to_object_id;
      std::unordered_map<std::string, hnswlib::labeltype> new_object_id_to_label;
      std::list<LRUEntry> new_lru_list;
      std::unordered_map<hnswlib::labeltype, std::list<LRUEntry>::iterator> new_label_to_lru_iter;

      hnswlib::labeltype new_label = 0;
      for (const auto& entry : lru_list_) {
        hnswlib::labeltype old_label = entry.label;

        // Skip deleted entries
        if (deleted_labels_.count(old_label)) {
          continue;
        }

        // Get the embedding from the old index
        std::vector<float> embedding = index_->getDataByLabel<float>(old_label);
        if (embedding.empty()) continue;

        // Add to new index
        new_index->addPoint(embedding.data(), new_label);

        // Update mappings
        auto it = label_to_object_id_.find(old_label);
        if (it != label_to_object_id_.end()) {
          new_label_to_object_id[new_label] = it->second;
          new_object_id_to_label[it->second] = new_label;

          // Update LRU tracking
          new_lru_list.push_back({new_label, entry.access_time});
          new_label_to_lru_iter[new_label] = std::prev(new_lru_list.end());
        }

        new_label++;
      }

      // Swap in the new data structures
      space_ = std::move(new_space);
      index_ = std::move(new_index);
      label_to_object_id_ = std::move(new_label_to_object_id);
      object_id_to_label_ = std::move(new_object_id_to_label);
      lru_list_ = std::move(new_lru_list);
      label_to_lru_iter_ = std::move(new_label_to_lru_iter);
      deleted_labels_.clear();
      current_count_ = new_label;
      max_elements_ = new_max;

      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

 private:
  // LRU entry: tracks label and access time
  struct LRUEntry {
    hnswlib::labeltype label;
    uint64_t access_time;
  };

  // Get live (non-deleted) count - caller must hold mutex
  size_t LiveCount() const {
    return label_to_object_id_.size();
  }

  // Update LRU on access - caller must hold mutex
  void UpdateLRULocked(hnswlib::labeltype label) const {
    auto it = label_to_lru_iter_.find(label);
    if (it != label_to_lru_iter_.end()) {
      // Move to back of list (most recently used)
      it->second->access_time = NowMicros();
      lru_list_.splice(lru_list_.end(), lru_list_, it->second);
    }
  }

  // Evict least recently used entry - caller must hold mutex
  void EvictLRULocked() {
    if (lru_list_.empty()) return;

    // Get the oldest entry (front of list)
    const LRUEntry& oldest = lru_list_.front();
    hnswlib::labeltype label = oldest.label;

    // Mark as deleted in HNSW (soft delete)
    deleted_labels_.insert(label);

    // Remove from mappings
    auto it = label_to_object_id_.find(label);
    if (it != label_to_object_id_.end()) {
      object_id_to_label_.erase(it->second);
      label_to_object_id_.erase(it);
    }

    // Remove from LRU tracking
    label_to_lru_iter_.erase(label);
    lru_list_.pop_front();

    total_evicted_++;
  }

  size_t dimension_;
  size_t max_elements_;
  int m_;
  int ef_construction_;
  int ef_search_;
  size_t max_entries_ = 0;        // 0 = unlimited
  size_t total_evicted_ = 0;      // Total entries evicted since creation

  std::unique_ptr<hnswlib::L2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

  // Mapping between internal labels and object_ids
  std::unordered_map<hnswlib::labeltype, std::string> label_to_object_id_;
  std::unordered_map<std::string, hnswlib::labeltype> object_id_to_label_;
  std::unordered_set<hnswlib::labeltype> deleted_labels_;

  // LRU tracking: ordered list (oldest at front) and iterator map for O(1) access
  mutable std::list<LRUEntry> lru_list_;
  mutable std::unordered_map<hnswlib::labeltype, std::list<LRUEntry>::iterator> label_to_lru_iter_;

  size_t current_count_ = 0;
  mutable std::mutex mutex_;
};

std::unique_ptr<VectorIndex> CreateHNSWIndex(
    size_t dimension,
    size_t max_elements,
    int m,
    int ef_construction,
    size_t max_entries) {
  return std::make_unique<HNSWIndex>(dimension, max_elements, m, ef_construction, max_entries);
}

#ifdef PRESTIGE_USE_FAISS
// FAISS implementation placeholder
std::unique_ptr<VectorIndex> CreateFAISSIndex(
    size_t dimension,
    int nlist,
    int pq_m,
    int pq_nbits) {
  // For now, fall back to HNSW
  return CreateHNSWIndex(dimension, 10000, 16, 200, 0);
}
#endif

}  // namespace prestige::internal

#endif  // PRESTIGE_ENABLE_SEMANTIC
