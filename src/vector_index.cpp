#include <prestige/vector_index.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace prestige::internal {

// HNSW implementation using hnswlib
class HNSWIndex : public VectorIndex {
 public:
  HNSWIndex(size_t dimension, size_t max_elements, int m, int ef_construction)
      : dimension_(dimension),
        max_elements_(max_elements),
        m_(m),
        ef_construction_(ef_construction),
        ef_search_(50) {
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

    deleted_labels_.insert(it->second);
    return true;
  }

  bool Save(const std::string& path) override {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
      // Save HNSW index
      index_->saveIndex(path);

      // Save metadata (object_id mappings and deleted set)
      std::string meta_path = path + ".meta";
      std::ofstream ofs(meta_path, std::ios::binary);
      if (!ofs) return false;

      // Write dimension and counts
      ofs.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
      ofs.write(reinterpret_cast<const char*>(&current_count_), sizeof(current_count_));

      size_t deleted_count = deleted_labels_.size();
      ofs.write(reinterpret_cast<const char*>(&deleted_count), sizeof(deleted_count));

      // Write label -> object_id mappings
      for (const auto& [label, obj_id] : label_to_object_id_) {
        ofs.write(reinterpret_cast<const char*>(&label), sizeof(label));
        size_t len = obj_id.size();
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        ofs.write(obj_id.data(), len);
      }

      // Write deleted labels
      for (hnswlib::labeltype label : deleted_labels_) {
        ofs.write(reinterpret_cast<const char*>(&label), sizeof(label));
      }

      return ofs.good();
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

      // Read dimension and counts
      size_t saved_dimension;
      ifs.read(reinterpret_cast<char*>(&saved_dimension), sizeof(saved_dimension));
      if (saved_dimension != dimension_) {
        return false;  // Dimension mismatch
      }

      ifs.read(reinterpret_cast<char*>(&current_count_), sizeof(current_count_));

      size_t deleted_count;
      ifs.read(reinterpret_cast<char*>(&deleted_count), sizeof(deleted_count));

      // Read label -> object_id mappings
      label_to_object_id_.clear();
      object_id_to_label_.clear();
      for (size_t i = 0; i < current_count_; ++i) {
        hnswlib::labeltype label;
        ifs.read(reinterpret_cast<char*>(&label), sizeof(label));
        size_t len;
        ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string obj_id(len, '\0');
        ifs.read(obj_id.data(), len);

        label_to_object_id_[label] = obj_id;
        object_id_to_label_[obj_id] = label;
      }

      // Read deleted labels
      deleted_labels_.clear();
      for (size_t i = 0; i < deleted_count; ++i) {
        hnswlib::labeltype label;
        ifs.read(reinterpret_cast<char*>(&label), sizeof(label));
        deleted_labels_.insert(label);
      }

      max_elements_ = index_->max_elements_;
      return ifs.good();
    } catch (const std::exception&) {
      return false;
    }
  }

  size_t Size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_count_ - deleted_labels_.size();
  }

  size_t Dimension() const override {
    return dimension_;
  }

  void SetSearchParam(const std::string& key, int value) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (key == "ef_search" || key == "ef") {
      ef_search_ = value;
      index_->setEf(value);
    }
  }

 private:
  size_t dimension_;
  size_t max_elements_;
  int m_;
  int ef_construction_;
  int ef_search_;

  std::unique_ptr<hnswlib::L2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

  // Mapping between internal labels and object_ids
  std::unordered_map<hnswlib::labeltype, std::string> label_to_object_id_;
  std::unordered_map<std::string, hnswlib::labeltype> object_id_to_label_;
  std::unordered_set<hnswlib::labeltype> deleted_labels_;

  size_t current_count_ = 0;
  mutable std::mutex mutex_;
};

std::unique_ptr<VectorIndex> CreateHNSWIndex(
    size_t dimension,
    size_t max_elements,
    int m,
    int ef_construction) {
  return std::make_unique<HNSWIndex>(dimension, max_elements, m, ef_construction);
}

#ifdef PRESTIGE_USE_FAISS
// FAISS implementation placeholder
// TODO: Implement when FAISS support is needed
std::unique_ptr<VectorIndex> CreateFAISSIndex(
    size_t dimension,
    int nlist,
    int pq_m,
    int pq_nbits) {
  // For now, fall back to HNSW
  return CreateHNSWIndex(dimension, 10000, 16, 200);
}
#endif

}  // namespace prestige::internal

#endif  // PRESTIGE_ENABLE_SEMANTIC
