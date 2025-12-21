// Semantic deduplication example for Prestige
//
// This example demonstrates semantic dedup using embeddings.
// Build with: cmake -DPRESTIGE_ENABLE_SEMANTIC=ON ..
//
// Before running, download an ONNX model:
//   pip install optimum[onnxruntime]
//   optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 \
//       --task feature-extraction ./minilm_onnx/

#include <prestige/store.hpp>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path-to-onnx-model>\n";
    std::cerr << "\nExample:\n";
    std::cerr << "  " << argv[0] << " ./minilm_onnx/model.onnx\n";
    return 1;
  }

  std::string model_path = argv[1];

  prestige::Options opt;

  // Configure semantic deduplication
  opt.dedup_mode = prestige::DedupMode::kSemantic;
  opt.semantic_model_path = model_path;
  opt.semantic_model_type = prestige::SemanticModel::kMiniLM;
  opt.semantic_threshold = 0.85f;  // Cosine similarity threshold for dedup
  opt.semantic_search_k = 10;      // Number of candidates to check

  // HNSW index parameters
  opt.hnsw_m = 16;
  opt.hnsw_ef_construction = 200;
  opt.hnsw_ef_search = 50;

  std::unique_ptr<prestige::Store> db;
  auto s = prestige::Store::Open("./prestige_semantic_db", &db, opt);
  if (!s.ok()) {
    std::cerr << "Open failed: " << s.ToString() << "\n";
    return 1;
  }

  std::cout << "Semantic dedup store opened successfully.\n\n";

  // These texts are semantically similar and should deduplicate
  const char* text1 = "The quick brown fox jumps over the lazy dog.";
  const char* text2 = "A fast brown fox leaps above a sleepy dog.";
  const char* text3 = "The swift brown fox hops over the tired hound.";

  // This text is semantically different
  const char* text4 = "Machine learning is a subset of artificial intelligence.";

  std::cout << "Inserting semantically similar texts:\n";
  std::cout << "  key1: \"" << text1 << "\"\n";
  s = db->Put("key1", text1);
  if (!s.ok()) {
    std::cerr << "Put key1 failed: " << s.ToString() << "\n";
    return 1;
  }

  std::cout << "  key2: \"" << text2 << "\"\n";
  s = db->Put("key2", text2);
  if (!s.ok()) {
    std::cerr << "Put key2 failed: " << s.ToString() << "\n";
    return 1;
  }

  std::cout << "  key3: \"" << text3 << "\"\n";
  s = db->Put("key3", text3);
  if (!s.ok()) {
    std::cerr << "Put key3 failed: " << s.ToString() << "\n";
    return 1;
  }

  std::cout << "\nInserting semantically different text:\n";
  std::cout << "  key4: \"" << text4 << "\"\n";
  s = db->Put("key4", text4);
  if (!s.ok()) {
    std::cerr << "Put key4 failed: " << s.ToString() << "\n";
    return 1;
  }

  // Count keys and unique values
  uint64_t key_count = 0;
  uint64_t unique_count = 0;

  s = db->CountKeys(&key_count);
  if (!s.ok()) {
    std::cerr << "CountKeys failed: " << s.ToString() << "\n";
    return 1;
  }

  s = db->CountUniqueValues(&unique_count);
  if (!s.ok()) {
    std::cerr << "CountUniqueValues failed: " << s.ToString() << "\n";
    return 1;
  }

  std::cout << "\nResults:\n";
  std::cout << "  Total keys: " << key_count << "\n";
  std::cout << "  Unique values: " << unique_count << "\n";

  if (unique_count < key_count) {
    std::cout << "\nSemantic dedup is working! Similar texts share storage.\n";
  } else {
    std::cout << "\nNo semantic dedup occurred. Try adjusting the threshold.\n";
  }

  // Retrieve and display values
  std::cout << "\nRetrieved values:\n";
  for (const char* key : {"key1", "key2", "key3", "key4"}) {
    std::string value;
    s = db->Get(key, &value);
    if (s.ok()) {
      std::cout << "  " << key << ": \"" << value << "\"\n";
    } else {
      std::cerr << "  " << key << ": " << s.ToString() << "\n";
    }
  }

  // Clean up
  for (const char* key : {"key1", "key2", "key3", "key4"}) {
    db->Delete(key);
  }

  std::cout << "\nDone.\n";
  return 0;
}
