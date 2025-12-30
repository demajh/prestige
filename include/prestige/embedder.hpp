#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace prestige::internal {

// Result of embedding computation
struct EmbeddingResult {
  std::vector<float> embedding;  // Normalized embedding vector
  bool success = false;
  std::string error_message;
};

// Supported model types
enum class EmbedderModelType {
  kMiniLM,    // all-MiniLM-L6-v2 (384 dimensions)
  kBGESmall,  // BGE-small-en-v1.5 (384 dimensions)
  kBGELarge   // BGE-large-en-v1.5 (1024 dimensions)
};

// ONNX-based text embedder
class Embedder {
 public:
  virtual ~Embedder() = default;

  // Factory: load model from ONNX file
  // Vocabulary file (vocab.txt) is auto-detected in the same directory as the model.
  // num_threads: 0 = use all available cores, >0 = use specified number of threads
  // Returns nullptr on failure, sets error_out if provided
  static std::unique_ptr<Embedder> Create(
      const std::string& model_path,
      EmbedderModelType type,
      int num_threads = 0,
      std::string* error_out = nullptr);

  // Compute embedding for text input
  // Input is assumed to be UTF-8 text
  // Returns normalized embedding vector (L2 norm = 1)
  virtual EmbeddingResult Embed(std::string_view text) const = 0;

  // Get embedding dimension (384 for MiniLM/BGE-small, 1024 for BGE-large)
  virtual size_t Dimension() const = 0;

  // Get model type
  virtual EmbedderModelType ModelType() const = 0;
};

}  // namespace prestige::internal
