#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace prestige::internal {

/**
 * Result of a text pair scoring operation.
 */
struct ScoringResult {
  bool success = false;
  float score = 0.0f;  // Range typically [0, 1], higher = more similar
  std::string error_message;
};

/**
 * Abstract base class for text pair rerankers.
 * 
 * Rerankers are cross-encoder models that score the similarity/relevance
 * between pairs of texts directly, providing more accurate scoring than
 * bi-encoder embeddings at the cost of higher computational overhead.
 */
class Reranker {
public:
  virtual ~Reranker() = default;
  
  /**
   * Score the similarity between two texts.
   * 
   * @param text1 First text
   * @param text2 Second text
   * @return Scoring result with success flag and score
   */
  virtual ScoringResult Score(std::string_view text1, 
                              std::string_view text2) const = 0;
  
  /**
   * Batch score multiple candidates against a query.
   * 
   * @param query Query text
   * @param candidates Vector of candidate texts to score
   * @return Vector of scoring results, one per candidate
   */
  virtual std::vector<ScoringResult> ScoreBatch(
      std::string_view query,
      const std::vector<std::string_view>& candidates) const = 0;
  
  /**
   * Get the maximum text length supported by this reranker.
   */
  virtual size_t MaxTextLength() const = 0;
  
  /**
   * Get the model type/name for debugging.
   */
  virtual std::string ModelType() const = 0;
};

/**
 * BGE-reranker-v2-m3 implementation.
 * 
 * This is a cross-encoder model from BAAI that provides high-quality
 * text pair scoring for semantic similarity and relevance ranking.
 */
class BGERerankerV2 : public Reranker {
public:
  /**
   * Create a BGE reranker.
   * 
   * @param num_threads Number of threads for ONNX inference (0 = all cores)
   */
  explicit BGERerankerV2(int num_threads = 0);
  
  ~BGERerankerV2();
  
  /**
   * Initialize the reranker with model files.
   * 
   * @param model_path Path to ONNX model file
   * @param vocab_path Path to vocabulary file (optional, auto-detected if empty)
   * @param error_out Error message output
   * @return true if initialization successful
   */
  bool Initialize(const std::string& model_path,
                  const std::string& vocab_path,
                  std::string* error_out);
  
  ScoringResult Score(std::string_view text1,
                     std::string_view text2) const override;
  
  std::vector<ScoringResult> ScoreBatch(
      std::string_view query,
      const std::vector<std::string_view>& candidates) const override;
  
  size_t MaxTextLength() const override { return max_length_; }
  
  std::string ModelType() const override { return "bge-reranker-v2-m3"; }

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  size_t max_length_ = 512;
};

/**
 * Factory function to create appropriate reranker based on model type.
 * 
 * @param model_path Path to model file
 * @param num_threads Number of threads for inference
 * @param error_out Error message output
 * @return Reranker instance or nullptr on failure
 */
std::unique_ptr<Reranker> CreateReranker(
    const std::string& model_path,
    int num_threads,
    std::string* error_out);

}  // namespace prestige::internal