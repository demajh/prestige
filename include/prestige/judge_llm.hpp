#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace prestige::internal {

/**
 * Result of a judge LLM evaluation.
 */
struct JudgeResult {
  bool success = false;           // Whether the evaluation completed successfully
  bool is_duplicate = false;      // Whether the judge determined texts are duplicates
  float confidence = 0.0f;        // Confidence score [0.0, 1.0]
  std::string reasoning;          // Optional reasoning from the judge
  std::string error_message;      // Error message if success is false
};

/**
 * Abstract base class for judge LLMs.
 *
 * Judge LLMs evaluate whether two texts are semantic duplicates by using
 * structured rubric-based evaluation. This is intended for "gray zone"
 * cases where embedding similarity is borderline.
 *
 * The primary implementation uses Prometheus 2 (prometheus-7b-v2.0),
 * an Apache-2.0 licensed evaluator model designed for rubric-based grading.
 */
class JudgeLLM {
 public:
  virtual ~JudgeLLM() = default;

  /**
   * Judge whether two texts are semantic duplicates.
   *
   * @param text1 First text to compare
   * @param text2 Second text to compare
   * @param similarity_score Precomputed embedding similarity (for context)
   * @return JudgeResult with duplicate determination and confidence
   */
  virtual JudgeResult Judge(std::string_view text1,
                            std::string_view text2,
                            float similarity_score) const = 0;

  /**
   * Get the model name/type for debugging.
   */
  virtual std::string ModelType() const = 0;

  /**
   * Get the maximum input text length supported.
   */
  virtual size_t MaxTextLength() const = 0;
};

/**
 * Prometheus 2 judge LLM implementation.
 *
 * Uses prometheus-7b-v2.0, a 7B parameter model trained for evaluation tasks.
 * The model uses rubric-based structured grading to determine semantic similarity.
 *
 * Model: prometheus-eval/prometheus-7b-v2.0
 * License: Apache-2.0
 * Source: https://huggingface.co/prometheus-eval/prometheus-7b-v2.0
 *
 * The model is loaded via llama.cpp for efficient local inference,
 * supporting both GGUF quantized models and GPU acceleration.
 */
class Prometheus2Judge : public JudgeLLM {
 public:
  /**
   * Create a Prometheus 2 judge.
   *
   * @param num_threads Number of threads for inference (0 = all cores)
   * @param context_size Context window size (default: 4096)
   * @param gpu_layers Number of layers to offload to GPU (0 = CPU only, -1 = all)
   * @param max_tokens Maximum tokens for response generation
   */
  explicit Prometheus2Judge(int num_threads = 0,
                            int context_size = 4096,
                            int gpu_layers = 0,
                            int max_tokens = 256);

  ~Prometheus2Judge();

  /**
   * Initialize the judge with a model file.
   *
   * @param model_path Path to GGUF model file
   * @param error_out Error message output
   * @return true if initialization successful
   */
  bool Initialize(const std::string& model_path, std::string* error_out);

  JudgeResult Judge(std::string_view text1,
                    std::string_view text2,
                    float similarity_score) const override;

  std::string ModelType() const override { return "prometheus-7b-v2.0"; }

  size_t MaxTextLength() const override { return max_text_length_; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  size_t max_text_length_ = 2048;  // Conservative limit for context window
  int num_threads_ = 0;
  int context_size_ = 4096;
  int gpu_layers_ = 0;
  int max_tokens_ = 256;
};

/**
 * Factory function to create a judge LLM.
 *
 * @param model_path Path to model file (GGUF format)
 * @param num_threads Number of threads for inference
 * @param context_size Context window size
 * @param gpu_layers GPU layers to offload
 * @param max_tokens Maximum response tokens
 * @param error_out Error message output
 * @return JudgeLLM instance or nullptr on failure
 */
std::unique_ptr<JudgeLLM> CreateJudgeLLM(
    const std::string& model_path,
    int num_threads = 0,
    int context_size = 4096,
    int gpu_layers = 0,
    int max_tokens = 256,
    std::string* error_out = nullptr);

}  // namespace prestige::internal
