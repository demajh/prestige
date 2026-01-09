#include <prestige/reranker.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <prestige/tokenizer.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace prestige::internal {

// Implementation class for BGERerankerV2
class BGERerankerV2::Impl {
public:
  Impl(int num_threads, InferenceDevice device)
      : num_threads_(num_threads),
        device_(device),
        env_(ORT_LOGGING_LEVEL_WARNING, "prestige_reranker"),
        memory_info_(Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault)) {}

  bool Initialize(const std::string& model_path,
                  const std::string& vocab_path_in,
                  std::string* error_out) {
    // Auto-detect tokenizer files if not provided
    std::string vocab_path = vocab_path_in;
    if (vocab_path.empty()) {
      // Try to find tokenizer files (prefer tokenizer.json over vocab.txt)
      std::string tokenizer_json_path = FindTokenizerPath(model_path, "tokenizer.json");
      std::string vocab_txt_path = FindTokenizerPath(model_path, "vocab.txt");
      
      if (!tokenizer_json_path.empty()) {
        // Use SentencePiece/HF tokenizer (BGE-reranker-v2-m3)
        if (error_out) {
          *error_out = "SentencePiece tokenization not yet implemented. Use BGE-reranker-base instead.";
        }
        return false;
      } else if (!vocab_txt_path.empty()) {
        vocab_path = vocab_txt_path;
      } else {
        if (error_out) {
          *error_out = "Could not find vocab.txt or tokenizer.json in model directory";
        }
        return false;
      }
    }

    // Load WordPiece tokenizer (for BGE-reranker-base)
    tokenizer_ = WordPieceTokenizer::Create(vocab_path, error_out);
    if (!tokenizer_) {
      if (error_out && error_out->empty()) {
        *error_out = "Failed to load tokenizer from: " + vocab_path;
      }
      return false;
    }

    try {
      Ort::SessionOptions session_options;
      session_options.SetIntraOpNumThreads(num_threads_);
      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

      // Configure execution provider based on device setting
      bool use_gpu = false;
      if (device_ == InferenceDevice::kGPU) {
        use_gpu = true;
      } else if (device_ == InferenceDevice::kAuto) {
        // Auto-detect: try GPU first
        use_gpu = true;
      }

      if (use_gpu) {
        try {
          OrtCUDAProviderOptions cuda_options;
          cuda_options.device_id = 0;
          session_options.AppendExecutionProvider_CUDA(cuda_options);
          using_gpu_ = true;
        } catch (const Ort::Exception& e) {
          // GPU not available
          if (device_ == InferenceDevice::kGPU) {
            // User explicitly requested GPU, fail
            if (error_out) *error_out = std::string("CUDA not available: ") + e.what();
            return false;
          }
          // Auto mode: fall back to CPU silently
          using_gpu_ = false;
        }
      }

      session_ = std::make_unique<Ort::Session>(
          env_, model_path.c_str(), session_options);

      // Cache input/output names
      Ort::AllocatorWithDefaultOptions allocator;

      size_t num_inputs = session_->GetInputCount();
      for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_str_.push_back(name.get());
      }

      size_t num_outputs = session_->GetOutputCount();
      for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_str_.push_back(name.get());
      }

      // Convert to const char* for ONNX Runtime
      for (const auto& s : input_names_str_) {
        input_names_.push_back(s.c_str());
      }
      for (const auto& s : output_names_str_) {
        output_names_.push_back(s.c_str());
      }

      return true;
    } catch (const Ort::Exception& e) {
      if (error_out) *error_out = e.what();
      return false;
    } catch (const std::exception& e) {
      if (error_out) *error_out = e.what();
      return false;
    }
  }

  ScoringResult Score(std::string_view text1, std::string_view text2) const {
    ScoringResult result;

    try {
      // Tokenize the text pair for cross-encoder
      auto tokens = TokenizePair(text1, text2);
      if (!tokens.success) {
        result.error_message = "Tokenization failed: " + tokens.error_message;
        return result;
      }

      // Create input tensors
      std::vector<int64_t> input_shape = {
          1, static_cast<int64_t>(tokens.input_ids.size())};

      std::vector<Ort::Value> input_tensors;
      for (const auto& name : input_names_str_) {
        if (name == "input_ids") {
          input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
              memory_info_, tokens.input_ids.data(), tokens.input_ids.size(),
              input_shape.data(), input_shape.size()));
        } else if (name == "attention_mask") {
          input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
              memory_info_, tokens.attention_mask.data(),
              tokens.attention_mask.size(), input_shape.data(),
              input_shape.size()));
        } else if (name == "token_type_ids") {
          input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
              memory_info_, tokens.token_type_ids.data(),
              tokens.token_type_ids.size(), input_shape.data(),
              input_shape.size()));
        } else {
          result.error_message = "Unknown input name: " + name;
          return result;
        }
      }

      // Run inference
      auto output_tensors = session_->Run(
          Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
          input_tensors.size(), output_names_.data(), output_names_.size());

      if (output_tensors.empty()) {
        result.error_message = "No output tensors";
        return result;
      }

      // Extract score from output
      auto& output_tensor = output_tensors[0];
      float* output_data = output_tensor.GetTensorMutableData<float>();
      
      // BGE reranker outputs a single score
      // Apply sigmoid to convert logit to probability [0, 1]
      float logit = output_data[0];
      result.score = 1.0f / (1.0f + std::exp(-logit));
      result.success = true;

    } catch (const Ort::Exception& e) {
      result.error_message = e.what();
    } catch (const std::exception& e) {
      result.error_message = e.what();
    }

    return result;
  }

  std::vector<ScoringResult> ScoreBatch(
      std::string_view query,
      const std::vector<std::string_view>& candidates) const {
    std::vector<ScoringResult> results;
    results.reserve(candidates.size());

    // For now, process sequentially
    // TODO: Implement true batch processing for efficiency
    for (const auto& candidate : candidates) {
      results.push_back(Score(query, candidate));
    }

    return results;
  }

private:
  TokenizerResult TokenizePair(std::string_view text1,
                                std::string_view text2) const {
    // For cross-encoder, we need to tokenize both texts together
    // Format: [CLS] text1 [SEP] text2 [SEP]
    
    // Combine texts with separator
    std::string combined;
    combined.reserve(text1.size() + text2.size() + 10);
    combined.append(text1);
    combined.append(" [SEP] ");
    combined.append(text2);
    
    // Use regular tokenizer with the combined text
    return tokenizer_->Tokenize(combined, 512);
  }

  std::string FindTokenizerPath(const std::string& model_path, const std::string& filename) const {
    // Get directory of model file
    size_t last_slash = model_path.find_last_of("/\\");
    std::string dir = (last_slash != std::string::npos)
                          ? model_path.substr(0, last_slash + 1)
                          : "./";

    // Look for the specified tokenizer file in the same directory
    std::string tokenizer_path = dir + filename;
    std::ifstream check(tokenizer_path);
    if (check.good()) {
      return tokenizer_path;
    }

    return "";
  }

  int num_threads_;
  InferenceDevice device_;
  bool using_gpu_ = false;
  std::unique_ptr<WordPieceTokenizer> tokenizer_;

  Ort::Env env_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;
  
  std::vector<std::string> input_names_str_;
  std::vector<std::string> output_names_str_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

// BGERerankerV2 implementation
BGERerankerV2::BGERerankerV2(int num_threads, InferenceDevice device)
    : impl_(std::make_unique<Impl>(num_threads, device)) {}

BGERerankerV2::~BGERerankerV2() = default;

bool BGERerankerV2::Initialize(const std::string& model_path,
                                const std::string& vocab_path,
                                std::string* error_out) {
  return impl_->Initialize(model_path, vocab_path, error_out);
}

ScoringResult BGERerankerV2::Score(std::string_view text1,
                                    std::string_view text2) const {
  return impl_->Score(text1, text2);
}

std::vector<ScoringResult> BGERerankerV2::ScoreBatch(
    std::string_view query,
    const std::vector<std::string_view>& candidates) const {
  return impl_->ScoreBatch(query, candidates);
}

// Factory function
std::unique_ptr<Reranker> CreateReranker(
    const std::string& model_path,
    int num_threads,
    InferenceDevice device,
    std::string* error_out) {
  // For now, only support BGE reranker
  // Could extend to support other models based on model_path detection
  auto reranker = std::make_unique<BGERerankerV2>(num_threads, device);

  if (!reranker->Initialize(model_path, "", error_out)) {
    return nullptr;
  }

  return reranker;
}

}  // namespace prestige::internal

#else  // !PRESTIGE_ENABLE_SEMANTIC

// Stub implementation when semantic features are disabled
namespace prestige::internal {

BGERerankerV2::BGERerankerV2(int, InferenceDevice) {}
BGERerankerV2::~BGERerankerV2() = default;

bool BGERerankerV2::Initialize(const std::string&, const std::string&, std::string* error_out) {
  if (error_out) {
    *error_out = "Reranker requires PRESTIGE_ENABLE_SEMANTIC=ON";
  }
  return false;
}

ScoringResult BGERerankerV2::Score(std::string_view, std::string_view) const {
  return ScoringResult{false, 0.0f, "Semantic features disabled"};
}

std::vector<ScoringResult> BGERerankerV2::ScoreBatch(
    std::string_view, const std::vector<std::string_view>&) const {
  return {};
}

std::unique_ptr<Reranker> CreateReranker(const std::string&, int, InferenceDevice, std::string* error_out) {
  if (error_out) {
    *error_out = "Reranker requires PRESTIGE_ENABLE_SEMANTIC=ON";
  }
  return nullptr;
}

}  // namespace prestige::internal

#endif  // PRESTIGE_ENABLE_SEMANTIC