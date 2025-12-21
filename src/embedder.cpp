#include <prestige/embedder.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace prestige::internal {

class OnnxEmbedder : public Embedder {
 public:
  OnnxEmbedder(EmbedderModelType type, size_t dimension)
      : model_type_(type),
        dimension_(dimension),
        env_(ORT_LOGGING_LEVEL_WARNING, "prestige_embedder"),
        memory_info_(Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault)) {}

  bool Initialize(const std::string& model_path, std::string* error_out) {
    try {
      Ort::SessionOptions session_options;
      session_options.SetIntraOpNumThreads(1);
      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

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

  EmbeddingResult Embed(std::string_view text) const override {
    EmbeddingResult result;

    try {
      // For sentence-transformers models exported with optimum,
      // the model typically expects tokenized input (input_ids, attention_mask).
      // However, some exports include the tokenizer in the graph.
      //
      // This implementation assumes a model that accepts:
      // - input_ids: int64 tensor [batch_size, sequence_length]
      // - attention_mask: int64 tensor [batch_size, sequence_length]
      //
      // For simplicity, we use a basic tokenization approach here.
      // A production implementation would use a proper tokenizer.

      // Simple word-piece-like tokenization (placeholder)
      // In practice, you'd use the model's actual tokenizer
      std::vector<int64_t> input_ids;
      std::vector<int64_t> attention_mask;

      // CLS token
      input_ids.push_back(101);  // [CLS]
      attention_mask.push_back(1);

      // Simple character-based tokenization (placeholder)
      // Real implementation needs proper WordPiece/BPE tokenizer
      size_t max_tokens = 510;  // Reserve space for [CLS] and [SEP]
      std::string text_str(text);

      // Very basic word splitting
      size_t pos = 0;
      size_t token_count = 0;
      while (pos < text_str.size() && token_count < max_tokens) {
        // Skip whitespace
        while (pos < text_str.size() && std::isspace(text_str[pos])) {
          pos++;
        }
        if (pos >= text_str.size()) break;

        // Find end of word
        size_t end = pos;
        while (end < text_str.size() && !std::isspace(text_str[end])) {
          end++;
        }

        // Use a hash of the word as a pseudo-token ID
        // This is a placeholder - real tokenization needed
        std::string word = text_str.substr(pos, end - pos);
        int64_t token_id = 1000 + (std::hash<std::string>{}(word) % 28000);
        input_ids.push_back(token_id);
        attention_mask.push_back(1);
        token_count++;
        pos = end;
      }

      // SEP token
      input_ids.push_back(102);  // [SEP]
      attention_mask.push_back(1);

      // Create input tensors
      std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

      std::vector<Ort::Value> input_tensors;
      input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
          memory_info_, input_ids.data(), input_ids.size(),
          input_shape.data(), input_shape.size()));
      input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
          memory_info_, attention_mask.data(), attention_mask.size(),
          input_shape.data(), input_shape.size()));

      // Run inference
      auto output_tensors = session_->Run(
          Ort::RunOptions{nullptr},
          input_names_.data(), input_tensors.data(), input_tensors.size(),
          output_names_.data(), output_names_.size());

      if (output_tensors.empty()) {
        result.error_message = "No output tensors";
        return result;
      }

      // Get output tensor info
      auto& output_tensor = output_tensors[0];
      auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();

      // Output shape is typically [batch_size, sequence_length, hidden_size]
      // or [batch_size, hidden_size] for pooled output
      float* output_data = output_tensor.GetTensorMutableData<float>();
      size_t total_elements = tensor_info.GetElementCount();

      // Mean pooling over sequence dimension if needed
      if (shape.size() == 3) {
        // [batch, seq_len, hidden]
        int64_t seq_len = shape[1];
        int64_t hidden_size = shape[2];

        result.embedding.resize(hidden_size, 0.0f);

        // Mean pooling with attention mask
        for (int64_t i = 0; i < seq_len; ++i) {
          if (i < static_cast<int64_t>(attention_mask.size()) && attention_mask[i]) {
            for (int64_t j = 0; j < hidden_size; ++j) {
              result.embedding[j] += output_data[i * hidden_size + j];
            }
          }
        }

        // Divide by number of non-padding tokens
        float mask_sum = static_cast<float>(
            std::count(attention_mask.begin(), attention_mask.end(), 1));
        if (mask_sum > 0) {
          for (float& v : result.embedding) {
            v /= mask_sum;
          }
        }
      } else if (shape.size() == 2) {
        // [batch, hidden] - already pooled
        int64_t hidden_size = shape[1];
        result.embedding.assign(output_data, output_data + hidden_size);
      } else {
        result.error_message = "Unexpected output tensor shape";
        return result;
      }

      // L2 normalize
      float norm = 0.0f;
      for (float v : result.embedding) {
        norm += v * v;
      }
      norm = std::sqrt(norm);
      if (norm > 1e-12f) {
        for (float& v : result.embedding) {
          v /= norm;
        }
      }

      result.success = true;

    } catch (const Ort::Exception& e) {
      result.error_message = e.what();
    } catch (const std::exception& e) {
      result.error_message = e.what();
    }

    return result;
  }

  size_t Dimension() const override {
    return dimension_;
  }

  EmbedderModelType ModelType() const override {
    return model_type_;
  }

 private:
  EmbedderModelType model_type_;
  size_t dimension_;

  Ort::Env env_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;

  std::vector<std::string> input_names_str_;
  std::vector<std::string> output_names_str_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

std::unique_ptr<Embedder> Embedder::Create(
    const std::string& model_path,
    EmbedderModelType type,
    std::string* error_out) {

  size_t dimension = 384;  // Both MiniLM and BGE-small use 384 dimensions

  auto embedder = std::make_unique<OnnxEmbedder>(type, dimension);
  if (!embedder->Initialize(model_path, error_out)) {
    return nullptr;
  }

  return embedder;
}

}  // namespace prestige::internal

#else  // !PRESTIGE_ENABLE_SEMANTIC

namespace prestige::internal {

std::unique_ptr<Embedder> Embedder::Create(
    const std::string& /*model_path*/,
    EmbedderModelType /*type*/,
    std::string* error_out) {
  if (error_out) {
    *error_out = "Semantic dedup not enabled. Rebuild with PRESTIGE_ENABLE_SEMANTIC=ON";
  }
  return nullptr;
}

}  // namespace prestige::internal

#endif  // PRESTIGE_ENABLE_SEMANTIC
