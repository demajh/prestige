#include <prestige/embedder.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <prestige/tokenizer.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
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

  bool Initialize(const std::string& model_path,
                  const std::string& vocab_path,
                  std::string* error_out) {
    // Load tokenizer
    tokenizer_ = WordPieceTokenizer::Create(vocab_path, error_out);
    if (!tokenizer_) {
      if (error_out && error_out->empty()) {
        *error_out = "Failed to load tokenizer from: " + vocab_path;
      }
      return false;
    }

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
      // For BGE models, prepend instruction prefix for better accuracy
      // See: https://huggingface.co/BAAI/bge-small-en-v1.5
      std::string prefixed_text;
      std::string_view text_to_embed = text;
      if (model_type_ == EmbedderModelType::kBGESmall ||
          model_type_ == EmbedderModelType::kBGELarge) {
        static constexpr std::string_view kBGEPrefix =
            "Represent this sentence for searching relevant passages: ";
        prefixed_text.reserve(kBGEPrefix.size() + text.size());
        prefixed_text.append(kBGEPrefix);
        prefixed_text.append(text);
        text_to_embed = prefixed_text;
      }

      // Tokenize the input text using WordPiece tokenizer
      TokenizerResult tokens = tokenizer_->Tokenize(text_to_embed, 512);
      if (!tokens.success) {
        result.error_message = "Tokenization failed: " + tokens.error_message;
        return result;
      }

      // Create input tensors
      std::vector<int64_t> input_shape = {
          1, static_cast<int64_t>(tokens.input_ids.size())};

      // Build input tensors in the order expected by the model
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

      // Get output tensor info
      auto& output_tensor = output_tensors[0];
      auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();

      // Output shape is typically [batch_size, sequence_length, hidden_size]
      // or [batch_size, hidden_size] for pooled output
      float* output_data = output_tensor.GetTensorMutableData<float>();

      // Mean pooling over sequence dimension if needed
      if (shape.size() == 3) {
        // [batch, seq_len, hidden]
        int64_t seq_len = shape[1];
        int64_t hidden_size = shape[2];

        result.embedding.resize(hidden_size, 0.0f);

        // Mean pooling with attention mask
        for (int64_t i = 0; i < seq_len; ++i) {
          if (i < static_cast<int64_t>(tokens.attention_mask.size()) &&
              tokens.attention_mask[i]) {
            for (int64_t j = 0; j < hidden_size; ++j) {
              result.embedding[j] += output_data[i * hidden_size + j];
            }
          }
        }

        // Divide by number of non-padding tokens
        float mask_sum = static_cast<float>(std::count(
            tokens.attention_mask.begin(), tokens.attention_mask.end(), 1));
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

  size_t Dimension() const override { return dimension_; }

  EmbedderModelType ModelType() const override { return model_type_; }

 private:
  EmbedderModelType model_type_;
  size_t dimension_;

  std::unique_ptr<WordPieceTokenizer> tokenizer_;

  Ort::Env env_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;

  std::vector<std::string> input_names_str_;
  std::vector<std::string> output_names_str_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

namespace {

// Find vocab.txt in the same directory as the model
std::string FindVocabPath(const std::string& model_path) {
  // Get directory of model file
  size_t last_slash = model_path.find_last_of("/\\");
  std::string dir = (last_slash != std::string::npos)
                        ? model_path.substr(0, last_slash + 1)
                        : "./";

  // Look for vocab.txt in the same directory
  std::string vocab_path = dir + "vocab.txt";
  std::ifstream check(vocab_path);
  if (check.good()) {
    return vocab_path;
  }

  // Try without path (current directory)
  check.open("vocab.txt");
  if (check.good()) {
    return "vocab.txt";
  }

  return "";
}

}  // namespace

std::unique_ptr<Embedder> Embedder::Create(const std::string& model_path,
                                           EmbedderModelType type,
                                           std::string* error_out) {
  // Auto-detect vocab path
  std::string vocab_path = FindVocabPath(model_path);
  if (vocab_path.empty()) {
    if (error_out) {
      *error_out = "Could not find vocab.txt in the same directory as " +
                   model_path + ". Please place vocab.txt alongside the model.";
    }
    return nullptr;
  }

  // Determine embedding dimension based on model type
  size_t dimension;
  switch (type) {
    case EmbedderModelType::kMiniLM:
    case EmbedderModelType::kBGESmall:
      dimension = 384;
      break;
    case EmbedderModelType::kBGELarge:
      dimension = 1024;
      break;
  }

  auto embedder = std::make_unique<OnnxEmbedder>(type, dimension);
  if (!embedder->Initialize(model_path, vocab_path, error_out)) {
    return nullptr;
  }

  return embedder;
}

}  // namespace prestige::internal

#else  // !PRESTIGE_ENABLE_SEMANTIC

namespace prestige::internal {

std::unique_ptr<Embedder> Embedder::Create(const std::string& /*model_path*/,
                                           EmbedderModelType /*type*/,
                                           std::string* error_out) {
  if (error_out) {
    *error_out =
        "Semantic dedup not enabled. Rebuild with PRESTIGE_ENABLE_SEMANTIC=ON";
  }
  return nullptr;
}

}  // namespace prestige::internal

#endif  // PRESTIGE_ENABLE_SEMANTIC
