#include <prestige/judge_llm.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC

// Check if llama.cpp is available
#ifdef PRESTIGE_ENABLE_LLAMA
#include <llama.h>
#define HAVE_LLAMA 1
#else
#define HAVE_LLAMA 0
#endif

#include <algorithm>
#include <cmath>
#include <regex>
#include <sstream>
#include <iomanip>
#include <thread>

namespace prestige::internal {

// Prometheus 2 evaluation prompt template
// Based on the model's training format for pairwise comparison tasks
constexpr const char* kPrometheusPromptTemplate = R"(###Task Description:
You are a semantic similarity judge. Evaluate whether two texts convey the same essential meaning and should be considered duplicates.

###Evaluation Criteria:
[Semantic Duplicates]
Score 1: The texts are clearly different in meaning, topic, or intent.
Score 2: The texts share some superficial similarities but have different core meanings.
Score 3: The texts are somewhat similar but have notable differences in scope or detail.
Score 4: The texts convey very similar meanings with only minor differences in wording.
Score 5: The texts are semantic duplicates - they express the same meaning despite surface differences.
Score 6: The texts are near character-level duplicates, as well as being semantic duplicates.
Score 7: The texts are character-level duplicates.

###Input:
Text A: {text1}

Text B: {text2}

Embedding Similarity: {similarity}

###Evaluation:
Based on the rubric, provide your assessment in the following format:
[REASONING]: Brief explanation of why the texts are or are not semantic duplicates.
[SCORE]: A single number from 1-7.
[DUPLICATE]: YES or NO

###Response:
)";

// Parse the judge's response to extract the duplicate decision
struct ParsedResponse {
  bool valid = false;
  bool is_duplicate = false;
  int score = 0;
  float confidence = 0.0f;
  std::string reasoning;
};

ParsedResponse ParseJudgeResponse(const std::string& response) {
  ParsedResponse result;

  // Try to extract DUPLICATE field
  std::regex duplicate_regex(R"(\[DUPLICATE\]:\s*(YES|NO))", std::regex::icase);
  std::smatch match;
  if (std::regex_search(response, match, duplicate_regex)) {
    result.valid = true;
    std::string answer = match[1].str();
    std::transform(answer.begin(), answer.end(), answer.begin(), ::toupper);
    result.is_duplicate = (answer == "YES");
  }

  // Try to extract SCORE field (now 1-7 scale)
  std::regex score_regex(R"(\[SCORE\]:\s*(\d))");
  if (std::regex_search(response, match, score_regex)) {
    result.score = std::stoi(match[1].str());
    // Convert score to confidence (1-7 scale, where 4+ is duplicate)
    // Scores 4-7 indicate duplicates with increasing confidence
    if (result.score >= 4) {
      result.confidence = (result.score - 3) / 4.0f;  // Maps 4->0.25, 5->0.5, 6->0.75, 7->1.0
    } else {
      result.confidence = (4 - result.score) / 4.0f;  // Maps 1->0.75, 2->0.5, 3->0.25 (confidence it's NOT a duplicate)
    }

    // Use score as fallback for duplicate decision if DUPLICATE not found
    if (!result.valid) {
      result.valid = true;
      result.is_duplicate = (result.score >= 5);  // Require strong semantic match for high precision
    }
  }

  // Extract reasoning
  std::regex reasoning_regex(R"(\[REASONING\]:\s*([^\[]+))");
  if (std::regex_search(response, match, reasoning_regex)) {
    result.reasoning = match[1].str();
    // Trim whitespace
    result.reasoning.erase(0, result.reasoning.find_first_not_of(" \t\n\r"));
    result.reasoning.erase(result.reasoning.find_last_not_of(" \t\n\r") + 1);
  }

  return result;
}

std::string FormatPrompt(std::string_view text1, std::string_view text2, float similarity) {
  std::string prompt = kPrometheusPromptTemplate;

  // Replace placeholders
  size_t pos = prompt.find("{text1}");
  if (pos != std::string::npos) {
    prompt.replace(pos, 7, text1);
  }

  pos = prompt.find("{text2}");
  if (pos != std::string::npos) {
    prompt.replace(pos, 7, text2);
  }

  pos = prompt.find("{similarity}");
  if (pos != std::string::npos) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << similarity;
    prompt.replace(pos, 12, ss.str());
  }

  return prompt;
}

#if HAVE_LLAMA

// Implementation class for Prometheus2Judge using llama.cpp
class Prometheus2Judge::Impl {
 public:
  Impl(int num_threads, int context_size, int gpu_layers, int max_tokens)
      : num_threads_(num_threads),
        context_size_(context_size),
        gpu_layers_(gpu_layers),
        max_tokens_(max_tokens) {}

  ~Impl() {
    if (ctx_) {
      llama_free(ctx_);
    }
    if (model_) {
      llama_model_free(model_);
    }
  }

  bool Initialize(const std::string& model_path, std::string* error_out) {
    // Initialize llama.cpp backend
    llama_backend_init();

    // Set up model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers_;

    // Load the model
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
      if (error_out) {
        *error_out = "Failed to load model from: " + model_path;
      }
      return false;
    }

    // Set up context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size_;
    ctx_params.n_threads = num_threads_ > 0 ? num_threads_ : std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;

    // Create context
    ctx_ = llama_new_context_with_model(model_, ctx_params);
    if (!ctx_) {
      if (error_out) {
        *error_out = "Failed to create llama context";
      }
      llama_model_free(model_);
      model_ = nullptr;
      return false;
    }

    return true;
  }

  JudgeResult Judge(std::string_view text1, std::string_view text2, float similarity_score) const {
    JudgeResult result;

    if (!model_ || !ctx_) {
      result.error_message = "Judge LLM not initialized";
      return result;
    }

    // Format the prompt
    std::string prompt = FormatPrompt(text1, text2, similarity_score);

    // Tokenize the prompt
    std::vector<llama_token> tokens(context_size_);
    int n_tokens = llama_tokenize(
        model_, prompt.c_str(), prompt.size(),
        tokens.data(), tokens.size(), true, false);

    if (n_tokens < 0) {
      result.error_message = "Tokenization failed";
      return result;
    }
    tokens.resize(n_tokens);

    // Check if prompt fits in context
    if (n_tokens + max_tokens_ > context_size_) {
      result.error_message = "Prompt too long for context window";
      return result;
    }

    // Clear KV cache
    llama_kv_cache_clear(ctx_);

    // Create batch for prompt processing
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = (i == n_tokens - 1);  // Enable logits for last token only
    }

    // Process prompt
    if (llama_decode(ctx_, batch) != 0) {
      llama_batch_free(batch);
      result.error_message = "Failed to process prompt";
      return result;
    }
    llama_batch_free(batch);

    // Generate response
    std::string response;
    int n_cur = n_tokens;

    for (int i = 0; i < max_tokens_; ++i) {
      // Sample next token
      float* logits = llama_get_logits_ith(ctx_, -1);
      int n_vocab = llama_n_vocab(model_);

      // Simple greedy sampling (could use more sophisticated sampling)
      llama_token new_token = 0;
      float max_logit = logits[0];
      for (int j = 1; j < n_vocab; ++j) {
        if (logits[j] > max_logit) {
          max_logit = logits[j];
          new_token = j;
        }
      }

      // Check for end of generation
      if (llama_token_is_eog(model_, new_token)) {
        break;
      }

      // Decode token to text
      char buf[256];
      int n = llama_token_to_piece(model_, new_token, buf, sizeof(buf), 0, false);
      if (n > 0) {
        response.append(buf, n);
      }

      // Process new token
      llama_batch single_batch = llama_batch_init(1, 0, 1);
      single_batch.n_tokens = 1;
      single_batch.token[0] = new_token;
      single_batch.pos[0] = n_cur;
      single_batch.n_seq_id[0] = 1;
      single_batch.seq_id[0][0] = 0;
      single_batch.logits[0] = true;

      if (llama_decode(ctx_, single_batch) != 0) {
        llama_batch_free(single_batch);
        break;
      }
      llama_batch_free(single_batch);

      ++n_cur;
    }

    // Parse the response
    ParsedResponse parsed = ParseJudgeResponse(response);

    if (!parsed.valid) {
      result.error_message = "Failed to parse judge response: " + response;
      return result;
    }

    result.success = true;
    result.is_duplicate = parsed.is_duplicate;
    result.confidence = parsed.confidence;
    result.reasoning = parsed.reasoning;

    return result;
  }

 private:
  int num_threads_;
  int context_size_;
  int gpu_layers_;
  int max_tokens_;

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
};

#else  // !HAVE_LLAMA

// Stub implementation when llama.cpp is not available
class Prometheus2Judge::Impl {
 public:
  Impl(int, int, int, int) {}
  ~Impl() = default;

  bool Initialize(const std::string&, std::string* error_out) {
    if (error_out) {
      *error_out = "llama.cpp not available. Build with PRESTIGE_ENABLE_LLAMA=ON";
    }
    return false;
  }

  JudgeResult Judge(std::string_view, std::string_view, float) const {
    return JudgeResult{false, false, 0.0f, "", "llama.cpp not available"};
  }
};

#endif  // HAVE_LLAMA

// Prometheus2Judge implementation (shared between llama and stub)
Prometheus2Judge::Prometheus2Judge(int num_threads, int context_size, int gpu_layers, int max_tokens)
    : impl_(std::make_unique<Impl>(num_threads, context_size, gpu_layers, max_tokens)),
      num_threads_(num_threads),
      context_size_(context_size),
      gpu_layers_(gpu_layers),
      max_tokens_(max_tokens) {}

Prometheus2Judge::~Prometheus2Judge() = default;

bool Prometheus2Judge::Initialize(const std::string& model_path, std::string* error_out) {
  return impl_->Initialize(model_path, error_out);
}

JudgeResult Prometheus2Judge::Judge(std::string_view text1,
                                     std::string_view text2,
                                     float similarity_score) const {
  return impl_->Judge(text1, text2, similarity_score);
}

// Factory function
std::unique_ptr<JudgeLLM> CreateJudgeLLM(
    const std::string& model_path,
    int num_threads,
    int context_size,
    int gpu_layers,
    int max_tokens,
    std::string* error_out) {
  auto judge = std::make_unique<Prometheus2Judge>(
      num_threads, context_size, gpu_layers, max_tokens);

  if (!judge->Initialize(model_path, error_out)) {
    return nullptr;
  }

  return judge;
}

}  // namespace prestige::internal

#else  // !PRESTIGE_ENABLE_SEMANTIC

// Stub implementation when semantic features are disabled
namespace prestige::internal {

Prometheus2Judge::Prometheus2Judge(int, int, int, int) {}
Prometheus2Judge::~Prometheus2Judge() = default;

bool Prometheus2Judge::Initialize(const std::string&, std::string* error_out) {
  if (error_out) {
    *error_out = "Judge LLM requires PRESTIGE_ENABLE_SEMANTIC=ON";
  }
  return false;
}

JudgeResult Prometheus2Judge::Judge(std::string_view, std::string_view, float) const {
  return JudgeResult{false, false, 0.0f, "", "Semantic features disabled"};
}

std::unique_ptr<JudgeLLM> CreateJudgeLLM(
    const std::string&, int, int, int, int, std::string* error_out) {
  if (error_out) {
    *error_out = "Judge LLM requires PRESTIGE_ENABLE_SEMANTIC=ON";
  }
  return nullptr;
}

}  // namespace prestige::internal

#endif  // PRESTIGE_ENABLE_SEMANTIC
