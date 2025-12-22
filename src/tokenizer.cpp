#include <prestige/tokenizer.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace prestige::internal {

WordPieceTokenizer::~WordPieceTokenizer() = default;

std::unique_ptr<WordPieceTokenizer> WordPieceTokenizer::Create(
    const std::string& vocab_path,
    std::string* error_out) {
  auto tokenizer = std::unique_ptr<WordPieceTokenizer>(new WordPieceTokenizer());
  if (!tokenizer->LoadVocab(vocab_path, error_out)) {
    return nullptr;
  }
  return tokenizer;
}

bool WordPieceTokenizer::LoadVocab(const std::string& vocab_path,
                                    std::string* error_out) {
  std::ifstream file(vocab_path);
  if (!file.is_open()) {
    if (error_out) {
      *error_out = "Failed to open vocabulary file: " + vocab_path;
    }
    return false;
  }

  std::string line;
  int64_t id = 0;
  while (std::getline(file, line)) {
    // Remove trailing whitespace/newline
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                              line.back() == ' ' || line.back() == '\t')) {
      line.pop_back();
    }
    vocab_[line] = id;
    id++;
  }

  if (vocab_.empty()) {
    if (error_out) {
      *error_out = "Vocabulary file is empty";
    }
    return false;
  }

  // Find special token IDs
  auto find_token = [this](const std::string& token, int64_t default_id) -> int64_t {
    auto it = vocab_.find(token);
    return (it != vocab_.end()) ? it->second : default_id;
  };

  pad_token_id_ = find_token("[PAD]", 0);
  unk_token_id_ = find_token("[UNK]", 100);
  cls_token_id_ = find_token("[CLS]", 101);
  sep_token_id_ = find_token("[SEP]", 102);

  return true;
}

int64_t WordPieceTokenizer::TokenToId(const std::string& token) const {
  auto it = vocab_.find(token);
  return (it != vocab_.end()) ? it->second : unk_token_id_;
}

bool WordPieceTokenizer::IsWhitespace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool WordPieceTokenizer::IsPunctuation(char c) {
  // ASCII punctuation
  if ((c >= 33 && c <= 47) ||   // !"#$%&'()*+,-./
      (c >= 58 && c <= 64) ||   // :;<=>?@
      (c >= 91 && c <= 96) ||   // [\]^_`
      (c >= 123 && c <= 126)) { // {|}~
    return true;
  }
  return false;
}

bool WordPieceTokenizer::IsChineseChar(uint32_t cp) {
  // CJK Unified Ideographs and related blocks
  if ((cp >= 0x4E00 && cp <= 0x9FFF) ||
      (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) ||
      (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) ||
      (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) ||
      (cp >= 0x2F800 && cp <= 0x2FA1F)) {
    return true;
  }
  return false;
}

std::string WordPieceTokenizer::ToLowercase(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  for (char c : text) {
    result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return result;
}

std::vector<std::string> WordPieceTokenizer::BasicTokenize(std::string_view text) const {
  std::vector<std::string> tokens;

  // Convert to lowercase
  std::string normalized = ToLowercase(text);

  // Tokenize on whitespace and punctuation
  std::string current_token;

  for (size_t i = 0; i < normalized.size(); ++i) {
    char c = normalized[i];

    if (IsWhitespace(c)) {
      // End current token
      if (!current_token.empty()) {
        tokens.push_back(std::move(current_token));
        current_token.clear();
      }
    } else if (IsPunctuation(c)) {
      // End current token
      if (!current_token.empty()) {
        tokens.push_back(std::move(current_token));
        current_token.clear();
      }
      // Add punctuation as its own token
      tokens.push_back(std::string(1, c));
    } else {
      // Check for Chinese characters (simplified - assumes UTF-8)
      unsigned char uc = static_cast<unsigned char>(c);
      if (uc >= 0xE0 && i + 2 < normalized.size()) {
        // Potential 3-byte UTF-8 character
        uint32_t cp = ((uc & 0x0F) << 12) |
                      ((static_cast<unsigned char>(normalized[i + 1]) & 0x3F) << 6) |
                      (static_cast<unsigned char>(normalized[i + 2]) & 0x3F);
        if (IsChineseChar(cp)) {
          // End current token
          if (!current_token.empty()) {
            tokens.push_back(std::move(current_token));
            current_token.clear();
          }
          // Add Chinese character as its own token
          tokens.push_back(normalized.substr(i, 3));
          i += 2;
          continue;
        }
      }
      current_token += c;
    }
  }

  // Don't forget the last token
  if (!current_token.empty()) {
    tokens.push_back(std::move(current_token));
  }

  return tokens;
}

std::vector<std::string> WordPieceTokenizer::WordPieceTokenize(
    const std::string& word) const {
  std::vector<std::string> output_tokens;

  if (word.size() > kMaxWordLength) {
    output_tokens.push_back("[UNK]");
    return output_tokens;
  }

  bool is_bad = false;
  size_t start = 0;

  while (start < word.size()) {
    size_t end = word.size();
    std::string cur_substr;
    bool found = false;

    while (start < end) {
      std::string substr = word.substr(start, end - start);
      if (start > 0) {
        substr = "##" + substr;
      }

      auto it = vocab_.find(substr);
      if (it != vocab_.end()) {
        cur_substr = substr;
        found = true;
        break;
      }
      --end;
    }

    if (!found) {
      is_bad = true;
      break;
    }

    output_tokens.push_back(cur_substr);
    start = end;
  }

  if (is_bad) {
    output_tokens.clear();
    output_tokens.push_back("[UNK]");
  }

  return output_tokens;
}

TokenizerResult WordPieceTokenizer::Tokenize(std::string_view text,
                                              size_t max_length) const {
  TokenizerResult result;

  if (text.empty()) {
    // Return just [CLS] [SEP] for empty input
    result.input_ids = {cls_token_id_, sep_token_id_};
    result.attention_mask = {1, 1};
    result.token_type_ids = {0, 0};
    result.success = true;
    return result;
  }

  // Reserve space (estimate ~1.5 tokens per word)
  result.input_ids.reserve(max_length);
  result.attention_mask.reserve(max_length);
  result.token_type_ids.reserve(max_length);

  // Add [CLS] token
  result.input_ids.push_back(cls_token_id_);
  result.attention_mask.push_back(1);
  result.token_type_ids.push_back(0);

  // Basic tokenization (split on whitespace/punctuation)
  std::vector<std::string> words = BasicTokenize(text);

  // Apply WordPiece to each word
  for (const auto& word : words) {
    std::vector<std::string> sub_tokens = WordPieceTokenize(word);

    for (const auto& sub_token : sub_tokens) {
      // Check if we have room (reserve 1 for [SEP])
      if (result.input_ids.size() >= max_length - 1) {
        break;
      }

      int64_t token_id = TokenToId(sub_token);
      result.input_ids.push_back(token_id);
      result.attention_mask.push_back(1);
      result.token_type_ids.push_back(0);
    }

    if (result.input_ids.size() >= max_length - 1) {
      break;
    }
  }

  // Add [SEP] token
  result.input_ids.push_back(sep_token_id_);
  result.attention_mask.push_back(1);
  result.token_type_ids.push_back(0);

  result.success = true;
  return result;
}

}  // namespace prestige::internal

#endif  // PRESTIGE_ENABLE_SEMANTIC
