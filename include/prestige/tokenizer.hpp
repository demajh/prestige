#pragma once

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace prestige::internal {

/**
 * Result of tokenization: token IDs and attention mask.
 */
struct TokenizerResult {
  std::vector<int64_t> input_ids;
  std::vector<int64_t> attention_mask;
  std::vector<int64_t> token_type_ids;
  bool success = false;
  std::string error_message;
};

/**
 * WordPiece tokenizer for BERT-based models.
 *
 * Implements the WordPiece algorithm used by BERT, BGE, MiniLM, etc.
 * Loads vocabulary from a vocab.txt file (one token per line).
 */
class WordPieceTokenizer {
 public:
  ~WordPieceTokenizer();

  /**
   * Create a tokenizer from a vocabulary file.
   *
   * @param vocab_path Path to vocab.txt file
   * @param error_out Optional error message on failure
   * @return Tokenizer instance, or nullptr on failure
   */
  static std::unique_ptr<WordPieceTokenizer> Create(
      const std::string& vocab_path,
      std::string* error_out = nullptr);

  /**
   * Tokenize text into token IDs.
   *
   * @param text Input text
   * @param max_length Maximum sequence length (default 512)
   * @return TokenizerResult with input_ids, attention_mask, token_type_ids
   */
  TokenizerResult Tokenize(std::string_view text, size_t max_length = 512) const;

  /**
   * Get vocabulary size.
   */
  size_t VocabSize() const { return vocab_.size(); }

  /**
   * Get token ID for a token string.
   */
  int64_t TokenToId(const std::string& token) const;

  /**
   * Special token IDs.
   */
  int64_t PadTokenId() const { return pad_token_id_; }
  int64_t UnkTokenId() const { return unk_token_id_; }
  int64_t ClsTokenId() const { return cls_token_id_; }
  int64_t SepTokenId() const { return sep_token_id_; }

 private:
  WordPieceTokenizer() = default;

  bool LoadVocab(const std::string& vocab_path, std::string* error_out);

  // Normalize and split text into words
  std::vector<std::string> BasicTokenize(std::string_view text) const;

  // Apply WordPiece algorithm to a single word
  std::vector<std::string> WordPieceTokenize(const std::string& word) const;

  // Check if character is whitespace
  static bool IsWhitespace(char c);

  // Check if character is punctuation
  static bool IsPunctuation(char c);

  // Check if character is Chinese character (for CJK handling)
  static bool IsChineseChar(uint32_t cp);

  // Convert UTF-8 string to lowercase
  static std::string ToLowercase(std::string_view text);

  // Vocabulary: token -> id
  std::unordered_map<std::string, int64_t> vocab_;

  // Special token IDs
  int64_t pad_token_id_ = 0;
  int64_t unk_token_id_ = 100;
  int64_t cls_token_id_ = 101;
  int64_t sep_token_id_ = 102;

  // Maximum word length for WordPiece
  static constexpr size_t kMaxWordLength = 200;
};

}  // namespace prestige::internal

#endif  // PRESTIGE_ENABLE_SEMANTIC
