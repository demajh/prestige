// Tests for judge LLM functionality (Prometheus 2 integration)

#include <gtest/gtest.h>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <prestige/store.hpp>
#include <prestige/judge_llm.hpp>
#include <prestige/embedder.hpp>
#include <prestige/test_utils.hpp>

#include <filesystem>
#include <vector>
#include <cmath>

namespace {

// Mock judge LLM for testing
class MockJudgeLLM : public prestige::internal::JudgeLLM {
 public:
  // Mode for controlling judge behavior in tests
  enum class Mode {
    kAlwaysAccept,      // Always return is_duplicate=true
    kAlwaysReject,      // Always return is_duplicate=false
    kThresholdBased,    // Accept if similarity >= internal threshold
    kLengthBased,       // Accept if text lengths are similar
  };

  explicit MockJudgeLLM(Mode mode = Mode::kThresholdBased,
                        float threshold = 0.8f)
      : mode_(mode), threshold_(threshold) {}

  prestige::internal::JudgeResult Judge(
      std::string_view text1,
      std::string_view text2,
      float similarity_score) const override {
    prestige::internal::JudgeResult result;
    result.success = true;
    call_count_++;

    switch (mode_) {
      case Mode::kAlwaysAccept:
        result.is_duplicate = true;
        result.confidence = 0.95f;
        result.reasoning = "Mock: Always accepting as duplicate";
        break;

      case Mode::kAlwaysReject:
        result.is_duplicate = false;
        result.confidence = 0.95f;
        result.reasoning = "Mock: Always rejecting as non-duplicate";
        break;

      case Mode::kThresholdBased:
        result.is_duplicate = (similarity_score >= threshold_);
        result.confidence = similarity_score;
        result.reasoning = "Mock: Threshold-based decision";
        break;

      case Mode::kLengthBased: {
        // Accept if text lengths are within 20% of each other
        float len1 = static_cast<float>(text1.length());
        float len2 = static_cast<float>(text2.length());
        float ratio = (len1 < len2) ? (len1 / len2) : (len2 / len1);
        result.is_duplicate = (ratio >= 0.8f);
        result.confidence = ratio;
        result.reasoning = "Mock: Length-based decision";
        break;
      }
    }

    return result;
  }

  std::string ModelType() const override { return "mock-judge"; }
  size_t MaxTextLength() const override { return 2048; }

  // Test helpers
  int GetCallCount() const { return call_count_; }
  void ResetCallCount() { call_count_ = 0; }

 private:
  Mode mode_;
  float threshold_;
  mutable int call_count_ = 0;
};

// Custom embedder that allows registering specific embeddings for controlled similarity
class ControlledEmbedder : public prestige::internal::Embedder {
 public:
  explicit ControlledEmbedder(size_t dim = 384) : dim_(dim) {}

  prestige::internal::EmbeddingResult Embed(std::string_view text) const override {
    prestige::internal::EmbeddingResult result;
    result.success = true;
    result.embedding.resize(dim_);

    // Check for registered embedding
    std::string text_str(text);
    auto it = registered_.find(text_str);
    if (it != registered_.end()) {
      result.embedding = it->second;
      return result;
    }

    // Generate deterministic embedding based on hash
    std::hash<std::string> hasher;
    size_t hash = hasher(text_str);

    for (size_t i = 0; i < dim_; ++i) {
      result.embedding[i] = static_cast<float>((hash >> (i % 8)) & 0xFF) / 255.0f - 0.5f;
    }

    // Normalize
    float norm = 0.0f;
    for (float v : result.embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0) {
      for (float& v : result.embedding) v /= norm;
    }

    return result;
  }

  size_t Dimension() const override { return dim_; }

  prestige::internal::EmbedderModelType ModelType() const override {
    return prestige::internal::EmbedderModelType::kMiniLM;
  }

  void RegisterEmbedding(const std::string& text, const std::vector<float>& embedding) {
    registered_[text] = embedding;
  }

  // Helper to create embeddings with controlled similarity
  static std::vector<float> CreateUnitVector(size_t dim, float angle) {
    std::vector<float> v(dim, 0.0f);
    v[0] = std::cos(angle);
    v[1] = std::sin(angle);
    // Normalize (already unit in 2D subspace)
    float norm = std::sqrt(v[0] * v[0] + v[1] * v[1]);
    v[0] /= norm;
    v[1] /= norm;
    return v;
  }

 private:
  size_t dim_;
  std::map<std::string, std::vector<float>> registered_;
};

class JudgeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() /
                ("prestige_judge_test_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    store_.reset();
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  // Open store with judge enabled
  rocksdb::Status OpenStoreWithJudge(
      float semantic_threshold = 0.90f,
      float judge_threshold = 0.75f,
      MockJudgeLLM::Mode judge_mode = MockJudgeLLM::Mode::kThresholdBased,
      float judge_internal_threshold = 0.8f) {
    prestige::Options opt;
    opt.dedup_mode = prestige::DedupMode::kSemantic;
    opt.semantic_threshold = semantic_threshold;
    opt.semantic_verify_exact = true;

    // Create controlled embedder
    embedder_ = new ControlledEmbedder(384);
    opt.custom_embedder = embedder_;

    // Enable judge
    opt.semantic_judge_enabled = true;
    opt.semantic_judge_threshold = judge_threshold;
    mock_judge_ = new MockJudgeLLM(judge_mode, judge_internal_threshold);
    opt.custom_judge = mock_judge_;

    opt.semantic_index_save_interval = 0;  // Disable auto-save for tests

    return prestige::Store::Open(test_dir_.string(), &store_, opt);
  }

  std::filesystem::path test_dir_;
  std::unique_ptr<prestige::Store> store_;
  ControlledEmbedder* embedder_ = nullptr;  // Owned by store
  MockJudgeLLM* mock_judge_ = nullptr;      // Owned by store
};

// =============================================================================
// Basic Judge Tests
// =============================================================================

TEST_F(JudgeTest, JudgeNotCalledAboveThreshold) {
  // When similarity >= semantic_threshold, judge should NOT be called
  ASSERT_TRUE(OpenStoreWithJudge(0.90f, 0.75f).ok());

  // Register embeddings with very high similarity (0.99)
  // cos(0.1) ~= 0.995
  auto emb1 = ControlledEmbedder::CreateUnitVector(384, 0.0f);
  auto emb2 = ControlledEmbedder::CreateUnitVector(384, 0.1f);
  embedder_->RegisterEmbedding("text A", emb1);
  embedder_->RegisterEmbedding("text A similar", emb2);

  ASSERT_TRUE(store_->Put("key1", "text A").ok());
  ASSERT_TRUE(store_->Put("key2", "text A similar").ok());

  // Should be deduplicated without calling judge (similarity ~0.995 > 0.90)
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);

  // Judge should not have been called
  EXPECT_EQ(mock_judge_->GetCallCount(), 0);
}

TEST_F(JudgeTest, JudgeCalledInGrayZone) {
  // When judge_threshold <= similarity < semantic_threshold, judge should be called
  ASSERT_TRUE(OpenStoreWithJudge(
      0.90f,  // semantic_threshold
      0.75f,  // judge_threshold
      MockJudgeLLM::Mode::kAlwaysAccept).ok());

  // Register embeddings with similarity in gray zone (0.80, between 0.75 and 0.90)
  // cos(0.6435) ~= 0.80
  auto emb1 = ControlledEmbedder::CreateUnitVector(384, 0.0f);
  auto emb2 = ControlledEmbedder::CreateUnitVector(384, 0.6435f);
  embedder_->RegisterEmbedding("text X", emb1);
  embedder_->RegisterEmbedding("text Y gray zone", emb2);

  ASSERT_TRUE(store_->Put("key1", "text X").ok());
  ASSERT_TRUE(store_->Put("key2", "text Y gray zone").ok());

  // Judge should have been called (similarity ~0.80 is in gray zone)
  EXPECT_GE(mock_judge_->GetCallCount(), 1);

  // Since judge always accepts, should be deduplicated
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);
}

TEST_F(JudgeTest, JudgeRejectsInGrayZone) {
  // Judge can reject candidates in the gray zone
  ASSERT_TRUE(OpenStoreWithJudge(
      0.90f,  // semantic_threshold
      0.75f,  // judge_threshold
      MockJudgeLLM::Mode::kAlwaysReject).ok());

  // Register embeddings with similarity in gray zone
  auto emb1 = ControlledEmbedder::CreateUnitVector(384, 0.0f);
  auto emb2 = ControlledEmbedder::CreateUnitVector(384, 0.6435f);  // cos ~0.80
  embedder_->RegisterEmbedding("text P", emb1);
  embedder_->RegisterEmbedding("text Q different", emb2);

  ASSERT_TRUE(store_->Put("key1", "text P").ok());
  ASSERT_TRUE(store_->Put("key2", "text Q different").ok());

  // Judge should have been called
  EXPECT_GE(mock_judge_->GetCallCount(), 1);

  // Since judge always rejects, should NOT be deduplicated
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 2u);
}

TEST_F(JudgeTest, JudgeNotCalledBelowThreshold) {
  // When similarity < judge_threshold, judge should NOT be called
  ASSERT_TRUE(OpenStoreWithJudge(
      0.90f,  // semantic_threshold
      0.75f,  // judge_threshold
      MockJudgeLLM::Mode::kAlwaysAccept).ok());

  // Register embeddings with low similarity (0.50, below judge threshold)
  // cos(1.047) ~= 0.50
  auto emb1 = ControlledEmbedder::CreateUnitVector(384, 0.0f);
  auto emb2 = ControlledEmbedder::CreateUnitVector(384, 1.047f);
  embedder_->RegisterEmbedding("text M", emb1);
  embedder_->RegisterEmbedding("text N very different", emb2);

  ASSERT_TRUE(store_->Put("key1", "text M").ok());
  ASSERT_TRUE(store_->Put("key2", "text N very different").ok());

  // Judge should NOT have been called (similarity too low)
  EXPECT_EQ(mock_judge_->GetCallCount(), 0);

  // Should NOT be deduplicated
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 2u);
}

TEST_F(JudgeTest, JudgeDisabledWorks) {
  // When judge is disabled, gray zone candidates should be rejected
  prestige::Options opt;
  opt.dedup_mode = prestige::DedupMode::kSemantic;
  opt.semantic_threshold = 0.90f;
  opt.semantic_verify_exact = true;
  opt.semantic_judge_enabled = false;  // Disabled
  opt.custom_embedder = new ControlledEmbedder(384);
  opt.semantic_index_save_interval = 0;

  ASSERT_TRUE(prestige::Store::Open(test_dir_.string(), &store_, opt).ok());

  // Put some data
  ASSERT_TRUE(store_->Put("key1", "hello").ok());
  ASSERT_TRUE(store_->Put("key2", "world").ok());

  // Should work fine without judge
  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "hello");
}

TEST_F(JudgeTest, ThresholdBasedJudge) {
  // Judge uses its own internal threshold
  ASSERT_TRUE(OpenStoreWithJudge(
      0.90f,   // semantic_threshold
      0.70f,   // judge_threshold
      MockJudgeLLM::Mode::kThresholdBased,
      0.78f).ok());  // judge's internal threshold

  // Create two pairs: one at 0.80 similarity (above judge's 0.78), one at 0.75 (below)
  auto emb_base = ControlledEmbedder::CreateUnitVector(384, 0.0f);
  auto emb_high = ControlledEmbedder::CreateUnitVector(384, 0.6435f);  // cos ~0.80
  auto emb_low = ControlledEmbedder::CreateUnitVector(384, 0.7227f);   // cos ~0.75

  embedder_->RegisterEmbedding("base text", emb_base);
  embedder_->RegisterEmbedding("high sim text", emb_high);
  embedder_->RegisterEmbedding("low sim text", emb_low);

  // First pair: base + high sim (should be deduplicated, 0.80 >= 0.78)
  ASSERT_TRUE(store_->Put("key1", "base text").ok());
  ASSERT_TRUE(store_->Put("key2", "high sim text").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);  // Deduplicated

  // Second pair: base + low sim (should NOT be deduplicated, 0.75 < 0.78)
  ASSERT_TRUE(store_->Put("key3", "low sim text").ok());

  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 2u);  // Not deduplicated
}

TEST_F(JudgeTest, JudgeWithExactDuplicates) {
  // Exact duplicates should be handled by threshold, not judge
  ASSERT_TRUE(OpenStoreWithJudge(
      0.90f,
      0.75f,
      MockJudgeLLM::Mode::kAlwaysReject).ok());

  // Register identical embeddings (similarity = 1.0)
  auto emb = ControlledEmbedder::CreateUnitVector(384, 0.0f);
  embedder_->RegisterEmbedding("exact same text", emb);

  ASSERT_TRUE(store_->Put("key1", "exact same text").ok());
  ASSERT_TRUE(store_->Put("key2", "exact same text").ok());

  // Should be deduplicated by threshold (sim=1.0 > 0.90), judge not called
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);
  EXPECT_EQ(mock_judge_->GetCallCount(), 0);
}

TEST_F(JudgeTest, MultipleGrayZoneCandidates) {
  // Test that judge is called for multiple gray zone candidates
  // Use AlwaysReject to ensure judge rejects gray zone candidates
  ASSERT_TRUE(OpenStoreWithJudge(
      0.90f,
      0.70f,
      MockJudgeLLM::Mode::kAlwaysReject).ok());

  // Create embeddings that are each in gray zone with emb0, but NOT with each other
  // Use orthogonal directions to ensure they don't match each other
  auto emb0 = ControlledEmbedder::CreateUnitVector(384, 0.0f);       // angle 0
  auto emb1 = ControlledEmbedder::CreateUnitVector(384, 0.55f);      // cos(0.55) ~0.85 with emb0

  // For emb2, use a completely different direction in the embedding space
  // to avoid matching with emb1
  std::vector<float> emb2(384, 0.0f);
  emb2[2] = 0.85f;  // Put in a different dimension
  emb2[3] = 0.527f; // To make cos with emb0's [1,0,...] ~= 0 (orthogonal)

  embedder_->RegisterEmbedding("text alpha", emb0);
  embedder_->RegisterEmbedding("text beta", emb1);
  embedder_->RegisterEmbedding("text gamma", emb2);

  ASSERT_TRUE(store_->Put("key1", "text alpha").ok());
  mock_judge_->ResetCallCount();  // Reset to count calls for subsequent puts

  ASSERT_TRUE(store_->Put("key2", "text beta").ok());   // Gray zone with emb0, judge rejects
  int calls_after_key2 = mock_judge_->GetCallCount();
  EXPECT_GE(calls_after_key2, 1);  // Judge was called at least once

  ASSERT_TRUE(store_->Put("key3", "text gamma").ok());  // Not in gray zone (orthogonal), no judge

  // All should be unique because judge rejected and emb2 is orthogonal
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 3u);
}

}  // namespace

#else  // !PRESTIGE_ENABLE_SEMANTIC

// Stub test when semantic features are disabled
TEST(JudgeTestStub, SemanticDisabled) {
  GTEST_SKIP() << "Semantic features disabled, skipping judge tests";
}

#endif  // PRESTIGE_ENABLE_SEMANTIC
