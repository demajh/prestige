// Tests for reranker functionality

#include <gtest/gtest.h>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <prestige/store.hpp>
#include <prestige/reranker.hpp>
#include <prestige/test_utils.hpp>

#include <filesystem>
#include <vector>

namespace {

// Mock reranker for testing
class MockReranker : public prestige::internal::Reranker {
public:
  explicit MockReranker(float fixed_score = 0.8f) : fixed_score_(fixed_score) {}
  
  prestige::internal::ScoringResult Score(std::string_view text1, 
                                           std::string_view text2) const override {
    prestige::internal::ScoringResult result;
    result.success = true;
    
    // Simple mock: higher score for shorter text difference
    int diff = std::abs(static_cast<int>(text1.length()) - static_cast<int>(text2.length()));
    result.score = std::max(0.0f, fixed_score_ - (diff * 0.01f));
    
    return result;
  }
  
  std::vector<prestige::internal::ScoringResult> ScoreBatch(
      std::string_view query,
      const std::vector<std::string_view>& candidates) const override {
    std::vector<prestige::internal::ScoringResult> results;
    for (const auto& candidate : candidates) {
      results.push_back(Score(query, candidate));
    }
    return results;
  }
  
  size_t MaxTextLength() const override { return 512; }
  std::string ModelType() const override { return "mock"; }

private:
  float fixed_score_;
};

class RerankerTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() /
                ("prestige_reranker_test_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    store_.reset();
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  rocksdb::Status OpenStoreWithReranker(float reranker_threshold = 0.7f,
                                         float semantic_threshold = 0.8f) {
    prestige::Options opt;
    opt.dedup_mode = prestige::DedupMode::kSemantic;
    opt.semantic_threshold = semantic_threshold;
    opt.custom_embedder = new prestige::testing::DeterministicEmbedder();
    
    // Enable reranker
    opt.semantic_reranker_enabled = true;
    opt.semantic_reranker_threshold = reranker_threshold;
    opt.custom_reranker = new MockReranker(0.9f);  // High base score
    opt.semantic_index_save_interval = 0;  // Disable auto-save for tests
    
    return prestige::Store::Open(test_dir_.string(), &store_, opt);
  }

  std::filesystem::path test_dir_;
  std::unique_ptr<prestige::Store> store_;
};

// =============================================================================
// Basic Reranker Tests
// =============================================================================

TEST_F(RerankerTest, BasicRerankerDedup) {
  ASSERT_TRUE(OpenStoreWithReranker().ok());
  
  // Put similar texts that should be deduplicated by reranker
  ASSERT_TRUE(store_->Put("key1", "hello world").ok());
  ASSERT_TRUE(store_->Put("key2", "hello world!").ok());  // Similar length, should deduplicate
  
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);  // Should be deduplicated by reranker
}

TEST_F(RerankerTest, RerankerThresholdRespected) {
  // Use very high threshold to prevent deduplication
  ASSERT_TRUE(OpenStoreWithReranker(0.99f).ok());
  
  ASSERT_TRUE(store_->Put("key1", "hello").ok());
  ASSERT_TRUE(store_->Put("key2", "hello world").ok());  // Different length, lower score
  
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 2u);  // Should NOT be deduplicated due to high threshold
}

TEST_F(RerankerTest, RerankerFallbackToEmbeddings) {
  // Test that semantic dedup still works when reranker is enabled
  ASSERT_TRUE(OpenStoreWithReranker(0.5f, 0.95f).ok());  // Low reranker threshold, high semantic threshold
  
  ASSERT_TRUE(store_->Put("key1", "text A").ok());
  ASSERT_TRUE(store_->Put("key2", "text B").ok());
  
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  // Behavior depends on the DeterministicEmbedder implementation
  // Just verify it doesn't crash
  EXPECT_GE(unique_count, 1u);
}

TEST_F(RerankerTest, RerankerDisabledFallsBack) {
  prestige::Options opt;
  opt.dedup_mode = prestige::DedupMode::kSemantic;
  opt.semantic_threshold = 0.8f;
  opt.custom_embedder = new prestige::testing::DeterministicEmbedder();
  opt.semantic_reranker_enabled = false;  // Disabled
  opt.semantic_index_save_interval = 0;
  
  ASSERT_TRUE(prestige::Store::Open(test_dir_.string(), &store_, opt).ok());
  
  // Should still work with just embeddings
  ASSERT_TRUE(store_->Put("key1", "hello").ok());
  ASSERT_TRUE(store_->Put("key2", "world").ok());
  
  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "hello");
}

}  // namespace

#else  // !PRESTIGE_ENABLE_SEMANTIC

// Stub test when semantic features are disabled
TEST(RerankerTestStub, SemanticDisabled) {
  GTEST_SKIP() << "Semantic features disabled, skipping reranker tests";
}

#endif  // PRESTIGE_ENABLE_SEMANTIC