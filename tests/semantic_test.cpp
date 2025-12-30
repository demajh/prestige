// Semantic deduplication tests using deterministic embedder (no ONNX required)

#include <gtest/gtest.h>

#ifdef PRESTIGE_ENABLE_SEMANTIC

#include <prestige/store.hpp>
#include <prestige/test_utils.hpp>

#include <filesystem>
#include <random>
#include <thread>
#include <vector>

namespace {

class SemanticTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() /
                ("prestige_semantic_test_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    store_.reset();
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  rocksdb::Status OpenStore(float threshold = 0.95f,
                            prestige::internal::Embedder* embedder = nullptr) {
    prestige::Options opt;
    opt.dedup_mode = prestige::DedupMode::kSemantic;
    opt.semantic_threshold = threshold;
    opt.custom_embedder = embedder ? embedder : new prestige::testing::DeterministicEmbedder();
    opt.semantic_index_save_interval = 0;  // Disable auto-save for tests
    return prestige::Store::Open(test_dir_.string(), &store_, opt);
  }

  std::filesystem::path test_dir_;
  std::unique_ptr<prestige::Store> store_;
};

// =============================================================================
// Basic Semantic Operations
// =============================================================================

TEST_F(SemanticTest, BasicPutGet) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "hello world").ok());

  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "hello world");
}

TEST_F(SemanticTest, GetNonExistent) {
  ASSERT_TRUE(OpenStore().ok());

  std::string value;
  auto status = store_->Get("missing", &value);
  EXPECT_TRUE(status.IsNotFound());
}

TEST_F(SemanticTest, Delete) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "value1").ok());
  ASSERT_TRUE(store_->Delete("key1").ok());

  std::string value;
  EXPECT_TRUE(store_->Get("key1", &value).IsNotFound());
}

// =============================================================================
// Semantic Deduplication Hit/Miss
// =============================================================================

TEST_F(SemanticTest, ExactDuplicateDedup) {
  ASSERT_TRUE(OpenStore(0.95f).ok());

  // Exact same content should always deduplicate
  ASSERT_TRUE(store_->Put("key1", "identical content").ok());
  ASSERT_TRUE(store_->Put("key2", "identical content").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);  // Should be deduplicated

  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_EQ(key_count, 2u);  // Two keys point to same object
}

TEST_F(SemanticTest, DifferentContentNoDedup) {
  ASSERT_TRUE(OpenStore(0.95f).ok());

  // Completely different content should not deduplicate
  ASSERT_TRUE(store_->Put("key1", "the quick brown fox").ok());
  ASSERT_TRUE(store_->Put("key2", "completely unrelated text").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 2u);  // Should not be deduplicated
}

TEST_F(SemanticTest, CustomEmbedderControlledDedup) {
  // Create embedder with custom embeddings for precise control
  auto* embedder = new prestige::testing::DeterministicEmbedder(384);

  // Register embeddings that are very similar (high cosine similarity)
  std::vector<float> emb1(384, 0.0f);
  std::vector<float> emb2(384, 0.0f);

  // Set embeddings that differ only slightly (cosine sim ~0.99)
  for (size_t i = 0; i < 384; ++i) {
    emb1[i] = static_cast<float>(i) / 384.0f;
    emb2[i] = emb1[i] + 0.01f;
  }
  // Normalize
  float norm1 = 0.0f, norm2 = 0.0f;
  for (size_t i = 0; i < 384; ++i) {
    norm1 += emb1[i] * emb1[i];
    norm2 += emb2[i] * emb2[i];
  }
  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);
  for (size_t i = 0; i < 384; ++i) {
    emb1[i] /= norm1;
    emb2[i] /= norm2;
  }

  embedder->RegisterEmbedding("text A", emb1);
  embedder->RegisterEmbedding("text B", emb2);

  // Create embeddings that are very different
  std::vector<float> emb3(384);
  for (size_t i = 0; i < 384; ++i) {
    emb3[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  float norm3 = std::sqrt(384.0f);
  for (float& v : emb3) v /= norm3;
  embedder->RegisterEmbedding("text C", emb3);

  ASSERT_TRUE(OpenStore(0.95f, embedder).ok());

  // Similar texts should deduplicate
  ASSERT_TRUE(store_->Put("key1", "text A").ok());
  ASSERT_TRUE(store_->Put("key2", "text B").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);  // Should deduplicate

  // Very different text should not deduplicate
  ASSERT_TRUE(store_->Put("key3", "text C").ok());

  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 2u);  // Now 2 unique values
}

TEST_F(SemanticTest, ThresholdAffectsDedup) {
  // Test with very low threshold (almost nothing deduplicates)
  {
    auto* embedder = new prestige::testing::DeterministicEmbedder(384);
    ASSERT_TRUE(OpenStore(0.9999f, embedder).ok());

    ASSERT_TRUE(store_->Put("key1", "first text").ok());
    ASSERT_TRUE(store_->Put("key2", "second text").ok());

    uint64_t unique_count = 0;
    ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
    EXPECT_EQ(unique_count, 2u);  // High threshold = less dedup

    store_.reset();
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
    std::filesystem::create_directories(test_dir_);
  }

  // Test with low threshold and known-similar embeddings (should deduplicate)
  {
    auto* embedder = new prestige::testing::DeterministicEmbedder(384);

    // Register embeddings with known high cosine similarity (~0.99)
    // For threshold=0.5 (cos_sim >= 0.5), these should match
    std::vector<float> emb1(384, 0.0f);
    std::vector<float> emb2(384, 0.0f);
    for (size_t i = 0; i < 384; ++i) {
      emb1[i] = static_cast<float>(i) / 384.0f;
      emb2[i] = emb1[i] + 0.01f;  // Very similar
    }
    // Normalize
    float norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < 384; ++i) {
      norm1 += emb1[i] * emb1[i];
      norm2 += emb2[i] * emb2[i];
    }
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    for (size_t i = 0; i < 384; ++i) {
      emb1[i] /= norm1;
      emb2[i] /= norm2;
    }
    embedder->RegisterEmbedding("first text", emb1);
    embedder->RegisterEmbedding("second text", emb2);

    ASSERT_TRUE(OpenStore(0.5f, embedder).ok());

    ASSERT_TRUE(store_->Put("key1", "first text").ok());
    ASSERT_TRUE(store_->Put("key2", "second text").ok());

    uint64_t unique_count = 0;
    ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
    EXPECT_EQ(unique_count, 1u);  // Low threshold + similar embeddings = dedup
  }
}

// =============================================================================
// Overwrite Semantics
// =============================================================================

TEST_F(SemanticTest, OverwriteWithSameValue) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "original").ok());
  ASSERT_TRUE(store_->Put("key1", "original").ok());  // Same value

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);

  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "original");
}

TEST_F(SemanticTest, OverwriteWithDifferentValue) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "original value").ok());
  ASSERT_TRUE(store_->Put("key1", "new value").ok());  // Different value

  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "new value");
}

// =============================================================================
// Crash Recovery (Pending Vector Ops Replay)
// =============================================================================

TEST_F(SemanticTest, RecoveryPreservesData) {
  std::string key1_value, key2_value;

  // First session: write data
  {
    auto* embedder = new prestige::testing::DeterministicEmbedder(384);
    ASSERT_TRUE(OpenStore(0.95f, embedder).ok());

    ASSERT_TRUE(store_->Put("key1", "value one").ok());
    ASSERT_TRUE(store_->Put("key2", "value two").ok());

    store_->Close();
    store_.reset();
  }

  // Second session: verify data persisted
  {
    auto* embedder = new prestige::testing::DeterministicEmbedder(384);
    ASSERT_TRUE(OpenStore(0.95f, embedder).ok());

    ASSERT_TRUE(store_->Get("key1", &key1_value).ok());
    EXPECT_EQ(key1_value, "value one");

    ASSERT_TRUE(store_->Get("key2", &key2_value).ok());
    EXPECT_EQ(key2_value, "value two");

    uint64_t key_count = 0;
    ASSERT_TRUE(store_->CountKeys(&key_count).ok());
    EXPECT_EQ(key_count, 2u);
  }
}

TEST_F(SemanticTest, RecoveryPreservesDeduplication) {
  // First session: create deduplicated entries
  {
    auto* embedder = new prestige::testing::DeterministicEmbedder(384);
    ASSERT_TRUE(OpenStore(0.95f, embedder).ok());

    // Same content = should deduplicate
    ASSERT_TRUE(store_->Put("key1", "shared content").ok());
    ASSERT_TRUE(store_->Put("key2", "shared content").ok());

    uint64_t unique_count = 0;
    ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
    EXPECT_EQ(unique_count, 1u);

    store_->Close();
    store_.reset();
  }

  // Second session: verify deduplication persisted
  {
    auto* embedder = new prestige::testing::DeterministicEmbedder(384);
    ASSERT_TRUE(OpenStore(0.95f, embedder).ok());

    uint64_t unique_count = 0;
    ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
    EXPECT_EQ(unique_count, 1u);  // Still deduplicated

    uint64_t key_count = 0;
    ASSERT_TRUE(store_->CountKeys(&key_count).ok());
    EXPECT_EQ(key_count, 2u);  // Still 2 keys

    // Both keys should return same value
    std::string v1, v2;
    ASSERT_TRUE(store_->Get("key1", &v1).ok());
    ASSERT_TRUE(store_->Get("key2", &v2).ok());
    EXPECT_EQ(v1, v2);
  }
}

// =============================================================================
// Vector Index Behavior
// =============================================================================

TEST_F(SemanticTest, LargeNumberOfEntries) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumEntries = 100;
  for (int i = 0; i < kNumEntries; ++i) {
    std::string key = "key_" + std::to_string(i);
    std::string value = "value_" + std::to_string(i);
    ASSERT_TRUE(store_->Put(key, value).ok()) << "Failed at i=" << i;
  }

  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_EQ(key_count, static_cast<uint64_t>(kNumEntries));

  // Verify random access
  for (int i = 0; i < 10; ++i) {
    int idx = i * 10;
    std::string key = "key_" + std::to_string(idx);
    std::string expected = "value_" + std::to_string(idx);
    std::string value;
    ASSERT_TRUE(store_->Get(key, &value).ok());
    EXPECT_EQ(value, expected);
  }
}

TEST_F(SemanticTest, DeleteReducesUniqueCount) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "unique value 1").ok());
  ASSERT_TRUE(store_->Put("key2", "unique value 2").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 2u);

  ASSERT_TRUE(store_->Delete("key1").ok());

  // After GC, unique count should decrease
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);
}

TEST_F(SemanticTest, DeleteSharedValueKeepsOneReference) {
  ASSERT_TRUE(OpenStore().ok());

  // Two keys share same value
  ASSERT_TRUE(store_->Put("key1", "shared").ok());
  ASSERT_TRUE(store_->Put("key2", "shared").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);

  // Delete one key
  ASSERT_TRUE(store_->Delete("key1").ok());

  // Value should still exist (referenced by key2)
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);

  std::string value;
  ASSERT_TRUE(store_->Get("key2", &value).ok());
  EXPECT_EQ(value, "shared");

  // Delete second key
  ASSERT_TRUE(store_->Delete("key2").ok());

  // Now value should be gone
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 0u);
}

}  // namespace

#else  // !PRESTIGE_ENABLE_SEMANTIC

// Provide a single test when semantic mode is disabled to avoid empty test suite
namespace {
TEST(SemanticTestDisabled, SemanticModeNotEnabled) {
  GTEST_SKIP() << "Semantic mode is not enabled (PRESTIGE_ENABLE_SEMANTIC not defined)";
}
}  // namespace

#endif  // PRESTIGE_ENABLE_SEMANTIC
