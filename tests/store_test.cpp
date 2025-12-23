// Unit tests for prestige/store.hpp
// Tests: Core Store operations (Put, Get, Delete, Count, ListKeys)

#include <gtest/gtest.h>

#include <prestige/store.hpp>

#include <algorithm>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace prestige {
namespace {

// =============================================================================
// Test Fixture
// =============================================================================

class StoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create unique test directory for each test
    test_dir_ = std::filesystem::temp_directory_path() / ("prestige_test_" + RandomSuffix());
    std::filesystem::create_directories(test_dir_);
    db_path_ = (test_dir_ / "test_db").string();
  }

  void TearDown() override {
    // Close store before cleanup
    store_.reset();
    // Clean up test directory
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  rocksdb::Status OpenStore(const Options& opt = Options{}) {
    return Store::Open(db_path_, &store_, opt);
  }

  std::string RandomSuffix() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    return std::to_string(dis(gen));
  }

  std::string RandomData(size_t size) {
    std::string data(size, '\0');
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(dis(gen));
    }
    return data;
  }

  std::filesystem::path test_dir_;
  std::string db_path_;
  std::unique_ptr<Store> store_;
};

// =============================================================================
// Open/Close Tests
// =============================================================================

TEST_F(StoreTest, OpenCreatesNewDatabase) {
  ASSERT_TRUE(OpenStore().ok());
  EXPECT_TRUE(std::filesystem::exists(db_path_));
}

TEST_F(StoreTest, OpenExistingDatabase) {
  {
    ASSERT_TRUE(OpenStore().ok());
    ASSERT_TRUE(store_->Put("key1", "value1").ok());
    store_->Close();
  }

  // Reopen and verify data persisted
  ASSERT_TRUE(OpenStore().ok());
  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "value1");
}

TEST_F(StoreTest, OpenWithNullOutput) {
  auto status = Store::Open(db_path_, nullptr);
  EXPECT_TRUE(status.IsInvalidArgument());
}

TEST_F(StoreTest, CloseIsIdempotent) {
  ASSERT_TRUE(OpenStore().ok());
  store_->Close();
  store_->Close();  // Should not crash
  store_->Close();  // Should not crash
}

TEST_F(StoreTest, OperationsAfterClose) {
  ASSERT_TRUE(OpenStore().ok());
  store_->Close();

  // Operations after close should return appropriate errors
  std::string value;
  auto status = store_->Get("key", &value);
  EXPECT_TRUE(status.IsInvalidArgument());
}

// =============================================================================
// Basic Put/Get/Delete Tests
// =============================================================================

TEST_F(StoreTest, PutAndGet) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "value1").ok());

  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "value1");
}

TEST_F(StoreTest, GetNonExistent) {
  ASSERT_TRUE(OpenStore().ok());

  std::string value;
  auto status = store_->Get("nonexistent", &value);
  EXPECT_TRUE(status.IsNotFound());
}

TEST_F(StoreTest, PutOverwrite) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "value1").ok());
  ASSERT_TRUE(store_->Put("key1", "value2").ok());

  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "value2");
}

TEST_F(StoreTest, Delete) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "value1").ok());
  ASSERT_TRUE(store_->Delete("key1").ok());

  std::string value;
  auto status = store_->Get("key1", &value);
  EXPECT_TRUE(status.IsNotFound());
}

TEST_F(StoreTest, DeleteNonExistent) {
  ASSERT_TRUE(OpenStore().ok());

  auto status = store_->Delete("nonexistent");
  EXPECT_TRUE(status.IsNotFound());
}

TEST_F(StoreTest, EmptyKey) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("", "value").ok());

  std::string value;
  ASSERT_TRUE(store_->Get("", &value).ok());
  EXPECT_EQ(value, "value");
}

TEST_F(StoreTest, EmptyValue) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key", "").ok());

  std::string value;
  ASSERT_TRUE(store_->Get("key", &value).ok());
  EXPECT_EQ(value, "");
}

TEST_F(StoreTest, BinaryKeyAndValue) {
  ASSERT_TRUE(OpenStore().ok());

  std::string binary_key(16, '\0');
  std::string binary_value(256, '\0');
  for (int i = 0; i < 16; ++i) binary_key[i] = static_cast<char>(i);
  for (int i = 0; i < 256; ++i) binary_value[i] = static_cast<char>(i);

  ASSERT_TRUE(store_->Put(binary_key, binary_value).ok());

  std::string value;
  ASSERT_TRUE(store_->Get(binary_key, &value).ok());
  EXPECT_EQ(value, binary_value);
}

TEST_F(StoreTest, LargeValue) {
  ASSERT_TRUE(OpenStore().ok());

  std::string large_value(1024 * 1024, 'x');  // 1 MB
  ASSERT_TRUE(store_->Put("large_key", large_value).ok());

  std::string value;
  ASSERT_TRUE(store_->Get("large_key", &value).ok());
  EXPECT_EQ(value, large_value);
}

TEST_F(StoreTest, GetWithNullOutput) {
  ASSERT_TRUE(OpenStore().ok());
  ASSERT_TRUE(store_->Put("key", "value").ok());

  auto status = store_->Get("key", nullptr);
  EXPECT_TRUE(status.IsInvalidArgument());
}

// =============================================================================
// Deduplication Tests
// =============================================================================

TEST_F(StoreTest, DeduplicatesIdenticalValues) {
  ASSERT_TRUE(OpenStore().ok());

  std::string value = "deduplicated_value";
  ASSERT_TRUE(store_->Put("key1", value).ok());
  ASSERT_TRUE(store_->Put("key2", value).ok());
  ASSERT_TRUE(store_->Put("key3", value).ok());

  uint64_t key_count = 0;
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());

  EXPECT_EQ(key_count, 3u);
  EXPECT_EQ(unique_count, 1u);  // Only one unique value stored
}

TEST_F(StoreTest, DifferentValuesNotDeduplicated) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("key1", "value1").ok());
  ASSERT_TRUE(store_->Put("key2", "value2").ok());
  ASSERT_TRUE(store_->Put("key3", "value3").ok());

  uint64_t key_count = 0;
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());

  EXPECT_EQ(key_count, 3u);
  EXPECT_EQ(unique_count, 3u);
}

TEST_F(StoreTest, DeduplicationAfterDelete) {
  ASSERT_TRUE(OpenStore().ok());

  std::string value = "shared_value";
  ASSERT_TRUE(store_->Put("key1", value).ok());
  ASSERT_TRUE(store_->Put("key2", value).ok());

  // Delete one key
  ASSERT_TRUE(store_->Delete("key1").ok());

  // Value should still be accessible via key2
  std::string retrieved;
  ASSERT_TRUE(store_->Get("key2", &retrieved).ok());
  EXPECT_EQ(retrieved, value);

  // Only when last reference is deleted should value be removed
  ASSERT_TRUE(store_->Delete("key2").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 0u);
}

TEST_F(StoreTest, OverwritePreservesDedup) {
  ASSERT_TRUE(OpenStore().ok());

  std::string value = "shared_value";
  ASSERT_TRUE(store_->Put("key1", value).ok());
  ASSERT_TRUE(store_->Put("key2", value).ok());

  // Overwrite key1 with a different value
  ASSERT_TRUE(store_->Put("key1", "different_value").ok());

  // Original value should still be accessible via key2
  std::string retrieved;
  ASSERT_TRUE(store_->Get("key2", &retrieved).ok());
  EXPECT_EQ(retrieved, value);

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 2u);
}

TEST_F(StoreTest, OverwriteWithSameValue) {
  ASSERT_TRUE(OpenStore().ok());

  std::string value = "constant_value";
  ASSERT_TRUE(store_->Put("key1", value).ok());
  ASSERT_TRUE(store_->Put("key1", value).ok());  // Same value
  ASSERT_TRUE(store_->Put("key1", value).ok());  // Same value again

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);

  std::string retrieved;
  ASSERT_TRUE(store_->Get("key1", &retrieved).ok());
  EXPECT_EQ(retrieved, value);
}

// =============================================================================
// Count Tests
// =============================================================================

TEST_F(StoreTest, CountEmptyStore) {
  ASSERT_TRUE(OpenStore().ok());

  uint64_t key_count = 0;
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());

  EXPECT_EQ(key_count, 0u);
  EXPECT_EQ(unique_count, 0u);
}

TEST_F(StoreTest, CountWithNullOutput) {
  ASSERT_TRUE(OpenStore().ok());

  auto status = store_->CountKeys(nullptr);
  EXPECT_TRUE(status.IsInvalidArgument());

  status = store_->CountUniqueValues(nullptr);
  EXPECT_TRUE(status.IsInvalidArgument());
}

TEST_F(StoreTest, CountApproxEmptyStore) {
  ASSERT_TRUE(OpenStore().ok());

  uint64_t key_count = 0;
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountKeysApprox(&key_count).ok());
  ASSERT_TRUE(store_->CountUniqueValuesApprox(&unique_count).ok());

  // Approx counts may not be exactly 0, but should be small
  EXPECT_LE(key_count, 10u);
  EXPECT_LE(unique_count, 10u);
}

TEST_F(StoreTest, CountApproxReasonable) {
  ASSERT_TRUE(OpenStore().ok());

  // Insert 100 unique values
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(store_->Put("key_" + std::to_string(i), "value_" + std::to_string(i)).ok());
  }

  uint64_t exact_keys = 0;
  uint64_t approx_keys = 0;
  ASSERT_TRUE(store_->CountKeys(&exact_keys).ok());
  ASSERT_TRUE(store_->CountKeysApprox(&approx_keys).ok());

  EXPECT_EQ(exact_keys, 100u);
  // Approx should be in reasonable range (50-200% of actual)
  EXPECT_GE(approx_keys, 50u);
  EXPECT_LE(approx_keys, 200u);
}

// =============================================================================
// ListKeys Tests
// =============================================================================

TEST_F(StoreTest, ListKeysEmpty) {
  ASSERT_TRUE(OpenStore().ok());

  std::vector<std::string> keys;
  ASSERT_TRUE(store_->ListKeys(&keys).ok());
  EXPECT_TRUE(keys.empty());
}

TEST_F(StoreTest, ListKeysAll) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("apple", "v1").ok());
  ASSERT_TRUE(store_->Put("banana", "v2").ok());
  ASSERT_TRUE(store_->Put("cherry", "v3").ok());

  std::vector<std::string> keys;
  ASSERT_TRUE(store_->ListKeys(&keys).ok());

  ASSERT_EQ(keys.size(), 3u);
  std::sort(keys.begin(), keys.end());
  EXPECT_EQ(keys[0], "apple");
  EXPECT_EQ(keys[1], "banana");
  EXPECT_EQ(keys[2], "cherry");
}

TEST_F(StoreTest, ListKeysWithLimit) {
  ASSERT_TRUE(OpenStore().ok());

  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(store_->Put("key_" + std::to_string(i), "value").ok());
  }

  std::vector<std::string> keys;
  ASSERT_TRUE(store_->ListKeys(&keys, 5).ok());
  EXPECT_EQ(keys.size(), 5u);
}

TEST_F(StoreTest, ListKeysWithPrefix) {
  ASSERT_TRUE(OpenStore().ok());

  ASSERT_TRUE(store_->Put("user:1", "v1").ok());
  ASSERT_TRUE(store_->Put("user:2", "v2").ok());
  ASSERT_TRUE(store_->Put("user:3", "v3").ok());
  ASSERT_TRUE(store_->Put("item:1", "v4").ok());
  ASSERT_TRUE(store_->Put("item:2", "v5").ok());

  std::vector<std::string> keys;
  ASSERT_TRUE(store_->ListKeys(&keys, 0, "user:").ok());

  EXPECT_EQ(keys.size(), 3u);
  for (const auto& key : keys) {
    EXPECT_TRUE(key.find("user:") == 0);
  }
}

TEST_F(StoreTest, ListKeysWithPrefixAndLimit) {
  ASSERT_TRUE(OpenStore().ok());

  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(store_->Put("prefix_" + std::to_string(i), "value").ok());
  }

  std::vector<std::string> keys;
  ASSERT_TRUE(store_->ListKeys(&keys, 3, "prefix_").ok());
  EXPECT_EQ(keys.size(), 3u);
}

TEST_F(StoreTest, ListKeysWithNullOutput) {
  ASSERT_TRUE(OpenStore().ok());

  auto status = store_->ListKeys(nullptr);
  EXPECT_TRUE(status.IsInvalidArgument());
}

// =============================================================================
// Normalization Mode Tests
// =============================================================================

TEST_F(StoreTest, NormalizationModeWhitespace) {
  Options opt;
  opt.normalization_mode = NormalizationMode::kWhitespace;
  ASSERT_TRUE(OpenStore(opt).ok());

  // These should deduplicate due to whitespace normalization
  ASSERT_TRUE(store_->Put("key1", "hello  world").ok());
  ASSERT_TRUE(store_->Put("key2", "hello world").ok());
  ASSERT_TRUE(store_->Put("key3", "  hello world  ").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);

  // But original values are preserved
  std::string value;
  ASSERT_TRUE(store_->Get("key1", &value).ok());
  EXPECT_EQ(value, "hello  world");
}

TEST_F(StoreTest, NormalizationModeASCII) {
  Options opt;
  opt.normalization_mode = NormalizationMode::kASCII;
  ASSERT_TRUE(OpenStore(opt).ok());

  // These should deduplicate due to case folding
  ASSERT_TRUE(store_->Put("key1", "Hello World").ok());
  ASSERT_TRUE(store_->Put("key2", "HELLO WORLD").ok());
  ASSERT_TRUE(store_->Put("key3", "hello world").ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);
}

// =============================================================================
// Options Tests
// =============================================================================

TEST_F(StoreTest, CustomBlockCache) {
  Options opt;
  opt.block_cache_bytes = 64 * 1024 * 1024;  // 64 MB
  ASSERT_TRUE(OpenStore(opt).ok());
  // Should open without error
}

TEST_F(StoreTest, CustomLockTimeout) {
  Options opt;
  opt.lock_timeout_ms = 5000;
  ASSERT_TRUE(OpenStore(opt).ok());
}

TEST_F(StoreTest, DisableGC) {
  Options opt;
  opt.enable_gc = false;
  ASSERT_TRUE(OpenStore(opt).ok());

  std::string value = "test_value";
  ASSERT_TRUE(store_->Put("key1", value).ok());
  ASSERT_TRUE(store_->Delete("key1").ok());

  // With GC disabled, object count may not decrease immediately
  // (This is more of a behavior verification than a strict test)
}

// =============================================================================
// Health Stats Tests
// =============================================================================

TEST_F(StoreTest, GetHealthEmptyStore) {
  ASSERT_TRUE(OpenStore().ok());

  HealthStats stats;
  ASSERT_TRUE(store_->GetHealth(&stats).ok());

  EXPECT_EQ(stats.total_keys, 0u);
  EXPECT_EQ(stats.total_objects, 0u);
  EXPECT_EQ(stats.total_bytes, 0u);
}

TEST_F(StoreTest, GetHealthWithData) {
  ASSERT_TRUE(OpenStore().ok());

  std::string value1(100, 'a');
  std::string value2(200, 'b');
  ASSERT_TRUE(store_->Put("key1", value1).ok());
  ASSERT_TRUE(store_->Put("key2", value1).ok());  // Deduplicated
  ASSERT_TRUE(store_->Put("key3", value2).ok());

  HealthStats stats;
  ASSERT_TRUE(store_->GetHealth(&stats).ok());

  EXPECT_EQ(stats.total_keys, 3u);
  EXPECT_EQ(stats.total_objects, 2u);
  EXPECT_EQ(stats.total_bytes, 300u);
  EXPECT_DOUBLE_EQ(stats.dedup_ratio, 1.5);  // 3 keys / 2 objects
}

TEST_F(StoreTest, GetHealthWithNullOutput) {
  ASSERT_TRUE(OpenStore().ok());

  auto status = store_->GetHealth(nullptr);
  EXPECT_TRUE(status.IsInvalidArgument());
}

// =============================================================================
// Total Store Bytes Tests
// =============================================================================

TEST_F(StoreTest, GetTotalStoreBytes) {
  ASSERT_TRUE(OpenStore().ok());

  EXPECT_EQ(store_->GetTotalStoreBytes(), 0u);

  std::string value(1000, 'x');
  ASSERT_TRUE(store_->Put("key1", value).ok());

  // Total bytes should increase
  EXPECT_GE(store_->GetTotalStoreBytes(), 1000u);
}

TEST_F(StoreTest, GetTotalStoreBytesApprox) {
  ASSERT_TRUE(OpenStore().ok());

  uint64_t bytes = 0;
  ASSERT_TRUE(store_->GetTotalStoreBytesApprox(&bytes).ok());
  // Approx should be non-negative
  EXPECT_GE(bytes, 0u);
}

// =============================================================================
// Multi-key Operations Tests
// =============================================================================

TEST_F(StoreTest, ManyKeys) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumKeys = 1000;
  for (int i = 0; i < kNumKeys; ++i) {
    ASSERT_TRUE(store_->Put("key_" + std::to_string(i), "value_" + std::to_string(i)).ok());
  }

  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_EQ(key_count, kNumKeys);

  // Verify random access
  for (int i = 0; i < 100; ++i) {
    int idx = std::rand() % kNumKeys;
    std::string value;
    ASSERT_TRUE(store_->Get("key_" + std::to_string(idx), &value).ok());
    EXPECT_EQ(value, "value_" + std::to_string(idx));
  }
}

TEST_F(StoreTest, ManyKeysWithDedup) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumKeys = 1000;
  std::string shared_value = "shared_value_for_all_keys";

  for (int i = 0; i < kNumKeys; ++i) {
    ASSERT_TRUE(store_->Put("key_" + std::to_string(i), shared_value).ok());
  }

  uint64_t key_count = 0;
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());

  EXPECT_EQ(key_count, kNumKeys);
  EXPECT_EQ(unique_count, 1u);  // All deduplicated
}

// =============================================================================
// Persistence Tests
// =============================================================================

TEST_F(StoreTest, DataPersistsAfterClose) {
  const int kNumKeys = 100;

  {
    ASSERT_TRUE(OpenStore().ok());
    for (int i = 0; i < kNumKeys; ++i) {
      ASSERT_TRUE(store_->Put("key_" + std::to_string(i), "value_" + std::to_string(i)).ok());
    }
    store_->Close();
  }

  // Reopen and verify
  ASSERT_TRUE(OpenStore().ok());

  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_EQ(key_count, kNumKeys);

  for (int i = 0; i < kNumKeys; ++i) {
    std::string value;
    ASSERT_TRUE(store_->Get("key_" + std::to_string(i), &value).ok());
    EXPECT_EQ(value, "value_" + std::to_string(i));
  }
}

TEST_F(StoreTest, DedupPersistsAfterClose) {
  std::string shared_value = "shared_value";

  {
    ASSERT_TRUE(OpenStore().ok());
    ASSERT_TRUE(store_->Put("key1", shared_value).ok());
    ASSERT_TRUE(store_->Put("key2", shared_value).ok());
    store_->Close();
  }

  // Reopen and verify dedup
  ASSERT_TRUE(OpenStore().ok());

  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);

  // Adding same value should still deduplicate
  ASSERT_TRUE(store_->Put("key3", shared_value).ok());
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);
}

}  // namespace
}  // namespace prestige
