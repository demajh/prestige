// Integration tests for prestige Store
// Tests: Transaction isolation, concurrent access, crash recovery

#include <gtest/gtest.h>

#include <prestige/store.hpp>
#include <prestige/test_utils.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace prestige {
namespace {

// =============================================================================
// Test Fixture
// =============================================================================

class IntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() / ("prestige_int_test_" + RandomSuffix());
    std::filesystem::create_directories(test_dir_);
    db_path_ = (test_dir_ / "test_db").string();
  }

  void TearDown() override {
    store_.reset();
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

  std::filesystem::path test_dir_;
  std::string db_path_;
  std::unique_ptr<Store> store_;
};

// =============================================================================
// Transaction Isolation Tests
// =============================================================================

TEST_F(IntegrationTest, ConcurrentPutsSameKey) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumThreads = 4;
  const int kOperationsPerThread = 100;
  std::atomic<int> success_count{0};
  std::atomic<int> error_count{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, t, &success_count, &error_count]() {
      for (int i = 0; i < kOperationsPerThread; ++i) {
        std::string value = "thread_" + std::to_string(t) + "_op_" + std::to_string(i);
        auto status = store_->Put("contested_key", value);
        if (status.ok()) {
          success_count++;
        } else {
          error_count++;
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // All operations should succeed (transactions handle contention)
  EXPECT_EQ(success_count.load(), kNumThreads * kOperationsPerThread);
  EXPECT_EQ(error_count.load(), 0);

  // Final value should be set correctly
  std::string value;
  ASSERT_TRUE(store_->Get("contested_key", &value).ok());
  EXPECT_FALSE(value.empty());
}

TEST_F(IntegrationTest, ConcurrentPutsDifferentKeys) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumThreads = 4;
  const int kKeysPerThread = 100;
  testing::TestResultCollector results;

  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, t, &results]() {
      for (int i = 0; i < kKeysPerThread; ++i) {
        std::string key = "thread_" + std::to_string(t) + "_key_" + std::to_string(i);
        std::string value = "value_" + std::to_string(i);
        if (store_->Put(key, value).ok()) {
          results.RecordSuccess();
        } else {
          results.RecordFailure("Put failed for " + key);
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_TRUE(results.AllSucceeded()) << "Failures: " << results.FailureCount();

  // All keys should exist
  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_EQ(key_count, kNumThreads * kKeysPerThread);
}

TEST_F(IntegrationTest, ConcurrentReadsAndWrites) {
  ASSERT_TRUE(OpenStore().ok());

  // Pre-populate some data
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(store_->Put("key_" + std::to_string(i), "value_" + std::to_string(i)).ok());
  }

  const int kNumReaders = 4;
  const int kNumWriters = 2;
  const int kOperations = 100;
  std::atomic<bool> running{true};
  std::atomic<int> read_count{0};
  std::atomic<int> write_count{0};
  testing::TestResultCollector results;

  // Reader threads
  std::vector<std::thread> readers;
  for (int r = 0; r < kNumReaders; ++r) {
    readers.emplace_back([this, &running, &read_count]() {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(0, 99);

      while (running.load()) {
        std::string value;
        std::string key = "key_" + std::to_string(dis(gen));
        store_->Get(key, &value);  // May succeed or not found if deleted
        read_count++;
      }
    });
  }

  // Writer threads
  std::vector<std::thread> writers;
  for (int w = 0; w < kNumWriters; ++w) {
    writers.emplace_back([this, w, &write_count, &results]() {
      for (int i = 0; i < kOperations; ++i) {
        std::string key = "new_key_" + std::to_string(w) + "_" + std::to_string(i);
        if (store_->Put(key, "new_value").ok()) {
          results.RecordSuccess();
          write_count++;
        } else {
          results.RecordFailure("Put failed for " + key);
        }
      }
    });
  }

  // Wait for writers to finish
  for (auto& w : writers) {
    w.join();
  }

  // Stop readers
  running.store(false);
  for (auto& r : readers) {
    r.join();
  }

  EXPECT_TRUE(results.AllSucceeded()) << "Write failures: " << results.FailureCount();
  EXPECT_EQ(write_count.load(), kNumWriters * kOperations);
  EXPECT_GT(read_count.load(), 0);
}

TEST_F(IntegrationTest, ConcurrentPutAndDelete) {
  ASSERT_TRUE(OpenStore().ok());

  const int kIterations = 100;
  std::atomic<int> put_success{0};
  std::atomic<int> delete_success{0};

  for (int iter = 0; iter < kIterations; ++iter) {
    std::string key = "temp_key";

    // One thread puts, another deletes
    auto put_future = std::async(std::launch::async, [this, &put_success]() {
      if (store_->Put("temp_key", "temp_value").ok()) {
        put_success++;
      }
    });

    auto delete_future = std::async(std::launch::async, [this, &delete_success]() {
      auto status = store_->Delete("temp_key");
      if (status.ok() || status.IsNotFound()) {
        delete_success++;  // Both outcomes are valid
      }
    });

    put_future.wait();
    delete_future.wait();
  }

  // All operations should complete without errors
  EXPECT_EQ(put_success.load(), kIterations);
  EXPECT_EQ(delete_success.load(), kIterations);
}

TEST_F(IntegrationTest, ConcurrentDeduplication) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumThreads = 4;
  const int kKeysPerThread = 50;
  const std::string kSharedValue = "shared_deduplicated_value";
  testing::TestResultCollector results;

  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, t, &kSharedValue, &results]() {
      for (int i = 0; i < kKeysPerThread; ++i) {
        std::string key = "thread_" + std::to_string(t) + "_key_" + std::to_string(i);
        if (store_->Put(key, kSharedValue).ok()) {
          results.RecordSuccess();
        } else {
          results.RecordFailure("Put failed for " + key);
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_TRUE(results.AllSucceeded()) << "Failures: " << results.FailureCount();

  // All keys should exist
  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_EQ(key_count, kNumThreads * kKeysPerThread);

  // But only one unique value due to deduplication
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);
}

TEST_F(IntegrationTest, ConcurrentOverwrites) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumThreads = 4;
  const int kIterations = 50;

  // Pre-create key
  ASSERT_TRUE(store_->Put("overwrite_key", "initial_value").ok());

  std::atomic<int> success_count{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, t, &success_count]() {
      for (int i = 0; i < kIterations; ++i) {
        std::string value = "thread_" + std::to_string(t) + "_iter_" + std::to_string(i);
        if (store_->Put("overwrite_key", value).ok()) {
          success_count++;
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), kNumThreads * kIterations);

  // Key should have a valid value
  std::string value;
  ASSERT_TRUE(store_->Get("overwrite_key", &value).ok());
  EXPECT_FALSE(value.empty());

  // Should still be just one key
  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_EQ(key_count, 1u);
}

// =============================================================================
// Retry Logic Tests
// =============================================================================

TEST_F(IntegrationTest, RetryBackoffUnderContention) {
  Options opt;
  opt.max_retries = 16;
  opt.retry_base_delay_us = 1000;
  opt.retry_max_delay_us = 100000;
  opt.retry_jitter_factor = 0.5;
  ASSERT_TRUE(OpenStore(opt).ok());

  const int kNumThreads = 8;
  const int kOperations = 50;
  std::atomic<int> success_count{0};

  auto start = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, &success_count]() {
      for (int i = 0; i < kOperations; ++i) {
        if (store_->Put("hot_key", "value").ok()) {
          success_count++;
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  auto elapsed = std::chrono::steady_clock::now() - start;
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

  // All should succeed due to retries
  EXPECT_EQ(success_count.load(), kNumThreads * kOperations);

  // With backoff, should complete in reasonable time despite contention
  // (This is more of a sanity check than a strict timing test)
  EXPECT_LT(elapsed_ms, 30000);  // Should complete within 30 seconds
}

// =============================================================================
// Crash Recovery Tests
// =============================================================================

TEST_F(IntegrationTest, RecoveryAfterCleanClose) {
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

  // Verify all values
  for (int i = 0; i < kNumKeys; ++i) {
    std::string value;
    ASSERT_TRUE(store_->Get("key_" + std::to_string(i), &value).ok());
    EXPECT_EQ(value, "value_" + std::to_string(i));
  }
}

TEST_F(IntegrationTest, RecoveryPreservesDeduplication) {
  const std::string kSharedValue = "shared_value_for_recovery_test";

  {
    ASSERT_TRUE(OpenStore().ok());
    for (int i = 0; i < 10; ++i) {
      ASSERT_TRUE(store_->Put("key_" + std::to_string(i), kSharedValue).ok());
    }

    uint64_t unique_before = 0;
    ASSERT_TRUE(store_->CountUniqueValues(&unique_before).ok());
    EXPECT_EQ(unique_before, 1u);

    store_->Close();
  }

  // Reopen and verify deduplication persisted
  ASSERT_TRUE(OpenStore().ok());

  uint64_t unique_after = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_after).ok());
  EXPECT_EQ(unique_after, 1u);

  // Adding more keys with same value should still deduplicate
  ASSERT_TRUE(store_->Put("new_key", kSharedValue).ok());
  ASSERT_TRUE(store_->CountUniqueValues(&unique_after).ok());
  EXPECT_EQ(unique_after, 1u);
}

TEST_F(IntegrationTest, RecoveryAfterMultipleOpenClose) {
  for (int round = 0; round < 5; ++round) {
    ASSERT_TRUE(OpenStore().ok());

    // Add some data each round
    for (int i = 0; i < 10; ++i) {
      std::string key = "round_" + std::to_string(round) + "_key_" + std::to_string(i);
      ASSERT_TRUE(store_->Put(key, "value_" + std::to_string(i)).ok());
    }

    store_->Close();
  }

  // Final reopen and verify all data
  ASSERT_TRUE(OpenStore().ok());

  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_EQ(key_count, 50u);  // 5 rounds * 10 keys
}

TEST_F(IntegrationTest, RecoveryPreservesMetadata) {
  Options opt;
  opt.track_access_time = true;
  opt.lru_update_interval_seconds = 0;

  {
    ASSERT_TRUE(OpenStore(opt).ok());
    ASSERT_TRUE(store_->Put("test_key", "test_value").ok());

    // Access to update LRU
    std::string value;
    ASSERT_TRUE(store_->Get("test_key", &value).ok());

    HealthStats stats;
    ASSERT_TRUE(store_->GetHealth(&stats).ok());
    EXPECT_EQ(stats.total_keys, 1u);

    store_->Close();
  }

  // Reopen and verify metadata
  ASSERT_TRUE(OpenStore(opt).ok());

  HealthStats stats;
  ASSERT_TRUE(store_->GetHealth(&stats).ok());
  EXPECT_EQ(stats.total_keys, 1u);
  EXPECT_EQ(stats.total_objects, 1u);
}

// =============================================================================
// Simulated Crash Recovery Tests
// =============================================================================

TEST_F(IntegrationTest, RecoveryAfterUncleanShutdown) {
  // Simulate an unclean shutdown by not calling Close()
  const int kNumKeys = 50;

  {
    ASSERT_TRUE(OpenStore().ok());
    for (int i = 0; i < kNumKeys; ++i) {
      ASSERT_TRUE(store_->Put("key_" + std::to_string(i), "value_" + std::to_string(i)).ok());
    }
    // Intentionally NOT calling Close() - simulating crash
    store_.reset();  // Destructor should handle cleanup
  }

  // Reopen after "crash"
  ASSERT_TRUE(OpenStore().ok());

  // Data should still be recoverable (RocksDB WAL handles this)
  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  // May have all or most keys depending on WAL flush timing
  EXPECT_GT(key_count, 0u);
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(IntegrationTest, HighVolumeOperations) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumOperations = 1000;

  for (int i = 0; i < kNumOperations; ++i) {
    std::string key = "key_" + std::to_string(i);
    std::string value = "value_" + std::to_string(i);
    ASSERT_TRUE(store_->Put(key, value).ok());

    // Occasionally read back
    if (i % 100 == 0) {
      std::string read_value;
      ASSERT_TRUE(store_->Get(key, &read_value).ok());
      EXPECT_EQ(read_value, value);
    }

    // Occasionally delete
    if (i % 200 == 0 && i > 0) {
      std::string del_key = "key_" + std::to_string(i - 100);
      ASSERT_TRUE(store_->Delete(del_key).ok());
    }
  }

  // Verify final state
  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  EXPECT_GT(key_count, 0u);
}

TEST_F(IntegrationTest, LargeValueConcurrency) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumThreads = 4;
  const int kValuesPerThread = 10;
  const size_t kValueSize = 100 * 1024;  // 100 KB
  testing::TestResultCollector results;

  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, t, kValueSize, &results]() {
      std::string large_value(kValueSize, 'x');
      for (int i = 0; i < kValuesPerThread; ++i) {
        std::string key = "large_" + std::to_string(t) + "_" + std::to_string(i);
        if (store_->Put(key, large_value).ok()) {
          results.RecordSuccess();
        } else {
          results.RecordFailure("Put failed for " + key);
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_TRUE(results.AllSucceeded()) << "Failures: " << results.FailureCount();

  // All values should be deduplicated (same content)
  uint64_t unique_count = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
  EXPECT_EQ(unique_count, 1u);
}

// =============================================================================
// Data Integrity Tests
// =============================================================================

TEST_F(IntegrationTest, DataIntegrityUnderConcurrency) {
  ASSERT_TRUE(OpenStore().ok());

  const int kNumThreads = 4;
  const int kKeysPerThread = 50;
  testing::TestResultCollector results;

  // Each thread writes unique values
  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, t, &results]() {
      for (int i = 0; i < kKeysPerThread; ++i) {
        std::string key = "t" + std::to_string(t) + "_k" + std::to_string(i);
        std::string value = "THREAD_" + std::to_string(t) + "_VALUE_" + std::to_string(i);
        if (store_->Put(key, value).ok()) {
          results.RecordSuccess();
        } else {
          results.RecordFailure("Put failed for " + key);
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_TRUE(results.AllSucceeded()) << "Write failures: " << results.FailureCount();

  // Verify all values are correct
  for (int t = 0; t < kNumThreads; ++t) {
    for (int i = 0; i < kKeysPerThread; ++i) {
      std::string key = "t" + std::to_string(t) + "_k" + std::to_string(i);
      std::string expected = "THREAD_" + std::to_string(t) + "_VALUE_" + std::to_string(i);
      std::string actual;
      ASSERT_TRUE(store_->Get(key, &actual).ok());
      EXPECT_EQ(actual, expected) << "Data corruption for key: " << key;
    }
  }
}

// =============================================================================
// Mixed Workload Tests
// =============================================================================

TEST_F(IntegrationTest, MixedReadWriteDeleteWorkload) {
  ASSERT_TRUE(OpenStore().ok());

  const int kOperations = 50;
  std::atomic<bool> running{true};

  // Pre-populate
  for (int i = 0; i < 50; ++i) {
    ASSERT_TRUE(store_->Put("init_key_" + std::to_string(i), "init_value").ok());
  }

  std::vector<std::thread> threads;

  // Writer thread
  threads.emplace_back([this]() {
    for (int i = 0; i < 50; ++i) {
      store_->Put("write_" + std::to_string(i), "written_value");
    }
  });

  // Reader thread
  threads.emplace_back([this, &running]() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 49);
    int reads = 0;
    while (running.load() && reads < 200) {
      std::string value;
      store_->Get("init_key_" + std::to_string(dis(gen)), &value);
      reads++;
    }
  });

  // Deleter thread
  threads.emplace_back([this]() {
    for (int i = 0; i < 25; ++i) {
      store_->Delete("init_key_" + std::to_string(i));
    }
  });

  // Wait for writer and deleter
  threads[0].join();
  threads[2].join();

  running.store(false);
  threads[1].join();

  // Verify store is in consistent state
  uint64_t key_count = 0;
  ASSERT_TRUE(store_->CountKeys(&key_count).ok());
  // Some initial keys deleted, some new keys written
  EXPECT_GT(key_count, 0u);
}

}  // namespace
}  // namespace prestige
