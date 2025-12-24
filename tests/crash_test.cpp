// Crash harness for durability testing
// Tests that invariants hold after simulated crashes

#include <gtest/gtest.h>

#include <prestige/store.hpp>
#include <prestige/test_utils.hpp>

#include <atomic>
#include <csignal>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

namespace {

class CrashTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() /
                ("prestige_crash_test_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    store_.reset();
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  rocksdb::Status OpenStore(const prestige::Options& opt = prestige::Options{}) {
    return prestige::Store::Open(test_dir_.string(), &store_, opt);
  }

  // Check all invariants after "crash" (unclean close)
  struct InvariantResult {
    bool passed = true;
    std::vector<std::string> violations;

    void AddViolation(const std::string& msg) {
      passed = false;
      violations.push_back(msg);
    }
  };

  InvariantResult CheckInvariants() {
    InvariantResult result;

    // Invariant 1: Every key should map to a valid value
    std::vector<std::string> keys;
    auto s = store_->ListKeys(&keys);
    if (!s.ok()) {
      result.AddViolation("ListKeys failed: " + s.ToString());
      return result;
    }

    for (const auto& key : keys) {
      std::string value;
      s = store_->Get(key, &value);
      if (!s.ok() && !s.IsNotFound()) {
        result.AddViolation("Get(" + key + ") failed: " + s.ToString());
      }
      // Note: NotFound is acceptable if TTL expired
    }

    // Invariant 2: Key count should match ListKeys
    uint64_t key_count = 0;
    s = store_->CountKeys(&key_count);
    if (!s.ok()) {
      result.AddViolation("CountKeys failed: " + s.ToString());
    } else if (key_count != keys.size()) {
      result.AddViolation("CountKeys mismatch: expected " +
                          std::to_string(keys.size()) + ", got " +
                          std::to_string(key_count));
    }

    // Invariant 3: Unique values should be <= key count
    uint64_t unique_count = 0;
    s = store_->CountUniqueValues(&unique_count);
    if (!s.ok()) {
      result.AddViolation("CountUniqueValues failed: " + s.ToString());
    } else if (unique_count > key_count) {
      result.AddViolation("More unique values (" +
                          std::to_string(unique_count) + ") than keys (" +
                          std::to_string(key_count) + ")");
    }

    // Invariant 4: Health stats should be consistent
    prestige::HealthStats health;
    s = store_->GetHealth(&health);
    if (!s.ok()) {
      result.AddViolation("GetHealth failed: " + s.ToString());
    } else {
      if (health.total_keys != key_count) {
        result.AddViolation("Health.total_keys mismatch");
      }
      if (health.total_objects != unique_count) {
        result.AddViolation("Health.total_objects mismatch");
      }
    }

    return result;
  }

  std::filesystem::path test_dir_;
  std::unique_ptr<prestige::Store> store_;
};

// =============================================================================
// Basic Crash Recovery Tests
// =============================================================================

TEST_F(CrashTest, UncleanClosePreservesCommittedData) {
  // Write some data
  {
    ASSERT_TRUE(OpenStore().ok());

    for (int i = 0; i < 100; ++i) {
      std::string key = "key_" + std::to_string(i);
      std::string value = "value_" + std::to_string(i);
      ASSERT_TRUE(store_->Put(key, value).ok());
    }

    // Simulate crash: just reset without Close()
    store_.reset();
  }

  // Reopen and verify
  {
    ASSERT_TRUE(OpenStore().ok());

    auto invariants = CheckInvariants();
    EXPECT_TRUE(invariants.passed) << "Violations: "
        << (invariants.violations.empty() ? "none" : invariants.violations[0]);

    // All committed data should be present
    uint64_t key_count = 0;
    ASSERT_TRUE(store_->CountKeys(&key_count).ok());
    EXPECT_EQ(key_count, 100u);

    // Verify random samples
    for (int i = 0; i < 10; ++i) {
      int idx = i * 10;
      std::string key = "key_" + std::to_string(idx);
      std::string expected = "value_" + std::to_string(idx);
      std::string actual;
      ASSERT_TRUE(store_->Get(key, &actual).ok());
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(CrashTest, UncleanCloseAfterDelete) {
  // Write then delete some data
  {
    ASSERT_TRUE(OpenStore().ok());

    for (int i = 0; i < 50; ++i) {
      std::string key = "key_" + std::to_string(i);
      ASSERT_TRUE(store_->Put(key, "value").ok());
    }

    // Delete half
    for (int i = 0; i < 25; ++i) {
      std::string key = "key_" + std::to_string(i);
      ASSERT_TRUE(store_->Delete(key).ok());
    }

    // Crash!
    store_.reset();
  }

  // Reopen and verify
  {
    ASSERT_TRUE(OpenStore().ok());

    auto invariants = CheckInvariants();
    EXPECT_TRUE(invariants.passed);

    uint64_t key_count = 0;
    ASSERT_TRUE(store_->CountKeys(&key_count).ok());
    EXPECT_EQ(key_count, 25u);

    // Deleted keys should be gone
    for (int i = 0; i < 25; ++i) {
      std::string key = "key_" + std::to_string(i);
      std::string value;
      EXPECT_TRUE(store_->Get(key, &value).IsNotFound());
    }

    // Remaining keys should be present
    for (int i = 25; i < 50; ++i) {
      std::string key = "key_" + std::to_string(i);
      std::string value;
      EXPECT_TRUE(store_->Get(key, &value).ok());
    }
  }
}

// =============================================================================
// Crash During Operations
// =============================================================================

TEST_F(CrashTest, CrashDuringBulkPut) {
  const int kTotalOps = 1000;
  std::atomic<int> ops_completed{0};
  std::atomic<bool> stop_writing{false};

  // Start writing, then "crash" after some ops
  {
    ASSERT_TRUE(OpenStore().ok());

    std::thread writer([this, &ops_completed, &stop_writing]() {
      for (int i = 0; i < kTotalOps && !stop_writing.load(); ++i) {
        std::string key = "bulk_" + std::to_string(i);
        std::string value = "value_" + std::to_string(i);
        auto s = store_->Put(key, value);
        if (s.ok()) {
          ops_completed.fetch_add(1);
        }
      }
    });

    // Wait for some ops to complete
    while (ops_completed.load() < 100) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Signal writer to stop and wait for it to finish
    stop_writing.store(true);
    writer.join();

    // Simulate crash by resetting store without calling Close()
    store_.reset();
  }

  // Reopen and check invariants
  {
    ASSERT_TRUE(OpenStore().ok());

    auto invariants = CheckInvariants();
    EXPECT_TRUE(invariants.passed) << "Invariant violations after crash";

    // Some ops should have completed
    uint64_t key_count = 0;
    ASSERT_TRUE(store_->CountKeys(&key_count).ok());
    EXPECT_GE(key_count, 100u);  // At least what we waited for

    // All present keys should have valid values
    std::vector<std::string> keys;
    ASSERT_TRUE(store_->ListKeys(&keys).ok());
    for (const auto& key : keys) {
      std::string value;
      ASSERT_TRUE(store_->Get(key, &value).ok());
      EXPECT_FALSE(value.empty());
    }
  }
}

TEST_F(CrashTest, CrashDuringOverwrite) {
  // Pre-populate
  {
    ASSERT_TRUE(OpenStore().ok());
    for (int i = 0; i < 50; ++i) {
      ASSERT_TRUE(store_->Put("key_" + std::to_string(i), "original").ok());
    }
    store_->Close();
    store_.reset();
  }

  // Overwrite and crash mid-way
  {
    ASSERT_TRUE(OpenStore().ok());

    for (int i = 0; i < 25; ++i) {
      ASSERT_TRUE(store_->Put("key_" + std::to_string(i), "updated").ok());
    }

    // Crash!
    store_.reset();
  }

  // Verify consistency
  {
    ASSERT_TRUE(OpenStore().ok());

    auto invariants = CheckInvariants();
    EXPECT_TRUE(invariants.passed);

    // First 25 should be updated, rest should be original
    for (int i = 0; i < 50; ++i) {
      std::string key = "key_" + std::to_string(i);
      std::string value;
      ASSERT_TRUE(store_->Get(key, &value).ok());

      if (i < 25) {
        EXPECT_EQ(value, "updated");
      } else {
        EXPECT_EQ(value, "original");
      }
    }
  }
}

// =============================================================================
// Concurrent Crash Recovery
// =============================================================================

TEST_F(CrashTest, CrashWithConcurrentWriters) {
  const int kNumThreads = 4;
  const int kOpsPerThread = 100;
  std::atomic<bool> stop{false};
  std::atomic<int> total_ops{0};

  {
    ASSERT_TRUE(OpenStore().ok());

    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t) {
      threads.emplace_back([this, t, &stop, &total_ops]() {
        int i = 0;
        while (!stop.load() && i < kOpsPerThread) {
          std::string key = "t" + std::to_string(t) + "_k" + std::to_string(i);
          std::string value = "v" + std::to_string(i);
          auto s = store_->Put(key, value);
          if (s.ok()) {
            total_ops.fetch_add(1);
          }
          ++i;
        }
      });
    }

    // Wait for some ops
    while (total_ops.load() < 100) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Stop threads and crash
    stop.store(true);
    for (auto& t : threads) {
      t.join();
    }
    store_.reset();  // Crash!
  }

  // Verify recovery
  {
    ASSERT_TRUE(OpenStore().ok());

    auto invariants = CheckInvariants();
    EXPECT_TRUE(invariants.passed);

    // All successfully committed ops should be present
    std::vector<std::string> keys;
    ASSERT_TRUE(store_->ListKeys(&keys).ok());

    for (const auto& key : keys) {
      std::string value;
      ASSERT_TRUE(store_->Get(key, &value).ok());
      EXPECT_FALSE(value.empty());
    }
  }
}

// =============================================================================
// Multiple Crash Cycles
// =============================================================================

TEST_F(CrashTest, MultipleCrashCycles) {
  const int kCycles = 5;
  const int kOpsPerCycle = 50;

  for (int cycle = 0; cycle < kCycles; ++cycle) {
    {
      ASSERT_TRUE(OpenStore().ok());

      // Add data for this cycle
      for (int i = 0; i < kOpsPerCycle; ++i) {
        std::string key = "c" + std::to_string(cycle) + "_k" + std::to_string(i);
        std::string value = "v" + std::to_string(cycle) + "_" + std::to_string(i);
        ASSERT_TRUE(store_->Put(key, value).ok());
      }

      // Half the time, close cleanly; half the time, crash
      if (cycle % 2 == 0) {
        store_->Close();
      }
      store_.reset();
    }

    // Verify after each crash
    {
      ASSERT_TRUE(OpenStore().ok());

      auto invariants = CheckInvariants();
      EXPECT_TRUE(invariants.passed) << "Cycle " << cycle << " failed";

      // All previous cycles' data should be present
      for (int c = 0; c <= cycle; ++c) {
        for (int i = 0; i < kOpsPerCycle; ++i) {
          std::string key = "c" + std::to_string(c) + "_k" + std::to_string(i);
          std::string value;
          EXPECT_TRUE(store_->Get(key, &value).ok()) << "Missing: " << key;
        }
      }

      store_.reset();
    }
  }
}

// =============================================================================
// Deduplication Consistency After Crash
// =============================================================================

TEST_F(CrashTest, DeduplicationConsistentAfterCrash) {
  // Create deduplicated entries
  {
    ASSERT_TRUE(OpenStore().ok());

    // All these keys share the same value
    for (int i = 0; i < 100; ++i) {
      std::string key = "shared_" + std::to_string(i);
      ASSERT_TRUE(store_->Put(key, "shared_content").ok());
    }

    uint64_t unique_count = 0;
    ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
    EXPECT_EQ(unique_count, 1u);

    // Crash!
    store_.reset();
  }

  // Verify dedup still works
  {
    ASSERT_TRUE(OpenStore().ok());

    auto invariants = CheckInvariants();
    EXPECT_TRUE(invariants.passed);

    uint64_t unique_count = 0;
    ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
    EXPECT_EQ(unique_count, 1u);  // Still deduplicated

    uint64_t key_count = 0;
    ASSERT_TRUE(store_->CountKeys(&key_count).ok());
    EXPECT_EQ(key_count, 100u);

    // Add more with same value
    ASSERT_TRUE(store_->Put("shared_100", "shared_content").ok());

    ASSERT_TRUE(store_->CountUniqueValues(&unique_count).ok());
    EXPECT_EQ(unique_count, 1u);  // Still 1 unique
  }
}

}  // namespace
