// Unit tests for prestige/internal.hpp utilities
// Tests: SHA-256, ObjectMeta serialization, LRU encoding, embedding utilities

#include <gtest/gtest.h>

#include <prestige/internal.hpp>

#include <algorithm>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace prestige::internal {
namespace {

// =============================================================================
// SHA-256 Tests
// =============================================================================

class Sha256Test : public ::testing::Test {};

TEST_F(Sha256Test, EmptyInput) {
  auto digest = Sha256::Digest("");
  // SHA-256 of empty string is well-known
  // e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  EXPECT_EQ(digest[0], 0xe3);
  EXPECT_EQ(digest[1], 0xb0);
  EXPECT_EQ(digest[31], 0x55);
}

TEST_F(Sha256Test, KnownVector) {
  // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
  auto digest = Sha256::Digest("hello");
  EXPECT_EQ(digest[0], 0x2c);
  EXPECT_EQ(digest[1], 0xf2);
  EXPECT_EQ(digest[2], 0x4d);
  EXPECT_EQ(digest[31], 0x24);
}

TEST_F(Sha256Test, DifferentInputsDifferentHashes) {
  auto d1 = Sha256::Digest("hello");
  auto d2 = Sha256::Digest("world");
  auto d3 = Sha256::Digest("hello");

  EXPECT_NE(d1, d2);
  EXPECT_EQ(d1, d3);
}

TEST_F(Sha256Test, LargeInput) {
  std::string large_data(1024 * 1024, 'x');  // 1 MB
  auto digest = Sha256::Digest(large_data);

  // Should complete without error and produce valid digest
  EXPECT_EQ(digest.size(), 32u);

  // Same input should produce same hash
  auto digest2 = Sha256::Digest(large_data);
  EXPECT_EQ(digest, digest2);
}

TEST_F(Sha256Test, BinaryData) {
  std::string binary_data;
  for (int i = 0; i < 256; ++i) {
    binary_data.push_back(static_cast<char>(i));
  }

  auto digest = Sha256::Digest(binary_data);
  EXPECT_EQ(digest.size(), 32u);
}

// =============================================================================
// Timestamp Utility Tests
// =============================================================================

class TimestampTest : public ::testing::Test {};

TEST_F(TimestampTest, NowMicrosMonotonic) {
  uint64_t t1 = NowMicros();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  uint64_t t2 = NowMicros();

  EXPECT_GT(t2, t1);
}

TEST_F(TimestampTest, WallClockMicrosReasonable) {
  uint64_t now = WallClockMicros();

  // Should be after 2020-01-01 00:00:00 UTC (1577836800000000 us)
  EXPECT_GT(now, 1577836800000000ULL);

  // Should be before 2100-01-01 (reasonable upper bound)
  EXPECT_LT(now, 4102444800000000ULL);
}

// =============================================================================
// Encoding Utility Tests
// =============================================================================

class EncodingTest : public ::testing::Test {};

TEST_F(EncodingTest, EncodeDecodeU64LE) {
  std::vector<uint64_t> test_values = {0,
                                       1,
                                       255,
                                       256,
                                       0xFFFF,
                                       0xFFFFFFFF,
                                       0xFFFFFFFFFFFFFFFF,
                                       0x0102030405060708ULL};

  for (uint64_t expected : test_values) {
    std::string encoded = EncodeU64LE(expected);
    ASSERT_EQ(encoded.size(), 8u);

    uint64_t decoded = 0;
    ASSERT_TRUE(DecodeU64LE(encoded, &decoded));
    EXPECT_EQ(decoded, expected) << "Failed for value: " << expected;
  }
}

TEST_F(EncodingTest, DecodeU64LEInvalidLength) {
  uint64_t out = 0;
  EXPECT_FALSE(DecodeU64LE("", &out));
  EXPECT_FALSE(DecodeU64LE("short", &out));
  EXPECT_FALSE(DecodeU64LE("toolongstring", &out));
}

TEST_F(EncodingTest, ToBytesCorrectness) {
  std::array<uint8_t, 4> arr = {0x01, 0x02, 0x03, 0x04};
  std::string result = ToBytes(arr.data(), arr.size());

  EXPECT_EQ(result.size(), 4u);
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x01);
  EXPECT_EQ(static_cast<uint8_t>(result[1]), 0x02);
  EXPECT_EQ(static_cast<uint8_t>(result[2]), 0x03);
  EXPECT_EQ(static_cast<uint8_t>(result[3]), 0x04);
}

// =============================================================================
// RandomObjectId128 Tests
// =============================================================================

class RandomObjectIdTest : public ::testing::Test {};

TEST_F(RandomObjectIdTest, ProducesCorrectSize) {
  auto id = RandomObjectId128();
  EXPECT_EQ(id.size(), 16u);
}

TEST_F(RandomObjectIdTest, ProducesUniqueIds) {
  const int kNumIds = 1000;
  std::set<std::string> ids;

  for (int i = 0; i < kNumIds; ++i) {
    auto id = RandomObjectId128();
    std::string id_str(reinterpret_cast<const char*>(id.data()), id.size());
    ids.insert(id_str);
  }

  // All IDs should be unique
  EXPECT_EQ(ids.size(), kNumIds);
}

TEST_F(RandomObjectIdTest, ThreadSafety) {
  const int kNumThreads = 4;
  const int kIdsPerThread = 250;
  std::vector<std::thread> threads;
  std::vector<std::set<std::string>> per_thread_ids(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([t, &per_thread_ids]() {
      for (int i = 0; i < kIdsPerThread; ++i) {
        auto id = RandomObjectId128();
        std::string id_str(reinterpret_cast<const char*>(id.data()), id.size());
        per_thread_ids[t].insert(id_str);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Combine all IDs and check uniqueness
  std::set<std::string> all_ids;
  for (const auto& ids : per_thread_ids) {
    all_ids.insert(ids.begin(), ids.end());
  }

  EXPECT_EQ(all_ids.size(), kNumThreads * kIdsPerThread);
}

// =============================================================================
// IsRetryableTxnStatus Tests
// =============================================================================

class RetryableStatusTest : public ::testing::Test {};

TEST_F(RetryableStatusTest, RetryableStatuses) {
  EXPECT_TRUE(IsRetryableTxnStatus(rocksdb::Status::Busy()));
  EXPECT_TRUE(IsRetryableTxnStatus(rocksdb::Status::TimedOut()));
  EXPECT_TRUE(IsRetryableTxnStatus(rocksdb::Status::TryAgain()));
  EXPECT_TRUE(IsRetryableTxnStatus(rocksdb::Status::Aborted()));
}

TEST_F(RetryableStatusTest, NonRetryableStatuses) {
  EXPECT_FALSE(IsRetryableTxnStatus(rocksdb::Status::OK()));
  EXPECT_FALSE(IsRetryableTxnStatus(rocksdb::Status::NotFound()));
  EXPECT_FALSE(IsRetryableTxnStatus(rocksdb::Status::Corruption()));
  EXPECT_FALSE(IsRetryableTxnStatus(rocksdb::Status::IOError()));
  EXPECT_FALSE(IsRetryableTxnStatus(rocksdb::Status::InvalidArgument()));
}

// =============================================================================
// ObjectMeta Serialization Tests
// =============================================================================

class ObjectMetaTest : public ::testing::Test {};

TEST_F(ObjectMetaTest, SerializeDeserializeRoundtrip) {
  ObjectMeta original;
  original.digest_key = "test_digest_key_12345678901234567890";
  original.created_at_us = 1234567890123456ULL;
  original.last_accessed_us = 1234567890654321ULL;
  original.size_bytes = 1024 * 1024;

  std::string serialized = original.Serialize();

  ObjectMeta deserialized;
  ASSERT_TRUE(ObjectMeta::Deserialize(serialized, &deserialized));

  EXPECT_EQ(deserialized.digest_key, original.digest_key);
  EXPECT_EQ(deserialized.created_at_us, original.created_at_us);
  EXPECT_EQ(deserialized.last_accessed_us, original.last_accessed_us);
  EXPECT_EQ(deserialized.size_bytes, original.size_bytes);
}

TEST_F(ObjectMetaTest, EmptyDigestKey) {
  ObjectMeta original;
  original.digest_key = "";
  original.created_at_us = 100;
  original.last_accessed_us = 200;
  original.size_bytes = 300;

  std::string serialized = original.Serialize();

  ObjectMeta deserialized;
  ASSERT_TRUE(ObjectMeta::Deserialize(serialized, &deserialized));

  EXPECT_EQ(deserialized.digest_key, "");
  EXPECT_EQ(deserialized.created_at_us, 100u);
}

TEST_F(ObjectMetaTest, BinaryDigestKey) {
  ObjectMeta original;
  // Create a binary digest key (like SHA-256 output)
  original.digest_key.resize(32);
  for (int i = 0; i < 32; ++i) {
    original.digest_key[i] = static_cast<char>(i);
  }
  original.created_at_us = 999;
  original.last_accessed_us = 888;
  original.size_bytes = 777;

  std::string serialized = original.Serialize();

  ObjectMeta deserialized;
  ASSERT_TRUE(ObjectMeta::Deserialize(serialized, &deserialized));

  EXPECT_EQ(deserialized.digest_key.size(), 32u);
  EXPECT_EQ(deserialized.digest_key, original.digest_key);
}

TEST_F(ObjectMetaTest, LegacyFormatFallback) {
  // Simulate legacy format: just raw digest bytes
  std::string legacy_data = "legacy_digest_key";

  ObjectMeta deserialized;
  ASSERT_TRUE(ObjectMeta::Deserialize(legacy_data, &deserialized));

  // Legacy format: digest_key is the entire data, timestamps are 0
  EXPECT_EQ(deserialized.digest_key, legacy_data);
  EXPECT_EQ(deserialized.created_at_us, 0u);
  EXPECT_EQ(deserialized.last_accessed_us, 0u);
  EXPECT_EQ(deserialized.size_bytes, 0u);
  EXPECT_TRUE(deserialized.IsLegacy());
}

TEST_F(ObjectMetaTest, IsLegacyDetection) {
  ObjectMeta legacy;
  legacy.digest_key = "test";
  legacy.created_at_us = 0;
  legacy.last_accessed_us = 0;
  legacy.size_bytes = 0;
  EXPECT_TRUE(legacy.IsLegacy());

  ObjectMeta modern;
  modern.digest_key = "test";
  modern.created_at_us = 1;
  modern.last_accessed_us = 0;
  modern.size_bytes = 0;
  EXPECT_FALSE(modern.IsLegacy());
}

TEST_F(ObjectMetaTest, NullOutput) {
  std::string serialized = ObjectMeta().Serialize();
  EXPECT_FALSE(ObjectMeta::Deserialize(serialized, nullptr));
}

TEST_F(ObjectMetaTest, MaxValues) {
  ObjectMeta original;
  original.digest_key = std::string(1000, 'x');  // Large digest key
  original.created_at_us = UINT64_MAX;
  original.last_accessed_us = UINT64_MAX;
  original.size_bytes = UINT64_MAX;

  std::string serialized = original.Serialize();

  ObjectMeta deserialized;
  ASSERT_TRUE(ObjectMeta::Deserialize(serialized, &deserialized));

  EXPECT_EQ(deserialized.digest_key.size(), 1000u);
  EXPECT_EQ(deserialized.created_at_us, UINT64_MAX);
  EXPECT_EQ(deserialized.last_accessed_us, UINT64_MAX);
  EXPECT_EQ(deserialized.size_bytes, UINT64_MAX);
}

// =============================================================================
// LRU Key Encoding Tests
// =============================================================================

class LRUKeyTest : public ::testing::Test {};

TEST_F(LRUKeyTest, MakeAndParseLRUKey) {
  uint64_t timestamp = 1234567890123456ULL;
  std::string obj_id = "object_id_12345";

  std::string key = MakeLRUKey(timestamp, obj_id);

  uint64_t parsed_ts = 0;
  std::string parsed_id;
  ASSERT_TRUE(ParseLRUKey(key, &parsed_ts, &parsed_id));

  EXPECT_EQ(parsed_ts, timestamp);
  EXPECT_EQ(parsed_id, obj_id);
}

TEST_F(LRUKeyTest, BigEndianSortOrder) {
  // Earlier timestamps should sort before later ones
  std::string key1 = MakeLRUKey(100, "obj1");
  std::string key2 = MakeLRUKey(200, "obj2");
  std::string key3 = MakeLRUKey(100, "obj3");

  // Lexicographic comparison should reflect timestamp order
  EXPECT_LT(key1, key2);
  EXPECT_LT(key3, key2);
  // Same timestamp, different object IDs - sorted by object ID
  EXPECT_LT(key1, key3);  // "obj1" < "obj3"
}

TEST_F(LRUKeyTest, BinaryObjectId) {
  uint64_t timestamp = 12345;
  std::string binary_id(16, '\0');
  for (int i = 0; i < 16; ++i) {
    binary_id[i] = static_cast<char>(i);
  }

  std::string key = MakeLRUKey(timestamp, binary_id);

  uint64_t parsed_ts = 0;
  std::string parsed_id;
  ASSERT_TRUE(ParseLRUKey(key, &parsed_ts, &parsed_id));

  EXPECT_EQ(parsed_ts, timestamp);
  EXPECT_EQ(parsed_id, binary_id);
}

TEST_F(LRUKeyTest, ParseInvalidKey) {
  uint64_t ts = 0;
  std::string id;

  // Too short
  EXPECT_FALSE(ParseLRUKey("short", &ts, &id));
  EXPECT_FALSE(ParseLRUKey("", &ts, &id));

  // Exactly 8 bytes (timestamp only, no object ID) should work
  std::string min_key = MakeLRUKey(123, "");
  EXPECT_TRUE(ParseLRUKey(min_key, &ts, &id));
  EXPECT_EQ(id, "");
}

// =============================================================================
// Embedding Utility Tests
// =============================================================================

class EmbeddingTest : public ::testing::Test {};

TEST_F(EmbeddingTest, SerializeDeserializeRoundtrip) {
  std::vector<float> original = {1.0f, 2.0f, 3.0f, -1.5f, 0.0f, 1e10f};

  std::string serialized = SerializeEmbedding(original);
  EXPECT_EQ(serialized.size(), original.size() * sizeof(float));

  std::vector<float> deserialized;
  ASSERT_TRUE(DeserializeEmbedding(serialized, &deserialized));

  ASSERT_EQ(deserialized.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_FLOAT_EQ(deserialized[i], original[i]);
  }
}

TEST_F(EmbeddingTest, EmptyEmbedding) {
  std::vector<float> empty;
  std::string serialized = SerializeEmbedding(empty);
  EXPECT_EQ(serialized.size(), 0u);

  std::vector<float> deserialized;
  ASSERT_TRUE(DeserializeEmbedding(serialized, &deserialized));
  EXPECT_TRUE(deserialized.empty());
}

TEST_F(EmbeddingTest, DeserializeInvalidSize) {
  // Size not divisible by sizeof(float)
  std::string invalid(5, 'x');
  std::vector<float> out;
  EXPECT_FALSE(DeserializeEmbedding(invalid, &out));
}

TEST_F(EmbeddingTest, CosineSimilarityIdentical) {
  std::vector<float> a = {1.0f, 0.0f, 0.0f};
  EXPECT_FLOAT_EQ(CosineSimilarity(a, a), 1.0f);
}

TEST_F(EmbeddingTest, CosineSimilarityOrthogonal) {
  std::vector<float> a = {1.0f, 0.0f, 0.0f};
  std::vector<float> b = {0.0f, 1.0f, 0.0f};
  EXPECT_NEAR(CosineSimilarity(a, b), 0.0f, 1e-6f);
}

TEST_F(EmbeddingTest, CosineSimilarityOpposite) {
  std::vector<float> a = {1.0f, 0.0f, 0.0f};
  std::vector<float> b = {-1.0f, 0.0f, 0.0f};
  EXPECT_FLOAT_EQ(CosineSimilarity(a, b), -1.0f);
}

TEST_F(EmbeddingTest, CosineSimilaritySizeMismatch) {
  std::vector<float> a = {1.0f, 2.0f};
  std::vector<float> b = {1.0f, 2.0f, 3.0f};
  EXPECT_FLOAT_EQ(CosineSimilarity(a, b), 0.0f);
}

TEST_F(EmbeddingTest, CosineSimilarityEmpty) {
  std::vector<float> a;
  std::vector<float> b;
  EXPECT_FLOAT_EQ(CosineSimilarity(a, b), 0.0f);
}

TEST_F(EmbeddingTest, CosineToL2Conversion) {
  // For unit vectors, L2^2 = 2 - 2*cos(sim)
  // cos = 1 -> L2 = 0
  EXPECT_NEAR(CosineToL2Distance(1.0f), 0.0f, 1e-6f);

  // cos = 0 -> L2 = sqrt(2)
  EXPECT_NEAR(CosineToL2Distance(0.0f), std::sqrt(2.0f), 1e-6f);

  // cos = -1 -> L2 = 2
  EXPECT_NEAR(CosineToL2Distance(-1.0f), 2.0f, 1e-6f);
}

TEST_F(EmbeddingTest, L2ToCosineConversion) {
  // Inverse of CosineToL2Distance
  EXPECT_NEAR(L2DistanceToCosine(0.0f), 1.0f, 1e-6f);
  EXPECT_NEAR(L2DistanceToCosine(std::sqrt(2.0f)), 0.0f, 1e-6f);
  EXPECT_NEAR(L2DistanceToCosine(2.0f), -1.0f, 1e-6f);
}

TEST_F(EmbeddingTest, ConversionRoundtrip) {
  for (float cos_sim = -1.0f; cos_sim <= 1.0f; cos_sim += 0.1f) {
    float l2 = CosineToL2Distance(cos_sim);
    float recovered = L2DistanceToCosine(l2);
    EXPECT_NEAR(recovered, cos_sim, 1e-5f)
        << "Failed roundtrip for cosine similarity: " << cos_sim;
  }
}

// =============================================================================
// Constants Tests
// =============================================================================

TEST(ConstantsTest, EmbeddingDimensions) {
  EXPECT_EQ(kMiniLMDimensions, 384u);
  EXPECT_EQ(kBGESmallDimensions, 384u);
  EXPECT_EQ(kDefaultEmbeddingDimensions, 384u);
}

}  // namespace
}  // namespace prestige::internal
