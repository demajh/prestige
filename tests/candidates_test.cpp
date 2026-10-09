// Tests for value metadata, the candidates call and outcome records. See docs/candidates.md.

#include <gtest/gtest.h>

#include <prestige/store.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC
#include <prestige/test_utils.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace prestige {
namespace {

class CandidatesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() / ("prestige_candidates_test_" + RandomSuffix());
    std::filesystem::create_directories(test_dir_);
    db_path_ = (test_dir_ / "db").string();
  }

  void TearDown() override {
    store_.reset();
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  rocksdb::Status OpenStore(const Options& opt = Options{}) { return Store::Open(db_path_, &store_, opt); }

  static std::string RandomSuffix() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    return std::to_string(dis(gen));
  }

  std::vector<Candidate> Query(const std::string& value, CandidateQuery q = CandidateQuery{}) {
    std::vector<Candidate> out;
    EXPECT_TRUE(store_->Candidates(value, q, &out).ok());
    return out;
  }

  Metadata MetaOf(const std::string& key) {
    Metadata m;
    EXPECT_TRUE(store_->GetMetadata(key, &m).ok());
    return m;
  }

  static Outcome MakeOutcome(const std::string& family, OutcomeVerdict verdict, float similarity, uint32_t rank = 0) {
    Outcome o;
    o.family_id = family;
    o.verdict = verdict;
    o.similarity = similarity;
    o.rank = rank;
    return o;
  }

  std::filesystem::path test_dir_;
  std::string db_path_;
  std::unique_ptr<Store> store_;
};

TEST_F(CandidatesTest, ExactModeReturnsTheIdenticalValueWithMetadataOrNothing) {
  ASSERT_TRUE(OpenStore().ok());
  const Metadata md{{"family", "support-faq@v3"}, {"tool", "resolver/1.4"}, {"schema", "constraints@2"}};
  ASSERT_TRUE(store_->Put("k1", "the answer", md).ok());

  auto out = Query("the answer");
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].rank, 0u);
  EXPECT_FLOAT_EQ(out[0].similarity, 1.0f);
  EXPECT_FLOAT_EQ(out[0].reranker_score, -1.0f);
  EXPECT_EQ(out[0].metadata, md);
  EXPECT_EQ(out[0].size_bytes, 10u);
  EXPECT_GT(out[0].created_at_us, 0u);
  std::string digest;
  ASSERT_TRUE(store_->Digest("the answer", &digest).ok());
  EXPECT_EQ(out[0].digest, digest);
  EXPECT_TRUE(out[0].value.empty());

  CandidateQuery with_values;
  with_values.include_values = true;
  out = Query("the answer", with_values);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].value, "the answer");

  EXPECT_TRUE(Query("something else").empty());
}

TEST_F(CandidatesTest, FilterRequiresEveryPair) {
  ASSERT_TRUE(OpenStore().ok());
  ASSERT_TRUE(store_->Put("k1", "v", Metadata{{"family", "a"}, {"tool", "t1"}}).ok());
  CandidateQuery q;
  q.filter = {{"family", "a"}};
  EXPECT_EQ(Query("v", q).size(), 1u);
  q.filter = {{"family", "b"}};
  EXPECT_EQ(Query("v", q).size(), 0u);
  q.filter = {{"family", "a"}, {"tool", "t2"}};
  EXPECT_EQ(Query("v", q).size(), 0u);
  q.filter = {{"family", "a"}, {"tool", "t1"}};
  EXPECT_EQ(Query("v", q).size(), 1u);
}

TEST_F(CandidatesTest, MetadataBelongsToTheValueAndMergesWithLaterPairsWinning) {
  ASSERT_TRUE(OpenStore().ok());
  ASSERT_TRUE(store_->Put("k1", "v", Metadata{{"a", "1"}, {"b", "2"}}).ok());
  ASSERT_TRUE(store_->Put("k2", "v", Metadata{{"b", "3"}, {"c", "4"}}).ok());  // same bytes, new key
  const Metadata expected{{"a", "1"}, {"b", "3"}, {"c", "4"}};
  EXPECT_EQ(MetaOf("k1"), expected);
  EXPECT_EQ(MetaOf("k2"), expected);

  ASSERT_TRUE(store_->Put("k1", "v").ok());  // a plain write leaves metadata alone
  EXPECT_EQ(MetaOf("k1"), expected);

  // Same key, same value, new metadata: a no-op overwrite still records the metadata
  ASSERT_TRUE(store_->Put("k1", "v", Metadata{{"d", "5"}}).ok());
  EXPECT_EQ(MetaOf("k1").at("d"), "5");
  EXPECT_EQ(MetaOf("k1").size(), 4u);

  uint64_t objects = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&objects).ok());
  EXPECT_EQ(objects, 1u);
}

TEST_F(CandidatesTest, MetadataIsRemovedWithTheObject) {
  ASSERT_TRUE(OpenStore().ok());
  ASSERT_TRUE(store_->Put("k", "v", Metadata{{"a", "1"}}).ok());
  ASSERT_TRUE(store_->Delete("k").ok());
  ASSERT_TRUE(store_->Put("k2", "v").ok());  // fresh object for the same bytes
  EXPECT_TRUE(MetaOf("k2").empty());
  Metadata m;
  EXPECT_TRUE(store_->GetMetadata("missing", &m).IsNotFound());
}

TEST_F(CandidatesTest, MetadataValidation) {
  ASSERT_TRUE(OpenStore().ok());
  Metadata too_many;
  for (int i = 0; i < 65; ++i) too_many["k" + std::to_string(i)] = "v";
  EXPECT_TRUE(store_->Put("k", "v", too_many).IsInvalidArgument());
  EXPECT_TRUE(store_->Put("k", "v", Metadata{{"", "v"}}).IsInvalidArgument());
  uint64_t keys = 1;
  ASSERT_TRUE(store_->CountKeys(&keys).ok());
  EXPECT_EQ(keys, 0u);
}

TEST_F(CandidatesTest, DecisionAndMetadataCommitTogether) {
  ASSERT_TRUE(OpenStore().ok());
  Decision d;
  d.decision_id = "dec-1";
  d.policy_revision = "p1";
  ASSERT_TRUE(store_->PutWithDecision("k", "v", d, Metadata{{"family", "f@1"}}).ok());
  EXPECT_EQ(MetaOf("k").at("family"), "f@1");
  DecisionRecord rec;
  ASSERT_TRUE(store_->GetDecision("dec-1", &rec).ok());
  EXPECT_EQ(rec.user_key, "k");
}

TEST_F(CandidatesTest, OutcomesAreRecordedListedAndReported) {
  ASSERT_TRUE(OpenStore().ok());
  uint64_t seq = 0;
  Outcome accepted = MakeOutcome("support-faq@v3", OutcomeVerdict::kAccepted, 0.97f, 0);
  accepted.tool_version = "resolver/1.4";
  accepted.schema_version = "constraints@2";
  ASSERT_TRUE(store_->RecordOutcome(accepted, &seq).ok());
  EXPECT_EQ(seq, 1u);

  Outcome rejected = MakeOutcome("support-faq@v3", OutcomeVerdict::kRejected, 0.93f, 1);
  rejected.reason = "tenant mismatch";
  rejected.threshold = 0.92f;
  ASSERT_TRUE(store_->RecordOutcome(rejected, &seq).ok());
  EXPECT_EQ(seq, 2u);

  ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("support-faq@v3", OutcomeVerdict::kNoCandidate, -1.0f), &seq).ok());
  EXPECT_EQ(seq, 3u);
  ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("billing@v1", OutcomeVerdict::kAccepted, 0.99f), &seq).ok());
  EXPECT_EQ(seq, 4u);

  std::vector<OutcomeRecord> list;
  ASSERT_TRUE(store_->ListOutcomes("support-faq@v3", &list).ok());
  ASSERT_EQ(list.size(), 3u);
  EXPECT_EQ(list[0].sequence, 1u);
  EXPECT_EQ(list[1].outcome.reason, "tenant mismatch");
  EXPECT_FLOAT_EQ(list[1].outcome.threshold, 0.92f);
  EXPECT_EQ(list[2].outcome.verdict, OutcomeVerdict::kNoCandidate);
  EXPECT_GT(list[0].recorded_at_us, 0u);

  ASSERT_TRUE(store_->ListOutcomes("support-faq@v3", &list, 1, 1).ok());
  ASSERT_EQ(list.size(), 1u);
  EXPECT_EQ(list[0].sequence, 2u);

  FamilyReport r;
  ASSERT_TRUE(store_->GetFamilyReport("support-faq@v3", &r).ok());
  EXPECT_EQ(r.accepted, 1u);
  EXPECT_EQ(r.rejected, 1u);
  EXPECT_EQ(r.no_candidate, 1u);
  EXPECT_DOUBLE_EQ(r.false_accept_rate, 0.5);
  ASSERT_EQ(r.accepted_similarity_hist.size(), 20u);
  EXPECT_EQ(r.accepted_similarity_hist[19], 1u);  // 0.97 falls in [0.95, 1.0]
  EXPECT_EQ(r.rejected_similarity_hist[18], 1u);  // 0.93 falls in [0.90, 0.95)
  EXPECT_EQ(r.rejected_rank_hist[1], 1u);
  EXPECT_EQ(r.first_sequence, 1u);
  EXPECT_EQ(r.last_sequence, 3u);
  EXPECT_FLOAT_EQ(r.suggested_threshold, -1.0f);  // too little evidence

  std::vector<std::string> families;
  ASSERT_TRUE(store_->ListFamilies(&families).ok());
  ASSERT_EQ(families.size(), 2u);
  EXPECT_EQ(families[0], "billing@v1");
  EXPECT_EQ(families[1], "support-faq@v3");

  EXPECT_TRUE(store_->GetFamilyReport("unknown", &r).IsNotFound());
}

TEST_F(CandidatesTest, SuggestedThresholdNeedsEnoughEvidenceAndIsAdvisory) {
  ASSERT_TRUE(OpenStore().ok());
  for (int i = 0; i < 40; ++i) ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("f", OutcomeVerdict::kAccepted, 0.96f)).ok());
  for (int i = 0; i < 2; ++i) ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("f", OutcomeVerdict::kRejected, 0.97f, 0)).ok());
  for (int i = 0; i < 10; ++i) ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("f", OutcomeVerdict::kRejected, 0.91f, 2)).ok());

  FamilyReport r;
  ASSERT_TRUE(store_->GetFamilyReport("f", &r).ok());
  EXPECT_EQ(r.accepted, 40u);
  EXPECT_EQ(r.rejected, 12u);
  EXPECT_NEAR(r.false_accept_rate, 12.0 / 52.0, 1e-9);
  // Above 0.95: 40 accepted, 2 rejected (4.8%). Above 0.90: 12 rejected of 52 (23%). The advice is 0.95.
  EXPECT_FLOAT_EQ(r.suggested_threshold, 0.95f);
  EXPECT_EQ(r.rejected_rank_hist[2], 10u);
}

TEST_F(CandidatesTest, SuggestedThresholdNeverFallsBelowTheLowestJudgedBucket) {
  ASSERT_TRUE(OpenStore().ok());
  // Every judged candidate sits at or above 0.90, as happens when a family is gated at 0.90 from the start.
  for (int i = 0; i < 30; ++i) ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("g", OutcomeVerdict::kAccepted, 0.96f)).ok());
  ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("g", OutcomeVerdict::kRejected, 0.93f, 0)).ok());

  FamilyReport r;
  ASSERT_TRUE(store_->GetFamilyReport("g", &r).ok());
  // Above 0.95: 30 of 30 accepted. Above 0.90: 1 rejected of 31 (3.2%), still within 5%, so the edge moves down to
  // 0.90. Below that nothing was judged, so the cumulative counts would not change and the advice must stop there
  // instead of sliding to 0.00.
  EXPECT_FLOAT_EQ(r.suggested_threshold, 0.90f);
}

TEST_F(CandidatesTest, OutcomeValidation) {
  ASSERT_TRUE(OpenStore().ok());
  EXPECT_TRUE(store_->RecordOutcome(MakeOutcome("", OutcomeVerdict::kAccepted, 0.9f)).IsInvalidArgument());
  Outcome bad_verdict = MakeOutcome("f", static_cast<OutcomeVerdict>(7), 0.9f);
  EXPECT_TRUE(store_->RecordOutcome(bad_verdict).IsInvalidArgument());
  Outcome bad_digest = MakeOutcome("f", OutcomeVerdict::kAccepted, 0.9f);
  bad_digest.candidate_digest = "short";
  EXPECT_TRUE(store_->RecordOutcome(bad_digest).IsInvalidArgument());
  std::vector<std::string> families;
  ASSERT_TRUE(store_->ListFamilies(&families).ok());
  EXPECT_TRUE(families.empty());
}

TEST_F(CandidatesTest, OutcomesAndMetadataSurviveReopen) {
  ASSERT_TRUE(OpenStore().ok());
  ASSERT_TRUE(store_->Put("k", "v", Metadata{{"family", "f@1"}}).ok());
  ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("f@1", OutcomeVerdict::kAccepted, 0.9f)).ok());
  ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("f@1", OutcomeVerdict::kRejected, 0.9f)).ok());
  store_.reset();

  ASSERT_TRUE(OpenStore().ok());
  EXPECT_EQ(MetaOf("k").at("family"), "f@1");
  uint64_t seq = 0;
  ASSERT_TRUE(store_->RecordOutcome(MakeOutcome("f@1", OutcomeVerdict::kAccepted, 0.9f), &seq).ok());
  EXPECT_EQ(seq, 3u);
  FamilyReport r;
  ASSERT_TRUE(store_->GetFamilyReport("f@1", &r).ok());
  EXPECT_EQ(r.accepted + r.rejected, 3u);
}

#ifdef PRESTIGE_ENABLE_SEMANTIC

// Semantic-mode tests pin exact embeddings so similarities are known: e0 = (1,0,0,...), a value rotated towards
// the second axis has cosine 1/sqrt(1.04) = 0.98 with e0, the second axis itself has cosine 0 with e0 and 0.196
// with the rotated value, and the third axis is orthogonal to all of them.
class SemanticCandidatesTest : public CandidatesTest {
 protected:
  static std::vector<float> Axis(size_t i, float tilt_towards_second_axis = 0.0f) {
    std::vector<float> v(384, 0.0f);
    v[i] = 1.0f;
    if (tilt_towards_second_axis > 0.0f) v[1] = tilt_towards_second_axis;
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    for (float& x : v) x /= norm;
    return v;
  }

  rocksdb::Status OpenSemantic(float threshold = 0.95f) {
    // The store takes ownership of custom_embedder (it is released when the store closes), as in SemanticTest.
    embedder_ = new prestige::testing::DeterministicEmbedder();
    embedder_->RegisterEmbedding("alpha one", Axis(0));
    embedder_->RegisterEmbedding("alpha two", Axis(0, 0.2f));    // 0.98 to "alpha one": a semantic duplicate
    embedder_->RegisterEmbedding("alpha four", Axis(0, 0.2f));   // the query in the ranking test
    embedder_->RegisterEmbedding("beta two", Axis(1));           // 0.196 to the query, 0 to "alpha one"
    embedder_->RegisterEmbedding("gamma three", Axis(2));        // orthogonal to everything
    Options opt;
    opt.dedup_mode = DedupMode::kSemantic;
    opt.semantic_threshold = threshold;
    opt.custom_embedder = embedder_;
    opt.semantic_index_save_interval = 0;
    return OpenStore(opt);
  }
  prestige::testing::DeterministicEmbedder* embedder_ = nullptr;  // owned by the store once opened
};

TEST_F(SemanticCandidatesTest, RanksNeighboursWithScoresAndMetadata) {
  ASSERT_TRUE(OpenSemantic().ok());
  ASSERT_TRUE(store_->Put("k1", "alpha one", Metadata{{"family", "a"}}).ok());
  ASSERT_TRUE(store_->Put("k2", "beta two", Metadata{{"family", "b"}}).ok());
  ASSERT_TRUE(store_->Put("k3", "gamma three", Metadata{{"family", "a"}}).ok());
  uint64_t objects = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&objects).ok());
  ASSERT_EQ(objects, 3u);  // none of the three deduplicated against another

  CandidateQuery q;
  q.k = 10;
  q.include_values = true;
  auto out = Query("alpha four", q);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].rank, 0u);
  EXPECT_EQ(out[0].value, "alpha one");
  EXPECT_NEAR(out[0].similarity, 0.98f, 0.01f);
  EXPECT_EQ(out[0].metadata.at("family"), "a");
  EXPECT_FLOAT_EQ(out[0].reranker_score, -1.0f);  // no reranker configured
  EXPECT_EQ(out[0].size_bytes, 9u);
  EXPECT_EQ(out[1].value, "beta two");
  EXPECT_NEAR(out[1].similarity, 0.196f, 0.01f);
  EXPECT_EQ(out[2].value, "gamma three");
  EXPECT_NEAR(out[2].similarity, 0.0f, 0.01f);
  for (size_t i = 0; i < out.size(); ++i) EXPECT_EQ(out[i].rank, i);

  CandidateQuery two;
  two.k = 2;
  EXPECT_EQ(Query("alpha four", two).size(), 2u);

  CandidateQuery floor;
  floor.min_similarity = 0.9f;
  auto close = Query("alpha four", floor);
  ASSERT_EQ(close.size(), 1u);
  EXPECT_EQ(close[0].object_id, out[0].object_id);

  CandidateQuery filtered;
  filtered.filter = {{"family", "b"}};
  auto only_b = Query("alpha four", filtered);
  ASSERT_EQ(only_b.size(), 1u);
  EXPECT_EQ(only_b[0].metadata.at("family"), "b");
  EXPECT_EQ(only_b[0].rank, 0u);  // ranks are assigned after filtering

  // The store ranks; it never decides. The caller may reuse rank 0 and record what happened.
  Outcome o;
  o.family_id = "a@1";
  o.verdict = OutcomeVerdict::kAccepted;
  o.candidate_object_id = out[0].object_id;
  o.rank = static_cast<uint32_t>(out[0].rank);
  o.similarity = out[0].similarity;
  o.threshold = 0.9f;
  ASSERT_TRUE(store_->RecordOutcome(o).ok());
  FamilyReport r;
  ASSERT_TRUE(store_->GetFamilyReport("a@1", &r).ok());
  EXPECT_EQ(r.accepted, 1u);
  EXPECT_EQ(r.accepted_similarity_hist[19], 1u);
}

TEST_F(SemanticCandidatesTest, SemanticHitMergesMetadataIntoTheMatchedValue) {
  ASSERT_TRUE(OpenSemantic().ok());
  ASSERT_TRUE(store_->Put("k1", "alpha one", Metadata{{"family", "a"}}).ok());
  ASSERT_TRUE(store_->Put("k2", "alpha two", Metadata{{"tool", "t"}}).ok());  // 0.98 >= 0.95: dedups onto "alpha one"
  uint64_t objects = 0;
  ASSERT_TRUE(store_->CountUniqueValues(&objects).ok());
  EXPECT_EQ(objects, 1u);
  const Metadata expected{{"family", "a"}, {"tool", "t"}};
  EXPECT_EQ(MetaOf("k1"), expected);
  EXPECT_EQ(MetaOf("k2"), expected);

  // Same key, same value, fresh metadata: still merged, still one object
  ASSERT_TRUE(store_->Put("k2", "alpha two", Metadata{{"schema", "s"}}).ok());
  EXPECT_EQ(MetaOf("k1").at("schema"), "s");
  ASSERT_TRUE(store_->CountUniqueValues(&objects).ok());
  EXPECT_EQ(objects, 1u);
}

#endif  // PRESTIGE_ENABLE_SEMANTIC

}  // namespace
}  // namespace prestige
