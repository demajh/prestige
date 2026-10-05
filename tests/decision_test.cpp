// Tests for decision records: provenance committed in the same transaction as the write,
// the read-time guard, the repair queue and the sweep. See docs/provenance.md.

#include <gtest/gtest.h>

#include <prestige/store.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace prestige {
namespace {

class DecisionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() / ("prestige_decision_test_" + RandomSuffix());
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

  static Decision MakeDecision(const std::string& id, const std::string& policy = "policy-v1") {
    Decision d;
    d.decision_id = id;
    d.policy_revision = policy;
    return d;
  }

  HealthStats Health() {
    HealthStats h;
    EXPECT_TRUE(store_->GetHealth(&h).ok());
    return h;
  }

  std::filesystem::path test_dir_;
  std::string db_path_;
  std::unique_ptr<Store> store_;
};

// Counts the sweep's cursor checkpoints. The optional callback runs on each one; a test uses it to stand in
// for a crash.
class CheckpointSink : public MetricsSink {
 public:
  explicit CheckpointSink(std::function<void(uint64_t)> on_checkpoint = {})
      : on_checkpoint_(std::move(on_checkpoint)) {}

  void Counter(std::string_view name, uint64_t delta) override {
    if (name != "prestige.decision.sweep_checkpoint_total") return;
    checkpoints_ += delta;
    if (on_checkpoint_) on_checkpoint_(checkpoints_);
  }
  void Histogram(std::string_view, uint64_t) override {}

  uint64_t checkpoints() const { return checkpoints_; }

 private:
  std::function<void(uint64_t)> on_checkpoint_;
  uint64_t checkpoints_ = 0;
};

TEST_F(DecisionTest, RecordCommitsWithTheWriteAndResolves) {
  ASSERT_TRUE(OpenStore().ok());
  Decision d = MakeDecision("dec-1");
  d.parent_decision_id = "dec-0";
  d.note = "first write";
  ASSERT_TRUE(store_->PutWithDecision("k1", "value one", d).ok());

  std::string v;
  ASSERT_TRUE(store_->Get("k1", &v).ok());
  EXPECT_EQ(v, "value one");

  DecisionRecord rec;
  DecisionCheck check;
  ASSERT_TRUE(store_->GetDecision("dec-1", &rec, &check).ok());
  EXPECT_EQ(rec.decision.decision_id, "dec-1");
  EXPECT_EQ(rec.decision.policy_revision, "policy-v1");
  EXPECT_EQ(rec.decision.parent_decision_id, "dec-0");
  EXPECT_EQ(rec.decision.note, "first write");
  EXPECT_EQ(rec.user_key, "k1");
  EXPECT_EQ(rec.sequence, 1u);
  EXPECT_GT(rec.committed_at_us, 0u);

  std::string digest;
  ASSERT_TRUE(store_->Digest("value one", &digest).ok());
  EXPECT_EQ(digest.size(), 32u);
  EXPECT_EQ(rec.output_digest, digest);

  EXPECT_TRUE(check.resolved);
  EXPECT_TRUE(check.missing_digests.empty());
  EXPECT_FALSE(check.queued_for_repair);
  EXPECT_TRUE(store_->VerifyDecision("dec-1").ok());

  HealthStats h = Health();
  EXPECT_EQ(h.decisions_total, 1u);
  EXPECT_EQ(h.decision_queue_size, 0u);
  EXPECT_EQ(h.decision_dangling_reads, 0u);
}

TEST_F(DecisionTest, UnknownDecisionIsNotFound) {
  ASSERT_TRUE(OpenStore().ok());
  DecisionRecord rec;
  EXPECT_TRUE(store_->GetDecision("missing", &rec).IsNotFound());
  EXPECT_TRUE(store_->VerifyDecision("missing").IsNotFound());
}

TEST_F(DecisionTest, ValidationRejectsIncompleteDecisionsWithoutWriting) {
  ASSERT_TRUE(OpenStore().ok());
  Decision no_id = MakeDecision("");
  EXPECT_TRUE(store_->PutWithDecision("k", "v", no_id).IsInvalidArgument());
  Decision no_policy = MakeDecision("dec", "");
  EXPECT_TRUE(store_->PutWithDecision("k", "v", no_policy).IsInvalidArgument());
  Decision bad_digest = MakeDecision("dec");
  bad_digest.input_digests = {"not-32-bytes"};
  EXPECT_TRUE(store_->PutWithDecision("k", "v", bad_digest).IsInvalidArgument());

  uint64_t keys = 1;
  ASSERT_TRUE(store_->CountKeys(&keys).ok());
  EXPECT_EQ(keys, 0u);
  EXPECT_EQ(Health().decisions_total, 0u);
}

TEST_F(DecisionTest, ReplayIsIdempotentAndReuseForAnotherWriteIsRefused) {
  ASSERT_TRUE(OpenStore().ok());
  Decision d = MakeDecision("dec-1");
  ASSERT_TRUE(store_->PutWithDecision("k1", "v", d).ok());
  // Same decision, same key, same content: a crash-and-retry replay. No-op success.
  ASSERT_TRUE(store_->PutWithDecision("k1", "v", d).ok());

  std::vector<DecisionRecord> all;
  ASSERT_TRUE(store_->ListDecisions(&all).ok());
  ASSERT_EQ(all.size(), 1u);
  EXPECT_EQ(all[0].sequence, 1u);

  // Same id for a different write: refused, and nothing is written.
  EXPECT_TRUE(store_->PutWithDecision("k2", "other", d).IsInvalidArgument());
  std::string v;
  EXPECT_TRUE(store_->Get("k2", &v).IsNotFound());
  EXPECT_EQ(Health().decisions_total, 1u);
}

TEST_F(DecisionTest, SameValueOverwriteStillRecordsTheDecisionAndKeepsRefcountsRight) {
  ASSERT_TRUE(OpenStore().ok());
  ASSERT_TRUE(store_->Put("k1", "v").ok());
  Decision d = MakeDecision("dec-1");
  ASSERT_TRUE(store_->PutWithDecision("k1", "v", d).ok());  // same object, decision recorded
  EXPECT_EQ(Health().decisions_total, 1u);

  // One key still references the object; deleting it must GC the object (refcount was not double-counted).
  ASSERT_TRUE(store_->Delete("k1").ok());
  uint64_t objects = 1;
  ASSERT_TRUE(store_->CountUniqueValues(&objects).ok());
  EXPECT_EQ(objects, 0u);

  // The record outlives the body; the read-time guard now reports the gap.
  DecisionRecord rec;
  DecisionCheck check;
  ASSERT_TRUE(store_->GetDecision("dec-1", &rec, &check).ok());
  EXPECT_FALSE(check.resolved);
  ASSERT_EQ(check.missing_digests.size(), 1u);
  EXPECT_EQ(check.missing_digests[0], rec.output_digest);
}

TEST_F(DecisionTest, MissingInputBodyIsDetectedQueuedAndRepaired) {
  ASSERT_TRUE(OpenStore().ok());
  std::string input_digest;
  ASSERT_TRUE(store_->Digest("input body", &input_digest).ok());
  Decision d = MakeDecision("dec-1");
  d.input_digests = {input_digest};
  // The commitment is written now; the body has not arrived.
  ASSERT_TRUE(store_->PutWithDecision("k1", "derived value", d).ok());

  DecisionRecord rec;
  DecisionCheck check;
  ASSERT_TRUE(store_->GetDecision("dec-1", &rec, &check).ok());
  EXPECT_FALSE(check.resolved);
  ASSERT_EQ(check.missing_digests.size(), 1u);
  EXPECT_EQ(check.missing_digests[0], input_digest);
  EXPECT_TRUE(check.queued_for_repair);

  HealthStats h = Health();
  EXPECT_EQ(h.decision_queue_size, 1u);
  EXPECT_EQ(h.decision_dangling_reads, 1u);

  // A second read counts again but does not queue twice.
  ASSERT_TRUE(store_->GetDecision("dec-1", &rec, &check).ok());
  EXPECT_FALSE(check.queued_for_repair);
  h = Health();
  EXPECT_EQ(h.decision_queue_size, 1u);
  EXPECT_EQ(h.decision_dangling_reads, 2u);

  EXPECT_TRUE(store_->VerifyDecision("dec-1").IsCorruption());

  DecisionSweepStats st;
  ASSERT_TRUE(store_->SweepDecisions(0, &st).ok());
  EXPECT_EQ(st.dangling, 1u);
  EXPECT_EQ(st.repaired, 0u);
  EXPECT_EQ(st.queue_size, 1u);

  // The body arrives under any key; the commitment now resolves and the sweep drains the queue.
  ASSERT_TRUE(store_->Put("some-other-key", "input body").ok());
  ASSERT_TRUE(store_->SweepDecisions(0, &st).ok());
  EXPECT_EQ(st.repaired, 1u);
  EXPECT_EQ(st.queue_size, 0u);
  h = Health();
  EXPECT_EQ(h.decision_queue_size, 0u);
  EXPECT_EQ(h.decision_repairs, 1u);

  ASSERT_TRUE(store_->GetDecision("dec-1", &rec, &check).ok());
  EXPECT_TRUE(check.resolved);
  EXPECT_TRUE(store_->VerifyDecision("dec-1").ok());
}

TEST_F(DecisionTest, SweepWalksUnreadRecordsInWriteOrderAndListsThem) {
  ASSERT_TRUE(OpenStore().ok());
  std::string missing_digest;
  ASSERT_TRUE(store_->Digest("never written", &missing_digest).ok());

  ASSERT_TRUE(store_->PutWithDecision("k1", "v1", MakeDecision("dec-1")).ok());
  Decision d2 = MakeDecision("dec-2");
  d2.input_digests = {missing_digest};
  ASSERT_TRUE(store_->PutWithDecision("k2", "v2", d2).ok());
  ASSERT_TRUE(store_->PutWithDecision("k3", "v3", MakeDecision("dec-3")).ok());

  std::vector<DecisionRecord> all;
  ASSERT_TRUE(store_->ListDecisions(&all).ok());
  ASSERT_EQ(all.size(), 3u);
  EXPECT_EQ(all[0].decision.decision_id, "dec-1");
  EXPECT_EQ(all[1].decision.decision_id, "dec-2");
  EXPECT_EQ(all[2].decision.decision_id, "dec-3");
  EXPECT_EQ(all[2].sequence, 3u);

  std::vector<DecisionRecord> page;
  ASSERT_TRUE(store_->ListDecisions(&page, 1, 1).ok());
  ASSERT_EQ(page.size(), 1u);
  EXPECT_EQ(page[0].decision.decision_id, "dec-2");

  // Nobody has read dec-2. A bounded sweep finds it by walking in write order.
  DecisionSweepStats st;
  ASSERT_TRUE(store_->SweepDecisions(2, &st).ok());
  EXPECT_EQ(st.checked, 2u);
  EXPECT_EQ(st.dangling, 1u);
  EXPECT_EQ(st.queue_size, 1u);
  EXPECT_EQ(st.cursor_sequence, 2u);
  EXPECT_EQ(st.max_sequence, 3u);

  // Next sweep re-checks the queued record first, then finishes the walk.
  ASSERT_TRUE(store_->SweepDecisions(0, &st).ok());
  EXPECT_EQ(st.checked, 2u);
  EXPECT_EQ(st.dangling, 1u);
  EXPECT_EQ(st.queue_size, 1u);
  EXPECT_EQ(st.cursor_sequence, 3u);
  EXPECT_EQ(Health().decision_dangling_reads, 0u);  // sweep finds are not read-time misses
}

TEST_F(DecisionTest, SweepPersistsItsCursorWithQueuedRecordsAndEveryInterval) {
  auto sink = std::make_shared<CheckpointSink>();
  Options opt;
  opt.metrics = sink;
  opt.decision_sweep_cursor_interval = 4;
  ASSERT_TRUE(OpenStore(opt).ok());
  std::string missing_digest;
  ASSERT_TRUE(store_->Digest("never written", &missing_digest).ok());
  for (int i = 1; i <= 10; ++i) {
    Decision d = MakeDecision("dec-" + std::to_string(i));
    if (i == 2 || i == 9) d.input_digests = {missing_digest};
    ASSERT_TRUE(store_->PutWithDecision("k" + std::to_string(i), "v" + std::to_string(i), d).ok());
  }

  // Checkpoints: with dec-2 as it is queued, four records later (cursor 6), with dec-9, and at the end.
  DecisionSweepStats st;
  ASSERT_TRUE(store_->SweepDecisions(0, &st).ok());
  EXPECT_EQ(st.checked, 10u);
  EXPECT_EQ(st.dangling, 2u);
  EXPECT_EQ(st.queue_size, 2u);
  EXPECT_EQ(st.cursor_sequence, 10u);
  EXPECT_EQ(sink->checkpoints(), 4u);

  // Nothing new to walk: the queue is re-checked and the cursor is not rewritten.
  ASSERT_TRUE(store_->SweepDecisions(0, &st).ok());
  EXPECT_EQ(st.checked, 2u);
  EXPECT_EQ(st.dangling, 2u);
  EXPECT_EQ(st.cursor_sequence, 10u);
  EXPECT_EQ(sink->checkpoints(), 4u);
}

TEST_F(DecisionTest, SweepInterruptedAfterACheckpointResumesFromIt) {
  // A sink that throws at the second checkpoint stands in for a crash: the call never reaches its end-of-call
  // cursor write, and whatever it had persisted by then is where the next sweep starts.
  auto sink = std::make_shared<CheckpointSink>([](uint64_t n) {
    if (n == 2) throw std::runtime_error("crash after the second checkpoint");
  });
  Options opt;
  opt.metrics = sink;
  opt.decision_sweep_cursor_interval = 3;
  ASSERT_TRUE(OpenStore(opt).ok());
  for (int i = 1; i <= 10; ++i) {
    const std::string n = std::to_string(i);
    ASSERT_TRUE(store_->PutWithDecision("k" + n, "v" + n, MakeDecision("dec-" + n)).ok());
  }

  DecisionSweepStats st;
  EXPECT_THROW(store_->SweepDecisions(0, &st), std::runtime_error);  // checkpoints after 3 and 6, then "crash"

  // Restart. The walk resumes after record 6 instead of starting over.
  store_.reset();
  Options plain;
  plain.decision_sweep_cursor_interval = 3;
  ASSERT_TRUE(OpenStore(plain).ok());
  ASSERT_TRUE(store_->SweepDecisions(0, &st).ok());
  EXPECT_EQ(st.checked, 4u);
  EXPECT_EQ(st.cursor_sequence, 10u);
  EXPECT_EQ(st.max_sequence, 10u);
}

TEST_F(DecisionTest, RecordsSurviveReopenAndSequencesContinue) {
  ASSERT_TRUE(OpenStore().ok());
  ASSERT_TRUE(store_->PutWithDecision("k1", "v1", MakeDecision("dec-1")).ok());
  ASSERT_TRUE(store_->PutWithDecision("k2", "v2", MakeDecision("dec-2")).ok());
  store_.reset();

  ASSERT_TRUE(OpenStore().ok());
  DecisionRecord rec;
  ASSERT_TRUE(store_->GetDecision("dec-2", &rec).ok());
  EXPECT_EQ(rec.sequence, 2u);
  ASSERT_TRUE(store_->PutWithDecision("k3", "v3", MakeDecision("dec-3")).ok());
  ASSERT_TRUE(store_->GetDecision("dec-3", &rec).ok());
  EXPECT_EQ(rec.sequence, 3u);
  EXPECT_EQ(Health().decisions_total, 3u);
}

TEST_F(DecisionTest, DigestFollowsTheStoresNormalization) {
  Options opt;
  opt.normalization_mode = NormalizationMode::kASCII;
  ASSERT_TRUE(OpenStore(opt).ok());
  std::string a, b;
  ASSERT_TRUE(store_->Digest("Hello   World", &a).ok());
  ASSERT_TRUE(store_->Digest("hello world", &b).ok());
  EXPECT_EQ(a, b);
  ASSERT_TRUE(store_->PutWithDecision("k", "Hello   World", MakeDecision("dec-1")).ok());
  DecisionRecord rec;
  ASSERT_TRUE(store_->GetDecision("dec-1", &rec).ok());
  EXPECT_EQ(rec.output_digest, a);
}

}  // namespace
}  // namespace prestige
