#include <prestige/store.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// A tiny in-process metrics collector for demo purposes.
//
// In real usage, you'd adapt prestige::MetricsSink to your system's metrics
// library (Prometheus, OpenTelemetry metrics, StatsD, etc.).
class SimpleMetrics final : public prestige::MetricsSink {
 public:
  void Counter(std::string_view name, uint64_t delta) override {
    std::lock_guard<std::mutex> lk(mu_);
    counters_[std::string(name)] += delta;
  }

  void Histogram(std::string_view name, uint64_t value) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto& h = hists_[std::string(name)];
    h.count += 1;
    h.sum += value;
    if (value < h.min) h.min = value;
    if (value > h.max) h.max = value;
  }

  void Dump(std::ostream& os) const {
    std::lock_guard<std::mutex> lk(mu_);
    os << "\n== Counters ==\n";
    for (const auto& kv : counters_) {
      os << kv.first << " = " << kv.second << "\n";
    }
    os << "\n== Histograms (count, min, avg, max) ==\n";
    for (const auto& kv : hists_) {
      const auto& h = kv.second;
      const double avg = h.count ? static_cast<double>(h.sum) / static_cast<double>(h.count) : 0.0;
      os << kv.first
         << " count=" << h.count
         << " min=" << (h.count ? h.min : 0)
         << " avg=" << avg
         << " max=" << (h.count ? h.max : 0)
         << "\n";
    }
  }

 private:
  struct HistAgg {
    uint64_t count = 0;
    uint64_t sum = 0;
    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
  };

  mutable std::mutex mu_;
  std::unordered_map<std::string, uint64_t> counters_;
  std::unordered_map<std::string, HistAgg> hists_;
};

// A minimal tracer that prints each span when it ends.
class StdoutSpan final : public prestige::TraceSpan {
 public:
  explicit StdoutSpan(std::string_view name)
      : name_(name), start_(std::chrono::steady_clock::now()) {}

  void SetAttribute(std::string_view key, uint64_t value) override {
    attrs_.emplace_back(std::string(key), std::to_string(value));
  }

  void SetAttribute(std::string_view key, std::string_view value) override {
    attrs_.emplace_back(std::string(key), std::string(value));
  }

  void AddEvent(std::string_view name) override { events_.push_back(std::string(name)); }

  void End(const rocksdb::Status& status) override {
    using namespace std::chrono;
    const auto end = steady_clock::now();
    const auto dur_us = duration_cast<microseconds>(end - start_).count();

    std::cout << "\n[span] " << name_ << " status=" << status.ToString() << " dur_us=" << dur_us << "\n";
    if (!attrs_.empty()) {
      std::cout << "  attributes:\n";
      for (const auto& kv : attrs_) {
        std::cout << "    " << kv.first << " = " << kv.second << "\n";
      }
    }
    if (!events_.empty()) {
      std::cout << "  events:\n";
      for (const auto& e : events_) {
        std::cout << "    " << e << "\n";
      }
    }
  }

 private:
  std::string name_;
  std::chrono::steady_clock::time_point start_;
  std::vector<std::pair<std::string, std::string>> attrs_;
  std::vector<std::string> events_;
};

class StdoutTracer final : public prestige::Tracer {
 public:
  std::unique_ptr<prestige::TraceSpan> StartSpan(std::string_view name) override {
    return std::make_unique<StdoutSpan>(name);
  }
};

}  // namespace

int main() {
  auto metrics = std::make_shared<SimpleMetrics>();
  auto tracer = std::make_shared<StdoutTracer>();

  prestige::Options opt;
  opt.metrics = metrics;
  opt.tracer = tracer;

  std::unique_ptr<prestige::Store> db;
  auto s = prestige::Store::Open("./prestige_db", &db, opt);
  if (!s.ok()) {
    std::cerr << "Open failed: " << s.ToString() << "\n";
    return 1;
  }

  // A few ops to generate signals.
  (void)db->Put("k1", "HELLO");
  (void)db->Put("k2", "HELLO");  // should be a dedup hit

  std::string v;
  (void)db->Get("k2", &v);

  (void)db->Put("k2", "WORLD");
  (void)db->Delete("k1");
  (void)db->Delete("k2");

  metrics->Dump(std::cout);
  return 0;
}
