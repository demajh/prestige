#include <prestige/server/metrics.hpp>

#include <drogon/drogon.h>

#include <iomanip>
#include <sstream>

namespace prestige::server {

namespace {

// Histogram buckets for latency (in milliseconds)
const std::vector<double> kLatencyBuckets = {
    0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000};

size_t FindBucket(double value, const std::vector<double>& buckets) {
  for (size_t i = 0; i < buckets.size(); ++i) {
    if (value <= buckets[i]) {
      return i;
    }
  }
  return buckets.size();  // +Inf bucket
}

}  // namespace

// --- PrometheusMetrics ---

void PrometheusMetrics::Counter(std::string_view name, uint64_t delta) {
  std::lock_guard<std::mutex> lock(mu_);
  counters_[std::string(name)] += delta;
}

void PrometheusMetrics::Histogram(std::string_view name, uint64_t value) {
  std::lock_guard<std::mutex> lock(mu_);
  auto& h = histograms_[std::string(name)];
  if (h.buckets.empty()) {
    h.buckets.resize(kLatencyBuckets.size() + 1, 0);
  }
  size_t bucket = FindBucket(static_cast<double>(value), kLatencyBuckets);
  for (size_t i = bucket; i < h.buckets.size(); ++i) {
    h.buckets[i]++;
  }
  h.count++;
  h.sum += static_cast<double>(value);
}

void PrometheusMetrics::Gauge(std::string_view name, double value) {
  std::lock_guard<std::mutex> lock(mu_);
  gauges_[std::string(name)] = value;
}

void PrometheusMetrics::RecordHttpRequest(const std::string& method,
                                          const std::string& path,
                                          int status_code,
                                          double latency_ms) {
  std::lock_guard<std::mutex> lock(mu_);

  // Increment request counter with labels
  HttpMetricKey key{method, path, status_code};
  http_requests_[key]++;

  // Record latency histogram
  if (http_latency_.buckets.empty()) {
    http_latency_.buckets.resize(kLatencyBuckets.size() + 1, 0);
  }
  size_t bucket = FindBucket(latency_ms, kLatencyBuckets);
  for (size_t i = bucket; i < http_latency_.buckets.size(); ++i) {
    http_latency_.buckets[i]++;
  }
  http_latency_.count++;
  http_latency_.sum += latency_ms;
}

void PrometheusMetrics::RecordProxyCacheHit() {
  std::lock_guard<std::mutex> lock(mu_);
  counters_["prestige_proxy_cache_hits_total"]++;
}

void PrometheusMetrics::RecordProxyCacheMiss() {
  std::lock_guard<std::mutex> lock(mu_);
  counters_["prestige_proxy_cache_misses_total"]++;
}

void PrometheusMetrics::RecordProxyUpstreamLatency(double latency_ms) {
  std::lock_guard<std::mutex> lock(mu_);
  auto& h = histograms_["prestige_proxy_upstream_latency_ms"];
  if (h.buckets.empty()) {
    h.buckets.resize(kLatencyBuckets.size() + 1, 0);
  }
  size_t bucket = FindBucket(latency_ms, kLatencyBuckets);
  for (size_t i = bucket; i < h.buckets.size(); ++i) {
    h.buckets[i]++;
  }
  h.count++;
  h.sum += latency_ms;
}

std::string PrometheusMetrics::Export() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);

  // Export counters
  for (const auto& [name, value] : counters_) {
    out << "# TYPE " << name << " counter\n";
    out << name << " " << value << "\n";
  }

  // Export gauges
  for (const auto& [name, value] : gauges_) {
    out << "# TYPE " << name << " gauge\n";
    out << name << " " << value << "\n";
  }

  // Export histograms
  for (const auto& [name, data] : histograms_) {
    out << "# TYPE " << name << " histogram\n";
    for (size_t i = 0; i < kLatencyBuckets.size(); ++i) {
      out << name << "_bucket{le=\"" << kLatencyBuckets[i] << "\"} "
          << data.buckets[i] << "\n";
    }
    out << name << "_bucket{le=\"+Inf\"} " << data.buckets.back() << "\n";
    out << name << "_sum " << data.sum << "\n";
    out << name << "_count " << data.count << "\n";
  }

  // Export HTTP request counters with labels
  if (!http_requests_.empty()) {
    out << "# TYPE prestige_http_requests_total counter\n";
    for (const auto& [key, count] : http_requests_) {
      out << "prestige_http_requests_total{method=\"" << key.method
          << "\",path=\"" << key.path << "\",status=\"" << key.status_code
          << "\"} " << count << "\n";
    }
  }

  // Export HTTP latency histogram
  if (http_latency_.count > 0) {
    out << "# TYPE prestige_http_request_duration_ms histogram\n";
    for (size_t i = 0; i < kLatencyBuckets.size(); ++i) {
      out << "prestige_http_request_duration_ms_bucket{le=\""
          << kLatencyBuckets[i] << "\"} " << http_latency_.buckets[i] << "\n";
    }
    out << "prestige_http_request_duration_ms_bucket{le=\"+Inf\"} "
        << http_latency_.buckets.back() << "\n";
    out << "prestige_http_request_duration_ms_sum " << http_latency_.sum << "\n";
    out << "prestige_http_request_duration_ms_count " << http_latency_.count << "\n";
  }

  return out.str();
}

// --- Metrics Handler Registration ---

void RegisterMetricsHandler(std::shared_ptr<PrometheusMetrics> metrics,
                            Store* store) {
  drogon::app().registerHandler(
      "/metrics",
      [metrics, store](const drogon::HttpRequestPtr& req,
                       std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        // Collect store metrics
        if (store) {
          store->EmitCacheMetrics();

          // Add store stats as gauges
          uint64_t keys = 0, objects = 0, bytes = 0;
          if (store->CountKeysApprox(&keys).ok()) {
            metrics->Gauge("prestige_keys_total", static_cast<double>(keys));
          }
          if (store->CountUniqueValuesApprox(&objects).ok()) {
            metrics->Gauge("prestige_objects_total", static_cast<double>(objects));
          }
          if (store->GetTotalStoreBytesApprox(&bytes).ok()) {
            metrics->Gauge("prestige_store_bytes", static_cast<double>(bytes));
          }
        }

        // Export metrics
        std::string output = metrics->Export();

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setBody(output);
        resp->setContentTypeString("text/plain; version=0.0.4; charset=utf-8");
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Get});
}

// --- RequestTimer ---

RequestTimer::RequestTimer(std::shared_ptr<PrometheusMetrics> metrics,
                           std::string method,
                           std::string path)
    : metrics_(std::move(metrics)),
      method_(std::move(method)),
      path_(std::move(path)),
      start_(std::chrono::steady_clock::now()) {}

RequestTimer::~RequestTimer() {
  if (metrics_) {
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
    double latency_ms = static_cast<double>(duration.count()) / 1000.0;
    metrics_->RecordHttpRequest(method_, path_, status_code_, latency_ms);
  }
}

}  // namespace prestige::server
