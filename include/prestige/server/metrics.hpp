#pragma once

#include <prestige/store.hpp>

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace prestige::server {

/**
 * Prometheus-compatible metrics sink.
 *
 * Collects counters, histograms, and gauges and exports them
 * in Prometheus text exposition format.
 */
class PrometheusMetrics : public prestige::MetricsSink {
 public:
  PrometheusMetrics() = default;

  // MetricsSink interface
  void Counter(std::string_view name, uint64_t delta) override;
  void Histogram(std::string_view name, uint64_t value) override;
  void Gauge(std::string_view name, double value) override;

  /**
   * Generate Prometheus text format output.
   */
  std::string Export() const;

  /**
   * Record an HTTP request metric.
   */
  void RecordHttpRequest(const std::string& method,
                         const std::string& path,
                         int status_code,
                         double latency_ms);

  /**
   * Record a proxy cache hit.
   */
  void RecordProxyCacheHit();

  /**
   * Record a proxy cache miss.
   */
  void RecordProxyCacheMiss();

  /**
   * Record upstream request latency.
   */
  void RecordProxyUpstreamLatency(double latency_ms);

 private:
  mutable std::mutex mu_;

  // Counters: name -> value
  std::unordered_map<std::string, uint64_t> counters_;

  // Histograms: name -> bucket values (using predefined buckets)
  struct HistogramData {
    std::vector<uint64_t> buckets;
    uint64_t count = 0;
    double sum = 0.0;
  };
  std::unordered_map<std::string, HistogramData> histograms_;

  // Gauges: name -> value
  std::unordered_map<std::string, double> gauges_;

  // HTTP request metrics with labels
  struct HttpMetricKey {
    std::string method;
    std::string path;
    int status_code;

    bool operator==(const HttpMetricKey& other) const {
      return method == other.method && path == other.path &&
             status_code == other.status_code;
    }
  };

  struct HttpMetricKeyHash {
    size_t operator()(const HttpMetricKey& k) const {
      return std::hash<std::string>{}(k.method) ^
             (std::hash<std::string>{}(k.path) << 1) ^
             (std::hash<int>{}(k.status_code) << 2);
    }
  };

  std::unordered_map<HttpMetricKey, uint64_t, HttpMetricKeyHash> http_requests_;
  HistogramData http_latency_;
};

/**
 * Register the /metrics endpoint with the Drogon app.
 */
void RegisterMetricsHandler(std::shared_ptr<PrometheusMetrics> metrics,
                            Store* store);

/**
 * RAII helper for timing HTTP requests.
 */
class RequestTimer {
 public:
  RequestTimer(std::shared_ptr<PrometheusMetrics> metrics,
               std::string method,
               std::string path);

  ~RequestTimer();

  void SetStatusCode(int code) { status_code_ = code; }

 private:
  std::shared_ptr<PrometheusMetrics> metrics_;
  std::string method_;
  std::string path_;
  int status_code_ = 200;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace prestige::server
