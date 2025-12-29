#pragma once

#include <prestige/server/config.hpp>
#include <prestige/server/metrics.hpp>
#include <prestige/store.hpp>

#include <drogon/drogon.h>

#include <string>
#include <vector>

namespace prestige::server {

/**
 * Cache key for proxy requests.
 */
struct ProxyCacheKey {
  std::string method;
  std::string path;
  std::string query;
  std::string body_hash;
  std::string vary_headers;

  /**
   * Serialize to a string for use as prestige key.
   */
  std::string Serialize() const;

  /**
   * Create cache key from an HTTP request.
   */
  static ProxyCacheKey FromRequest(const drogon::HttpRequestPtr& req,
                                   const ProxyConfig& config);
};

/**
 * Cached response structure.
 */
struct CachedResponse {
  int status_code = 0;
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;
  uint64_t cached_at_ms = 0;
  uint64_t upstream_latency_ms = 0;

  /**
   * Serialize to bytes for storage in prestige.
   */
  std::string Serialize() const;

  /**
   * Deserialize from stored bytes.
   */
  static CachedResponse Deserialize(std::string_view data);

  /**
   * Check if deserialization succeeded.
   */
  bool IsValid() const { return status_code != 0; }
};

/**
 * Register proxy handler if proxy mode is enabled.
 *
 * All requests to /proxy/... are forwarded to the configured upstream
 * and responses are cached in prestige with deduplication.
 */
void RegisterProxyHandler(Store* store,
                          const ProxyConfig& config,
                          std::shared_ptr<PrometheusMetrics> metrics);

}  // namespace prestige::server
