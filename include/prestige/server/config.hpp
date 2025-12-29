#pragma once

#include <prestige/store.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace prestige::server {

/**
 * Server configuration.
 */
struct ServerConfig {
  std::string host = "0.0.0.0";
  uint16_t port = 8080;
  uint32_t threads = 0;  // 0 = auto-detect CPU cores
  std::string log_level = "info";
};

/**
 * Proxy mode configuration.
 */
struct ProxyConfig {
  bool enabled = false;
  std::string upstream_base_url;
  uint32_t timeout_ms = 30000;
  bool forward_host_header = false;
  std::vector<std::string> strip_request_headers;
  std::vector<std::string> strip_response_headers;
  std::vector<std::string> vary_headers = {"Accept", "Accept-Language"};
  bool cache_post = false;
  std::vector<int> cacheable_status_codes = {200, 201, 204, 301, 404};
};

/**
 * Metrics configuration.
 */
struct MetricsConfig {
  bool enabled = true;
  std::string path = "/metrics";
};

/**
 * Complete server configuration.
 */
struct Config {
  ServerConfig server;
  std::string db_path;
  prestige::Options store;
  ProxyConfig proxy;
  MetricsConfig metrics;

  /**
   * Load configuration from a YAML file.
   * @throws std::runtime_error if file cannot be read or parsed.
   */
  static Config LoadFromFile(const std::string& path);

  /**
   * Parse configuration from command-line arguments.
   * @param argc Argument count
   * @param argv Argument values
   * @return Parsed configuration
   * @throws std::runtime_error on invalid arguments.
   */
  static Config LoadFromArgs(int argc, char** argv);

  /**
   * Validate configuration.
   * @throws std::runtime_error if configuration is invalid.
   */
  void Validate() const;
};

}  // namespace prestige::server
