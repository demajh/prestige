// Server tests for prestige HTTP server
// Tests: Config parsing, metrics, and HTTP handlers

#include <gtest/gtest.h>

#include <prestige/server/config.hpp>
#include <prestige/server/metrics.hpp>
#include <prestige/server/handlers.hpp>
#include <prestige/server/proxy.hpp>
#include <prestige/store.hpp>

#include <drogon/drogon.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include <chrono>

namespace prestige::server {
namespace {

// =============================================================================
// Test Utilities
// =============================================================================

std::string RandomSuffix() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 999999);
  return std::to_string(dis(gen));
}

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("prestige_server_test_" + RandomSuffix());
    std::filesystem::create_directories(path_);
  }

  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  std::filesystem::path path() const { return path_; }
  std::string string() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

// =============================================================================
// Config Tests
// =============================================================================

class ConfigTest : public ::testing::Test {
 protected:
  TempDir temp_dir_;
};

TEST_F(ConfigTest, DefaultValues) {
  Config config;
  EXPECT_EQ(config.server.host, "0.0.0.0");
  EXPECT_EQ(config.server.port, 8080);
  EXPECT_EQ(config.server.threads, 0);
  EXPECT_EQ(config.server.log_level, "info");
  EXPECT_TRUE(config.db_path.empty());
  EXPECT_FALSE(config.proxy.enabled);
  EXPECT_TRUE(config.metrics.enabled);
}

TEST_F(ConfigTest, LoadFromArgs_DbPath) {
  const char* argv[] = {"prestige-server", "--db-path", "/data/test"};
  auto config = Config::LoadFromArgs(3, const_cast<char**>(argv));
  EXPECT_EQ(config.db_path, "/data/test");
}

TEST_F(ConfigTest, LoadFromArgs_Port) {
  const char* argv[] = {"prestige-server", "--db-path", "/data", "--port", "9090"};
  auto config = Config::LoadFromArgs(5, const_cast<char**>(argv));
  EXPECT_EQ(config.server.port, 9090);
}

TEST_F(ConfigTest, LoadFromArgs_ShortPort) {
  const char* argv[] = {"prestige-server", "--db-path", "/data", "-p", "9091"};
  auto config = Config::LoadFromArgs(5, const_cast<char**>(argv));
  EXPECT_EQ(config.server.port, 9091);
}

TEST_F(ConfigTest, LoadFromArgs_Host) {
  const char* argv[] = {"prestige-server", "--db-path", "/data", "--host", "127.0.0.1"};
  auto config = Config::LoadFromArgs(5, const_cast<char**>(argv));
  EXPECT_EQ(config.server.host, "127.0.0.1");
}

TEST_F(ConfigTest, LoadFromArgs_Threads) {
  const char* argv[] = {"prestige-server", "--db-path", "/data", "--threads", "4"};
  auto config = Config::LoadFromArgs(5, const_cast<char**>(argv));
  EXPECT_EQ(config.server.threads, 4);
}

TEST_F(ConfigTest, LoadFromArgs_LogLevel) {
  const char* argv[] = {"prestige-server", "--db-path", "/data", "--log-level", "debug"};
  auto config = Config::LoadFromArgs(5, const_cast<char**>(argv));
  EXPECT_EQ(config.server.log_level, "debug");
}

TEST_F(ConfigTest, LoadFromArgs_ProxyUpstream) {
  const char* argv[] = {"prestige-server", "--db-path", "/data",
                        "--proxy-upstream", "https://api.example.com"};
  auto config = Config::LoadFromArgs(5, const_cast<char**>(argv));
  EXPECT_TRUE(config.proxy.enabled);
  EXPECT_EQ(config.proxy.upstream_base_url, "https://api.example.com");
}

TEST_F(ConfigTest, LoadFromArgs_UnknownOption) {
  const char* argv[] = {"prestige-server", "--unknown-option"};
  EXPECT_THROW(Config::LoadFromArgs(2, const_cast<char**>(argv)), std::runtime_error);
}

TEST_F(ConfigTest, LoadFromArgs_MissingDbPathArg) {
  const char* argv[] = {"prestige-server", "--db-path"};
  EXPECT_THROW(Config::LoadFromArgs(2, const_cast<char**>(argv)), std::runtime_error);
}

TEST_F(ConfigTest, LoadFromFile_Basic) {
  auto config_path = temp_dir_.path() / "config.yaml";
  std::ofstream out(config_path);
  out << "server:\n"
      << "  host: \"127.0.0.1\"\n"
      << "  port: 9000\n"
      << "  threads: 8\n"
      << "store:\n"
      << "  path: \"/var/lib/prestige\"\n"
      << "  default_ttl_seconds: 7200\n";
  out.close();

  auto config = Config::LoadFromFile(config_path.string());
  EXPECT_EQ(config.server.host, "127.0.0.1");
  EXPECT_EQ(config.server.port, 9000);
  EXPECT_EQ(config.server.threads, 8);
  EXPECT_EQ(config.db_path, "/var/lib/prestige");
  EXPECT_EQ(config.store.default_ttl_seconds, 7200);
}

TEST_F(ConfigTest, LoadFromFile_Proxy) {
  auto config_path = temp_dir_.path() / "config.yaml";
  std::ofstream out(config_path);
  out << "store:\n"
      << "  path: \"/data\"\n"
      << "proxy:\n"
      << "  enabled: true\n"
      << "  upstream_base_url: \"https://backend.example.com\"\n"
      << "  timeout_ms: 5000\n";
  out.close();

  auto config = Config::LoadFromFile(config_path.string());
  EXPECT_TRUE(config.proxy.enabled);
  EXPECT_EQ(config.proxy.upstream_base_url, "https://backend.example.com");
  EXPECT_EQ(config.proxy.timeout_ms, 5000);
}

TEST_F(ConfigTest, LoadFromFile_Metrics) {
  auto config_path = temp_dir_.path() / "config.yaml";
  std::ofstream out(config_path);
  out << "store:\n"
      << "  path: \"/data\"\n"
      << "metrics:\n"
      << "  enabled: false\n"
      << "  path: \"/custom-metrics\"\n";
  out.close();

  auto config = Config::LoadFromFile(config_path.string());
  EXPECT_FALSE(config.metrics.enabled);
  EXPECT_EQ(config.metrics.path, "/custom-metrics");
}

TEST_F(ConfigTest, LoadFromFile_NonExistent) {
  EXPECT_THROW(Config::LoadFromFile("/nonexistent/path/config.yaml"),
               std::runtime_error);
}

TEST_F(ConfigTest, Validate_MissingDbPath) {
  Config config;
  // db_path is empty
  EXPECT_THROW(config.Validate(), std::runtime_error);
}

TEST_F(ConfigTest, Validate_InvalidPort) {
  Config config;
  config.db_path = "/data";
  config.server.port = 0;
  EXPECT_THROW(config.Validate(), std::runtime_error);
}

TEST_F(ConfigTest, Validate_InvalidLogLevel) {
  Config config;
  config.db_path = "/data";
  config.server.log_level = "invalid";
  EXPECT_THROW(config.Validate(), std::runtime_error);
}

TEST_F(ConfigTest, Validate_ProxyWithoutUpstream) {
  Config config;
  config.db_path = "/data";
  config.proxy.enabled = true;
  // upstream_base_url is empty
  EXPECT_THROW(config.Validate(), std::runtime_error);
}

TEST_F(ConfigTest, Validate_Valid) {
  Config config;
  config.db_path = "/data";
  EXPECT_NO_THROW(config.Validate());
}

TEST_F(ConfigTest, Validate_ValidWithProxy) {
  Config config;
  config.db_path = "/data";
  config.proxy.enabled = true;
  config.proxy.upstream_base_url = "https://api.example.com";
  EXPECT_NO_THROW(config.Validate());
}

// =============================================================================
// PrometheusMetrics Tests
// =============================================================================

class MetricsTest : public ::testing::Test {
 protected:
  PrometheusMetrics metrics_;
};

TEST_F(MetricsTest, Counter_Increments) {
  metrics_.Counter("test_counter", 1);
  metrics_.Counter("test_counter", 5);

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("test_counter 6"), std::string::npos);
}

TEST_F(MetricsTest, Counter_MultipleCounters) {
  metrics_.Counter("counter_a", 10);
  metrics_.Counter("counter_b", 20);

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("counter_a 10"), std::string::npos);
  EXPECT_NE(output.find("counter_b 20"), std::string::npos);
}

TEST_F(MetricsTest, Gauge_SetValue) {
  metrics_.Gauge("test_gauge", 42.5);

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("test_gauge 42.5"), std::string::npos);
}

TEST_F(MetricsTest, Gauge_Overwrite) {
  metrics_.Gauge("test_gauge", 10.0);
  metrics_.Gauge("test_gauge", 20.0);

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("test_gauge 20"), std::string::npos);
  EXPECT_EQ(output.find("test_gauge 10"), std::string::npos);
}

TEST_F(MetricsTest, Histogram_RecordsValues) {
  metrics_.Histogram("test_histogram", 5);
  metrics_.Histogram("test_histogram", 50);
  metrics_.Histogram("test_histogram", 500);

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("test_histogram_count 3"), std::string::npos);
  EXPECT_NE(output.find("test_histogram_sum 555"), std::string::npos);
  EXPECT_NE(output.find("test_histogram_bucket"), std::string::npos);
}

TEST_F(MetricsTest, RecordHttpRequest_CountsRequests) {
  metrics_.RecordHttpRequest("GET", "/api/v1/kv/key1", 200, 1.5);
  metrics_.RecordHttpRequest("GET", "/api/v1/kv/key2", 200, 2.0);
  metrics_.RecordHttpRequest("POST", "/api/v1/kv/key1", 201, 3.0);
  metrics_.RecordHttpRequest("GET", "/api/v1/kv/key1", 404, 0.5);

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("prestige_http_requests_total"), std::string::npos);
  EXPECT_NE(output.find("prestige_http_request_duration_ms"), std::string::npos);
}

TEST_F(MetricsTest, RecordProxyCacheHit) {
  metrics_.RecordProxyCacheHit();
  metrics_.RecordProxyCacheHit();
  metrics_.RecordProxyCacheHit();

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("prestige_proxy_cache_hits_total 3"), std::string::npos);
}

TEST_F(MetricsTest, RecordProxyCacheMiss) {
  metrics_.RecordProxyCacheMiss();
  metrics_.RecordProxyCacheMiss();

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("prestige_proxy_cache_misses_total 2"), std::string::npos);
}

TEST_F(MetricsTest, RecordProxyUpstreamLatency) {
  metrics_.RecordProxyUpstreamLatency(10.0);
  metrics_.RecordProxyUpstreamLatency(100.0);
  metrics_.RecordProxyUpstreamLatency(1000.0);

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("prestige_proxy_upstream_latency_ms_count 3"),
            std::string::npos);
}

TEST_F(MetricsTest, Export_IncludesTypeAnnotations) {
  metrics_.Counter("my_counter", 1);
  metrics_.Gauge("my_gauge", 1.0);
  metrics_.Histogram("my_histogram", 1);

  std::string output = metrics_.Export();
  EXPECT_NE(output.find("# TYPE my_counter counter"), std::string::npos);
  EXPECT_NE(output.find("# TYPE my_gauge gauge"), std::string::npos);
  EXPECT_NE(output.find("# TYPE my_histogram histogram"), std::string::npos);
}

// =============================================================================
// RequestTimer Tests
// =============================================================================

TEST(RequestTimerTest, RecordsLatency) {
  auto metrics = std::make_shared<PrometheusMetrics>();

  {
    RequestTimer timer(metrics, "GET", "/api/v1/test");
    timer.SetStatusCode(200);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::string output = metrics->Export();
  EXPECT_NE(output.find("prestige_http_requests_total"), std::string::npos);
  EXPECT_NE(output.find("prestige_http_request_duration_ms"), std::string::npos);
}

// =============================================================================
// ProxyCacheKey Tests
// =============================================================================

class ProxyCacheKeyTest : public ::testing::Test {
 protected:
  ProxyConfig config_;

  void SetUp() override {
    config_.vary_headers = {"Accept", "Accept-Language"};
  }
};

TEST_F(ProxyCacheKeyTest, Serialize_Basic) {
  ProxyCacheKey key;
  key.method = "GET";
  key.path = "/users/123";
  key.query = "";
  key.body_hash = "";
  key.vary_headers = "";

  std::string serialized = key.Serialize();
  EXPECT_EQ(serialized, "proxy:GET:/users/123");
}

TEST_F(ProxyCacheKeyTest, Serialize_WithQuery) {
  ProxyCacheKey key;
  key.method = "GET";
  key.path = "/search";
  key.query = "q=hello&limit=10";
  key.body_hash = "";
  key.vary_headers = "";

  std::string serialized = key.Serialize();
  EXPECT_EQ(serialized, "proxy:GET:/search?q=hello&limit=10");
}

TEST_F(ProxyCacheKeyTest, Serialize_WithBodyHash) {
  ProxyCacheKey key;
  key.method = "POST";
  key.path = "/data";
  key.query = "";
  key.body_hash = "abc123";
  key.vary_headers = "";

  std::string serialized = key.Serialize();
  EXPECT_NE(serialized.find(":body=abc123"), std::string::npos);
}

TEST_F(ProxyCacheKeyTest, Serialize_WithVaryHeaders) {
  ProxyCacheKey key;
  key.method = "GET";
  key.path = "/data";
  key.query = "";
  key.body_hash = "";
  key.vary_headers = "Accept=application/json;";

  std::string serialized = key.Serialize();
  EXPECT_NE(serialized.find(":vary=Accept=application/json;"), std::string::npos);
}

// =============================================================================
// CachedResponse Tests
// =============================================================================

TEST(CachedResponseTest, SerializeDeserialize_Basic) {
  CachedResponse original;
  original.status_code = 200;
  original.headers = {{"Content-Type", "application/json"}};
  original.body = R"({"key": "value"})";
  original.cached_at_ms = 1234567890;
  original.upstream_latency_ms = 42;

  std::string serialized = original.Serialize();
  CachedResponse restored = CachedResponse::Deserialize(serialized);

  EXPECT_TRUE(restored.IsValid());
  EXPECT_EQ(restored.status_code, 200);
  EXPECT_EQ(restored.headers.size(), 1);
  EXPECT_EQ(restored.headers[0].first, "Content-Type");
  EXPECT_EQ(restored.headers[0].second, "application/json");
  EXPECT_EQ(restored.body, R"({"key": "value"})");
  EXPECT_EQ(restored.cached_at_ms, 1234567890);
  EXPECT_EQ(restored.upstream_latency_ms, 42);
}

TEST(CachedResponseTest, SerializeDeserialize_BinaryBody) {
  CachedResponse original;
  original.status_code = 200;
  original.body = std::string("\x00\x01\x02\x03\xff\xfe", 6);
  original.cached_at_ms = 1000;
  original.upstream_latency_ms = 10;

  std::string serialized = original.Serialize();
  CachedResponse restored = CachedResponse::Deserialize(serialized);

  EXPECT_TRUE(restored.IsValid());
  EXPECT_EQ(restored.body.size(), 6);
  EXPECT_EQ(restored.body, original.body);
}

TEST(CachedResponseTest, SerializeDeserialize_MultipleHeaders) {
  CachedResponse original;
  original.status_code = 201;
  original.headers = {
      {"Content-Type", "text/plain"},
      {"X-Custom-Header", "custom-value"},
      {"Cache-Control", "max-age=3600"}};
  original.body = "Hello, World!";
  original.cached_at_ms = 2000;
  original.upstream_latency_ms = 5;

  std::string serialized = original.Serialize();
  CachedResponse restored = CachedResponse::Deserialize(serialized);

  EXPECT_TRUE(restored.IsValid());
  EXPECT_EQ(restored.status_code, 201);
  EXPECT_EQ(restored.headers.size(), 3);
}

TEST(CachedResponseTest, Deserialize_Invalid) {
  CachedResponse restored = CachedResponse::Deserialize("not valid json");
  EXPECT_FALSE(restored.IsValid());
  EXPECT_EQ(restored.status_code, 0);
}

TEST(CachedResponseTest, Deserialize_Empty) {
  CachedResponse restored = CachedResponse::Deserialize("");
  EXPECT_FALSE(restored.IsValid());
}

// =============================================================================
// Handler Integration Tests (require Drogon test server)
// =============================================================================

class HandlerIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    temp_dir_ = std::make_unique<TempDir>();
    db_path_ = (temp_dir_->path() / "test_db").string();

    // Open store
    auto status = Store::Open(db_path_, &store_);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override {
    store_.reset();
    temp_dir_.reset();
  }

  std::unique_ptr<TempDir> temp_dir_;
  std::string db_path_;
  std::unique_ptr<Store> store_;
};

TEST_F(HandlerIntegrationTest, MakeErrorResponse_NotFound) {
  auto status = rocksdb::Status::NotFound("key not found");
  auto resp = MakeErrorResponse(status, "Get failed");

  EXPECT_EQ(resp->statusCode(), drogon::k404NotFound);

  auto json = resp->getJsonObject();
  ASSERT_TRUE(json != nullptr);
  EXPECT_EQ((*json)["error"].asString(), "not_found");
  EXPECT_EQ((*json)["code"].asInt(), 404);
}

TEST_F(HandlerIntegrationTest, MakeErrorResponse_InvalidArgument) {
  auto status = rocksdb::Status::InvalidArgument("bad input");
  auto resp = MakeErrorResponse(status, "Validation failed");

  EXPECT_EQ(resp->statusCode(), drogon::k400BadRequest);

  auto json = resp->getJsonObject();
  ASSERT_TRUE(json != nullptr);
  EXPECT_EQ((*json)["error"].asString(), "invalid_argument");
  EXPECT_EQ((*json)["code"].asInt(), 400);
}

TEST_F(HandlerIntegrationTest, MakeErrorResponse_Timeout) {
  auto status = rocksdb::Status::TimedOut("operation timed out");
  auto resp = MakeErrorResponse(status, "Timeout");

  EXPECT_EQ(resp->statusCode(), drogon::k504GatewayTimeout);

  auto json = resp->getJsonObject();
  ASSERT_TRUE(json != nullptr);
  EXPECT_EQ((*json)["error"].asString(), "timeout");
  EXPECT_EQ((*json)["code"].asInt(), 504);
}

TEST_F(HandlerIntegrationTest, MakeErrorResponse_Busy) {
  auto status = rocksdb::Status::Busy("resource busy");
  auto resp = MakeErrorResponse(status, "Busy");

  EXPECT_EQ(resp->statusCode(), drogon::k503ServiceUnavailable);

  auto json = resp->getJsonObject();
  ASSERT_TRUE(json != nullptr);
  EXPECT_EQ((*json)["error"].asString(), "service_busy");
  EXPECT_EQ((*json)["code"].asInt(), 503);
}

TEST_F(HandlerIntegrationTest, MakeErrorResponse_InternalError) {
  auto status = rocksdb::Status::Corruption("data corruption");
  auto resp = MakeErrorResponse(status, "Internal error");

  EXPECT_EQ(resp->statusCode(), drogon::k500InternalServerError);

  auto json = resp->getJsonObject();
  ASSERT_TRUE(json != nullptr);
  EXPECT_EQ((*json)["error"].asString(), "internal_error");
  EXPECT_EQ((*json)["code"].asInt(), 500);
}

// =============================================================================
// End-to-End Server Tests (using Drogon test client)
// =============================================================================

class ServerE2ETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    temp_dir_ = std::make_unique<TempDir>();
    db_path_ = (temp_dir_->path() / "e2e_test_db").string();

    // Open store
    auto status = Store::Open(db_path_, &store_);
    ASSERT_TRUE(status.ok()) << status.ToString();

    // Register handlers
    Config config;
    config.db_path = db_path_;
    RegisterHandlers(store_.get(), config);

    // Start Drogon in test mode on a random port
    port_ = 18080 + (std::random_device{}() % 1000);

    drogon::app()
        .addListener("127.0.0.1", port_)
        .setThreadNum(1)
        .disableSession();

    // Start server in background thread
    server_thread_ = std::thread([]() { drogon::app().run(); });

    // Wait for server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  static void TearDownTestSuite() {
    drogon::app().quit();
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    store_.reset();
    temp_dir_.reset();
  }

  static std::string BaseUrl() {
    return "http://127.0.0.1:" + std::to_string(port_);
  }

  static std::unique_ptr<TempDir> temp_dir_;
  static std::string db_path_;
  static std::unique_ptr<Store> store_;
  static uint16_t port_;
  static std::thread server_thread_;
};

std::unique_ptr<TempDir> ServerE2ETest::temp_dir_;
std::string ServerE2ETest::db_path_;
std::unique_ptr<Store> ServerE2ETest::store_;
uint16_t ServerE2ETest::port_ = 0;
std::thread ServerE2ETest::server_thread_;

TEST_F(ServerE2ETest, Health_Liveness) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());
  auto req = drogon::HttpRequest::newHttpRequest();
  req->setPath("/health");
  req->setMethod(drogon::Get);

  std::promise<void> promise;
  auto future = promise.get_future();

  client->sendRequest(
      req,
      [&promise](drogon::ReqResult result,
                 const drogon::HttpResponsePtr& resp) {
        EXPECT_EQ(result, drogon::ReqResult::Ok);
        EXPECT_EQ(resp->statusCode(), drogon::k200OK);

        auto json = resp->getJsonObject();
        ASSERT_TRUE(json != nullptr);
        EXPECT_EQ((*json)["status"].asString(), "healthy");

        promise.set_value();
      });

  ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
}

TEST_F(ServerE2ETest, Health_Readiness) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());
  auto req = drogon::HttpRequest::newHttpRequest();
  req->setPath("/health/ready");
  req->setMethod(drogon::Get);

  std::promise<void> promise;
  auto future = promise.get_future();

  client->sendRequest(
      req,
      [&promise](drogon::ReqResult result,
                 const drogon::HttpResponsePtr& resp) {
        EXPECT_EQ(result, drogon::ReqResult::Ok);
        EXPECT_EQ(resp->statusCode(), drogon::k200OK);

        auto json = resp->getJsonObject();
        ASSERT_TRUE(json != nullptr);
        EXPECT_EQ((*json)["status"].asString(), "healthy");
        EXPECT_TRUE(json->isMember("total_keys"));
        EXPECT_TRUE(json->isMember("total_objects"));
        EXPECT_TRUE(json->isMember("dedup_ratio"));

        promise.set_value();
      });

  ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
}

TEST_F(ServerE2ETest, KV_PutGetDelete) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());

  // PUT
  {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/v1/kv/test-key");
    req->setMethod(drogon::Put);
    req->setBody("test-value");

    std::promise<void> promise;
    auto future = promise.get_future();

    client->sendRequest(
        req,
        [&promise](drogon::ReqResult result,
                   const drogon::HttpResponsePtr& resp) {
          EXPECT_EQ(result, drogon::ReqResult::Ok);
          EXPECT_EQ(resp->statusCode(), drogon::k201Created);
          promise.set_value();
        });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }

  // GET
  {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/v1/kv/test-key");
    req->setMethod(drogon::Get);

    std::promise<void> promise;
    auto future = promise.get_future();

    client->sendRequest(
        req,
        [&promise](drogon::ReqResult result,
                   const drogon::HttpResponsePtr& resp) {
          EXPECT_EQ(result, drogon::ReqResult::Ok);
          EXPECT_EQ(resp->statusCode(), drogon::k200OK);
          EXPECT_EQ(std::string(resp->body()), "test-value");
          promise.set_value();
        });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }

  // DELETE
  {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/v1/kv/test-key");
    req->setMethod(drogon::Delete);

    std::promise<void> promise;
    auto future = promise.get_future();

    client->sendRequest(
        req,
        [&promise](drogon::ReqResult result,
                   const drogon::HttpResponsePtr& resp) {
          EXPECT_EQ(result, drogon::ReqResult::Ok);
          EXPECT_EQ(resp->statusCode(), drogon::k204NoContent);
          promise.set_value();
        });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }

  // GET after DELETE should return 404
  {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/v1/kv/test-key");
    req->setMethod(drogon::Get);

    std::promise<void> promise;
    auto future = promise.get_future();

    client->sendRequest(
        req,
        [&promise](drogon::ReqResult result,
                   const drogon::HttpResponsePtr& resp) {
          EXPECT_EQ(result, drogon::ReqResult::Ok);
          EXPECT_EQ(resp->statusCode(), drogon::k404NotFound);
          promise.set_value();
        });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }
}

TEST_F(ServerE2ETest, KV_PutWithJson) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());

  // PUT with JSON
  {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/v1/kv/json-key");
    req->setMethod(drogon::Put);
    req->setContentTypeString("application/json");
    req->setBody(R"({"value": "json-value"})");

    std::promise<void> promise;
    auto future = promise.get_future();

    client->sendRequest(
        req,
        [&promise](drogon::ReqResult result,
                   const drogon::HttpResponsePtr& resp) {
          EXPECT_EQ(result, drogon::ReqResult::Ok);
          EXPECT_EQ(resp->statusCode(), drogon::k201Created);
          promise.set_value();
        });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }

  // GET and verify
  {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/v1/kv/json-key");
    req->setMethod(drogon::Get);

    std::promise<void> promise;
    auto future = promise.get_future();

    client->sendRequest(
        req,
        [&promise](drogon::ReqResult result,
                   const drogon::HttpResponsePtr& resp) {
          EXPECT_EQ(result, drogon::ReqResult::Ok);
          EXPECT_EQ(resp->statusCode(), drogon::k200OK);
          EXPECT_EQ(std::string(resp->body()), "json-value");
          promise.set_value();
        });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }
}

TEST_F(ServerE2ETest, KV_List) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());

  // Add some keys
  for (int i = 0; i < 5; ++i) {
    store_->Put("list-test-key-" + std::to_string(i), "value-" + std::to_string(i));
  }

  // List all keys with prefix
  {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/v1/kv");
    req->setParameter("prefix", "list-test-key-");
    req->setMethod(drogon::Get);

    std::promise<void> promise;
    auto future = promise.get_future();

    client->sendRequest(
        req,
        [&promise](drogon::ReqResult result,
                   const drogon::HttpResponsePtr& resp) {
          EXPECT_EQ(result, drogon::ReqResult::Ok);
          EXPECT_EQ(resp->statusCode(), drogon::k200OK);

          auto json = resp->getJsonObject();
          ASSERT_TRUE(json != nullptr);
          EXPECT_TRUE(json->isArray());
          EXPECT_EQ(json->size(), 5);

          promise.set_value();
        });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }

  // List with limit
  {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/v1/kv");
    req->setParameter("prefix", "list-test-key-");
    req->setParameter("limit", "2");
    req->setMethod(drogon::Get);

    std::promise<void> promise;
    auto future = promise.get_future();

    client->sendRequest(
        req,
        [&promise](drogon::ReqResult result,
                   const drogon::HttpResponsePtr& resp) {
          EXPECT_EQ(result, drogon::ReqResult::Ok);
          EXPECT_EQ(resp->statusCode(), drogon::k200OK);

          auto json = resp->getJsonObject();
          ASSERT_TRUE(json != nullptr);
          EXPECT_TRUE(json->isArray());
          EXPECT_EQ(json->size(), 2);

          promise.set_value();
        });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }
}

TEST_F(ServerE2ETest, KV_Count) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());

  auto req = drogon::HttpRequest::newHttpRequest();
  req->setPath("/api/v1/kv/_count");
  req->setMethod(drogon::Get);

  std::promise<void> promise;
  auto future = promise.get_future();

  client->sendRequest(
      req,
      [&promise](drogon::ReqResult result,
                 const drogon::HttpResponsePtr& resp) {
        EXPECT_EQ(result, drogon::ReqResult::Ok);
        EXPECT_EQ(resp->statusCode(), drogon::k200OK);

        auto json = resp->getJsonObject();
        ASSERT_TRUE(json != nullptr);
        EXPECT_TRUE(json->isMember("keys"));
        EXPECT_TRUE(json->isMember("unique_values"));
        EXPECT_TRUE((*json)["exact"].asBool());

        promise.set_value();
      });

  ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
}

TEST_F(ServerE2ETest, KV_CountApprox) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());

  auto req = drogon::HttpRequest::newHttpRequest();
  req->setPath("/api/v1/kv/_count/approx");
  req->setMethod(drogon::Get);

  std::promise<void> promise;
  auto future = promise.get_future();

  client->sendRequest(
      req,
      [&promise](drogon::ReqResult result,
                 const drogon::HttpResponsePtr& resp) {
        EXPECT_EQ(result, drogon::ReqResult::Ok);
        EXPECT_EQ(resp->statusCode(), drogon::k200OK);

        auto json = resp->getJsonObject();
        ASSERT_TRUE(json != nullptr);
        EXPECT_TRUE(json->isMember("keys"));
        EXPECT_TRUE(json->isMember("unique_values"));
        EXPECT_TRUE(json->isMember("bytes"));
        EXPECT_FALSE((*json)["exact"].asBool());

        promise.set_value();
      });

  ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
}

TEST_F(ServerE2ETest, Admin_Sweep) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());

  auto req = drogon::HttpRequest::newHttpRequest();
  req->setPath("/api/v1/admin/sweep");
  req->setMethod(drogon::Post);

  std::promise<void> promise;
  auto future = promise.get_future();

  client->sendRequest(
      req,
      [&promise](drogon::ReqResult result,
                 const drogon::HttpResponsePtr& resp) {
        EXPECT_EQ(result, drogon::ReqResult::Ok);
        EXPECT_EQ(resp->statusCode(), drogon::k200OK);

        auto json = resp->getJsonObject();
        ASSERT_TRUE(json != nullptr);
        EXPECT_EQ((*json)["status"].asString(), "ok");
        EXPECT_TRUE(json->isMember("deleted"));

        promise.set_value();
      });

  ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
}

TEST_F(ServerE2ETest, Admin_Flush) {
  auto client = drogon::HttpClient::newHttpClient(BaseUrl());

  auto req = drogon::HttpRequest::newHttpRequest();
  req->setPath("/api/v1/admin/flush");
  req->setMethod(drogon::Post);

  std::promise<void> promise;
  auto future = promise.get_future();

  client->sendRequest(
      req,
      [&promise](drogon::ReqResult result,
                 const drogon::HttpResponsePtr& resp) {
        EXPECT_EQ(result, drogon::ReqResult::Ok);
        EXPECT_EQ(resp->statusCode(), drogon::k200OK);

        auto json = resp->getJsonObject();
        ASSERT_TRUE(json != nullptr);
        EXPECT_EQ((*json)["status"].asString(), "ok");

        promise.set_value();
      });

  ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
}

}  // namespace
}  // namespace prestige::server
