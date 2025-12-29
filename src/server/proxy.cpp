#include <prestige/server/proxy.hpp>

#include <drogon/drogon.h>
#include <json/json.h>

#include <algorithm>
#include <chrono>
#include <openssl/sha.h>
#include <set>
#include <sstream>

namespace prestige::server {

namespace {

// Compute SHA-256 hash of data and return as hex string
std::string Sha256Hex(std::string_view data) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash);

  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
    oss << std::setw(2) << static_cast<int>(hash[i]);
  }
  return oss.str();
}

uint64_t NowMillis() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

bool ShouldCache(int status_code, const ProxyConfig& config) {
  return std::find(config.cacheable_status_codes.begin(),
                   config.cacheable_status_codes.end(),
                   status_code) != config.cacheable_status_codes.end();
}

drogon::HttpResponsePtr CreateResponse(const CachedResponse& cached,
                                        const ProxyConfig& config) {
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(static_cast<drogon::HttpStatusCode>(cached.status_code));

  for (const auto& [name, value] : cached.headers) {
    // Check if this header should be stripped
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                   ::tolower);
    bool strip = false;
    for (const auto& strip_header : config.strip_response_headers) {
      std::string lower_strip = strip_header;
      std::transform(lower_strip.begin(), lower_strip.end(), lower_strip.begin(),
                     ::tolower);
      if (lower_name == lower_strip) {
        strip = true;
        break;
      }
    }
    if (!strip) {
      resp->addHeader(name, value);
    }
  }

  resp->setBody(cached.body);
  return resp;
}

}  // namespace

// --- ProxyCacheKey ---

std::string ProxyCacheKey::Serialize() const {
  std::ostringstream oss;
  oss << "proxy:" << method << ":" << path;
  if (!query.empty()) {
    oss << "?" << query;
  }
  if (!body_hash.empty()) {
    oss << ":body=" << body_hash;
  }
  if (!vary_headers.empty()) {
    oss << ":vary=" << vary_headers;
  }
  return oss.str();
}

ProxyCacheKey ProxyCacheKey::FromRequest(const drogon::HttpRequestPtr& req,
                                         const ProxyConfig& config) {
  ProxyCacheKey key;
  key.method = req->methodString();
  key.path = std::string(req->path());
  key.query = req->query();

  // Hash request body for POST/PUT/PATCH
  auto body = req->body();
  if (!body.empty() &&
      (req->method() == drogon::Post || req->method() == drogon::Put ||
       req->method() == drogon::Patch)) {
    key.body_hash = Sha256Hex(body);
  }

  // Build vary headers string
  std::ostringstream vary_oss;
  for (const auto& header_name : config.vary_headers) {
    auto value = req->getHeader(header_name);
    if (!value.empty()) {
      vary_oss << header_name << "=" << value << ";";
    }
  }
  key.vary_headers = vary_oss.str();

  return key;
}

// --- CachedResponse ---

std::string CachedResponse::Serialize() const {
  Json::Value json;
  json["status_code"] = status_code;
  json["cached_at_ms"] = static_cast<Json::UInt64>(cached_at_ms);
  json["upstream_latency_ms"] = static_cast<Json::UInt64>(upstream_latency_ms);

  Json::Value headers_json(Json::arrayValue);
  for (const auto& [name, value] : headers) {
    Json::Value h;
    h["name"] = name;
    h["value"] = value;
    headers_json.append(h);
  }
  json["headers"] = headers_json;

  // Base64 encode body to handle binary data
  json["body"] = drogon::utils::base64Encode(
      reinterpret_cast<const unsigned char*>(body.data()), body.size());

  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  return Json::writeString(builder, json);
}

CachedResponse CachedResponse::Deserialize(std::string_view data) {
  CachedResponse resp;

  Json::Value json;
  Json::CharReaderBuilder builder;
  std::string errors;
  std::istringstream stream{std::string(data)};

  if (!Json::parseFromStream(builder, stream, &json, &errors)) {
    return resp;  // Invalid, status_code will be 0
  }

  resp.status_code = json.get("status_code", 0).asInt();
  resp.cached_at_ms = json.get("cached_at_ms", 0).asUInt64();
  resp.upstream_latency_ms = json.get("upstream_latency_ms", 0).asUInt64();

  const auto& headers_json = json["headers"];
  if (headers_json.isArray()) {
    for (const auto& h : headers_json) {
      resp.headers.emplace_back(h.get("name", "").asString(),
                                h.get("value", "").asString());
    }
  }

  // Decode base64 body
  std::string encoded_body = json.get("body", "").asString();
  resp.body = drogon::utils::base64Decode(encoded_body);

  return resp;
}

// --- Proxy Handler Registration ---

void RegisterProxyHandler(Store* store,
                          const ProxyConfig& config,
                          std::shared_ptr<PrometheusMetrics> metrics) {
  if (!config.enabled) {
    return;
  }

  // Capture config by value for the lambda
  ProxyConfig proxy_config = config;

  // Register handler for all methods under /proxy/
  drogon::app().registerHandler(
      "/proxy/{path:.*}",
      [store, proxy_config, metrics](
          const drogon::HttpRequestPtr& req,
          std::function<void(const drogon::HttpResponsePtr&)>&& callback,
          const std::string& path) {
        // Generate cache key
        auto cache_key = ProxyCacheKey::FromRequest(req, proxy_config);
        std::string key_str = cache_key.Serialize();

        // Check if POST should be cached
        if (req->method() == drogon::Post && !proxy_config.cache_post) {
          // Don't cache, just forward
          auto start = std::chrono::steady_clock::now();

          // Build upstream URL
          std::string upstream_url = proxy_config.upstream_base_url;
          if (!upstream_url.empty() && upstream_url.back() == '/') {
            upstream_url.pop_back();
          }
          upstream_url += "/" + path;

          auto client = drogon::HttpClient::newHttpClient(proxy_config.upstream_base_url);

          auto upstream_req = drogon::HttpRequest::newHttpRequest();
          upstream_req->setMethod(req->method());
          upstream_req->setPath("/" + path);

          if (!req->body().empty()) {
            upstream_req->setBody(std::string(req->body()));
          }

          client->sendRequest(
              upstream_req,
              [callback, start, metrics](drogon::ReqResult result,
                                          const drogon::HttpResponsePtr& resp) {
                auto end = std::chrono::steady_clock::now();
                auto duration_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                        .count();

                if (metrics) {
                  metrics->RecordProxyUpstreamLatency(static_cast<double>(duration_ms));
                }

                if (result != drogon::ReqResult::Ok) {
                  auto error_resp = drogon::HttpResponse::newHttpResponse();
                  error_resp->setStatusCode(drogon::k502BadGateway);
                  error_resp->setBody("Upstream request failed");
                  callback(error_resp);
                  return;
                }

                callback(resp);
              },
              proxy_config.timeout_ms / 1000.0);
          return;
        }

        // Try to get from cache
        std::string cached_data;
        auto status = store->Get(key_str, &cached_data);

        if (status.ok()) {
          // Cache hit
          auto cached = CachedResponse::Deserialize(cached_data);
          if (cached.IsValid()) {
            if (metrics) {
              metrics->RecordProxyCacheHit();
            }

            // Add cache headers
            auto resp = CreateResponse(cached, proxy_config);
            resp->addHeader("X-Prestige-Cache", "HIT");
            resp->addHeader("X-Prestige-Cached-At",
                            std::to_string(cached.cached_at_ms));
            callback(resp);
            return;
          }
        }

        // Cache miss - forward to upstream
        if (metrics) {
          metrics->RecordProxyCacheMiss();
        }

        auto start = std::chrono::steady_clock::now();

        // Build upstream URL
        std::string upstream_url = proxy_config.upstream_base_url;
        if (!upstream_url.empty() && upstream_url.back() == '/') {
          upstream_url.pop_back();
        }

        auto client = drogon::HttpClient::newHttpClient(proxy_config.upstream_base_url);

        auto upstream_req = drogon::HttpRequest::newHttpRequest();
        upstream_req->setMethod(req->method());
        upstream_req->setPath("/" + path);

        // Copy headers
        static const std::set<std::string> skip_headers = {
            "host", "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"};

        for (const auto& [name, value] : req->headers()) {
          std::string lower_name = name;
          std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                         ::tolower);
          if (skip_headers.find(lower_name) == skip_headers.end()) {
            bool strip = false;
            for (const auto& strip_header : proxy_config.strip_request_headers) {
              std::string lower_strip = strip_header;
              std::transform(lower_strip.begin(), lower_strip.end(),
                             lower_strip.begin(), ::tolower);
              if (lower_name == lower_strip) {
                strip = true;
                break;
              }
            }
            if (!strip) {
              upstream_req->addHeader(name, value);
            }
          }
        }

        if (!req->body().empty()) {
          upstream_req->setBody(std::string(req->body()));
        }

        client->sendRequest(
            upstream_req,
            [store, key_str, proxy_config, metrics, callback, start](
                drogon::ReqResult result,
                const drogon::HttpResponsePtr& resp) {
              auto end = std::chrono::steady_clock::now();
              auto duration_ms =
                  std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                      .count();

              if (metrics) {
                metrics->RecordProxyUpstreamLatency(static_cast<double>(duration_ms));
              }

              CachedResponse cached;

              if (result != drogon::ReqResult::Ok) {
                cached.status_code = 502;  // Bad Gateway
                cached.body = "Upstream request failed";
                cached.cached_at_ms = NowMillis();
                cached.upstream_latency_ms = duration_ms;
                callback(CreateResponse(cached, proxy_config));
                return;
              }

              cached.status_code = static_cast<int>(resp->statusCode());
              cached.cached_at_ms = NowMillis();
              cached.upstream_latency_ms = duration_ms;

              // Copy headers
              static const std::set<std::string> skip_resp_headers = {
                  "connection", "keep-alive", "transfer-encoding"};

              for (const auto& [name, value] : resp->headers()) {
                std::string lower_name = name;
                std::transform(lower_name.begin(), lower_name.end(),
                               lower_name.begin(), ::tolower);
                if (skip_resp_headers.find(lower_name) == skip_resp_headers.end()) {
                  cached.headers.emplace_back(name, value);
                }
              }

              cached.body = std::string(resp->body());

              // Cache the response if it should be cached
              if (ShouldCache(cached.status_code, proxy_config)) {
                std::string serialized = cached.Serialize();
                auto put_status = store->Put(key_str, serialized);
                if (!put_status.ok()) {
                  LOG_WARN << "Failed to cache proxy response: "
                           << put_status.ToString();
                }
              }

              auto http_resp = CreateResponse(cached, proxy_config);
              http_resp->addHeader("X-Prestige-Cache", "MISS");
              callback(http_resp);
            },
            proxy_config.timeout_ms / 1000.0);
      },
      {drogon::Get, drogon::Post, drogon::Put, drogon::Delete,
       drogon::Patch, drogon::Head, drogon::Options});
}

}  // namespace prestige::server
