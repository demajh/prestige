#include <prestige/server/handlers.hpp>

#include <drogon/drogon.h>
#include <json/json.h>

#include <algorithm>
#include <sstream>

namespace prestige::server {

// --- Error Response Helper ---

drogon::HttpResponsePtr MakeErrorResponse(const rocksdb::Status& status,
                                           const std::string& context) {
  Json::Value json;
  int http_code = 500;

  if (status.IsNotFound()) {
    json["error"] = "not_found";
    json["code"] = 404;
    http_code = 404;
  } else if (status.IsInvalidArgument()) {
    json["error"] = "invalid_argument";
    json["code"] = 400;
    http_code = 400;
  } else if (status.IsTimedOut()) {
    json["error"] = "timeout";
    json["code"] = 504;
    http_code = 504;
  } else if (status.IsBusy() || status.IsTryAgain()) {
    json["error"] = "service_busy";
    json["code"] = 503;
    http_code = 503;
  } else {
    json["error"] = "internal_error";
    json["code"] = 500;
    http_code = 500;
  }

  json["message"] = context + ": " + status.ToString();

  auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
  resp->setStatusCode(static_cast<drogon::HttpStatusCode>(http_code));
  return resp;
}

// --- Handler Registration ---

void RegisterHandlers(Store* store, const Config& config) {
  auto& app = drogon::app();

  // ==========================================================================
  // KV Endpoints
  // ==========================================================================

  // PUT /api/v1/kv/{key} - Store a value
  app.registerHandler(
      "/api/v1/kv/{key}",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback,
              const std::string& key) {
        std::string value;

        // Check content type (case-insensitive)
        auto content_type = req->getHeader("Content-Type");
        if (content_type.empty()) {
          content_type = req->getHeader("content-type");
        }
        std::string lower_ct = content_type;
        std::transform(lower_ct.begin(), lower_ct.end(), lower_ct.begin(), ::tolower);

        if (lower_ct.find("application/json") != std::string::npos) {
          // Parse JSON body - try getJsonObject first, then manual parse
          auto json = req->getJsonObject();
          if (!json) {
            // Manual parse as fallback
            Json::Value parsed;
            Json::CharReaderBuilder builder;
            std::string errors;
            std::string body_str(req->body());
            std::istringstream stream(body_str);
            if (Json::parseFromStream(builder, stream, &parsed, &errors)) {
              json = std::make_shared<Json::Value>(parsed);
            }
          }

          if (json && json->isMember("value")) {
            value = (*json)["value"].asString();
          } else {
            Json::Value error;
            error["error"] = "invalid_argument";
            error["message"] = "JSON body must contain 'value' field";
            error["code"] = 400;
            auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
            resp->setStatusCode(drogon::k400BadRequest);
            callback(resp);
            return;
          }
        } else {
          // Treat body as raw value
          value = std::string(req->body());
        }

        // Store the value
        auto status = store->Put(key, value);
        if (!status.ok()) {
          callback(MakeErrorResponse(status, "Put failed for key '" + key + "'"));
          return;
        }

        Json::Value json;
        json["status"] = "ok";
        json["key"] = key;

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k201Created);
        callback(resp);
      },
      {drogon::Put});

  // GET /api/v1/kv/{key} - Retrieve a value
  app.registerHandler(
      "/api/v1/kv/{key}",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback,
              const std::string& key) {
        std::string value;
        auto status = store->Get(key, &value);

        if (!status.ok()) {
          callback(MakeErrorResponse(status, "Get failed for key '" + key + "'"));
          return;
        }

        // Check Accept header to determine response format
        auto accept = req->getHeader("Accept");
        if (accept.find("application/json") != std::string::npos) {
          Json::Value json;
          json["key"] = key;
          json["value"] = value;
          auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
          resp->setStatusCode(drogon::k200OK);
          callback(resp);
        } else {
          // Return raw value
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setBody(value);
          resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);
          resp->setStatusCode(drogon::k200OK);
          callback(resp);
        }
      },
      {drogon::Get});

  // DELETE /api/v1/kv/{key} - Delete a key
  app.registerHandler(
      "/api/v1/kv/{key}",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback,
              const std::string& key) {
        auto status = store->Delete(key);

        if (status.IsNotFound()) {
          callback(MakeErrorResponse(status, "Key not found: " + key));
          return;
        }

        if (!status.ok()) {
          callback(MakeErrorResponse(status, "Delete failed for key '" + key + "'"));
          return;
        }

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k204NoContent);
        callback(resp);
      },
      {drogon::Delete});

  // GET /api/v1/kv - List keys
  app.registerHandler(
      "/api/v1/kv",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        std::string prefix = req->getParameter("prefix");
        uint64_t limit = 0;

        auto limit_str = req->getParameter("limit");
        if (!limit_str.empty()) {
          try {
            limit = std::stoull(limit_str);
          } catch (...) {
            Json::Value error;
            error["error"] = "invalid_argument";
            error["message"] = "Invalid limit parameter";
            error["code"] = 400;
            auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
            resp->setStatusCode(drogon::k400BadRequest);
            callback(resp);
            return;
          }
        }

        std::vector<std::string> keys;
        auto status = store->ListKeys(&keys, limit, prefix);

        if (!status.ok()) {
          callback(MakeErrorResponse(status, "ListKeys failed"));
          return;
        }

        Json::Value json(Json::arrayValue);
        for (const auto& key : keys) {
          json.append(key);
        }

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Get});

  // GET /api/v1/kv/_count - Count keys (exact)
  app.registerHandler(
      "/api/v1/kv/_count",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        uint64_t key_count = 0;
        uint64_t unique_count = 0;

        auto status = store->CountKeys(&key_count);
        if (!status.ok()) {
          callback(MakeErrorResponse(status, "CountKeys failed"));
          return;
        }

        status = store->CountUniqueValues(&unique_count);
        if (!status.ok()) {
          callback(MakeErrorResponse(status, "CountUniqueValues failed"));
          return;
        }

        Json::Value json;
        json["keys"] = static_cast<Json::UInt64>(key_count);
        json["unique_values"] = static_cast<Json::UInt64>(unique_count);
        json["exact"] = true;

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Get});

  // GET /api/v1/kv/_count/approx - Count keys (approximate)
  app.registerHandler(
      "/api/v1/kv/_count/approx",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        uint64_t key_count = 0;
        uint64_t unique_count = 0;
        uint64_t bytes = 0;

        auto status = store->CountKeysApprox(&key_count);
        if (!status.ok()) {
          callback(MakeErrorResponse(status, "CountKeysApprox failed"));
          return;
        }

        status = store->CountUniqueValuesApprox(&unique_count);
        if (!status.ok()) {
          callback(MakeErrorResponse(status, "CountUniqueValuesApprox failed"));
          return;
        }

        status = store->GetTotalStoreBytesApprox(&bytes);
        if (!status.ok()) {
          callback(MakeErrorResponse(status, "GetTotalStoreBytesApprox failed"));
          return;
        }

        Json::Value json;
        json["keys"] = static_cast<Json::UInt64>(key_count);
        json["unique_values"] = static_cast<Json::UInt64>(unique_count);
        json["bytes"] = static_cast<Json::UInt64>(bytes);
        json["exact"] = false;

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Get});

  // ==========================================================================
  // Admin Endpoints
  // ==========================================================================

  // POST /api/v1/admin/sweep - Sweep expired/orphaned objects
  app.registerHandler(
      "/api/v1/admin/sweep",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        uint64_t deleted = 0;
        auto status = store->Sweep(&deleted);

        if (!status.ok()) {
          callback(MakeErrorResponse(status, "Sweep failed"));
          return;
        }

        Json::Value json;
        json["status"] = "ok";
        json["deleted"] = static_cast<Json::UInt64>(deleted);

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Post});

  // POST /api/v1/admin/prune - Prune by age/idle time
  app.registerHandler(
      "/api/v1/admin/prune",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        auto json_body = req->getJsonObject();
        if (!json_body) {
          Json::Value error;
          error["error"] = "invalid_argument";
          error["message"] = "Request body must be JSON with max_age_s and/or max_idle_s";
          error["code"] = 400;
          auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
          resp->setStatusCode(drogon::k400BadRequest);
          callback(resp);
          return;
        }

        uint64_t max_age_s = 0;
        uint64_t max_idle_s = 0;

        if (json_body->isMember("max_age_s")) {
          max_age_s = (*json_body)["max_age_s"].asUInt64();
        }
        if (json_body->isMember("max_idle_s")) {
          max_idle_s = (*json_body)["max_idle_s"].asUInt64();
        }

        uint64_t deleted = 0;
        auto status = store->Prune(max_age_s, max_idle_s, &deleted);

        if (!status.ok()) {
          callback(MakeErrorResponse(status, "Prune failed"));
          return;
        }

        Json::Value json;
        json["status"] = "ok";
        json["deleted"] = static_cast<Json::UInt64>(deleted);

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Post});

  // POST /api/v1/admin/evict - LRU eviction
  app.registerHandler(
      "/api/v1/admin/evict",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        auto json_body = req->getJsonObject();
        if (!json_body || !json_body->isMember("target_bytes")) {
          Json::Value error;
          error["error"] = "invalid_argument";
          error["message"] = "Request body must be JSON with target_bytes";
          error["code"] = 400;
          auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
          resp->setStatusCode(drogon::k400BadRequest);
          callback(resp);
          return;
        }

        uint64_t target_bytes = (*json_body)["target_bytes"].asUInt64();
        uint64_t evicted = 0;

        auto status = store->EvictLRU(target_bytes, &evicted);

        if (!status.ok()) {
          callback(MakeErrorResponse(status, "Evict failed"));
          return;
        }

        Json::Value json;
        json["status"] = "ok";
        json["evicted"] = static_cast<Json::UInt64>(evicted);

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Post});

  // POST /api/v1/admin/flush - Flush to disk
  app.registerHandler(
      "/api/v1/admin/flush",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        auto status = store->Flush();

        if (!status.ok()) {
          callback(MakeErrorResponse(status, "Flush failed"));
          return;
        }

        Json::Value json;
        json["status"] = "ok";

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Post});

  // ==========================================================================
  // Health Endpoints
  // ==========================================================================

  // GET /health - Liveness check
  app.registerHandler(
      "/health",
      [](const drogon::HttpRequestPtr& req,
         std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        Json::Value json;
        json["status"] = "healthy";

        auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
      },
      {drogon::Get});

  // GET /health/ready - Readiness check with stats
  app.registerHandler(
      "/health/ready",
      [store](const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        HealthStats stats;
        auto status = store->GetHealth(&stats);

        Json::Value json;
        if (status.ok()) {
          json["status"] = "healthy";
          json["total_keys"] = static_cast<Json::UInt64>(stats.total_keys);
          json["total_objects"] = static_cast<Json::UInt64>(stats.total_objects);
          json["total_bytes"] = static_cast<Json::UInt64>(stats.total_bytes);
          json["expired_objects"] = static_cast<Json::UInt64>(stats.expired_objects);
          json["orphaned_objects"] = static_cast<Json::UInt64>(stats.orphaned_objects);
          json["oldest_object_age_s"] = static_cast<Json::UInt64>(stats.oldest_object_age_s);
          json["newest_access_age_s"] = static_cast<Json::UInt64>(stats.newest_access_age_s);
          json["dedup_ratio"] = stats.dedup_ratio;

          auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
          resp->setStatusCode(drogon::k200OK);
          callback(resp);
        } else {
          json["status"] = "unhealthy";
          json["error"] = status.ToString();

          auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
          resp->setStatusCode(drogon::k503ServiceUnavailable);
          callback(resp);
        }
      },
      {drogon::Get});
}

}  // namespace prestige::server
