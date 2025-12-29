#pragma once

#include <prestige/store.hpp>
#include <prestige/server/config.hpp>

#include <drogon/HttpSimpleController.h>
#include <drogon/HttpAppFramework.h>

namespace prestige::server {

/**
 * Create an error response from a RocksDB status.
 */
drogon::HttpResponsePtr MakeErrorResponse(const rocksdb::Status& status,
                                           const std::string& context);

/**
 * Register all KV and admin handlers with the Drogon app.
 * Uses lambda handlers to capture the Store pointer.
 */
void RegisterHandlers(Store* store, const Config& config);

}  // namespace prestige::server
