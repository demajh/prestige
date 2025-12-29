#pragma once

#include <prestige/server/config.hpp>
#include <prestige/store.hpp>

#include <memory>
#include <string>

namespace prestige::server {

/**
 * Prestige HTTP Server.
 *
 * Wraps a prestige::Store with a REST API using Drogon.
 * Supports two modes:
 *   1. KV Store: Direct access to Put/Get/Delete operations
 *   2. Proxy: Cache upstream API responses with deduplication
 */
class Server {
 public:
  /**
   * Create a server with the given configuration.
   */
  explicit Server(const Config& config);

  ~Server();

  // Non-copyable, non-movable
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  /**
   * Start the server (blocking).
   * Returns when the server shuts down.
   */
  void Run();

  /**
   * Request shutdown (async).
   * The server will stop accepting new connections and drain existing ones.
   */
  void Shutdown();

  /**
   * Get the underlying store (for advanced use).
   */
  Store* GetStore() { return store_.get(); }
  const Store* GetStore() const { return store_.get(); }

 private:
  void SetupRoutes();
  void SetupShutdown();

  Config config_;
  std::unique_ptr<Store> store_;
  bool running_ = false;
};

}  // namespace prestige::server
