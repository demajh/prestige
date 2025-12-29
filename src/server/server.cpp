#include <prestige/server/server.hpp>
#include <prestige/server/handlers.hpp>
#include <prestige/server/metrics.hpp>
#include <prestige/server/proxy.hpp>
#include <prestige/shutdown.hpp>

#include <drogon/drogon.h>

#include <iostream>
#include <thread>

namespace prestige::server {

Server::Server(const Config& config) : config_(config) {
  // Validate configuration
  config_.Validate();

  // Open the prestige store
  auto status = Store::Open(config_.db_path, &store_, config_.store);
  if (!status.ok()) {
    throw std::runtime_error("Failed to open store at " + config_.db_path +
                             ": " + status.ToString());
  }
}

Server::~Server() {
  if (running_) {
    Shutdown();
  }
}

void Server::SetupRoutes() {
  // Create shared metrics sink if metrics are enabled
  std::shared_ptr<PrometheusMetrics> metrics;
  if (config_.metrics.enabled) {
    metrics = std::make_shared<PrometheusMetrics>();

    // Wire up metrics to the store's options
    // Note: This would require reopening the store or setting metrics after open
    // For now, metrics are collected at the HTTP layer
  }

  // Register all handlers
  RegisterHandlers(store_.get(), config_);

  // Register metrics endpoint if enabled
  if (config_.metrics.enabled && metrics) {
    RegisterMetricsHandler(metrics, store_.get());
  }

  // Register proxy handler if enabled
  if (config_.proxy.enabled) {
    RegisterProxyHandler(store_.get(), config_.proxy, metrics);
  }
}

void Server::SetupShutdown() {
  // Register with global shutdown handler
  GlobalShutdownHandler().RegisterStore(store_.get());
  GlobalShutdownHandler().InstallSignalHandlers();

  // Add callback to stop Drogon
  GlobalShutdownHandler().OnShutdown([]() {
    std::cout << "Shutting down HTTP server..." << std::endl;
    drogon::app().quit();
  });
}

void Server::Run() {
  running_ = true;

  // Configure Drogon
  auto& app = drogon::app();

  // Set listener address and port
  app.addListener(config_.server.host, config_.server.port);

  // Set number of threads
  uint32_t threads = config_.server.threads;
  if (threads == 0) {
    threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;  // Fallback
  }
  app.setThreadNum(threads);

  // Configure timeouts and limits
  app.setMaxConnectionNum(10000);
  app.setMaxConnectionNumPerIP(100);
  app.setIdleConnectionTimeout(60);
  app.setKeepaliveRequestsNumber(100);

  // Disable session support (not needed for API server)
  app.disableSession();

  // Set up routes
  SetupRoutes();

  // Set up graceful shutdown
  SetupShutdown();

  std::cout << "Prestige server starting on " << config_.server.host
            << ":" << config_.server.port << " with " << threads << " threads"
            << std::endl;

  if (config_.proxy.enabled) {
    std::cout << "Proxy mode enabled, upstream: " << config_.proxy.upstream_base_url
              << std::endl;
  }

  // Run Drogon (blocking)
  app.run();

  running_ = false;
  std::cout << "Server stopped." << std::endl;
}

void Server::Shutdown() {
  if (running_) {
    GlobalShutdownHandler().Shutdown();
  }
}

}  // namespace prestige::server
