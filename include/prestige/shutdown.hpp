#pragma once

#include <atomic>
#include <csignal>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace prestige {

class Store;  // Forward declaration

/**
 * ShutdownHandler provides graceful shutdown for prestige stores.
 *
 * Usage:
 *   1. Create a ShutdownHandler instance (typically one per process)
 *   2. Register stores with RegisterStore()
 *   3. Call InstallSignalHandlers() to catch SIGTERM/SIGINT/SIGHUP
 *   4. On signal, all registered stores are automatically closed
 *
 * Thread-safe: All methods can be called from any thread.
 *
 * Example:
 *   prestige::ShutdownHandler shutdown_handler;
 *   shutdown_handler.InstallSignalHandlers();
 *   shutdown_handler.RegisterStore(my_store.get());
 *   // ... use store ...
 *   // On SIGTERM/SIGINT, store is automatically closed
 */
class ShutdownHandler {
 public:
  ShutdownHandler();
  ~ShutdownHandler();

  // Non-copyable, non-movable
  ShutdownHandler(const ShutdownHandler&) = delete;
  ShutdownHandler& operator=(const ShutdownHandler&) = delete;

  /**
   * Register a store for automatic shutdown.
   * The store pointer must remain valid until Unregister() or shutdown.
   */
  void RegisterStore(Store* store);

  /**
   * Unregister a store (e.g., before destroying it).
   */
  void UnregisterStore(Store* store);

  /**
   * Install signal handlers for SIGTERM, SIGINT, and SIGHUP.
   * On any of these signals, Shutdown() is called automatically.
   * Returns true if handlers were installed successfully.
   *
   * Note: This modifies global signal handlers. Only call once per process.
   */
  bool InstallSignalHandlers();

  /**
   * Restore original signal handlers.
   */
  void RestoreSignalHandlers();

  /**
   * Trigger shutdown: close all registered stores.
   * Thread-safe and idempotent (safe to call multiple times).
   * Returns true if shutdown was performed, false if already shut down.
   */
  bool Shutdown();

  /**
   * Check if shutdown has been requested.
   */
  bool IsShutdownRequested() const;

  /**
   * Register a custom callback to run during shutdown.
   * Callbacks are invoked after stores are closed, in registration order.
   */
  void OnShutdown(std::function<void()> callback);

  /**
   * Block until shutdown is complete.
   * Useful for main() to wait for graceful termination.
   */
  void WaitForShutdown();

 private:
  static void SignalHandler(int signum);

  std::mutex mutex_;
  std::vector<Store*> stores_;
  std::vector<std::function<void()>> callbacks_;
  std::atomic<bool> shutdown_requested_{false};
  std::atomic<bool> shutdown_complete_{false};
  bool handlers_installed_ = false;

  // Original signal handlers to restore
  struct sigaction old_sigterm_;
  struct sigaction old_sigint_;
  struct sigaction old_sighup_;
};

/**
 * Global shutdown handler instance.
 * Use this for simple single-instance deployments.
 */
ShutdownHandler& GlobalShutdownHandler();

}  // namespace prestige
