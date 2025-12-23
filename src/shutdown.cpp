#include <prestige/shutdown.hpp>
#include <prestige/store.hpp>

#include <algorithm>
#include <csignal>
#include <condition_variable>
#include <thread>

namespace prestige {

namespace {
// Global instance for signal handler access
ShutdownHandler* g_handler = nullptr;
std::mutex g_handler_mutex;
std::condition_variable g_shutdown_cv;
}  // namespace

ShutdownHandler::ShutdownHandler() {
  std::lock_guard<std::mutex> lock(g_handler_mutex);
  if (!g_handler) {
    g_handler = this;
  }
}

ShutdownHandler::~ShutdownHandler() {
  RestoreSignalHandlers();
  Shutdown();

  std::lock_guard<std::mutex> lock(g_handler_mutex);
  if (g_handler == this) {
    g_handler = nullptr;
  }
}

void ShutdownHandler::RegisterStore(Store* store) {
  if (!store) return;

  std::lock_guard<std::mutex> lock(mutex_);
  // Avoid duplicates
  for (const auto* s : stores_) {
    if (s == store) return;
  }
  stores_.push_back(store);
}

void ShutdownHandler::UnregisterStore(Store* store) {
  if (!store) return;

  std::lock_guard<std::mutex> lock(mutex_);
  stores_.erase(std::remove(stores_.begin(), stores_.end(), store), stores_.end());
}

void ShutdownHandler::SignalHandler(int signum) {
  (void)signum;

  ShutdownHandler* handler = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_handler_mutex);
    handler = g_handler;
  }

  if (handler) {
    handler->Shutdown();
  }

  // Notify any waiters
  g_shutdown_cv.notify_all();
}

bool ShutdownHandler::InstallSignalHandlers() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (handlers_installed_) {
    return true;  // Already installed
  }

  struct sigaction sa;
  sa.sa_handler = SignalHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;  // Restart interrupted syscalls

  // Install SIGTERM handler
  if (sigaction(SIGTERM, &sa, &old_sigterm_) != 0) {
    return false;
  }

  // Install SIGINT handler (Ctrl+C)
  if (sigaction(SIGINT, &sa, &old_sigint_) != 0) {
    sigaction(SIGTERM, &old_sigterm_, nullptr);
    return false;
  }

  // Install SIGHUP handler (terminal hangup)
  if (sigaction(SIGHUP, &sa, &old_sighup_) != 0) {
    sigaction(SIGTERM, &old_sigterm_, nullptr);
    sigaction(SIGINT, &old_sigint_, nullptr);
    return false;
  }

  handlers_installed_ = true;
  return true;
}

void ShutdownHandler::RestoreSignalHandlers() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!handlers_installed_) {
    return;
  }

  sigaction(SIGTERM, &old_sigterm_, nullptr);
  sigaction(SIGINT, &old_sigint_, nullptr);
  sigaction(SIGHUP, &old_sighup_, nullptr);

  handlers_installed_ = false;
}

bool ShutdownHandler::Shutdown() {
  // Atomically check and set shutdown flag
  bool expected = false;
  if (!shutdown_requested_.compare_exchange_strong(expected, true)) {
    // Already shutting down, wait for completion
    while (!shutdown_complete_.load()) {
      std::this_thread::yield();
    }
    return false;
  }

  // Copy stores under lock to avoid holding lock during Close()
  std::vector<Store*> stores_copy;
  std::vector<std::function<void()>> callbacks_copy;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stores_copy = stores_;
    callbacks_copy = callbacks_;
    stores_.clear();
  }

  // Close all registered stores
  for (Store* store : stores_copy) {
    if (store) {
      store->Close();
    }
  }

  // Run shutdown callbacks
  for (const auto& callback : callbacks_copy) {
    if (callback) {
      callback();
    }
  }

  shutdown_complete_.store(true);
  g_shutdown_cv.notify_all();
  return true;
}

bool ShutdownHandler::IsShutdownRequested() const {
  return shutdown_requested_.load();
}

void ShutdownHandler::OnShutdown(std::function<void()> callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  callbacks_.push_back(std::move(callback));
}

void ShutdownHandler::WaitForShutdown() {
  std::unique_lock<std::mutex> lock(g_handler_mutex);
  g_shutdown_cv.wait(lock, [this] { return shutdown_complete_.load(); });
}

ShutdownHandler& GlobalShutdownHandler() {
  static ShutdownHandler instance;
  return instance;
}

}  // namespace prestige
