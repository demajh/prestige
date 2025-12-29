#include <prestige/server/config.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace prestige::server {

namespace {

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " [options]\n"
            << "\nOptions:\n"
            << "  --config, -c <path>       Path to YAML config file\n"
            << "  --host <addr>             Bind address (default: 0.0.0.0)\n"
            << "  --port, -p <port>         Listen port (default: 8080)\n"
            << "  --threads <n>             Worker threads (default: auto)\n"
            << "  --db-path <path>          Database path (required)\n"
            << "  --proxy-upstream <url>    Enable proxy mode with upstream URL\n"
            << "  --log-level <level>       Log level: debug, info, warn, error\n"
            << "  --help, -h                Show this help\n"
            << "\nExamples:\n"
            << "  " << argv0 << " --db-path /data/prestige --port 8080\n"
            << "  " << argv0 << " --config /etc/prestige/server.yaml\n"
            << "  " << argv0 << " --db-path /data/cache --proxy-upstream https://api.example.com\n";
}

// Simple YAML-like parser for basic config files
// Format:
//   key: value
//   section:
//     key: value
std::string Trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

Config Config::LoadFromFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open config file: " + path);
  }

  Config config;
  std::string current_section;
  std::string line;

  while (std::getline(file, line)) {
    line = Trim(line);

    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Check for section header (no leading whitespace in original line)
    size_t colon_pos = line.find(':');
    if (colon_pos != std::string::npos) {
      std::string key = Trim(line.substr(0, colon_pos));
      std::string value = Trim(line.substr(colon_pos + 1));

      // If value is empty, this is a section header
      if (value.empty()) {
        current_section = key;
        continue;
      }

      // Remove quotes from value if present
      if (value.size() >= 2 &&
          ((value.front() == '"' && value.back() == '"') ||
           (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1, value.size() - 2);
      }

      // Parse based on current section
      if (current_section == "server") {
        if (key == "host") {
          config.server.host = value;
        } else if (key == "port") {
          config.server.port = static_cast<uint16_t>(std::stoul(value));
        } else if (key == "threads") {
          config.server.threads = static_cast<uint32_t>(std::stoul(value));
        } else if (key == "log_level") {
          config.server.log_level = value;
        }
      } else if (current_section == "store") {
        if (key == "path") {
          config.db_path = value;
        } else if (key == "block_cache_bytes") {
          config.store.block_cache_bytes = std::stoull(value);
        } else if (key == "default_ttl_seconds") {
          config.store.default_ttl_seconds = std::stoull(value);
        } else if (key == "max_store_bytes") {
          config.store.max_store_bytes = std::stoull(value);
        }
      } else if (current_section == "proxy") {
        if (key == "enabled") {
          config.proxy.enabled = (value == "true" || value == "1" || value == "yes");
        } else if (key == "upstream_base_url") {
          config.proxy.upstream_base_url = value;
        } else if (key == "timeout_ms") {
          config.proxy.timeout_ms = static_cast<uint32_t>(std::stoul(value));
        }
      } else if (current_section == "metrics") {
        if (key == "enabled") {
          config.metrics.enabled = (value == "true" || value == "1" || value == "yes");
        } else if (key == "path") {
          config.metrics.path = value;
        }
      } else if (current_section.empty()) {
        // Top-level keys
        if (key == "db_path") {
          config.db_path = value;
        }
      }
    }
  }

  return config;
}

Config Config::LoadFromArgs(int argc, char** argv) {
  Config config;
  std::string config_file;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--config" || arg == "-c") {
      if (++i >= argc) {
        throw std::runtime_error("--config requires a path argument");
      }
      config_file = argv[i];
    } else if (arg == "--host") {
      if (++i >= argc) {
        throw std::runtime_error("--host requires an address argument");
      }
      config.server.host = argv[i];
    } else if (arg == "--port" || arg == "-p") {
      if (++i >= argc) {
        throw std::runtime_error("--port requires a port number");
      }
      config.server.port = static_cast<uint16_t>(std::stoul(argv[i]));
    } else if (arg == "--threads") {
      if (++i >= argc) {
        throw std::runtime_error("--threads requires a number");
      }
      config.server.threads = static_cast<uint32_t>(std::stoul(argv[i]));
    } else if (arg == "--db-path") {
      if (++i >= argc) {
        throw std::runtime_error("--db-path requires a path");
      }
      config.db_path = argv[i];
    } else if (arg == "--proxy-upstream") {
      if (++i >= argc) {
        throw std::runtime_error("--proxy-upstream requires a URL");
      }
      config.proxy.enabled = true;
      config.proxy.upstream_base_url = argv[i];
    } else if (arg == "--log-level") {
      if (++i >= argc) {
        throw std::runtime_error("--log-level requires a level");
      }
      config.server.log_level = argv[i];
    } else if (arg[0] == '-') {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  // If a config file was specified, load it first then override with CLI args
  if (!config_file.empty()) {
    Config file_config = LoadFromFile(config_file);

    // Merge: CLI args override file config
    if (config.server.host == "0.0.0.0" && file_config.server.host != "0.0.0.0") {
      config.server.host = file_config.server.host;
    }
    if (config.server.port == 8080 && file_config.server.port != 8080) {
      config.server.port = file_config.server.port;
    }
    if (config.server.threads == 0 && file_config.server.threads != 0) {
      config.server.threads = file_config.server.threads;
    }
    if (config.db_path.empty()) {
      config.db_path = file_config.db_path;
    }
    if (!config.proxy.enabled && file_config.proxy.enabled) {
      config.proxy = file_config.proxy;
    }

    // Always inherit store options from file unless explicitly overridden
    config.store = file_config.store;
    config.metrics = file_config.metrics;
  }

  return config;
}

void Config::Validate() const {
  if (db_path.empty()) {
    throw std::runtime_error("db_path is required (use --db-path or config file)");
  }

  if (server.port == 0 || server.port > 65535) {
    throw std::runtime_error("Invalid port number: " + std::to_string(server.port));
  }

  if (proxy.enabled && proxy.upstream_base_url.empty()) {
    throw std::runtime_error("proxy.upstream_base_url is required when proxy is enabled");
  }

  // Validate log level
  if (server.log_level != "debug" && server.log_level != "info" &&
      server.log_level != "warn" && server.log_level != "error") {
    throw std::runtime_error("Invalid log_level: " + server.log_level +
                             " (must be debug, info, warn, or error)");
  }
}

}  // namespace prestige::server
