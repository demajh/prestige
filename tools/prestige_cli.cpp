#include <prestige/store.hpp>

#include <iostream>
#include <string>

static void usage(const char* argv0) {
  std::cerr
      << "usage:\n"
      << "  " << argv0 << " <db_path> put <key> <value>\n"
      << "  " << argv0 << " <db_path> get <key>\n"
      << "  " << argv0 << " <db_path> del <key>\n"
      << "  " << argv0 << " <db_path> count          (exact, O(N) scan)\n"
      << "  " << argv0 << " <db_path> count-approx   (fast O(1) estimate)\n"
      << "  " << argv0 << " <db_path> keys [prefix] [limit]\n"
      << "  " << argv0 << " <db_path> sweep\n"
      << "  " << argv0 << " <db_path> prune <max_age_s> <max_idle_s>\n"
      << "  " << argv0 << " <db_path> evict <target_bytes>\n"
      << "  " << argv0 << " <db_path> health\n";
}

int main(int argc, char** argv) {
  if (argc < 3) { usage(argv[0]); return 2; }

  std::string db_path = argv[1];
  std::string cmd = argv[2];

  prestige::Options opt;
  std::unique_ptr<prestige::Store> db;
  auto s = prestige::Store::Open(db_path, &db, opt);
  if (!s.ok()) {
    std::cerr << "Open failed: " << s.ToString() << "\n";
    return 1;
  }

  if (cmd == "put") {
    if (argc != 5) { usage(argv[0]); return 2; }
    s = db->Put(argv[3], argv[4]);
    if (!s.ok()) {
      std::cerr << "Put failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << "OK\n";
    return 0;
  } else if (cmd == "get") {
    if (argc != 4) { usage(argv[0]); return 2; }
    std::string v;
    s = db->Get(argv[3], &v);
    if (!s.ok()) {
      std::cerr << "Get failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << v << "\n";
    return 0;
  } else if (cmd == "del") {
    if (argc != 4) { usage(argv[0]); return 2; }
    s = db->Delete(argv[3]);
    if (!s.ok() && !s.IsNotFound()) {
      std::cerr << "Delete failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << "OK\n";
    return 0;
  } else if (cmd == "count") {
    if (argc != 3) { usage(argv[0]); return 2; }
    uint64_t keys = 0;
    uint64_t uniq = 0;
    s = db->CountKeys(&keys);
    if (!s.ok()) {
      std::cerr << "CountKeys failed: " << s.ToString() << "\n";
      return 1;
    }
    s = db->CountUniqueValues(&uniq);
    if (!s.ok()) {
      std::cerr << "CountUniqueValues failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << "keys=" << keys << " unique_values=" << uniq << "\n";
    return 0;
  } else if (cmd == "count-approx") {
    if (argc != 3) { usage(argv[0]); return 2; }
    uint64_t keys = 0;
    uint64_t uniq = 0;
    uint64_t bytes = 0;
    s = db->CountKeysApprox(&keys);
    if (!s.ok()) {
      std::cerr << "CountKeysApprox failed: " << s.ToString() << "\n";
      return 1;
    }
    s = db->CountUniqueValuesApprox(&uniq);
    if (!s.ok()) {
      std::cerr << "CountUniqueValuesApprox failed: " << s.ToString() << "\n";
      return 1;
    }
    s = db->GetTotalStoreBytesApprox(&bytes);
    if (!s.ok()) {
      std::cerr << "GetTotalStoreBytesApprox failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << "keys~=" << keys << " unique_values~=" << uniq
              << " bytes~=" << bytes << " (approximate)\n";
    return 0;
  } else if (cmd == "keys") {
    // keys [prefix] [limit]
    std::string prefix;
    uint64_t limit = 0;
    if (argc == 4) {
      prefix = argv[3];
    } else if (argc == 5) {
      prefix = argv[3];
      try {
        limit = static_cast<uint64_t>(std::stoull(argv[4]));
      } catch (...) {
        std::cerr << "Invalid limit: " << argv[4] << "\n";
        return 2;
      }
    } else if (argc != 3) {
      usage(argv[0]);
      return 2;
    }

    std::vector<std::string> out;
    s = db->ListKeys(&out, limit, prefix);
    if (!s.ok()) {
      std::cerr << "ListKeys failed: " << s.ToString() << "\n";
      return 1;
    }
    for (const auto& k : out) {
      std::cout << k << "\n";
    }
    return 0;
  } else if (cmd == "sweep") {
    if (argc != 3) { usage(argv[0]); return 2; }
    uint64_t deleted = 0;
    s = db->Sweep(&deleted);
    if (!s.ok()) {
      std::cerr << "Sweep failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << "deleted=" << deleted << "\n";
    return 0;
  } else if (cmd == "prune") {
    if (argc != 5) { usage(argv[0]); return 2; }
    uint64_t max_age_s = 0, max_idle_s = 0;
    try {
      max_age_s = std::stoull(argv[3]);
      max_idle_s = std::stoull(argv[4]);
    } catch (...) {
      std::cerr << "Invalid arguments\n";
      return 2;
    }
    uint64_t deleted = 0;
    s = db->Prune(max_age_s, max_idle_s, &deleted);
    if (!s.ok()) {
      std::cerr << "Prune failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << "deleted=" << deleted << "\n";
    return 0;
  } else if (cmd == "evict") {
    if (argc != 4) { usage(argv[0]); return 2; }
    uint64_t target_bytes = 0;
    try {
      target_bytes = std::stoull(argv[3]);
    } catch (...) {
      std::cerr << "Invalid target_bytes\n";
      return 2;
    }
    uint64_t evicted = 0;
    s = db->EvictLRU(target_bytes, &evicted);
    if (!s.ok()) {
      std::cerr << "Evict failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << "evicted=" << evicted << "\n";
    return 0;
  } else if (cmd == "health") {
    if (argc != 3) { usage(argv[0]); return 2; }
    prestige::HealthStats stats;
    s = db->GetHealth(&stats);
    if (!s.ok()) {
      std::cerr << "Health failed: " << s.ToString() << "\n";
      return 1;
    }
    std::cout << "total_keys=" << stats.total_keys << "\n"
              << "total_objects=" << stats.total_objects << "\n"
              << "total_bytes=" << stats.total_bytes << "\n"
              << "expired_objects=" << stats.expired_objects << "\n"
              << "orphaned_objects=" << stats.orphaned_objects << "\n"
              << "oldest_object_age_s=" << stats.oldest_object_age_s << "\n"
              << "newest_access_age_s=" << stats.newest_access_age_s << "\n"
              << "dedup_ratio=" << stats.dedup_ratio << "\n";
    return 0;
  } else {
    usage(argv[0]);
    return 2;
  }
}
