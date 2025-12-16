\
#include <prestige/store.hpp>

#include <iostream>
#include <string>

static void usage(const char* argv0) {
  std::cerr
      << "usage:\n"
      << "  " << argv0 << " <db_path> put <key> <value>\n"
      << "  " << argv0 << " <db_path> get <key>\n"
      << "  " << argv0 << " <db_path> del <key>\n";
}

int main(int argc, char** argv) {
  if (argc < 4) { usage(argv[0]); return 2; }

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
  } else {
    usage(argv[0]);
    return 2;
  }
}
