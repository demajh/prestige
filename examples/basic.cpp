#include <prestige/store.hpp>

#include <iostream>

int main() {
  prestige::Options opt;
  std::unique_ptr<prestige::Store> db;

  auto s = prestige::Store::Open("./prestige_db", &db, opt);
  if (!s.ok()) {
    std::cerr << "Open failed: " << s.ToString() << "\n";
    return 1;
  }

  s = db->Put("k1", "HELLO");
  if (!s.ok()) std::cerr << "Put k1 failed: " << s.ToString() << "\n";

  s = db->Put("k2", "HELLO");
  if (!s.ok()) std::cerr << "Put k2 failed: " << s.ToString() << "\n";

  std::string v;
  s = db->Get("k2", &v);
  if (!s.ok()) {
    std::cerr << "Get k2 failed: " << s.ToString() << "\n";
    return 1;
  }
  std::cout << "k2=" << v << "\n";

  // Overwrite adjusts refcounts and may GC old objects.
  s = db->Put("k2", "WORLD");
  if (!s.ok()) std::cerr << "Put overwrite failed: " << s.ToString() << "\n";

  s = db->Get("k2", &v);
  if (!s.ok()) {
    std::cerr << "Get k2 failed: " << s.ToString() << "\n";
    return 1;
  }
  std::cout << "k2=" << v << "\n";

  // Delete keys; GC should reclaim objects once refcount hits zero.
  s = db->Delete("k1");
  if (!s.ok() && !s.IsNotFound()) std::cerr << "Delete k1 failed: " << s.ToString() << "\n";

  s = db->Delete("k2");
  if (!s.ok() && !s.IsNotFound()) std::cerr << "Delete k2 failed: " << s.ToString() << "\n";

  std::cout << "done\n";
  return 0;
}
