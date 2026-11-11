// Normalization test vectors: run every vector in examples/normalization_vectors.tsv through
// prestige::internal::Normalize and report, by vector id, which ones do not produce the expected
// bytes.
//
// The vector file is the reference a reader's own implementation of the rule is checked against
// (docs/normalization.md). This program checks the other direction: that the file is true of this
// repository's implementation. It needs only the normalizer, not RocksDB or the store.
//
// Build and run from the repository root:
//   c++ -std=c++17 -Iinclude examples/normalization_vectors.cpp src/normalize.cpp \
//       -o normalization_vectors
//   ./normalization_vectors [path/to/normalization_vectors.tsv]
//
// Exit status: 0 when every vector passes, 1 when any fails, 2 when the file cannot be read or
// parsed.

#include <prestige/normalize.hpp>
#include <prestige/version.hpp>

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

// "-" stands for zero bytes; anything else is an even number of hex digits.
bool FromHex(const std::string& hex, std::string* out) {
  out->clear();
  if (hex == "-") return true;
  if (hex.empty() || hex.size() % 2 != 0) return false;
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  for (size_t i = 0; i < hex.size(); i += 2) {
    const int hi = nibble(hex[i]);
    const int lo = nibble(hex[i + 1]);
    if (hi < 0 || lo < 0) return false;
    out->push_back(static_cast<char>((hi << 4) | lo));
  }
  return true;
}

std::string ToHex(const std::string& bytes) {
  if (bytes.empty()) return "-";
  static const char kDigits[] = "0123456789ABCDEF";
  std::string hex;
  hex.reserve(bytes.size() * 2);
  for (unsigned char c : bytes) {
    hex.push_back(kDigits[c >> 4]);
    hex.push_back(kDigits[c & 0x0F]);
  }
  return hex;
}

bool ParseMode(const std::string& name, prestige::NormalizationMode* mode) {
  if (name == "kNone") {
    *mode = prestige::NormalizationMode::kNone;
  } else if (name == "kWhitespace") {
    *mode = prestige::NormalizationMode::kWhitespace;
  } else if (name == "kASCII") {
    *mode = prestige::NormalizationMode::kASCII;
  } else if (name == "kUnicode") {
    *mode = prestige::NormalizationMode::kUnicode;
  } else {
    return false;
  }
  return true;
}

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream in(line);
  while (std::getline(in, field, '\t')) fields.push_back(field);
  return fields;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "examples/normalization_vectors.tsv";
  std::ifstream file(path);
  if (!file) {
    std::cerr << "cannot open " << path << " (run from the repository root or pass the path)\n";
    return 2;
  }

  std::string vector_set = "?";
  std::set<std::string> seen;
  int total = 0;
  int failed = 0;
  int line_no = 0;
  std::string line;
  while (std::getline(file, line)) {
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '#') {
      const std::string tag = "# vector-set:";
      if (line.compare(0, tag.size(), tag) == 0) {
        vector_set = line.substr(tag.size());
        while (!vector_set.empty() && vector_set.front() == ' ') vector_set.erase(0, 1);
      }
      continue;
    }

    const std::vector<std::string> fields = SplitTabs(line);
    prestige::NormalizationMode mode;
    std::string input;
    std::string expected;
    if (fields.size() < 4 || !ParseMode(fields[1], &mode) || !FromHex(fields[2], &input) ||
        !FromHex(fields[3], &expected)) {
      std::cerr << path << ":" << line_no
                << ": expected id, mode, input_hex, expected_hex[, note]\n";
      return 2;
    }
    if (!seen.insert(fields[0]).second) {
      std::cerr << path << ":" << line_no << ": vector id " << fields[0] << " is used twice\n";
      return 2;
    }

    ++total;
    const std::string observed = prestige::internal::Normalize(input, mode);
    if (observed == expected) {
      std::cout << "PASS " << fields[0] << " " << fields[1] << "\n";
    } else {
      ++failed;
      std::cout << "FAIL " << fields[0] << " " << fields[1] << "\n"
                << "     input:    " << ToHex(input) << "\n"
                << "     expected: " << ToHex(expected) << "\n"
                << "     observed: " << ToHex(observed) << "\n";
    }
  }

  std::cout << "vector set " << vector_set << ", prestige " << prestige::Version() << ": "
            << (total - failed) << " of " << total << " vectors pass\n";
  return failed == 0 ? 0 : 1;
}
