// Outcome records demo: per-family false-accept reports from a synthetic workload.
//
// The store ranks and the caller decides (docs/candidates.md). This program opens a temporary store, loads a small
// workload of task families from a tab-separated file (stored values carrying a ground-truth item id, and queries
// labelled with the item each one should be served by) and, for every query:
//
//   1. asks Candidates() for the nearest stored values of the family;
//   2. applies the family's similarity threshold (the similarity gate);
//   3. runs the caller's constraint check on each candidate above the gate, in rank order: does the candidate's
//      "item" metadata name the item the query expects? Here the ground-truth label stands in for the tenant,
//      schema or side-effect checks a real caller would run;
//   4. records every verdict with RecordOutcome(): kAccepted when the check passed, kRejected when the gate let a
//      different item through (a false accept of the similarity gate), kNoCandidate when nothing was above the gate.
//
// It ends with GetFamilyReport() for every family: outcomes, false-accept rate, the similarity and rank
// distributions of the false accepts, and the advisory suggested threshold. The run is deterministic: fixed data,
// no randomness, single-threaded CPU inference.
//
// Build (semantic mode; see docs/outcomes-demo.md):
//   cmake -S . -B build -DPRESTIGE_ENABLE_SEMANTIC=ON ...
//   cmake --build build --target prestige_example_outcomes
// Run:
//   ./build/prestige_example_outcomes --model models/bge-small-en-v1.5_onnx/model.onnx
//       [--model-type bge-small|minilm] [--workload examples/outcomes_workload.tsv] [--trace]
//
// Without --model, or in a build without PRESTIGE_ENABLE_SEMANTIC, the store runs in exact mode: Candidates()
// returns the identical value at similarity 1.0 or nothing, so only verbatim repeats are served and a false accept
// cannot happen. The output says so.

#include <prestige/store.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef PRESTIGE_OUTCOMES_WORKLOAD
#define PRESTIGE_OUTCOMES_WORKLOAD "examples/outcomes_workload.tsv"
#endif

namespace {

constexpr const char* kToolVersion = "outcomes-demo/1";
constexpr const char* kSchemaVersion = "ground-truth@1";
constexpr size_t kCandidatesPerQuery = 5;

struct Item {
  std::string id;
  std::string text;
};

struct Query {
  std::string expected;  // the item that should serve this query; empty when no stored item does
  std::string text;
};

struct Family {
  std::string id;
  float threshold = 0.85f;
  std::vector<Item> items;
  std::vector<Query> queries;
};

struct Args {
  std::string model_path;
  std::string model_type = "bge-small";
  std::string workload = PRESTIGE_OUTCOMES_WORKLOAD;
  bool trace = false;
};

// What the demo itself counted per family, next to what the store reports.
struct Tally {
  uint64_t queries = 0;
  uint64_t served = 0;  // queries that ended in an accepted candidate
};

void Usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " [--model <model.onnx>] [--model-type bge-small|minilm]\n"
            << "       [--workload <file.tsv>] [--trace]\n"
            << "\nWithout --model the store runs in exact mode (similarity is exact-only).\n"
            << "vocab.txt must sit next to the model file.\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](std::string* out) {
      if (i + 1 >= argc) {
        std::cerr << a << " needs a value\n";
        return false;
      }
      *out = argv[++i];
      return true;
    };
    if (a == "--model") {
      if (!value(&args->model_path)) return false;
    } else if (a == "--model-type") {
      if (!value(&args->model_type)) return false;
      if (args->model_type != "bge-small" && args->model_type != "minilm") {
        std::cerr << "--model-type must be bge-small or minilm\n";
        return false;
      }
    } else if (a == "--workload") {
      if (!value(&args->workload)) return false;
    } else if (a == "--trace") {
      args->trace = true;
    } else if (a == "--help" || a == "-h") {
      Usage(argv[0]);
      return false;
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      Usage(argv[0]);
      return false;
    }
  }
  return true;
}

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (true) {
    const size_t tab = line.find('\t', start);
    if (tab == std::string::npos) {
      fields.push_back(line.substr(start));
      return fields;
    }
    fields.push_back(line.substr(start, tab - start));
    start = tab + 1;
  }
}

bool LoadWorkload(const std::string& path, std::vector<Family>* families, std::string* error) {
  std::ifstream in(path);
  if (!in) {
    *error = "cannot open workload file " + path;
    return false;
  }
  std::map<std::string, size_t> index;  // family id -> position, keeping first-seen order
  std::string line;
  size_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    const auto f = SplitTabs(line);
    if (f.size() != 4) {
      *error = path + ":" + std::to_string(line_no) + ": expected 4 tab-separated columns, got " +
               std::to_string(f.size());
      return false;
    }
    auto it = index.find(f[0]);
    if (it == index.end()) {
      it = index.emplace(f[0], families->size()).first;
      families->push_back(Family{});
      families->back().id = f[0];
    }
    Family& fam = (*families)[it->second];
    if (f[1] == "threshold") {
      fam.threshold = std::stof(f[3]);
    } else if (f[1] == "item") {
      fam.items.push_back(Item{f[2], f[3]});
    } else if (f[1] == "query") {
      fam.queries.push_back(Query{f[2] == "-" ? std::string() : f[2], f[3]});
    } else {
      *error = path + ":" + std::to_string(line_no) + ": unknown kind " + f[1];
      return false;
    }
  }
  if (families->empty()) {
    *error = path + ": no families";
    return false;
  }
  return true;
}

std::string RandomSuffix() {
  std::random_device rd;
  std::uniform_int_distribution<int> dis(0, 999999);
  return std::to_string(dis(rd));
}

// Removes the temporary store directory when the demo ends, after the store has been closed.
struct TempDir {
  std::filesystem::path path;
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

const char* VerdictName(prestige::OutcomeVerdict v) {
  switch (v) {
    case prestige::OutcomeVerdict::kAccepted: return "accepted";
    case prestige::OutcomeVerdict::kRejected: return "false accept";
    case prestige::OutcomeVerdict::kNoCandidate: return "no candidate";
  }
  return "?";
}

std::string Fixed(double v, int digits) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(digits) << v;
  return os.str();
}

// Judge one query: the similarity gate, then the constraint check on every candidate it let through, each verdict
// recorded under the family id. Returns true when a candidate was accepted.
bool JudgeQuery(prestige::Store* db, const Family& fam, const Query& query, bool trace) {
  prestige::CandidateQuery q;
  q.k = kCandidatesPerQuery;
  q.filter = {{"family", fam.id}};  // only values written for this family
  std::vector<prestige::Candidate> cands;
  rocksdb::Status s = db->Candidates(query.text, q, &cands);
  if (!s.ok()) {
    std::cerr << "Candidates failed: " << s.ToString() << "\n";
    std::exit(1);
  }

  bool served = false;
  bool any_above_gate = false;
  std::string trace_line;
  for (const auto& c : cands) {
    if (c.similarity < fam.threshold) break;  // the similarity gate; candidates are ranked by similarity
    any_above_gate = true;

    prestige::Outcome o;
    o.family_id = fam.id;
    o.candidate_object_id = c.object_id;
    o.candidate_digest = c.digest;
    o.rank = static_cast<uint32_t>(c.rank);
    o.similarity = c.similarity;
    o.reranker_score = c.reranker_score;
    o.threshold = fam.threshold;
    o.tool_version = kToolVersion;
    o.schema_version = kSchemaVersion;

    // The constraint check. A real caller tests what the embedding cannot see (tenant, schema version, side
    // effects); the demo tests the one constraint it knows, the ground-truth item written as metadata.
    const auto item = c.metadata.find("item");
    const std::string candidate_item = item == c.metadata.end() ? std::string("?") : item->second;
    if (!query.expected.empty() && candidate_item == query.expected) {
      o.verdict = prestige::OutcomeVerdict::kAccepted;
    } else {
      o.verdict = prestige::OutcomeVerdict::kRejected;
      o.reason = query.expected.empty() ? "no stored item answers this query; candidate " + candidate_item
                                        : "expected " + query.expected + "; candidate " + candidate_item;
    }
    s = db->RecordOutcome(o);
    if (!s.ok()) {
      std::cerr << "RecordOutcome failed: " << s.ToString() << "\n";
      std::exit(1);
    }
    if (trace) {
      trace_line += "    rank " + std::to_string(c.rank) + "  " + candidate_item + "  sim " +
                    Fixed(c.similarity, 3) + "  " + VerdictName(o.verdict) + "\n";
    }
    if (o.verdict == prestige::OutcomeVerdict::kAccepted) {
      served = true;
      break;  // the caller reuses this candidate; lower ranks are never judged
    }
  }

  if (!any_above_gate) {
    prestige::Outcome o;
    o.family_id = fam.id;
    o.verdict = prestige::OutcomeVerdict::kNoCandidate;
    o.threshold = fam.threshold;
    o.tool_version = kToolVersion;
    o.schema_version = kSchemaVersion;
    s = db->RecordOutcome(o);
    if (!s.ok()) {
      std::cerr << "RecordOutcome failed: " << s.ToString() << "\n";
      std::exit(1);
    }
    if (trace) {
      std::string best = "nothing returned";
      if (!cands.empty()) {
        const auto item = cands[0].metadata.find("item");
        best = "best " + (item == cands[0].metadata.end() ? std::string("?") : item->second) + " sim " +
               Fixed(cands[0].similarity, 3);
      }
      trace_line = "    no candidate above " + Fixed(fam.threshold, 2) + " (" + best + ")\n";
    }
  }

  if (trace) {
    std::cout << "  [" << fam.id << "] \"" << query.text << "\" expects "
              << (query.expected.empty() ? std::string("nothing") : query.expected) << "\n"
              << trace_line;
  }
  return served;
}

void PrintSummary(const std::vector<Family>& families, const std::map<std::string, prestige::FamilyReport>& reports,
                  const std::map<std::string, Tally>& tallies) {
  std::cout << std::left << std::setw(17) << "family" << std::right << std::setw(9) << "threshold" << std::setw(8)
            << "queries" << std::setw(7) << "served" << std::setw(9) << "outcomes" << std::setw(9) << "accepted"
            << std::setw(14) << "false_accepts" << std::setw(13) << "no_candidate" << std::setw(8) << "fa_rate"
            << std::setw(10) << "suggested" << "\n";
  for (const auto& fam : families) {
    const auto& r = reports.at(fam.id);
    const auto& t = tallies.at(fam.id);
    std::cout << std::left << std::setw(17) << fam.id << std::right << std::setw(9) << Fixed(fam.threshold, 2)
              << std::setw(8) << t.queries << std::setw(7) << t.served << std::setw(9)
              << (r.accepted + r.rejected + r.no_candidate) << std::setw(9) << r.accepted << std::setw(14)
              << r.rejected << std::setw(13) << r.no_candidate << std::setw(8) << Fixed(r.false_accept_rate, 3)
              << std::setw(10)
              << (r.suggested_threshold < 0.0f ? std::string("none") : Fixed(r.suggested_threshold, 2)) << "\n";
  }
}

void PrintDistributions(const Family& fam, const prestige::FamilyReport& r) {
  std::cout << "\n" << fam.id << ": " << (r.accepted + r.rejected) << " judged candidates, " << r.rejected
            << " false accepts (rate " << Fixed(r.false_accept_rate, 3) << "), suggested threshold "
            << (r.suggested_threshold < 0.0f ? std::string("none (insufficient evidence)")
                                             : Fixed(r.suggested_threshold, 2))
            << "\n";
  std::cout << "  " << std::left << std::setw(14) << "similarity" << std::right << std::setw(9) << "accepted"
            << std::setw(15) << "false_accepts" << "\n";
  bool any = false;
  for (int b = 19; b >= 0; --b) {
    const uint64_t acc = r.accepted_similarity_hist[static_cast<size_t>(b)];
    const uint64_t rej = r.rejected_similarity_hist[static_cast<size_t>(b)];
    if (acc == 0 && rej == 0) continue;
    any = true;
    const std::string label =
        "[" + Fixed(b / 20.0, 2) + ", " + Fixed((b + 1) / 20.0, 2) + (b == 19 ? "]" : ")");
    std::cout << "  " << std::left << std::setw(14) << label << std::right << std::setw(9) << acc << std::setw(15)
              << rej << "\n";
  }
  if (!any) std::cout << "  (no judged candidates)\n";
  std::cout << "  false accepts by rank:";
  bool any_rank = false;
  for (size_t rank = 0; rank < r.rejected_rank_hist.size(); ++rank) {
    if (r.rejected_rank_hist[rank] == 0) continue;
    any_rank = true;
    std::cout << " rank " << rank << (rank == 15 ? "+" : "") << ": " << r.rejected_rank_hist[rank];
  }
  std::cout << (any_rank ? "" : " none") << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) return 1;

  std::vector<Family> families;
  std::string error;
  if (!LoadWorkload(args.workload, &families, &error)) {
    std::cerr << error << "\n";
    return 1;
  }

  // The store lives in a temporary directory and is removed at the end (TempDir outlives the store).
  TempDir tmp{std::filesystem::temp_directory_path() / ("prestige_outcomes_demo_" + RandomSuffix())};
  std::filesystem::create_directories(tmp.path);

  prestige::Options opt;
  std::string mode = "exact";
#ifdef PRESTIGE_ENABLE_SEMANTIC
  if (!args.model_path.empty()) {
    opt.dedup_mode = prestige::DedupMode::kSemantic;
    opt.semantic_model_path = args.model_path;
    opt.semantic_model_type =
        args.model_type == "minilm" ? prestige::SemanticModel::kMiniLM : prestige::SemanticModel::kBGESmall;
    // Put() merges nothing: every demo value stays its own object, so item identity is unambiguous. The only
    // threshold under test is the caller's, applied to the Candidates() list below.
    opt.semantic_threshold = 1.0f;
    opt.semantic_device = prestige::SemanticDevice::kCPU;  // deterministic
    opt.semantic_num_threads = 1;
    opt.semantic_index_save_interval = 0;
    mode = "semantic";
  }
#else
  if (!args.model_path.empty()) {
    std::cerr << "This binary was built without PRESTIGE_ENABLE_SEMANTIC; --model is ignored.\n";
  }
#endif

  std::unique_ptr<prestige::Store> db;
  rocksdb::Status s = prestige::Store::Open((tmp.path / "db").string(), &db, opt);
  if (!s.ok()) {
    std::cerr << "Open failed: " << s.ToString() << "\n";
    return 1;
  }

  std::cout << "prestige outcomes demo\n";
  if (mode == "semantic") {
    std::cout << "mode: semantic, cosine similarity from " << args.model_type << " (" << args.model_path
              << "), CPU, 1 thread\n";
  } else {
    std::cout << "mode: exact. Similarity is EXACT-ONLY: Candidates() returns the identical value at 1.0 or "
                 "nothing,\n      so only verbatim repeats are served, paraphrases record no_candidate and a false "
                 "accept\n      cannot happen. Build with PRESTIGE_ENABLE_SEMANTIC=ON and pass --model for real "
                 "numbers.\n";
  }

  // Store every item with its family and ground-truth item id as value metadata.
  uint64_t stored = 0;
  size_t total_queries = 0;
  for (const auto& fam : families) {
    for (const auto& item : fam.items) {
      s = db->Put(fam.id + "/" + item.id, item.text,
                  prestige::Metadata{{"family", fam.id}, {"item", item.id}, {"tool", kToolVersion},
                                     {"schema", kSchemaVersion}});
      if (!s.ok()) {
        std::cerr << "Put failed: " << s.ToString() << "\n";
        return 1;
      }
      ++stored;
    }
    total_queries += fam.queries.size();
  }
  uint64_t objects = 0;
  s = db->CountUniqueValues(&objects);
  if (!s.ok()) {
    std::cerr << "CountUniqueValues failed: " << s.ToString() << "\n";
    return 1;
  }
  if (objects != stored) {
    std::cerr << "the store merged " << (stored - objects) << " of " << stored
              << " values on Put; item identity would be ambiguous\n";
    return 1;
  }
  std::cout << "workload: " << args.workload << ": " << families.size() << " families, " << stored
            << " stored values (" << objects << " distinct objects), " << total_queries << " queries\n";
  std::cout << "gate: candidates at or above the family threshold are checked in rank order until one passes\n";
  if (args.trace) std::cout << "\n";

  std::map<std::string, Tally> tallies;
  for (const auto& fam : families) {
    Tally& t = tallies[fam.id];
    for (const auto& query : fam.queries) {
      ++t.queries;
      if (JudgeQuery(db.get(), fam, query, args.trace)) ++t.served;
    }
  }

  // The store aggregates per family; the caller reads the reports and recalibrates its thresholds from them.
  std::vector<std::string> family_ids;
  s = db->ListFamilies(&family_ids);
  if (!s.ok()) {
    std::cerr << "ListFamilies failed: " << s.ToString() << "\n";
    return 1;
  }
  std::map<std::string, prestige::FamilyReport> reports;
  for (const auto& id : family_ids) {
    prestige::FamilyReport r;
    s = db->GetFamilyReport(id, &r);
    if (!s.ok()) {
      std::cerr << "GetFamilyReport(" << id << ") failed: " << s.ToString() << "\n";
      return 1;
    }
    reports[id] = std::move(r);
  }

  std::cout << "\nper-family report (" << family_ids.size() << " families with outcomes)\n";
  PrintSummary(families, reports, tallies);
  for (const auto& fam : families) PrintDistributions(fam, reports.at(fam.id));

  std::cout << "\nfa_rate = false_accepts / (accepted + false_accepts). suggested is advisory: the lowest 0.05 "
               "bucket edge\nwith at least 20 judged candidates above it and at most 5% false accepts among them; "
               "none when no\nedge qualifies. The store never changes a threshold; the caller recalibrates per "
               "family.\n";
  return 0;
}
