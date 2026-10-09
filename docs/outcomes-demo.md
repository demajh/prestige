# Outcomes demo: per-family false-accept reports

`examples/outcomes.cpp` (and its Python twin, `python/examples/outcomes_demo.py`) runs the loop described in
[candidates.md](candidates.md) end to end: the store ranks candidates, the caller applies its similarity gate and
its constraint check, every verdict is recorded under a versioned task-family id, and the store reports the
false-accept rate, the similarity and rank distributions of the failures and an advisory threshold per family.
The workload is synthetic but labelled, so every number below has a known ground truth behind it.

## What the example does

The workload lives in `examples/outcomes_workload.tsv`, one tab-separated line per entry
(`family`, `kind`, `item`, `text`). It defines three task families, each with twelve stored values and 54 to 56
queries:

| Family | Stored values | Queries |
|---|---|---|
| `support-faq@v2` | canonical support questions (`pw-reset-web`, `invoice-vat`, ...) | incoming user questions |
| `code-search@v1` | one-line descriptions of cached snippets (`lru-cache`, `debounce`, ...) | developer searches |
| `ticket-dedup@v1` | open incident tickets (`T-101` ... `T-112`) | new tickets that may duplicate one |

Each family's queries fall into five groups: verbatim repeats, near-duplicates (the same text with small edits,
the bulk of real cache traffic), paraphrases, same-topic-different-constraint queries (a password reset *in the
Android app* when the stored item covers the mobile app; a 502 *in the APAC region* when the open tickets cover the
EU and US regions; `LFU cache` next to a stored `LRU cache`) and unrelated text. Every query is labelled with the
stored item that should serve it, or `-` when no stored item does. That label is the ground truth; in the example
it plays the part of the tenant, schema or side-effect check a real caller would run, because it is the one
constraint the demo knows about and the embedding does not.

For every family the program:

1. stores each value with `Put(key, text, Metadata{{"family", ...}, {"item", ...}, {"tool", ...}, {"schema", ...}})`;
2. for every query calls `Candidates()` with `k = 5` and `filter = {{"family", <family>}}`, so only values written
   for that family are ranked;
3. applies the family's similarity threshold, the **similarity gate**, and walks the candidates at or above it in
   rank order;
4. runs the **constraint check** on each: does the candidate's `item` metadata name the item the query expects?
   A pass is recorded as `kAccepted` and the walk stops (the caller reuses that value). A failure is recorded as
   `kRejected` with the candidate's rank, similarity, the threshold applied and a reason: the gate said
   "duplicate" and the check said "different item", which is a **false accept of the similarity gate**. When no
   candidate reaches the gate at all, `kNoCandidate` is recorded and a fresh computation would follow;
5. ends with `ListFamilies()` and `GetFamilyReport()` for each family, printed as a plain table.

Two settings matter for reading the numbers:

- Every family starts at a deliberately loose threshold of **0.70**. A gate only produces evidence above itself:
  the report sees the candidates the gate let through and nothing below it. Started at 0.85 instead, the same
  workload judges 31, 34 and 38 candidates per family: `code-search@v1` records no false accept at all and gets no
  suggestion, `support-faq@v2` records one and is advised 0.85, its own gate, because the advice never falls below
  the lowest bucket in which a candidate was judged, and only `ticket-dedup@v1` still answers 0.95. Starting loose
  is the learning phase; the report then says where the gate belongs, and every outcome carries the threshold that
  was applied when it was recorded.
- The store's own dedup threshold (`semantic_threshold`) is set to **1.0**, so `Put` merges nothing and each of
  the 36 values stays its own object with its own `item`. The only threshold under test is the caller's, applied
  to the `Candidates()` list. (In exact mode the question does not arise.)

The run is deterministic: fixed data, no random numbers, CPU inference on one thread. Two runs produce identical
output, and `--trace`, which prints every query with its candidates and verdicts, changes nothing in the summary.

## Build and run

Semantic mode needs ONNX Runtime and an embedding model with its `vocab.txt` next to it. On this machine
(macOS, Apple Silicon, Homebrew) the following worked, starting from the repository root:

```bash
brew install rocksdb onnxruntime

mkdir -p models/bge-small-en-v1.5_onnx && cd models/bge-small-en-v1.5_onnx
curl -LO https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model.onnx
curl -LO https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/vocab.txt
cd ../..

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPRESTIGE_BUILD_TESTS=OFF \
  -DPRESTIGE_ENABLE_SEMANTIC=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/opt/homebrew/include/onnxruntime \
  -DONNXRUNTIME_LIBRARY=/opt/homebrew/lib/libonnxruntime.dylib \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/rocksdb -DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@3
cmake --build build --target prestige_example_outcomes -j

./build/prestige_example_outcomes --model models/bge-small-en-v1.5_onnx/model.onnx \
  --workload examples/outcomes_workload.tsv
```

`models/*_onnx/`, `build/` and `cmake-build-*/` are ignored by git. Options: `--trace` prints every query, candidate and verdict;
`--model-type minilm` switches the tokenizer/prefix handling for an `all-MiniLM-L6-v2` export; `--workload` points
at another file in the same format, which is the easy way to run the loop over your own data.

Without `--model`, or in a build without `PRESTIGE_ENABLE_SEMANTIC`, the example still builds and runs, in exact
mode. It then says so in its second line, because the numbers mean something different: `Candidates()` returns the
identical value at similarity 1.0 or nothing, so only the two verbatim repeats per family are served, every
paraphrase records `no_candidate`, no false accept is possible and the suggested threshold is `none`.

### Python

The Python twin uses `put(metadata=...)`, `candidates()`, `record_outcome()`, `list_families()` and
`family_report()` and prints the same format, so the two outputs can be diffed. It needs a source build of the
bindings with semantic mode (the PyPI wheel is exact-only):

```bash
python3 -m venv .venv && .venv/bin/pip install pybind11
cmake -S . -B cmake-build-python -DCMAKE_BUILD_TYPE=Release -DPRESTIGE_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE=$PWD/.venv/bin/python -DPRESTIGE_BUILD_TESTS=OFF -DPRESTIGE_BUILD_EXAMPLES=OFF \
  -DPRESTIGE_ENABLE_SEMANTIC=ON -DONNXRUNTIME_INCLUDE_DIR=/opt/homebrew/include/onnxruntime \
  -DONNXRUNTIME_LIBRARY=/opt/homebrew/lib/libonnxruntime.dylib \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/rocksdb -DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@3
cmake --build cmake-build-python --target _prestige -j      # writes python/prestige/_prestige.*.so

PYTHONPATH=python .venv/bin/python python/examples/outcomes_demo.py \
  --model models/bge-small-en-v1.5_onnx/model.onnx --workload examples/outcomes_workload.tsv
```

On this machine its output is byte-for-byte the C++ output below.

## Output on this machine

macOS 26.5 (Apple Silicon), ONNX Runtime 1.30.0 from Homebrew, `Xenova/bge-small-en-v1.5` fp32 ONNX export, mean
pooling, CPU, one thread. Similarities are properties of the model and pooling, so another model, or another
ONNX Runtime build, will move them; the method is what transfers.

```
prestige outcomes demo
mode: semantic, cosine similarity from bge-small (models/bge-small-en-v1.5_onnx/model.onnx), CPU, 1 thread
workload: examples/outcomes_workload.tsv: 3 families, 36 stored values (36 distinct objects), 166 queries
gate: candidates at or above the family threshold are checked in rank order until one passes

per-family report (3 families with outcomes)
family           threshold queries served outcomes accepted false_accepts no_candidate fa_rate suggested
support-faq@v2        0.70      54     39       55       39             8            8   0.170      0.85
code-search@v1        0.70      56     41       57       41             8            8   0.163      0.80
ticket-dedup@v1       0.70      56     44       58       44            10            4   0.185      0.95

support-faq@v2: 47 judged candidates, 8 false accepts (rate 0.170), suggested threshold 0.85
  similarity     accepted  false_accepts
  [0.95, 1.00]         27              0
  [0.90, 0.95)          1              0
  [0.85, 0.90)          2              1
  [0.80, 0.85)          3              1
  [0.75, 0.80)          2              2
  [0.70, 0.75)          4              4
  false accepts by rank: rank 0: 7 rank 1: 1

code-search@v1: 49 judged candidates, 8 false accepts (rate 0.163), suggested threshold 0.80
  similarity     accepted  false_accepts
  [0.95, 1.00]         23              0
  [0.90, 0.95)          7              0
  [0.85, 0.90)          4              0
  [0.80, 0.85)          4              2
  [0.75, 0.80)          3              4
  [0.70, 0.75)          0              2
  false accepts by rank: rank 0: 8

ticket-dedup@v1: 54 judged candidates, 10 false accepts (rate 0.185), suggested threshold 0.95
  similarity     accepted  false_accepts
  [0.95, 1.00]         27              0
  [0.90, 0.95)          3              3
  [0.85, 0.90)          4              1
  [0.80, 0.85)          7              2
  [0.75, 0.80)          2              1
  [0.70, 0.75)          1              3
  false accepts by rank: rank 0: 8 rank 1: 2

fa_rate = false_accepts / (accepted + false_accepts). suggested is advisory: the lowest 0.05 bucket edge
with at least 20 judged candidates above it and at most 5% false accepts among them; none when no
edge qualifies. The store never changes a threshold; the caller recalibrates per family.
```

Some of the false accepts, from `--trace` (query, then the candidate the gate let through):

| Family | Query | Candidate | Similarity |
|---|---|---|---|
| `ticket-dedup@v1` | Login page returns 502 Bad Gateway for users in the APAC region. | `T-102` (US region), then `T-101` (EU region) at rank 1 | 0.934, 0.912 |
| `ticket-dedup@v1` | Export to CSV cuts off the last column of wide tables. | `T-111` (export to PDF) | 0.923 |
| `ticket-dedup@v1` | Login page returns 504 Gateway Timeout for users in the EU region. | `T-101` (502 Bad Gateway) | 0.890 |
| `support-faq@v2` | How do I export my data as JSON instead of CSV? | `export-data` (as CSV) | 0.851 |
| `code-search@v1` | LFU cache with fixed capacity | `lru-cache` | 0.828 |
| `code-search@v1` | check if a number is a palindrome | `is-palindrome` (for strings) | 0.826 |
| `support-faq@v2` | Where do I rotate my API key? | `api-key` (where to find it) | 0.804 |
| `code-search@v1` | throttle a function to run at most once per interval | `debounce` | 0.715 |

## How to read the numbers

**Columns.** `queries` is the number of queries issued for the family and `served` how many ended in an accepted
candidate; the difference is the number of fresh computations. `outcomes` is the number of records the store
holds for the family (`accepted + false_accepts + no_candidate`). It can exceed `queries`: a query whose top
candidate fails the constraint check has its next candidate above the gate judged too, so one query can produce
two records (the APAC ticket above produced two false accepts, at ranks 0 and 1). `no_candidate` counts only
queries where nothing reached the gate; the other unserved queries were caught by the constraint check, and the
false-accept records are their trace.

**What a false accept is.** The similarity gate is a claim: "a candidate at or above the threshold is the same
item". A false accept is one of those claims the constraint check refuted. It is a property of the *gate*, not of
the candidate list: of the 26 false accepts recorded, 25 come from queries that have no correct answer in the
store but still look, to the embedding, like something that is there. The one exception, `turn a list of lists
into one list`, was served by `flatten-list` at rank 1 after `merge-sorted` at rank 0 (0.768 against 0.766) and
is the only judged query whose expected item was not at rank 0. `fa_rate` is
`false_accepts / (accepted + false_accepts)`: the share of the gate's claims that were wrong. Accepts are recorded
as well as rejects precisely so that this rate has a denominator.

**Why the three rates differ.** Same model, same starting gate, three distributions. `support-faq@v2`'s false
accepts sit between 0.70 and 0.85, so the family separates cleanly above 0.85. `ticket-dedup@v1`'s sit as high as
0.934: tickets that differ only in a region, a status code or an export format are near-identical sentences, and
the embedding sees a near-identical sentence. No similarity threshold short of "verbatim" separates them, which is
exactly the case for carrying the region or the format as metadata and filtering or checking on it rather than
hoping the score will. That is why the report is per family and the numbers of one family say nothing about
another.

**The suggested threshold.** `GetFamilyReport()` walks the 0.05 similarity buckets from the top, accumulating
accepted and rejected candidates, and reports the lowest bucket edge above which at least 20 candidates were
judged and at most 5% of them were false accepts; `none` when no edge qualifies or no rejection has been recorded.
From the histograms above:

- `support-faq@v2`: above 0.85, 30 accepted and 1 false accept (3.2%); above 0.80, 33 and 2 (5.7%). Suggested
  0.85.
- `code-search@v1`: above 0.80, 38 accepted and 2 false accepts, 5.0% exactly; above 0.75, 41 and 6 (12.8%).
  Suggested 0.80, sitting right on the line: one more false accept in `[0.80, 0.85)` would move it to 0.85.
- `ticket-dedup@v1`: above 0.90, 30 accepted and 3 false accepts (9.1%); only `[0.95, 1.00]` is clean. Suggested
  0.95, which in practice means "serve near-verbatim repeats from the cache and nothing else" for this family.

**Why it is advisory.** The store knows the counts; it does not know what a false accept costs against a miss
for this family, and the sample is small enough that one record moves the answer (see `code-search@v1`). The 20
and 5% are conventions chosen so that a suggestion is never made from a handful of records. The caller also
weighs coverage, which the report does not: at the suggested 0.85, `support-faq@v2` would serve its 30
near-verbatim queries and recompute the nine paraphrases it currently serves between 0.70 and 0.85. Whether that
trade is right depends on what a wrong answer costs, so the number is a suggestion and the decision stays with
the caller. The store never changes a threshold on its own, and the threshold the caller applied is recorded
with every outcome, so a recalibration is auditable after the fact. One more reason to read it as advice: the
report only sees what the gate let through, so a suggestion equal to the threshold that was applied (what a 0.85
start produces for `support-faq@v2`) says nothing about the region below the gate, where no candidate was ever
judged; the advice is floored at the lowest judged bucket for exactly that reason.

**Why per family, never per pair.** A threshold tuned per stored value, or per (query, candidate) pair, fits the
noise of individual sentences and cannot be explained or transferred; the unit over which a similarity
distribution is stable is the task family: an explicit, versioned id (`support-faq@v2`), with the tool and
constraint-schema versions carried as metadata. Outcomes are aggregated at that granularity, thresholds are read
back at that granularity, and when the tool or the schema changes the family id changes with it and calibration
starts again with a clean denominator rather than mixing two distributions under one name.

**Exact mode.** The same program built without semantic mode, or run without `--model`, reports two accepted
candidates per family (the verbatim repeats), 52 to 54 `no_candidate`, a rate of 0.000 and `none` for every
suggestion, under a header that says similarity is exact-only. Those numbers are correct for exact matching and
useless for calibration, which is why the program says so rather than printing them quietly.
