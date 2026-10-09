#!/usr/bin/env python3
"""Outcome records demo through the Python bindings: per-family false-accept reports from a synthetic workload.

The store ranks and the caller decides (docs/candidates.md). This script is the Python twin of
examples/outcomes.cpp: it opens a temporary store, loads examples/outcomes_workload.tsv (stored values carrying a
ground-truth item id, and queries labelled with the item each one should be served by) and, for every query,

  1. asks candidates() for the nearest stored values of the family,
  2. applies the family's similarity threshold (the similarity gate),
  3. runs the caller's constraint check on each candidate above the gate, in rank order: does the candidate's
     "item" metadata name the item the query expects? The ground-truth label stands in for the tenant, schema or
     side-effect checks a real caller would run,
  4. records every verdict with record_outcome(): "accepted" when the check passed, "rejected" when the gate let a
     different item through (a false accept of the similarity gate), "no_candidate" when nothing was above the gate.

It ends with family_report() for every family. Same data, same gate and same output format as the C++ example, so
the two outputs can be diffed.

Semantic mode needs a build with PRESTIGE_ENABLE_SEMANTIC=ON (docs/outcomes-demo.md) and an ONNX model with
vocab.txt next to it:

    python python/examples/outcomes_demo.py --model models/bge-small-en-v1.5_onnx/model.onnx

Without --model, or with the PyPI wheel (built without semantic mode), the store runs in exact mode: candidates()
returns the identical value at similarity 1.0 or nothing, so only verbatim repeats are served and a false accept
cannot happen. The output says so.
"""

import argparse
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import prestige

TOOL_VERSION = "outcomes-demo/1"
SCHEMA_VERSION = "ground-truth@1"
CANDIDATES_PER_QUERY = 5
DEFAULT_WORKLOAD = Path(__file__).resolve().parents[2] / "examples" / "outcomes_workload.tsv"
VERDICT_NAMES = {"accepted": "accepted", "rejected": "false accept", "no_candidate": "no candidate"}


def as_float32(x):
    """The C++ example compares float similarities against a float threshold; do the same here."""
    return struct.unpack("f", struct.pack("f", x))[0]


def load_workload(path):
    families = {}  # family id -> dict, in first-seen order
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 4:
                sys.exit(f"{path}:{line_no}: expected 4 tab-separated columns, got {len(fields)}")
            family, kind, item, text = fields
            fam = families.setdefault(family, {"id": family, "threshold": 0.85, "items": [], "queries": []})
            if kind == "threshold":
                fam["threshold"] = float(text)
            elif kind == "item":
                fam["items"].append((item, text))
            elif kind == "query":
                fam["queries"].append(("" if item == "-" else item, text))
            else:
                sys.exit(f"{path}:{line_no}: unknown kind {kind}")
    if not families:
        sys.exit(f"{path}: no families")
    return list(families.values())


def judge_query(store, fam, expected, text, trace):
    """The similarity gate, then the constraint check on every candidate it let through, each verdict recorded
    under the family id. Returns True when a candidate was accepted."""
    threshold = as_float32(fam["threshold"])
    cands = store.candidates(text, k=CANDIDATES_PER_QUERY, filter={"family": fam["id"]})
    served = False
    any_above_gate = False
    lines = []
    for c in cands:
        if c["similarity"] < threshold:
            break  # the similarity gate; candidates are ranked by similarity
        any_above_gate = True
        # The constraint check: a real caller tests what the embedding cannot see (tenant, schema version, side
        # effects); the demo tests the one constraint it knows, the ground-truth item written as metadata.
        candidate_item = c["metadata"].get("item", "?")
        if expected and candidate_item == expected:
            verdict, reason = "accepted", ""
        else:
            verdict = "rejected"
            reason = (f"expected {expected}; candidate {candidate_item}" if expected
                      else f"no stored item answers this query; candidate {candidate_item}")
        # A dict from candidates() supplies the object id, digest, rank and scores.
        store.record_outcome(fam["id"], verdict, candidate=c, threshold=fam["threshold"],
                             tool_version=TOOL_VERSION, schema_version=SCHEMA_VERSION, reason=reason)
        if trace:
            lines.append(f"    rank {c['rank']}  {candidate_item}  sim {c['similarity']:.3f}  "
                         f"{VERDICT_NAMES[verdict]}")
        if verdict == "accepted":
            served = True
            break  # the caller reuses this candidate; lower ranks are never judged

    if not any_above_gate:
        store.record_outcome(fam["id"], "no_candidate", threshold=fam["threshold"],
                             tool_version=TOOL_VERSION, schema_version=SCHEMA_VERSION)
        if trace:
            if cands:
                best = f"best {cands[0]['metadata'].get('item', '?')} sim {cands[0]['similarity']:.3f}"
            else:
                best = "nothing returned"
            lines = [f"    no candidate above {fam['threshold']:.2f} ({best})"]

    if trace:
        print(f"  [{fam['id']}] \"{text}\" expects {expected or 'nothing'}")
        for line in lines:
            print(line)
    return served


def print_summary(families, reports, tallies):
    print(f"{'family':<17}{'threshold':>9}{'queries':>8}{'served':>7}{'outcomes':>9}{'accepted':>9}"
          f"{'false_accepts':>14}{'no_candidate':>13}{'fa_rate':>8}{'suggested':>10}")
    for fam in families:
        r = reports[fam["id"]]
        t = tallies[fam["id"]]
        suggested = "none" if r["suggested_threshold"] is None else f"{r['suggested_threshold']:.2f}"
        print(f"{fam['id']:<17}{fam['threshold']:>9.2f}{t['queries']:>8}{t['served']:>7}"
              f"{r['accepted'] + r['rejected'] + r['no_candidate']:>9}{r['accepted']:>9}{r['rejected']:>14}"
              f"{r['no_candidate']:>13}{r['false_accept_rate']:>8.3f}{suggested:>10}")


def print_distributions(fam, r):
    suggested = ("none (insufficient evidence)" if r["suggested_threshold"] is None
                 else f"{r['suggested_threshold']:.2f}")
    print(f"\n{fam['id']}: {r['accepted'] + r['rejected']} judged candidates, {r['rejected']} false accepts "
          f"(rate {r['false_accept_rate']:.3f}), suggested threshold {suggested}")
    print(f"  {'similarity':<14}{'accepted':>9}{'false_accepts':>15}")
    any_bucket = False
    for b in range(19, -1, -1):
        acc = r["accepted_similarity_hist"][b]
        rej = r["rejected_similarity_hist"][b]
        if acc == 0 and rej == 0:
            continue
        any_bucket = True
        label = f"[{b / 20:.2f}, {(b + 1) / 20:.2f}{']' if b == 19 else ')'}"
        print(f"  {label:<14}{acc:>9}{rej:>15}")
    if not any_bucket:
        print("  (no judged candidates)")
    ranks = "".join(f" rank {i}{'+' if i == 15 else ''}: {n}" for i, n in enumerate(r["rejected_rank_hist"]) if n)
    print("  false accepts by rank:" + (ranks or " none"))


def main():
    parser = argparse.ArgumentParser(description="Per-family false-accept reports from a synthetic workload.")
    parser.add_argument("--model", help="ONNX embedding model (vocab.txt next to it); omit for exact mode")
    parser.add_argument("--model-type", choices=["bge-small", "minilm"], default="bge-small")
    parser.add_argument("--workload", default=str(DEFAULT_WORKLOAD))
    parser.add_argument("--trace", action="store_true", help="print every query and verdict")
    args = parser.parse_args()

    families = load_workload(args.workload)

    opts = prestige.Options()
    mode = "exact"
    if args.model and prestige.SEMANTIC_AVAILABLE:
        opts.dedup_mode = prestige.DedupMode.SEMANTIC
        opts.semantic_model_path = args.model
        opts.semantic_model_type = (prestige.SemanticModel.MINILM if args.model_type == "minilm"
                                    else prestige.SemanticModel.BGE_SMALL)
        # put() merges nothing: every demo value stays its own object, so item identity is unambiguous. The only
        # threshold under test is the caller's, applied to the candidates() list in judge_query().
        opts.semantic_threshold = 1.0
        opts.semantic_device = prestige.SemanticDevice.CPU  # deterministic
        opts.semantic_num_threads = 1
        opts.semantic_index_save_interval = 0
        mode = "semantic"
    elif args.model:
        print("This prestige build has no semantic mode; --model is ignored.", file=sys.stderr)

    tmp = tempfile.mkdtemp(prefix="prestige_outcomes_demo_")
    try:
        with prestige.open(str(Path(tmp) / "db"), opts) as store:
            print("prestige outcomes demo")
            if mode == "semantic":
                print(f"mode: semantic, cosine similarity from {args.model_type} ({args.model}), CPU, 1 thread")
            else:
                print("mode: exact. Similarity is EXACT-ONLY: Candidates() returns the identical value at 1.0 or "
                      "nothing,\n      so only verbatim repeats are served, paraphrases record no_candidate and a "
                      "false accept\n      cannot happen. Build with PRESTIGE_ENABLE_SEMANTIC=ON and pass --model "
                      "for real numbers.")

            # Store every item with its family and ground-truth item id as value metadata.
            stored = 0
            total_queries = 0
            for fam in families:
                for item, text in fam["items"]:
                    store.put(f"{fam['id']}/{item}", text, metadata={"family": fam["id"], "item": item,
                                                                     "tool": TOOL_VERSION, "schema": SCHEMA_VERSION})
                    stored += 1
                total_queries += len(fam["queries"])
            objects = store.count_unique_values()
            if objects != stored:
                sys.exit(f"the store merged {stored - objects} of {stored} values on put; "
                         "item identity would be ambiguous")
            print(f"workload: {args.workload}: {len(families)} families, {stored} stored values "
                  f"({objects} distinct objects), {total_queries} queries")
            print("gate: candidates at or above the family threshold are checked in rank order until one passes")
            if args.trace:
                print()

            tallies = {}
            for fam in families:
                t = tallies.setdefault(fam["id"], {"queries": 0, "served": 0})
                for expected, text in fam["queries"]:
                    t["queries"] += 1
                    if judge_query(store, fam, expected, text, args.trace):
                        t["served"] += 1

            # The store aggregates per family; the caller reads the reports and recalibrates its thresholds.
            family_ids = store.list_families()
            reports = {fid: store.family_report(fid) for fid in family_ids}

            print(f"\nper-family report ({len(family_ids)} families with outcomes)")
            print_summary(families, reports, tallies)
            for fam in families:
                print_distributions(fam, reports[fam["id"]])
            print("\nfa_rate = false_accepts / (accepted + false_accepts). suggested is advisory: the lowest 0.05 "
                  "bucket edge\nwith at least 20 judged candidates above it and at most 5% false accepts among "
                  "them; none when no\nedge qualifies. The store never changes a threshold; the caller "
                  "recalibrates per family.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
