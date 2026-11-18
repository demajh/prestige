#!/usr/bin/env python3
"""Run the text-normalisation fixture against the library and print pass/fail per row.

The table is examples/normalization_fixture.tsv (columns and reading notes in docs/normalization-fixture.md). Every
row is one input under one normalisation mode, with the bytes the dedup key is computed over, the key class the row
belongs to, the bytes the store holds for the row's key, and the raw byte span of every code point of the input.

For every row the script checks, in this order:

  spans   the spans tile the input bytes exactly, each span is the UTF-8 of the code point it names, and the
          concatenated span outputs equal expected_digest_input_hex (the table agrees with itself);
  key     Store.digest(input) == SHA-256(expected_digest_input_hex);
  span    for every span that is not whitespace, Store.digest(span bytes) == SHA-256(span output), so "kept" means
          the library hashed those bytes as they came in, and "folded"/"mapped" means it hashed the replacement;
  alone   written into an empty store, Get returns the input bytes unchanged;
  stored  written into one store per mode in table order, Get returns expected_stored_hex.

Then, per mode: two rows have the same key exactly when they carry the same key_class, and the store's unique-value
count equals the number of key classes.

    python python/examples/normalization_fixture.py                # needs the prestige package (pip install ./python)
    python python/examples/normalization_fixture.py --table-only   # the "spans" check alone, no library needed

Exit status is 0 when every check passed and 1 otherwise.
"""

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "examples" / "normalization_fixture.tsv"
COLUMNS = [
    "row",
    "mode",
    "input_hex",
    "spans",
    "expected_stored_hex",
    "expected_digest_input_hex",
    "key_class",
    "input",
]
MODES = ["none", "whitespace", "ascii", "unicode"]
ACTIONS = {"kept", "folded", "mapped", "ws"}


class Span:
    def __init__(self, cell):
        # offset+length:U+XXXX:action[>output hex]
        where, codepoint, rest = cell.split(":")
        offset, length = where.split("+")
        self.offset, self.length = int(offset), int(length)
        self.codepoint = int(codepoint.replace("U+", "0x"), 0)
        self.action, _, out = rest.partition(">")
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r} in span {cell!r}")
        self.has_output = ">" in rest
        self.output_hex = out
        self.cell = cell


class Row:
    def __init__(self, fields, line_no):
        if len(fields) != len(COLUMNS):
            raise ValueError(
                f"line {line_no}: expected {len(COLUMNS)} tab-separated fields, got {len(fields)}"
            )
        rec = dict(zip(COLUMNS, fields))
        self.id = rec["row"]
        self.mode = rec["mode"]
        if self.mode not in MODES:
            raise ValueError(f"line {line_no}: unknown mode {self.mode!r}")
        self.input = bytes.fromhex(rec["input_hex"])
        self.spans = [Span(cell) for cell in rec["spans"].split(" ")]
        self.stored = bytes.fromhex(rec["expected_stored_hex"])
        self.digest_input = bytes.fromhex(rec["expected_digest_input_hex"])
        self.key_class = rec["key_class"]
        self.description = rec["input"]


def load(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if line_no == 1:
                if fields != COLUMNS:
                    raise ValueError(f"unexpected header: {fields}")
                continue
            rows.append(Row(fields, line_no))
    return rows


def span_bytes(row, span):
    return row.input[span.offset : span.offset + span.length]


def span_output(row, span):
    """The bytes this span contributes to the dedup-key input."""
    return bytes.fromhex(span.output_hex) if span.has_output else span_bytes(row, span)


def check_spans(row):
    """Table self-consistency; returns a list of problems (empty when the row is consistent)."""
    problems = []
    cursor = 0
    for span in row.spans:
        if span.offset != cursor:
            problems.append(f"span {span.cell} starts at {span.offset}, expected {cursor}")
        raw = span_bytes(row, span)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        if len(text) != 1 or ord(text) != span.codepoint:
            problems.append(
                f"span {span.cell} covers {raw.hex()}, which is not U+{span.codepoint:04X}"
            )
        if span.action == "kept" and span.has_output:
            problems.append(f"span {span.cell} is kept but names an output")
        if span.action != "kept" and not span.has_output:
            problems.append(f"span {span.cell} is {span.action} but names no output")
        cursor = span.offset + span.length
    if cursor != len(row.input):
        problems.append(f"spans cover {cursor} bytes, input has {len(row.input)}")
    joined = b"".join(span_output(row, s) for s in row.spans)
    if joined != row.digest_input:
        problems.append(
            f"span outputs join to {joined.hex()}, expected_digest_input_hex is {row.digest_input.hex()}"
        )
    return problems


def open_store(prestige, root, name, mode):
    options = prestige.Options()
    options.normalization_mode = getattr(prestige.NormalizationMode, mode.upper())
    return prestige.open(str(root / name), options)


def check_row_against_library(prestige, root, shared, row):
    """Checks key, span, alone and stored for one row; returns ({check: problem-or-None}, key)."""
    results = {}

    key = shared.digest(row.input)
    want = hashlib.sha256(row.digest_input).digest()
    results["key"] = (
        None
        if key == want
        else (
            f"digest is {key.hex()}, SHA-256 of {row.digest_input.hex() or '(empty)'} is {want.hex()}"
        )
    )

    bad = []
    for span in row.spans:
        if span.action == "ws":
            continue  # what whitespace becomes depends on its neighbours; the key check covers it
        got = shared.digest(span_bytes(row, span))
        if got != hashlib.sha256(span_output(row, span)).digest():
            bad.append(
                f"{span.cell}: library does not hash {span_bytes(row, span).hex()} as {span_output(row, span).hex()}"
            )
    results["span"] = "; ".join(bad) or None

    with open_store(prestige, root, f"alone-{row.id}", row.mode) as alone:
        alone.put("k", row.input)
        got = alone.get("k")
    results["alone"] = (
        None if got == row.input else f"empty store returned {got.hex()}, wrote {row.input.hex()}"
    )

    shared.put(row.id, row.input)
    got = shared.get(row.id)
    results["stored"] = (
        None if got == row.stored else f"Get returned {got.hex()}, expected {row.stored.hex()}"
    )
    return results, key


def print_row(row, results, key=None):
    failed = {name: problem for name, problem in results.items() if problem}
    names = " ".join(f"{name}={'FAIL' if name in failed else 'ok'}" for name in results)
    key_text = f"  key={key.hex()[:16]}" if key is not None else ""
    print(
        f"{'FAIL' if failed else 'PASS'}  {row.id:<3} {row.mode:<10} in={row.input.hex():<28} "
        f"class={row.key_class:<5} stored={row.stored.hex():<28}{key_text}  {names}"
    )
    for name, problem in failed.items():
        print(f"        {name}: {problem}")
    return not failed


def main():
    parser = argparse.ArgumentParser(
        description="Run examples/normalization_fixture.tsv against prestige."
    )
    parser.add_argument(
        "--fixture", type=Path, default=DEFAULT_FIXTURE, help="path to the fixture table"
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="check only that the table agrees with itself; does not import prestige",
    )
    args = parser.parse_args()

    rows = load(args.fixture)
    ids = [row.id for row in rows]
    if len(set(ids)) != len(ids):
        sys.exit("duplicate row ids in the fixture")
    print(f"fixture: {args.fixture} ({len(rows)} rows)")

    prestige = None
    if not args.table_only:
        import prestige

        print(f"library: prestige {getattr(prestige, '__version__', '(no version attribute)')}")
    print()

    passed = failed = 0
    root = Path(tempfile.mkdtemp(prefix="prestige-normalization-fixture-"))
    try:
        for mode in MODES:
            mode_rows = [row for row in rows if row.mode == mode]
            if not mode_rows:
                continue
            keys = {}
            shared = open_store(prestige, root, f"shared-{mode}", mode) if prestige else None
            try:
                for row in mode_rows:
                    problems = check_spans(row)
                    results = {"spans": "; ".join(problems) or None}
                    key = None
                    if shared is not None:
                        library_results, key = check_row_against_library(
                            prestige, root, shared, row
                        )
                        results.update(library_results)
                        keys[row.id] = key
                    if print_row(row, results, key):
                        passed += 1
                    else:
                        failed += 1

                # Per mode: the key classes must agree with the expected key inputs, and with the library's keys.
                problems = []
                for i, a in enumerate(mode_rows):
                    for b in mode_rows[i + 1 :]:
                        same_class = a.key_class == b.key_class
                        if same_class != (a.digest_input == b.digest_input):
                            problems.append(
                                f"{a.id}/{b.id}: key_class and expected_digest_input_hex disagree"
                            )
                        if keys and same_class != (keys[a.id] == keys[b.id]):
                            problems.append(
                                f"{a.id}/{b.id}: expected {'one key' if same_class else 'two keys'}, "
                                f"library gave {'one' if keys[a.id] == keys[b.id] else 'two'}"
                            )
                classes = len({row.key_class for row in mode_rows})
                detail = f"{len(mode_rows)} rows, {classes} key classes"
                if shared is not None:
                    unique = shared.count_unique_values()
                    detail += f", {shared.count_keys()} keys, {unique} unique values in the store"
                    if unique != classes:
                        problems.append(f"store holds {unique} unique values, expected {classes}")
                print(f"{'FAIL' if problems else 'PASS'}  mode {mode}: {detail}")
                for problem in problems:
                    print(f"        {problem}")
                if problems:
                    failed += 1
                else:
                    passed += 1
                print()
            finally:
                if shared is not None:
                    shared.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    what = (
        "table checks only (library not loaded)" if args.table_only else "table and library checks"
    )
    print(f"{passed} passed, {failed} failed; {what}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
