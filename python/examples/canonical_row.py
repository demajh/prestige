#!/usr/bin/env python3
"""Hash a record's canonical bytes, stage by stage, against published vectors.

A record becomes one line of bytes, the line gets a SHA-256, and each field's
value gets a SHA-256 of its own byte span inside the line. The expected values
for every stage are in examples/canonical_row_vectors.json and in
docs/canonical-row.md, so a stranger can test an encoder before trusting any
hash built on it.

Stages (run one with --stage, or all of them):
  bytes      the worked row's canonical line as hex, with its byte count
  hash       sha256 of the line
  fields     per field: typed value, byte span in the line, sha256 of the span
  edges      null, negative int, integer above 2^53, empty string, non-ASCII
  near-miss  the worked row with one field altered
  control    the worked row through three wrong-but-plausible encoders
  decoders   the lines parsed back with exact integers and with doubles
  chain      heads that link each row to the previous head

Needs only the Python standard library (3.8+). --store PATH additionally
writes the lines to a prestige store and checks that the store's content key
is the same SHA-256; that part needs the prestige package.

A matching line proves agreement with the published encoder, not correctness.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ENCODER_VERSION = "example-row-json/1"
ENCODER_LINE = 'json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")'
SAFE_INT = 2**53 - 1
VECTORS = Path(__file__).resolve().parents[2] / "examples" / "canonical_row_vectors.json"
STAGES = ["bytes", "hash", "fields", "edges", "near-miss", "control", "decoders", "chain"]

# Written in an order that is not sorted, so an encoder that keeps insertion
# order produces a different line.
WORKED = {
    "seq": 41,
    "attempt_id": "att-0041",
    "agent": "probe-3",
    "state": "finished",
    "leaf_index": 12,
    "skew_ms": 250,
    "credits": 1800,
    "outcome_of": "att-0040",
    "note": "leaf index read from the proof response",
}

# Each edge row is the worked row with one field replaced.
EDGES = [
    ("null", "outcome_of", None),
    ("negative-int", "skew_ms", -250),
    ("int-above-2^53", "credits", 2**53 + 1),
    ("empty-string", "note", ""),
    ("non-ascii", "note", "café 東京 ✓ \U0001f9ed"),
]
NEAR_MISS = ("near-miss", "leaf_index", 13)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def altered(field, value):
    row = dict(WORKED)
    row[field] = value
    return row


# ---------------------------------------------------------------------------
# The published encoder
# ---------------------------------------------------------------------------


def prepare(row):
    """Check the row against what version 1 can carry and wrap large integers.

    A row is a flat object whose values are strings, integers or null. An
    integer outside +/-(2^53 - 1) is written as {"int": "<decimal digits>"}:
    a parser that reads numbers as doubles cannot hold it, a string survives.
    """
    out = {}
    for name, value in row.items():
        if not isinstance(name, str) or not name.isascii():
            raise ValueError(f"field names are ASCII strings in version 1: {name!r}")
        if value is None or isinstance(value, str):
            out[name] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            out[name] = value if abs(value) <= SAFE_INT else {"int": str(value)}
        else:
            raise ValueError(
                f"version 1 carries strings, integers and null; {name} is {type(value).__name__}"
            )
    return out


def encode(row):
    """The canonical line. No trailing newline: the hash covers exactly these bytes."""
    return json.dumps(
        prepare(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Wrong-but-plausible encoders (controls)
# ---------------------------------------------------------------------------


def encode_insertion_order(row):
    """Right separators and number forms, keys left in the order they were written."""
    return json.dumps(
        prepare(row), separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def encode_unsorted_floats(row):
    """Keys in insertion order and every bare integer written as a float."""
    prepared = {k: float(v) if isinstance(v, int) else v for k, v in prepare(row).items()}
    return json.dumps(prepared, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )


def encode_python_default(row):
    """json.dumps(row) with no arguments: insertion order, ', ' and ': ', \\u escapes."""
    return json.dumps(prepare(row)).encode("utf-8")


CONTROLS = [
    (
        "insertion-order-keys",
        encode_insertion_order,
        'json.dumps(row, separators=(",", ":"), ensure_ascii=False)',
    ),
    (
        "insertion-order-keys+ints-as-floats",
        encode_unsorted_floats,
        'json.dumps({k: float(v) if is_int(v) else v ...}, separators=(",", ":"), ensure_ascii=False)',
    ),
    ("python-default", encode_python_default, "json.dumps(row)"),
]


# ---------------------------------------------------------------------------
# Field spans: read from the line itself, whatever encoder wrote it
# ---------------------------------------------------------------------------


def field_spans(line):
    """Return {field: (start, end)} byte offsets of each top-level value, end exclusive."""
    text = line.decode("utf-8")
    decoder = json.JSONDecoder()

    def skip(i):
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        return i

    def byte_offset(i):
        return len(text[:i].encode("utf-8"))

    i = skip(0)
    if i >= len(text) or text[i] != "{":
        raise ValueError("the line is not a JSON object")
    i = skip(i + 1)
    spans = {}
    while text[i] != "}":
        name, i = decoder.raw_decode(text, i)
        i = skip(i)
        if text[i] != ":":
            raise ValueError("expected ':' at character %d" % i)
        start = skip(i + 1)
        _, end = decoder.raw_decode(text, start)
        spans[name] = (byte_offset(start), byte_offset(end))
        i = skip(end)
        if text[i] == ",":
            i = skip(i + 1)
    return spans


def typed(value):
    """(type, value as a string) so that no reader of the vectors needs a number parser."""
    if value is None:
        return "null", None
    if isinstance(value, str):
        return "string", value
    return "int", str(value)


def describe_line(line, row=None):
    """Everything published about one line: hex, length, hash, per-field spans and digests."""
    spans = field_spans(line)
    fields = []
    for name in sorted(spans, key=lambda n: spans[n][0]):
        start, end = spans[name]
        entry = {"name": name}
        if row is not None:
            entry["type"], entry["value"] = typed(row[name])
        entry.update(
            {
                "span": [start, end],
                "span_text": line[start:end].decode("utf-8"),
                "span_sha256": sha256_hex(line[start:end]),
            }
        )
        fields.append(entry)
    return {
        "line_utf8": line.decode("utf-8"),
        "line_hex": line.hex(),
        "byte_count": len(line),
        "line_sha256": sha256_hex(line),
        "fields": fields,
    }


def compare(reference, other):
    """Per-field comparison of two described lines. Returns (same_line, moved, shifted, missing)."""
    ref = {f["name"]: f for f in reference["fields"]}
    oth = {f["name"]: f for f in other["fields"]}
    moved = sorted(n for n in ref if n in oth and ref[n]["span_sha256"] != oth[n]["span_sha256"])
    shifted = sorted(n for n in ref if n in oth and ref[n]["span"] != oth[n]["span"])
    missing = sorted(set(ref) ^ set(oth))
    return reference["line_sha256"] == other["line_sha256"], moved, shifted, missing


def verdict(reference, other):
    same, moved, _shifted, missing = compare(reference, other)
    if same:
        return "line hash matches: this encoder agrees with the published one on this row"
    if missing:
        return "field set differs: {}".format(", ".join(missing))
    if not moved:
        return (
            "line hash differs, every span digest matches: encoder layout (key order or separators)"
        )
    return "line hash differs, span digest moved for: {}".format(", ".join(moved))


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


def unwrap(parsed):
    return {
        k: int(v["int"]) if isinstance(v, dict) and set(v) == {"int"} else v
        for k, v in parsed.items()
    }


def decode_exact(line):
    """Integers stay integers of any size (Python's default)."""
    return unwrap(json.loads(line.decode("utf-8")))


def decode_doubles(line):
    """Every number goes through an IEEE-754 double, as JSON.parse does."""
    return unwrap(json.loads(line.decode("utf-8"), parse_int=lambda s: int(float(s))))


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


def head_line(index, prev_head, row_sha256, encoder=ENCODER_VERSION):
    """A head is itself a row, encoded by the same encoder; the first has prev null."""
    return encode({"index": index, "encoder": encoder, "prev": prev_head, "row_sha256": row_sha256})


def chain(lines, encoder=ENCODER_VERSION):
    heads, prev = [], None
    for index, line in enumerate(lines):
        hl = head_line(index, prev, sha256_hex(line), encoder)
        prev = sha256_hex(hl)
        heads.append(
            {
                "index": index,
                "head_line_utf8": hl.decode("utf-8"),
                "byte_count": len(hl),
                "head": prev,
            }
        )
    return heads


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------


def build_vectors():
    worked = describe_line(encode(WORKED), WORKED)
    vectors = {
        "encoder": {
            "version": ENCODER_VERSION,
            "line": ENCODER_LINE,
            "hash": "sha256 over the line's bytes, lowercase hex",
            "trailing_newline_inside_hash": False,
            "text": "UTF-8, written as given: no Unicode normalization, no \\u escapes except the ones JSON requires",
            "keys": "ASCII, sorted by code point",
            "integers": "decimal digits, '-' for negatives, no exponent, no fraction, when |n| <= 2^53 - 1",
            "large_integers": 'outside that range the value is the object {"int":"<decimal digits>"}',
            "null": "null",
            "span": "byte offsets [start, end) of a field's value in the line, quotes included, end exclusive",
            "values_in_this_file": "each field's value is given as a type and a string, so no number parser is needed",
        },
        "note": "A matching line proves agreement with the published encoder, not correctness.",
        "worked_row": worked,
        "edge_rows": [],
    }
    for name, field, value in EDGES:
        row = altered(field, value)
        entry = {"name": name, "field": field}
        entry.update(describe_line(encode(row), row))
        vectors["edge_rows"].append(entry)

    name, field, value = NEAR_MISS
    row = altered(field, value)
    near = {"name": name, "field": field}
    near.update(describe_line(encode(row), row))
    _, moved, shifted, _ = compare(worked, near)
    near["span_digests_that_differ_from_worked_row"] = moved
    vectors["near_miss_row"] = near

    vectors["control_rows"] = []
    for name, fn, how in CONTROLS:
        entry = {"name": name, "encoder_line": how, "row": "worked_row"}
        entry.update(describe_line(fn(WORKED)))
        _, moved, shifted, _ = compare(worked, entry)
        entry["span_digests_that_differ_from_worked_row"] = moved
        entry["spans_at_other_offsets"] = shifted
        vectors["control_rows"].append(entry)

    vectors["decoders"] = decoder_vectors()

    lines = [encode(WORKED)] + [encode(altered(f, v)) for _, f, v in EDGES]
    tampered = [encode(altered(NEAR_MISS[1], NEAR_MISS[2]))] + lines[1:]
    vectors["chain"] = {
        "head_rule": "head = sha256 of the canonical line of "
        '{"encoder": version, "index": n, "prev": previous head or null, "row_sha256": sha256 of row n}',
        "rows": ["worked_row"] + [name for name, _, _ in EDGES],
        "heads": chain(lines),
        "heads_with_near_miss_as_row_0": [h["head"] for h in chain(tampered)],
        "heads_with_encoder_version_2": [h["head"] for h in chain(lines, "example-row-json/2")],
    }
    return vectors


def decoder_vectors():
    big_row = altered("credits", 2**53 + 1)
    bare = json.dumps(big_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    cases = [
        ("worked_row", encode(WORKED), encode),
        ("int-above-2^53, typed as a string (the published form)", encode(big_row), encode),
        (
            "int-above-2^53, written as a bare number (not the published form)",
            bare,
            lambda row: json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
        ),
    ]
    out = []
    for name, line, reencode in cases:
        before = describe_line(line)
        entry = {"line": name, "line_sha256": before["line_sha256"]}
        for label, decoder in (("exact_integers", decode_exact), ("doubles", decode_doubles)):
            row = decoder(line)
            after = describe_line(reencode(row))
            _, moved, _, _ = compare(before, after)
            entry[label] = {
                "credits_after_decoding": str(row["credits"]),
                "reencoded_line_sha256": after["line_sha256"],
                "round_trips": after["line_sha256"] == before["line_sha256"],
                "span_digests_that_moved": moved,
            }
        out.append(entry)
    return out


def dump_vectors(vectors):
    return json.dumps(vectors, indent=2, ensure_ascii=True) + "\n"


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_line(d, with_fields=True):
    print("  line        {}".format(d["line_utf8"]))
    print("  hex         {}".format(d["line_hex"]))
    print("  byte count  %d (no trailing newline; none is hashed)" % d["byte_count"])
    print("  sha256      {}".format(d["line_sha256"]))
    if with_fields:
        print_fields(d)


def print_fields(d):
    for f in d["fields"]:
        value = ""
        if "type" in f:
            value = "  {} {}".format(
                f["type"],
                "null" if f["value"] is None else json.dumps(f["value"], ensure_ascii=False),
            )
        print(
            "    %-11s [%3d,%3d)  %s%s"
            % (f["name"], f["span"][0], f["span"][1], f["span_sha256"], value)
        )


def stage_bytes(v):
    print("== bytes: the worked row as one canonical line")
    print("  encoder     {}".format(v["encoder"]["version"]))
    print("  encoder line  {}".format(v["encoder"]["line"]))
    w = v["worked_row"]
    print("  line        {}".format(w["line_utf8"]))
    print("  hex         {}".format(w["line_hex"]))
    print("  byte count  %d (no trailing newline; none is hashed)" % w["byte_count"])


def stage_hash(v):
    print("== hash: sha256 of those bytes")
    print("  sha256      {}".format(v["worked_row"]["line_sha256"]))


def stage_fields(v):
    print("== fields: value, byte span [start,end) in the line, sha256 of the span")
    print_fields(v["worked_row"])


def stage_edges(v):
    print("== edges: the worked row with one field replaced")
    for e in v["edge_rows"]:
        print("- {} (field {})".format(e["name"], e["field"]))
        print_line(e, with_fields=False)
        f = next(f for f in e["fields"] if f["name"] == e["field"])
        print(
            "    %-11s [%3d,%3d)  %s  span text %s"
            % (f["name"], f["span"][0], f["span"][1], f["span_sha256"], f["span_text"])
        )


def stage_near_miss(v):
    n = v["near_miss_row"]
    print("== near-miss: leaf_index 12 -> 13, one byte of the line")
    print_line(n, with_fields=False)
    print("  {}".format(verdict(v["worked_row"], n)))


def stage_control(v):
    print("== control: the worked row through wrong-but-plausible encoders")
    for c in v["control_rows"]:
        print("- {}: {}".format(c["name"], c["encoder_line"]))
        print_line(c, with_fields=False)
        print("  {}".format(verdict(v["worked_row"], c)))
    print("  {}".format(v["note"]))


def stage_decoders(v):
    print("== decoders: parse the line, encode it again, compare")
    for d in v["decoders"]:
        print("- {}".format(d["line"]))
        for label in ("exact_integers", "doubles"):
            r = d[label]
            what = (
                "round-trips"
                if r["round_trips"]
                else "does NOT round-trip, moved: {}".format(
                    ", ".join(r["span_digests_that_moved"])
                )
            )
            print("    %-15s credits=%s  %s" % (label, r["credits_after_decoding"], what))


def stage_chain(v):
    c = v["chain"]
    print("== chain: each head commits to the row, the encoder version and the previous head")
    for h, name in zip(c["heads"], c["rows"]):
        print("  %d %-15s %s" % (h["index"], name, h["head"]))
        print("      {}".format(h["head_line_utf8"]))
    print("  with the near-miss row as row 0, every head differs:")
    for i, h in enumerate(c["heads_with_near_miss_as_row_0"]):
        print("  %d %-15s %s" % (i, "", h))


PRINTERS = {
    "bytes": stage_bytes,
    "hash": stage_hash,
    "fields": stage_fields,
    "edges": stage_edges,
    "near-miss": stage_near_miss,
    "control": stage_control,
    "decoders": stage_decoders,
    "chain": stage_chain,
}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def find_row(vectors, name):
    rows = {"worked": vectors["worked_row"], "near-miss": vectors["near_miss_row"]}
    rows.update({e["name"]: e for e in vectors["edge_rows"]})
    if name not in rows:
        raise SystemExit("unknown row {!r}; choose from: {}".format(name, ", ".join(rows)))
    return rows[name]


def cmd_compare(vectors, hex_line, row_name):
    """Diagnose a line produced by someone else's encoder for one of the published rows."""
    reference = find_row(vectors, row_name)
    try:
        line = bytes.fromhex("".join(hex_line.split()))
        other = describe_line(line)
    except ValueError as e:
        print(f"cannot read the line: {e}")
        return 2
    print("published  %s  %d bytes" % (reference["line_sha256"], reference["byte_count"]))
    print("yours      %s  %d bytes" % (other["line_sha256"], other["byte_count"]))
    same, moved, shifted, _missing = compare(reference, other)
    ref = {f["name"]: f for f in reference["fields"]}
    for f in other["fields"]:
        r = ref.get(f["name"])
        if r is None:
            state = "not in the published row"
        elif f["name"] in moved:
            state = "DIFFERS  published {}  yours {}".format(r["span_text"], f["span_text"])
        elif f["name"] in shifted:
            state = "same bytes, at [%d,%d) instead of [%d,%d)" % (
                f["span"][0],
                f["span"][1],
                r["span"][0],
                r["span"][1],
            )
        else:
            state = "same bytes, same span"
        print("  %-11s %s" % (f["name"], state))
    print(verdict(reference, other))
    if line.endswith(b"\n"):
        print(
            "your line ends with a newline; the published line does not, and no newline is hashed"
        )
    return 0 if same else 1


def cmd_store(vectors, path):
    """Write every published line to a prestige store and compare its content key."""
    try:
        import prestige
    except ImportError:
        print(
            "--store needs the prestige package (see docs/python-bindings.md); the other stages do not"
        )
        return 2
    rows = [("worked", vectors["worked_row"])] + [(e["name"], e) for e in vectors["edge_rows"]]
    ok = True
    with prestige.open(path) as store:
        if not hasattr(store, "digest"):
            print(
                "this prestige build ({}) has no Store.digest; build the bindings from this repository".format(
                    getattr(prestige, "__version__", "unknown version")
                )
            )
            return 2
        for name, d in rows:
            line = bytes.fromhex(d["line_hex"])
            key = "rows/" + name
            store.put(key, line)
            digest = store.digest(line).hex()
            back = store.get(key)
            good = digest == d["line_sha256"] and back == line
            ok = ok and good
            print(
                "  %-15s store.digest %s  %s"
                % (name, digest, "matches, bytes returned unchanged" if good else "MISMATCH")
            )
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--stage", choices=STAGES + ["all"], default="all", help="print one stage (default: all)"
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="recompute every vector and compare with the published file",
    )
    p.add_argument(
        "--vectors",
        default=str(VECTORS),
        help="path of the published vectors (default: %(default)s)",
    )
    p.add_argument(
        "--write-vectors", action="store_true", help="rewrite the vectors file from this script"
    )
    p.add_argument(
        "--compare",
        metavar="HEX",
        help="diagnose your encoder's line (hex, or - for stdin) against a published row",
    )
    p.add_argument(
        "--row", default="worked", help="row --compare is checked against (default: worked)"
    )
    p.add_argument(
        "--store",
        metavar="PATH",
        help="also write the lines to a prestige store at PATH and compare digests",
    )
    args = p.parse_args(argv)

    computed = build_vectors()
    if args.write_vectors:
        Path(args.vectors).write_text(dump_vectors(computed), encoding="utf-8")
        print(f"wrote {args.vectors}")
        return 0
    if args.check:
        published = json.loads(Path(args.vectors).read_text(encoding="utf-8"))
        if published != computed:
            for key in computed:
                if published.get(key) != computed[key]:
                    print(f"MISMATCH in {key}")
            return 1
        print(
            "all vectors in %s reproduced (%d rows, %d controls, %d heads)"
            % (
                args.vectors,
                2 + len(computed["edge_rows"]),
                len(computed["control_rows"]),
                len(computed["chain"]["heads"]),
            )
        )
        return 0
    if args.compare:
        hex_line = sys.stdin.read() if args.compare == "-" else args.compare
        return cmd_compare(computed, hex_line, args.row)
    if args.store:
        return cmd_store(computed, args.store)

    for stage in STAGES if args.stage == "all" else [args.stage]:
        PRINTERS[stage](computed)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
