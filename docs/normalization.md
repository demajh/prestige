# Text Normalization

Prestige supports optional text normalization to ensure stable dedup keys across harmless formatting differences. Normalization is applied only for computing the dedup key; **original values are always stored unchanged**.

## Normalization Levels

| Level | Description |
|-------|-------------|
| `kNone` | No normalization (default). Byte-exact deduplication. |
| `kWhitespace` | Collapse multiple whitespace to single space, trim edges. |
| `kASCII` | Whitespace + ASCII case folding (A-Z -> a-z). |
| `kUnicode` | Whitespace + ASCII + common Unicode normalizations. |

## Unicode Normalizations (kUnicode mode)

The `kUnicode` mode handles common cases without external dependencies:

- **Ligatures**: ﬀ -> ff, ﬁ -> fi, ﬂ -> fl, ﬃ -> ffi, ﬄ -> ffl, ﬅ -> st, ﬆ -> st
- **Superscripts**: ⁰¹²³⁴⁵⁶⁷⁸⁹ -> 0123456789, ⁺⁻⁼⁽⁾ⁿ -> +-=()n
- **Subscripts**: ₀₁₂₃₄₅₆₇₈₉ -> 0123456789, ₊₋₌₍₎ -> +-=()
- **Fractions**: ½ -> 1/2, ¼ -> 1/4, ¾ -> 3/4
- **Fullwidth ASCII** (U+FF01 to U+FF5E): Ａ-Ｚ -> a-z, ａ-ｚ -> a-z, ０-９ -> 0-9, fullwidth punctuation -> ASCII punctuation
- **Latin-1 case folding** (U+00C0 to U+00DE): À -> à, É -> é, Þ -> þ, etc.; × (U+00D7) is left alone

It is a fixed table, not Unicode NFKC: combining marks are neither composed nor decomposed, letters outside ASCII and
Latin-1 are not case folded, and only the six ASCII whitespace bytes (space, tab, LF, CR, VT, FF) count as whitespace. The table and
the loop that applies it are in `src/normalize.cpp`.

## Configuration

```cpp
prestige::Options opt;
opt.normalization_mode = prestige::NormalizationMode::kUnicode;
opt.normalization_max_bytes = 1024 * 1024;  // Skip for values > 1MB (default)
```

| Option | Default | Description |
|--------|---------|-------------|
| `normalization_mode` | `kNone` | Normalization level |
| `normalization_max_bytes` | 1MB | Skip normalization for values larger than this |

## Example

```cpp
prestige::Options opt;
opt.normalization_mode = prestige::NormalizationMode::kUnicode;

auto store = prestige::Store::Open("./my_db", &db, opt);

// These will deduplicate despite formatting differences:
store->Put("k1", "Hello World");     // stored as-is
store->Put("k2", "  hello  world");  // stored as-is, but dedup key matches k1
store->Put("k3", "HELLO WORLD");     // stored as-is, but dedup key matches k1

// Get returns original values:
std::string v;
store->Get("k2", &v);  // Returns "  hello  world"
```

## Test Vectors

A reader who recomputes a dedup key outside the store (to check an `output_digest` in a
[decision record](provenance.md), say) has to reproduce the rule byte for byte, since the key is the SHA-256 of the
normalized bytes. The vectors below pin what the rule does, including the places a second implementation is most
likely to differ: what counts as whitespace, what is not folded, and what happens to bytes that are not valid UTF-8.
A value larger than `normalization_max_bytes` is hashed as stored, with no rule applied.

The ids and the separate version are Mark's design, from a conversation on Moltbook: "give each vector an id so a
failure report can name it, and give the vector set a version of its own so adding a vector isn't mistaken for
changing the rule."

Two things are versioned, and they move independently:

- **The rule** is the code in `src/normalize.cpp`. It has no number of its own; it is named by the prestige version
  that contains it (`PRESTIGE_VERSION_STRING`, `prestige::Version()`). These vectors describe the rule in prestige
  0.2.0, the first release with normalization.
- **The vector set** is at version **1**, recorded in the `# vector-set:` line of the vector file. Adding or
  retiring a vector raises it and says nothing about the rule. A vector's id is never reused and its bytes never
  change: if the rule changes so that a vector no longer holds, that vector is retired and the new behaviour gets a
  new id.

The machine-readable copy is [`examples/normalization_vectors.tsv`](../examples/normalization_vectors.tsv):
tab-separated `id`, `mode`, `input_hex`, `expected_hex`, `note`, with `-` for zero bytes. The table shows the same
bytes, spaced for reading.

| Id | Mode | Input bytes | Expected output bytes | What it pins |
|----|------|-------------|-----------------------|--------------|
| `NV-001` | `kNone` | `20 20 48 65 6C 6C 6F 20 09 20 57 6F 72 6C 64 20 20` | `20 20 48 65 6C 6C 6F 20 09 20 57 6F 72 6C 64 20 20` | kNone leaves every byte alone |
| `NV-002` | `kWhitespace` | `-` | `-` | empty input stays empty |
| `NV-003` | `kWhitespace` | `20 20 68 65 6C 6C 6F 20 09 0A 20 77 6F 72 6C 64 20 20` | `68 65 6C 6C 6F 20 77 6F 72 6C 64` | leading and trailing whitespace trimmed, inner run collapsed to one space |
| `NV-004` | `kWhitespace` | `61 20 09 0A 0D 0B 0C 62` | `61 20 62` | all six whitespace bytes (20 09 0A 0D 0B 0C) in one run |
| `NV-005` | `kWhitespace` | `20 09 0D 0A 20` | `-` | whitespace only becomes empty |
| `NV-006` | `kWhitespace` | `48 65 6C 6C 6F 20 20 57 6F 72 6C 64` | `48 65 6C 6C 6F 20 57 6F 72 6C 64` | case is kept in kWhitespace |
| `NV-007` | `kWhitespace` | `61 C2 A0 62 E3 80 80 63` | `61 C2 A0 62 E3 80 80 63` | U+00A0 and U+3000 are not whitespace to the rule |
| `NV-008` | `kWhitespace` | `6F 6B 20 F0 9F 98 80 20 20 6F 6B` | `6F 6B 20 F0 9F 98 80 20 6F 6B` | 4-byte sequence passes through |
| `NV-009` | `kASCII` | `20 20 48 45 4C 4C 4F 20 20 57 6F 72 6C 64 20` | `68 65 6C 6C 6F 20 77 6F 72 6C 64` | whitespace rule plus A-Z to a-z |
| `NV-010` | `kASCII` | `5B 41 5D 5E 5F 60 7B 5A 7D 7E 20 30 39` | `5B 61 5D 5E 5F 60 7B 7A 7D 7E 20 30 39` | only A-Z change; neighbouring punctuation and digits do not |
| `NV-011` | `kASCII` | `C3 89 43 4F 4C 45` | `C3 89 63 6F 6C 65` | kASCII does not fold non-ASCII letters |
| `NV-012` | `kASCII` | `EF AC 81 6C 65 20 78 C2 B2` | `EF AC 81 6C 65 20 78 C2 B2` | kASCII leaves ligatures and superscripts alone |
| `NV-013` | `kASCII` | `E2 41 42 43` | `E2 41 42 63` | lead byte E2 without continuation bytes: the next two bytes are copied unfolded as part of its 3-byte length |
| `NV-014` | `kUnicode` | `EF AC 80 20 EF AC 81 20 EF AC 82 20 EF AC 83 20 EF AC 84` | `66 66 20 66 69 20 66 6C 20 66 66 69 20 66 66 6C` | ligatures U+FB00..U+FB04 |
| `NV-015` | `kUnicode` | `EF AC 85 20 EF AC 86` | `73 74 20 73 74` | U+FB05 (long s t) and U+FB06 both become st |
| `NV-016` | `kUnicode` | `E2 81 B0 C2 B9 C2 B2 C2 B3 E2 81 B4 E2 81 B5 E2 81 B6 E2 81 B7 E2 81 B8 E2 81 B9` | `30 31 32 33 34 35 36 37 38 39` | superscript digits 0-9 |
| `NV-017` | `kUnicode` | `E2 81 BA E2 81 BB E2 81 BC E2 81 BD E2 81 BE E2 81 BF` | `2B 2D 3D 28 29 6E` | superscript + - = ( ) n |
| `NV-018` | `kUnicode` | `78 E2 81 B1` | `78 E2 81 B1` | U+2071 superscript i is not in the table |
| `NV-019` | `kUnicode` | `E2 82 80 E2 82 81 E2 82 82 E2 82 83 E2 82 84 E2 82 85 E2 82 86 E2 82 87 E2 82 88 E2 82 89` | `30 31 32 33 34 35 36 37 38 39` | subscript digits 0-9 |
| `NV-020` | `kUnicode` | `E2 82 8A E2 82 8B E2 82 8C E2 82 8D E2 82 8E` | `2B 2D 3D 28 29` | subscript + - = ( ) |
| `NV-021` | `kUnicode` | `C2 BD 20 C2 BC 20 C2 BE` | `31 2F 32 20 31 2F 34 20 33 2F 34` | fractions become three ASCII bytes each |
| `NV-022` | `kUnicode` | `EF BC A1 EF BC A2 EF BC BA 20 EF BD 81 EF BD 82 EF BD 9A 20 EF BC 90 EF BC 99` | `61 62 7A 20 61 62 7A 20 30 39` | fullwidth letters fold to lowercase ASCII, fullwidth digits to ASCII digits |
| `NV-023` | `kUnicode` | `EF BC 81 EF BC A0 EF BC BB EF BD 80 EF BD 9B EF BD 9E` | `21 40 5B 60 7B 7E` | fullwidth punctuation U+FF01..U+FF5E |
| `NV-024` | `kUnicode` | `61 E3 80 80 62 EF BD 9F` | `61 E3 80 80 62 EF BD 9F` | U+3000 ideographic space and U+FF5F are outside the fullwidth range and unchanged |
| `NV-025` | `kUnicode` | `C3 80 C3 89 C3 8E C3 95 C3 9C C3 9E` | `C3 A0 C3 A9 C3 AE C3 B5 C3 BC C3 BE` | Latin-1 uppercase U+00C0..U+00DE folds to lowercase |
| `NV-026` | `kUnicode` | `C3 97 20 C3 9F 20 C3 B7` | `C3 97 20 C3 9F 20 C3 B7` | multiplication sign, sharp s and division sign are unchanged |
| `NV-027` | `kUnicode` | `C4 80 20 C5 B8 20 CE A3 20 D0 96` | `C4 80 20 C5 B8 20 CE A3 20 D0 96` | letters outside Latin-1 are not folded |
| `NV-028` | `kUnicode` | `45 CC 81` | `65 CC 81` | no composition: E + U+0301 becomes e + U+0301, which differs from the output for U+00C9 |
| `NV-029` | `kUnicode` | `C3 89` | `C3 A9` | precomposed U+00C9 becomes U+00E9; compare the previous vector |
| `NV-030` | `kUnicode` | `61 C2 A0 62` | `61 C2 A0 62` | U+00A0 is neither whitespace nor mapped |
| `NV-031` | `kUnicode` | `41 FF 42` | `61 FF 62` | invalid byte FF is copied; the bytes around it still fold |
| `NV-032` | `kUnicode` | `41 EF AC` | `61 EF AC` | truncated ligature prefix at end of input is copied |
| `NV-033` | `kUnicode` | `20 20 EF BC A8 65 6C 6C 6F 20 20 EF AC 81 6C 65 20 C2 BD 20 C3 89 43 4F 4C 45 20 78 C2 B2 20` | `68 65 6C 6C 6F 20 66 69 6C 65 20 31 2F 32 20 C3 A9 63 6F 6C 65 20 78 32` | every rule in one value |

`NV-013`, `NV-031` and `NV-032` record what the code does with malformed UTF-8 rather than a choice anyone would
defend on its own: the bytes are copied through, and in `NV-013` a lead byte that announces a three-byte sequence
takes the next two bytes with it unfolded, whatever they are.

`examples/normalization_vectors.cpp` runs the file through the repository's own normalizer and prints `PASS` or
`FAIL` per id, with the input, expected and observed bytes for a failure. It needs only the normalizer, not RocksDB:

```bash
c++ -std=c++17 -Iinclude examples/normalization_vectors.cpp src/normalize.cpp -o normalization_vectors
./normalization_vectors        # reads examples/normalization_vectors.tsv; exit status 1 if any vector fails
```

```
PASS NV-001 kNone
...
PASS NV-033 kUnicode
vector set 1, prestige 0.2.0: 33 of 33 vectors pass
```

### Reporting a failing vector

Failure reports from readers are accepted. If your implementation, or a build of prestige, disagrees with a vector,
open an issue at https://github.com/demajh/prestige/issues with the vector id and the vector-set version, the input
bytes, the expected output, the output you observed (all three as hex) and your library versions: the prestige
version if prestige produced the output, otherwise the language, runtime and Unicode or string library your reader
uses. If the bytes that disagree are not in the set, send them the same way without an id.

## Important Notes

- Normalization affects **only the dedup key computation**
- Original values are stored and returned unchanged
- User keys are **not** normalized (they remain opaque byte sequences)
- Values larger than `normalization_max_bytes` skip normalization entirely
