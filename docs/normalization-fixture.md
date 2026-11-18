# Normalization fixture: a table to check row by row

`examples/normalization_fixture.tsv` is a small table of inputs, one row per input per normalization mode, with what
the store is expected to do with each: the bytes the dedup key is computed over, which rows share a key, and the bytes
the store holds afterwards. `python/examples/normalization_fixture.py` runs the table against the library and prints
pass or fail per row. [normalization.md](normalization.md) describes the modes; this page is the evidence for it.

The shape of the table follows skitter-swarmmemo-2, who asked for the negative control and said how to assert it:

> The pair: U+00E9 (precomposed) and U+0065 U+0301 (combining). They are equal under NFC/NFD normalization and unequal
> as byte strings. A table that maps neither must leave both byte strings unchanged and report no mapping — that is one
> assertion: output bytes equal input bytes, and lookup returns absent.

and for the raw byte span of each input, so that "unmapped" can be told from "mapped to itself". Prestige's table
maps neither form, so the fixture makes that one assertion, per span, and then a second one about what is stored.

## The inputs

Nine inputs, each under the four modes (`none`, `whitespace`, `ascii`, `unicode`), 36 rows:

| # | Input | Bytes (hex) | Why it is here |
|---|---|---|---|
| 1 | `Hello World` | `48656c6c6f20576f726c64` | positive control |
| 2 | `  hello  world` (two leading spaces, two between the words) | `202068656c6c6f2020776f726c64` | positive control |
| 3 | `HELLO WORLD` | `48454c4c4f20574f524c44` | positive control |
| 4 | U+00E9, precomposed e-acute | `c3a9` | negative control |
| 5 | U+0065 U+0301, `e` + combining acute | `65cc81` | negative control |
| 6 | U+00C9, precomposed capital E-acute | `c389` | the Latin-1 case fold reaches the precomposed form |
| 7 | U+0045 U+0301, `E` + combining acute | `45cc81` | the ASCII case fold reaches the combining form |
| 8 | U+FB01, the fi ligature | `efac81` | a sequence the `kUnicode` table does map |
| 9 | `fi` | `6669` | what the ligature maps to |

## Columns

Tab-separated, one header line, hex in lowercase without separators.

| Column | Meaning |
|---|---|
| `row` | Row id: the mode's initial and the input number (`U4` is input 4 under `unicode`). |
| `mode` | `none`, `whitespace`, `ascii` or `unicode`: `NormalizationMode::kNone`, `kWhitespace`, `kASCII`, `kUnicode` (`prestige.NormalizationMode.NONE` ... `UNICODE` in Python). |
| `input_hex` | The bytes written. |
| `spans` | One entry per code point of the input, space-separated: `offset+length:U+XXXX:action`, with `>hex` appended when the action replaces the bytes. Offsets and lengths are in bytes. |
| `expected_stored_hex` | The bytes `Get` returns for this row's key after the mode's rows have been written to one store in table order. |
| `expected_digest_input_hex` | The bytes the dedup key is computed over. The key is their SHA-256. |
| `key_class` | Rows of one mode with the same label share a dedup key; rows with different labels do not. |
| `input` | The input in words. |

The four span actions:

| Action | Meaning |
|---|---|
| `kept` | The span's bytes go into the key input exactly as they came in. No mapping was applied. |
| `folded>hex` | Case fold: ASCII `A`-`Z` under `ascii` and `unicode`, U+00C0-U+00DE except U+00D7 under `unicode`. |
| `mapped>hex` | An entry of the `kUnicodeMappings` table in `src/normalize.cpp` replaced the span (`unicode` only). |
| `ws>hex` | Whitespace: `ws>20` is the single space a run collapses to, `ws>` is dropped (leading, trailing or the rest of a run). |

## Running it

```bash
pip install ./python                                  # from the repository root, builds the bindings
python python/examples/normalization_fixture.py
```

For every row the script checks:

- **spans**: the spans tile the input exactly, each is the UTF-8 of the code point it names, and the span outputs
  joined equal `expected_digest_input_hex`. This is the table agreeing with itself and needs no library:
  `python python/examples/normalization_fixture.py --table-only` runs it alone.
- **key**: `Store.digest(input)` equals SHA-256 of `expected_digest_input_hex`.
- **span**: for every span that is not whitespace, `Store.digest(span bytes)` equals SHA-256 of the span's output.
  This is the per-span assertion: a `kept` span is one whose bytes the library hashed as they came in.
- **alone**: written into an empty store, `Get` returns the input bytes unchanged.
- **stored**: written into one store per mode, in table order, `Get` returns `expected_stored_hex`.

and per mode: two rows have the same key exactly when they carry the same `key_class`, and
`count_unique_values()` equals the number of key classes. The exit status is 1 if anything failed.

Output for the `unicode` rows, from the library built from this repository:

```
PASS  U1  unicode    in=48656c6c6f20576f726c64       class=U-K1  stored=48656c6c6f20576f726c64        key=b94d27b9934d3e08  spans=ok key=ok span=ok alone=ok stored=ok
PASS  U2  unicode    in=202068656c6c6f2020776f726c64 class=U-K1  stored=48656c6c6f20576f726c64        key=b94d27b9934d3e08  spans=ok key=ok span=ok alone=ok stored=ok
PASS  U3  unicode    in=48454c4c4f20574f524c44       class=U-K1  stored=48656c6c6f20576f726c64        key=b94d27b9934d3e08  spans=ok key=ok span=ok alone=ok stored=ok
PASS  U4  unicode    in=c3a9                         class=U-K2  stored=c3a9                          key=4a99557e4033c353  spans=ok key=ok span=ok alone=ok stored=ok
PASS  U5  unicode    in=65cc81                       class=U-K3  stored=65cc81                        key=bf12767b0f2a56b2  spans=ok key=ok span=ok alone=ok stored=ok
PASS  U6  unicode    in=c389                         class=U-K2  stored=c3a9                          key=4a99557e4033c353  spans=ok key=ok span=ok alone=ok stored=ok
PASS  U7  unicode    in=45cc81                       class=U-K3  stored=65cc81                        key=bf12767b0f2a56b2  spans=ok key=ok span=ok alone=ok stored=ok
PASS  U8  unicode    in=efac81                       class=U-K4  stored=efac81                        key=b4bdc848109722a3  spans=ok key=ok span=ok alone=ok stored=ok
PASS  U9  unicode    in=6669                         class=U-K4  stored=efac81                        key=b4bdc848109722a3  spans=ok key=ok span=ok alone=ok stored=ok
PASS  mode unicode: 9 rows, 4 key classes, 9 keys, 4 unique values in the store
```

The whole run ends with `40 passed, 0 failed; table and library checks` (36 rows and one line per mode). `key=` is the
first eight bytes of the 32-byte key.

## What the rows say

**Positive control.** Inputs 1 to 3 have three keys under `none` and `whitespace` and one key under `ascii` and
`unicode` (`A-K1`, `U-K1`), computed over `68656c6c6f20776f726c64`, `hello world`.

**Negative control.** Rows `U4` and `U5`:

| Row | Input | Spans | Key input | Key class | Stored |
|---|---|---|---|---|---|
| `U4` | `c3a9` | `0+2:U+00E9:kept` | `c3a9` | `U-K2` | `c3a9` |
| `U5` | `65cc81` | `0+1:U+0065:kept 1+2:U+0301:kept` | `65cc81` | `U-K3` | `65cc81` |

Every span is `kept`, the key input equals the input, the two rows are in different key classes, and each key returns
its own bytes. The same holds for inputs 4 and 5 under the other three modes. `kUnicode` does not implement canonical
equivalence: a precomposed and a combining e-acute are two values.

**Unmapped, not mapped to itself.** The fixture can show that the bytes went into the key unchanged; that no table
entry was involved is read from the source. In `kUnicodeMappings` every source sequence starts with `c2`, `e2 81`,
`e2 82`, `ef ac`, `ef bc` or `ef bd`, and every destination is ASCII, so no entry maps a sequence to itself and none
matches `c3 a9` or `cc 81`. The Latin-1 case fold applies to `c3 80` through `c3 9e` only. Both spans fall through
to the pass-through at the end of `Normalize`, which copies the sequence as it is.

**The case folds reach both forms, separately.** Under `unicode`, `U6` (`c389`, É) folds to `c3a9` and joins `U4`;
`U7` (`45cc81`) folds its `E` and joins `U5`. Under `ascii` only `A7` joins `A5`; `A6` keeps its own key because
`ascii` does not touch bytes above `7f`. So `unicode` puts é and É under one key and `e`+U+0301 and `E`+U+0301 under
another, and still never joins the two.

**A mapped sequence.** `U8` (`efac81`) is `0+3:U+FB01:mapped>6669` and shares `U-K4` with `U9` (`fi`). Under the
other modes the ligature is `kept` and has a key of its own.

**What is stored is the first writer's bytes.** The store keeps one value per dedup key: the bytes of the first write
that produced that key. A later write under the same key adds a reference to that value and its own bytes are not
kept. So the store never holds the normalized form in place of what was written (`alone` passes on every row), but
`Get` on a later key returns the earlier writer's bytes:

| Row | Written | `Get` returns | Because |
|---|---|---|---|
| `U2` | `  hello  world` | `Hello World` | `U1` was written first under `U-K1` |
| `U3` | `HELLO WORLD` | `Hello World` | same |
| `U6` | `c389` (É) | `c3a9` (é) | `U4` was written first under `U-K2` |
| `U7` | `45cc81` | `65cc81` | `U5` was written first under `U-K3` |
| `U9` | `fi` | `efac81` (the ligature) | `U8` was written first under `U-K4` |

`expected_stored_hex` therefore depends on write order, which is why the script writes each mode's rows in table
order. With a normalization mode on, a key whose value shared a dedup key with an earlier one does not get its own
bytes back; under `none` every row does.

## Limits of the fixture

- It runs through the Python bindings, which do not expose `Normalize` itself. The key input is checked through the
  key: `Store.digest` must equal SHA-256 of the expected bytes.
- Whitespace spans are not checked on their own, because what a whitespace byte becomes depends on its neighbours. They
  are covered by the row's `key` check.
- It covers exact mode. In semantic mode `Store.digest` is not defined and keys are not content hashes.
- Nine inputs are a sample of the rules, not the whole mapping table; `tests/normalize_test.cpp` covers more of it.
