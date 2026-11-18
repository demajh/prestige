# Text Normalization

Prestige supports optional text normalization to ensure stable dedup keys across harmless formatting differences. Normalization is applied only for computing the dedup key; **the normalized form is never stored**. The store keeps one value per dedup key, the bytes of the first write that produced it (see [Important Notes](#important-notes)).

[normalization-fixture.md](normalization-fixture.md) is a table of inputs with expected keys and stored bytes, and a script that checks it against the library row by row.

## Normalization Levels

| Level | Description |
|-------|-------------|
| `kNone` | No normalization (default). Byte-exact deduplication. |
| `kWhitespace` | Collapse multiple whitespace to single space, trim edges. |
| `kASCII` | Whitespace + ASCII case folding (A-Z -> a-z). |
| `kUnicode` | Whitespace + ASCII + common Unicode normalizations. |

## Unicode Normalizations (kUnicode mode)

The `kUnicode` mode handles common cases without external dependencies:

- **Ligatures**: U+FB00-U+FB06 (ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ) -> ff, fi, fl, ffi, ffl, st, st
- **Superscripts**: U+00B9, U+00B2, U+00B3, U+2070, U+2074-U+207F (¹²³⁰⁴-⁹ ⁺⁻⁼⁽⁾ⁿ) -> digits, `+ - = ( )`, n
- **Subscripts**: U+2080-U+208E (₀-₉ ₊₋₌₍₎) -> digits, `+ - = ( )`
- **Fractions**: ½ -> 1/2, ¼ -> 1/4, ¾ -> 3/4
- **Fullwidth ASCII**: U+FF01-U+FF5E -> U+0021-U+007E, letters to lowercase (Ａ-Ｚ and ａ-ｚ -> a-z, ０-９ -> 0-9)
- **Latin-1 case folding**: U+00C0-U+00DE except U+00D7 -> lowercase (À -> à, É -> é, Þ -> þ)

Anything else above U+007F passes through unchanged. In particular `kUnicode` is not NFC, NFD or NFKC and does not
implement canonical equivalence: precomposed U+00E9 (`c3 a9`) and `e` followed by combining U+0301 (`65 cc 81`) keep
their bytes in the key and are two values. Accents are not stripped, and case folding stops at Latin-1.

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

// These share one dedup key (SHA-256 of "hello world"); one value is stored:
store->Put("k1", "Hello World");     // first write under that key: these bytes are stored
store->Put("k2", "  hello  world");  // same dedup key: k2 references the value k1 wrote
store->Put("k3", "HELLO WORLD");     // same dedup key: k3 references it too

// Get returns the stored bytes, which are the first writer's:
std::string v;
store->Get("k2", &v);  // Returns "Hello World"
```

## Important Notes

- Normalization affects **only the dedup key computation**
- The normalized form is never stored. A value is stored as written by the first `Put` that produced its dedup key
- A later `Put` whose value has the same dedup key references that stored value; `Get` on its key returns the first
  writer's bytes, not the bytes that `Put` was given. With `kNone` the two are always identical
- User keys are **not** normalized (they remain opaque byte sequences)
- Values larger than `normalization_max_bytes` skip normalization entirely
