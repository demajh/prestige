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

- **Ligatures**: ﬁ -> fi, ﬂ -> fl, ﬀ -> ff, ﬃ -> ffi, ﬄ -> ffl
- **Superscripts**: 1234567890 -> 1234567890
- **Subscripts**: ₀₁₂₃₄₅₆₇₈₉ -> 0123456789
- **Fractions**: ½ -> 1/2, ¼ -> 1/4, ¾ -> 3/4
- **Fullwidth ASCII**: Ａ-Ｚ -> a-z, ａ-ｚ -> a-z, ０-９ -> 0-9
- **Latin-1 case folding**: A -> a, E -> e, etc.

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

## Important Notes

- Normalization affects **only the dedup key computation**
- Original values are stored and returned unchanged
- User keys are **not** normalized (they remain opaque byte sequences)
- Values larger than `normalization_max_bytes` skip normalization entirely
