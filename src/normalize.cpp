#include <prestige/normalize.hpp>

#include <cstdint>
#include <cstring>

namespace prestige::internal {

namespace {

// ASCII case folding table [0-127]: maps A-Z to a-z, everything else unchanged
constexpr char kASCIILower[128] = {
    // 0-31: control chars -> unchanged
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
    16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    // 32-63: space, punctuation, digits -> unchanged
    ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
    // 64-95: @, A-Z -> @, a-z, then [\]^_
    '@', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '[', '\\', ']', '^', '_',
    // 96-127: `, a-z -> unchanged, then {|}~ DEL
    '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 127
};

// Latin-1 Supplement case folding (bytes 0xC0-0xDE in Latin-1 map to 0xE0-0xFE)
// Index: (byte - 0xC0), returns lowercase equivalent or same byte
// Only applies to 2-byte UTF-8 sequences starting with 0xC3
constexpr uint8_t kLatin1CaseFold[32] = {
    // 0xC0-0xCF: A-grave through I-umlaut uppercase -> lowercase (+0x20)
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,  // A-grave..C-cedilla
    0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF,  // E-grave..I-umlaut
    // 0xD0-0xDF: D-eth through Y-acute, then special chars
    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xD7,  // D-eth..O-umlaut, multiplication (unchanged)
    0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xDF   // O-stroke..Thorn, Y-acute, eszett (unchanged)
};

// Unicode decomposition mappings for common multi-byte sequences
// Format: { src_bytes, src_len, dst_bytes, dst_len }
struct UnicodeMapping {
    uint8_t src[4];
    uint8_t src_len;
    uint8_t dst[4];
    uint8_t dst_len;
};

// Sorted by first byte for faster rejection
constexpr UnicodeMapping kUnicodeMappings[] = {
    // Latin-1 supplement fractions (0xC2 prefix)
    {{0xC2, 0xB2, 0, 0}, 2, {'2', 0, 0, 0}, 1},           // superscript 2
    {{0xC2, 0xB3, 0, 0}, 2, {'3', 0, 0, 0}, 1},           // superscript 3
    {{0xC2, 0xB9, 0, 0}, 2, {'1', 0, 0, 0}, 1},           // superscript 1
    {{0xC2, 0xBC, 0, 0}, 2, {'1', '/', '4', 0}, 3},       // 1/4
    {{0xC2, 0xBD, 0, 0}, 2, {'1', '/', '2', 0}, 3},       // 1/2
    {{0xC2, 0xBE, 0, 0}, 2, {'3', '/', '4', 0}, 3},       // 3/4

    // Superscripts U+2070-207F (0xE2 0x81 prefix)
    {{0xE2, 0x81, 0xB0, 0}, 3, {'0', 0, 0, 0}, 1},        // superscript 0
    {{0xE2, 0x81, 0xB4, 0}, 3, {'4', 0, 0, 0}, 1},        // superscript 4
    {{0xE2, 0x81, 0xB5, 0}, 3, {'5', 0, 0, 0}, 1},        // superscript 5
    {{0xE2, 0x81, 0xB6, 0}, 3, {'6', 0, 0, 0}, 1},        // superscript 6
    {{0xE2, 0x81, 0xB7, 0}, 3, {'7', 0, 0, 0}, 1},        // superscript 7
    {{0xE2, 0x81, 0xB8, 0}, 3, {'8', 0, 0, 0}, 1},        // superscript 8
    {{0xE2, 0x81, 0xB9, 0}, 3, {'9', 0, 0, 0}, 1},        // superscript 9
    {{0xE2, 0x81, 0xBA, 0}, 3, {'+', 0, 0, 0}, 1},        // superscript +
    {{0xE2, 0x81, 0xBB, 0}, 3, {'-', 0, 0, 0}, 1},        // superscript -
    {{0xE2, 0x81, 0xBC, 0}, 3, {'=', 0, 0, 0}, 1},        // superscript =
    {{0xE2, 0x81, 0xBD, 0}, 3, {'(', 0, 0, 0}, 1},        // superscript (
    {{0xE2, 0x81, 0xBE, 0}, 3, {')', 0, 0, 0}, 1},        // superscript )
    {{0xE2, 0x81, 0xBF, 0}, 3, {'n', 0, 0, 0}, 1},        // superscript n

    // Subscripts U+2080-208E (0xE2 0x82 prefix)
    {{0xE2, 0x82, 0x80, 0}, 3, {'0', 0, 0, 0}, 1},        // subscript 0
    {{0xE2, 0x82, 0x81, 0}, 3, {'1', 0, 0, 0}, 1},        // subscript 1
    {{0xE2, 0x82, 0x82, 0}, 3, {'2', 0, 0, 0}, 1},        // subscript 2
    {{0xE2, 0x82, 0x83, 0}, 3, {'3', 0, 0, 0}, 1},        // subscript 3
    {{0xE2, 0x82, 0x84, 0}, 3, {'4', 0, 0, 0}, 1},        // subscript 4
    {{0xE2, 0x82, 0x85, 0}, 3, {'5', 0, 0, 0}, 1},        // subscript 5
    {{0xE2, 0x82, 0x86, 0}, 3, {'6', 0, 0, 0}, 1},        // subscript 6
    {{0xE2, 0x82, 0x87, 0}, 3, {'7', 0, 0, 0}, 1},        // subscript 7
    {{0xE2, 0x82, 0x88, 0}, 3, {'8', 0, 0, 0}, 1},        // subscript 8
    {{0xE2, 0x82, 0x89, 0}, 3, {'9', 0, 0, 0}, 1},        // subscript 9
    {{0xE2, 0x82, 0x8A, 0}, 3, {'+', 0, 0, 0}, 1},        // subscript +
    {{0xE2, 0x82, 0x8B, 0}, 3, {'-', 0, 0, 0}, 1},        // subscript -
    {{0xE2, 0x82, 0x8C, 0}, 3, {'=', 0, 0, 0}, 1},        // subscript =
    {{0xE2, 0x82, 0x8D, 0}, 3, {'(', 0, 0, 0}, 1},        // subscript (
    {{0xE2, 0x82, 0x8E, 0}, 3, {')', 0, 0, 0}, 1},        // subscript )

    // Latin ligatures U+FB00-FB06 (0xEF 0xAC prefix)
    {{0xEF, 0xAC, 0x80, 0}, 3, {'f', 'f', 0, 0}, 2},      // ff
    {{0xEF, 0xAC, 0x81, 0}, 3, {'f', 'i', 0, 0}, 2},      // fi
    {{0xEF, 0xAC, 0x82, 0}, 3, {'f', 'l', 0, 0}, 2},      // fl
    {{0xEF, 0xAC, 0x83, 0}, 3, {'f', 'f', 'i', 0}, 3},    // ffi
    {{0xEF, 0xAC, 0x84, 0}, 3, {'f', 'f', 'l', 0}, 3},    // ffl
    {{0xEF, 0xAC, 0x85, 0}, 3, {'s', 't', 0, 0}, 2},      // st (long s + t)
    {{0xEF, 0xAC, 0x86, 0}, 3, {'s', 't', 0, 0}, 2},      // st

    // Fullwidth ASCII U+FF01-FF5E -> U+0021-007E (0xEF 0xBC/0xBD prefix)
    // Punctuation and digits (0xEF 0xBC 0x81-0xBF)
    {{0xEF, 0xBC, 0x81, 0}, 3, {'!', 0, 0, 0}, 1},        // fullwidth !
    {{0xEF, 0xBC, 0x82, 0}, 3, {'"', 0, 0, 0}, 1},        // fullwidth "
    {{0xEF, 0xBC, 0x83, 0}, 3, {'#', 0, 0, 0}, 1},        // fullwidth #
    {{0xEF, 0xBC, 0x84, 0}, 3, {'$', 0, 0, 0}, 1},        // fullwidth $
    {{0xEF, 0xBC, 0x85, 0}, 3, {'%', 0, 0, 0}, 1},        // fullwidth %
    {{0xEF, 0xBC, 0x86, 0}, 3, {'&', 0, 0, 0}, 1},        // fullwidth &
    {{0xEF, 0xBC, 0x87, 0}, 3, {'\'', 0, 0, 0}, 1},       // fullwidth '
    {{0xEF, 0xBC, 0x88, 0}, 3, {'(', 0, 0, 0}, 1},        // fullwidth (
    {{0xEF, 0xBC, 0x89, 0}, 3, {')', 0, 0, 0}, 1},        // fullwidth )
    {{0xEF, 0xBC, 0x8A, 0}, 3, {'*', 0, 0, 0}, 1},        // fullwidth *
    {{0xEF, 0xBC, 0x8B, 0}, 3, {'+', 0, 0, 0}, 1},        // fullwidth +
    {{0xEF, 0xBC, 0x8C, 0}, 3, {',', 0, 0, 0}, 1},        // fullwidth ,
    {{0xEF, 0xBC, 0x8D, 0}, 3, {'-', 0, 0, 0}, 1},        // fullwidth -
    {{0xEF, 0xBC, 0x8E, 0}, 3, {'.', 0, 0, 0}, 1},        // fullwidth .
    {{0xEF, 0xBC, 0x8F, 0}, 3, {'/', 0, 0, 0}, 1},        // fullwidth /
    {{0xEF, 0xBC, 0x90, 0}, 3, {'0', 0, 0, 0}, 1},        // fullwidth 0
    {{0xEF, 0xBC, 0x91, 0}, 3, {'1', 0, 0, 0}, 1},        // fullwidth 1
    {{0xEF, 0xBC, 0x92, 0}, 3, {'2', 0, 0, 0}, 1},        // fullwidth 2
    {{0xEF, 0xBC, 0x93, 0}, 3, {'3', 0, 0, 0}, 1},        // fullwidth 3
    {{0xEF, 0xBC, 0x94, 0}, 3, {'4', 0, 0, 0}, 1},        // fullwidth 4
    {{0xEF, 0xBC, 0x95, 0}, 3, {'5', 0, 0, 0}, 1},        // fullwidth 5
    {{0xEF, 0xBC, 0x96, 0}, 3, {'6', 0, 0, 0}, 1},        // fullwidth 6
    {{0xEF, 0xBC, 0x97, 0}, 3, {'7', 0, 0, 0}, 1},        // fullwidth 7
    {{0xEF, 0xBC, 0x98, 0}, 3, {'8', 0, 0, 0}, 1},        // fullwidth 8
    {{0xEF, 0xBC, 0x99, 0}, 3, {'9', 0, 0, 0}, 1},        // fullwidth 9
    {{0xEF, 0xBC, 0x9A, 0}, 3, {':', 0, 0, 0}, 1},        // fullwidth :
    {{0xEF, 0xBC, 0x9B, 0}, 3, {';', 0, 0, 0}, 1},        // fullwidth ;
    {{0xEF, 0xBC, 0x9C, 0}, 3, {'<', 0, 0, 0}, 1},        // fullwidth <
    {{0xEF, 0xBC, 0x9D, 0}, 3, {'=', 0, 0, 0}, 1},        // fullwidth =
    {{0xEF, 0xBC, 0x9E, 0}, 3, {'>', 0, 0, 0}, 1},        // fullwidth >
    {{0xEF, 0xBC, 0x9F, 0}, 3, {'?', 0, 0, 0}, 1},        // fullwidth ?
    {{0xEF, 0xBC, 0xA0, 0}, 3, {'@', 0, 0, 0}, 1},        // fullwidth @
    // Fullwidth A-Z (map to lowercase a-z)
    {{0xEF, 0xBC, 0xA1, 0}, 3, {'a', 0, 0, 0}, 1},        // fullwidth A -> a
    {{0xEF, 0xBC, 0xA2, 0}, 3, {'b', 0, 0, 0}, 1},        // fullwidth B -> b
    {{0xEF, 0xBC, 0xA3, 0}, 3, {'c', 0, 0, 0}, 1},        // fullwidth C -> c
    {{0xEF, 0xBC, 0xA4, 0}, 3, {'d', 0, 0, 0}, 1},        // fullwidth D -> d
    {{0xEF, 0xBC, 0xA5, 0}, 3, {'e', 0, 0, 0}, 1},        // fullwidth E -> e
    {{0xEF, 0xBC, 0xA6, 0}, 3, {'f', 0, 0, 0}, 1},        // fullwidth F -> f
    {{0xEF, 0xBC, 0xA7, 0}, 3, {'g', 0, 0, 0}, 1},        // fullwidth G -> g
    {{0xEF, 0xBC, 0xA8, 0}, 3, {'h', 0, 0, 0}, 1},        // fullwidth H -> h
    {{0xEF, 0xBC, 0xA9, 0}, 3, {'i', 0, 0, 0}, 1},        // fullwidth I -> i
    {{0xEF, 0xBC, 0xAA, 0}, 3, {'j', 0, 0, 0}, 1},        // fullwidth J -> j
    {{0xEF, 0xBC, 0xAB, 0}, 3, {'k', 0, 0, 0}, 1},        // fullwidth K -> k
    {{0xEF, 0xBC, 0xAC, 0}, 3, {'l', 0, 0, 0}, 1},        // fullwidth L -> l
    {{0xEF, 0xBC, 0xAD, 0}, 3, {'m', 0, 0, 0}, 1},        // fullwidth M -> m
    {{0xEF, 0xBC, 0xAE, 0}, 3, {'n', 0, 0, 0}, 1},        // fullwidth N -> n
    {{0xEF, 0xBC, 0xAF, 0}, 3, {'o', 0, 0, 0}, 1},        // fullwidth O -> o
    {{0xEF, 0xBC, 0xB0, 0}, 3, {'p', 0, 0, 0}, 1},        // fullwidth P -> p
    {{0xEF, 0xBC, 0xB1, 0}, 3, {'q', 0, 0, 0}, 1},        // fullwidth Q -> q
    {{0xEF, 0xBC, 0xB2, 0}, 3, {'r', 0, 0, 0}, 1},        // fullwidth R -> r
    {{0xEF, 0xBC, 0xB3, 0}, 3, {'s', 0, 0, 0}, 1},        // fullwidth S -> s
    {{0xEF, 0xBC, 0xB4, 0}, 3, {'t', 0, 0, 0}, 1},        // fullwidth T -> t
    {{0xEF, 0xBC, 0xB5, 0}, 3, {'u', 0, 0, 0}, 1},        // fullwidth U -> u
    {{0xEF, 0xBC, 0xB6, 0}, 3, {'v', 0, 0, 0}, 1},        // fullwidth V -> v
    {{0xEF, 0xBC, 0xB7, 0}, 3, {'w', 0, 0, 0}, 1},        // fullwidth W -> w
    {{0xEF, 0xBC, 0xB8, 0}, 3, {'x', 0, 0, 0}, 1},        // fullwidth X -> x
    {{0xEF, 0xBC, 0xB9, 0}, 3, {'y', 0, 0, 0}, 1},        // fullwidth Y -> y
    {{0xEF, 0xBC, 0xBA, 0}, 3, {'z', 0, 0, 0}, 1},        // fullwidth Z -> z
    {{0xEF, 0xBC, 0xBB, 0}, 3, {'[', 0, 0, 0}, 1},        // fullwidth [
    {{0xEF, 0xBC, 0xBC, 0}, 3, {'\\', 0, 0, 0}, 1},       // fullwidth backslash
    {{0xEF, 0xBC, 0xBD, 0}, 3, {']', 0, 0, 0}, 1},        // fullwidth ]
    {{0xEF, 0xBC, 0xBE, 0}, 3, {'^', 0, 0, 0}, 1},        // fullwidth ^
    {{0xEF, 0xBC, 0xBF, 0}, 3, {'_', 0, 0, 0}, 1},        // fullwidth _
    // Fullwidth ` and a-z (0xEF 0xBD prefix)
    {{0xEF, 0xBD, 0x80, 0}, 3, {'`', 0, 0, 0}, 1},        // fullwidth `
    {{0xEF, 0xBD, 0x81, 0}, 3, {'a', 0, 0, 0}, 1},        // fullwidth a
    {{0xEF, 0xBD, 0x82, 0}, 3, {'b', 0, 0, 0}, 1},        // fullwidth b
    {{0xEF, 0xBD, 0x83, 0}, 3, {'c', 0, 0, 0}, 1},        // fullwidth c
    {{0xEF, 0xBD, 0x84, 0}, 3, {'d', 0, 0, 0}, 1},        // fullwidth d
    {{0xEF, 0xBD, 0x85, 0}, 3, {'e', 0, 0, 0}, 1},        // fullwidth e
    {{0xEF, 0xBD, 0x86, 0}, 3, {'f', 0, 0, 0}, 1},        // fullwidth f
    {{0xEF, 0xBD, 0x87, 0}, 3, {'g', 0, 0, 0}, 1},        // fullwidth g
    {{0xEF, 0xBD, 0x88, 0}, 3, {'h', 0, 0, 0}, 1},        // fullwidth h
    {{0xEF, 0xBD, 0x89, 0}, 3, {'i', 0, 0, 0}, 1},        // fullwidth i
    {{0xEF, 0xBD, 0x8A, 0}, 3, {'j', 0, 0, 0}, 1},        // fullwidth j
    {{0xEF, 0xBD, 0x8B, 0}, 3, {'k', 0, 0, 0}, 1},        // fullwidth k
    {{0xEF, 0xBD, 0x8C, 0}, 3, {'l', 0, 0, 0}, 1},        // fullwidth l
    {{0xEF, 0xBD, 0x8D, 0}, 3, {'m', 0, 0, 0}, 1},        // fullwidth m
    {{0xEF, 0xBD, 0x8E, 0}, 3, {'n', 0, 0, 0}, 1},        // fullwidth n
    {{0xEF, 0xBD, 0x8F, 0}, 3, {'o', 0, 0, 0}, 1},        // fullwidth o
    {{0xEF, 0xBD, 0x90, 0}, 3, {'p', 0, 0, 0}, 1},        // fullwidth p
    {{0xEF, 0xBD, 0x91, 0}, 3, {'q', 0, 0, 0}, 1},        // fullwidth q
    {{0xEF, 0xBD, 0x92, 0}, 3, {'r', 0, 0, 0}, 1},        // fullwidth r
    {{0xEF, 0xBD, 0x93, 0}, 3, {'s', 0, 0, 0}, 1},        // fullwidth s
    {{0xEF, 0xBD, 0x94, 0}, 3, {'t', 0, 0, 0}, 1},        // fullwidth t
    {{0xEF, 0xBD, 0x95, 0}, 3, {'u', 0, 0, 0}, 1},        // fullwidth u
    {{0xEF, 0xBD, 0x96, 0}, 3, {'v', 0, 0, 0}, 1},        // fullwidth v
    {{0xEF, 0xBD, 0x97, 0}, 3, {'w', 0, 0, 0}, 1},        // fullwidth w
    {{0xEF, 0xBD, 0x98, 0}, 3, {'x', 0, 0, 0}, 1},        // fullwidth x
    {{0xEF, 0xBD, 0x99, 0}, 3, {'y', 0, 0, 0}, 1},        // fullwidth y
    {{0xEF, 0xBD, 0x9A, 0}, 3, {'z', 0, 0, 0}, 1},        // fullwidth z
    {{0xEF, 0xBD, 0x9B, 0}, 3, {'{', 0, 0, 0}, 1},        // fullwidth {
    {{0xEF, 0xBD, 0x9C, 0}, 3, {'|', 0, 0, 0}, 1},        // fullwidth |
    {{0xEF, 0xBD, 0x9D, 0}, 3, {'}', 0, 0, 0}, 1},        // fullwidth }
    {{0xEF, 0xBD, 0x9E, 0}, 3, {'~', 0, 0, 0}, 1},        // fullwidth ~
};

constexpr size_t kUnicodeMappingsCount = sizeof(kUnicodeMappings) / sizeof(kUnicodeMappings[0]);

inline bool IsWhitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

inline int UTF8ByteLength(uint8_t first_byte) {
    if ((first_byte & 0x80) == 0) return 1;      // 0xxxxxxx
    if ((first_byte & 0xE0) == 0xC0) return 2;   // 110xxxxx
    if ((first_byte & 0xF0) == 0xE0) return 3;   // 1110xxxx
    if ((first_byte & 0xF8) == 0xF0) return 4;   // 11110xxx
    return 1;  // Invalid, treat as single byte
}

// Linear search for Unicode mapping (table is small enough)
const UnicodeMapping* FindUnicodeMapping(const uint8_t* src, size_t max_len) {
    if (max_len == 0 || src[0] < 0x80) return nullptr;

    for (size_t i = 0; i < kUnicodeMappingsCount; ++i) {
        const auto& m = kUnicodeMappings[i];
        if (max_len >= m.src_len) {
            bool match = true;
            for (size_t j = 0; j < m.src_len && match; ++j) {
                if (src[j] != m.src[j]) match = false;
            }
            if (match) return &m;
        }
    }
    return nullptr;
}

}  // namespace

std::string Normalize(std::string_view input, NormalizationMode mode) {
    if (mode == NormalizationMode::kNone || input.empty()) {
        return std::string(input);
    }

    std::string result;
    result.reserve(input.size());

    bool in_whitespace = true;  // Start true to trim leading whitespace
    const bool needs_case_fold = (mode >= NormalizationMode::kASCII);
    const bool needs_unicode = (mode == NormalizationMode::kUnicode);

    size_t i = 0;
    while (i < input.size()) {
        uint8_t c = static_cast<uint8_t>(input[i]);

        // Whitespace handling (all modes except kNone)
        if (IsWhitespace(static_cast<char>(c))) {
            if (!in_whitespace) {
                result += ' ';
                in_whitespace = true;
            }
            ++i;
            continue;
        }

        in_whitespace = false;

        // ASCII fast path
        if (c < 0x80) {
            if (needs_case_fold) {
                result += kASCIILower[c];
            } else {
                result += static_cast<char>(c);
            }
            ++i;
            continue;
        }

        // Unicode handling
        if (needs_unicode) {
            // Try explicit Unicode decomposition mapping
            const UnicodeMapping* mapping = FindUnicodeMapping(
                reinterpret_cast<const uint8_t*>(input.data() + i),
                input.size() - i);

            if (mapping) {
                for (size_t j = 0; j < mapping->dst_len; ++j) {
                    result += static_cast<char>(mapping->dst[j]);
                }
                i += mapping->src_len;
                continue;
            }

            // Latin-1 case folding for 2-byte UTF-8 (0xC3 0x80-0xBF range)
            // This covers uppercase accented Latin letters
            if (c == 0xC3 && i + 1 < input.size()) {
                uint8_t c2 = static_cast<uint8_t>(input[i + 1]);
                if (c2 >= 0x80 && c2 <= 0x9E) {
                    // Map to lowercase: 0xC3 0x80-0x9E -> 0xC3 0xA0-0xBE
                    uint8_t latin1_offset = c2 - 0x80;  // 0-30
                    if (latin1_offset < 32) {
                        uint8_t lower = kLatin1CaseFold[latin1_offset];
                        if (lower != (0xC0 + latin1_offset)) {
                            // Case changed: encode back to UTF-8
                            result += static_cast<char>(0xC3);
                            result += static_cast<char>(lower - 0x40);
                            i += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // Pass through unrecognized UTF-8 sequences unchanged
        int byte_len = UTF8ByteLength(c);
        for (int j = 0; j < byte_len && i < input.size(); ++j) {
            result += input[i++];
        }
    }

    // Trim trailing whitespace
    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

}  // namespace prestige::internal
