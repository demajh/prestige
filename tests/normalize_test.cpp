// Unit tests for prestige/normalize.hpp
// Tests: Text normalization modes (None, Whitespace, ASCII, Unicode)

#include <gtest/gtest.h>

#include <prestige/normalize.hpp>

#include <string>

namespace prestige::internal {
namespace {

// =============================================================================
// NormalizationMode::kNone Tests
// =============================================================================

class NormalizeNoneTest : public ::testing::Test {};

TEST_F(NormalizeNoneTest, EmptyString) {
  EXPECT_EQ(Normalize("", NormalizationMode::kNone), "");
}

TEST_F(NormalizeNoneTest, PreservesExactBytes) {
  std::string input = "Hello World!\n\t  Multiple   Spaces";
  EXPECT_EQ(Normalize(input, NormalizationMode::kNone), input);
}

TEST_F(NormalizeNoneTest, PreservesBinaryData) {
  std::string binary;
  for (int i = 0; i < 256; ++i) {
    binary.push_back(static_cast<char>(i));
  }
  EXPECT_EQ(Normalize(binary, NormalizationMode::kNone), binary);
}

// =============================================================================
// NormalizationMode::kWhitespace Tests
// =============================================================================

class NormalizeWhitespaceTest : public ::testing::Test {};

TEST_F(NormalizeWhitespaceTest, EmptyString) {
  EXPECT_EQ(Normalize("", NormalizationMode::kWhitespace), "");
}

TEST_F(NormalizeWhitespaceTest, TrimsLeadingWhitespace) {
  EXPECT_EQ(Normalize("   hello", NormalizationMode::kWhitespace), "hello");
  EXPECT_EQ(Normalize("\t\nhello", NormalizationMode::kWhitespace), "hello");
}

TEST_F(NormalizeWhitespaceTest, TrimsTrailingWhitespace) {
  EXPECT_EQ(Normalize("hello   ", NormalizationMode::kWhitespace), "hello");
  EXPECT_EQ(Normalize("hello\t\n", NormalizationMode::kWhitespace), "hello");
}

TEST_F(NormalizeWhitespaceTest, CollapsesInternalWhitespace) {
  EXPECT_EQ(Normalize("hello   world", NormalizationMode::kWhitespace), "hello world");
  EXPECT_EQ(Normalize("a\t\t\tb", NormalizationMode::kWhitespace), "a b");
  EXPECT_EQ(Normalize("a\n\nb", NormalizationMode::kWhitespace), "a b");
}

TEST_F(NormalizeWhitespaceTest, PreservesCase) {
  EXPECT_EQ(Normalize("Hello World", NormalizationMode::kWhitespace), "Hello World");
  EXPECT_EQ(Normalize("UPPER lower", NormalizationMode::kWhitespace), "UPPER lower");
}

TEST_F(NormalizeWhitespaceTest, AllWhitespace) {
  EXPECT_EQ(Normalize("   ", NormalizationMode::kWhitespace), "");
  EXPECT_EQ(Normalize("\t\n\r\v\f", NormalizationMode::kWhitespace), "");
}

TEST_F(NormalizeWhitespaceTest, WhitespaceTypes) {
  // All ASCII whitespace characters
  EXPECT_EQ(Normalize("a b", NormalizationMode::kWhitespace), "a b");    // space
  EXPECT_EQ(Normalize("a\tb", NormalizationMode::kWhitespace), "a b");   // tab
  EXPECT_EQ(Normalize("a\nb", NormalizationMode::kWhitespace), "a b");   // newline
  EXPECT_EQ(Normalize("a\rb", NormalizationMode::kWhitespace), "a b");   // carriage return
  EXPECT_EQ(Normalize("a\vb", NormalizationMode::kWhitespace), "a b");   // vertical tab
  EXPECT_EQ(Normalize("a\fb", NormalizationMode::kWhitespace), "a b");   // form feed
}

// =============================================================================
// NormalizationMode::kASCII Tests
// =============================================================================

class NormalizeASCIITest : public ::testing::Test {};

TEST_F(NormalizeASCIITest, EmptyString) {
  EXPECT_EQ(Normalize("", NormalizationMode::kASCII), "");
}

TEST_F(NormalizeASCIITest, LowercasesASCII) {
  EXPECT_EQ(Normalize("Hello", NormalizationMode::kASCII), "hello");
  EXPECT_EQ(Normalize("HELLO", NormalizationMode::kASCII), "hello");
  EXPECT_EQ(Normalize("HeLLo WoRLd", NormalizationMode::kASCII), "hello world");
}

TEST_F(NormalizeASCIITest, PreservesNonAlpha) {
  EXPECT_EQ(Normalize("Hello123!", NormalizationMode::kASCII), "hello123!");
  EXPECT_EQ(Normalize("A-B_C", NormalizationMode::kASCII), "a-b_c");
}

TEST_F(NormalizeASCIITest, AlsoNormalizesWhitespace) {
  EXPECT_EQ(Normalize("  Hello   World  ", NormalizationMode::kASCII), "hello world");
}

TEST_F(NormalizeASCIITest, AllASCIIUppercase) {
  std::string upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::string lower = "abcdefghijklmnopqrstuvwxyz";
  EXPECT_EQ(Normalize(upper, NormalizationMode::kASCII), lower);
}

TEST_F(NormalizeASCIITest, PreservesNonASCII) {
  // Non-ASCII bytes should pass through unchanged in kASCII mode
  // UTF-8 for "Héllo" (H é l l o)
  std::string input = "H\xc3\xa9llo";  // é is 0xC3 0xA9 in UTF-8
  std::string expected = "h\xc3\xa9llo";
  EXPECT_EQ(Normalize(input, NormalizationMode::kASCII), expected);
}

// =============================================================================
// NormalizationMode::kUnicode Tests
// =============================================================================

class NormalizeUnicodeTest : public ::testing::Test {};

TEST_F(NormalizeUnicodeTest, EmptyString) {
  EXPECT_EQ(Normalize("", NormalizationMode::kUnicode), "");
}

TEST_F(NormalizeUnicodeTest, AlsoDoesASCIIAndWhitespace) {
  EXPECT_EQ(Normalize("  HELLO   World  ", NormalizationMode::kUnicode), "hello world");
}

TEST_F(NormalizeUnicodeTest, SuperscriptDigits) {
  // Superscript 1, 2, 3 (Latin-1 supplement)
  EXPECT_EQ(Normalize("\xc2\xb9", NormalizationMode::kUnicode), "1");  // ¹
  EXPECT_EQ(Normalize("\xc2\xb2", NormalizationMode::kUnicode), "2");  // ²
  EXPECT_EQ(Normalize("\xc2\xb3", NormalizationMode::kUnicode), "3");  // ³
}

TEST_F(NormalizeUnicodeTest, Fractions) {
  // Vulgar fractions
  EXPECT_EQ(Normalize("\xc2\xbc", NormalizationMode::kUnicode), "1/4");  // ¼
  EXPECT_EQ(Normalize("\xc2\xbd", NormalizationMode::kUnicode), "1/2");  // ½
  EXPECT_EQ(Normalize("\xc2\xbe", NormalizationMode::kUnicode), "3/4");  // ¾
}

TEST_F(NormalizeUnicodeTest, LatinLigatures) {
  // ff, fi, fl, ffi, ffl ligatures (U+FB00-FB04)
  EXPECT_EQ(Normalize("\xef\xac\x80", NormalizationMode::kUnicode), "ff");   // ff ligature
  EXPECT_EQ(Normalize("\xef\xac\x81", NormalizationMode::kUnicode), "fi");   // fi ligature
  EXPECT_EQ(Normalize("\xef\xac\x82", NormalizationMode::kUnicode), "fl");   // fl ligature
  EXPECT_EQ(Normalize("\xef\xac\x83", NormalizationMode::kUnicode), "ffi");  // ffi ligature
  EXPECT_EQ(Normalize("\xef\xac\x84", NormalizationMode::kUnicode), "ffl");  // ffl ligature
}

TEST_F(NormalizeUnicodeTest, FullwidthASCII) {
  // Fullwidth A -> a, fullwidth 1 -> 1
  EXPECT_EQ(Normalize("\xef\xbc\xa1", NormalizationMode::kUnicode), "a");  // Fullwidth A
  EXPECT_EQ(Normalize("\xef\xbc\x91", NormalizationMode::kUnicode), "1");  // Fullwidth 1
}

TEST_F(NormalizeUnicodeTest, Latin1CaseFolding) {
  // Uppercase accented letters should be lowercased
  // À (U+00C0) -> à (U+00E0)
  // Both are 2-byte UTF-8: 0xC3 0x80 -> 0xC3 0xA0
  EXPECT_EQ(Normalize("\xc3\x80", NormalizationMode::kUnicode), "\xc3\xa0");  // À -> à

  // Ü (U+00DC) -> ü (U+00FC)
  EXPECT_EQ(Normalize("\xc3\x9c", NormalizationMode::kUnicode), "\xc3\xbc");  // Ü -> ü
}

TEST_F(NormalizeUnicodeTest, MixedContent) {
  // "½ café"
  std::string input = "\xc2\xbd caf\xc3\xa9";
  std::string expected = "1/2 caf\xc3\xa9";
  EXPECT_EQ(Normalize(input, NormalizationMode::kUnicode), expected);
}

TEST_F(NormalizeUnicodeTest, UnmappedUTFPassesThrough) {
  // Chinese character 中 (U+4E2D) should pass through unchanged
  std::string chinese = "\xe4\xb8\xad";
  EXPECT_EQ(Normalize(chinese, NormalizationMode::kUnicode), chinese);
}

TEST_F(NormalizeUnicodeTest, SuperscriptMathSymbols) {
  // Superscript + - = ( )
  EXPECT_EQ(Normalize("\xe2\x81\xba", NormalizationMode::kUnicode), "+");  // ⁺
  EXPECT_EQ(Normalize("\xe2\x81\xbb", NormalizationMode::kUnicode), "-");  // ⁻
  EXPECT_EQ(Normalize("\xe2\x81\xbc", NormalizationMode::kUnicode), "=");  // ⁼
}

TEST_F(NormalizeUnicodeTest, SubscriptDigits) {
  // Subscript 0-9 (U+2080-2089)
  EXPECT_EQ(Normalize("\xe2\x82\x80", NormalizationMode::kUnicode), "0");  // ₀
  EXPECT_EQ(Normalize("\xe2\x82\x81", NormalizationMode::kUnicode), "1");  // ₁
  EXPECT_EQ(Normalize("\xe2\x82\x89", NormalizationMode::kUnicode), "9");  // ₉
}

// =============================================================================
// Edge Cases
// =============================================================================

class NormalizeEdgeCasesTest : public ::testing::Test {};

TEST_F(NormalizeEdgeCasesTest, SingleCharacter) {
  EXPECT_EQ(Normalize("a", NormalizationMode::kWhitespace), "a");
  EXPECT_EQ(Normalize("A", NormalizationMode::kASCII), "a");
  EXPECT_EQ(Normalize(" ", NormalizationMode::kWhitespace), "");
}

TEST_F(NormalizeEdgeCasesTest, OnlyPunctuation) {
  EXPECT_EQ(Normalize("!@#$%", NormalizationMode::kASCII), "!@#$%");
}

TEST_F(NormalizeEdgeCasesTest, LongString) {
  std::string input(10000, 'A');
  std::string expected(10000, 'a');
  EXPECT_EQ(Normalize(input, NormalizationMode::kASCII), expected);
}

TEST_F(NormalizeEdgeCasesTest, AlternatingWhitespace) {
  EXPECT_EQ(Normalize("a b c d e", NormalizationMode::kWhitespace), "a b c d e");
}

TEST_F(NormalizeEdgeCasesTest, InvalidUTF8Passthrough) {
  // Invalid UTF-8 sequence should pass through (treated as single bytes)
  std::string invalid = "\xff\xfe";
  // In kNone mode, passes through
  EXPECT_EQ(Normalize(invalid, NormalizationMode::kNone), invalid);
  // In other modes, also passes through (no mapping found)
  std::string result = Normalize(invalid, NormalizationMode::kUnicode);
  EXPECT_EQ(result.size(), 2u);
}

// =============================================================================
// Idempotency Tests
// =============================================================================

class NormalizeIdempotencyTest : public ::testing::Test {};

TEST_F(NormalizeIdempotencyTest, Idempotent) {
  std::vector<std::string> inputs = {"Hello World", "  multiple   spaces  ", "UPPERCASE",
                                     "MiXeD CaSe", "\xc2\xbd caf\xc3\xa9"};

  for (NormalizationMode mode : {NormalizationMode::kWhitespace, NormalizationMode::kASCII,
                                 NormalizationMode::kUnicode}) {
    for (const auto& input : inputs) {
      std::string once = Normalize(input, mode);
      std::string twice = Normalize(once, mode);
      EXPECT_EQ(once, twice) << "Not idempotent for input: " << input;
    }
  }
}

// =============================================================================
// Deduplication Equivalence Tests
// =============================================================================

class NormalizeDedupTest : public ::testing::Test {};

TEST_F(NormalizeDedupTest, WhitespaceVariantsMatch) {
  std::string a = "hello world";
  std::string b = "hello  world";
  std::string c = "  hello world  ";
  std::string d = "hello\tworld";

  auto na = Normalize(a, NormalizationMode::kWhitespace);
  auto nb = Normalize(b, NormalizationMode::kWhitespace);
  auto nc = Normalize(c, NormalizationMode::kWhitespace);
  auto nd = Normalize(d, NormalizationMode::kWhitespace);

  EXPECT_EQ(na, nb);
  EXPECT_EQ(na, nc);
  EXPECT_EQ(na, nd);
}

TEST_F(NormalizeDedupTest, CaseVariantsMatch) {
  std::string a = "hello world";
  std::string b = "HELLO WORLD";
  std::string c = "Hello World";
  std::string d = "hElLo WoRlD";

  auto na = Normalize(a, NormalizationMode::kASCII);
  auto nb = Normalize(b, NormalizationMode::kASCII);
  auto nc = Normalize(c, NormalizationMode::kASCII);
  auto nd = Normalize(d, NormalizationMode::kASCII);

  EXPECT_EQ(na, nb);
  EXPECT_EQ(na, nc);
  EXPECT_EQ(na, nd);
}

TEST_F(NormalizeDedupTest, LigatureVariantsMatch) {
  // "office" with ff ligature vs normal
  std::string with_ligature = "o\xef\xac\x80ice";  // oﬀice
  std::string without = "office";

  auto na = Normalize(with_ligature, NormalizationMode::kUnicode);
  auto nb = Normalize(without, NormalizationMode::kUnicode);

  EXPECT_EQ(na, nb);
}

}  // namespace
}  // namespace prestige::internal
