#pragma once

#include <string>
#include <string_view>

namespace prestige {

/** Normalization level for content deduplication.
 *  Normalization is applied ONLY to compute the dedup key;
 *  the original value is always stored as-is.
 */
enum class NormalizationMode {
  kNone,           // No normalization (byte-exact dedup, current default)
  kWhitespace,     // Collapse whitespace runs to single space, trim edges
  kASCII,          // Whitespace + ASCII case folding (A-Z -> a-z)
  kUnicode         // Whitespace + ASCII + Unicode-lite (common ligatures,
                   //   superscripts, Latin-1 case folding, etc.)
};

namespace internal {

/**
 * Normalize text according to the specified mode.
 *
 * This function normalizes text for dedup key computation:
 * - kNone: Returns input unchanged
 * - kWhitespace: Collapses whitespace, trims edges
 * - kASCII: + ASCII lowercase (A-Z -> a-z)
 * - kUnicode: + Common Unicode normalizations (ligatures, superscripts, etc.)
 *
 * @param input The text to normalize
 * @param mode The normalization level
 * @return Normalized string (copy of input for kNone)
 */
std::string Normalize(std::string_view input, NormalizationMode mode);

}  // namespace internal
}  // namespace prestige
