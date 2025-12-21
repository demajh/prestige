#pragma once

#define PRESTIGE_VERSION_MAJOR 0
#define PRESTIGE_VERSION_MINOR 1
#define PRESTIGE_VERSION_PATCH 0

#define PRESTIGE_VERSION_STRING "0.1.0"

// For compile-time version checks
#define PRESTIGE_VERSION \
  (PRESTIGE_VERSION_MAJOR * 10000 + PRESTIGE_VERSION_MINOR * 100 + PRESTIGE_VERSION_PATCH)

namespace prestige {

inline const char* Version() { return PRESTIGE_VERSION_STRING; }

}  // namespace prestige
