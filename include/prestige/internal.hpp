#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <thread>

#include <rocksdb/status.h>

namespace prestige::internal {

class Sha256 {
 public:
  static constexpr size_t kDigestBytes = 32;

  static std::array<uint8_t, kDigestBytes> Digest(std::string_view data) {
    Sha256 ctx;
    ctx.Update(reinterpret_cast<const uint8_t*>(data.data()), data.size());
    return ctx.Final();
  }

 private:
  static constexpr uint32_t kInitH[8] = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
  };

  static constexpr uint32_t kK[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
  };

  static inline uint32_t ROTR(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }
  static inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
  static inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
  static inline uint32_t Sig0(uint32_t x) { return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22); }
  static inline uint32_t Sig1(uint32_t x) { return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25); }
  static inline uint32_t sig0(uint32_t x) { return ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3); }
  static inline uint32_t sig1(uint32_t x) { return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10); }

  Sha256() {
    std::memcpy(h_, kInitH, sizeof(h_));
    bit_len_ = 0;
    buf_len_ = 0;
  }

  void Update(const uint8_t* data, size_t len) {
    bit_len_ += static_cast<uint64_t>(len) * 8ull;

    if (buf_len_ > 0) {
      size_t take = (len < (64 - buf_len_)) ? len : (64 - buf_len_);
      std::memcpy(buf_ + buf_len_, data, take);
      buf_len_ += take;
      data += take;
      len -= take;
      if (buf_len_ == 64) {
        Compress(buf_);
        buf_len_ = 0;
      }
    }

    while (len >= 64) {
      Compress(data);
      data += 64;
      len -= 64;
    }

    if (len > 0) {
      std::memcpy(buf_, data, len);
      buf_len_ = len;
    }
  }

  std::array<uint8_t, kDigestBytes> Final() {
    uint8_t pad[128];
    size_t pad_len = 0;

    pad[pad_len++] = 0x80u;
    size_t rem = (buf_len_ + pad_len) % 64;
    size_t zeros = (rem <= 56) ? (56 - rem) : (56 + 64 - rem);
    std::memset(pad + pad_len, 0, zeros);
    pad_len += zeros;

    uint64_t bl = bit_len_;
    for (int i = 7; i >= 0; --i) {
      pad[pad_len++] = static_cast<uint8_t>((bl >> (i * 8)) & 0xffu);
    }

    Update(pad, pad_len);
    assert(buf_len_ == 0);

    std::array<uint8_t, kDigestBytes> out{};
    for (int i = 0; i < 8; ++i) {
      out[i * 4 + 0] = static_cast<uint8_t>((h_[i] >> 24) & 0xffu);
      out[i * 4 + 1] = static_cast<uint8_t>((h_[i] >> 16) & 0xffu);
      out[i * 4 + 2] = static_cast<uint8_t>((h_[i] >> 8) & 0xffu);
      out[i * 4 + 3] = static_cast<uint8_t>((h_[i] >> 0) & 0xffu);
    }
    return out;
  }

  void Compress(const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      uint32_t b0 = block[i * 4 + 0];
      uint32_t b1 = block[i * 4 + 1];
      uint32_t b2 = block[i * 4 + 2];
      uint32_t b3 = block[i * 4 + 3];
      w[i] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    }
    for (int i = 16; i < 64; ++i) {
      w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];

    for (int i = 0; i < 64; ++i) {
      uint32_t t1 = hh + Sig1(e) + Ch(e, f, g) + kK[i] + w[i];
      uint32_t t2 = Sig0(a) + Maj(a, b, c);
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }

    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
  }

  uint32_t h_[8];
  uint64_t bit_len_;
  uint8_t buf_[64];
  size_t buf_len_;
};

inline std::string ToBytes(const uint8_t* p, size_t n) {
  return std::string(reinterpret_cast<const char*>(p), n);
}

inline std::string EncodeU64LE(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) {
    s[i] = static_cast<char>(v & 0xffu);
    v >>= 8;
  }
  return s;
}

inline bool DecodeU64LE(std::string_view s, uint64_t* out) {
  if (s.size() != 8) return false;
  uint64_t v = 0;
  // little endian decode
  for (int i = 7; i >= 0; --i) {
    v <<= 8;
    v |= static_cast<uint8_t>(s[static_cast<size_t>(i)]);
  }
  *out = v;
  return true;
}

inline std::array<uint8_t, 16> RandomObjectId128() {
  thread_local std::mt19937_64 rng([]{
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    seed ^= static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return seed;
  }());

  std::array<uint8_t, 16> id{};
  uint64_t a = rng();
  uint64_t b = rng();
  std::memcpy(id.data() + 0, &a, 8);
  std::memcpy(id.data() + 8, &b, 8);
  return id;
}

inline bool IsRetryableTxnStatus(const rocksdb::Status& s) {
  return s.IsBusy() || s.IsTimedOut() || s.IsTryAgain() || s.IsAborted();
}

}  // namespace prestige::internal
