/***************************************************************************//**
**  Mimamo - Mighty Matrix Module
**      -- a header-only C++17 matrix library with matlab/octave-like interfaces
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2026
**
**  License: BSD 3-Clause, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mmm_zlib.hpp
@brief   deflate and inflate, the zlib and gzip framings, and zlibmt

This file is mimamo's zlib: a compressor, a decompressor, the checksums,
and the two framings around them, with nothing to link. mima's `gzip`,
`unzip` and `zlibencode` have to work in a binary that mimac produced,
which by design links nothing but libc, and the PNG writer and the .pmat
codec need a compressor too.

*The compressor* is a real deflate. It matches with hash chains over a
32 KiB window, greedily at levels 1 to 3 and lazily at 4 to 9, and writes
each block with whichever of dynamic Huffman codes, the fixed codes or a
stored block is smallest. Level 0 is stored blocks. Its output is not
zlib's byte for byte, but it is standard, and ratio and speed are in the
same range.

*The decompressor* is table driven: a ten-bit first-level table for
literals and lengths, eight bits for distances, and second-level tables
for longer codes, read from a 64-bit bit buffer that is refilled eight
bytes at a time. It handles stored, fixed and dynamic blocks, which is all
RFC 1951 defines.

**zlibmt** is how a large payload is compressed on every core while still
being one ordinary zlib stream. The input is cut into blocks (4 MiB by
default); each thread deflates one block and ends it with a full flush --
an empty stored block, the bytes `00 00 FF FF` -- so no block refers back
into another. The blocks are joined behind one zlib header, an empty
final block and one Adler-32 over the whole follow, and any inflate reads
the result. The bytes depend only on the data, the level, the block size
and the engine, never on the thread count. This is the construction of
the author's own zlibmt, in zmat and in pyjdata's `jdata.zlibmt`, and with
the system zlib as the engine the two produce the same bytes.

**Inflating in parallel.** zmat and pyjdata hand a block index to the
caller, which stores it beside the stream (`_ArrayZipOffsets_`), because
a full flush leaves no reliable mark: `00 00 FF FF` also occurs by chance
inside compressed data. The .pmat format stores no index, so this file
finds the boundaries in the stream itself and proves each one. The
compressed bytes are split among the threads; each looks for the first
`00 00 FF FF` in its share and inflates speculatively from just after
it, with no history, stopping at the first block boundary past the next
share's starting point. A start is accepted only when the inflate before
it, itself accepted, ends *exactly* there, at a block boundary: the
serial parse then provably reaches the same point, and since the
speculative inflate refused any back-reference before its own start, its
output is what a serial inflate would have produced. A chance match, or a
stream with no full flushes at all, fails that test, and the gap is
inflated serially from the last proven point, with the history in place.
The Adler-32 (or CRC) is checked over the whole result either way. So any
zlib stream inflates correctly, and a zlibmt one inflates on every core.

**The system zlib.** When `libz.so.1` (`libz.dylib`, `zlib1.dll`) can be
opened at run time it is used for zlibmt's per-block deflate and for
inflate. Nothing links against it; MIMA_NO_LIBZ=1 in the environment, or
Engine::Builtin, forces the built-in code. The plain zlib_deflate() is
always the built-in engine, so its output does not depend on what is
installed.
*******************************************************************************/

#ifndef _MMM_ZLIB_H
#define _MMM_ZLIB_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace mimamo {

// ===========================================================================
// CRC-32 and Adler-32
// ===========================================================================

namespace zlib {
namespace detail {

inline std::uint32_t load32le(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t load64le(const unsigned char* p) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
#else
    return static_cast<std::uint64_t>(load32le(p)) |
           (static_cast<std::uint64_t>(load32le(p + 4)) << 32);
#endif
}

/** \brief Eight CRC tables, for the slicing-by-eight update */

struct CrcTables {
    std::uint32_t t[8][256];

    CrcTables() {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;

            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }

            t[0][i] = c;
        }

        for (int k = 1; k < 8; ++k) {
            for (int i = 0; i < 256; ++i) {
                t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xFF];
            }
        }
    }
};

inline const CrcTables& crc_tables() {
    static const CrcTables tables;
    return tables;
}

/** \brief The CRC register over p, not inverted at either end */

inline std::uint32_t crc_update(std::uint32_t crc, const unsigned char* p, std::size_t n) {
    const auto& t = crc_tables().t;

    while (n >= 8) {
        const std::uint32_t lo = crc ^ load32le(p);
        const std::uint32_t hi = load32le(p + 4);
        crc = t[7][lo & 0xFF] ^ t[6][(lo >> 8) & 0xFF] ^ t[5][(lo >> 16) & 0xFF] ^
              t[4][lo >> 24] ^ t[3][hi & 0xFF] ^ t[2][(hi >> 8) & 0xFF] ^
              t[1][(hi >> 16) & 0xFF] ^ t[0][hi >> 24];
        p += 8;
        n -= 8;
    }

    while (n--) {
        crc = t[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    }

    return crc;
}

/** \brief a times b, modulo the CRC polynomial, both bit-reflected */

inline std::uint32_t crc_mulmod(std::uint32_t a, std::uint32_t b) {
    std::uint32_t m = 1u << 31, p = 0;

    for (;;) {
        if (a & m) {
            p ^= b;

            if ((a & (m - 1)) == 0) {
                break;
            }
        }

        m >>= 1;
        b = (b & 1) ? ((b >> 1) ^ 0xEDB88320u) : (b >> 1);
    }

    return p;
}

} // namespace detail

/** \brief CRC-32 as zlib's crc32(): start from 0, the result is finished */

inline std::uint32_t crc32(std::uint32_t crc, const void* p, std::size_t n) {
    return ~detail::crc_update(~crc, static_cast<const unsigned char*>(p), n);
}

/**
 * \brief The CRC of A followed by B, from the CRCs of each and B's length
 *
 * Appending len2 bytes multiplies A's contribution by x^(8 len2); the
 * power is built by repeated squaring, so this is a few hundred steps
 * whatever the length.
 */

inline std::uint32_t crc32_combine(std::uint32_t crc1, std::uint32_t crc2, std::uint64_t len2) {
    if (len2 == 0) {
        return crc1;
    }

    std::uint32_t x2k = 1u << 30;                 // x^1, reflected
    std::uint32_t power = 1u << 31;               // x^0
    std::uint64_t bits = len2 * 8;

    while (bits) {
        if (bits & 1) {
            power = detail::crc_mulmod(x2k, power);
        }

        x2k = detail::crc_mulmod(x2k, x2k);
        bits >>= 1;
    }

    return detail::crc_mulmod(power, crc1) ^ crc2;
}

/** \brief Adler-32 as zlib's adler32(): start from 1 */

inline std::uint32_t adler32(std::uint32_t adler, const void* data, std::size_t n) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::uint32_t a = adler & 0xFFFF, b = adler >> 16;

    while (n) {
        // 5552 is the most bytes before b can pass 2^32
        std::size_t k = std::min<std::size_t>(n, 5552);
        n -= k;

        while (k >= 8) {
            a += p[0];
            b += a;
            a += p[1];
            b += a;
            a += p[2];
            b += a;
            a += p[3];
            b += a;
            a += p[4];
            b += a;
            a += p[5];
            b += a;
            a += p[6];
            b += a;
            a += p[7];
            b += a;
            p += 8;
            k -= 8;
        }

        while (k--) {
            a += *p++;
            b += a;
        }

        a %= 65521;
        b %= 65521;
    }

    return (b << 16) | a;
}

/**
 * \brief The Adler-32 of A followed by B
 *
 * Each byte of B adds its running sum to b, and that running sum is A's
 * a, less one, plus B's own; so b gains len2 (a1 - 1) on top of b2.
 */

inline std::uint32_t adler32_combine(std::uint32_t adler1, std::uint32_t adler2,
                                     std::uint64_t len2) {
    const std::uint64_t M = 65521;
    const std::uint64_t a1 = adler1 & 0xFFFF, b1 = adler1 >> 16;
    const std::uint64_t a2 = adler2 & 0xFFFF, b2 = adler2 >> 16;
    const std::uint64_t a = (a1 + a2 + M - 1) % M;
    const std::uint64_t b = (b1 + b2 + (len2 % M) * ((a1 + M - 1) % M)) % M;
    return static_cast<std::uint32_t>((b << 16) | a);
}

} // namespace zlib

/** \brief The CRC register over p, for callers that finish it themselves */

inline std::uint32_t crc32_of(const unsigned char* p, std::size_t n,
                              std::uint32_t crc = 0xFFFFFFFFu) {
    return zlib::detail::crc_update(crc, p, n);
}

/** \brief The finished CRC of a whole buffer, ready to store */

inline std::uint32_t crc32_str(const std::string& s) {
    return zlib::crc32(0, s.data(), s.size());
}

inline void put32(std::string& s, std::uint32_t v) {
    s += static_cast<char>((v >> 24) & 0xFF);
    s += static_cast<char>((v >> 16) & 0xFF);
    s += static_cast<char>((v >> 8) & 0xFF);
    s += static_cast<char>(v & 0xFF);
}

namespace zlib {

// ===========================================================================
// A byte buffer, engines, options
// ===========================================================================

/**
 * \class Buffer mmm_zlib.hpp
 * \brief A growable byte buffer that never zero-fills
 *
 * Growing a std::string clears the new room, and a gigabyte about to be
 * written over would be written twice.
 */

class Buffer {
  public:
    Buffer() = default;

    ~Buffer() {
        std::free(m_p);
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& o) noexcept : m_p(o.m_p), m_n(o.m_n), m_cap(o.m_cap) {
        o.m_p = nullptr;
        o.m_n = o.m_cap = 0;
    }

    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) {
            std::free(m_p);
            m_p = o.m_p;
            m_n = o.m_n;
            m_cap = o.m_cap;
            o.m_p = nullptr;
            o.m_n = o.m_cap = 0;
        }

        return *this;
    }

    void reserve(std::size_t total) {
        if (total <= m_cap) {
            return;
        }

        void* q = std::realloc(m_p, total);

        if (!q) {
            throw std::bad_alloc();
        }

        m_p = static_cast<std::uint8_t*>(q);
        m_cap = total;
    }

    /** @brief k more bytes at the end, uninitialised; where they start */
    std::uint8_t* grow(std::size_t k) {
        if (m_n + k > m_cap) {
            reserve(std::max(m_n + k, std::max<std::size_t>(m_cap * 2, 4096)));
        }

        std::uint8_t* at = m_p + m_n;
        m_n += k;
        return at;
    }

    void append(const void* p, std::size_t k) {
        if (k) {
            std::memcpy(grow(k), p, k);
        }
    }

    /** @brief Set the size; new bytes are uninitialised */
    void resize(std::size_t n) {
        reserve(n);
        m_n = n;
    }

    void clear() {
        m_n = 0;
    }

    std::uint8_t* data() {
        return m_p;
    }

    const std::uint8_t* data() const {
        return m_p;
    }

    std::size_t size() const {
        return m_n;
    }

    std::size_t capacity() const {
        return m_cap;
    }

    std::string str() const {
        return m_n ? std::string(reinterpret_cast<const char*>(m_p), m_n) : std::string();
    }

  private:
    std::uint8_t* m_p = nullptr;
    std::size_t m_n = 0;
    std::size_t m_cap = 0;
};

/** \brief Which deflate and inflate do the work */

enum class Engine {
    Auto,      ///< the system zlib when it loads and MIMA_NO_LIBZ is unset, else Builtin
    Builtin,   ///< the code in this file
    System     ///< the system zlib, falling back to Builtin when it is absent
};

/** \brief The framing around a deflate stream */

enum class Format {
    Zlib,      ///< RFC 1950: two header bytes, deflate, Adler-32
    Gzip,      ///< RFC 1952: a header, deflate, CRC-32 and length
    Raw        ///< RFC 1951 alone
};

/** \brief How zlibmt compresses */

struct Options {
    int level = 6;                       ///< 0 (stored) to 9
    std::size_t block_bytes = 4u << 20;  ///< bytes of input per independent block
    int threads = 0;                     ///< 0 for every hardware thread
    Engine engine = Engine::Auto;
};

/**
 * \brief Bytes [offset, offset+len) of a virtual input, written to dst
 *
 * For an input that is not contiguous in memory -- the real and imaginary
 * halves of an interleaved complex array, say -- so it need not be copied
 * whole before it is compressed.
 */

using Gather = std::function<void(std::size_t offset, std::size_t len, std::uint8_t* dst)>;

inline int hardware_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n ? static_cast<int>(n) : 1;
}

namespace detail {

/**
 * Run f(i) for i in [0, count) on up to `threads` threads, which pull the
 * next index from a shared counter. f returns false to report a failure;
 * the others still finish.
 */

template <class F>
bool parallel_for(std::size_t count, int threads, F&& f) {
    if (threads <= 0) {
        threads = hardware_threads();
    }

    const std::size_t nt = std::min<std::size_t>(static_cast<std::size_t>(threads), count);

    if (nt <= 1) {
        bool ok = true;

        for (std::size_t i = 0; i < count; ++i) {
            ok = f(i) && ok;
        }

        return ok;
    }

    std::atomic<std::size_t> next(0);
    std::atomic<bool> ok(true);
    auto work = [&]() {
        for (;;) {
            const std::size_t i = next.fetch_add(1);

            if (i >= count) {
                return;
            }

            try {
                if (!f(i)) {
                    ok = false;
                }
            } catch (...) {
                ok = false;
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nt - 1);

    for (std::size_t t = 1; t < nt; ++t) {
        pool.emplace_back(work);
    }

    work();

    for (auto& th : pool) {
        th.join();
    }

    return ok;
}

// ===========================================================================
// The system zlib, opened at run time
// ===========================================================================

/** \brief zlib's z_stream, whose layout is part of its ABI */

struct ZStream {
    const unsigned char* next_in;
    unsigned avail_in;
    unsigned long total_in;
    unsigned char* next_out;
    unsigned avail_out;
    unsigned long total_out;
    const char* msg;
    void* state;
    void* zalloc;
    void* zfree;
    void* opaque;
    int data_type;
    unsigned long adler;
    unsigned long reserved;
};

constexpr int Z_NO_FLUSH = 0, Z_FULL_FLUSH = 3, Z_BLOCK = 5;
constexpr int Z_OK = 0, Z_STREAM_END = 1, Z_BUF_ERROR = -5;

#define MMM_LIBZ_FUNCS(X)                                                        \
    X(const char*,   zlibVersion,          (void))                               \
    X(int,           deflateInit2_,        (ZStream*, int, int, int, int, int,   \
                                            const char*, int))                   \
    X(int,           deflate,              (ZStream*, int))                      \
    X(int,           deflateEnd,           (ZStream*))                           \
    X(int,           deflateReset,         (ZStream*))                           \
    X(unsigned long, deflateBound,         (ZStream*, unsigned long))            \
    X(int,           inflateInit2_,        (ZStream*, int, const char*, int))    \
    X(int,           inflate,              (ZStream*, int))                      \
    X(int,           inflateEnd,           (ZStream*))                           \
    X(int,           inflateSetDictionary, (ZStream*, const unsigned char*, unsigned)) \
    X(int,           inflatePrime,         (ZStream*, int, int))

/** \struct LibzApi mmm_zlib.hpp \brief Function pointers resolved from libz */

struct LibzApi {
#define MMM_LIBZ_DECL(ret, name, args) ret (*name) args = nullptr;
    MMM_LIBZ_FUNCS(MMM_LIBZ_DECL)
#undef MMM_LIBZ_DECL

    void* lib = nullptr;
    bool ok = false;
    std::string version;
    std::string error;
};

inline LibzApi& libz() {
    static LibzApi a;
    return a;
}

inline void* open_lib(const char* path) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
#endif
}

inline void* sym(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

/** \brief Open the system zlib once and resolve what this file uses */

inline bool libz_load() {
    static std::once_flag once;
    std::call_once(once, []() {
        LibzApi& a = libz();
        static const char* candidates[] = {
#if defined(_WIN32)
            "zlib1.dll", "zlib.dll",
#elif defined(__APPLE__)
            "libz.1.dylib", "libz.dylib",
#else
            "libz.so.1", "libz.so",
#endif
            nullptr
        };

        for (int i = 0; candidates[i] && !a.lib; ++i) {
            a.lib = open_lib(candidates[i]);
        }

        if (!a.lib) {
            a.error = "no system zlib";
            return;
        }

        std::string missing;
#define MMM_LIBZ_LOAD(ret, name, args)                                          \
    a.name = reinterpret_cast<ret (*) args>(sym(a.lib, #name));             \
    if (!a.name) { missing += missing.empty() ? "" : ", "; missing += #name; }
        MMM_LIBZ_FUNCS(MMM_LIBZ_LOAD)
#undef MMM_LIBZ_LOAD

        if (!missing.empty()) {
            a.error = "the system zlib lacks " + missing;
            return;
        }

        const char* v = a.zlibVersion();

        // deflateInit2_ refuses another major version anyway
        if (!v || v[0] != '1') {
            a.error = std::string("system zlib version ") + (v ? v : "?") + " is not 1.x";
            return;
        }

        a.version = v;
        a.ok = true;
    });
    return libz().ok;
}

} // namespace detail

/** \brief True when MIMA_NO_LIBZ asks for the built-in code */

inline bool system_disabled() {
    const char* e = std::getenv("MIMA_NO_LIBZ");
    return e && *e && std::strcmp(e, "0") != 0;
}

/** \brief Whether the system zlib is loaded and allowed */

inline bool system_available() {
    return !system_disabled() && detail::libz_load();
}

/** \brief The system zlib's version, empty when there is none */

inline std::string system_version() {
    return detail::libz_load() ? detail::libz().version : std::string();
}

/** \brief The engine that will actually run for a request */

inline Engine resolve(Engine e) {
    if (e == Engine::Builtin) {
        return Engine::Builtin;
    }

    return system_available() ? Engine::System : Engine::Builtin;
}

// ===========================================================================
// The tables RFC 1951 defines
// ===========================================================================

namespace detail {

constexpr std::uint16_t kLenBase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43,
    51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
};
constexpr std::uint8_t kLenExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
};
constexpr std::uint16_t kDistBase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
};
constexpr std::uint8_t kDistExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
    12, 12, 13, 13
};
constexpr std::uint8_t kClOrder[19] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
};

inline std::uint32_t reverse_bits(std::uint32_t code, int len) {
    std::uint32_t r = 0;

    for (int i = 0; i < len; ++i) {
        r = (r << 1) | (code & 1);
        code >>= 1;
    }

    return r;
}

/** \brief Canonical codes for a set of lengths, bit-reversed for writing */

inline void huff_codes(const std::uint8_t* len, int n, std::uint16_t* code) {
    std::uint16_t count[16] = {0}, next[16] = {0};

    for (int i = 0; i < n; ++i) {
        count[len[i]]++;
    }

    count[0] = 0;
    std::uint32_t c = 0;

    for (int b = 1; b < 16; ++b) {
        c = (c + count[b - 1]) << 1;
        next[b] = static_cast<std::uint16_t>(c);
    }

    for (int i = 0; i < n; ++i) {
        code[i] = len[i] ? static_cast<std::uint16_t>(reverse_bits(next[len[i]]++, len[i])) : 0;
    }
}

/** \brief What the compressor looks up for every symbol */

struct DeflateTables {
    std::uint8_t len_sym[259] = {0};  ///< a match length, 3 to 258, to its code 0 to 28
    std::uint8_t dist_lo[256] = {0};  ///< distance - 1 below 256 to its code
    std::uint8_t dist_hi[256] = {0};  ///< (distance - 1) >> 7 above that
    std::uint8_t fixed_len[288];
    std::uint16_t fixed_code[288];
    std::uint8_t fixed_dlen[30];
    std::uint16_t fixed_dcode[30];

    DeflateTables() {
        for (int c = 0; c < 28; ++c) {
            for (int l = kLenBase[c]; l < kLenBase[c] + (1 << kLenExtra[c]); ++l) {
                len_sym[l] = static_cast<std::uint8_t>(c);
            }
        }

        len_sym[258] = 28;

        for (int c = 0; c < 30; ++c) {
            for (int d = kDistBase[c]; d < kDistBase[c] + (1 << kDistExtra[c]); ++d) {
                if (d <= 256) {
                    dist_lo[d - 1] = static_cast<std::uint8_t>(c);
                } else {
                    dist_hi[(d - 1) >> 7] = static_cast<std::uint8_t>(c);
                }
            }
        }

        for (int i = 0; i < 288; ++i) {
            fixed_len[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
        }

        huff_codes(fixed_len, 288, fixed_code);

        for (int i = 0; i < 30; ++i) {
            fixed_dlen[i] = 5;
        }

        huff_codes(fixed_dlen, 30, fixed_dcode);
    }

    int dist_sym(unsigned d) const {
        return d <= 256 ? dist_lo[d - 1] : dist_hi[(d - 1) >> 7];
    }
};

inline const DeflateTables& deflate_tables() {
    static const DeflateTables t;
    return t;
}

/**
 * \brief Code lengths of at most `limit` bits for these frequencies
 *
 * The lengths of an optimal code come from Moffat and Katajainen's
 * in-place method over the frequencies in ascending order. When the
 * longest is over the limit, the count of codes at each length is
 * repaired -- everything longer moves to the limit, then leaves are
 * pushed down from the deepest shorter length until the code is complete
 * again -- and the lengths are dealt back out, longest to the rarest.
 * Fewer than two used symbols get two codes of one bit, which every
 * inflate accepts.
 */

inline void huff_lengths(const std::uint32_t* freq, int n, int limit, std::uint8_t* len) {
    int sym[288];
    std::uint32_t a[288];
    int m = 0;
    std::memset(len, 0, static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        if (freq[i]) {
            sym[m++] = i;
        }
    }

    if (m < 2) {
        const int s0 = m ? sym[0] : 0;
        len[s0] = 1;
        len[s0 == 0 ? 1 : 0] = 1;
        return;
    }

    std::sort(sym, sym + m, [freq](int x, int y) {
        return freq[x] != freq[y] ? freq[x] < freq[y] : x < y;
    });

    for (int i = 0; i < m; ++i) {
        a[i] = freq[sym[i]];
    }

    // pass one: pair the two smallest of leaves and internal nodes, leaving
    // in a[] the internal weights and then parent pointers
    a[0] += a[1];
    int root = 0, leaf = 2;

    for (int next = 1; next < m - 1; ++next) {
        if (leaf >= m || a[root] < a[leaf]) {
            a[next] = a[root];
            a[root++] = static_cast<std::uint32_t>(next);
        } else {
            a[next] = a[leaf++];
        }

        if (leaf >= m || (root < next && a[root] < a[leaf])) {
            a[next] += a[root];
            a[root++] = static_cast<std::uint32_t>(next);
        } else {
            a[next] += a[leaf++];
        }
    }

    // pass two: the depth of every internal node, from the root down
    a[m - 2] = 0;

    for (int next = m - 3; next >= 0; --next) {
        a[next] = a[a[next]] + 1;
    }

    // pass three: the depth of every leaf
    int avail = 1, used = 0, depth = 0;
    root = m - 2;
    int next = m - 1;

    while (avail > 0) {
        while (root >= 0 && static_cast<int>(a[root]) == depth) {
            ++used;
            --root;
        }

        while (avail > used) {
            a[next--] = static_cast<std::uint32_t>(depth);
            --avail;
        }

        avail = 2 * used;
        ++depth;
        used = 0;
    }

    int count[300] = {0};
    int maxlen = 0;

    for (int i = 0; i < m; ++i) {
        count[a[i]]++;
        maxlen = std::max(maxlen, static_cast<int>(a[i]));
    }

    if (maxlen > limit) {
        for (int l = limit + 1; l <= maxlen; ++l) {
            count[limit] += count[l];
            count[l] = 0;
        }

        std::uint64_t total = 0;

        for (int l = 1; l <= limit; ++l) {
            total += static_cast<std::uint64_t>(count[l]) << (limit - l);
        }

        while (total > (1ull << limit)) {
            count[limit]--;

            for (int l = limit - 1; l > 0; --l) {
                if (count[l]) {
                    count[l]--;
                    count[l + 1] += 2;
                    break;
                }
            }

            --total;
        }

        maxlen = limit;
    }

    int at = 0;

    for (int l = maxlen; l > 0; --l) {
        for (int k = 0; k < count[l]; ++k) {
            len[sym[at++]] = static_cast<std::uint8_t>(l);
        }
    }
}

// ===========================================================================
// The compressor
// ===========================================================================

/** \brief A bit writer, least significant bit first, as deflate wants */

struct BitWriter {
    Buffer& out;
    std::uint64_t acc = 0;
    int n = 0;

    explicit BitWriter(Buffer& o) : out(o) {}

    /** @brief k bits of v (at most 32, v no wider) */
    void put(std::uint32_t v, int k) {
        acc |= static_cast<std::uint64_t>(v) << n;
        n += k;

        if (n >= 32) {
            std::uint8_t* q = out.grow(4);
            q[0] = static_cast<std::uint8_t>(acc);
            q[1] = static_cast<std::uint8_t>(acc >> 8);
            q[2] = static_cast<std::uint8_t>(acc >> 16);
            q[3] = static_cast<std::uint8_t>(acc >> 24);
            acc >>= 32;
            n -= 32;
        }
    }

    /** @brief Pad to a byte with zeros and write out what is held */
    void align() {
        while (n > 0) {
            *out.grow(1) = static_cast<std::uint8_t>(acc);
            acc >>= 8;
            n -= 8;
        }

        acc = 0;
        n = 0;
    }
};

/** \brief Matcher settings per level: when to shorten, when to stop */

struct LevelConfig {
    std::uint16_t good;    ///< a match this long already: search a quarter as far
    std::uint16_t lazy;    ///< no lazy search beyond this (greedy: insert no further)
    std::uint16_t nice;    ///< stop searching at a match this long
    std::uint16_t chain;   ///< the most chain links to follow
    bool lazy_mode;
};

constexpr LevelConfig kLevels[10] = {
    {0, 0, 0, 0, false},
    {4, 4, 8, 4, false},
    {4, 5, 16, 8, false},
    {4, 6, 32, 32, false},
    {4, 4, 16, 16, true},
    {8, 16, 32, 32, true},
    {8, 16, 128, 128, true},
    {8, 32, 128, 256, true},
    {32, 128, 258, 1024, true},
    {32, 258, 258, 4096, true}
};

/**
 * \class Deflater mmm_zlib.hpp
 * \brief Raw deflate of a buffer in memory, reusable across calls
 *
 * The whole input is in memory, so the matcher reads it where it lies:
 * positions are offsets into it, and a hash chain is followed back no
 * further than 32 KiB. Matches and literals are collected, 32 Ki of them
 * at most, and then written as one block in its cheapest form.
 */

class Deflater {
  public:
    explicit Deflater(int level)
        : m_level(std::max(0, std::min(9, level))), m_cfg(kLevels[m_level]) {
        if (m_level) {
            m_head.resize(HSIZE);
            m_prev.resize(WSIZE);
            m_syms.resize(SYMMAX);
        }
    }

    /**
     * @brief Deflate p[0, n), n below 4 GiB, as complete blocks
     *
     * \param last   the final block of the stream carries BFINAL
     * \param flush  end with a full flush (an empty stored block), so what
     *               follows starts afresh on a byte boundary
     */

    void run(const std::uint8_t* p, std::size_t n, BitWriter& w, bool last, bool flush) {
        if (m_level == 0) {
            if (n || last) {
                stored(p, n, w, last);
            }
        } else {
            std::fill(m_head.begin(), m_head.end(), 0u);
            m_p = p;
            m_n = n;
            m_nsym = 0;
            m_start = m_emitted = 0;
            std::fill(m_lfreq, m_lfreq + 286, 0u);
            std::fill(m_dfreq, m_dfreq + 30, 0u);

            if (m_cfg.lazy_mode) {
                lazy(w);
            } else {
                greedy(w);
            }

            if (m_nsym || last) {
                block(w, last);
            }
        }

        if (flush && !last) {
            w.put(0, 3);
            w.align();
            static const std::uint8_t marker[4] = {0, 0, 0xFF, 0xFF};
            w.out.append(marker, 4);
        }
    }

  private:
    static constexpr int HBITS = 15;
    static constexpr std::size_t HSIZE = std::size_t(1) << HBITS;
    static constexpr std::size_t WSIZE = 32768;
    static constexpr std::size_t WMASK = WSIZE - 1;
    static constexpr std::size_t SYMMAX = 32768;
    static constexpr std::size_t TOO_FAR = 4096;

    int m_level;
    LevelConfig m_cfg;
    std::vector<std::uint32_t> m_head;    // position + 1 of the newest with this hash
    std::vector<std::uint32_t> m_prev;    // position + 1 of the one before, by position
    std::vector<std::uint32_t> m_syms;    // (length or literal) << 16 | distance
    std::size_t m_nsym = 0;
    std::uint32_t m_lfreq[286];
    std::uint32_t m_dfreq[30];
    const std::uint8_t* m_p = nullptr;
    std::size_t m_n = 0;
    std::size_t m_start = 0;               // the first byte of the open block
    std::size_t m_emitted = 0;             // the first byte not yet in a symbol

    static void stored(const std::uint8_t* p, std::size_t n, BitWriter& w, bool last) {
        std::size_t at = 0;

        do {
            const std::size_t k = std::min<std::size_t>(65535, n - at);
            const bool fin = last && at + k >= n;
            w.put(fin ? 1 : 0, 1);
            w.put(0, 2);
            w.align();
            std::uint8_t* q = w.out.grow(4);
            q[0] = static_cast<std::uint8_t>(k);
            q[1] = static_cast<std::uint8_t>(k >> 8);
            q[2] = static_cast<std::uint8_t>(~k);
            q[3] = static_cast<std::uint8_t>((~k) >> 8);
            w.out.append(p + at, k);
            at += k;
        } while (at < n);
    }

    std::uint32_t hash(std::size_t pos) const {
        const std::uint32_t v = static_cast<std::uint32_t>(m_p[pos]) |
                                (static_cast<std::uint32_t>(m_p[pos + 1]) << 8) |
                                (static_cast<std::uint32_t>(m_p[pos + 2]) << 16);
        return (v * 2654435761u) >> (32 - HBITS);
    }

    /** @brief Put pos on its chain; the chain's previous head, as position + 1 */
    std::uint32_t insert(std::size_t pos) {
        const std::uint32_t h = hash(pos);
        const std::uint32_t old = m_head[h];
        m_prev[pos & WMASK] = old;
        m_head[h] = static_cast<std::uint32_t>(pos + 1);
        return old;
    }

    static std::size_t common(const std::uint8_t* a, const std::uint8_t* b, std::size_t max) {
        std::size_t l = 0;
#if defined(__GNUC__) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

        while (l + 8 <= max) {
            std::uint64_t x, y;
            std::memcpy(&x, a + l, 8);
            std::memcpy(&y, b + l, 8);

            if (x != y) {
                return l + static_cast<std::size_t>(__builtin_ctzll(x ^ y) >> 3);
            }

            l += 8;
        }

#endif

        while (l < max && a[l] == b[l]) {
            ++l;
        }

        return l;
    }

    /**
     * @brief The longest match at pos longer than `better`, following the
     * chain from `cand`; 0 when there is none
     */

    std::size_t longest(std::size_t pos, std::uint32_t cand, std::size_t better,
                        std::size_t& dist) const {
        const std::size_t max = std::min<std::size_t>(258, m_n - pos);

        if (max < 3 || better >= max) {
            return 0;
        }

        const std::uint8_t* s = m_p + pos;
        const std::size_t nice = std::min<std::size_t>(m_cfg.nice, max);
        unsigned chain = m_cfg.chain;

        if (better >= m_cfg.good) {
            chain >>= 2;
        }

        std::size_t best = better, found = 0;

        while (cand) {
            const std::size_t c = cand - 1;

            if (pos - c > WSIZE) {
                break;
            }

            const std::uint8_t* t = m_p + c;

            if (t[best] == s[best] && t[0] == s[0] && t[1] == s[1]) {
                const std::size_t l = common(t, s, max);

                if (l > best) {
                    best = l;
                    found = pos - c;

                    if (l >= nice) {
                        break;
                    }
                }
            }

            if (--chain == 0) {
                break;
            }

            const std::uint32_t nx = m_prev[c & WMASK];

            if (nx == 0 || nx - 1 >= c) {
                break;
            }

            cand = nx;
        }

        if (!found) {
            return 0;
        }

        dist = found;
        return best;
    }

    void literal(std::size_t pos) {
        const std::uint8_t b = m_p[pos];
        m_syms[m_nsym++] = static_cast<std::uint32_t>(b) << 16;
        m_lfreq[b]++;
        m_emitted = pos + 1;
    }

    void match(std::size_t pos, std::size_t len, std::size_t dist) {
        const DeflateTables& t = deflate_tables();
        m_syms[m_nsym++] = (static_cast<std::uint32_t>(len) << 16) | static_cast<std::uint32_t>(dist);
        m_lfreq[257 + t.len_sym[len]]++;
        m_dfreq[t.dist_sym(static_cast<unsigned>(dist))]++;
        m_emitted = pos + len;
    }

    void greedy(BitWriter& w) {
        std::size_t pos = 0;

        while (pos < m_n) {
            const std::uint32_t cand = pos + 3 <= m_n ? insert(pos) : 0;
            std::size_t len = 0, dist = 0;

            if (cand) {
                len = longest(pos, cand, 2, dist);
            }

            if (len >= 3 && !(len == 3 && dist > TOO_FAR)) {
                match(pos, len, dist);

                if (len <= m_cfg.lazy) {
                    for (std::size_t k = pos + 1; k < pos + len && k + 3 <= m_n; ++k) {
                        insert(k);
                    }
                }

                pos += len;
            } else {
                literal(pos);
                ++pos;
            }

            if (m_nsym >= SYMMAX - 2) {
                block(w, false);
            }
        }
    }

    /**
     * The lazy matcher: a match found at pos-1 waits one byte, and is
     * written only when the match at pos is no longer.
     */

    void lazy(BitWriter& w) {
        std::size_t pos = 0, plen = 0, pdist = 0;
        bool pending = false;

        while (pos < m_n) {
            const std::uint32_t cand = pos + 3 <= m_n ? insert(pos) : 0;
            std::size_t len = 0, dist = 0;

            if (cand && plen < m_cfg.lazy) {
                len = longest(pos, cand, std::max<std::size_t>(plen, 2), dist);

                if (len == 3 && dist > TOO_FAR) {
                    len = 0;
                }
            }

            if (pending && plen >= 3 && len <= plen) {
                match(pos - 1, plen, pdist);
                const std::size_t end = pos - 1 + plen;

                for (std::size_t k = pos + 1; k < end && k + 3 <= m_n; ++k) {
                    insert(k);
                }

                pos = end;
                pending = false;
                plen = 0;
            } else {
                if (pending) {
                    literal(pos - 1);
                }

                pending = true;
                plen = len;
                pdist = dist;
                ++pos;
            }

            if (m_nsym >= SYMMAX - 2) {
                block(w, false);
            }
        }

        if (pending) {
            literal(m_n - 1);
        }
    }

    /** \brief Write the collected symbols as one block, in its cheapest form */

    void block(BitWriter& w, bool last) {
        const DeflateTables& t = deflate_tables();
        m_lfreq[256] = 1;
        std::uint8_t llen[286], dlen[30];
        huff_lengths(m_lfreq, 286, 15, llen);
        huff_lengths(m_dfreq, 30, 15, dlen);
        int hlit = 286, hdist = 30;

        while (hlit > 257 && llen[hlit - 1] == 0) {
            --hlit;
        }

        while (hdist > 1 && dlen[hdist - 1] == 0) {
            --hdist;
        }

        // the code lengths, run-length coded with 16, 17 and 18
        std::uint8_t all[316];
        std::memcpy(all, llen, static_cast<std::size_t>(hlit));
        std::memcpy(all + hlit, dlen, static_cast<std::size_t>(hdist));
        const int total = hlit + hdist;
        std::uint8_t rle[316], rlx[316];
        int nr = 0;
        std::uint32_t clfreq[19] = {0};
        auto emit = [&](int s, int x) {
            rle[nr] = static_cast<std::uint8_t>(s);
            rlx[nr++] = static_cast<std::uint8_t>(x);
            clfreq[s]++;
        };

        for (int i = 0; i < total;) {
            const std::uint8_t cur = all[i];
            int run = 1;

            while (i + run < total && all[i + run] == cur) {
                ++run;
            }

            i += run;

            if (cur == 0) {
                while (run >= 11) {
                    const int r = std::min(run, 138);
                    emit(18, r - 11);
                    run -= r;
                }

                if (run >= 3) {
                    emit(17, run - 3);
                    run = 0;
                }
            } else {
                emit(cur, 0);
                --run;

                while (run >= 3) {
                    const int r = std::min(run, 6);
                    emit(16, r - 3);
                    run -= r;
                }
            }

            while (run-- > 0) {
                emit(cur, 0);
            }
        }

        std::uint8_t cllen[19];
        huff_lengths(clfreq, 19, 7, cllen);
        int hclen = 19;

        while (hclen > 4 && cllen[kClOrder[hclen - 1]] == 0) {
            --hclen;
        }

        // the bits each form would take
        std::uint64_t extra = 0;

        for (int c = 0; c < 29; ++c) {
            extra += static_cast<std::uint64_t>(m_lfreq[257 + c]) * kLenExtra[c];
        }

        for (int c = 0; c < 30; ++c) {
            extra += static_cast<std::uint64_t>(m_dfreq[c]) * kDistExtra[c];
        }

        std::uint64_t dyn = 17 + 3 * static_cast<std::uint64_t>(hclen) + extra;
        std::uint64_t fix = 3 + extra;

        for (int s = 0; s < 19; ++s) {
            dyn += static_cast<std::uint64_t>(clfreq[s]) * cllen[s];
        }

        dyn += 2 * clfreq[16] + 3 * clfreq[17] + 7 * clfreq[18];

        for (int s = 0; s < 286; ++s) {
            dyn += static_cast<std::uint64_t>(m_lfreq[s]) * llen[s];
            fix += static_cast<std::uint64_t>(m_lfreq[s]) * t.fixed_len[s];
        }

        for (int s = 0; s < 30; ++s) {
            dyn += static_cast<std::uint64_t>(m_dfreq[s]) * dlen[s];
            fix += static_cast<std::uint64_t>(m_dfreq[s]) * 5;
        }

        const std::size_t raw = m_emitted - m_start;
        const std::uint64_t chunks = std::max<std::uint64_t>(1, (raw + 65534) / 65535);
        const std::uint64_t sto = chunks * 35 + 7 + 8 * static_cast<std::uint64_t>(raw);

        w.out.reserve(w.out.size() + static_cast<std::size_t>(std::min(dyn, fix) / 8) + 64);

        if (raw > 0 && sto < dyn && sto < fix) {
            stored(m_p + m_start, raw, w, last);
        } else if (fix <= dyn) {
            w.put(last ? 1 : 0, 1);
            w.put(1, 2);
            symbols(w, t.fixed_code, t.fixed_len, t.fixed_dcode, t.fixed_dlen);
        } else {
            std::uint16_t lcode[286], dcode[30], clcode[19];
            huff_codes(llen, 286, lcode);
            huff_codes(dlen, 30, dcode);
            huff_codes(cllen, 19, clcode);
            w.put(last ? 1 : 0, 1);
            w.put(2, 2);
            w.put(static_cast<std::uint32_t>(hlit - 257), 5);
            w.put(static_cast<std::uint32_t>(hdist - 1), 5);
            w.put(static_cast<std::uint32_t>(hclen - 4), 4);

            for (int i = 0; i < hclen; ++i) {
                w.put(cllen[kClOrder[i]], 3);
            }

            static const int xbits[3] = {2, 3, 7};

            for (int i = 0; i < nr; ++i) {
                w.put(clcode[rle[i]], cllen[rle[i]]);

                if (rle[i] >= 16) {
                    w.put(rlx[i], xbits[rle[i] - 16]);
                }
            }

            symbols(w, lcode, llen, dcode, dlen);
        }

        m_nsym = 0;
        m_start = m_emitted;
        std::fill(m_lfreq, m_lfreq + 286, 0u);
        std::fill(m_dfreq, m_dfreq + 30, 0u);
    }

    void symbols(BitWriter& w, const std::uint16_t* lcode, const std::uint8_t* llen,
                 const std::uint16_t* dcode, const std::uint8_t* dlen) const {
        const DeflateTables& t = deflate_tables();

        for (std::size_t i = 0; i < m_nsym; ++i) {
            const std::uint32_t s = m_syms[i];
            const std::uint32_t v = s >> 16, d = s & 0xFFFF;

            if (!d) {
                w.put(lcode[v], llen[v]);
                continue;
            }

            const int c = t.len_sym[v];
            std::uint32_t bits = lcode[257 + c];
            int nb = llen[257 + c];

            if (kLenExtra[c]) {
                bits |= (v - kLenBase[c]) << nb;
                nb += kLenExtra[c];
            }

            w.put(bits, nb);
            const int dc = t.dist_sym(d);
            bits = dcode[dc] | ((d - kDistBase[dc]) << dlen[dc]);
            w.put(bits, dlen[dc] + kDistExtra[dc]);
        }

        w.put(lcode[256], llen[256]);
    }
};

/** \brief The system zlib's raw deflate, for one zlibmt block at a time */

class LibzDeflater {
  public:
    explicit LibzDeflater(int level) {
        std::memset(&m_z, 0, sizeof(m_z));
        const LibzApi& a = libz();
        m_ok = a.deflateInit2_(&m_z, level, 8, -15, 8, 0, a.version.c_str(),
                               static_cast<int>(sizeof(ZStream))) == Z_OK;
    }

    ~LibzDeflater() {
        if (m_ok) {
            libz().deflateEnd(&m_z);
        }
    }

    LibzDeflater(const LibzDeflater&) = delete;
    LibzDeflater& operator=(const LibzDeflater&) = delete;

    /**
     * @brief p[0, n), then a full flush, appended to out
     *
     * The input goes in with Z_NO_FLUSH and the flush follows on its own,
     * as pyjdata's zlibmt does it, so that the two give the same bytes.
     */

    bool block(const std::uint8_t* p, std::size_t n, Buffer& out) {
        const LibzApi& a = libz();

        if (!m_ok || a.deflateReset(&m_z) != Z_OK) {
            return false;
        }

        out.reserve(out.size() + a.deflateBound(&m_z, static_cast<unsigned long>(n)) + 64);
        std::size_t at = 0;

        while (at < n) {
            const std::size_t k = std::min<std::size_t>(n - at, 1u << 30);
            m_z.next_in = p + at;
            m_z.avail_in = static_cast<unsigned>(k);

            while (m_z.avail_in) {
                if (!step(out, Z_NO_FLUSH)) {
                    return false;
                }
            }

            at += k;
        }

        do {
            if (!step(out, Z_FULL_FLUSH)) {
                return false;
            }
        } while (m_z.avail_out == 0);

        return true;
    }

  private:
    ZStream m_z;
    bool m_ok = false;

    bool step(Buffer& out, int flush) {
        if (out.capacity() - out.size() < 4096) {
            out.reserve(out.capacity() * 2 + 65536);
        }

        const std::size_t room = std::min<std::size_t>(out.capacity() - out.size(), 1u << 30);
        m_z.next_out = out.data() + out.size();
        m_z.avail_out = static_cast<unsigned>(room);
        const int r = libz().deflate(&m_z, flush);
        out.resize(out.size() + (room - m_z.avail_out));
        return r == Z_OK || r == Z_BUF_ERROR;
    }
};

// ===========================================================================
// The decompressor
// ===========================================================================

/**
 * Decoding table entries: bits 0-3 the code bits to drop, 4-7 the kind,
 * 8-15 the extra bits that follow (or a second-level table's index bits),
 * 16-31 the literal, the base length or distance, or the table's offset.
 */

enum : std::uint32_t { OP_LIT = 0, OP_LEN = 1, OP_EOB = 2, OP_SUB = 3, OP_BAD = 4 };

constexpr std::uint32_t entry(std::uint32_t op, std::uint32_t bits, std::uint32_t extra,
                              std::uint32_t value) {
    return (value << 16) | (extra << 8) | (op << 4) | bits;
}

struct HuffTable {
    std::vector<std::uint32_t> t;
    int root = 0;
};

/** \brief False for an over-subscribed set of code lengths */

inline bool lengths_ok(const std::uint8_t* len, int n) {
    int count[16] = {0};

    for (int i = 0; i < n; ++i) {
        count[len[i]]++;
    }

    int left = 1;

    for (int b = 1; b < 16; ++b) {
        left = (left << 1) - count[b];

        if (left < 0) {
            return false;
        }
    }

    return true;
}

/**
 * \brief Build a decoding table from code lengths
 *
 * \param dist  the distance alphabet rather than literal/length
 * \return false for an over-subscribed set of lengths. An incomplete one
 *         is accepted; its unused codes decode as errors.
 */

inline bool huff_table(const std::uint8_t* len, int n, int root, bool dist, HuffTable& h) {
    if (!lengths_ok(len, n)) {
        return false;
    }

    int count[16] = {0};

    for (int i = 0; i < n; ++i) {
        count[len[i]]++;
    }

    count[0] = 0;
    std::uint32_t next[16] = {0}, c = 0;

    for (int b = 1; b < 16; ++b) {
        c = (c + static_cast<std::uint32_t>(count[b - 1])) << 1;
        next[b] = c;
    }

    const std::uint32_t rsize = 1u << root, rmask = rsize - 1;
    h.root = root;
    h.t.assign(rsize, entry(OP_BAD, 0, 0, 0));
    std::uint8_t subbits[1 << 10] = {0};
    std::uint32_t rev[320];

    for (int s = 0; s < n; ++s) {
        if (len[s]) {
            rev[s] = reverse_bits(next[len[s]]++, len[s]);

            if (len[s] > root) {
                const std::uint32_t pre = rev[s] & rmask;
                subbits[pre] = std::max<std::uint8_t>(subbits[pre],
                                                      static_cast<std::uint8_t>(len[s] - root));
            }
        }
    }

    for (std::uint32_t pre = 0; pre < rsize; ++pre) {
        if (subbits[pre]) {
            h.t[pre] = entry(OP_SUB, static_cast<std::uint32_t>(root), subbits[pre],
                             static_cast<std::uint32_t>(h.t.size()));
            h.t.resize(h.t.size() + (std::size_t(1) << subbits[pre]), entry(OP_BAD, 0, 0, 0));
        }
    }

    for (int s = 0; s < n; ++s) {
        const int l = len[s];

        if (!l) {
            continue;
        }

        std::uint32_t op = OP_BAD, extra = 0, value = 0;

        if (dist) {
            if (s < 30) {
                op = OP_LEN;
                extra = kDistExtra[s];
                value = kDistBase[s];
            }
        } else if (s < 256) {
            op = OP_LIT;
            value = static_cast<std::uint32_t>(s);
        } else if (s == 256) {
            op = OP_EOB;
        } else if (s < 286) {
            op = OP_LEN;
            extra = kLenExtra[s - 257];
            value = kLenBase[s - 257];
        }

        if (l <= root) {
            const std::uint32_t e = entry(op, static_cast<std::uint32_t>(l), extra, value);

            for (std::uint32_t k = rev[s]; k < rsize; k += 1u << l) {
                h.t[k] = e;
            }
        } else {
            const std::uint32_t pre = rev[s] & rmask;
            const std::uint32_t off = h.t[pre] >> 16;
            const std::uint32_t sb = subbits[pre];
            const std::uint32_t e = entry(op, static_cast<std::uint32_t>(l - root), extra, value);

            for (std::uint32_t k = rev[s] >> root; k < (1u << sb); k += 1u << (l - root)) {
                h.t[off + k] = e;
            }
        }
    }

    return true;
}

struct FixedTables {
    HuffTable lit, dist;

    FixedTables() {
        std::uint8_t l[288], d[30];

        for (int i = 0; i < 288; ++i) {
            l[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
        }

        for (int i = 0; i < 30; ++i) {
            d[i] = 5;
        }

        huff_table(l, 288, 10, false, lit);
        huff_table(d, 30, 8, true, dist);
    }
};

inline const FixedTables& fixed_tables() {
    static const FixedTables t;
    return t;
}

/**
 * \brief Where inflate writes: a fixed span, or a Buffer that grows
 *
 * Everything before `op` is the history back-references may reach.
 */

struct Out {
    std::uint8_t* base = nullptr;
    std::size_t op = 0;
    std::size_t cap = 0;
    Buffer* buf = nullptr;
    std::size_t stop = SIZE_MAX;   ///< stop at a block boundary once op reaches this (a growing Out)

    static Out fixed(void* p, std::size_t n) {
        Out o;
        o.base = static_cast<std::uint8_t*>(p);
        o.cap = n;
        return o;
    }

    static Out growing(Buffer& b) {
        Out o;
        o.buf = &b;
        o.base = b.data();
        o.op = b.size();
        o.cap = b.capacity();
        return o;
    }

    bool room(std::size_t k) {
        if (op + k <= cap) {
            return true;
        }

        // a prefix stops at the end of the block that reaches `stop`; a
        // block running on far past it is a stream cut short (an inflater
        // reads zeros past the end), or one no reader of prefixes expects
        if (!buf || (stop != SIZE_MAX && op + k > stop && op + k - stop > (std::size_t(16) << 20))) {
            return false;
        }

        buf->resize(op);
        buf->reserve(std::max(op + k, std::max<std::size_t>(cap * 2, 1 << 16)));
        base = buf->data();
        cap = buf->capacity();
        return true;
    }

    void commit() {
        if (buf) {
            buf->resize(op);
        }
    }
};

enum class Seg { Final, Stopped, Error };

struct SegResult {
    Seg st = Seg::Error;
    std::size_t end_bit = 0;     ///< where the stream stood when it stopped
};

/**
 * \class Inflater mmm_zlib.hpp
 * \brief The built-in raw inflate, from any bit to any block boundary
 */

class Inflater {
  public:
    Inflater(const std::uint8_t* p, std::size_t n, std::size_t start_bit, Out& out)
        : m_p(p), m_n(n), m_i(start_bit >> 3), m_out(out) {
        refill();
        drop(static_cast<int>(start_bit & 7));
    }

    /**
     * @brief Inflate blocks until the final one, or until a block boundary
     * at or past stop_bit
     */

    SegResult run(std::size_t stop_bit) {
        SegResult r;
        HuffTable lit, dist;

        for (;;) {
            if (pos() >= stop_bit || m_out.op >= m_out.stop) {
                r.st = Seg::Stopped;
                break;
            }

            need(3);
            const unsigned last = take(1);
            const unsigned type = take(2);
            bool ok = false;

            if (type == 0) {
                ok = stored();
            } else if (type == 1) {
                ok = codes(fixed_tables().lit, fixed_tables().dist);
            } else if (type == 2) {
                ok = header(lit, dist) && codes(lit, dist);
            }

            if (!ok || pos() > m_n * 8) {
                r.st = Seg::Error;
                break;
            }

            if (last) {
                r.st = Seg::Final;
                break;
            }
        }

        r.end_bit = pos();
        return r;
    }

  private:
    const std::uint8_t* m_p;
    std::size_t m_n;
    std::size_t m_i;          // the next byte to load; past m_n, zeros are loaded
    std::uint64_t m_buf = 0;
    int m_cnt = 0;
    Out& m_out;

    std::size_t pos() const {
        return m_i * 8 - static_cast<std::size_t>(m_cnt);
    }

    void refill() {
        if (m_i + 8 <= m_n) {
            m_buf |= load64le(m_p + m_i) << m_cnt;
            m_i += static_cast<std::size_t>((63 - m_cnt) >> 3);
            m_cnt |= 56;
        } else {
            while (m_cnt <= 56) {
                const std::uint64_t b = m_i < m_n ? m_p[m_i] : 0;
                m_buf |= b << m_cnt;
                ++m_i;
                m_cnt += 8;
            }
        }
    }

    void need(int k) {
        if (m_cnt < k) {
            refill();
        }
    }

    void drop(int k) {
        m_buf >>= k;
        m_cnt -= k;
    }

    unsigned take(int k) {
        const unsigned v = static_cast<unsigned>(m_buf & ((1ull << k) - 1));
        drop(k);
        return v;
    }

    /** @brief One symbol through a table; the entry, its code dropped */
    std::uint32_t decode(const HuffTable& h) {
        std::uint32_t e = h.t[m_buf & ((1u << h.root) - 1)];

        if (((e >> 4) & 15) == OP_SUB) {
            drop(static_cast<int>(e & 15));
            e = h.t[(e >> 16) + (m_buf & ((1u << ((e >> 8) & 0xFF)) - 1))];
        }

        drop(static_cast<int>(e & 15));
        return e;
    }

    bool stored() {
        drop(m_cnt & 7);
        need(32);
        const unsigned len = take(16);
        const unsigned nlen = take(16);

        if ((len ^ 0xFFFFu) != nlen) {
            return false;
        }

        // hand the whole bytes still held back to the input
        m_i -= static_cast<std::size_t>(m_cnt >> 3);
        m_buf = 0;
        m_cnt = 0;

        if (m_i + len > m_n || !m_out.room(len)) {
            return false;
        }

        if (len) {
            std::memcpy(m_out.base + m_out.op, m_p + m_i, len);
        }

        m_out.op += len;
        m_i += len;
        return true;
    }

    bool header(HuffTable& lit, HuffTable& dist) {
        need(14);
        const int hlit = static_cast<int>(take(5)) + 257;
        const int hdist = static_cast<int>(take(5)) + 1;
        const int hclen = static_cast<int>(take(4)) + 4;

        if (hlit > 286 || hdist > 30) {
            return false;
        }

        std::uint8_t cl[19] = {0};

        for (int i = 0; i < hclen; ++i) {
            need(3);
            cl[kClOrder[i]] = static_cast<std::uint8_t>(take(3));
        }

        if (!lengths_ok(cl, 19)) {
            return false;
        }

        // the code-length code is at most seven bits: one direct table
        std::uint8_t clsym[128], clbits[128] = {0};
        {
            std::uint32_t next[8] = {0}, c = 0;
            int count[8] = {0};

            for (int i = 0; i < 19; ++i) {
                count[cl[i]]++;
            }

            count[0] = 0;

            for (int b = 1; b < 8; ++b) {
                c = (c + static_cast<std::uint32_t>(count[b - 1])) << 1;
                next[b] = c;
            }

            for (int s = 0; s < 19; ++s) {
                if (cl[s]) {
                    const std::uint32_t r = reverse_bits(next[cl[s]]++, cl[s]);

                    for (std::uint32_t k = r; k < 128; k += 1u << cl[s]) {
                        clsym[k] = static_cast<std::uint8_t>(s);
                        clbits[k] = cl[s];
                    }
                }
            }
        }

        std::uint8_t lens[316];
        const int total = hlit + hdist;

        for (int i = 0; i < total;) {
            need(14);
            const unsigned k = static_cast<unsigned>(m_buf & 127);

            if (!clbits[k]) {
                return false;
            }

            drop(clbits[k]);
            const int s = clsym[k];

            if (s < 16) {
                lens[i++] = static_cast<std::uint8_t>(s);
                continue;
            }

            int rep = 0;
            std::uint8_t v = 0;

            if (s == 16) {
                if (i == 0) {
                    return false;
                }

                v = lens[i - 1];
                rep = 3 + static_cast<int>(take(2));
            } else if (s == 17) {
                rep = 3 + static_cast<int>(take(3));
            } else {
                rep = 11 + static_cast<int>(take(7));
            }

            if (i + rep > total) {
                return false;
            }

            while (rep--) {
                lens[i++] = v;
            }
        }

        if (lens[256] == 0) {
            return false;
        }

        return huff_table(lens, hlit, 10, false, lit) &&
               huff_table(lens + hlit, hdist, 8, true, dist);
    }

    bool codes(const HuffTable& lit, const HuffTable& dist) {
        Out& o = m_out;

        for (;;) {
            if (m_cnt < 48) {
                refill();
            }

            std::uint32_t e = decode(lit);
            const std::uint32_t op = (e >> 4) & 15;

            if (op == OP_LIT) {
                if (o.op >= o.cap && !o.room(1)) {
                    return false;
                }

                o.base[o.op++] = static_cast<std::uint8_t>(e >> 16);
                continue;
            }

            if (op != OP_LEN) {
                return op == OP_EOB;
            }

            const std::size_t len = (e >> 16) + take(static_cast<int>((e >> 8) & 0xFF));
            e = decode(dist);

            if (((e >> 4) & 15) != OP_LEN) {
                return false;
            }

            const std::size_t d = (e >> 16) + take(static_cast<int>((e >> 8) & 0xFF));

            if (d > o.op) {
                return false;                      // before the start of the history
            }

            if (o.op + len + 8 > o.cap && !o.room(len + 8)) {
                if (!o.room(len)) {
                    return false;
                }

                std::uint8_t* q = o.base + o.op;

                for (std::size_t k = 0; k < len; ++k) {
                    q[k] = q[k - d];
                }

                o.op += len;
                continue;
            }

            std::uint8_t* q = o.base + o.op;
            const std::uint8_t* s = q - d;

            if (d >= 8) {
                for (std::size_t k = 0; k < len; k += 8) {
                    std::memcpy(q + k, s + k, 8);
                }
            } else if (d == 1) {
                std::memset(q, s[0], len);
            } else {
                for (std::size_t k = 0; k < len; ++k) {
                    q[k] = s[k];
                }
            }

            o.op += len;
        }
    }
};

/**
 * \brief The system zlib's raw inflate, from any bit to any block boundary
 *
 * The history already in `out` is given as the dictionary, a start inside
 * a byte is primed with that byte's remaining bits, and Z_BLOCK makes
 * inflate return at each block boundary so the position can be checked.
 */

inline SegResult libz_segment(const std::uint8_t* p, std::size_t n, std::size_t start_bit,
                              std::size_t stop_bit, Out& out) {
    const LibzApi& a = libz();
    SegResult r;
    ZStream z;
    std::memset(&z, 0, sizeof(z));

    if (a.inflateInit2_(&z, -15, a.version.c_str(), static_cast<int>(sizeof(ZStream))) != Z_OK) {
        return r;
    }

    std::size_t at = start_bit >> 3;
    const int skip = static_cast<int>(start_bit & 7);
    bool ok = at <= n;

    if (ok && out.op) {
        const std::size_t k = std::min<std::size_t>(out.op, 32768);
        ok = a.inflateSetDictionary(&z, out.base + out.op - k, static_cast<unsigned>(k)) == Z_OK;
    }

    if (ok && skip) {
        ok = at < n && a.inflatePrime(&z, 8 - skip, p[at] >> skip) == Z_OK;
        ++at;
    }

    std::size_t fed = at;        // the first byte not yet handed to zlib
    std::uint8_t spare;

    while (ok) {
        if (z.avail_in == 0 && fed < n) {
            const std::size_t k = std::min<std::size_t>(n - fed, 1u << 30);
            z.next_in = p + fed;
            z.avail_in = static_cast<unsigned>(k);
            fed += k;
        }

        bool scratch = false;

        if (out.op >= out.cap && !out.room(65536)) {
            z.next_out = &spare;   // a fixed span is full: anything more is too much
            z.avail_out = 1;
            scratch = true;
        } else {
            const std::size_t room = std::min<std::size_t>(out.cap - out.op, 1u << 30);
            z.next_out = out.base + out.op;
            z.avail_out = static_cast<unsigned>(room);
        }

        const unsigned before_out = z.avail_out, before_in = z.avail_in;
        const int ret = a.inflate(&z, Z_BLOCK);
        const std::size_t made = before_out - z.avail_out;

        if (scratch) {
            if (made) {
                break;
            }
        } else {
            out.op += made;
        }

        const std::size_t bit = (fed - z.avail_in) * 8 - static_cast<std::size_t>(z.data_type & 63);

        if (ret == Z_STREAM_END) {
            r.st = Seg::Final;
            r.end_bit = bit;
            break;
        }

        if (ret != Z_OK && ret != Z_BUF_ERROR) {
            break;
        }

        if ((z.data_type & 128) && (bit >= stop_bit || out.op >= out.stop)) {
            r.st = Seg::Stopped;
            r.end_bit = bit;
            break;
        }

        if (made == 0 && before_in == z.avail_in && fed >= n && ret == Z_BUF_ERROR) {
            break;                 // out of input before the end
        }
    }

    a.inflateEnd(&z);
    return r;
}

inline SegResult segment(Engine e, const std::uint8_t* p, std::size_t n, std::size_t start_bit,
                         std::size_t stop_bit, Out& out) {
    if (e == Engine::System) {
        return libz_segment(p, n, start_bit, stop_bit, out);
    }

    Inflater f(p, n, start_bit, out);
    return f.run(stop_bit);
}

/** \brief Just past the first full-flush mark `00 00 FF FF` in [lo, hi), or 0 */

inline std::size_t find_flush(const std::uint8_t* p, std::size_t lo, std::size_t hi) {
    std::size_t i = std::max<std::size_t>(lo, 2);

    while (i + 1 < hi) {
        const void* q = std::memchr(p + i, 0xFF, hi - 1 - i);

        if (!q) {
            return 0;
        }

        const std::size_t j = static_cast<std::size_t>(static_cast<const std::uint8_t*>(q) - p);

        if (p[j + 1] == 0xFF && p[j - 1] == 0 && p[j - 2] == 0) {
            return j + 2;
        }

        i = j + 1;
    }

    return 0;
}

/** \brief Adler-32 or CRC-32 of a large buffer, in pieces on every thread */

inline std::uint32_t checksum(bool crc, const std::uint8_t* p, std::size_t n, int threads) {
    const std::size_t piece = 4u << 20;
    const std::size_t np = (n + piece - 1) / piece;

    if (np <= 1 || threads == 1) {
        return crc ? zlib::crc32(0, p, n) : adler32(1, p, n);
    }

    std::vector<std::uint32_t> part(np);
    parallel_for(np, threads, [&](std::size_t i) {
        const std::size_t len = std::min(piece, n - i * piece);
        part[i] = crc ? zlib::crc32(0, p + i * piece, len) : adler32(1, p + i * piece, len);
        return true;
    });
    std::uint32_t s = part[0];

    for (std::size_t i = 1; i < np; ++i) {
        const std::size_t len = std::min(piece, n - i * piece);
        s = crc ? crc32_combine(s, part[i], len) : adler32_combine(s, part[i], len);
    }

    return s;
}

/**
 * \brief Raw deflate data into `out`, in parallel where it can be proven
 *
 * The method is the file comment's: speculative inflates from each
 * share's first full-flush mark, each accepted only when the proven parse
 * before it ends exactly at its start, and serial inflate over any gap.
 * Accepted pieces are copied into place in parallel, just before the
 * next serial stretch needs them as history, and at the end.
 */

inline bool inflate_body(const std::uint8_t* p, std::size_t n, Out& out, int threads,
                         Engine e, std::size_t& end_bit) {
    if (threads <= 0) {
        threads = hardware_threads();
    }

    const std::size_t share = 1u << 20;
    std::vector<std::size_t> cand;

    if (threads > 1 && n >= 4 * share) {
        const std::size_t ns = std::min<std::size_t>(n / share, static_cast<std::size_t>(threads) * 8);

        for (std::size_t k = 1; k < ns; ++k) {
            const std::size_t at = find_flush(p, k * n / ns, (k + 1) * n / ns);

            if (at && at < n && (cand.empty() || at > cand.back())) {
                cand.push_back(at);
            }
        }
    }

    if (cand.empty()) {
        const SegResult r = segment(e, p, n, 0, SIZE_MAX, out);
        out.commit();
        end_bit = r.end_bit;
        return r.st == Seg::Final;
    }

    const std::size_t m = cand.size();
    std::vector<Buffer> piece(m);
    std::vector<SegResult> res(m + 1);
    parallel_for(m + 1, threads, [&](std::size_t j) {
        if (j == 0) {
            res[0] = segment(e, p, n, 0, cand[0] * 8, out);
            return true;
        }

        const std::size_t to = j < m ? cand[j] : n;
        piece[j - 1].reserve((to - cand[j - 1]) * 3 + 65536);
        Out o = Out::growing(piece[j - 1]);
        res[j] = segment(e, p, n, cand[j - 1] * 8, j < m ? cand[j] * 8 : SIZE_MAX, o);
        o.commit();
        return true;
    });

    struct Copy {
        const std::uint8_t* src;
        std::size_t len, at;
    };
    std::vector<Copy> pending;
    auto place = [&]() {
        parallel_for(pending.size(), threads, [&](std::size_t i) {
            if (pending[i].len) {
                std::memcpy(out.base + pending[i].at, pending[i].src, pending[i].len);
            }

            return true;
        });
        pending.clear();
    };
    SegResult cur = res[0];
    std::size_t j = 1;                // the next candidate is cand[j - 1], its result res[j]

    while (cur.st == Seg::Stopped) {
        const std::size_t bit = cur.end_bit;

        while (j <= m && cand[j - 1] * 8 < bit) {
            ++j;
        }

        if (j <= m && cand[j - 1] * 8 == bit && res[j].st != Seg::Error) {
            const Buffer& b = piece[j - 1];

            if (!out.room(b.size())) {
                return false;
            }

            pending.push_back({b.data(), b.size(), out.op});
            out.op += b.size();
            cur = res[j++];
            continue;
        }

        if (j <= m && cand[j - 1] * 8 == bit) {
            ++j;                      // a real boundary whose piece failed
        }

        place();
        cur = segment(e, p, n, bit, j <= m ? cand[j - 1] * 8 : SIZE_MAX, out);
    }

    place();
    out.commit();
    end_bit = cur.end_bit;
    return cur.st == Seg::Final;
}

/** \brief Where a gzip member's deflate data starts, and its stored name */

inline bool gzip_header(const std::uint8_t* g, std::size_t n, std::size_t& at, std::string* name) {
    if (n < 18 || g[0] != 0x1F || g[1] != 0x8B || g[2] != 8) {
        return false;
    }

    const std::uint8_t flg = g[3];
    at = 10;

    if (flg & 4) {                                   // FEXTRA
        if (at + 2 > n) {
            return false;
        }

        at += 2 + (static_cast<std::size_t>(g[at]) | (static_cast<std::size_t>(g[at + 1]) << 8));
    }

    for (int field = 8; field <= 16; field <<= 1) {  // FNAME, FCOMMENT
        if (flg & field) {
            std::string s;

            while (at < n && g[at] != 0) {
                s += static_cast<char>(g[at++]);
            }

            ++at;

            if (field == 8 && name) {
                *name = s;
            }
        }
    }

    if (flg & 2) {                                   // FHCRC
        at += 2;
    }

    return at < n;
}

inline std::uint32_t be32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) | p[3];
}

/**
 * \brief Inflate a framed stream into `out`
 *
 * \param lenient  accept a zlib stream whose Adler-32 is missing (but not
 *                 one whose Adler-32 is wrong)
 * \param eaten    the input bytes consumed
 */

inline bool inflate_framed(const std::uint8_t* p, std::size_t n, Out& out, Format f, int threads,
                           Engine e, bool lenient, std::size_t* eaten, std::string* name) {
    e = resolve(e);
    std::size_t end_bit = 0;

    if (f == Format::Raw) {
        const bool ok = inflate_body(p, n, out, threads, e, end_bit);

        if (eaten) {
            *eaten = (end_bit + 7) / 8;
        }

        return ok;
    }

    if (f == Format::Zlib) {
        if (n < 2 || (p[0] & 0x0F) != 8 || (p[0] >> 4) > 7 || ((p[0] << 8) | p[1]) % 31 != 0 ||
                (p[1] & 0x20)) {
            return false;                  // not deflate, a bad check, or a preset dictionary
        }

        const std::size_t start = out.op;

        if (!inflate_body(p + 2, n - 2, out, threads, e, end_bit)) {
            return false;
        }

        const std::size_t end = 2 + (end_bit + 7) / 8;

        if (eaten) {
            *eaten = std::min(n, end + 4);
        }

        if (end + 4 > n) {
            return lenient;
        }

        return be32(p + end) == checksum(false, out.base + start, out.op - start, threads);
    }

    std::size_t at = 0;
    bool first = true;

    while (first || (n - at >= 18 && p[at] == 0x1F && p[at + 1] == 0x8B)) {
        std::size_t body = 0;

        if (!gzip_header(p + at, n - at, body, first ? name : nullptr)) {
            return false;
        }

        const std::size_t start = out.op;

        if (!inflate_body(p + at + body, n - at - body, out, threads, e, end_bit)) {
            return false;
        }

        at += body + (end_bit + 7) / 8;

        if (at + 8 > n) {
            return false;
        }

        const std::uint32_t crc = load32le(p + at);
        const std::uint32_t len = load32le(p + at + 4);

        if (crc != checksum(true, out.base + start, out.op - start, threads) ||
                len != static_cast<std::uint32_t>(out.op - start)) {
            return false;
        }

        at += 8;
        first = false;
    }

    if (eaten) {
        *eaten = at;
    }

    return true;
}

/** \brief pyjdata's FLEVEL for a zlib header, so the headers agree */

inline void zlib_head(Buffer& out, int level) {
    static const int flevel[10] = {0, 0, 1, 1, 1, 1, 2, 3, 3, 3};
    const unsigned cmf = 0x78;
    unsigned flg = static_cast<unsigned>(flevel[level]) << 6;
    flg |= 31 - ((cmf << 8 | flg) % 31);
    std::uint8_t* q = out.grow(2);
    q[0] = static_cast<std::uint8_t>(cmf);
    q[1] = static_cast<std::uint8_t>(flg);
}

inline void put_be32(Buffer& out, std::uint32_t v) {
    std::uint8_t* q = out.grow(4);
    q[0] = static_cast<std::uint8_t>(v >> 24);
    q[1] = static_cast<std::uint8_t>(v >> 16);
    q[2] = static_cast<std::uint8_t>(v >> 8);
    q[3] = static_cast<std::uint8_t>(v);
}

inline void put_le32(Buffer& out, std::uint32_t v) {
    std::uint8_t* q = out.grow(4);
    q[0] = static_cast<std::uint8_t>(v);
    q[1] = static_cast<std::uint8_t>(v >> 8);
    q[2] = static_cast<std::uint8_t>(v >> 16);
    q[3] = static_cast<std::uint8_t>(v >> 24);
}

/**
 * \brief zlibmt: blocks deflated on every thread, joined as one stream
 *
 * Exactly one of `in` and `gather` gives the input. Each worker keeps its
 * own deflater, and a gather buffer one block long.
 */

inline bool compress_mt(const std::uint8_t* in, const Gather* gather, std::size_t n,
                        Buffer& out, const Options& opt, Format f) {
    const int level = std::max(0, std::min(9, opt.level));
    std::size_t bs = opt.block_bytes ? opt.block_bytes : (4u << 20);
    bs = std::min<std::size_t>(bs, 1u << 30);
    const std::size_t nb = n ? (n + bs - 1) / bs : 1;
    // stored blocks are copying, and how libz splits them depends on the
    // room it is given for output, so level 0 is always the built-in's
    const Engine e = level == 0 ? Engine::Builtin : resolve(opt.engine);
    const bool gz = f == Format::Gzip;
    std::vector<Buffer> seg(nb);
    std::vector<std::uint32_t> sum(nb);
    std::atomic<std::size_t> next(0);
    std::atomic<bool> ok(true);
    int threads = opt.threads > 0 ? opt.threads : hardware_threads();
    threads = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(threads), nb));

    auto worker = [&]() {
        try {
            std::unique_ptr<Deflater> d;
            std::unique_ptr<LibzDeflater> z;
            Buffer local;

            if (e == Engine::System) {
                z.reset(new LibzDeflater(level));
            } else {
                d.reset(new Deflater(level));
            }

            for (;;) {
                const std::size_t i = next.fetch_add(1);

                if (i >= nb) {
                    return;
                }

                const std::size_t off = i * bs, len = std::min(bs, n - off);
                const std::uint8_t* src = in ? in + off : nullptr;

                if (!in) {
                    local.resize(len);

                    if (len) {
                        (*gather)(off, len, local.data());
                    }

                    src = local.data();
                }

                sum[i] = gz ? zlib::crc32(0, src, len) : adler32(1, src, len);

                if (z) {
                    if (!z->block(src, len, seg[i])) {
                        ok = false;
                    }
                } else {
                    seg[i].reserve(len + len / 8 + 64);
                    BitWriter w(seg[i]);
                    d->run(src, len, w, false, true);
                }
            }
        } catch (...) {
            ok = false;
        }
    };
    std::vector<std::thread> pool;

    for (int t = 1; t < threads; ++t) {
        pool.emplace_back(worker);
    }

    worker();

    for (auto& th : pool) {
        th.join();
    }

    if (!ok) {
        return false;
    }

    std::size_t total = 32;

    for (const Buffer& b : seg) {
        total += b.size();
    }

    out.reserve(out.size() + total);

    if (f == Format::Zlib) {
        zlib_head(out, level);
    } else if (gz) {
        static const std::uint8_t head[10] = {0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF};
        out.append(head, 10);
    }

    for (const Buffer& b : seg) {
        out.append(b.data(), b.size());
    }

    static const std::uint8_t tail[2] = {0x03, 0x00};   // an empty, final, fixed block
    out.append(tail, 2);
    std::uint32_t s = sum[0];

    for (std::size_t i = 1; i < nb; ++i) {
        const std::size_t len = std::min(bs, n - i * bs);
        s = gz ? crc32_combine(s, sum[i], len) : adler32_combine(s, sum[i], len);
    }

    if (f == Format::Zlib) {
        put_be32(out, s);
    } else if (gz) {
        put_le32(out, s);
        put_le32(out, static_cast<std::uint32_t>(n));
    }

    return true;
}

} // namespace detail

// ===========================================================================
// The interface
// ===========================================================================

/**
 * @brief zlibmt: compress n bytes into one standard stream, appended to out
 *
 * The bytes are a function of the data, opt.level, opt.block_bytes and
 * the engine; opt.threads changes only the speed.
 */

inline bool compress(const void* in, std::size_t n, Buffer& out, const Options& opt = Options(),
                     Format f = Format::Zlib) {
    return detail::compress_mt(static_cast<const std::uint8_t*>(in), nullptr, n, out, opt, f);
}

/** @brief zlibmt over a virtual input of n bytes that `gather` supplies */

inline bool compress(std::size_t n, const Gather& gather, Buffer& out,
                     const Options& opt = Options(), Format f = Format::Zlib) {
    return detail::compress_mt(nullptr, &gather, n, out, opt, f);
}

/**
 * @brief One deflate stream, not in blocks, by the built-in engine
 *
 * \param last  give the final block BFINAL; with false the stream ends
 *              with a full flush instead, ready for more
 */

inline void deflate_raw(const void* in, std::size_t n, Buffer& out, int level = 6,
                        bool last = true) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(in);
    detail::Deflater d(level);
    detail::BitWriter w(out);
    std::size_t at = 0;

    do {
        const std::size_t k = std::min<std::size_t>(n - at, 1u << 30);
        const bool end = at + k >= n;
        d.run(p + at, k, w, last && end, !last && end);
        at += k;
    } while (at < n);

    w.align();
}

/**
 * @brief Inflate into exactly outn bytes
 *
 * False for a corrupt stream, a wrong checksum, or one that does not come
 * out at outn bytes. threads 0 is every hardware thread.
 */

inline bool inflate(const void* in, std::size_t n, void* out, std::size_t outn,
                    Format f = Format::Zlib, int threads = 0, Engine e = Engine::Auto) {
    detail::Out o = detail::Out::fixed(out, outn);
    return detail::inflate_framed(static_cast<const std::uint8_t*>(in), n, o, f, threads, e,
                                  false, nullptr, nullptr) && o.op == outn;
}

/** @brief Inflate, appending to a Buffer of whatever size comes out */

inline bool inflate(const void* in, std::size_t n, Buffer& out, Format f = Format::Zlib,
                    int threads = 0, Engine e = Engine::Auto) {
    detail::Out o = detail::Out::growing(out);
    const bool ok = detail::inflate_framed(static_cast<const std::uint8_t*>(in), n, o, f, threads,
                                           e, false, nullptr, nullptr);
    o.commit();
    return ok;
}

/**
 * @brief Inflate the first `want` bytes of a stream, and no more than needed
 *
 * The stream is inflated in order until `want` bytes have come out, and
 * stops at the end of the deflate block that got there, so `out` may gain
 * a little more than `want`. This is how a slice near the front of a large
 * compressed array is read without inflating the rest. The Adler-32 is
 * checked only when the stream is read to its end, which it isn't when it
 * stops early. A gzip stream is inflated whole.
 *
 * @return false for a corrupt stream, or one that ends before `want` bytes
 */

inline bool inflate_prefix(const void* in, std::size_t n, Buffer& out, std::size_t want,
                           Format f = Format::Zlib, Engine e = Engine::Auto) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(in);
    const std::size_t start = out.size();

    if (f == Format::Gzip) {
        return inflate(in, n, out, f, 0, e) && out.size() - start >= want;
    }

    std::size_t skip = 0;

    if (f == Format::Zlib) {
        if (n < 2 || (p[0] & 0x0F) != 8 || (p[0] >> 4) > 7 || ((p[0] << 8) | p[1]) % 31 != 0 ||
                (p[1] & 0x20)) {
            return false;
        }

        skip = 2;
    }

    detail::Out o = detail::Out::growing(out);
    o.stop = start + want;
    const detail::SegResult r = detail::segment(resolve(e), p + skip, n - skip, 0,
                                                SIZE_MAX, o);
    o.commit();

    if (r.st == detail::Seg::Error || out.size() - start < want) {
        return false;
    }

    if (r.st == detail::Seg::Final && f == Format::Zlib) {
        const std::size_t end = skip + (r.end_bit + 7) / 8;
        return end + 4 <= n &&
               detail::be32(p + end) == adler32(1, out.data() + start, out.size() - start);
    }

    return true;
}

} // namespace zlib

// ===========================================================================
// The string interface that mima's builtins, the PNG writer and the MAT
// reader use
// ===========================================================================

/** \brief A zlib stream of raw at `level`, by the built-in engine */

inline std::string zlib_deflate(const std::string& raw, int level = 6) {
    zlib::Buffer out;
    out.reserve(raw.size() / 2 + 64);
    zlib::detail::zlib_head(out, std::max(0, std::min(9, level)));
    zlib::deflate_raw(raw.data(), raw.size(), out, level);
    zlib::detail::put_be32(out, zlib::adler32(1, raw.data(), raw.size()));
    return out.str();
}

/** \brief zlib's stored-block form: no compressor, still a valid stream */

inline std::string zlib_stored(const std::string& raw) {
    return zlib_deflate(raw, 0);
}

/**
 * \brief Raw deflate, RFC 1951: no zlib or gzip framing
 *
 * \param in     the compressed bytes
 * \param out    appended to
 * \param eaten  how many input bytes were consumed, for a framed caller
 * \return false on a corrupt stream
 */

inline bool inflate_raw(const std::string& in, std::string& out, std::size_t* eaten = nullptr) {
    zlib::Buffer b;
    zlib::detail::Out o = zlib::detail::Out::growing(b);
    const bool ok = zlib::detail::inflate_framed(
                        reinterpret_cast<const std::uint8_t*>(in.data()), in.size(), o,
                        zlib::Format::Raw, 0, zlib::Engine::Auto, false, eaten, nullptr);
    o.commit();

    if (ok) {
        out.append(reinterpret_cast<const char*>(b.data()), b.size());
    }

    return ok;
}

/**
 * \brief A zlib stream (RFC 1950): two header bytes, deflate, adler32
 *
 * The Adler-32 is checked when it is there; a stream cut off just before
 * it is accepted, as it always was here.
 */

inline bool zlib_inflate(const std::string& z, std::string& out) {
    zlib::Buffer b;
    zlib::detail::Out o = zlib::detail::Out::growing(b);
    const bool ok = zlib::detail::inflate_framed(
                        reinterpret_cast<const std::uint8_t*>(z.data()), z.size(), o,
                        zlib::Format::Zlib, 0, zlib::Engine::Auto, true, nullptr, nullptr);
    o.commit();

    if (ok) {
        out.append(reinterpret_cast<const char*>(b.data()), b.size());
    }

    return ok;
}

// ===========================================================================
// gzip framing (RFC 1952), which is a header, raw deflate, crc and length
// ===========================================================================

/** \brief Wrap raw deflate output as a gzip member */

inline std::string gzip_pack(const std::string& raw, const std::string& deflated,
                             const std::string& name = std::string()) {
    std::string g;
    g += static_cast<char>(0x1F);
    g += static_cast<char>(0x8B);
    g += static_cast<char>(8);                       // deflate
    g += static_cast<char>(name.empty() ? 0 : 8);    // FNAME
    g += std::string(4, '\0');                       // mtime: zero, so the
    g += static_cast<char>(0);                       // output is reproducible
    g += static_cast<char>(3);                       // unix

    if (!name.empty()) {
        g += name;
        g += '\0';
    }

    g += deflated;
    const std::uint32_t crc = crc32_str(raw);
    const std::uint32_t n = static_cast<std::uint32_t>(raw.size());

    for (int i = 0; i < 4; ++i) {
        g += static_cast<char>((crc >> (8 * i)) & 0xFF);
    }

    for (int i = 0; i < 4; ++i) {
        g += static_cast<char>((n >> (8 * i)) & 0xFF);
    }

    return g;
}

/**
 * \brief Unwrap and inflate a gzip file, every member of it
 *
 * \param name  filled with the first member's stored name when there is
 *              one, which is what gunzip's output is called when the
 *              caller gave no other clue
 */

inline bool gzip_unpack(const std::string& g, std::string& out, std::string* name = nullptr) {
    zlib::Buffer b;
    zlib::detail::Out o = zlib::detail::Out::growing(b);
    const bool ok = zlib::detail::inflate_framed(
                        reinterpret_cast<const std::uint8_t*>(g.data()), g.size(), o,
                        zlib::Format::Gzip, 0, zlib::Engine::Auto, false, nullptr, name);
    o.commit();

    if (ok) {
        out.append(reinterpret_cast<const char*>(b.data()), b.size());
    }

    return ok;
}

} // namespace mimamo

#endif
