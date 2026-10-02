/*******************************************************************************/
/*! \file    zlibmt.h
    \brief   zlib / gzip / base64 for the NIfTI and JData I/O, on every core

    A thin adapter over mimamo's `mmm_zlib.hpp` and `mmm_base64.hpp`
    (https://github.com/fangq/mimamo, BSD-3-Clause; vendored unchanged),
    which replaced zmat / miniz here.

    Compression is **zlibmt**: the input is cut into 4 MiB blocks, each
    deflated on its own thread and ended with a full flush, and the blocks
    are joined behind one zlib (or gzip) header and one checksum -- a single
    standard stream any inflate reads (jsonlab, pyjdata, nibabel, gzip). Its
    bytes depend on the data and the level only, never on the thread count.
    Inflate is parallel too, finding and proving the block boundaries in the
    stream itself. The system zlib (`libz.so.1`, opened at run time, never
    linked) does the per-block work when present; mimamo's own deflate /
    inflate otherwise (MIMA_NO_LIBZ=1 forces it).

    Threads: ZLIBMT_THREADS in the environment, else every hardware thread.
    Level: ZLIBMT_LEVEL in the environment, else the caller's (default 6,
    zlib's own).

    The same file is shared by siamize and gpu_brain2mesh; C++11 or later.
*/
/*******************************************************************************/

#ifndef SIAM_ZLIBMT_H
#define SIAM_ZLIBMT_H

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "mmm_base64.hpp"
#include "mmm_zlib.hpp"

namespace zlibmt {

namespace detail {

inline int env_int(const char* name, int dflt) {
    const char* s = std::getenv(name);
    return (s && *s) ? std::atoi(s) : dflt;
}

inline mimamo::zlib::Options options(int level = 6) {
    mimamo::zlib::Options o;
    o.level = env_int("ZLIBMT_LEVEL", level);
    o.threads = env_int("ZLIBMT_THREADS", 0);   // 0: every hardware thread
    return o;
}

inline std::vector<uint8_t> deflate(const uint8_t* in, size_t n, mimamo::zlib::Format f, int level) {
    mimamo::zlib::Buffer z;

    if (!mimamo::zlib::compress(in, n, z, options(level), f)) {
        throw std::runtime_error("zlibmt: compression failed");
    }

    return std::vector<uint8_t>(z.data(), z.data() + z.size());
}

inline std::vector<uint8_t> inflate(const uint8_t* in, size_t n, size_t expected, mimamo::zlib::Format f,
                                    const char* what) {
    const int th = options().threads;

    if (expected > 0) {   // (the size is known: straight into the result)
        std::vector<uint8_t> out(expected);

        if (!mimamo::zlib::inflate(in, n, out.data(), expected, f, th)) {
            throw std::runtime_error(std::string(what) + " decode failed (corrupt, or not " + std::to_string(expected) +
                                     " bytes)");
        }

        return out;
    }

    mimamo::zlib::Buffer b;

    if (!mimamo::zlib::inflate(in, n, b, f, th)) {
        throw std::runtime_error(std::string(what) + " decode failed (corrupt stream)");
    }

    return std::vector<uint8_t>(b.data(), b.data() + b.size());
}

}  // namespace detail

/** A zlib (RFC 1950) stream of in[0, n), at `level` (0-9) unless ZLIBMT_LEVEL is set. */
inline std::vector<uint8_t> zlib_compress(const uint8_t* in, size_t n, int level = 6) {
    return detail::deflate(in, n, mimamo::zlib::Format::Zlib, level);
}

/** Inflate a zlib stream; `expected` > 0: it must come out at that many bytes. */
inline std::vector<uint8_t> zlib_decompress(const uint8_t* in, size_t n, size_t expected = 0) {
    return detail::inflate(in, n, expected, mimamo::zlib::Format::Zlib, "zlib");
}

/** A gzip (RFC 1952) stream of in[0, n), as a .gz file holds. */
inline std::vector<uint8_t> gzip_compress(const uint8_t* in, size_t n, int level = 6) {
    return detail::deflate(in, n, mimamo::zlib::Format::Gzip, level);
}

/** Inflate a gzip stream. */
inline std::vector<uint8_t> gzip_decompress(const uint8_t* in, size_t n, size_t expected = 0) {
    return detail::inflate(in, n, expected, mimamo::zlib::Format::Gzip, "gzip");
}

/** Base64 (RFC 4648, no line breaks) of in[0, n). */
inline std::string base64_encode(const uint8_t* in, size_t n) {
    return mimamo::base64_encode(std::string(reinterpret_cast<const char*>(in), n));
}

/** Decode base64 text (whitespace ignored). */
inline std::vector<uint8_t> base64_decode(const char* in, size_t n) {
    std::string out;

    if (!mimamo::base64_decode(std::string(in, n), out)) {
        throw std::runtime_error("base64 decode failed");
    }

    return std::vector<uint8_t>(out.begin(), out.end());
}

}  // namespace zlibmt

#endif  // SIAM_ZLIBMT_H
