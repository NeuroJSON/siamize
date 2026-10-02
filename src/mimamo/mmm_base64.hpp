/***************************************************************************//**
**  Mimamo - Mighty Matrix Module
**      -- a header-only C++17 matrix library with matlab/octave-like interfaces
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2026
**
**  License: BSD 3-Clause, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mmm_base64.hpp
@brief   RFC 4648 base64, which is how binary travels inside text

JSON, BJData's text sibling, XML and every HTTP header that carries bytes
spell them in base64, so a library that reads and writes those formats
needs it and should not need a toolbox for it. matlab's own answer is
`matlab.net.base64encode`, which is a thin wrapper over the JVM; there is
no jvm here, and this is forty lines.

The optional line wrapping is not decoration: MIME and several of the
older encoders break at 72 characters, and a decoder that cannot skip a
newline cannot read what they wrote. So the encoder can produce it and the
decoder ignores every character outside the alphabet, which is what makes
the pair interoperable with anything.

Transcribed from zmat's zmatlib.c, which is the same author's C and is
itself from Jouni Malinen's BSD-licensed original.
*******************************************************************************/

#ifndef _MMM_BASE64_H
#define _MMM_BASE64_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace mimamo {

namespace detail {

/** @brief The sixty-four digits, in RFC 4648's order */

inline const char* base64_alphabet() {
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}

} // namespace detail

/**
 * \brief Bytes as base64 text
 *
 * \param[in] raw: the bytes
 * \param[in] wrap: characters per line, or 0 for one unbroken line
 * \param[in] trailing_newline: whether to end a wrapped answer with one
 */

inline std::string base64_encode(const std::string& raw, std::size_t wrap = 0,
                                 bool trailing_newline = false) {
    const char* tbl = detail::base64_alphabet();
    const unsigned char* in =
        reinterpret_cast<const unsigned char*>(raw.data());
    const std::size_t n = raw.size();
    std::string out;
    out.reserve(n / 3 * 4 + 8 + (wrap ? n / (wrap ? wrap : 1) : 0));

    std::size_t line = 0;
    std::size_t i = 0;

    for (; i + 3 <= n; i += 3) {
        out += tbl[in[i] >> 2];
        out += tbl[((in[i] & 0x03) << 4) | (in[i + 1] >> 4)];
        out += tbl[((in[i + 1] & 0x0f) << 2) | (in[i + 2] >> 6)];
        out += tbl[in[i + 2] & 0x3f];
        line += 4;

        if (wrap && line >= wrap) {
            out += '\n';
            line = 0;
        }
    }

    if (i < n) {
        out += tbl[in[i] >> 2];

        if (n - i == 1) {
            out += tbl[(in[i] & 0x03) << 4];
            out += '=';
        } else {
            out += tbl[((in[i] & 0x03) << 4) | (in[i + 1] >> 4)];
            out += tbl[(in[i + 1] & 0x0f) << 2];
        }

        out += '=';
        line += 4;
    }

    if (trailing_newline && line) {
        out += '\n';
    }

    return out;
}

/**
 * \brief base64 text back to bytes, ignoring anything outside the alphabet
 *
 * Whitespace, line breaks and stray characters are skipped rather than
 * refused, because a base64 payload that has been through an editor, an
 * email or a JSON pretty-printer has some of all three in it. A length
 * that is not a multiple of four is the one thing that cannot be
 * recovered from, and it returns false.
 */

inline bool base64_decode(const std::string& text, std::string& out) {
    unsigned char table[256];

    for (int i = 0; i < 256; ++i) {
        table[i] = 0x80;
    }

    const char* tbl = detail::base64_alphabet();

    for (int i = 0; i < 64; ++i) {
        table[static_cast<unsigned char>(tbl[i])] = static_cast<unsigned char>(i);
    }

    table[static_cast<unsigned char>('=')] = 0;

    std::size_t count = 0;

    for (char c : text) {
        if (table[static_cast<unsigned char>(c)] != 0x80) {
            ++count;
        }
    }

    out.clear();

    if (count == 0) {
        return true;
    }

    if (count % 4) {
        return false;
    }

    out.reserve(count / 4 * 3);
    unsigned char block[4] = {0, 0, 0, 0};
    std::size_t have = 0;
    int pad = 0;

    for (char c : text) {
        const unsigned char v = table[static_cast<unsigned char>(c)];

        if (v == 0x80) {
            continue;
        }

        if (c == '=') {
            ++pad;
        }

        block[have++] = v;

        if (have < 4) {
            continue;
        }

        out += static_cast<char>((block[0] << 2) | (block[1] >> 4));
        out += static_cast<char>((block[1] << 4) | (block[2] >> 2));
        out += static_cast<char>((block[2] << 6) | block[3]);
        have = 0;

        if (pad) {
            if (pad > 2) {
                return false;
            }

            out.resize(out.size() - static_cast<std::size_t>(pad));
            break;
        }
    }

    return true;
}

} // namespace mimamo

#endif
