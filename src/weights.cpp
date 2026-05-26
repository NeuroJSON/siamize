/***************************************************************************//**
**  \mainpage siamize - native C++/ONNX port of SIAM v0.3 brain segmentation
**
**  \author    Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2026
**
**  \section sref Reference:
**  \li \c (\b Valabregue2026) Romain Valabregue, Ikram Khemir, Eric Bardinet,
**         Francois Rousseau, Guillaume Auzias, Reuben Dorent, "SIAM: Head and
**         Brain MRI Segmentation from Few High-Quality Templates via Synthetic
**         Training," arXiv:2605.02737 (2026).
**         <a href="https://arxiv.org/abs/2605.02737">arxiv.org/abs/2605.02737</a>
**  \li \c (\b NeuroJSON) NeuroJSON project,
**         <a href="https://neurojson.org">neurojson.org</a>
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    weights.cpp
\brief   Lazy weight cache + NeuroJSON auto-download implementation

When the user does not supply a local `.onnx` weight path, this
module shells out to the system `curl` to fetch the requested fold
from NeuroJSON's CGI endpoint. The downloaded `.onnx.gz` is
inflated in-place via zmat's miniz instance (the same one
nifti_io.cpp activates with ZMAT_IMPLEMENTATION), so the binary
keeps zero dependency on a system zlib.

The cache directory and the base URL are environment-overridable
(`SIAMIZE_CACHE_DIR`, `SIAMIZE_WEIGHTS_BASE_URL`) for CI runners
and offline / proxied installations.
*******************************************************************************/

#include "weights.h"
#include "siam_log.h"

/**
 * @brief Forward declaration of zmat's bundled miniz one-shot gzip inflater
 *
 * zmat.h gates its implementation behind `ZMAT_IMPLEMENTATION`, which
 * is defined exactly once (in nifti_io.cpp). Forward-declaring the
 * helper here lets us call into the same linked-in copy without
 * pulling in the whole 6500-line header.
 */
int miniz_gzip_uncompress(void* in_data, size_t in_len,
                          void** out_data, size_t* out_len);

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace siam {

namespace {

/*******************************************************************************/
/*! \fn    const char* env_or(const char* name, const char* fallback)
    \brief Read an environment variable or fall back to a default string
    \param  name      environment variable name
    \param  fallback  used when the variable is unset or empty
    \return           the variable value or \a fallback
*/
const char* env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : fallback;
}

/*******************************************************************************/
/*! \fn    std::string quote(const std::string& s)
    \brief Shell-quote a path argument for use inside a `std::system` command

    POSIX path: wraps in single quotes and escapes embedded `'` by
    closing/reopening the quoting; Windows path: wraps in double
    quotes and escapes embedded `"`. Only the embedded quote
    character is special-cased; everything else passes through
    literally, which is what curl and the shell expect for a file
    path argument.

    \param  s  the raw argument string
    \return    the quoted form, safe for command-line use
*/
std::string quote(const std::string& s) {
#ifdef _WIN32
    std::string out = "\"";

    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }

    out += "\"";
    return out;
#else
    std::string out = "'";

    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }

    out += "'";
    return out;
#endif
}

/*******************************************************************************/
/*! \fn    bool run_silent(const std::string& cmd)
    \brief Run a shell command and return whether it exited with status 0
    \param  cmd  the command line to pass to std::system
    \return      true on exit code 0
*/
bool run_silent(const std::string& cmd) {
    return std::system(cmd.c_str()) == 0;
}

/*******************************************************************************/
/*! \fn    std::vector<uint8_t> read_file_bytes(const std::string& path)
    \brief Slurp a file's bytes into memory
    \param  path  source file path
    \return       file contents as a byte vector
*/
std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);

    if (!f) {
        throw std::runtime_error("failed to open " + path);
    }

    auto sz = static_cast<std::streamsize>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));

    if (sz > 0 && !f.read(reinterpret_cast<char*>(buf.data()), sz)) {
        throw std::runtime_error("failed to read " + path);
    }

    return buf;
}

/*******************************************************************************/
/*! \fn    void write_file_bytes(const std::string& path,
                                 const uint8_t* data, size_t n)
    \brief Write \a n bytes of \a data to \a path, truncating any existing file
    \param  path  destination file path
    \param  data  byte buffer
    \param  n     number of bytes
*/
void write_file_bytes(const std::string& path, const uint8_t* data, size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);

    if (!f) {
        throw std::runtime_error("failed to open " + path + " for writing");
    }

    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));

    if (!f) {
        throw std::runtime_error("failed to write " + path);
    }
}

/*******************************************************************************/
/*! \fn    std::vector<uint8_t> gunzip(const uint8_t* in, size_t n)
    \brief Inflate an in-memory gzip blob via zmat/miniz

    Calls the forward-declared `miniz_gzip_uncompress` whose definition
    lives in zmat.h's ZMAT_IMPLEMENTATION block (instantiated by
    nifti_io.cpp). The output buffer is malloc'd by miniz; we copy
    into a std::vector and free the original.

    \param  in  gzip-encoded buffer (must start with magic 0x1F 0x8B)
    \param  n   number of bytes in \a in
    \return     decompressed bytes
*/
std::vector<uint8_t> gunzip(const uint8_t* in, size_t n) {
    void* out = nullptr;
    size_t outlen = 0;
    int rc = miniz_gzip_uncompress(const_cast<uint8_t*>(in), n, &out, &outlen);

    if (rc != 0 || out == nullptr) {
        if (out) {
            std::free(out);
        }

        throw std::runtime_error("gzip decode failed (rc=" + std::to_string(rc) + ")");
    }

    std::vector<uint8_t> result(static_cast<uint8_t*>(out),
                                static_cast<uint8_t*>(out) + outlen);
    std::free(out);
    return result;
}

}  // anonymous namespace

/*******************************************************************************/
/*! \fn    std::string default_cache_dir()
    \brief Resolve siamize's local weight-cache directory
    \return  absolute path to the cache directory (no trailing slash)
*/
std::string default_cache_dir() {
    const char* env = std::getenv("SIAMIZE_CACHE_DIR");

    if (env && *env) {
        return env;
    }

#ifdef _WIN32
    const char* base = env_or("LOCALAPPDATA", env_or("USERPROFILE", "."));
    return std::string(base) + "\\siamize\\models";
#else
    const char* home = env_or("HOME", ".");
    return std::string(home) + "/.cache/siamize/models";
#endif
}

/*******************************************************************************/
/*! \fn    std::string default_weights_url()
    \brief Resolve the NeuroJSON URL prefix for auto-fetched weights
    \return  URL prefix (env-overridable; see weights.h)
*/
/*******************************************************************************/
/*! \fn    const char* variant_str(WeightVariant variant)
    \brief Map WeightVariant to its on-server `doc=` value / cache subdir name
    \return  null-terminated string ("dynshape", "fixshape", "coreml")
*/
const char* variant_str(WeightVariant variant) {
    switch (variant) {
        case WeightVariant::FIXSHAPE: return "fixshape";
        case WeightVariant::COREML:   return "coreml";
        case WeightVariant::DYNSHAPE:
        default:                      return "dynshape";
    }
}

std::string default_weights_url(WeightVariant variant) {
    const char* env = std::getenv("SIAMIZE_WEIGHTS_BASE_URL");

    if (env && *env) {
        return env;
    }

    // doc=dynshape: fp16 ONNX with dynamic_axes={D,H,W}. Default for
    //   CUDA / TensorRT / CPU EPs; supports --lowmem patch shrink.
    // doc=fixshape: fp16 ONNX locked to 256x256x192. Legacy CI bundle.
    // doc=coreml:   fp16 fixed-shape ONNX with rank-5 InstanceNorm
    //   rewritten to rank-3 (see tools/onnx_export/rewrite_for_coreml.py)
    //   so Apple's mlcompilerd accepts it. Used by the CoreML EP.
    return std::string(
               "https://neurojson.org/io/stat.cgi?action=get&db=siam_v03"
               "&doc=") + variant_str(variant) + "&file=";
}

/*******************************************************************************/
/*! \fn    std::string resolve_model_path(const std::string& spec, bool verbose)
    \brief Resolve a model spec to an on-disk path, fetching from NeuroJSON if needed

    Three-stage lookup: (1) spec is an existing file; (2) the spec's
    basename is already in the cache; (3) `curl`-download the weight
    from NeuroJSON (trying `<basename>.gz` first, falling back to
    `<basename>` raw) into the cache. The gzipped path is decompressed
    in-memory via the zmat-bundled miniz so we don't depend on `gunzip`
    being on PATH.

    \param  spec     fold spec (filename, digit shortcut, or full path)
    \param  verbose  print per-step progress to stderr
    \return          absolute path to the resolved .onnx file
    \throws std::runtime_error if none of the lookup stages produces a file
*/
std::string resolve_model_path(const std::string& spec, bool verbose,
                               WeightVariant variant) {
    namespace fs = std::filesystem;

    if (spec.empty()) {
        throw std::runtime_error("empty model spec");
    }

    // 1. Path exists as-is (absolute, relative, or in CWD).
    std::error_code ec;

    if (fs::exists(spec, ec)) {
        return spec;
    }

    // 2. Look up <cache_dir>/<basename>. Non-default variants live in
    //    a per-variant subdir so the same basename (fold_0_fp16.onnx)
    //    doesn't collide across the three weight sets on disk.
    fs::path cache = default_cache_dir();

    if (variant != WeightVariant::DYNSHAPE) {
        cache /= variant_str(variant);
    }

    fs::path base = fs::path(spec).filename();
    fs::path cached = cache / base;

    if (fs::exists(cached, ec)) {
        siam::log_tag("weights", "cached: %s", cached.string().c_str());
        (void)verbose;
        return cached.string();
    }

    // 3. Download from default URL into the cache dir. Try the .gz form
    //    first; decompress via zmat's bundled miniz (no external tool).
    //    Fall back to the raw uncompressed URL if the .gz form 404s.
    fs::create_directories(cache, ec);
    std::string url_base = default_weights_url(variant);
    std::string verbose_curl = verbose ? "-#" : "-s";

    // url_base is expected to end with the parameter that takes the
    // filename (e.g. `...&file=`); concatenate the basename directly.
    std::string url_gz = url_base + base.string() + ".gz";
    std::string archive_gz = (cache / (base.string() + ".gz")).string();

    siam::log_tag("weights", "fetching %s", url_gz.c_str());

    std::string cmd_dl_gz = "curl -fL " + verbose_curl + " -o "
                            + quote(archive_gz) + " " + quote(url_gz);

    if (run_silent(cmd_dl_gz) && fs::exists(archive_gz, ec)) {
        try {
            auto gz_bytes = read_file_bytes(archive_gz);
            auto raw_bytes = gunzip(gz_bytes.data(), gz_bytes.size());
            write_file_bytes(cached.string(), raw_bytes.data(), raw_bytes.size());
            siam::log_tag("weights", "decompressed %s (%zu -> %zu bytes)",
                          base.string().c_str(),
                          gz_bytes.size(), raw_bytes.size());
            fs::remove(archive_gz, ec);
            return cached.string();
        } catch (const std::exception& e) {
            // Fall through to the raw fallback below.
            siam::log_tag("weights", "gzip decode failed: %s", e.what());
            fs::remove(archive_gz, ec);
        }
    }

    // Try raw uncompressed URL as a last resort.
    std::string url_raw = url_base + base.string();
    siam::log_tag("weights", "fetching %s -> %s",
                  url_raw.c_str(), cached.string().c_str());

    std::string cmd_dl_raw = "curl -fL " + verbose_curl + " -o "
                             + quote(cached.string()) + " " + quote(url_raw);

    if (run_silent(cmd_dl_raw) && fs::exists(cached, ec)) {
        return cached.string();
    }

    fs::remove(cached, ec);
    throw std::runtime_error(
        "could not find or fetch model: " + spec
        + "\n  looked in: " + cache.string()
        + "\n  tried URL: " + url_gz
        + "\n  also tried: " + url_raw
        + "\nensure `curl` is on PATH, or pre-place the file at the cache path.");
}

}  // namespace siam
