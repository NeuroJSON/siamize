#include "weights.h"

// zmat's implementation lives in nifti_io.cpp; the public header only
// exposes prototypes under ZMAT_IMPLEMENTATION, so we forward-declare the
// single function we need here. The linker resolves it against the
// definition in the same binary.
// Matches the C++ linkage used when nifti_io.cpp pulls in zmat.h with
// ZMAT_IMPLEMENTATION. Same forward-declaration both sides; the linker
// resolves it inside the same binary.
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

const char* env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : fallback;
}

// Quote a shell argument with single quotes (POSIX) or double quotes (Windows).
// We only escape the embedded quote char to keep paths sane in the resulting
// `system()` invocation.
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

bool run_silent(const std::string& cmd) {
    return std::system(cmd.c_str()) == 0;
}

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

// Decompress an in-memory gzip blob via zmat's miniz_gzip_uncompress.
// The output buffer is malloc'd inside miniz; we copy it to a vector and
// free the original.
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

}  // namespace

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

std::string default_weights_url() {
    const char* env = std::getenv("SIAMIZE_WEIGHTS_BASE_URL");

    if (env && *env) {
        return env;
    }

    return "https://neurojson.org/siamize/weights/siam_v03";
}

std::string resolve_model_path(const std::string& spec, bool verbose) {
    namespace fs = std::filesystem;

    if (spec.empty()) {
        throw std::runtime_error("empty model spec");
    }

    // 1. Path exists as-is (absolute, relative, or in CWD).
    std::error_code ec;

    if (fs::exists(spec, ec)) {
        return spec;
    }

    // 2. Look up <cache_dir>/<basename>.
    fs::path cache = default_cache_dir();
    fs::path base = fs::path(spec).filename();
    fs::path cached = cache / base;

    if (fs::exists(cached, ec)) {
        if (verbose) {
            std::fprintf(stderr, "  using cached weight: %s\n", cached.string().c_str());
        }

        return cached.string();
    }

    // 3. Download from default URL into the cache dir. Try the .gz form
    //    first; decompress via zmat's bundled miniz (no external tool).
    //    Fall back to the raw uncompressed URL if the .gz form 404s.
    fs::create_directories(cache, ec);
    std::string url_base = default_weights_url();
    std::string verbose_curl = verbose ? "-#" : "-s";

    std::string url_gz = url_base + "/" + base.string() + ".gz";
    std::string archive_gz = (cache / (base.string() + ".gz")).string();

    if (verbose) {
        std::fprintf(stderr, "  fetching %s\n", url_gz.c_str());
    }

    std::string cmd_dl_gz = "curl -fL " + verbose_curl + " -o "
                            + quote(archive_gz) + " " + quote(url_gz);

    if (run_silent(cmd_dl_gz) && fs::exists(archive_gz, ec)) {
        try {
            auto gz_bytes = read_file_bytes(archive_gz);
            auto raw_bytes = gunzip(gz_bytes.data(), gz_bytes.size());
            write_file_bytes(cached.string(), raw_bytes.data(), raw_bytes.size());

            if (verbose) {
                std::fprintf(stderr,
                             "  decompressed %s (%zu -> %zu bytes)\n",
                             base.string().c_str(),
                             gz_bytes.size(), raw_bytes.size());
            }

            fs::remove(archive_gz, ec);
            return cached.string();
        } catch (const std::exception& e) {
            // Fall through to the raw fallback below.
            if (verbose) {
                std::fprintf(stderr, "  gzip decode failed: %s\n", e.what());
            }

            fs::remove(archive_gz, ec);
        }
    }

    // Try raw uncompressed URL as a last resort.
    std::string url_raw = url_base + "/" + base.string();

    if (verbose) {
        std::fprintf(stderr, "  fetching %s -> %s\n", url_raw.c_str(), cached.string().c_str());
    }

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
