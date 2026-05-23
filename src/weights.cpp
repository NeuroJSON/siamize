#include "weights.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

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

    // 3. Download from default URL into the cache dir.
    fs::create_directories(cache, ec);
    std::string url_base = default_weights_url();
    std::string verbose_curl = verbose ? "-#" : "-s";

    // Try compressed (.7z) first.
    std::string url_7z = url_base + "/" + base.string() + ".7z";
    std::string archive = (cache / (base.string() + ".7z")).string();

    if (verbose) {
        std::fprintf(stderr, "  fetching %s -> %s\n", url_7z.c_str(), archive.c_str());
    }

    std::string cmd_dl_7z = "curl -fL " + verbose_curl + " -o "
                            + quote(archive) + " " + quote(url_7z);

    if (run_silent(cmd_dl_7z)) {
        std::string cmd_extract = "7z x -y -bso0 -bsp0 -o" + quote(cache.string())
                                  + " " + quote(archive);

        if (run_silent(cmd_extract) && fs::exists(cached, ec)) {
            fs::remove(archive, ec);
            return cached.string();
        }

        // Fall through to the raw fallback below.
        fs::remove(archive, ec);
    }

    // Try uncompressed fallback.
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
        + "\n  tried URL: " + url_7z
        + "\n  also tried: " + url_raw
        + "\nensure `curl` and `7z` are on PATH, or pre-place the file at the cache path.");
}

}  // namespace siam
