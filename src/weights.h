// Auto-fetch SIAM model weights from the NeuroJSON server when the user
// hasn't supplied a local path. Pattern mirrors umcx's `runcmd("curl ...")`
// trick: shell out to a system `curl` (no library dependency).
//
// Default URLs / paths can be overridden by environment variables so power
// users and CI runners don't need to recompile.

#pragma once
#include <string>

namespace siam {

// Local directory siamize uses for cached / auto-downloaded weights.
//   Linux/macOS : $HOME/.cache/siamize/models/
//   Windows     : %LOCALAPPDATA%/siamize/models/  (fallback %USERPROFILE%)
//   Override    : env SIAMIZE_CACHE_DIR
// Created on demand by the resolver below; the function itself is pure.
std::string default_cache_dir();

// URL prefix that auto-fetched weight files live under. The prefix is
// expected to end with the NeuroJSON query parameter that takes the
// filename (e.g. `...&file=`); the resolver appends `<basename>.gz`
// (preferred) or `<basename>` (raw fallback) directly, with no `/`
// separator.
//   Default  : https://neurojson.org/io/stat.cgi?action=get&db=siam_v03&doc=dynshape&size=95360591&file=
//   Override : env SIAMIZE_WEIGHTS_BASE_URL
// The `size=` query parameter on the default URL is informational only;
// the NeuroJSON server does not validate it against the actual file
// size, so the same constant is used for every fold.
std::string default_weights_url();

// Resolve a model spec to an existing local file path:
//   1. If `spec` exists as-is on disk, return its path verbatim.
//   2. Otherwise compute basename(spec) and check <cache_dir>/<basename>.
//   3. Otherwise curl-download <url_base>/<basename>.7z, extract via 7z,
//      place at <cache_dir>/<basename>, return its path. If the .7z form
//      404s, fall back to the raw <url_base>/<basename>.
// Throws std::runtime_error if none of the above produced a file.
// `verbose` controls progress messages on stderr.
std::string resolve_model_path(const std::string& spec, bool verbose);

}  // namespace siam
