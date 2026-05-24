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
\file    weights.h

@brief   Lazy model-weight cache + NeuroJSON auto-download interface

Auto-fetches SIAM fp16 ONNX fold weights from NeuroJSON's CGI
endpoint when the user has not supplied a local path. The pattern
mirrors umcx's `runcmd("curl ...")` trick: shell out to system
`curl` so the binary keeps zero library-level dependencies on
HTTP / TLS.

Both the cache directory and the URL prefix are configurable via
environment variables so CI runners and offline users can override
without recompiling.
*******************************************************************************/

#ifndef SIAMIZE_WEIGHTS_H
#define SIAMIZE_WEIGHTS_H

#include <string>

namespace siam {

/**
 * @brief Resolve siamize's local model-weight cache directory
 *
 * Default lookup:
 *
 *   - Linux/macOS: `$HOME/.cache/siamize/models/`
 *   - Windows:     `%LOCALAPPDATA%/siamize/models/`
 *                  (fallback `%USERPROFILE%/.cache/siamize/models/`)
 *
 * Override via the `SIAMIZE_CACHE_DIR` environment variable.
 * Created on demand by the resolver below; this function itself
 * is pure and does no I/O.
 *
 * @return absolute path to the cache directory (no trailing slash)
 */
std::string default_cache_dir();

/**
 * @brief Resolve the NeuroJSON URL prefix used for auto-fetched weights
 *
 * The prefix is expected to end with the CGI query parameter that
 * takes the filename (e.g. `...&file=`); the resolver appends the
 * weight basename directly with no separator. The `size=` query
 * parameter in the default URL is informational only -- NeuroJSON
 * does not validate it -- so the same constant works for every
 * fold.
 *
 * Default:
 *
 *     https://neurojson.org/io/stat.cgi?action=get&db=siam_v03&doc=dynshape&size=95360591&file=
 *
 * Override via the `SIAMIZE_WEIGHTS_BASE_URL` environment variable.
 *
 * @return the URL prefix (or its override)
 */
std::string default_weights_url();

/**
 * @brief Resolve a model spec to an existing local file path, fetching if needed
 *
 * Lookup chain:
 *
 *   1. If \a spec exists as-is on disk, return its path verbatim.
 *   2. Otherwise compute `basename(spec)` and check
 *      `<cache_dir>/<basename>`; return that if present.
 *   3. Otherwise `curl`-download `<url_base>/<basename>` into
 *      `<cache_dir>/<basename>` (with a `.gz`-suffixed attempt
 *      first when supported), and return the cached path.
 *
 * Throws `std::runtime_error` if none of the above produces a
 * file. Use \a verbose to surface per-step progress messages on
 * stderr.
 *
 * @param  spec     fold spec: filename, digit shortcut, or full path
 * @param  verbose  true to print progress to stderr
 * @return          absolute path to the resolved .onnx file
 */
std::string resolve_model_path(const std::string& spec, bool verbose);

}  // namespace siam

#endif  // SIAMIZE_WEIGHTS_H
