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
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    siam_log.h

@brief   Verbose-mode console logger with tagged, column-aligned output

Centralized logging API used by the CLI and MEX paths. Output is
formatted as:

    [tag]      <body>

with the bracketed tag left-justified in a 10-character column and
a single separator space before the body. Continuation lines indent
to align under the body column. Long tags ("[resample]", "[fallback]",
...) overflow gracefully without padding.

Routing:

  - CLI builds emit through `std::fprintf(stderr, ...)`.
  - MEX builds (`MATLAB_MEX_FILE` defined) emit through `mexPrintf`
    so messages reach the MATLAB / Octave command window portably.

Verbose-mode gating:

  - log_tag() and log_cont() are no-ops when verbose() is false.
  - log_warn() always emits, regardless of the verbose flag.

Call set_verbose(true) once at startup (from main() / mexFunction)
to enable progress output.
*******************************************************************************/

#ifndef SIAMIZE_SIAM_LOG_H
#define SIAMIZE_SIAM_LOG_H

namespace siam {

/**
 * @brief Enable or disable verbose-mode output globally
 *
 * Defaults to false. Callers set this once at the top of main() (or
 * mexFunction) based on the user's -v / opts.verbose flag, then call
 * log_tag / log_cont freely without gating each call.
 *
 * @param  on  true to print log_tag / log_cont messages
 */
void set_verbose(bool on);

/**
 * @brief Query the current verbose-mode flag
 * @return value last passed to set_verbose() (false if never called)
 */
bool verbose();

/**
 * @brief Emit a tagged status line
 *
 * Formats as `[tag]<padding> <body>\n`. The bracketed tag is padded
 * to a 10-character left-justified column; if the tag itself is
 * longer than 8 chars, it overflows without padding and the body
 * starts one column further right.
 *
 *     siam::log_tag("input", "%s", path.c_str());
 *     -> [input]    /tmp/sub-01_T1w.nii.gz
 *
 * No-op when verbose() is false.
 *
 * @param  tag  short tag name (no brackets; the helper adds them)
 * @param  fmt  printf-style format for the body
 */
void log_tag(const char* tag, const char* fmt, ...)
__attribute__((format(printf, 2, 3)));

/**
 * @brief Continuation line aligned under the tag column
 *
 * Useful when one logical line carries more than fits cleanly on a
 * single row. The 11-space indent matches the column the body
 * starts at after a `[tag]` of standard width.
 *
 *     siam::log_tag("input", "%s", path.c_str());
 *     siam::log_cont("shape (%lld,%lld,%lld)", Z, Y, X);
 *     -> [input]    /tmp/sub-01_T1w.nii.gz
 *     ->            shape (192,192,160)
 *
 * No-op when verbose() is false.
 */
void log_cont(const char* fmt, ...)
__attribute__((format(printf, 1, 2)));

/**
 * @brief Unconditional warning (ignores the verbose flag)
 *
 * Emitted with a `[warn]` tag so it lines up with the rest of the
 * verbose stream when verbose is on, but still prints in quiet mode.
 * Intended for warnings the user must see regardless: missing
 * weights, fallback paths, deprecation notices, etc.
 *
 * @param  fmt  printf-style format
 */
void log_warn(const char* fmt, ...)
__attribute__((format(printf, 1, 2)));

}  // namespace siam

#endif  // SIAMIZE_SIAM_LOG_H
