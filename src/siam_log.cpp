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
\file    siam_log.cpp
\brief   siam_log API implementation (CLI stderr / MEX mexPrintf routing)

In the CLI build, log lines go to `stderr` via `std::fprintf`. In a
MEX build (`MATLAB_MEX_FILE` defined by both MATLAB's matlab_add_mex
and Octave's mkoctfile --mex), lines go to `mexPrintf` so the MATLAB
or Octave command window receives them on every platform.

The verbose-mode flag is a single global bool that callers set once
at startup; subsequent log_tag / log_cont calls check it internally
so call sites stay uncluttered.
*******************************************************************************/

#include "siam_log.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#ifdef MATLAB_MEX_FILE
    #include "mex.h"
#endif

#ifdef _WIN32
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

// A Windows console *is* a tty, but it prints raw escape bytes unless
// ENABLE_VIRTUAL_TERMINAL_PROCESSING is enabled (Win10+: Windows Terminal,
// PowerShell, modern conhost). Enable it once on stderr and report whether
// stderr is now an ANSI-capable console. Redirected stderr (pipe/file) is not
// a console -> GetConsoleMode fails -> returns false -> no color (correct).
inline bool siam_stderr_ansi() {
    static const bool ok = [] {
        if (_isatty(_fileno(stderr)) == 0) {
            return false;
        }

        HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stderr)));

        if (h == nullptr || h == INVALID_HANDLE_VALUE) {
            return false;
        }

        DWORD mode = 0;

        if (GetConsoleMode(h, &mode) == 0) {
            return false;
        }

        if (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) {
            return true;
        }

        return SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
    }();
    return ok;
}
#define SIAM_ISATTY_STDERR() (siam_stderr_ansi())
#else
#include <unistd.h>
#define SIAM_ISATTY_STDERR() (isatty(fileno(stderr)) != 0)
#endif

namespace siam {

namespace {

bool g_verbose = false;          ///< verbose-mode flag (see set_verbose)

/// Width of the bracketed-tag column in characters. Tags wider than
/// (LOG_TAG_COL - 2) overflow without padding so the body column
/// shifts right by the overflow amount on those lines.
constexpr int LOG_TAG_COL = 10;

/*******************************************************************************/
/*! \fn    const char* tag_color(const char* tag)
    \brief Pick an ANSI color escape for a tag name (TTY-only path)

    Three buckets:

      - `[warn]`            -> bold yellow (highest visibility)
      - `[hint]`            -> bold cyan (suggestion)
      - everything else     -> bold cyan (default accent)

    The CLI calls log_tag heavily; using a single accent for the
    default keeps the eye anchored on the column without
    rainbow-colouring every distinct phase.
*/
const char* tag_color(const char* tag) {
    if (!tag) {
        return "\033[1;36m";
    }

    if (std::strcmp(tag, "warn") == 0) {
        return "\033[1;33m";    // bold yellow
    }

    if (std::strcmp(tag, "hint") == 0) {
        return "\033[1;36m";    // bold cyan
    }

    return "\033[1;36m";        // default: bold cyan
}

#define SIAM_ANSI_RESET "\033[0m"

/*******************************************************************************/
/*! \fn    void format_padded_tag(char* out, size_t outsz, const char* tag, bool use_color)
    \brief Build the column-aligned `[tag]<padding> ` prefix string

    The visible column width is LOG_TAG_COL + 1 (one trailing space).
    When `use_color` is true the bracketed prefix is wrapped in ANSI
    color codes, but the padding is computed against the *visible*
    length so column alignment is preserved on coloured output.
*/
void format_padded_tag(char* out, size_t outsz, const char* tag, bool use_color) {
    const int visible = 2 + static_cast<int>(std::strlen(tag));  // "[%s]"
    int pad = LOG_TAG_COL - visible;

    if (pad < 0) {
        pad = 0;
    }

    if (use_color) {
        std::snprintf(out, outsz, "%s[%s]%s%*s ",
                      tag_color(tag), tag, SIAM_ANSI_RESET, pad, "");
    } else {
        std::snprintf(out, outsz, "[%s]%*s ", tag, pad, "");
    }
}

/*******************************************************************************/
/*! \fn    void emit(const char* s)
    \brief Send a fully-formed log line to the current output channel

    Stays a single-line helper so log_tag / log_cont / log_warn can
    share one routing decision: stderr for CLI, mexPrintf for MEX.

    \param  s  null-terminated string already containing its trailing newline
*/
void emit(const char* s) {
#ifdef MATLAB_MEX_FILE
    mexPrintf("%s", s);
    // MATLAB buffers mexPrintf output until the MEX function returns
    // (Octave doesn't, but the call is cheap on Octave anyway). A
    // drawnow forces the Command Window to flush the buffer so the
    // user sees [tile] / [resample] / etc. live during the run
    // instead of all at once at the end. mexEvalString is safe to
    // call from any MEX context; siam_log helpers are only invoked
    // from the main thread (not inside OMP parallel regions).
    mexEvalString("drawnow;");
#else
    std::fputs(s, stderr);
#endif
}

/*******************************************************************************/
/*! \fn    void emit_formatted(const char* head, const char* fmt, va_list ap)
    \brief Format <head><body>\n into a stack buffer and dispatch to emit()

    Two-stage formatting (body then head + body + newline) keeps the
    string operations bounded: body is capped at 4096 chars, full
    line at 4160. Truncation is silent (defensive only -- siamize's
    log strings are short).

    \param  head  prefix string (already aligned/padded)
    \param  fmt   printf-style body format
    \param  ap    varargs list, already started by the caller
*/
void emit_formatted(const char* head, const char* fmt, va_list ap) {
    char body[4096];
    std::vsnprintf(body, sizeof(body), fmt, ap);
    char line[4160];
    std::snprintf(line, sizeof(line), "%s%s\n", head, body);
    emit(line);
}

}  // anonymous namespace

/*******************************************************************************/
/*! \fn    void set_verbose(bool on)
    \brief Enable or disable verbose-mode output globally
    \param  on  true to print log_tag / log_cont messages
*/
void set_verbose(bool on) {
    g_verbose = on;
}

/*******************************************************************************/
/*! \fn    bool verbose()
    \brief Return the current verbose-mode flag
*/
bool verbose() {
    return g_verbose;
}

/*******************************************************************************/
/*! \fn    void log_tag(const char* tag, const char* fmt, ...)
    \brief Emit a tagged, column-aligned status line (verbose-gated)
*/
void log_tag(const char* tag, const char* fmt, ...) {
    if (!g_verbose) {
        return;
    }

    bool color = false;
#ifndef MATLAB_MEX_FILE
    color = SIAM_ISATTY_STDERR();
#endif
    char head[96];
    format_padded_tag(head, sizeof(head), tag, color);

    va_list ap;
    va_start(ap, fmt);
    emit_formatted(head, fmt, ap);
    va_end(ap);
}

/*******************************************************************************/
/*! \fn    void log_cont(const char* fmt, ...)
    \brief Emit a continuation line indented under the body column
*/
void log_cont(const char* fmt, ...) {
    if (!g_verbose) {
        return;
    }

    // (LOG_TAG_COL + 1) spaces: one for each character of the padded
    // bracketed tag plus the separator space that follows it.
    const char* head = "           ";  // 11 spaces (matches LOG_TAG_COL=10 + 1)

    va_list ap;
    va_start(ap, fmt);
    emit_formatted(head, fmt, ap);
    va_end(ap);
}

/*******************************************************************************/
/*! \fn    void log_progress(const char* tag, long long current, long long total)
    \brief Render a TTY-aware progress bar; one-line-per-call fallback otherwise

    Strategy:

      - When stderr is a TTY and not in a MEX context: emit
        "\\r[tag]      [#####---------] N/M (XX%)" with trailing
        spaces to pad over residue from any previous longer line.
        On the final step (current >= total) append a newline so
        the next log line starts fresh.
      - Otherwise (piped stderr, redirected output, or MEX context):
        emit a normal log_tag-style line per call. The user gets a
        progress trail that survives capture, at the cost of N
        lines of output instead of one self-updating row.
*/
void log_progress(const char* tag, long long current, long long total) {
    if (!g_verbose) {
        return;
    }

    if (total <= 0) {
        return;
    }

    bool color = false;
#ifndef MATLAB_MEX_FILE
    color = SIAM_ISATTY_STDERR();
#endif
    char head[96];
    format_padded_tag(head, sizeof(head), tag, color);

    // Fixed-width 20-cell bar, ASCII so it renders on every terminal.
    const int BAR_W = 20;
    int filled = static_cast<int>((current * static_cast<long long>(BAR_W)) / total);

    if (filled < 0) {
        filled = 0;
    }

    if (filled > BAR_W) {
        filled = BAR_W;
    }

    char bar[BAR_W + 1];

    for (int i = 0; i < BAR_W; ++i) {
        bar[i] = (i < filled) ? '#' : '-';
    }

    bar[BAR_W] = '\0';

    double pct = (current * 100.0) / total;

#ifdef MATLAB_MEX_FILE
    // MEX: no TTY semantics; emit a normal line per call.
    char line[256];
    std::snprintf(line, sizeof(line), "%s[%s] %lld/%lld (%.0f%%)\n",
                  head, bar, current, total, pct);
    mexPrintf("%s", line);
    mexEvalString("drawnow;");   // flush; see emit() above for rationale
#else

    if (SIAM_ISATTY_STDERR()) {
        // TTY: \r-redraw on the same line, trailing newline only at finish.
        // The 8 trailing spaces overwrite residue from any shorter prior
        // line (e.g. "9/100" -> "10/100" gains a digit; the pad makes
        // sure the previous string is fully clobbered).
        char line[256];
        std::snprintf(line, sizeof(line), "\r%s[%s] %lld/%lld (%.0f%%)        ",
                      head, bar, current, total, pct);
        std::fputs(line, stderr);

        if (current >= total) {
            std::fputc('\n', stderr);
        }

        std::fflush(stderr);
    } else {
        // Piped / redirected: one full line per call so the log
        // remains usable when captured to a file.
        char line[256];
        std::snprintf(line, sizeof(line), "%s[%s] %lld/%lld (%.0f%%)\n",
                      head, bar, current, total, pct);
        std::fputs(line, stderr);
    }

#endif
}

/*******************************************************************************/
/*! \fn    void log_warn(const char* fmt, ...)
    \brief Emit an unconditional warning line (ignores the verbose flag)
*/
void log_warn(const char* fmt, ...) {
    bool color = false;
#ifndef MATLAB_MEX_FILE
    color = SIAM_ISATTY_STDERR();
#endif
    char head[96];
    format_padded_tag(head, sizeof(head), "warn", color);

    va_list ap;
    va_start(ap, fmt);
    emit_formatted(head, fmt, ap);
    va_end(ap);
}

/*******************************************************************************/
/*! \fn    void log_hint(const char* fmt, ...)
    \brief Emit an unconditional hint line (ignores the verbose flag)
*/
void log_hint(const char* fmt, ...) {
    bool color = false;
#ifndef MATLAB_MEX_FILE
    color = SIAM_ISATTY_STDERR();
#endif
    char head[96];
    format_padded_tag(head, sizeof(head), "hint", color);

    va_list ap;
    va_start(ap, fmt);
    emit_formatted(head, fmt, ap);
    va_end(ap);
}

}  // namespace siam
