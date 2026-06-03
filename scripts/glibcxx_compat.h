/* glibcxx_compat.h -- force-included into the MNN build by fetch_mnn.sh.
 *
 * Binds std::condition_variable::wait(std::unique_lock<std::mutex>&) to its
 * long-standing GLIBCXX_3.4.11 symbol instead of the gcc-12 default
 * @GLIBCXX_3.4.30.
 *
 * Why: MNN's thread pool (ThreadPool.cpp / WorkerThread.cpp) is the only user
 * of this symbol, and libMNN's .o files carry an UNVERSIONED reference -- the
 * version is stamped at the final link against the build host's libstdc++.so.
 * On a gcc-12 host (Ubuntu 22.04+) the default for this symbol is 3.4.30, which
 * is ABSENT from MATLAB's bundled libstdc++, so an MNN-backed MEX built there
 * fails to load with "version `GLIBCXX_3.4.30' not found". The 3.4.11 alias is
 * present in every libstdc++ (incl. MATLAB's and gcc-9/10/11/12 runtimes), so
 * pinning it keeps the MEX / CLI / docker binaries portable -- without building
 * in an old-libstdc++ container.
 *
 * This is a pure assembler directive: it needs no libstdc++ header, so it works
 * via -include before any header is parsed. Do NOT guard on __GLIBCXX__ -- that
 * macro is undefined at force-include time (before any C++ header is read), so
 * guarding on it would silently no-op the directive. The 3.4.11 symbol is the
 * long-standing implementation of wait(lock); fine for MNN's basic usage.
 *
 * ONLY for libMNN that will be linked with DYNAMIC libstdc++ (the MEX). A
 * versioned reference cannot be resolved by `-static-libstdc++` (the static
 * libstdc++.a has no version nodes -> "undefined reference to ...@3.4.11"), so
 * the CLI / docker static builds must NOT use this. fetch_mnn.sh gates it
 * behind SIAMIZE_GLIBCXX_COMPAT=1 and leaves it off by default.
 *
 * Inert outside Linux/ELF (clang on macOS, MSVC on Windows) and in C TUs.
 */
#if defined(__linux__) && defined(__ELF__)
__asm__(".symver _ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE,"
        "_ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE@GLIBCXX_3.4.11");
#endif
