cmake_minimum_required(VERSION 3.16)
project(siamize LANGUAGES CXX)

# siamize's own source is C++11-clean (no structured bindings, if-constexpr,
# std::optional/string_view/filesystem, nested-namespace, or [[nodiscard]]).
# We require C++17 strictly because the ONNX Runtime C API uses `noexcept`
# as part of function-pointer typedefs (the C++17 P0012R1 feature where
# exception specs became part of the type system, e.g.
# `typedef void (*Fn)() noexcept;`). Pre-C++17 compilers reject those
# typedefs even with the `noexcept` keyword itself available. ORT also
# uses C++14-relaxed `constexpr` constructors. Bottom line: ORT pins us
# to C++17; we don't get a choice.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- options ----------------------------------------------------------------
option(SIAMIZE_STATIC_LINK
    "Statically link the C++ runtime / OpenMP into the binary so the only \
non-system shared lib at runtime is libonnxruntime." ON)

option(SIAMIZE_BUILD_MATLAB_MEX
    "Build the MATLAB MEX wrapper (src/siamize_mex.cpp). Auto-skipped if MATLAB \
is not found." OFF)

option(SIAMIZE_BUILD_OCTAVE_MEX
    "Build the GNU Octave MEX wrapper via mkoctfile. Auto-skipped if mkoctfile \
is not found." OFF)

# GPU backend selection. "none" = CPU only (the smallest deployable bundle).
# "cuda" = enable the ONNX Runtime CUDA Execution Provider; requires the
# onnxruntime-gpu prebuilt (fetched via ORT_BUILD=gpu scripts/fetch_deps.sh)
# and a runtime CUDA + cuDNN install. At runtime the binary tries CUDA first
# and falls back to CPU if libcudart / libcudnn cannot be loaded.
set(SIAMIZE_GPU "none" CACHE STRING "GPU backend: none, cuda")
set_property(CACHE SIAMIZE_GPU PROPERTY STRINGS none cuda)

# ---- platform detection -----------------------------------------------------
if(WIN32)
    set(SIAMIZE_PLATFORM "win")
elseif(APPLE)
    set(SIAMIZE_PLATFORM "mac")
else()
    set(SIAMIZE_PLATFORM "linux")
endif()

# ---- ORT (prebuilt) under third_party/onnxruntime/ --------------------------
set(ORT_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
set(ORT_LIB_DIR ${ORT_ROOT}/lib)
if(NOT EXISTS ${ORT_ROOT}/include/onnxruntime_cxx_api.h)
    message(FATAL_ERROR
        "ORT not found at ${ORT_ROOT}. Run scripts/fetch_deps.sh (Linux/macOS) "
        "or scripts/fetch_deps.ps1 (Windows) first.")
endif()

if(WIN32)
    set(ORT_RUNTIME_DLL ${ORT_ROOT}/lib/onnxruntime.dll)
    set(ORT_IMPORT_LIB  ${ORT_ROOT}/lib/onnxruntime.lib)
elseif(APPLE)
    set(ORT_RUNTIME_DLL ${ORT_ROOT}/lib/libonnxruntime.dylib)
else()
    set(ORT_RUNTIME_DLL ${ORT_ROOT}/lib/libonnxruntime.so.1)
endif()

# If a GPU backend was requested, sanity-check that the fetched ORT actually
# contains the right provider plugin(s). The CPU-only tarball does not ship
# them; the GPU tarball ships both CUDA and TensorRT plugin shared libs.
if(SIAMIZE_GPU STREQUAL "cuda" OR SIAMIZE_GPU STREQUAL "tensorrt")
    if(WIN32)
        set(_cuda_provider "${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll")
        set(_trt_provider  "${ORT_ROOT}/lib/onnxruntime_providers_tensorrt.dll")
    else()
        set(_cuda_provider "${ORT_ROOT}/lib/libonnxruntime_providers_cuda.so")
        set(_trt_provider  "${ORT_ROOT}/lib/libonnxruntime_providers_tensorrt.so")
    endif()
    if(NOT EXISTS "${_cuda_provider}")
        message(FATAL_ERROR
            "SIAMIZE_GPU=${SIAMIZE_GPU} but the CUDA provider library was not "
            "found at ${_cuda_provider}.\n"
            "You probably fetched the CPU-only ORT prebuilt. Re-fetch with:\n"
            "  rm -rf third_party/onnxruntime\n"
            "  ORT_BUILD=gpu scripts/fetch_deps.sh")
    endif()
    if(SIAMIZE_GPU STREQUAL "tensorrt" AND NOT EXISTS "${_trt_provider}")
        message(FATAL_ERROR
            "SIAMIZE_GPU=tensorrt but the TensorRT provider library was not "
            "found at ${_trt_provider}.")
    endif()
endif()

# ---- zmat (bundled, single-header at src/zmat/zmat.h) ----------------------
# zmat.h is dual-licensed Apache-2.0 for siamize (see header). It includes
# miniz internally (public domain). No external zlib, no external nifti_clib.
set(ZMAT_INCLUDE ${CMAKE_SOURCE_DIR}/src/zmat)
if(NOT EXISTS ${ZMAT_INCLUDE}/zmat.h)
    message(FATAL_ERROR "zmat.h not found at ${ZMAT_INCLUDE}/zmat.h.")
endif()

# ---- OpenMP (optional everywhere) -------------------------------------------
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "OpenMP not found; the OpenMP pragmas degrade to serial code.")
endif()

# ---- main binary ------------------------------------------------------------
add_executable(siamize
    src/siamize.cpp
    src/nifti_io.cpp
    src/jnifti_io.cpp
    src/orient.cpp
    src/preprocess.cpp
    src/sliding.cpp
    src/weights.cpp
    src/siam_log.cpp
)
target_include_directories(siamize PRIVATE
    src
    ${ORT_ROOT}/include
    ${ZMAT_INCLUDE}
)
target_link_directories(siamize PRIVATE ${ORT_ROOT}/lib)
if(WIN32)
    target_link_libraries(siamize PRIVATE ${ORT_IMPORT_LIB})
else()
    target_link_libraries(siamize PRIVATE onnxruntime)
endif()

# zmat / miniz are C code inside a header; silence a few stylistic warnings
# coming from that block.
if(MSVC)
    target_compile_definitions(siamize PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    target_compile_options(siamize PRIVATE
        /wd4068   # unknown pragma (zmat.h has #pragma GCC blocks)
        /wd4244   # implicit narrowing conversions
        /wd4267   # size_t -> int narrowing
    )
else()
    target_compile_options(siamize PRIVATE -Wall -Wno-deprecated-declarations
        -Wno-unused-function -Wno-unused-variable -Wno-sign-compare
        -Wno-unused-but-set-variable -Wno-misleading-indentation)
endif()


# GPU backend preprocessor defines — drive the runtime EP-try blocks.
# tensorrt implies cuda (TRT EP is registered first, CUDA second as fallback).
if(SIAMIZE_GPU STREQUAL "cuda" OR SIAMIZE_GPU STREQUAL "tensorrt")
    target_compile_definitions(siamize PRIVATE SIAMIZE_HAS_CUDA)
endif()
if(SIAMIZE_GPU STREQUAL "tensorrt")
    target_compile_definitions(siamize PRIVATE SIAMIZE_HAS_TENSORRT)
endif()

# OpenMP linkage
if(OpenMP_CXX_FOUND)
    set(_omp_static_linked FALSE)

    # OpenMP_CXX_FLAGS may be a space-separated string (e.g. on macOS:
    # "-Xclang -fopenmp"). target_compile_options treats each list element as
    # one argv entry, so we must split before adding.
    separate_arguments(_omp_compile_opts UNIX_COMMAND "${OpenMP_CXX_FLAGS}")

    if(SIAMIZE_STATIC_LINK AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Linux/GCC: link libgomp.a directly so the binary doesn't pull in
        # libgomp.so.1 at runtime.
        find_library(GOMP_STATIC_LIB NAMES libgomp.a
            HINTS /usr/lib/gcc/x86_64-linux-gnu/12
                  /usr/lib/gcc/x86_64-linux-gnu/11
                  /usr/lib/gcc/x86_64-linux-gnu/10
                  /usr/lib/gcc/x86_64-linux-gnu/9
        )
        if(GOMP_STATIC_LIB)
            message(STATUS "libgomp static: ${GOMP_STATIC_LIB}")
            target_compile_options(siamize PRIVATE ${_omp_compile_opts})
            # libgomp.a's target.o has unresolved refs to dlopen/dlclose
            # (it dlopens plugin-host runtimes lazily). Modern ld won't
            # add transitive DSO deps automatically, so we list libdl
            # explicitly here. pthread is also a hard dep of libgomp.
            target_link_libraries(siamize PRIVATE ${GOMP_STATIC_LIB} pthread dl)
            set(_omp_static_linked TRUE)
        else()
            message(WARNING "libgomp.a not found; OpenMP will link dynamically.")
        endif()
    elseif(SIAMIZE_STATIC_LINK AND APPLE
           AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # macOS/Apple-Clang: Homebrew's libomp ships both .dylib and .a.
        # Find the .a from $OpenMP_ROOT (set by `brew --prefix libomp` in CI),
        # or fall back to the standard Homebrew install paths for arm64 / x86_64.
        find_library(LIBOMP_STATIC_LIB NAMES libomp.a
            HINTS $ENV{OpenMP_ROOT}/lib
                  /opt/homebrew/opt/libomp/lib
                  /usr/local/opt/libomp/lib
        )
        if(LIBOMP_STATIC_LIB)
            message(STATUS "libomp static: ${LIBOMP_STATIC_LIB}")
            # Apple clang needs `-Xclang -fopenmp` (passed as TWO argv entries)
            # plus libomp headers; CMake's FindOpenMP stores both in
            # OpenMP_CXX_FLAGS as a space-separated string which we split above.
            target_compile_options(siamize PRIVATE ${_omp_compile_opts})
            target_include_directories(siamize PRIVATE
                $ENV{OpenMP_ROOT}/include
                /opt/homebrew/opt/libomp/include
                /usr/local/opt/libomp/include
            )
            target_link_libraries(siamize PRIVATE ${LIBOMP_STATIC_LIB})
            set(_omp_static_linked TRUE)
        else()
            message(WARNING "libomp.a not found; OpenMP will link dynamically. "
                            "Run `brew install libomp` and set OpenMP_ROOT.")
        endif()
    endif()

    if(NOT _omp_static_linked)
        target_link_libraries(siamize PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# C/C++ runtime: static where the toolchain supports it
if(SIAMIZE_STATIC_LINK)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(siamize PRIVATE -static-libstdc++ -static-libgcc)
    elseif(MSVC)
        target_compile_options(siamize PRIVATE
            $<$<CONFIG:Release>:/MT>
            $<$<CONFIG:Debug>:/MTd>
            $<$<CONFIG:RelWithDebInfo>:/MT>
            $<$<CONFIG:MinSizeRel>:/MT>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND APPLE)
        message(STATUS "macOS: relying on system libc++ (no static-stdlib option).")
    endif()
endif()

# ---- RPATH / DLL placement --------------------------------------------------
if(WIN32)
    # Always copy the main ORT DLL next to siamize.exe (no RPATH on Windows).
    add_custom_command(TARGET siamize POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_RUNTIME_DLL} $<TARGET_FILE_DIR:siamize>
        COMMENT "Copying onnxruntime.dll next to siamize.exe")

    # For GPU builds, also copy the provider plugin DLLs so ORT can dlopen them.
    if(SIAMIZE_GPU STREQUAL "cuda" OR SIAMIZE_GPU STREQUAL "tensorrt")
        add_custom_command(TARGET siamize POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_ROOT}/lib/onnxruntime_providers_shared.dll"
                "${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll"
                $<TARGET_FILE_DIR:siamize>
            COMMENT "Copying ORT CUDA provider DLLs next to siamize.exe")
    endif()
    if(SIAMIZE_GPU STREQUAL "tensorrt")
        add_custom_command(TARGET siamize POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_ROOT}/lib/onnxruntime_providers_tensorrt.dll"
                $<TARGET_FILE_DIR:siamize>
            COMMENT "Copying ORT TensorRT provider DLL next to siamize.exe")
    endif()
elseif(APPLE)
    set_target_properties(siamize PROPERTIES
        BUILD_RPATH   "${ORT_ROOT}/lib;@executable_path;@loader_path"
        INSTALL_RPATH "@executable_path"
    )
else()
    set_target_properties(siamize PROPERTIES
        BUILD_RPATH   "${ORT_ROOT}/lib;\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
    )
endif()

# ---- MATLAB MEX target ------------------------------------------------------
set(_siamize_mex_sources
    src/siamize_mex.cpp
    src/nifti_io.cpp
    src/orient.cpp
    src/preprocess.cpp
    src/sliding.cpp
    src/weights.cpp
    src/siam_log.cpp
)
set(_siamize_mex_includes
    src
    ${ORT_ROOT}/include
    ${ZMAT_INCLUDE}
)

if(SIAMIZE_BUILD_MATLAB_MEX)
    find_package(Matlab COMPONENTS MX_LIBRARY)

    if(Matlab_FOUND)
        message(STATUS "MATLAB found at ${Matlab_ROOT_DIR}; building MEX")
        # The MEX binary is the low-level entry point (`siamex.mex*`);
        # the user-facing dispatcher is `matlab/siamize.m`, which handles the
        # weight download (urlread + gunzip) and then calls siamex.
        matlab_add_mex(
            NAME siamize_mex
            SRC ${_siamize_mex_sources}
            OUTPUT_NAME siamex
        )
        # Drop the .mex* next to matlab/siamize.m so the wrapper finds it
        # without a manual addpath. siamize.m's auto-addpath block looks
        # in `here = fileparts(mfilename('fullpath'))` and only adds the
        # path if siamex.<mexext> exists there.
        set_target_properties(siamize_mex PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/matlab"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/matlab"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_SOURCE_DIR}/matlab"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_SOURCE_DIR}/matlab"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/matlab"
            RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_SOURCE_DIR}/matlab"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_SOURCE_DIR}/matlab"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_SOURCE_DIR}/matlab"
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/matlab"
            LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_SOURCE_DIR}/matlab"
        )
        target_include_directories(siamize_mex PRIVATE ${_siamize_mex_includes})
        target_link_directories(siamize_mex PRIVATE ${ORT_LIB_DIR})
        # Propagate SIAMIZE_HAS_CUDA / SIAMIZE_HAS_TENSORRT to the MEX
        # exactly as we do for the CLI -- sliding.cpp's EP probe blocks
        # are gated on these and #ifdef'd out otherwise. Combined with
        # ort-gpu (fetched by `make cudamex`) this gives the MATLAB MEX
        # the same -c cuda / -c tensorrt paths as siamize(1).
        if(SIAMIZE_GPU STREQUAL "cuda" OR SIAMIZE_GPU STREQUAL "tensorrt")
            target_compile_definitions(siamize_mex PRIVATE SIAMIZE_HAS_CUDA)
        endif()
        if(SIAMIZE_GPU STREQUAL "tensorrt")
            target_compile_definitions(siamize_mex PRIVATE SIAMIZE_HAS_TENSORRT)
        endif()
        # FindMatlab.cmake's matlab_add_mex() uses target_link_libraries
        # in the plain (non-keyword) form. CMake forbids mixing plain and
        # keyword signatures on the same target, so we follow suit here.
        if(WIN32)
            target_link_libraries(siamize_mex ${ORT_IMPORT_LIB})
        else()
            target_link_libraries(siamize_mex onnxruntime)
        endif()
        if(OpenMP_CXX_FOUND)
            target_link_libraries(siamize_mex OpenMP::OpenMP_CXX)
        endif()
        # C++ runtime handling for the MEX:
        #   * MSVC: /MT (static CRT). Windows MATLAB ships its own MSVCRT
        #     copies but the C++ ABI doesn't fight the way glibc++ does
        #     on Linux, so static-link is safe and avoids the runtime
        #     redistribute headache.
        #   * GNU on Linux: link libstdc++ DYNAMICALLY. Statically
        #     embedding a second libstdc++ inside a MEX fights MATLAB's
        #     own already-loaded libstdc++ (type_info, vtables,
        #     std::cout) and aborts at MEX-load. MATLAB R2024a's bundled
        #     libstdc++.so.6 may be older than the host toolchain's, in
        #     which case the user / CI must LD_PRELOAD the system
        #     libstdc++ before launching MATLAB:
        #         LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab
        #     The matlab/README.md documents this; CI does it in the
        #     "Smoke-load MEX (MATLAB)" step.
        if(SIAMIZE_STATIC_LINK AND MSVC)
            target_compile_options(siamize_mex PRIVATE
                $<$<CONFIG:Release>:/MT>
                $<$<CONFIG:Debug>:/MTd>
                $<$<CONFIG:RelWithDebInfo>:/MT>
                $<$<CONFIG:MinSizeRel>:/MT>
            )
        endif()
        if(MSVC)
            target_compile_definitions(siamize_mex PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
            target_compile_options(siamize_mex PRIVATE /wd4068 /wd4244 /wd4267)
        else()
            target_compile_options(siamize_mex PRIVATE -Wall
                -Wno-deprecated-declarations -Wno-unused-function
                -Wno-unused-variable -Wno-sign-compare
                -Wno-unused-but-set-variable -Wno-misleading-indentation)
        endif()
        # Make libonnxruntime findable when MATLAB loads the .mex*.
        if(APPLE)
            set_target_properties(siamize_mex PROPERTIES
                BUILD_RPATH   "${ORT_LIB_DIR};@loader_path"
                INSTALL_RPATH "@loader_path"
            )
        elseif(WIN32)
            # Windows: copy onnxruntime.dll next to siamize.mexw64 so the
            # PE loader resolves it at MEX load time.
            add_custom_command(TARGET siamize_mex POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ORT_RUNTIME_DLL} $<TARGET_FILE_DIR:siamize_mex>
                COMMENT "Copying onnxruntime.dll next to siamize.mexw64")
        else()
            set_target_properties(siamize_mex PROPERTIES
                BUILD_RPATH   "${ORT_LIB_DIR};\$ORIGIN"
                INSTALL_RPATH "\$ORIGIN"
            )
        endif()
    else()
        message(STATUS "SIAMIZE_BUILD_MATLAB_MEX=ON but MATLAB not found; skipping")
    endif()
endif()

# ---- GNU Octave MEX target --------------------------------------------------
# Octave doesn't have a CMake target wrapper like FindMatlab, so we invoke
# mkoctfile via a custom command. Output is siamize.mex (Octave's binary mex).
if(SIAMIZE_BUILD_OCTAVE_MEX)
    find_program(MKOCTFILE NAMES mkoctfile)
    if(MKOCTFILE)
        message(STATUS "Octave mkoctfile found: ${MKOCTFILE}")
        # Drop the .mex next to matlab/siamize.m for the same auto-
        # addpath reason as the MATLAB MEX above. matlab/siamize.m
        # locates siamex.<mexext> via `mfilename('fullpath')`.
        set(_oct_out ${CMAKE_SOURCE_DIR}/matlab/siamex.mex)
        set(_oct_sources "")
        foreach(s IN LISTS _siamize_mex_sources)
            list(APPEND _oct_sources ${CMAKE_SOURCE_DIR}/${s})
        endforeach()
        set(_oct_inc "")
        foreach(i IN LISTS _siamize_mex_includes)
            if(IS_ABSOLUTE ${i})
                list(APPEND _oct_inc "-I${i}")
            else()
                list(APPEND _oct_inc "-I${CMAKE_SOURCE_DIR}/${i}")
            endif()
        endforeach()
        # mkoctfile defaults to C++11 on Octave 6.x; ORT 1.26 requires
        # C++17 typedef-noexcept support. Inject -std=c++17 via env.
        # mkoctfile honors LDFLAGS env for link-stage flags; -L/-l args
        # on the command line get dropped silently.
        # Two -rpath entries: the absolute build-tree path (lets the MEX
        # find libonnxruntime when running from the source checkout) plus
        # the literal $ORIGIN token (lets the MEX find a libonnxruntime
        # that lives next to it in a package bundle). $ORIGIN is expanded
        # at runtime by the ELF loader, not by cmake/make.
        #
        # The "\\\$" escape is load-bearing: mkoctfile is a Perl script
        # that runs the link command via `system($cmd)`, which goes
        # through /bin/sh. A bare $ORIGIN in LDFLAGS gets shell-expanded
        # to the empty string before ld ever sees it. The triple-escape
        # makes the CMake string `\$ORIGIN`, which the shell reads as a
        # literal `$ORIGIN` and forwards intact to the linker.
        set(_oct_ld "-L${ORT_LIB_DIR} -lonnxruntime -Wl,-rpath,${ORT_LIB_DIR} -Wl,-rpath,\\\$ORIGIN")
        # Octave loads its MEX into a process that already has a
        # dynamically-loaded libstdc++. Statically embedding a second
        # copy of libstdc++ inside the MEX causes ABI conflicts at
        # load time (std::cout, type_info, exception handling) and
        # SIGABRT during siamex() entry. So we always link libstdc++
        # dynamically for the Octave MEX even with SIAMIZE_STATIC_LINK=ON.
        # (The MATLAB MEX is different -- MATLAB ships an older bundled
        #  libstdc++.so.6 that lacks recent GLIBCXX_x.y.z symbols, so
        #  there static-linking is the only thing that loads.)
        # mkoctfile --mex does NOT auto-define MATLAB_MEX_FILE (only
        # MEX_DEBUG), so we set it explicitly. CMake's matlab_add_mex
        # already does the same for the MATLAB target. siam_log.cpp
        # uses MATLAB_MEX_FILE to route through mexPrintf instead of
        # stderr; without it the Octave MEX would silently print to
        # the launching terminal rather than the command window.
        # Same SIAMIZE_HAS_CUDA / SIAMIZE_HAS_TENSORRT propagation as the
        # MATLAB MEX target above, but via CXXFLAGS since mkoctfile does
        # not honor target_compile_definitions.
        set(_oct_cxxflags "-std=c++17 -O2 -fPIC -DMATLAB_MEX_FILE")
        if(SIAMIZE_GPU STREQUAL "cuda" OR SIAMIZE_GPU STREQUAL "tensorrt")
            set(_oct_cxxflags "${_oct_cxxflags} -DSIAMIZE_HAS_CUDA")
        endif()
        if(SIAMIZE_GPU STREQUAL "tensorrt")
            set(_oct_cxxflags "${_oct_cxxflags} -DSIAMIZE_HAS_TENSORRT")
        endif()

        add_custom_command(
            OUTPUT ${_oct_out}
            COMMAND ${CMAKE_COMMAND} -E env
                "CXXFLAGS=${_oct_cxxflags}"
                "LDFLAGS=${_oct_ld}"
                ${MKOCTFILE} --mex
                ${_oct_inc}
                ${_oct_sources}
                $<$<BOOL:${OpenMP_CXX_FOUND}>:-fopenmp>
                -Wno-unused -Wno-unknown-pragmas
                -o ${_oct_out}
            DEPENDS ${_oct_sources}
            COMMENT "Building Octave MEX via mkoctfile -> ${_oct_out}"
            VERBATIM)
        add_custom_target(siamize_oct ALL DEPENDS ${_oct_out})
    else()
        message(STATUS "SIAMIZE_BUILD_OCTAVE_MEX=ON but mkoctfile not found; skipping")
    endif()
endif()

# ---- summary ----------------------------------------------------------------
message(STATUS "")
message(STATUS "siamize configuration:")
message(STATUS "  platform:         ${SIAMIZE_PLATFORM}")
message(STATUS "  compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  binary:           siamize${CMAKE_EXECUTABLE_SUFFIX}")
message(STATUS "  static link:      ${SIAMIZE_STATIC_LINK}")
message(STATUS "  ORT:              ${ORT_ROOT}")
message(STATUS "  zmat header:      ${ZMAT_INCLUDE}/zmat.h (bundled)")
message(STATUS "  build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP:           ${OpenMP_CXX_FOUND}")
message(STATUS "  GPU backend:      ${SIAMIZE_GPU}")
message(STATUS "  MATLAB MEX:       ${SIAMIZE_BUILD_MATLAB_MEX}")
message(STATUS "  Octave MEX:       ${SIAMIZE_BUILD_OCTAVE_MEX}")
message(STATUS "")
