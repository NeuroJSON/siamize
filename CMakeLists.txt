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

# Inference backend selection (build-time, exclusive). "ort" is the
# default and links ONNX Runtime; "mnn" links MNN (with the int64
# patches from tools/mnn_probe/patches/ for SIAM-class workloads).
# Both produce the same `siamize` binary -- the user picks one at
# build time depending on which runtime / hardware target they want.
set(SIAMIZE_BACKEND "ort" CACHE STRING
    "Inference backend: ort (ONNX Runtime, default) or mnn (Alibaba MNN). \
Exactly one is linked per build. ORT supports CPU + CUDA + TensorRT + \
CoreML; MNN supports CPU + OpenCL + Vulkan + Metal. See README for \
the per-backend trade-offs.")
set_property(CACHE SIAMIZE_BACKEND PROPERTY STRINGS ort mnn)
if(NOT (SIAMIZE_BACKEND STREQUAL "ort" OR SIAMIZE_BACKEND STREQUAL "mnn"))
    message(FATAL_ERROR
        "SIAMIZE_BACKEND must be one of: ort, mnn (got '${SIAMIZE_BACKEND}')")
endif()

# GPU backend selection. Only meaningful when SIAMIZE_BACKEND=ort.
# "none" = CPU only (the smallest deployable bundle).
# "cuda" = enable the ONNX Runtime CUDA Execution Provider; requires the
# onnxruntime-gpu prebuilt (fetched via ORT_BUILD=gpu scripts/fetch_deps.sh)
# and a runtime CUDA + cuDNN install. At runtime the binary tries CUDA first
# and falls back to CPU if libcudart / libcudnn cannot be loaded.
set(SIAMIZE_GPU "none" CACHE STRING
    "Hardware-acceleration backend (ORT-only): none, cuda, tensorrt, coreml. \
Ignored when SIAMIZE_BACKEND=mnn (MNN's device is picked at runtime via \
--device). 'cuda' uses the ORT CUDA EP (NVIDIA GPUs). 'tensorrt' layers \
the ORT TensorRT EP on top of CUDA. 'coreml' uses the ORT CoreML EP \
on Apple Silicon (CPU + Metal GPU + Neural Engine, selectable via \
--coreml-units). 'coreml' is macOS-only.")
set_property(CACHE SIAMIZE_GPU PROPERTY STRINGS none cuda tensorrt coreml)
if(SIAMIZE_GPU STREQUAL "coreml" AND NOT APPLE)
    message(FATAL_ERROR
        "SIAMIZE_GPU=coreml is macOS-only (CoreML EP needs the CoreML.framework).")
endif()
if(SIAMIZE_BACKEND STREQUAL "mnn" AND NOT SIAMIZE_GPU STREQUAL "none")
    message(WARNING
        "SIAMIZE_GPU=${SIAMIZE_GPU} is ignored when SIAMIZE_BACKEND=mnn. "
        "Pass --device=opencl|vulkan|metal at runtime instead.")
endif()

# ---- platform detection -----------------------------------------------------
if(WIN32)
    set(SIAMIZE_PLATFORM "win")
elseif(APPLE)
    set(SIAMIZE_PLATFORM "mac")
else()
    set(SIAMIZE_PLATFORM "linux")
endif()

# ---- ORT (prebuilt) under third_party/onnxruntime/ --------------------------
# Only required when SIAMIZE_BACKEND=ort. Skipped entirely when building
# the MNN backend so the user doesn't need an ORT install at all.
set(ORT_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
set(ORT_LIB_DIR ${ORT_ROOT}/lib)
if(SIAMIZE_BACKEND STREQUAL "ort" AND NOT EXISTS ${ORT_ROOT}/include/onnxruntime_cxx_api.h)
    message(FATAL_ERROR
        "ORT not found at ${ORT_ROOT}. Run scripts/fetch_deps.sh (Linux/macOS) "
        "or scripts/fetch_deps.ps1 (Windows) first.")
endif()

if(SIAMIZE_BACKEND STREQUAL "ort")
    if(WIN32)
        set(ORT_RUNTIME_DLL ${ORT_ROOT}/lib/onnxruntime.dll)
        set(ORT_IMPORT_LIB  ${ORT_ROOT}/lib/onnxruntime.lib)
    elseif(APPLE)
        set(ORT_RUNTIME_DLL ${ORT_ROOT}/lib/libonnxruntime.dylib)
    else()
        set(ORT_RUNTIME_DLL ${ORT_ROOT}/lib/libonnxruntime.so.1)
    endif()
endif()

# ---- MNN (prebuilt under third_party/mnn/) ---------------------------------
# Required when SIAMIZE_BACKEND=mnn. The expected layout is:
#   third_party/mnn/include/MNN/Interpreter.hpp
#   third_party/mnn/lib/libMNN.so          (Linux)
#                       libMNN.dylib       (macOS)
#                       MNN.dll / MNN.lib  (Windows)
# Build MNN from the patched source at
# tools/mnn_probe/patches/mnn-int64-fixes-for-siam.patch -- stock MNN
# 3.5.0 silently produces wrong logits on SIAM-class workloads.
set(MNN_ROOT ${CMAKE_SOURCE_DIR}/third_party/mnn)
set(MNN_LIB_DIR ${MNN_ROOT}/lib)
if(SIAMIZE_BACKEND STREQUAL "mnn")
    if(NOT EXISTS ${MNN_ROOT}/include/MNN/Interpreter.hpp)
        message(FATAL_ERROR
            "MNN not found at ${MNN_ROOT}/include/MNN/Interpreter.hpp.\n"
            "Apply tools/mnn_probe/patches/mnn-int64-fixes-for-siam.patch to a \n"
            "MNN checkout, build it with -DMNN_OPENCL=ON -DMNN_BUILD_CONVERTER=ON, \n"
            "and stage include/ + lib/libMNN.so under ${MNN_ROOT}/.")
    endif()
    if(WIN32)
        set(MNN_RUNTIME_DLL ${MNN_LIB_DIR}/MNN.dll)
        set(MNN_IMPORT_LIB  ${MNN_LIB_DIR}/MNN.lib)
    elseif(APPLE)
        set(MNN_RUNTIME_DLL ${MNN_LIB_DIR}/libMNN.dylib)
    else()
        set(MNN_RUNTIME_DLL ${MNN_LIB_DIR}/libMNN.so)
    endif()
endif()

# If a GPU backend was requested, sanity-check that the fetched ORT actually
# contains the right provider plugin(s). The CPU-only tarball does not ship
# them; the GPU tarball ships both CUDA and TensorRT plugin shared libs.
if(SIAMIZE_BACKEND STREQUAL "ort"
   AND (SIAMIZE_GPU STREQUAL "cuda" OR SIAMIZE_GPU STREQUAL "tensorrt"))
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
# Build-time backend selection. The backend-specific implementation file
# (engine_mnn.cpp at minimum; an engine_ort.cpp refactor of sliding.cpp's
# inner ORT calls is the natural follow-up) is appended to the source list
# based on SIAMIZE_BACKEND.
set(SIAMIZE_SOURCES
    src/siamize.cpp
    src/nifti_io.cpp
    src/jnifti_io.cpp
    src/orient.cpp
    src/preprocess.cpp
    src/sliding.cpp
    src/weights.cpp
    src/siam_log.cpp
)
if(SIAMIZE_BACKEND STREQUAL "ort")
    list(APPEND SIAMIZE_SOURCES src/engine_ort.cpp)
elseif(SIAMIZE_BACKEND STREQUAL "mnn")
    list(APPEND SIAMIZE_SOURCES src/engine_mnn.cpp)
endif()

add_executable(siamize ${SIAMIZE_SOURCES})

if(SIAMIZE_BACKEND STREQUAL "ort")
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
    target_compile_definitions(siamize PRIVATE SIAMIZE_HAS_ORT)
elseif(SIAMIZE_BACKEND STREQUAL "mnn")
    target_include_directories(siamize PRIVATE
        src
        ${MNN_ROOT}/include
        ${ZMAT_INCLUDE}
    )
    target_link_directories(siamize PRIVATE ${MNN_LIB_DIR})
    if(WIN32)
        target_link_libraries(siamize PRIVATE ${MNN_IMPORT_LIB})
    else()
        target_link_libraries(siamize PRIVATE MNN)
    endif()
    target_compile_definitions(siamize PRIVATE SIAMIZE_HAS_MNN)
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


# GPU backend preprocessor defines — drive sliding.cpp's runtime EP-try
# blocks. These are ORT-EP-specific and inert for the MNN backend.
# tensorrt implies cuda (TRT EP is registered first, CUDA second as fallback).
if(SIAMIZE_BACKEND STREQUAL "ort")
    if(SIAMIZE_GPU STREQUAL "cuda" OR SIAMIZE_GPU STREQUAL "tensorrt")
        target_compile_definitions(siamize PRIVATE SIAMIZE_HAS_CUDA)
    endif()
    if(SIAMIZE_GPU STREQUAL "tensorrt")
        target_compile_definitions(siamize PRIVATE SIAMIZE_HAS_TENSORRT)
    endif()
    # CoreML EP is statically baked into ORT's macOS dylib (no separate
    # provider plugin .so to fetch). We only need the system frameworks
    # CoreML uses underneath at runtime, plus the SIAMIZE_HAS_COREML define
    # that gates sliding.cpp's CoreML probe block.
    if(SIAMIZE_GPU STREQUAL "coreml")
        target_compile_definitions(siamize PRIVATE SIAMIZE_HAS_COREML)
        target_link_libraries(siamize PRIVATE
            "-framework CoreML"
            "-framework Foundation")
    endif()
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
if(SIAMIZE_BACKEND STREQUAL "ort")
    set(_runtime_dll ${ORT_RUNTIME_DLL})
    set(_lib_dir ${ORT_ROOT}/lib)
else()
    set(_runtime_dll ${MNN_RUNTIME_DLL})
    set(_lib_dir ${MNN_LIB_DIR})
endif()

if(WIN32)
    # Always copy the main runtime DLL next to siamize.exe (no RPATH on Windows).
    add_custom_command(TARGET siamize POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_runtime_dll} $<TARGET_FILE_DIR:siamize>
        COMMENT "Copying ${SIAMIZE_BACKEND} runtime DLL next to siamize.exe")

    # For ORT GPU builds, also copy the provider plugin DLLs so ORT can dlopen them.
    if(SIAMIZE_BACKEND STREQUAL "ort"
       AND (SIAMIZE_GPU STREQUAL "cuda" OR SIAMIZE_GPU STREQUAL "tensorrt"))
        add_custom_command(TARGET siamize POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_ROOT}/lib/onnxruntime_providers_shared.dll"
                "${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll"
                $<TARGET_FILE_DIR:siamize>
            COMMENT "Copying ORT CUDA provider DLLs next to siamize.exe")
    endif()
    if(SIAMIZE_BACKEND STREQUAL "ort" AND SIAMIZE_GPU STREQUAL "tensorrt")
        add_custom_command(TARGET siamize POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_ROOT}/lib/onnxruntime_providers_tensorrt.dll"
                $<TARGET_FILE_DIR:siamize>
            COMMENT "Copying ORT TensorRT provider DLL next to siamize.exe")
    endif()
elseif(APPLE)
    set_target_properties(siamize PROPERTIES
        BUILD_RPATH   "${_lib_dir};@executable_path;@loader_path"
        INSTALL_RPATH "@executable_path"
    )
else()
    set_target_properties(siamize PROPERTIES
        BUILD_RPATH   "${_lib_dir};\$ORIGIN"
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
if(SIAMIZE_BACKEND STREQUAL "ort")
    list(APPEND _siamize_mex_sources src/engine_ort.cpp)
elseif(SIAMIZE_BACKEND STREQUAL "mnn")
    list(APPEND _siamize_mex_sources src/engine_mnn.cpp)
endif()
set(_siamize_mex_includes
    src
    ${ORT_ROOT}/include
    ${ZMAT_INCLUDE}
)

# MEX targets currently support the ORT backend only. MEX wiring for
# MNN (different include dir, libMNN linkage, no provider-plugin DLLs)
# is a follow-up; for now we skip MEX builds when SIAMIZE_BACKEND=mnn
# rather than fail at compile time with a confusing missing-header error.
if(SIAMIZE_BUILD_MATLAB_MEX AND SIAMIZE_BACKEND STREQUAL "mnn")
    message(STATUS "MATLAB MEX skipped: SIAMIZE_BACKEND=mnn (MNN-MEX wiring is TODO)")
endif()
if(SIAMIZE_BUILD_OCTAVE_MEX AND SIAMIZE_BACKEND STREQUAL "mnn")
    message(STATUS "Octave MEX skipped: SIAMIZE_BACKEND=mnn (MNN-MEX wiring is TODO)")
endif()

if(SIAMIZE_BUILD_MATLAB_MEX AND SIAMIZE_BACKEND STREQUAL "ort")
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
        if(SIAMIZE_GPU STREQUAL "coreml")
            target_compile_definitions(siamize_mex PRIVATE SIAMIZE_HAS_COREML)
            target_link_libraries(siamize_mex
                "-framework CoreML"
                "-framework Foundation")
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
if(SIAMIZE_BUILD_OCTAVE_MEX AND SIAMIZE_BACKEND STREQUAL "ort")
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
        # CoreML: add the define so sliding.cpp's CoreML probe compiles
        # in, and append the macOS framework link flags to LDFLAGS so
        # mkoctfile passes them to clang.
        if(SIAMIZE_GPU STREQUAL "coreml")
            set(_oct_cxxflags "${_oct_cxxflags} -DSIAMIZE_HAS_COREML")
            set(_oct_ld "${_oct_ld} -framework CoreML -framework Foundation")
        endif()

        # OpenMP flag is compiler-dependent. Linux GCC accepts plain
        # `-fopenmp`; AppleClang rejects it and wants `-Xclang -fopenmp`
        # plus a dynamic -lomp link (Homebrew libomp). CMake's
        # FindOpenMP populates OpenMP_CXX_FLAGS with the right
        # invocation per compiler, so we inject that into CXXFLAGS
        # (env var read by mkoctfile) rather than hard-coding.
        #
        # Linking is platform-different too: on Linux we let mkoctfile
        # add the dynamic libgomp.so.1 via `-fopenmp` -- libgomp.a's
        # TLS relocations aren't PIC-compatible for shared libs (the
        # .mex is a shared lib), so we CAN'T static-link it the way
        # the CLI does. On macOS we link the static libomp.a from
        # Homebrew if we found it (no system libomp present).
        if(OpenMP_CXX_FOUND)
            if(APPLE)
                # AppleClang: needs -Xclang -fopenmp via CXXFLAGS plus
                # a static libomp.a link.
                set(_oct_cxxflags "${_oct_cxxflags} ${OpenMP_CXX_FLAGS}")

                if(LIBOMP_STATIC_LIB)
                    set(_oct_ld "${_oct_ld} ${LIBOMP_STATIC_LIB}")
                endif()
            else()
                # Linux GCC: -fopenmp on the compile+link line. mkoctfile
                # also pulls in dynamic libgomp.so.1 transitively from
                # this flag, which works for shared-library output where
                # libgomp.a's TLS relocs would not.
                list(APPEND _oct_extra_flags "-fopenmp")
            endif()
        endif()

        add_custom_command(
            OUTPUT ${_oct_out}
            COMMAND ${CMAKE_COMMAND} -E env
                "CXXFLAGS=${_oct_cxxflags}"
                "LDFLAGS=${_oct_ld}"
                ${MKOCTFILE} --mex
                ${_oct_inc}
                ${_oct_sources}
                ${_oct_extra_flags}
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
message(STATUS "  backend:          ${SIAMIZE_BACKEND}")
if(SIAMIZE_BACKEND STREQUAL "ort")
    message(STATUS "  ORT:              ${ORT_ROOT}")
    message(STATUS "  GPU backend:      ${SIAMIZE_GPU}")
else()
    message(STATUS "  MNN:              ${MNN_ROOT}")
    message(STATUS "  GPU backend:      runtime --device (cpu / opencl / vulkan / metal)")
endif()
message(STATUS "  zmat header:      ${ZMAT_INCLUDE}/zmat.h (bundled)")
message(STATUS "  build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP:           ${OpenMP_CXX_FOUND}")
message(STATUS "  MATLAB MEX:       ${SIAMIZE_BUILD_MATLAB_MEX}")
message(STATUS "  Octave MEX:       ${SIAMIZE_BUILD_OCTAVE_MEX}")
message(STATUS "")
