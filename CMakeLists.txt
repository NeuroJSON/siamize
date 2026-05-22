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
    src/main.cpp
    src/nifti_io.cpp
    src/preprocess.cpp
    src/sliding.cpp
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
else()
    target_compile_options(siamize PRIVATE -Wall -Wno-deprecated-declarations
        -Wno-unused-function -Wno-unused-variable -Wno-sign-compare
        -Wno-unused-but-set-variable -Wno-misleading-indentation)
endif()

# OpenMP linkage
if(OpenMP_CXX_FOUND)
    if(SIAMIZE_STATIC_LINK AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_library(GOMP_STATIC_LIB NAMES libgomp.a
            HINTS /usr/lib/gcc/x86_64-linux-gnu/12
                  /usr/lib/gcc/x86_64-linux-gnu/11
                  /usr/lib/gcc/x86_64-linux-gnu/10
                  /usr/lib/gcc/x86_64-linux-gnu/9
        )
        if(GOMP_STATIC_LIB)
            message(STATUS "libgomp static: ${GOMP_STATIC_LIB}")
            target_compile_options(siamize PRIVATE ${OpenMP_CXX_FLAGS})
            target_link_libraries(siamize PRIVATE ${GOMP_STATIC_LIB} pthread)
        else()
            message(WARNING "libgomp.a not found; OpenMP will link dynamically.")
            target_link_libraries(siamize PRIVATE OpenMP::OpenMP_CXX)
        endif()
    else()
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
    add_custom_command(TARGET siamize POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_RUNTIME_DLL} $<TARGET_FILE_DIR:siamize>
        COMMENT "Copying onnxruntime.dll next to siamize.exe")
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
message(STATUS "")
