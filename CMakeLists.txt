cmake_minimum_required(VERSION 3.16)
project(siamize LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- options ----------------------------------------------------------------
option(SIAMIZE_STATIC_LINK
    "Statically link the C/C++ runtime / OpenMP / zlib into the binary so the \
only non-system shared lib at runtime is libonnxruntime." ON)

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

# Find the runtime library file for ORT (the per-platform extension differs).
if(WIN32)
    set(ORT_LIB_DIR ${ORT_ROOT}/lib)
    set(ORT_RUNTIME_DLL ${ORT_LIB_DIR}/onnxruntime.dll)
    set(ORT_IMPORT_LIB  ${ORT_LIB_DIR}/onnxruntime.lib)
elseif(APPLE)
    set(ORT_LIB_DIR ${ORT_ROOT}/lib)
    set(ORT_RUNTIME_DLL ${ORT_LIB_DIR}/libonnxruntime.dylib)
else()
    set(ORT_LIB_DIR ${ORT_ROOT}/lib)
    set(ORT_RUNTIME_DLL ${ORT_LIB_DIR}/libonnxruntime.so.1)
endif()

# ---- nifti_clib (vendored sources) ------------------------------------------
set(NIFTI_ROOT ${CMAKE_SOURCE_DIR}/third_party/nifti_clib)
if(NOT EXISTS ${NIFTI_ROOT}/niftilib/nifti1_io.c)
    message(FATAL_ERROR "nifti_clib not found at ${NIFTI_ROOT}.")
endif()

# zlib resolution strategy, in order of preference:
#   1. POSIX + static-link build: link the explicit libz.a from the host.
#   2. Otherwise: find_package(ZLIB) and link the system zlib.
#   3. If find_package fails (typical on Windows GHA runners where no zlib
#      is preinstalled): FetchContent the upstream zlib v1.3.1 source and
#      build it as a static target. Adds ~30 s to the first configure.
set(_zlib_static_found FALSE)
if(SIAMIZE_STATIC_LINK AND NOT WIN32)
    find_library(ZLIB_STATIC_LIB NAMES libz.a)
    if(ZLIB_STATIC_LIB)
        message(STATUS "zlib static: ${ZLIB_STATIC_LIB}")
        set(_zlib_static_found TRUE)
    endif()
endif()
if(NOT _zlib_static_found)
    find_package(ZLIB QUIET)
    if(NOT ZLIB_FOUND)
        message(STATUS "System zlib not found; fetching upstream v1.3.1 via FetchContent.")
        include(FetchContent)
        # Suppress zlib's own CMake noise.
        set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(zlib_src
            URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
            URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(zlib_src)
        # zlib's CMakeLists creates both `zlib` (shared) and `zlibstatic`
        # (static). Wire both names to ZLIB::ZLIB so downstream targets
        # below find the static one through the canonical alias.
        if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
            add_library(ZLIB::ZLIB ALIAS zlibstatic)
        endif()
        # ZLIB_INCLUDE_DIR needs to be set for users of FindZLIB.cmake idioms.
        get_target_property(_zlib_inc zlibstatic INCLUDE_DIRECTORIES)
    endif()
endif()

add_library(nifti_static STATIC
    ${NIFTI_ROOT}/znzlib/znzlib.c
    ${NIFTI_ROOT}/niftilib/nifti1_io.c
)
target_include_directories(nifti_static PUBLIC
    ${NIFTI_ROOT}/znzlib
    ${NIFTI_ROOT}/niftilib
)
target_compile_definitions(nifti_static PUBLIC HAVE_ZLIB)
if(_zlib_static_found)
    target_link_libraries(nifti_static PUBLIC ${ZLIB_STATIC_LIB})
else()
    target_link_libraries(nifti_static PUBLIC ZLIB::ZLIB)
endif()
set_target_properties(nifti_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_definitions(nifti_static PUBLIC _CRT_SECURE_NO_WARNINGS)
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
)
target_link_directories(siamize PRIVATE ${ORT_LIB_DIR})

if(WIN32)
    target_link_libraries(siamize PRIVATE nifti_static ${ORT_IMPORT_LIB})
else()
    target_link_libraries(siamize PRIVATE nifti_static onnxruntime)
endif()

# OpenMP linkage
if(OpenMP_CXX_FOUND)
    if(SIAMIZE_STATIC_LINK AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Static libgomp.a from the GCC install.
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
        # MSVC: static CRT (/MT in Release, /MTd in Debug).
        target_compile_options(siamize PRIVATE
            $<$<CONFIG:Release>:/MT>
            $<$<CONFIG:Debug>:/MTd>
            $<$<CONFIG:RelWithDebInfo>:/MT>
            $<$<CONFIG:MinSizeRel>:/MT>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND APPLE)
        # Apple clang does not support -static-libstdc++. libc++ ships with
        # macOS and has an ABI guarantee, so dynamic linking is the norm.
        message(STATUS "macOS: relying on system libc++ (no static-stdlib option).")
    endif()
endif()

# Common warnings
if(MSVC)
    target_compile_options(siamize PRIVATE /W3 /wd4244 /wd4267)
    target_compile_definitions(siamize PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(siamize PRIVATE -Wall -Wno-deprecated-declarations)
endif()

# ---- RPATH / DLL placement --------------------------------------------------
if(WIN32)
    # Windows has no RPATH; copy onnxruntime.dll next to siamize.exe at build
    # time so the binary can find it.
    add_custom_command(TARGET siamize POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_RUNTIME_DLL} $<TARGET_FILE_DIR:siamize>
        COMMENT "Copying onnxruntime.dll next to siamize.exe")
elseif(APPLE)
    # macOS: look in $ORIGIN equivalents.
    set_target_properties(siamize PROPERTIES
        BUILD_RPATH   "${ORT_LIB_DIR};@executable_path;@loader_path"
        INSTALL_RPATH "@executable_path"
    )
else()
    # Linux: $ORIGIN means "directory of the binary at runtime".
    set_target_properties(siamize PROPERTIES
        BUILD_RPATH   "${ORT_LIB_DIR};\$ORIGIN"
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
message(STATUS "  ORT runtime:      ${ORT_RUNTIME_DLL}")
message(STATUS "  nifti_clib:       ${NIFTI_ROOT}")
message(STATUS "  build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP:           ${OpenMP_CXX_FOUND}")
message(STATUS "")
