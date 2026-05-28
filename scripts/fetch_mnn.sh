#!/usr/bin/env bash
# Fetch and build the patched MNN (NeuroJSON/MNN v3.5-int64fix) into
# third_party/mnn/ so siamize built with `cmake -DSIAMIZE_BACKEND=mnn`
# can find the include headers and libMNN.so / .dylib / .dll.
#
# Why a patched MNN: stock MNN 3.5.0 silently produces wrong logits on
# SIAM-class workloads (>2 GB intermediate tensors after Conv3DTurn2D).
# The NeuroJSON/MNN fork's `v3.5-int64fix` tag carries a 12-file int64
# audit that fixes the offset arithmetic on the hot paths. See
# tools/mnn_probe/patches/README.md.
#
# This script is heavyweight on first run (~15-20 min for cmake + make
# of the MNN C++ tree). It caches build artifacts under
# third_party/mnn-build/ so reruns skip the configure step. To force a
# clean rebuild: `rm -rf third_party/mnn third_party/mnn-build`.
#
# Environment variables:
#   MNN_TAG       Git tag/branch in NeuroJSON/MNN. Default v3.5-int64fix.
#   MNN_OPENCL    1 (default) to enable MNN's OpenCL backend, 0 to skip.
#   MNN_JOBS      Parallel make jobs. Default $(nproc) on Linux, sysctl
#                 hw.logicalcpu on macOS, else 4.
#   FORCE         1 = remove third_party/mnn and rebuild even if it
#                 already exists. Default 0 (idempotent).

set -euo pipefail

# MNN_REF can be a branch, tag, or commit SHA. Default points to the
# NeuroJSON/MNN branch carrying the int64-overflow fixes needed for
# SIAM-class workloads. Note: the v3.5-int64fix *tag* on that repo
# currently points to a pre-patch commit (sha 59061a4f); the actual
# patched HEAD lives on the siam-int64-fixes branch (sha 3ba7a4d4+).
# `MNN_TAG` is accepted as a synonym for back-compat with the script's
# first revision.
MNN_REF="${MNN_REF:-${MNN_TAG:-siam-int64-fixes}}"
MNN_TAG="$MNN_REF"   # used in stage-dir naming below
MNN_OPENCL="${MNN_OPENCL:-1}"
FORCE="${FORCE:-0}"

# Resolve siamize root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIAMIZE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MNN_STAGE="$SIAMIZE_ROOT/third_party/mnn"
MNN_BUILD="$SIAMIZE_ROOT/third_party/mnn-build"

OS_NAME="$(uname -s)"
case "$OS_NAME" in
    Linux)         LIB_NAME="libMNN.so" ;;
    Darwin)        LIB_NAME="libMNN.dylib" ;;
    MINGW*|MSYS*|CYGWIN*) LIB_NAME="MNN.dll" ;;
    *) echo "[fetch_mnn] Unsupported OS: $OS_NAME" >&2; exit 1 ;;
esac

if [[ -z "${MNN_JOBS:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        MNN_JOBS="$(nproc)"
    elif [[ "$OS_NAME" == "Darwin" ]]; then
        MNN_JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
    else
        MNN_JOBS=4
    fi
fi

# Idempotency: skip if already staged unless FORCE=1.
if [[ "$FORCE" == "0" && -f "$MNN_STAGE/include/MNN/Interpreter.hpp" && -f "$MNN_STAGE/lib/$LIB_NAME" ]]; then
    echo "[fetch_mnn] $MNN_STAGE already populated (set FORCE=1 to rebuild)"
    echo "[fetch_mnn]   include: $MNN_STAGE/include/MNN/Interpreter.hpp"
    echo "[fetch_mnn]   lib:     $MNN_STAGE/lib/$LIB_NAME"
    exit 0
fi

# Wipe the stage on rebuild but keep the build tree around so cmake
# incremental rebuilds work when you bump MNN_TAG.
rm -rf "$MNN_STAGE"
mkdir -p "$MNN_STAGE/include" "$MNN_STAGE/lib"

# Fetch the source if we don't have a clean checkout for this tag.
mkdir -p "$MNN_BUILD"
SRC_DIR="$MNN_BUILD/MNN-$MNN_TAG"
ZIP_PATH="$MNN_BUILD/MNN-$MNN_TAG.zip"

if [[ ! -d "$SRC_DIR" ]]; then
    # GitHub's /archive/<ref>.zip endpoint accepts any ref kind
    # (tag, branch, or commit SHA), unlike /archive/refs/tags/<tag>.zip
    # which only resolves tag names. We rely on this for the default
    # branch fetch and any user-supplied ref form.
    URL="https://github.com/NeuroJSON/MNN/archive/${MNN_REF}.zip"
    echo "[fetch_mnn] fetching $URL"
    # Use curl with -L (follow redirects: GitHub redirects to codeload).
    # -f fails on HTTP error so the script doesn't silently install a
    # 404 page as the source archive.
    curl -fL --retry 3 -o "$ZIP_PATH" "$URL"
    echo "[fetch_mnn] extracting $ZIP_PATH"
    # GitHub zip archives extract to a directory named MNN-<tagslug>/
    # where tagslug is the tag name with v stripped from some auto
    # archives but kept for explicit tags. Detect and rename if needed.
    unzip -q "$ZIP_PATH" -d "$MNN_BUILD"
    if [[ ! -d "$SRC_DIR" ]]; then
        # Try variants GitHub may use: MNN-<tag>, MNN-<tag-with-v-stripped>
        for d in "$MNN_BUILD"/MNN-*/; do
            mv "$d" "$SRC_DIR"
            break
        done
    fi
    if [[ ! -d "$SRC_DIR" ]]; then
        echo "[fetch_mnn] FAIL: extraction did not produce $SRC_DIR" >&2
        ls -la "$MNN_BUILD"/ >&2 || true
        exit 1
    fi
fi

# CMake configure + build the C++ library.
CMAKE_BUILD="$SRC_DIR/build-siamize"
mkdir -p "$CMAKE_BUILD"

CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DMNN_BUILD_SHARED_LIBS=ON
    -DMNN_SEP_BUILD=OFF         # fold every backend into libMNN.so so
                                # siamize links a single library; with
                                # SEP_BUILD=ON the OpenCL backend lives in
                                # a separate libMNN_CL.so that siamize
                                # would need to dlopen explicitly.
    -DMNN_USE_THREAD_POOL=ON
    -DMNN_OPENMP=OFF
    -DMNN_BUILD_CONVERTER=OFF   # we only ship the runtime, not mnnconvert
    -DMNN_BUILD_TOOLS=OFF
    -DMNN_BUILD_TEST=OFF
    -DMNN_BUILD_DEMO=OFF
    -DMNN_BUILD_BENCHMARK=OFF
    -DMNN_BUILD_TRAIN=OFF
    -DMNN_BUILD_QUANTOOLS=OFF
)
if [[ "$MNN_OPENCL" == "1" ]]; then
    CMAKE_FLAGS+=(-DMNN_OPENCL=ON)
fi

echo "[fetch_mnn] cmake configure (tag=$MNN_TAG, opencl=$MNN_OPENCL)"
(cd "$CMAKE_BUILD" && cmake "${CMAKE_FLAGS[@]}" "$SRC_DIR")

echo "[fetch_mnn] cmake build -j$MNN_JOBS (this is the slow step, ~15-20 min)"
(cd "$CMAKE_BUILD" && make -j"$MNN_JOBS" MNN)

# Stage headers + library at third_party/mnn/.
echo "[fetch_mnn] staging headers"
cp -r "$SRC_DIR/include/MNN" "$MNN_STAGE/include/"

echo "[fetch_mnn] staging $LIB_NAME"
case "$LIB_NAME" in
    libMNN.so)
        # The build may produce libMNN.so or libMNN.so.<version>; copy
        # whatever exists and symlink the canonical name.
        if [[ -e "$CMAKE_BUILD/libMNN.so" ]]; then
            cp "$CMAKE_BUILD/libMNN.so" "$MNN_STAGE/lib/"
        else
            so=$(ls "$CMAKE_BUILD"/libMNN.so* 2>/dev/null | head -1)
            cp "$so" "$MNN_STAGE/lib/libMNN.so"
        fi
        ;;
    libMNN.dylib)
        cp "$CMAKE_BUILD/libMNN.dylib" "$MNN_STAGE/lib/"
        ;;
    MNN.dll)
        cp "$CMAKE_BUILD/MNN.dll" "$MNN_STAGE/lib/"
        # Windows also needs the import lib.
        if [[ -f "$CMAKE_BUILD/MNN.lib" ]]; then
            cp "$CMAKE_BUILD/MNN.lib" "$MNN_STAGE/lib/"
        fi
        ;;
esac

echo ""
echo "[fetch_mnn] done"
echo "[fetch_mnn]   include:  $MNN_STAGE/include/MNN/Interpreter.hpp"
echo "[fetch_mnn]   library:  $MNN_STAGE/lib/$LIB_NAME"
echo "[fetch_mnn]   tag:      $MNN_TAG"
echo ""
echo "Next:"
echo "  cmake -S . -B build-mnn -DSIAMIZE_BACKEND=mnn"
echo "  cmake --build build-mnn -j"
echo "  ./build-mnn/siamize -i tests/sub-01_T1w.nii.gz -M 0 -c cpu --device cpu"
