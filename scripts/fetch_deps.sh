#!/usr/bin/env bash
# Fetch C++ build dependencies into third_party/.
#   - ONNX Runtime prebuilt for the host platform (Linux x64/aarch64,
#     macOS arm64, Windows x64 via Git Bash), from its NuGet packages.
#
# zmat (the single-header compression library) is bundled directly in this
# repo under src/zmat/zmat.h — no fetch needed.
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.26.0}"
ORT_BUILD="${ORT_BUILD:-cpu}"        # cpu | gpu
ORT_CUDA="${ORT_CUDA:-}"             # must stay blank: the NuGet GPU packages are CUDA 12

# Source: Microsoft's official ONNX Runtime NuGet packages (nuget.org). The
# GitHub release tarballs this script used to download are gone
# (github.com/microsoft/onnxruntime returns 404 since 2026-11). A .nupkg is a
# zip holding the headers (build[Transitive]/native/include) and the native
# libraries (runtimes/<rid>/native); they are rearranged into the old
# tarball layout, third_party/onnxruntime/{include,lib}, so CMake is unchanged.
#   CPU:              Microsoft.ML.OnnxRuntime           (linux-x64/arm64, osx-arm64, win-x64)
#   GPU (CUDA 12):    Microsoft.ML.OnnxRuntime.Gpu.Linux / .Gpu.Windows
NUGET_BASE="${NUGET_BASE:-https://api.nuget.org/v3-flatcontainer}"

OS_NAME="$(uname -s)"
ARCH="$(uname -m)"
case "$OS_NAME" in
    Linux)
        case "$ARCH" in
            x86_64)         ORT_RID="linux-x64" ;;
            aarch64|arm64)  ORT_RID="linux-arm64" ;;
            *) echo "Unsupported Linux arch: $ARCH" >&2; exit 1 ;;
        esac
        ;;
    Darwin)
        case "$ARCH" in
            arm64)  ORT_RID="osx-arm64" ;;
            *) echo "Unsupported macOS arch: $ARCH (the ORT NuGet package ships osx-arm64 only)" >&2; exit 1 ;;
        esac
        ;;
    MINGW*|MSYS*|CYGWIN*)
        # Git Bash / MSYS2 / Cygwin on Windows.
        ORT_RID="win-x64"
        ;;
    *)
        echo "Unsupported OS: $OS_NAME" >&2
        exit 1
        ;;
esac

# Optional GPU build: CUDA packages exist for Linux x64 and Windows x64.
# macOS / aarch64 have none -- they fall back to CPU.
ORT_PKG="microsoft.ml.onnxruntime"
if [[ "$ORT_BUILD" == "gpu" ]]; then
    if [[ -n "$ORT_CUDA" ]]; then
        echo "[fetch_deps] ORT_CUDA=$ORT_CUDA: the NuGet GPU packages are CUDA 12 only" >&2
        exit 1
    fi
    case "$ORT_RID" in
        linux-x64) ORT_PKG="microsoft.ml.onnxruntime.gpu.linux" ;;
        win-x64)   ORT_PKG="microsoft.ml.onnxruntime.gpu.windows" ;;
        *)
            echo "[fetch_deps] GPU build not available for $ORT_RID; falling back to CPU." >&2
            ;;
    esac
fi

ORT_URL="${NUGET_BASE}/${ORT_PKG}/${ORT_VERSION}/${ORT_PKG}.${ORT_VERSION}.nupkg"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TP="$HERE/third_party"
mkdir -p "$TP"

# unzip_to <archive> <dir>: whatever extractor the host has (GNU tar can't
# read .zip; bsdtar, unzip, 7z and Python's zipfile can)
unzip_to() {
    if command -v unzip >/dev/null 2>&1; then
        unzip -q "$1" -d "$2"
    elif [[ -x /c/Windows/System32/tar.exe ]]; then
        /c/Windows/System32/tar.exe -xf "$1" -C "$2"
    elif command -v 7z >/dev/null 2>&1; then
        7z x "$1" -o"$2" >/dev/null
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m zipfile -e "$1" "$2"
    else
        echo "no unzip / bsdtar / 7z / python3 available to extract $1" >&2
        exit 1
    fi
}

if [[ -d "$TP/onnxruntime" ]]; then
    echo "[fetch_deps] onnxruntime already present, skipping"
else
    echo "[fetch_deps] downloading ORT v${ORT_VERSION} (${ORT_PKG}, ${ORT_RID}) ..."
    tmp="$(mktemp -d)"
    trap "rm -rf '$tmp'" EXIT
    curl -fsSL "$ORT_URL" -o "$tmp/ort.nupkg"
    mkdir -p "$tmp/pkg"
    unzip_to "$tmp/ort.nupkg" "$tmp/pkg"

    out="$tmp/onnxruntime-${ORT_VERSION}"
    mkdir -p "$out/include" "$out/lib"
    inc=""
    for d in "$tmp/pkg/build/native/include" "$tmp/pkg/buildTransitive/native/include"; do
        if [[ -f "$d/onnxruntime_cxx_api.h" ]]; then
            inc="$d"
            break
        fi
    done
    native="$tmp/pkg/runtimes/$ORT_RID/native"
    if [[ -z "$inc" || ! -d "$native" ]]; then
        echo "[fetch_deps] unexpected package layout in $ORT_PKG $ORT_VERSION (no headers or runtimes/$ORT_RID)" >&2
        ls -R "$tmp/pkg" | head -40 >&2
        exit 1
    fi
    cp "$inc"/*.h "$out/include/"
    cp "$native"/* "$out/lib/"

    # The names the tarballs had and the loaders look for: the Linux library's
    # SONAME is libonnxruntime.so.1, the macOS one's install name is
    # @rpath/libonnxruntime.1.dylib.
    case "$ORT_RID" in
        linux-*)
            mv "$out/lib/libonnxruntime.so" "$out/lib/libonnxruntime.so.${ORT_VERSION}"
            ln -s "libonnxruntime.so.${ORT_VERSION}" "$out/lib/libonnxruntime.so.1"
            ln -s "libonnxruntime.so.1" "$out/lib/libonnxruntime.so"
            ;;
        osx-*)
            ln -s "libonnxruntime.dylib" "$out/lib/libonnxruntime.1.dylib"
            ;;
    esac
    mv "$out" "$TP/onnxruntime"
    echo "[fetch_deps] ORT installed at $TP/onnxruntime"
fi

echo "[fetch_deps] done. ORT at $TP/onnxruntime"
