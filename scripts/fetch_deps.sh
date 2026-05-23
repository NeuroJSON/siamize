#!/usr/bin/env bash
# Fetch C++ build dependencies into third_party/.
#   - ONNX Runtime prebuilt for the host platform (Linux x64/aarch64,
#     macOS x86_64/arm64, Windows x64 via Git Bash).
#
# zmat (the single-header compression library) is bundled directly in this
# repo under src/zmat/zmat.h — no fetch needed.
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.26.0}"
ORT_BUILD="${ORT_BUILD:-cpu}"        # cpu | gpu
ORT_CUDA="${ORT_CUDA:-}"             # blank (default CUDA), or 13 for cuda13 variant

OS_NAME="$(uname -s)"
ARCH="$(uname -m)"
ORT_EXT="tgz"
case "$OS_NAME" in
    Linux)
        case "$ARCH" in
            x86_64)         ORT_ARCH="linux-x64" ;;
            aarch64|arm64)  ORT_ARCH="linux-aarch64" ;;
            *) echo "Unsupported Linux arch: $ARCH" >&2; exit 1 ;;
        esac
        ;;
    Darwin)
        case "$ARCH" in
            arm64)  ORT_ARCH="osx-arm64" ;;
            x86_64) ORT_ARCH="osx-x86_64" ;;
            *) echo "Unsupported macOS arch: $ARCH" >&2; exit 1 ;;
        esac
        ;;
    MINGW*|MSYS*|CYGWIN*)
        # Git Bash / MSYS2 / Cygwin on Windows.
        ORT_ARCH="win-x64"
        ORT_EXT="zip"
        ;;
    *)
        echo "Unsupported OS: $OS_NAME" >&2
        exit 1
        ;;
esac

# Optional GPU build. NVIDIA CUDA prebuilts exist for Linux x64 and Windows x64.
# macOS / aarch64 have no CUDA prebuilts — they fall back to CPU silently.
if [[ "$ORT_BUILD" == "gpu" ]]; then
    case "$ORT_ARCH" in
        linux-x64|win-x64)
            ORT_ARCH="${ORT_ARCH}-gpu"
            if [[ -n "$ORT_CUDA" ]]; then
                ORT_ARCH="${ORT_ARCH}_cuda${ORT_CUDA}"   # e.g. linux-x64-gpu_cuda13
            fi
            ;;
        *)
            echo "[fetch_deps] GPU build not available for $ORT_ARCH; falling back to CPU." >&2
            ;;
    esac
fi

ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_ARCH}-${ORT_VERSION}.${ORT_EXT}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TP="$HERE/third_party"
mkdir -p "$TP"

if [[ -d "$TP/onnxruntime" ]]; then
    echo "[fetch_deps] onnxruntime already present, skipping"
else
    echo "[fetch_deps] downloading ORT v${ORT_VERSION} (${ORT_ARCH}) ..."
    tmp="$(mktemp -d)"
    trap "rm -rf '$tmp'" EXIT
    archive="$tmp/ort.$ORT_EXT"
    curl -fsSL "$ORT_URL" -o "$archive"
    case "$ORT_EXT" in
        tgz)
            tar xzf "$archive" -C "$tmp"
            ;;
        zip)
            # Try multiple unzip strategies — what's available depends on the
            # host (GNU tar can't handle .zip; bsdtar can; unzip and Python's
            # zipfile module are also usually around).
            if command -v unzip >/dev/null 2>&1; then
                unzip -q "$archive" -d "$tmp"
            elif [[ -x /c/Windows/System32/tar.exe ]]; then
                /c/Windows/System32/tar.exe -xf "$archive" -C "$tmp"
            elif command -v 7z >/dev/null 2>&1; then
                7z x "$archive" -o"$tmp" >/dev/null
            elif command -v python3 >/dev/null 2>&1; then
                python3 -m zipfile -e "$archive" "$tmp"
            else
                echo "no unzip / bsdtar / 7z / python3 available to extract $archive" >&2
                exit 1
            fi
            ;;
    esac
    # The extracted top-level directory name doesn't always match ORT_ARCH
    # verbatim (e.g. the gpu_cuda13 tarball extracts into a plain "gpu" dir).
    # Just take whichever onnxruntime-* directory landed in $tmp.
    extracted="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d -name 'onnxruntime-*' | head -1)"
    if [[ -z "$extracted" ]]; then
        echo "[fetch_deps] no onnxruntime-* dir found inside the tarball" >&2
        ls -la "$tmp" >&2
        exit 1
    fi
    mv "$extracted" "$TP/onnxruntime"
    echo "[fetch_deps] ORT installed at $TP/onnxruntime"
fi

echo "[fetch_deps] done. ORT at $TP/onnxruntime"
