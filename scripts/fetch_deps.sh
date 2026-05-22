#!/usr/bin/env bash
# Fetch C++ build dependencies into third_party/.
#   - ONNX Runtime prebuilt for the host platform (Linux x64/aarch64,
#     macOS x86_64/arm64, Windows x64 via Git Bash).
#   - nifti_clib (git clone)
# Re-runnable; skips already-present trees.
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.26.0}"
NIFTI_REPO="${NIFTI_REPO:-https://github.com/NIFTI-Imaging/nifti_clib.git}"

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
    mv "$tmp/onnxruntime-${ORT_ARCH}-${ORT_VERSION}" "$TP/onnxruntime"
    echo "[fetch_deps] ORT installed at $TP/onnxruntime"
fi

if [[ -d "$TP/nifti_clib" ]]; then
    echo "[fetch_deps] nifti_clib already present, skipping"
else
    echo "[fetch_deps] cloning nifti_clib ..."
    git clone --depth 1 "$NIFTI_REPO" "$TP/nifti_clib"
fi

echo "[fetch_deps] done."
echo "  ORT:    $TP/onnxruntime"
echo "  NIfTI:  $TP/nifti_clib"
