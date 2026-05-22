#!/usr/bin/env bash
# Fetch siamize's fp16 ONNX folds into models/.
#
# This script expects the .onnx files to be published as a GitHub Release on
# the siamize repo. Until that release is cut, you can produce them locally
# from the upstream SIAM weights with:
#
#   bash tools/onnx_export/export_all_folds.sh
#
# which assumes the original SIAM model dir at
# $HOME/siam_params/v0.3/pred_DS108_LcsfP_Ano (the default `siam-pred` cache).
set -euo pipefail

# SIAMIZE_RELEASE_BASE: base URL hosting the fp16 ONNX folds.
#   Each fold is expected at  <base>/fold_<N>_fp16.onnx
# SIAMIZE_FOLDS: comma-separated list of folds to fetch (default 0,1,2,3,4).
#   Set SIAMIZE_FOLDS=0 to grab just fold 0 (CI smoke tests).
RELEASE_BASE="${SIAMIZE_RELEASE_BASE:-}"
FOLDS="${SIAMIZE_FOLDS:-0,1,2,3,4}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$HERE/models"
mkdir -p "$MODELS"

if [[ -z "$RELEASE_BASE" ]]; then
    cat <<EOF
[fetch_weights] No release URL configured yet.

To re-export the .onnx files locally from the upstream SIAM weights, run:

    bash tools/onnx_export/export_all_folds.sh

That requires PyTorch + dynamic_network_architectures + the SIAM model dir
(\$HOME/siam_params/v0.3/pred_DS108_LcsfP_Ano by default).

Once a release is published, re-run this script with:

    SIAMIZE_RELEASE_BASE=https://github.com/.../releases/download/v0.X.Y \\
        bash scripts/fetch_weights.sh

EOF
    exit 1
fi

# Try compressed (.onnx.7z) first since it's ~3x smaller; fall back to raw.
extract_7z() {
    if command -v 7z >/dev/null 2>&1; then
        7z x -y "$1" -o"$2" >/dev/null
    elif command -v 7zz >/dev/null 2>&1; then  # newer 7-Zip name on some distros
        7zz x -y "$1" -o"$2" >/dev/null
    else
        echo "ERROR: 7z not found. Install with: apt install p7zip-full / brew install p7zip" >&2
        return 1
    fi
}

IFS=',' read -ra FOLD_LIST <<< "$FOLDS"
for f in "${FOLD_LIST[@]}"; do
    out="$MODELS/fold_${f}_fp16.onnx"
    if [[ -s "$out" ]]; then
        echo "[fetch_weights] fold_${f}_fp16.onnx already present"
        continue
    fi

    url_7z="${RELEASE_BASE}/fold_${f}_fp16.onnx.7z"
    url_raw="${RELEASE_BASE}/fold_${f}_fp16.onnx"

    if curl -fsI -o /dev/null "$url_7z" 2>/dev/null; then
        echo "[fetch_weights] downloading $url_7z (compressed)"
        tmp7z="$MODELS/.fold_${f}.7z.tmp"
        curl -fsSL "$url_7z" -o "$tmp7z"
        extract_7z "$tmp7z" "$MODELS"
        rm -f "$tmp7z"
    else
        echo "[fetch_weights] downloading $url_raw (uncompressed)"
        curl -fsSL "$url_raw" -o "$out"
    fi
done

echo "[fetch_weights] done. $(ls -1 "$MODELS"/*.onnx | wc -l) folds in $MODELS"
