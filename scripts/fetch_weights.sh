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

# SIAMIZE_RELEASE_BASE: URL prefix hosting the fp16 ONNX folds. The
# script appends `fold_<N>_fp16.onnx[.gz]` directly to the prefix
# without a `/` separator, so the prefix is expected to end with the
# parameter that takes the filename (e.g. NeuroJSON's `...&file=`).
# Defaults to the NeuroJSON URL hosting the dynamic-shape SIAM v0.3
# folds; override to point at a self-hosted mirror.
# SIAMIZE_FOLDS: comma-separated list of folds to fetch (default 0,1,2,3,4).
#   Set SIAMIZE_FOLDS=0 to grab just fold 0 (CI smoke tests).
RELEASE_BASE="${SIAMIZE_RELEASE_BASE:-https://neurojson.org/io/stat.cgi?action=get&db=siam_v03&doc=dynshape&size=95360591&file=}"
FOLDS="${SIAMIZE_FOLDS:-0,1,2,3,4}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$HERE/models"
mkdir -p "$MODELS"

# Try compressed (.onnx.gz) first since it's ~3x smaller; fall back to raw.
# gzip is universally available on Linux/macOS/Windows (Git Bash); no extra
# install step needed.
IFS=',' read -ra FOLD_LIST <<< "$FOLDS"
for f in "${FOLD_LIST[@]}"; do
    out="$MODELS/fold_${f}_fp16.onnx"
    if [[ -s "$out" ]]; then
        echo "[fetch_weights] fold_${f}_fp16.onnx already present"
        continue
    fi

    url_gz="${RELEASE_BASE}fold_${f}_fp16.onnx.gz"
    url_raw="${RELEASE_BASE}fold_${f}_fp16.onnx"

    if curl -fsI -o /dev/null "$url_gz" 2>/dev/null; then
        echo "[fetch_weights] downloading $url_gz (gzip)"
        tmp_gz="$MODELS/fold_${f}_fp16.onnx.gz"
        curl -fsSL "$url_gz" -o "$tmp_gz"
        gunzip -f "$tmp_gz"
    else
        echo "[fetch_weights] downloading $url_raw (uncompressed)"
        curl -fsSL "$url_raw" -o "$out"
    fi
done

echo "[fetch_weights] done. $(ls -1 "$MODELS"/*.onnx | wc -l) folds in $MODELS"
