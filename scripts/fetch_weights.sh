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

# TODO: replace with a real release tag URL once the first siamize release
# is cut. Example URL pattern:
#   https://github.com/NeuroJSON/siamize/releases/download/v0.1.0/fold_${f}_fp16.onnx
RELEASE_BASE="${SIAMIZE_RELEASE_BASE:-}"

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

for f in 0 1 2 3 4; do
    out="$MODELS/fold_${f}_fp16.onnx"
    if [[ -s "$out" ]]; then
        echo "[fetch_weights] fold_${f}_fp16.onnx already present"
        continue
    fi
    url="${RELEASE_BASE}/fold_${f}_fp16.onnx"
    echo "[fetch_weights] downloading $url"
    curl -fsSL "$url" -o "$out"
done

echo "[fetch_weights] done. $(ls -1 "$MODELS"/*.onnx | wc -l) folds in $MODELS"
