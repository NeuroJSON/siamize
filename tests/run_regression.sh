#!/usr/bin/env bash
# Run the bundled sample through siamize and report agreement with the
# checked-in reference (pred_ref_allfolds.nii.gz).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

BIN="${SIAMIZE_BIN:-build/siamize}"
if [[ ! -x "$BIN" ]]; then
    echo "$BIN not found. Build it first: cmake -S . -B build && cmake --build build -j" >&2
    exit 2
fi

MODELS=()
for f in 0 1 2 3 4; do
    p="models/fold_${f}_fp16.onnx"
    if [[ ! -s "$p" ]]; then
        echo "$p not found. Run scripts/fetch_weights.sh or tools/onnx_export/export_all_folds.sh" >&2
        exit 2
    fi
    MODELS+=("$p")
done
MODELS_CSV="$(IFS=,; echo "${MODELS[*]}")"

OUT="tests/pred_regression.nii.gz"
"$BIN" -i tests/sub-01_T1w.nii.gz -o "$OUT" -M "$MODELS_CSV" -t 8 -v

python3 tools/onnx_export/compare.py --ref tests/pred_ref_allfolds.nii.gz --new "$OUT"
