#!/usr/bin/env bash
# Export all 5 folds to ONNX (fp32 + fp16). Skips parity check (use
# `python3 export.py --skip-export ...` to re-validate any single fold).
set -euo pipefail
HERE="$(dirname "$0")"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
mkdir -p models
for f in 0 1 2 3 4; do
    echo "=== fold $f ==="
    python3 tools/onnx_export/export.py \
        --fold "$f" \
        --out  "models/fold_${f}_fp32.onnx" \
        --fp16-out "models/fold_${f}_fp16.onnx" \
        --no-verify
    # Drop the fp32 once the fp16 is written, to save disk.
    rm -f "models/fold_${f}_fp32.onnx"
done
echo "all folds done:"
ls -lh models/
