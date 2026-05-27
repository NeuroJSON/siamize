#!/usr/bin/env bash
# Stage 2: Dump the converted .mnn to JSON and tally the op-type
# histogram. The point is to verify that MNN's 3.0+ deprecation rewrite
# actually fired -- i.e., that no Conv3D / ConvTranspose3D / Resize3D ops
# survive in the runtime graph. If they do, the OpenCL backend will hit
# the same coverage walls that killed the ncnn-based exploration.
#
# Inputs (env vars):
#   PROBE_DIR  default /tmp/mnn-probe
#   MNN_IN     converted .mnn from Stage 1, default $PROBE_DIR/fold_0_fp16.mnn
#
# Decision gates:
#   PASS:    pure 2D Conv + Add + Reshape + Reduce + elementwise + 1D Interp.
#            No 3D ops surviving.
#   PARTIAL: InstanceNorm3D / Resize3D still listed (likely will run on CPU
#            even with OpenCL backend, but everything else accelerates).
#   FAIL:    Conv3D or ConvTranspose3D survive -- the deprecation rewrite
#            didn't fire. Stop and investigate MNN build / version.

set -e

PROBE_DIR="${PROBE_DIR:-/tmp/mnn-probe}"
MNN_IN="${MNN_IN:-$PROBE_DIR/fold_0_fp16.mnn}"
JSON_OUT="${PROBE_DIR}/fold_0_fp16.json"

if [ ! -f "$MNN_IN" ]; then
    echo "FAIL: $MNN_IN not found. Run 01_convert.sh first." >&2
    exit 1
fi

if ! command -v MNNDump2Json >/dev/null; then
    echo "FAIL: MNNDump2Json not in PATH." >&2
    echo "  Provided by the same MNN build/install that supplied MNNConvert." >&2
    exit 1
fi

cd "$PROBE_DIR"

echo "=== dumping $MNN_IN -> $JSON_OUT ==="
MNNDump2Json "$MNN_IN" "$JSON_OUT"
ls -lh "$JSON_OUT"

if ! command -v jq >/dev/null; then
    echo "WARN: jq not installed; printing raw op-type lines instead." >&2
    grep -oE '"type"\s*:\s*"[^"]+"' "$JSON_OUT" | sort | uniq -c | sort -rn
    exit 0
fi

echo
echo "=== Op-type histogram (sorted by frequency) ==="
jq -r '.oplists[].type' "$JSON_OUT" | sort | uniq -c | sort -rn

echo
echo "=== Total op count ==="
jq '.oplists | length' "$JSON_OUT"

echo
echo "=== 3D-shaped ops still present (FAIL gate: this should be empty) ==="
surviving_3d=$(jq -r '.oplists[].type' "$JSON_OUT" \
               | grep -iE "Conv(olution)?3D|ConvTranspose3D|Interp3D|Pool3D" \
               || true)
if [ -n "$surviving_3d" ]; then
    echo "$surviving_3d"
    echo
    echo "VERDICT: FAIL -- 3D ops survived the deprecation rewrite." >&2
    echo "  MNN 3.0+ should have decomposed Conv3D / ConvTranspose3D to 2D." >&2
    echo "  Check MNNConvert version + MNN_SUPPORT_DEPRECATED_OPV2 build flag." >&2
    exit 2
fi
echo "(none -- good, the deprecation rewrite fired)"

echo
echo "=== InstanceNorm and Resize variants (likely surviving as runtime ops) ==="
nominal_3d=$(jq -r '.oplists[].type' "$JSON_OUT" \
             | grep -iE "InstanceNorm|Interp|Resize" \
             || true)
if [ -n "$nominal_3d" ]; then
    echo "$nominal_3d"
    echo
    echo "NOTE: these ops typically still exist as runtime ops even"
    echo "in MNN 3.0+. Stage 4 will tell us whether their OpenCL"
    echo "kernels accept the 3D shapes SIAM feeds them."
else
    echo "(none -- ideal; everything decomposed to elementary 2D primitives)"
fi

echo
echo "VERDICT: PASS -- proceed to Stage 3."
