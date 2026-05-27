#!/usr/bin/env bash
# Stage 1: Convert SIAM's ONNX to MNN format and inspect the conversion log
# for any warnings that would signal unsupported ops.
#
# Inputs (env vars, with defaults):
#   PROBE_DIR  workspace, default /tmp/mnn-probe
#   ONNX_IN    source ONNX path, default $PROBE_DIR/fold_0_fp16.onnx
#   MNN_OUT    target .mnn path, default $PROBE_DIR/fold_0_fp16.mnn
#
# Decision gates:
#   PASS:    no "not support" / "unsupport" / "unknown op" / "fail" in the
#            conversion log.
#   PARTIAL: conversion succeeded but some 3D ops were warned about; they
#            will likely fall back to CPU even when the OpenCL backend is
#            selected at run time.
#   FAIL:    "Convolution3D not supported" or similar fatal error -- the
#            MNN 3.0 deprecation rewrite (Conv3D -> 2D primitives) didn't
#            fire for SIAM's specific Conv3D patterns. Stop.

set -e

PROBE_DIR="${PROBE_DIR:-/tmp/mnn-probe}"
ONNX_IN="${ONNX_IN:-$PROBE_DIR/fold_0_fp16.onnx}"
MNN_OUT="${MNN_OUT:-$PROBE_DIR/fold_0_fp16.mnn}"

if [ ! -f "$ONNX_IN" ]; then
    echo "FAIL: source ONNX not found at $ONNX_IN" >&2
    echo "  Place fold_0_fp16.onnx at $ONNX_IN, e.g.:" >&2
    echo "    cp <siamize-repo>/models/fold_0_fp16.onnx $ONNX_IN" >&2
    echo "  Or fetch the dynshape variant from NeuroJSON:" >&2
    echo "    URL='https://neurojson.org/io/stat.cgi?action=get&db=siam_v03&doc=dynshape&file=fold_0_fp16.onnx.gz'" >&2
    echo "    curl -sSL \"\$URL\" | gunzip > $ONNX_IN" >&2
    exit 1
fi

if ! command -v MNNConvert >/dev/null; then
    echo "FAIL: MNNConvert not in PATH." >&2
    echo "  Either pip install MNN==3.5.0, or build from source:" >&2
    echo "    cmake -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TOOLS=ON ..." >&2
    exit 1
fi

mkdir -p "$PROBE_DIR"
cd "$PROBE_DIR"

echo "=== MNNConvert version ==="
MNNConvert --version || true
echo

echo "=== converting $ONNX_IN -> $MNN_OUT ==="
MNNConvert -f ONNX \
    --modelFile "$ONNX_IN" \
    --MNNModel  "$MNN_OUT" \
    --bizCode   SIAM_PROBE \
    2>&1 | tee convert.log

if [ ! -f "$MNN_OUT" ]; then
    echo "FAIL: conversion produced no output file" >&2
    exit 2
fi

echo
echo "=== sizes ==="
ls -lh "$ONNX_IN" "$MNN_OUT"

echo
echo "=== unsupported-op warnings (PASS gate: this section should be empty) ==="
unsupported=$(grep -iE "not support|unsupport|unknown op|skipped" convert.log || true)
if [ -n "$unsupported" ]; then
    echo "$unsupported"
    echo
    echo "VERDICT: PARTIAL -- some ops will fall back at runtime."
    # not exit 1 -- caller can decide to continue
else
    echo "(none)"
fi

echo
echo "=== decomposition / rewrite notices ==="
grep -iE "decompos|rewrit|emulat" convert.log || echo "(none -- conversion silent, which is normal)"

echo
echo "=== critical errors (FAIL gate: this section MUST be empty) ==="
critical=$(grep -iE "^.*error|fail|fatal" convert.log | head -20 || true)
if [ -n "$critical" ]; then
    echo "$critical"
    echo
    echo "VERDICT: FAIL -- stop here, investigate." >&2
    exit 3
fi

echo "(none)"
echo
echo "VERDICT: ${unsupported:+PARTIAL}${unsupported:-PASS} -- proceed to Stage 2."
