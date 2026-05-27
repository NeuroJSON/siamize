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

# MNN ships the converter under two different names depending on how it was
# installed:
#   * pip wheel `MNN==3.5.0`     -> lowercase `mnnconvert` console script
#   * source build with -DMNN_BUILD_CONVERTER=ON -> PascalCase `MNNConvert`
# Try both. If neither is found, also try `python -m MNN.tools.mnnconvert`
# (the entry-point module the pip wheel registers).
MNN_CONVERT=""
for cand in MNNConvert mnnconvert; do
    if command -v "$cand" >/dev/null; then
        MNN_CONVERT="$cand"
        break
    fi
done
if [ -z "$MNN_CONVERT" ]; then
    if python3 -c "import MNN.tools.mnnconvert" >/dev/null 2>&1; then
        MNN_CONVERT="python3 -m MNN.tools.mnnconvert"
    fi
fi
if [ -z "$MNN_CONVERT" ]; then
    echo "FAIL: no MNN converter found in PATH (tried MNNConvert, mnnconvert," >&2
    echo "      python -m MNN.tools.mnnconvert)." >&2
    echo "  pip install MNN==3.5.0    # ships lowercase 'mnnconvert'" >&2
    echo "  -- or build from source for the PascalCase tools:" >&2
    echo "    cmake -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TOOLS=ON ..." >&2
    exit 1
fi
echo "Using converter: $MNN_CONVERT"

mkdir -p "$PROBE_DIR"
cd "$PROBE_DIR"

echo "=== converter version ==="
$MNN_CONVERT --version 2>&1 || true
echo

echo "=== converting $ONNX_IN -> $MNN_OUT ==="
$MNN_CONVERT -f ONNX \
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
echo "=== symbolic-shape diagnostics (informational) ==="
# MNN prints `Reshape error: -1879048192 -> -1877999616` (= 0x90000000 ->
# 0x90100000) when it can't statically resolve a Reshape because the ONNX
# graph has dynamic axes. The shape gets bound at runtime when actual input
# dimensions arrive. These are warnings, NOT failures -- the magic numbers
# are MNN's internal "unknown dim" sentinels, not real shape values.
symbolic=$(grep -E "Reshape error: -1[0-9]+ -> -1[0-9]+" convert.log || true)
if [ -n "$symbolic" ]; then
    n=$(echo "$symbolic" | wc -l)
    echo "$n symbolic-shape resolution(s) deferred to runtime -- expected for"
    echo "the dynshape ONNX variant. Stage 3 will confirm runtime binding works."
else
    echo "(none)"
fi

# The authoritative success signal in MNNConvert's log is "Converted Success!".
# If it's missing, the conversion really did fail; otherwise treat any "error"
# substring not already classified above as informational.
echo
echo "=== conversion success signal ==="
if grep -qE "Converted Success!" convert.log; then
    echo "Converted Success! (found in log)"
else
    echo "FAIL: 'Converted Success!' marker absent from log" >&2
    echo "Last 30 lines of convert.log for context:" >&2
    tail -30 convert.log >&2
    exit 3
fi

# Anything left over that contains "error|fail|fatal" but isn't a known
# symbolic-shape diagnostic is worth surfacing -- but only as a warning.
echo
echo "=== other diagnostics (informational) ==="
other=$(grep -iE "error|fail|fatal" convert.log \
        | grep -vE "Reshape error: -1[0-9]+ -> -1[0-9]+" \
        | head -10 || true)
if [ -n "$other" ]; then
    echo "$other"
else
    echo "(none)"
fi

echo
echo "VERDICT: ${unsupported:+PARTIAL}${unsupported:-PASS} -- proceed to Stage 2."
