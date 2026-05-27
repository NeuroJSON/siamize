#!/usr/bin/env bash
# Convenience wrapper: run the five probe stages in order, stopping at
# the first FAIL. Drop-in usage:
#
#   cd <siamize-repo>/tools/mnn_probe
#   ./run_all.sh
#
# Each stage's PASS / FAIL semantics are documented in its own header
# comment (and consolidated in README.md). Stages 1-2 are bash; stages
# 3-5 are python and assume the venv set up in README.md is active.
#
# Inputs (env vars, with defaults):
#   PROBE_DIR       /tmp/mnn-probe
#   SIAMIZE_REPO    /drives/bixi1/users/fangq/git/Temp/siamize
#   ONNX_IN         $PROBE_DIR/fold_0_fp16.onnx
#   MNN_IN          $PROBE_DIR/fold_0_fp16.mnn
#   INPUT_NII       $SIAMIZE_REPO/tests/sub-01_T1w.nii.gz
#   SKIP_STAGES     comma-separated stage numbers to skip (e.g. "5")

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_DIR="${PROBE_DIR:-/tmp/mnn-probe}"
SIAMIZE_REPO="${SIAMIZE_REPO:-$(cd "$HERE/../.." && pwd)}"
ONNX_IN="${ONNX_IN:-$PROBE_DIR/fold_0_fp16.onnx}"
MNN_IN="${MNN_IN:-$PROBE_DIR/fold_0_fp16.mnn}"
INPUT_NII="${INPUT_NII:-$SIAMIZE_REPO/tests/sub-01_T1w.nii.gz}"
SKIP_STAGES="${SKIP_STAGES:-}"

export PROBE_DIR SIAMIZE_REPO ONNX_IN MNN_IN INPUT_NII

echo "Workspace: $PROBE_DIR"
echo "ONNX:      $ONNX_IN"
echo "MNN:       $MNN_IN"
echo "Test NIfTI: $INPUT_NII"
echo

mkdir -p "$PROBE_DIR"

skip() {
    case ",$SKIP_STAGES," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Stage 1: convert
if skip 1; then
    echo "=== SKIP Stage 1 ==="
else
    echo "############################"
    echo "# Stage 1: convert"
    echo "############################"
    bash "$HERE/01_convert.sh"
    echo
fi

# Stage 2: op histogram (Python -- the pip wheel doesn't ship MNNDump2Json,
# so we walk the graph via MNN.expr.load_as_list instead).
if skip 2; then
    echo "=== SKIP Stage 2 ==="
else
    echo "############################"
    echo "# Stage 2: op histogram"
    echo "############################"
    python3 "$HERE/02_op_histogram.py"
    echo
fi

# Stage 3: CPU parity
if skip 3; then
    echo "=== SKIP Stage 3 ==="
else
    echo "############################"
    echo "# Stage 3: CPU parity (ORT vs MNN CPU)"
    echo "############################"
    python3 "$HERE/03_cpu_parity.py"
    echo
fi

# Stage 4: OpenCL parity
if skip 4; then
    echo "=== SKIP Stage 4 ==="
else
    echo "############################"
    echo "# Stage 4: OpenCL parity (MNN CPU vs MNN OpenCL)"
    echo "############################"
    echo "NOTE: start a GPU utility monitor in another terminal NOW if"
    echo "you want to observe whether OpenCL is actually engaging the GPU."
    echo "  nvidia-smi dmon -s u -c 60"
    echo "  intel_gpu_top -L     (Intel iGPU)"
    echo "  radeontop -d - -l 1  (AMD)"
    echo
    python3 "$HERE/04_opencl_parity.py"
    echo
fi

# Stage 5: end-to-end performance
if skip 5; then
    echo "=== SKIP Stage 5 ==="
else
    echo "############################"
    echo "# Stage 5: end-to-end performance"
    echo "############################"
    python3 "$HERE/05_e2e_perf.py"
    echo
fi

echo
echo "All stages complete. Fill out tools/mnn_probe/REPORT_TEMPLATE.md"
echo "and save under tools/mnn_probe/reports/<YYYY-MM-DD>-<hostname>.md"
