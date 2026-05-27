"""
Stage 2: Op-type histogram of the converted MNN graph.

Verifies that MNN's 3.0+ deprecation rewrite actually fired -- i.e., that
no Conv3D / ConvTranspose3D / Resize3D ops survive as runtime ops. If
they do, the OpenCL backend will hit the same coverage walls that killed
the ncnn-based exploration.

This is a Python re-implementation of what the legacy bash shell did via
the `MNNDump2Json` source-build-only tool. The pip wheel `MNN==3.5.0`
ships only `mnn`, `mnnconvert`, and `mnnquant` CLIs, not the dumper --
but the same information is reachable through `MNN.expr.load_as_list`,
which returns every Var in the graph (each one carries its producing
op_type).

Inputs (env vars):
  PROBE_DIR  default /tmp/mnn-probe
  MNN_IN     converted .mnn from Stage 1, default $PROBE_DIR/fold_0_fp16.mnn

Decision gates:
  PASS:    pure 2D Conv + Add + Reshape + Reduce + elementwise + 1D Interp.
           No 3D ops surviving.
  PARTIAL: InstanceNorm3D / Resize3D still listed (likely will run on CPU
           even with OpenCL backend, but everything else accelerates).
  FAIL:    Conv3D or ConvTranspose3D survive -- the deprecation rewrite
           didn't fire. Stop and investigate MNN build / version.

Usage:
  python3 02_op_histogram.py
"""

import os
import re
import sys
from collections import Counter

PROBE_DIR = os.environ.get("PROBE_DIR", "/tmp/mnn-probe")
MNN_IN = os.environ.get("MNN_IN", os.path.join(PROBE_DIR, "fold_0_fp16.mnn"))

if not os.path.isfile(MNN_IN):
    print(f"FAIL: {MNN_IN} not found. Run 01_convert.sh first.", file=sys.stderr)
    sys.exit(1)

import MNN.expr as F

print(f"=== loading {MNN_IN} via MNN.expr.load_as_list ===")
vars_ = F.load_as_list(MNN_IN)
print(f"graph has {len(vars_)} vars in topological order")
print()

# Tally op_type frequencies. Each Var carries the op_type of the op that
# produced it; that's the histogram we want.
hist = Counter(v.op_type for v in vars_)

print("=== Op-type histogram (sorted by frequency) ===")
for op_type, n in hist.most_common():
    print(f"  {n:5d}  {op_type}")
print()

print(f"=== unique op types: {len(hist)} ===")
print(f"=== total ops: {sum(hist.values())} ===")
print()

# Split 3D-survivor analysis into two buckets:
#   FATAL_3D_RX: ops whose absence of OpenCL kernel coverage would be a
#                hard blocker -- the Conv3D family carries 99% of SIAM's
#                compute. If these survive, the deprecation rewrite did
#                not fire and OpenCL acceleration is impossible.
#   OTHER_3D_RX: small 3D ops (pooling, normalization, interpolation)
#                that survive in MNN 3.x even after Conv3D decomposition.
#                Their OpenCL kernel coverage is determined empirically
#                in Stage 4 (silent-fallback detection).
FATAL_3D_RX = re.compile(
    r"(Conv(olution)?3D$|ConvTranspose3D$|Convolution3DDepthwise)", re.I
)
OTHER_3D_RX = re.compile(r"^[A-Za-z]+3D$")

fatal_3d = sorted({op for op in hist if FATAL_3D_RX.search(op)})
other_3d = sorted({op for op in hist if OTHER_3D_RX.search(op) and op not in fatal_3d})

print("=== Conv3D-family ops surviving (FAIL gate: this MUST be empty) ===")
if fatal_3d:
    for op in fatal_3d:
        print(f"  {hist[op]:5d}  {op}")
    print()
    print(
        "VERDICT: FAIL -- Conv3D-family ops survived the deprecation rewrite.",
        file=sys.stderr,
    )
    print(
        "  MNN 3.0+ should have decomposed these to 2D Conv primitives.",
        file=sys.stderr,
    )
    print("  Check the MNNConvert version (Stage 1 log).", file=sys.stderr)
    sys.exit(2)
print("(none -- good, the Conv3D deprecation rewrite fired)")
print()

print("=== Other 3D ops surviving (PARTIAL: Stage 4 decides OpenCL impact) ===")
if other_3d:
    for op in other_3d:
        print(f"  {hist[op]:5d}  {op}")
    print()
    print("These ops are small in count but execute on rank-5 tensors. Whether")
    print("MNN's OpenCL backend has kernels for them is the real Stage 4")
    print("question. If OpenCL kernels are missing, they will silently fall")
    print("back to CPU -- correct output, but expensive H<->D round-trips.")
    partial = True
else:
    print("(none -- ideal; every 3D op decomposed to elementary 2D primitives)")
    partial = False
print()

# Conv2D count is a useful sanity check: the decomposition should produce
# a LOT of Conv2D operations from each original Conv3D (typically one
# Conv2D per Z-slice of the 3D kernel).
conv_like = {op: n for op, n in hist.items() if "conv" in op.lower()}
if conv_like:
    print("=== Conv-family ops ===")
    for op, n in sorted(conv_like.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {op}")
    print()

verdict = "PARTIAL" if partial else "PASS"
print(f"VERDICT: {verdict} -- proceed to Stage 3 (CPU parity).")
