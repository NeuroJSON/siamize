"""
Stage 3: Numerical parity check between ORT and MNN on the CPU backend.

Feeds a deterministic synthetic (1, 1, 256, 256, 192) tile through both
ORT (CPU EP, fp32) and MNN (CPU backend), then compares the resulting
18-class logit tensors and per-voxel argmax labelmaps.

Decision gates:
  PASS:    max|diff| < 1e-3, argmax agreement >= 99.9%
  PARTIAL: mild logit drift (1-5% relative) but argmax agreement >= 99%.
           Likely indicates fp16/fp32 cast handling difference inside MNN;
           usually acceptable for inference use.
  FAIL:    argmax agreement < 99%. Conversion produced a math-different
           graph -- e.g. a decomposed 3D conv that doesn't exactly
           reproduce the original. Stop and file an MNN issue.

Inputs (env vars):
  PROBE_DIR  default /tmp/mnn-probe
  ONNX_IN    default $PROBE_DIR/fold_0.onnx
  MNN_IN     default $PROBE_DIR/fold_0.mnn

Usage:
  python3 03_cpu_parity.py
"""

import os
import sys
import time

import numpy as np

PROBE_DIR = os.environ.get("PROBE_DIR", "/tmp/mnn-probe")
ONNX_IN = os.environ.get("ONNX_IN", os.path.join(PROBE_DIR, "fold_0.onnx"))
MNN_IN = os.environ.get("MNN_IN", os.path.join(PROBE_DIR, "fold_0.mnn"))

PATCH_SHAPE = (1, 1, 256, 256, 192)  # SIAM's nominal patch
SEED = 0

# ---------------------------------------------------------------- ORT
print("=" * 60)
print("ORT CPU baseline")
print("=" * 60)

import onnxruntime as ort

rng = np.random.default_rng(SEED)
x_np = rng.standard_normal(PATCH_SHAPE).astype(np.float32)
print(f"input shape={x_np.shape}, dtype={x_np.dtype}, seed={SEED}")

t0 = time.time()
ort_sess = ort.InferenceSession(ONNX_IN, providers=["CPUExecutionProvider"])
y_ort = ort_sess.run(["logits"], {"x": x_np})[0]
ort_time = time.time() - t0
print(f"ORT CPU run:    {ort_time:.2f} s")
print(f"ORT output:     shape={y_ort.shape}, dtype={y_ort.dtype}")
print(f"ORT logit range: [{y_ort.min():.3f}, {y_ort.max():.3f}]")

# ---------------------------------------------------------------- MNN
print()
print("=" * 60)
print("MNN CPU backend")
print("=" * 60)

import MNN

# Backend dispatch:
#   CPU = 0  CUDA = 2  OPENCL = 3  OPENGL = 6  VULKAN = 7  METAL = 1
interp = MNN.Interpreter(MNN_IN)
sess_cfg = {"backend": "CPU", "numThread": 8, "precision": "high"}
sess = interp.createSession(sess_cfg)
print(f"MNN session config: {sess_cfg}")
print(
    f"MNN inputs:  {[(n, t.getShape()) for n, t in interp.getSessionInputAll(sess).items()]}"
)
print(
    f"MNN outputs: {[(n, t.getShape()) for n, t in interp.getSessionOutputAll(sess).items()]}"
)

in_tensor = interp.getSessionInput(sess)
# Resize if MNN's declared input shape doesn't match (some converters
# fix the shape to whatever was in the ONNX I/O signature).
if tuple(in_tensor.getShape()) != PATCH_SHAPE:
    print(
        f"resizing input from {tuple(in_tensor.getShape())} to {PATCH_SHAPE}",
        flush=True,
    )
    interp.resizeTensor(in_tensor, list(PATCH_SHAPE))
    interp.resizeSession(sess)

# Copy data in. MNN's Halide_Type_Float is fp32; Caffe layout = NCHW
# (extended to NCDHW for 3D).
mnn_in = MNN.Tensor(
    PATCH_SHAPE, MNN.Halide_Type_Float, x_np, MNN.Tensor_DimensionType_Caffe
)
in_tensor.copyFrom(mnn_in)

t0 = time.time()
interp.runSession(sess)
mnn_time = time.time() - t0

out_tensor = interp.getSessionOutput(sess)
y_mnn = np.array(out_tensor.getData(), dtype=np.float32).reshape(out_tensor.getShape())
print(f"MNN CPU run:    {mnn_time:.2f} s")
print(f"MNN output:     shape={y_mnn.shape}, dtype={y_mnn.dtype}")
print(f"MNN logit range: [{y_mnn.min():.3f}, {y_mnn.max():.3f}]")

# ---------------------------------------------------------------- Compare
print()
print("=" * 60)
print("Parity comparison (ORT CPU vs MNN CPU)")
print("=" * 60)

if y_ort.shape != y_mnn.shape:
    print(
        f"FAIL: output shape mismatch. ORT={y_ort.shape}, MNN={y_mnn.shape}",
        file=sys.stderr,
    )
    sys.exit(2)

diff = np.abs(y_ort.astype(np.float64) - y_mnn.astype(np.float64))
argmax_ort = y_ort.argmax(axis=1)  # axis 1 is class
argmax_mnn = y_mnn.argmax(axis=1)
agreement = float((argmax_ort == argmax_mnn).mean())

print(f"max|diff|:       {diff.max():.4e}")
print(f"mean|diff|:      {diff.mean():.4e}")
print(f"median|diff|:    {np.median(diff):.4e}")
print(
    f"argmax agree:    {agreement * 100:.4f}%  ({(argmax_ort == argmax_mnn).sum()} / {argmax_ort.size} voxels)"
)

# Decision
print()
if agreement >= 0.999 and diff.max() < 1e-3:
    print("VERDICT: PASS -- ORT and MNN CPU outputs match within fp16 noise.")
    print("Proceed to Stage 4 (04_opencl_parity.py).")
elif agreement >= 0.99:
    print("VERDICT: PARTIAL -- mild logit drift but argmax mostly agrees.")
    print("Likely fp16-cast difference inside MNN. Continue to Stage 4 with caveat.")
    sys.exit(0)
else:
    print(f"VERDICT: FAIL -- argmax agreement {agreement*100:.4f}% < 99%.")
    print("Conversion produced a math-different graph. Stop.")
    print()
    print("Possible causes:")
    print("  - MNN's Conv3D-to-2D decomposition has a bug for SIAM's kernel patterns")
    print("  - InstanceNorm3D is being decomposed incorrectly")
    print("  - fp16/fp32 conversion inside MNN is lossy in a problematic way")
    sys.exit(3)
