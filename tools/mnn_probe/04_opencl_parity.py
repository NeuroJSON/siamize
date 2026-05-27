"""
Stage 4: OpenCL backend correctness + silent-fallback detection.

Repeats Stage 3's single-tile comparison but with MNN's OpenCL backend.
Compares the OpenCL output to both ORT (ground truth) AND MNN CPU (which
should match if conversion is correct). Then samples GPU utilization to
verify the GPU is actually engaged -- MNN may silently fall back to CPU
for ops without OpenCL kernels, which would defeat the entire reason for
trying MNN.

Decision gates:
  PASS:    OpenCL output matches MNN CPU output within ~1 ULP fp16 noise,
           AND nvidia-smi / intel_gpu_top / radeontop shows non-zero GPU
           utilization during the run.
  FAIL:    OpenCL output diverges from MNN CPU -- specific OpenCL kernel
           has a numerical bug. File MNN issue.
  SILENT
  FALLBACK: OpenCL "runs" but GPU utilization stays at 0% throughout.
            One or more ops are silently running on CPU. The remaining
            ops accelerate but the unsupported ones force expensive
            host<->device round-trips per call.

Inputs (env vars):
  PROBE_DIR  default /tmp/mnn-probe
  MNN_IN     default $PROBE_DIR/fold_0_fp16.mnn

Usage:
  # In one terminal, start a GPU-utilization monitor before running:
  #   NVIDIA:  nvidia-smi dmon -s u -c 60
  #   Intel:   sudo intel_gpu_top -L
  #   AMD:     sudo radeontop -d - -l 1
  python3 04_opencl_parity.py
"""

import os
import sys
import time

import numpy as np

PROBE_DIR = os.environ.get("PROBE_DIR", "/tmp/mnn-probe")
MNN_IN = os.environ.get("MNN_IN", os.path.join(PROBE_DIR, "fold_0_fp16.mnn"))
ONNX_IN = os.environ.get("ONNX_IN", os.path.join(PROBE_DIR, "fold_0_fp16.onnx"))

PATCH_SHAPE = (1, 1, 256, 256, 192)
SEED = 0

rng = np.random.default_rng(SEED)
x_np = rng.standard_normal(PATCH_SHAPE).astype(np.float32)
print(f"Input: shape={x_np.shape}, dtype={x_np.dtype}, seed={SEED}")
print()

import MNN


def run_mnn(backend: str, num_iters: int = 1):
    """Returns (output_array, total_seconds)."""
    interp = MNN.Interpreter(MNN_IN)
    sess_cfg = {"backend": backend, "numThread": 0, "precision": "high"}
    sess = interp.createSession(sess_cfg)
    print(f"MNN {backend} session config: {sess_cfg}")

    in_tensor = interp.getSessionInput(sess)
    if tuple(in_tensor.getShape()) != PATCH_SHAPE:
        interp.resizeTensor(in_tensor, list(PATCH_SHAPE))
        interp.resizeSession(sess)

    mnn_in = MNN.Tensor(
        PATCH_SHAPE, MNN.Halide_Type_Float, x_np, MNN.Tensor_DimensionType_Caffe
    )

    # Warm-up: OpenCL JIT-compiles kernels on first run, which can take
    # 5-30 s. We don't time the warm-up.
    in_tensor.copyFrom(mnn_in)
    print(f"  warm-up run (compiling OpenCL kernels if applicable)...", flush=True)
    t0 = time.time()
    interp.runSession(sess)
    print(f"  warm-up: {time.time() - t0:.2f} s")

    t0 = time.time()
    for _ in range(num_iters):
        in_tensor.copyFrom(mnn_in)
        interp.runSession(sess)
    timed = (time.time() - t0) / num_iters

    out_tensor = interp.getSessionOutput(sess)
    y = np.array(out_tensor.getData(), dtype=np.float32).reshape(out_tensor.getShape())
    return y, timed


# ---------------------------------------------------------------- MNN CPU
print("=" * 60)
print("MNN CPU (reference for OpenCL comparison)")
print("=" * 60)
y_cpu, cpu_time = run_mnn("CPU", num_iters=1)
print(f"MNN CPU run:    {cpu_time:.2f} s")
print(f"Output shape:   {y_cpu.shape}")
print()

# ---------------------------------------------------------------- MNN OpenCL
print("=" * 60)
print("MNN OpenCL backend (the actual test)")
print("=" * 60)
print()
print("Reminder: in another terminal, start a GPU utility monitor BEFORE")
print("this script reaches `runSession()`. Watch for non-zero GPU %.")
print("  NVIDIA:  nvidia-smi dmon -s u -c 60")
print("  Intel:   sudo intel_gpu_top -L")
print("  AMD:     sudo radeontop -d - -l 1")
print()

# Sleep briefly so the user can switch terminals / start the monitor
# if they haven't already.
if os.environ.get("NO_PAUSE") != "1":
    print("Pausing 5s; set NO_PAUSE=1 to skip.")
    time.sleep(5)

try:
    y_opencl, opencl_time = run_mnn("OPENCL", num_iters=3)
except Exception as e:
    print(f"FAIL: OpenCL session creation / run failed: {e}", file=sys.stderr)
    print(
        "Likely cause: MNN was built without OPENCL support (need cmake",
        file=sys.stderr,
    )
    print("  -DMNN_OPENCL=ON), or the host has no OpenCL ICD loader.", file=sys.stderr)
    sys.exit(2)

print(f"MNN OpenCL run: {opencl_time:.2f} s (avg of 3 iterations after warm-up)")
print(f"Output shape:   {y_opencl.shape}")
print()

# ---------------------------------------------------------------- Compare
print("=" * 60)
print("Numerical comparison (MNN OpenCL vs MNN CPU)")
print("=" * 60)

if y_cpu.shape != y_opencl.shape:
    print(
        f"FAIL: output shape mismatch. CPU={y_cpu.shape}, OpenCL={y_opencl.shape}",
        file=sys.stderr,
    )
    sys.exit(2)

diff = np.abs(y_cpu.astype(np.float64) - y_opencl.astype(np.float64))
argmax_cpu = y_cpu.argmax(axis=1)
argmax_ocl = y_opencl.argmax(axis=1)
agreement = float((argmax_cpu == argmax_ocl).mean())

print(f"max|diff|:       {diff.max():.4e}")
print(f"mean|diff|:      {diff.mean():.4e}")
print(f"argmax agree:    {agreement * 100:.4f}%")
print()
print(f"Wall-time:       CPU={cpu_time:.2f} s   OpenCL={opencl_time:.2f} s")
print(f"OpenCL speedup:  {cpu_time / opencl_time:.2f}x")

# Decision
print()
if agreement >= 0.999 and diff.max() < 1e-2:
    print("VERDICT (numerical): PASS -- OpenCL output matches CPU within fp16 noise.")
elif agreement >= 0.99:
    print("VERDICT (numerical): PARTIAL -- OpenCL drifts slightly but argmax mostly")
    print("                     agrees. Acceptable for inference.")
else:
    print(f"VERDICT (numerical): FAIL -- argmax agreement {agreement*100:.4f}% < 99%.")
    print("OpenCL kernel for some decomposed op has a numerical bug.")
    print("Stop and file an MNN GitHub issue with the op histogram from Stage 2.")
    sys.exit(3)

print()
print("=" * 60)
print("MANUAL CHECK: was the GPU monitor showing non-zero utilization?")
print("=" * 60)
print()
print("If GPU utilization stayed at 0% throughout, MNN silently fell back")
print("to CPU for one or more ops. The run completed but no acceleration")
print("happened. This is the 'silent fallback' failure mode.")
print()
print("If the speedup is poor (~1x) AND GPU utilization was high, the")
print("OpenCL backend ran the graph but kernel-launch overhead dominated --")
print("typical for a Conv3D-decomposed-to-many-Conv2D graph on OpenCL.")
print()
print("Record what you observed in REPORT_TEMPLATE.md before moving to Stage 5.")
