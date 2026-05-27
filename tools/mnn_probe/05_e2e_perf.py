"""
Stage 5: End-to-end performance characterization on a real input.

Drives MNN through siamize's actual preprocessing + sliding-window
inference + postprocessing pipeline on the bundled `sub-01_T1w.nii.gz`.
Times each backend (ORT CPU baseline, MNN CPU, MNN OpenCL) on a full
single-fold inference and reports the speedups.

This is the question Stages 1-4 don't answer: even if MNN works
correctly, is its OpenCL backend FAST enough to justify integration?
Decomposing a Conv3D into many Conv2D ops can be correct but
dispatch-overhead-heavy on OpenCL.

Decision gates:
  STRONG PASS:  MNN OpenCL >= 5x faster than MNN CPU. Worth integrating.
  MARGINAL:     2-5x. Useful only where ORT CUDA isn't available
                (AMD/Intel Linux GPUs, integrated GPUs).
  FAIL:         < 2x. Decomposed graph is too dispatch-heavy. Stop.
  SIDE-FAIL:    MNN CPU is itself > 1.5x slower than ORT CPU. MNN's CPU
                backend is uncompetitive; OpenCL won't save it.

Inputs (env vars):
  PROBE_DIR    default /tmp/mnn-probe
  ONNX_IN      default $PROBE_DIR/fold_0.onnx
  MNN_IN       default $PROBE_DIR/fold_0.mnn
  INPUT_NII    default <siamize-repo>/tests/sub-01_T1w.nii.gz
  SIAMIZE_REPO default /drives/bixi1/users/fangq/git/Temp/siamize

Usage:
  python3 05_e2e_perf.py
"""

import os
import sys
import time

import numpy as np

SIAMIZE_REPO = os.environ.get(
    "SIAMIZE_REPO", "/drives/bixi1/users/fangq/git/Temp/siamize"
)
PROBE_DIR = os.environ.get("PROBE_DIR", "/tmp/mnn-probe")
ONNX_IN = os.environ.get("ONNX_IN", os.path.join(PROBE_DIR, "fold_0.onnx"))
MNN_IN = os.environ.get("MNN_IN", os.path.join(PROBE_DIR, "fold_0.mnn"))
INPUT_NII = os.environ.get(
    "INPUT_NII", os.path.join(SIAMIZE_REPO, "tests", "sub-01_T1w.nii.gz")
)

# Use siamize's Python reference for the preprocessing + sliding window
# pipeline. Keeps the comparison apples-to-apples -- we're only swapping
# the inner inference call between ORT and MNN.
sys.path.insert(0, os.path.join(SIAMIZE_REPO, "py"))
try:
    import siam_ref  # noqa: F401
except ImportError as e:
    print(f"FAIL: cannot import siam_ref from {SIAMIZE_REPO}/py: {e}", file=sys.stderr)
    print(f"  Set SIAMIZE_REPO env var to the siamize checkout root.", file=sys.stderr)
    sys.exit(1)

import nibabel as nib

# Inspect what utilities siam_ref exposes; the helper names may evolve
# between versions, so be defensive.
REQUIRED = [
    "load_fold",  # builds the PyTorch network + plans
    "nib_to_zyx",
    "crop_to_nonzero",
    "zscore_normalize",
    "compute_new_shape",
    "resample_image",
    "pad_to_patch",
]
missing = [name for name in REQUIRED if not hasattr(siam_ref, name)]
if missing:
    print(f"FAIL: siam_ref missing expected helpers: {missing}", file=sys.stderr)
    print(
        f"  This probe was written against the siam_ref API as of 2026-05-26.",
        file=sys.stderr,
    )
    print(f"  Update siam_ref or this probe script.", file=sys.stderr)
    sys.exit(1)


def preprocess(nii_path: str):
    """Mirror siamize's preprocessing: nibabel -> RAS canonical -> crop ->
    zscore -> resample to 0.75 mm isotropic -> pad to patch. Returns the
    preprocessed volume as fp32 numpy of shape (1, Z, Y, X).
    """
    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)
    d = np.asarray(img.dataobj, dtype=np.float32)
    d = siam_ref.nib_to_zyx(d)[None]  # add channel axis
    spacing = tuple(float(z) for z in reversed(img.header.get_zooms()))
    d, _ = siam_ref.crop_to_nonzero(d)
    d = siam_ref.zscore_normalize(d)
    new_shape = siam_ref.compute_new_shape(d.shape[1:], spacing, (0.75, 0.75, 0.75))
    d = siam_ref.resample_image(d, new_shape, order=3)
    d, _ = siam_ref.pad_to_patch(d, (256, 256, 192))
    return d


def sliding_window(volume_zyx, model_call, patch=(256, 256, 192), step=0.5):
    """Lightweight sliding window. `model_call(x_np)` is a callable that
    runs one tile through whatever backend you want; this function adds
    the tile-coordinate loop and Gaussian-weighted accumulator on top.
    """
    C_out = 18  # SIAM v0.3 class count
    pz, py, px = patch
    Z, Y, X = volume_zyx.shape[1:]
    sz, sy, sx = int(pz * step), int(py * step), int(px * step)

    starts_z = list(range(0, max(1, Z - pz + 1), sz))
    starts_y = list(range(0, max(1, Y - py + 1), sy))
    starts_x = list(range(0, max(1, X - px + 1), sx))

    if not starts_z or starts_z[-1] + pz < Z:
        starts_z.append(max(0, Z - pz))
    if not starts_y or starts_y[-1] + py < Y:
        starts_y.append(max(0, Y - py))
    if not starts_x or starts_x[-1] + px < X:
        starts_x.append(max(0, X - px))

    accum = np.zeros((C_out, Z, Y, X), dtype=np.float32)
    weight = np.zeros((Z, Y, X), dtype=np.float32)

    n_tiles = len(starts_z) * len(starts_y) * len(starts_x)
    print(f"  sliding window: {n_tiles} tiles over (Z={Z}, Y={Y}, X={X})")
    tile_idx = 0
    for z0 in starts_z:
        for y0 in starts_y:
            for x0 in starts_x:
                tile_idx += 1
                patch_np = volume_zyx[:, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px][
                    None
                ].astype(np.float32)
                logits = model_call(patch_np)[0]  # drop batch axis
                accum[:, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += logits
                weight[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += 1.0
                if tile_idx == 1 or tile_idx == n_tiles or tile_idx % 4 == 0:
                    print(f"    tile {tile_idx}/{n_tiles}", flush=True)
    accum /= np.maximum(weight, 1e-6)
    return accum


# ---------------------------------------------------------------- model callables
def make_ort_callable():
    import onnxruntime as ort

    sess = ort.InferenceSession(ONNX_IN, providers=["CPUExecutionProvider"])

    def call(x_np):
        return sess.run(["logits"], {"x": x_np})[0]

    return call


def make_mnn_callable(backend: str):
    import MNN

    interp = MNN.Interpreter(MNN_IN)
    sess = interp.createSession(
        {"backend": backend, "numThread": 0, "precision": "high"}
    )

    def call(x_np):
        in_t = interp.getSessionInput(sess)
        if tuple(in_t.getShape()) != tuple(x_np.shape):
            interp.resizeTensor(in_t, list(x_np.shape))
            interp.resizeSession(sess)
        mnn_t = MNN.Tensor(
            x_np.shape, MNN.Halide_Type_Float, x_np, MNN.Tensor_DimensionType_Caffe
        )
        in_t.copyFrom(mnn_t)
        interp.runSession(sess)
        out = interp.getSessionOutput(sess)
        return np.array(out.getData(), dtype=np.float32).reshape(out.getShape())

    return call


# ---------------------------------------------------------------- main
print(f"Loading + preprocessing {INPUT_NII}", flush=True)
t0 = time.time()
preproc = preprocess(INPUT_NII)
print(f"  preproc done in {time.time() - t0:.1f} s, shape={preproc.shape}")
print()

results = {}

for label, factory in [
    ("ORT CPU", lambda: make_ort_callable()),
    ("MNN CPU", lambda: make_mnn_callable("CPU")),
    ("MNN OpenCL", lambda: make_mnn_callable("OPENCL")),
]:
    print("=" * 60)
    print(label)
    print("=" * 60)
    try:
        call = factory()
    except Exception as e:
        print(f"  SKIP -- backend unavailable: {e}")
        results[label] = None
        print()
        continue

    t0 = time.time()
    logits = sliding_window(preproc, call)
    wall = time.time() - t0
    print(f"  end-to-end wall: {wall:.1f} s")
    print(f"  output shape:    {logits.shape}")
    print(f"  logits range:    [{logits.min():.3f}, {logits.max():.3f}]")
    results[label] = (wall, logits)
    print()

# Decision
print("=" * 60)
print("Performance summary")
print("=" * 60)
for label, val in results.items():
    if val is None:
        print(f"  {label:12} SKIPPED")
    else:
        print(f"  {label:12} {val[0]:7.1f} s")
print()

if results.get("ORT CPU") and results.get("MNN CPU"):
    ratio_mnn_vs_ort = results["MNN CPU"][0] / results["ORT CPU"][0]
    print(f"MNN CPU / ORT CPU: {ratio_mnn_vs_ort:.2f}x")
    if ratio_mnn_vs_ort > 1.5:
        print("SIDE-FAIL: MNN's CPU backend is uncompetitive with ORT's CPU EP.")
        print("           Even if OpenCL speeds it up, the baseline is wrong.")

if results.get("MNN CPU") and results.get("MNN OpenCL"):
    speedup = results["MNN CPU"][0] / results["MNN OpenCL"][0]
    print(f"MNN OpenCL speedup vs MNN CPU: {speedup:.2f}x")
    if speedup >= 5:
        print(
            "VERDICT: STRONG PASS -- worth integrating MNN as a vendor-neutral backend."
        )
    elif speedup >= 2:
        print("VERDICT: MARGINAL -- useful where ORT CUDA isn't available.")
    else:
        print(
            "VERDICT: FAIL -- OpenCL acceleration is too small to justify integration."
        )
        print("         Decomposed Conv3D-as-many-Conv2D dispatch overhead dominates.")

# Numerical agreement (ORT vs MNN OpenCL) sanity check at the end-to-end level
if results.get("ORT CPU") and results.get("MNN OpenCL"):
    y_ort = results["ORT CPU"][1]
    y_ocl = results["MNN OpenCL"][1]
    if y_ort.shape == y_ocl.shape:
        agree = (y_ort.argmax(0) == y_ocl.argmax(0)).mean() * 100
        print(
            f"End-to-end label-volume agreement (ORT CPU vs MNN OpenCL): {agree:.3f}%"
        )
