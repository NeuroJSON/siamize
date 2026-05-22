"""
SIAM v0.3 inference using ONNX Runtime instead of PyTorch.

Same preprocessing / sliding-window / postprocessing as siam_ref.py,
but the per-patch forward is delegated to one or more ORT
InferenceSessions. The PyTorch dependency at runtime is gone.

Usage:
    python siam_ort.py -i input.nii.gz -o pred.nii.gz \
        --models models/fold_0_fp16.onnx[,fold_1_fp16.onnx,...] -v
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import nibabel as nib
import numpy as np
import onnxruntime as ort
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Geometry helpers (mirror siam_ref.py)
# ---------------------------------------------------------------------------


def nib_to_zyx(d):
    return np.ascontiguousarray(np.transpose(d, (2, 1, 0)))


def zyx_to_nib(d):
    return np.ascontiguousarray(np.transpose(d, (2, 1, 0)))


def crop_to_nonzero(data: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    mask = data[0] != 0
    for c in range(1, data.shape[0]):
        mask |= data[c] != 0
    mask = binary_fill_holes(mask)
    if not mask.any():
        return data, [[0, s] for s in data.shape[1:]]
    coords = np.where(mask)
    bbox = [[int(c.min()), int(c.max()) + 1] for c in coords]
    sl = tuple(slice(b[0], b[1]) for b in bbox)
    return data[(slice(None),) + sl], bbox


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        m = float(out[c].mean())
        s = max(float(out[c].std()), 1e-8)
        out[c] = (out[c] - m) / s
    return out


def compute_new_shape(old_shape, old_spacing, new_spacing):
    return tuple(
        int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)
    )


def resample_image(data: np.ndarray, new_shape, order: int = 3) -> np.ndarray:
    out = np.zeros((data.shape[0],) + tuple(new_shape), dtype=np.float32)
    for c in range(data.shape[0]):
        out[c] = resize(
            data[c].astype(np.float32),
            new_shape,
            order=order,
            mode="edge",
            anti_aliasing=False,
        )
    return out


def compute_gaussian(
    tile_size, sigma_scale: float = 1.0 / 8, value_scaling: float = 10.0
):
    tmp = np.zeros(tile_size, dtype=np.float32)
    center = tuple(s // 2 for s in tile_size)
    sigmas = [s * sigma_scale for s in tile_size]
    tmp[center] = 1
    g = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    g = g / (g.max() / value_scaling)
    nz = g[g > 0]
    g = np.where(g == 0, float(nz.min()), g).astype(np.float32)
    return g


def compute_steps(image_size, tile_size, step_ratio: float = 0.5):
    target_step = [t * step_ratio for t in tile_size]
    num_steps = [
        int(np.ceil((i - t) / s)) + 1
        for i, s, t in zip(image_size, target_step, tile_size)
    ]
    steps = []
    for d in range(len(tile_size)):
        max_step = image_size[d] - tile_size[d]
        actual = max_step / (num_steps[d] - 1) if num_steps[d] > 1 else 1e12
        steps.append([int(np.round(actual * i)) for i in range(num_steps[d])])
    return steps


def pad_to_patch(data: np.ndarray, patch_size):
    spat = data.shape[1:]
    pad = []
    for s, p in zip(spat, patch_size):
        diff = max(p - s, 0)
        pad.append((diff // 2, diff - diff // 2))
    if all(l == 0 and r == 0 for l, r in pad):
        return data, tuple(slice(0, s) for s in spat)
    padded = np.pad(data, [(0, 0)] + pad, mode="constant", constant_values=0)
    sl = tuple(slice(l, l + s) for (l, _), s in zip(pad, spat))
    return padded, sl


# ---------------------------------------------------------------------------
# Sliding-window with ONNX Runtime
# ---------------------------------------------------------------------------


def make_session(onnx_path: str, providers, num_threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Cap how much memory ORT keeps cached between runs.
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def predict_sliding_window_ort(
    onnx_paths: List[str],
    data: np.ndarray,
    patch_size,
    num_classes: int,
    providers=("CPUExecutionProvider",),
    num_threads: int = 4,
    step_ratio: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    data, revert_pad = pad_to_patch(data, patch_size)
    spat = data.shape[1:]
    steps = compute_steps(spat, patch_size, step_ratio)
    n_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])
    if verbose:
        print(
            f"sliding window: image {spat}, patch {patch_size}, "
            f"step {step_ratio}, {n_tiles} tiles, {len(onnx_paths)} fold(s)"
        )

    import gc

    gauss = compute_gaussian(patch_size).astype(np.float32)

    # 18-channel logits accumulator in fp16 (matches the original PyTorch
    # sliding-window) to keep this big buffer small. Weights stay fp32
    # because gauss values at the patch periphery underflow to 0 in fp16,
    # which would produce divide-by-zero at the normalization step.
    logits = np.zeros((num_classes,) + tuple(spat), dtype=np.float16)
    weights = np.zeros(spat, dtype=np.float32)

    for fi, onnx_path in enumerate(onnx_paths):
        sess = make_session(onnx_path, list(providers), num_threads)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        if verbose:
            print(
                f"  fold {fi+1}/{len(onnx_paths)}: {os.path.basename(onnx_path)}",
                flush=True,
            )

        idx = 0
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    sl_data = (
                        slice(None),
                        slice(sx, sx + patch_size[0]),
                        slice(sy, sy + patch_size[1]),
                        slice(sz, sz + patch_size[2]),
                    )
                    tile = data[sl_data][None].astype(np.float32, copy=False)
                    pred = sess.run([out_name], {in_name: tile})[0][0]  # (C,Z,Y,X) fp32
                    pred *= gauss  # fp32 in-place
                    logits[sl_data] += pred.astype(np.float16, copy=False)
                    if fi == 0:
                        weights[sl_data[1:]] += gauss
                    del tile, pred
                    idx += 1
                    if verbose and idx % 4 == 0:
                        print(f"    tile {idx}/{n_tiles}", flush=True)

        del sess  # release session memory before loading next fold
        gc.collect()

    if not np.isfinite(logits).all():
        raise RuntimeError("non-finite values in logits")
    # cast back to fp32 for the final divide so post-resample is well-conditioned
    logits = logits.astype(np.float32) / (weights * len(onnx_paths))
    logits = logits[(slice(None),) + revert_pad]
    return logits


def argmax_labels(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=0).astype(np.uint8)


def insert_into_bbox(labels, bbox, full_shape):
    out = np.zeros(full_shape, dtype=labels.dtype)
    out[tuple(slice(b[0], b[1]) for b in bbox)] = labels
    return out


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run(
    input_path,
    output_path,
    onnx_paths,
    dataset_json_path,
    plans_path,
    providers,
    num_threads,
    verbose=False,
):
    with open(plans_path) as f:
        plans = json.load(f)
    with open(dataset_json_path) as f:
        ds = json.load(f)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = tuple(cfg["patch_size"])
    target_spacing = tuple(cfg["spacing"])
    num_classes = len(ds["labels"])

    t0 = time.time()
    img = nib.load(input_path)
    img_canon = nib.as_closest_canonical(img)
    data_xyz = np.asarray(img_canon.dataobj, dtype=np.float32)
    if verbose:
        print(f"input: {input_path}")
        print(
            f"  shape XYZ: {data_xyz.shape}, zooms XYZ: {img_canon.header.get_zooms()}"
        )
        print(f"  orient: {nib.aff2axcodes(img.affine)} -> RAS")

    data = nib_to_zyx(data_xyz)[None]
    original_spacing = tuple(float(z) for z in reversed(img_canon.header.get_zooms()))

    shape_before_cropping = data.shape[1:]
    data, bbox = crop_to_nonzero(data)
    shape_after_cropping = data.shape[1:]
    if verbose:
        print(f"  bbox crop: {shape_before_cropping} -> {shape_after_cropping}")

    data = zscore_normalize(data)

    new_shape = compute_new_shape(
        shape_after_cropping, original_spacing, target_spacing
    )
    if verbose:
        print(
            f"  resample {shape_after_cropping}@{original_spacing} -> {new_shape}@{target_spacing}"
        )
    data_res = resample_image(data, new_shape, order=3)

    logits = predict_sliding_window_ort(
        onnx_paths,
        data_res,
        patch_size,
        num_classes,
        providers=providers,
        num_threads=num_threads,
        step_ratio=0.5,
        verbose=verbose,
    )

    if logits.shape[1:] != shape_after_cropping:
        if verbose:
            print(
                f"  resample logits back {logits.shape[1:]} -> {shape_after_cropping}"
            )
        logits = resample_image(logits, shape_after_cropping, order=1)

    labels = argmax_labels(logits)
    labels_full = insert_into_bbox(labels, bbox, shape_before_cropping)
    labels_xyz = zyx_to_nib(labels_full)

    if not np.allclose(img.affine, img_canon.affine):
        canon_lbl = nib.Nifti1Image(labels_xyz, img_canon.affine)
        from nibabel.processing import resample_from_to

        out_img = resample_from_to(canon_lbl, img, order=0)
        out_img.set_data_dtype(np.uint8)
    else:
        out_img = nib.Nifti1Image(labels_xyz.astype(np.uint8), img.affine, img.header)
        out_img.set_data_dtype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    nib.save(out_img, output_path)
    if verbose:
        print(f"saved {output_path} in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument(
        "--models",
        required=True,
        help="comma-separated list of .onnx model files (one per fold)",
    )
    ap.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/siam_params/v0.3/pred_DS108_LcsfP_Ano"),
        help="Original SIAM model dir (used only to read plans.json + dataset.json)",
    )
    ap.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="comma-separated ORT providers, e.g. CUDAExecutionProvider,CPUExecutionProvider",
    )
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    onnx_paths = [p.strip() for p in args.models.split(",") if p.strip()]
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    run(
        args.input,
        args.output,
        onnx_paths,
        dataset_json_path=os.path.join(args.model_dir, "dataset.json"),
        plans_path=os.path.join(args.model_dir, "plans.json"),
        providers=providers,
        num_threads=args.threads,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
