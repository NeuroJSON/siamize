"""
SIAM v0.3 slim reference inference.

Dependencies: torch, numpy, nibabel, scipy, skimage,
              dynamic_network_architectures (PyTorch-only, no nnUNet).

Mirrors nnUNetPredictor's behaviour for SIAM v0.3 closely enough that
output should match the original `siam-pred` pipeline within numerical
noise on isotropic 3D fullres inputs.
"""

import argparse
import json
import os
import pydoc
import sys
import time
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.transform import resize

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _resolve_class(s):
    return pydoc.locate(s) if isinstance(s, str) else s


def build_network(plans: dict, dataset_json: dict, deep_supervision: bool = False):
    cfg = plans["configurations"]["3d_fullres"]
    arch = cfg["architecture"]
    kwargs = dict(arch["arch_kwargs"])
    for k in arch["_kw_requires_import"]:
        if kwargs.get(k) is not None:
            kwargs[k] = _resolve_class(kwargs[k])
    num_classes = len(dataset_json["labels"])
    input_channels = len(dataset_json["channel_names"])
    return ResidualEncoderUNet(
        input_channels=input_channels,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        **kwargs,
    )


def load_plans_and_dataset(model_dir: str):
    with open(os.path.join(model_dir, "plans.json")) as f:
        plans = json.load(f)
    with open(os.path.join(model_dir, "dataset.json")) as f:
        dataset_json = json.load(f)
    return plans, dataset_json


def load_fold_state_dict(model_dir: str, fold: int) -> dict:
    ck_path = os.path.join(model_dir, f"fold_{fold}", "checkpoint_final.pth")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["network_weights"]
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def load_fold(model_dir: str, fold: int, device: torch.device):
    """Convenience wrapper: build + load one fold onto device."""
    plans, dataset_json = load_plans_and_dataset(model_dir)
    net = build_network(plans, dataset_json, deep_supervision=False)
    net.load_state_dict(load_fold_state_dict(model_dir, fold), strict=True)
    net.to(device).eval()
    return net, plans, dataset_json


# ---------------------------------------------------------------------------
# Geometry: nibabel (X,Y,Z) <-> SITK / nnunet (Z,Y,X)
# ---------------------------------------------------------------------------


def nib_to_zyx(data_xyz: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(data_xyz, (2, 1, 0)))


def zyx_to_nib(data_zyx: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(data_zyx, (2, 1, 0)))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def crop_to_nonzero(data: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    """data: (C, Z, Y, X). Returns cropped data + bbox = [[lo,hi]*3]."""
    mask = data[0] != 0
    for c in range(1, data.shape[0]):
        mask |= data[c] != 0
    mask = binary_fill_holes(mask)
    if not mask.any():
        bbox = [[0, s] for s in data.shape[1:]]
        return data, bbox
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
    """data: (C, ...). Linear/cubic resize per channel."""
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


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------


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
    assert all(
        i >= t for i, t in zip(image_size, tile_size)
    ), f"image {image_size} smaller than patch {tile_size}"
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


def pad_to_patch(data: np.ndarray, patch_size) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    spat = data.shape[1:]
    pad = []
    for s, p in zip(spat, patch_size):
        diff = max(p - s, 0)
        left = diff // 2
        pad.append((left, diff - left))
    if all(l == 0 and r == 0 for l, r in pad):
        return data, tuple(slice(0, s) for s in spat)
    padded = np.pad(data, [(0, 0)] + pad, mode="constant", constant_values=0)
    sl = tuple(slice(l, l + s) for (l, _), s in zip(pad, spat))
    return padded, sl


@torch.inference_mode()
def predict_sliding_window(
    model_dir: str,
    folds: List[int],
    plans: dict,
    dataset_json: dict,
    data: np.ndarray,
    patch_size,
    num_classes: int,
    device: torch.device,
    step_ratio: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Fold-major sliding-window inference: loads one fold to GPU at a time.

    data: (C, Z, Y, X) float32 already normalized.
    Returns logits: (num_classes, Z, Y, X) float32 numpy.
    """
    data, revert_pad = pad_to_patch(data, patch_size)
    spat = data.shape[1:]
    steps = compute_steps(spat, patch_size, step_ratio)
    n_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])
    if verbose:
        print(
            f"sliding window: image {spat}, patch {patch_size}, "
            f"step {step_ratio}, {n_tiles} tiles, {len(folds)} fold(s)"
        )

    data_t = torch.from_numpy(data).to(device, dtype=torch.float32)
    gauss_np = compute_gaussian(patch_size)
    gauss = torch.from_numpy(gauss_np).to(device, dtype=torch.float16)

    logits = torch.zeros(
        (num_classes,) + tuple(spat), dtype=torch.float16, device=device
    )
    weights = torch.zeros(spat, dtype=torch.float16, device=device)

    autocast = device.type == "cuda"
    cm_outer = torch.autocast(device.type, enabled=True) if autocast else _noop_ctx()

    # Build the network shell once; reload weights per fold.
    net = build_network(plans, dataset_json, deep_supervision=False)
    net.to(device).eval()

    for fi, fold in enumerate(folds):
        sd = load_fold_state_dict(model_dir, fold)
        net.load_state_dict(sd, strict=True)
        if verbose:
            print(f"  fold {fold} loaded", flush=True)

        idx = 0
        with cm_outer:
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        sl = (
                            slice(None),
                            slice(sx, sx + patch_size[0]),
                            slice(sy, sy + patch_size[1]),
                            slice(sz, sz + patch_size[2]),
                        )
                        tile = data_t[sl].unsqueeze(0).contiguous()
                        out = net(tile)
                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        out = out[0].to(dtype=torch.float16) * gauss
                        logits[sl] += out
                        if fi == 0:
                            weights[sl[1:]] += gauss
                        idx += 1
                        if verbose and idx % 8 == 0:
                            print(f"    tile {idx}/{n_tiles}", flush=True)
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if torch.any(torch.isinf(logits)):
        raise RuntimeError(
            "inf in logits — try lowering value_scaling or fp32 accumulator"
        )
    # ensemble: divide by (per-tile gauss-weight sum) * num_folds
    logits = (logits / (weights * len(folds))).float()
    logits = logits[(slice(None),) + revert_pad]
    return logits.cpu().numpy()


class _noop_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def argmax_labels(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=0).astype(np.uint8)


def insert_into_bbox(labels: np.ndarray, bbox, full_shape) -> np.ndarray:
    out = np.zeros(full_shape, dtype=labels.dtype)
    sl = tuple(slice(b[0], b[1]) for b in bbox)
    out[sl] = labels
    return out


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run(
    input_path: str,
    output_path: str,
    model_dir: str,
    folds: List[int],
    device: torch.device,
    verbose: bool = False,
):
    t0 = time.time()

    plans, ds = load_plans_and_dataset(model_dir)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = tuple(cfg["patch_size"])  # (256, 256, 192) [Z, Y, X]
    target_spacing = tuple(cfg["spacing"])  # (0.75, 0.75, 0.75) [Z, Y, X]
    num_classes = len(ds["labels"])

    # Load NIfTI (X,Y,Z) and reorient to RAS canonical
    img = nib.load(input_path)
    img_canon = nib.as_closest_canonical(img)
    data_xyz = np.asarray(img_canon.dataobj, dtype=np.float32)
    affine_canon = img_canon.affine.copy()
    zooms_canon_xyz = img_canon.header.get_zooms()  # (sx, sy, sz)
    if verbose:
        print(f"input: {input_path}")
        print(f"  shape XYZ: {data_xyz.shape}, zooms XYZ: {zooms_canon_xyz}")
        print(f"  orient: {nib.aff2axcodes(img.affine)} -> RAS")

    # Convert to (C, Z, Y, X) and spacing (z, y, x) to match SimpleITK/nnunet convention
    data = nib_to_zyx(data_xyz)[None]  # (1, Z, Y, X)
    original_spacing = tuple(float(z) for z in reversed(zooms_canon_xyz))  # (z, y, x)

    shape_before_cropping = data.shape[1:]
    data, bbox = crop_to_nonzero(data)
    shape_after_cropping = data.shape[1:]
    if verbose:
        print(f"  bbox crop: {shape_before_cropping} -> {shape_after_cropping}")

    # Normalize BEFORE resampling (nnunet order)
    data = zscore_normalize(data)

    # Resample to target spacing
    new_shape = compute_new_shape(
        shape_after_cropping, original_spacing, target_spacing
    )
    if verbose:
        print(
            f"  resample {shape_after_cropping}@{original_spacing} -> {new_shape}@{target_spacing}"
        )
    data_res = resample_image(data, new_shape, order=3)

    # Sliding-window inference (fold-major to keep GPU memory low)
    logits = predict_sliding_window(
        model_dir,
        folds,
        plans,
        ds,
        data_res,
        patch_size,
        num_classes,
        device,
        step_ratio=0.5,
        verbose=verbose,
    )

    # Resample logits back to pre-resample (cropped) shape with order=1
    if logits.shape[1:] != shape_after_cropping:
        if verbose:
            print(
                f"  resample logits back {logits.shape[1:]} -> {shape_after_cropping}"
            )
        logits = resample_image(logits, shape_after_cropping, order=1)

    # Argmax → label volume; un-crop into original-pre-crop shape
    labels = argmax_labels(logits)
    labels_full = insert_into_bbox(labels, bbox, shape_before_cropping)

    # Back to (X, Y, Z) for nibabel
    labels_xyz = zyx_to_nib(labels_full)

    # If the input was not RAS, reorient labels back to the original orientation
    if not np.allclose(img.affine, img_canon.affine):
        # Build a NIfTI in canonical space, then resample (NN) into the original geometry
        canon_lbl = nib.Nifti1Image(labels_xyz, affine_canon)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_folds(s: str) -> List[int]:
    s = s.strip().lower()
    if s == "all":
        return [0, 1, 2, 3, 4]
    return [int(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/siam_params/v0.3/pred_DS108_LcsfP_Ano"),
    )
    p.add_argument("--folds", default="0", help="comma-separated folds or 'all'")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
        if args.device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    folds = _parse_folds(args.folds)
    print(f"device: {dev}, folds: {folds}", flush=True)
    run(args.input, args.output, args.model_dir, folds, dev, verbose=args.verbose)


if __name__ == "__main__":
    main()
