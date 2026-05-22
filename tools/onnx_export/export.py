"""
Export one SIAM fold to ONNX.

Usage:
    python export.py --fold 0 --out models/fold_0_fp32.onnx [--fp16]

Validation flow:
1. Build the PyTorch network (deep_supervision=False) with the chosen fold's weights.
2. Trace via torch.onnx.export on a fixed-shape patch (1, 1, 256, 256, 192).
3. Optionally cast to fp16 via onnxconverter-common.
4. Compare PyTorch output vs onnxruntime output on a random patch.
"""

import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
from siam_ref import build_network, load_fold_state_dict, load_plans_and_dataset


def export_fp32(model_dir: str, fold: int, out_path: str, opset: int = 17) -> tuple:
    plans, ds = load_plans_and_dataset(model_dir)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = tuple(cfg["patch_size"])
    in_ch = len(ds["channel_names"])
    num_classes = len(ds["labels"])

    net = build_network(plans, ds, deep_supervision=False)
    net.load_state_dict(load_fold_state_dict(model_dir, fold), strict=True)
    net.eval()

    dummy = torch.randn(1, in_ch, *patch_size)
    print(
        f"exporting fold_{fold} -> {out_path} (input shape {tuple(dummy.shape)})",
        flush=True,
    )
    t0 = time.time()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            net,
            (dummy,),
            out_path,
            input_names=["x"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
        )
    print(
        f"  ONNX written in {time.time() - t0:.1f}s, size {os.path.getsize(out_path) / 1e6:.1f} MB"
    )

    return patch_size, in_ch, num_classes, net


def cast_to_fp16(in_path: str, out_path: str):
    import onnx
    from onnxconverter_common import float16

    print(f"casting {in_path} -> {out_path} (fp16)", flush=True)
    model = onnx.load(in_path)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # keep input/output in fp32 for ergonomics
        disable_shape_infer=False,
    )
    onnx.save(model_fp16, out_path)
    print(f"  fp16 size {os.path.getsize(out_path) / 1e6:.1f} MB")


def _make_random_patch(patch_size, in_ch, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, in_ch, *patch_size)).astype(np.float32)


def _make_real_brain_patch(patch_size, in_ch):
    """Extract a centered preprocessed patch from the bundled sub-01_T1w."""
    sample = os.path.join(
        os.path.dirname(__file__), "..", "..", "tests", "sub-01_T1w.nii.gz"
    )
    sample = os.path.abspath(sample)
    if not os.path.isfile(sample):
        return None
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
    from siam_ref import (
        nib_to_zyx,
        crop_to_nonzero,
        zscore_normalize,
        compute_new_shape,
        resample_image,
        pad_to_patch,
    )

    img = nib.load(sample)
    img = nib.as_closest_canonical(img)
    d = np.asarray(img.dataobj, dtype=np.float32)
    d = nib_to_zyx(d)[None]
    spc = tuple(float(z) for z in reversed(img.header.get_zooms()))
    d, _ = crop_to_nonzero(d)
    d = zscore_normalize(d)
    new_shape = compute_new_shape(d.shape[1:], spc, (0.75, 0.75, 0.75))
    d = resample_image(d, new_shape, order=3)
    d, _ = pad_to_patch(d, patch_size)
    # Center-crop a single patch
    spat = d.shape[1:]
    starts = [(s - p) // 2 for s, p in zip(spat, patch_size)]
    sl = (slice(None),) + tuple(slice(s, s + p) for s, p in zip(starts, patch_size))
    return d[sl][None].astype(np.float32, copy=False)


def verify_parity(
    net: torch.nn.Module,
    onnx_path: str,
    patch_size,
    in_ch: int,
    argmax_floor: float = 0.99,
    label: str = "",
):
    """
    Compare PyTorch (fp32) reference vs ONNX Runtime on (a) random noise and
    (b) a real preprocessed brain patch. Only argmax agreement is asserted —
    raw logit drift on random noise can be large for big nets due to MLAS vs
    oneDNN kernel-level rounding differences; what matters is the final
    segmentation decision.
    """
    import onnxruntime as ort

    print(
        f"loading ORT session on CPUExecutionProvider for parity check {label}",
        flush=True,
    )
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(
        f"  inputs: {[i.name for i in sess.get_inputs()]}, "
        f"outputs: {[o.name for o in sess.get_outputs()]}"
    )
    net = net.eval()

    cases = [("random noise", _make_random_patch(patch_size, in_ch, seed=0))]
    real = _make_real_brain_patch(patch_size, in_ch)
    if real is not None:
        cases.append(("real brain patch", real))

    # NOTE: run PyTorch BEFORE ORT in each pair. With fp16 ORT sessions held
    # in memory, a subsequent PyTorch fwd on the same big patch can stall for
    # minutes (suspected allocator contention via oneDNN's thread pool).
    worst_agreement = 1.0
    for name, x_np in cases:
        with torch.inference_mode():
            y_torch = net(torch.from_numpy(x_np)).numpy()
        y_ort = sess.run(["logits"], {"x": x_np})[0]
        diff = np.abs(y_torch - y_ort)
        argmax_t = y_torch.argmax(axis=1)
        argmax_o = y_ort.argmax(axis=1)
        agreement = float((argmax_t == argmax_o).mean())
        worst_agreement = min(worst_agreement, agreement)
        print(
            f"  [{name}] max|diff|={diff.max():.3e}, mean|diff|={diff.mean():.3e}, "
            f"argmax agreement {agreement*100:.3f}%",
            flush=True,
        )

    assert worst_agreement >= argmax_floor, (
        f"parity check failed: argmax agreement {worst_agreement*100:.3f}% < "
        f"{argmax_floor*100:.3f}% (label={label})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/siam_params/v0.3/pred_DS108_LcsfP_Ano"),
    )
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", required=True, help="output .onnx path (fp32)")
    ap.add_argument(
        "--fp16-out",
        default=None,
        help="if set, also write fp16-cast .onnx to this path",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip ORT parity check (saves a few minutes per fold)",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--skip-export",
        action="store_true",
        help="don't re-trace; reuse the existing .onnx (still need fold/model-dir for parity)",
    )
    args = ap.parse_args()

    if args.skip_export:
        plans, ds = load_plans_and_dataset(args.model_dir)
        cfg = plans["configurations"]["3d_fullres"]
        patch_size = tuple(cfg["patch_size"])
        in_ch = len(ds["channel_names"])
        net = build_network(plans, ds, deep_supervision=False)
        net.load_state_dict(
            load_fold_state_dict(args.model_dir, args.fold), strict=True
        )
        net.eval()
    else:
        patch_size, in_ch, num_classes, net = export_fp32(
            args.model_dir, args.fold, args.out, opset=args.opset
        )
        if args.fp16_out:
            cast_to_fp16(args.out, args.fp16_out)

    if not args.no_verify:
        print("\n=== parity check: fp32 ONNX ===")
        verify_parity(net, args.out, patch_size, in_ch, argmax_floor=0.99, label="fp32")
        if args.fp16_out:
            print("\n=== parity check: fp16 ONNX ===")
            verify_parity(
                net, args.fp16_out, patch_size, in_ch, argmax_floor=0.985, label="fp16"
            )


if __name__ == "__main__":
    main()
