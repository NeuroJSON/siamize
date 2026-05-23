# tools/onnx_export/

PyTorch → ONNX export pipeline for SIAM v0.3.

## Scripts

| File | Purpose |
|---|---|
| `export.py`         | Export one fold to fp32 ONNX, optionally cast to fp16, optionally run parity check against PyTorch on a real brain patch. |
| `export_all_folds.sh` | Wrapper around `export.py` to do all 5 folds and drop the fp32 intermediates. |
| `siam_ort.py`       | End-to-end inference using ONNX Runtime instead of PyTorch — useful for cross-checking the exported models without rebuilding the C++ binary. |
| `compare.py`        | Voxel + Dice comparison between two NIfTI label volumes. Used by `tests/run_regression.sh`. |

## Typical workflow

```bash
# from repo root
bash tools/onnx_export/export_all_folds.sh
# → produces models/fold_{0..4}_fp16.onnx (~270 MB each, ~1.35 GB total)

# Sanity check the exported model end-to-end:
python tools/onnx_export/siam_ort.py \
    -i tests/sub-01_T1w.nii.gz \
    -o /tmp/pred_ort.nii.gz \
    --models models/fold_0_fp16.onnx,models/fold_1_fp16.onnx,models/fold_2_fp16.onnx,models/fold_3_fp16.onnx,models/fold_4_fp16.onnx \
    --threads 8 -v

python tools/onnx_export/compare.py \
    --ref tests/pred_ref_allfolds.nii.gz \
    --new /tmp/pred_ort.nii.gz
```

## Parity check

`export.py` (without `--no-verify`) runs a single PyTorch forward and a
single ORT forward on (a) random noise at the default patch size,
(b) a centered brain patch from `tests/sub-01_T1w.nii.gz`, and (c) a
half-size random patch (to exercise the dynamic spatial axes baked
into the export), then reports max/mean abs-diff and argmax agreement
per case. Real-brain argmax agreement should be > 99% for fp32 and >
98.5% for fp16.

If argmax agreement is low: the most likely cause is an opset mismatch on
a 3D op. Bump `--opset` or check `pip show onnx onnxruntime`.

## Input shape contract

The exported ONNX has **dynamic batch + spatial axes** (`x` and `logits`
are both `[batch, channel, D, H, W]` with `batch`/`D`/`H`/`W` symbolic).
Concrete shape requirements come from the SIAM v0.3 plans.json stride
pattern `[[1,1,1], [2,2,2]×5, [2,2,1]]`:

- Channel dim is fixed at 1 (single-modality T1w in SIAM v0.3).
- **Z and Y** must each be a multiple of **64** (six stride-2 stages).
- **X** must be a multiple of **32** (the final stage's X-stride is 1).
- Mixing different multiples across axes works — only the per-axis
  divisor matters.

Common useful sizes (`Z x Y x X` — these are the ones siamize accepts):

| `--patch` | Notes |
|---|---|
| `256x256x192` | Training plan; identical to the original SIAM. Default. |
| `192x192x192` | -23% VRAM, virtually no accuracy cost. |
| `192x192x160` | -32% VRAM. |
| `192x192x128` | -44% VRAM. Fits ~7 GB free on a laptop 4070. |
| `128x128x128` | -75% VRAM. Tight-fit configurations. |
| `128x128x96`  | -78% VRAM. The smallest validated in our parity test. |

Sizes that look reasonable but **don't** work because they break the
divisibility constraint and the network's residual Adds disagree on
spatial dims by 1: `224x224x160` (Z/Y not /64), `160x160x128`
(Z/Y not /64), anything with Z or Y in {64+32, 192+32, 224, ...}.
