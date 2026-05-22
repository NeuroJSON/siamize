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
single ORT forward on (a) random noise and (b) a centered brain patch from
`tests/sub-01_T1w.nii.gz`, then reports max/mean abs-diff and argmax
agreement. Real-brain argmax agreement should be > 99% for fp32 and > 98.5%
for fp16.

If argmax agreement is low: the most likely cause is an opset mismatch on
a 3D op. Bump `--opset` or check `pip show onnx onnxruntime`.
