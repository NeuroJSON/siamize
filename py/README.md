# py/ — slim Python reference

`siam_ref.py` is a ~400-line PyTorch reference implementation of SIAM v0.3
inference, built using only:

- `torch`
- `numpy`
- `nibabel`
- `scipy`
- `scikit-image`
- `dynamic_network_architectures` (the network module from upstream nnU-Net,
  which is itself dependency-clean: only `torch`)

It drops nnU-Net, torchio, batchgenerators, SimpleITK, and the rest of
SIAM's deployment dependencies. Output matches the original `siam-pred`
to 99.989% voxel agreement (5-fold ensemble; see top-level README).

## Usage

```bash
python siam_ref.py \
    -i input.nii.gz \
    -o output.nii.gz \
    --model-dir ~/siam_params/v0.3/pred_DS108_LcsfP_Ano \
    --folds all \
    -v
```

`--folds 0`, `--folds 0,2,4`, or `--folds all`.

## Role in the project

This module is the validation baseline for the ONNX export pipeline
(`tools/onnx_export/`) and the C++ binary (`src/`). If you change the
preprocessing, the gauss map, the resampling kernel, or anything else that
could move the network's output, this is the file to run against the
reference NIfTI in `tests/`.
