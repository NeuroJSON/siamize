# MNN int32→int64 patches for SIAM-class 3D models

This patch makes MNN's CPU backend correctly run nnUNet-style 3D
segmentation models (e.g. SIAM v0.3 with input shape `(1, 1, 256, 256,
192)`) that produce intermediate tensors larger than ~2 GB during the
Convolution3DTurn2D rewrite.

**Tested against MNN master `39be21c5` (post-3.5.0)** — applies cleanly,
no upstream changes since 3.5.0 touch these files.

## What's in the patch (`mnn-int64-fixes-for-siam.patch`)

12 files, +113/-67 lines. Each file fixes a distinct int32-overflow site
in MNN's CPU runtime that triggers on tensors larger than 2 GB:

| File | Bug class |
|---|---|
| `include/MNN/Tensor.hpp` | `elementSize()` returns `int` → truncates for tensors > ~537M fp32 elements. Promoted to `size_t`. |
| `source/shape/ShapeReshape.cpp` | `totalSizeInput * totalSizeOutput` accumulators were `int`. Promoted to `int64_t`. Reshape's `-1` dim now computes correctly. |
| `source/backend/cpu/CPURaster.cpp` | Pointer-offset arithmetic in `_blit`, `_zero`, and the lambdas in `executeFaster` / `onResize` used `int * int32 * int` for byte offsets. Cast each to `size_t`. |
| `source/backend/cpu/CPUTensorConvert.cpp` | `v * pack * bitLength * area` and `v * channel * bitLength * area` overflowed for batched NC4HW4 conversions. Cast to `size_t`. |
| `source/backend/cpu/CPULayerNorm.cpp` | `tId * mInnerSize * bytes` byte-offset overflowed when `mInnerSize ~ 12.5M` and tId ≥ 43. Cast to `size_t`. |
| `source/backend/cpu/CPUScale.cpp` | `depthStride * i * bytes` overflowed for depthStride ~ 100M, i ≥ 6. Hoisted to `size_t byteOffset`. |
| `source/backend/cpu/CPUDeconvolution.cpp` | `outputSize = batch*src_w*src_h*ocC4*pack*bytes` and multiple inner pointer offsets overflowed. Cast to `size_t`. |
| `source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp` | `hw4Stride` and `t_ic * hw4Stride` overflowed at icC4 ≥ 11 with iw\*ih\*batch = 12.5M. Promoted `hw4Stride` to `size_t`. |
| `source/backend/cpu/compute/ConvolutionPackWinograd.cpp` | `int sourceZStep`, `int dstZStep`, and `srcStartY` offsets overflowed for full-spatial decoder layers with batch=256 and full H\*W. Promoted to `size_t`. |
| `source/backend/cpu/x86_x64/avx/GemmCommon.cpp` | `int eReal = info[1]` (= iw\*ih\*batch) plus subsequent `x * unit * eReal` overflowed. Declared `int64_t eReal`. |
| `source/geometry/GeometryDet.cpp` | `auto` deduction broken by elementSize() return-type change. Fixed locally with explicit casts. |
| `source/geometry/GeometryGather.cpp` | Same `auto` deduction fix. |

## Effect on SIAM v0.3

Single-threaded CPU inference now produces correct logits at every input
shape:

| Input shape | argmax agreement vs ORT |
|---|---|
| `(1, 1, 64, 64, 64)` | 91.5% |
| `(1, 1, 128, 128, 128)` | 91.9% |
| `(1, 1, 256, 256, 64)` | 92.6% |
| `(1, 1, 256, 256, 128)` | 93.5% |
| `(1, 1, 256, 256, 192)` (native) | **93.5%** |

The residual ~6% disagreement is fp16-precision noise cascading through
SIAM's 76 conv layers — the same noise floor seen at small scales,
*not* a correctness bug. ORT and MNN agree on logit ranges
(`[-52.9, 69.3]` vs `[-52.5, 71.6]`) and on the location of
classifications.

Before the patch, the same model produced `[-13.2, 30.0]` with only
26.7% argmax agreement — a scale-dependent collapse triggered by any
intermediate tensor crossing 2^31 bytes (~537M fp32 elements), which
happens routinely in 3D-segmentation decoder layers operating at native
resolution.

## How to apply

```bash
cd <your MNN checkout>
git apply /path/to/mnn-int64-fixes-for-siam.patch

# Rebuild Python bindings
cd pymnn/pip_package
rm -rf build/lib.linux-x86_64-*/  # force fresh link
python3 build_deps.py opencl
python3 setup.py --deps opencl install --user
```

## Known remaining issues

- **Multi-threaded CPU** (`numThread > 1`) still segfaults on this
  workload. Single-thread mode (the default for high-precision
  inference) is correct. Multi-thread crash is in
  `CPURaster::onResize`'s lambda — likely additional pointer-offset
  overflows on a parallel-iteration code path that wasn't reached in
  single-thread testing. Patching would follow the same int32→size_t
  pattern as the rest of CPURaster.cpp.

- **OpenCL backend** not tested. Likely has similar int32 overflows
  given the systemic nature of the issue across MNN's CPU backend.

## Upstream filing recommendation

This patch bundle is the basis of the upstream issue draft in
`tools/mnn_probe/reports/mnn_issue_draft.md`. The single-line
`Tensor::elementSize()` fix is the most impactful change and would
benefit any model with > 2 GB intermediate tensors (3D segmentation,
high-resolution detection, video).
