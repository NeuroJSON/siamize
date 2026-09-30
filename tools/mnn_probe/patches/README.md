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
shape. Synthetic-Gaussian input (the worst case — every voxel is a
boundary):

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

**End-to-end on a real T1 brain volume** (`tests/sub-01_T1w.nii.gz`,
full siamize sliding-window pipeline, single fold), MNN vs ORT:

- **Voxel-level argmax agreement: 99.67 %**
- **Mean foreground Dice: 0.992**
- Per-class Dice ranges from 0.972 (vascular) to 1.000 (anomalies)
- MNN-vs-PyTorch-reference Dice (0.928) is within 0.001 of
  ORT-vs-PyTorch-reference Dice (0.929) — MNN adds essentially no
  error beyond what ORT already adds.

The synthetic-Gaussian 93.5 % was a worst-case diagnostic; real brain
anatomy is dominated by easy interior voxels and only the <0.4 %
boundary fringe sees fp16 drift.

**OpenCL backend** (NVIDIA OpenCL ICD on A100, no additional patches
needed — the CPU patches in this directory are sufficient):

- Correctness: 93.5 % argmax / matches ORT logit range, GPU sm-util
  actually hits 100 % (no silent CPU fallback).
- Performance: 65 s warm wall vs ORT-CPU 27 s — *slower* than CPU on
  this many-core host. The Conv3D-to-Conv2D decomposition produces a
  3382-op graph whose OpenCL kernel-launch overhead dominates wall
  time. ORT CUDA on the same A100 is ~6× faster than MNN OpenCL.

Per-host verification reports are in `tools/mnn_probe/reports/`
(gitignored).

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

- **Multi-threaded CPU** (`numThread > 1`) still segfaults on SIAM-class
  workloads. The patch includes a defensive workaround in
  `CPURaster::onResize` that demotes any Raster op with a per-stride
  byte step > 16 MB to single-thread, but this does not fully eliminate
  the crash; the failing call stack is consistently
  `_blit → memmove` inside `CPURaster::onResize::lambda#4`, even with
  `mUseThreads` forcibly cleared. The root cause is therefore not in
  the lambda's per-thread offset arithmetic (extensively audited and
  patched) but somewhere else triggered by `numThread > 1` in the
  parent backend's setup. **Workaround: pass `numThread=1` or `2`** in
  `createSession`. Both values produce correct output and complete
  inference; `numThread=2` is also ~1.5× faster than `numThread=1`.

  Investigation summary of MT crash (for upstream MNN maintainers):
  - The crash reproduces on the minimal `dec_272` truncated SIAM
    (`/decoder/stages.5` Conv at full spatial 256×256×192).
  - `numThread=1, 2`: no crash, output correct.
  - `numThread=4, 8`: SIGSEGV in `_blit`'s inner `memcpy`, *even when*
    the workaround forces every Raster op in this op-chain to
    single-thread via `mUseThreads=false`. The lambda's captured
    `threadNum=1` was verified with instrumentation.
  - Audit-and-patch passes on every visible int32 byte-offset
    arithmetic site (CPURaster offsets, AVX pack `eReal`, Winograd
    zSteps, CPULayerNorm `tId * mInnerSize`, CPUScale `depthStride*i`,
    CPUDeconvolution `outputSize`, CPUTensorConvert offsets) did *not*
    eliminate the multi-thread failure.
  - The most likely remaining cause is in a path that isn't reached
    when `threadNumber == 1` from the start, but is reached when
    `threadNumber == 4` even after every per-op lambda capture is
    demoted — possibly thread-pool initialization, or a static state
    in a shared kernel.

- **OpenCL backend** not tested. Likely has similar int32 overflows
  given the systemic nature of the issue across MNN's CPU backend.

## Upstream filing recommendation

This patch bundle is the basis of the upstream issue draft in
`tools/mnn_probe/reports/mnn_issue_draft.md`. The single-line
`Tensor::elementSize()` fix is the most impactful change and would
benefit any model with > 2 GB intermediate tensors (3D segmentation,
high-resolution detection, video).

# Vulkan native Conv3D in the NeuroJSON/MNN fork

Branch `siam-vulkan-conv3d`, tag `v3.5-vulkan-conv3d`, six commits on top of
`v3.5-opencl-conv3d` (siamize's default `MNN_REF` is now `v3.5-gpu-opt`, below). Only the Vulkan
backend changes; build it with `MNN_VULKAN=1 scripts/fetch_mnn.sh`
(`-DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF`).

| Commit | Change |
|---|---|
| `[Vulkan:Perf] allocate device memory from a DEVICE_LOCAL type` | `VulkanMemoryPool` took the first allowed memory type; on NVIDIA's driver that is a flag-less type on the system-RAM heap, so every tensor lived across PCIe (~40x slower end-to-end). |
| `[Vulkan:Bugfix] Scale / PRelu: cover every spatial dim of 5D tensors` | Plane size was `width() * height() * batch()`, which drops W for (N, C, D, H, W) -- wrong InstanceNorm affine in every block. |
| `[Vulkan:Bugfix] report device out-of-memory instead of crashing` | Failed `vkAllocateMemory` now propagates as `OUT_OF_MEMORY` from resize (it used to bind a null allocation and segfault in the driver); `VulkanRaster` temp sizes `int` -> `size_t` (3.2 GB concat at 256x256x192). |
| `[Vulkan:Feature] MNN_VULKAN_DEVICE selects the physical device` | `sharedContext` is read as a full `MNNVulkanContext`, so it cannot carry an index. |
| `[Vulkan:Feature] native Conv3D / ConvTranspose3D for the buffer backend` | `VulkanConvolution3D.cpp` + `glsl/conv3d.comp`: one shader for conv and gather-form transposed conv; kernel / stride / pad / NC4HW4-vs-NCHW layouts / tiling are specialization constants; `OC_TILE` x 4 channels x `W_TILE` interleaved voxels per invocation, fp32 accumulation; weights in device-local memory. `compiler/splice_shader.py` adds one shader to the generated tables without regenerating all 8 MB (rerun `python3 splice_shader.py conv3d.comp` after editing the shader). |
| `[Vulkan:Perf] fast host<->device copies: transfer queue, cached staging` | `HOST_CACHED` staging (CPU reads of write-combined memory ran at ~0.65 GB/s), threaded memcpy / fp16 conversion for copies >= 16 MB, copies on a dedicated transfer queue (RTX 5090: 5.7 GB/s on the graphics/compute queue vs 28.4 GB/s on the copy engines; readback 79 -> 24 ms per 340 MB tile), `CONCURRENT` buffer sharing across the two families, and an opt-in `VK_EXT_external_memory_host` path (`MNN_VULKAN_HOST_IMPORT=1`; per-copy page pinning makes it slower than staging on the GPUs tested). |

Every commit builds on its own. Validation (`tools/mnn_probe/layercheck.cpp`,
fold 0, random N(0,1) input, 128^3, fp32): all 529 op outputs within 2e-3
relative of MNN CPU, final argmax agreement 99.975 % on TITAN V and on the AMD
Raphael iGPU (RADV). End-to-end on `sub-01_T1w`: 99.9999 % voxel agreement
with MNN-OpenCL, 99.99 % with ORT-CUDA, bit-identical across TITAN V and
RTX 5090. `VK_LAYER_KHRONOS_validation` (errors, warnings, perf,
synchronization validation) reports nothing, with and without host import.

# GPU kernel optimizations (`v3.5-gpu-opt`, siamize's default `MNN_REF`)

Four more commits on `siam-vulkan-conv3d`, from profiling one SIAM v0.3 tile
(192x192x128) on an RTX 5090. `tools/mnn_probe/clbench.cpp` benchmarks OpenCL
Conv3D kernels on every SIAM layer shape and dumps their PTX.

| Commit | Change |
|---|---|
| `[OpenCL:Perf] Conv3D: FLOAT4-typed, register-blocked kernel (1.9x)` | The `conv_3d_buf` kernels read `FLOAT*` through `vload4`; NVIDIA's compiler can only assume 4-byte alignment and emits four 32-bit loads per vector (no `ld.global.v4` in the PTX), capping the inner loop at ~18 TFLOP/s. `conv_3d_buf_opt` types the buffers `FLOAT4` + `restrict` and register-blocks 4x2 / 2x2 / 1x1 output voxels per work-item by layer size (~43 TFLOP/s on the large layers). Conv3D 211 -> 111 ms per tile. `MNN_CONV3D_LEGACY=1` keeps the old kernels. |
| `[OpenCL:Perf] LayerNorm: split few very long rows across work-groups` | One work-group per row meant 32 work-groups for InstanceNorm at full resolution. Rows >= 64k values now run as per-chunk statistics, a per-row Chan merge and a parallel normalize: 16.4 -> 7.5 ms per tile. `MNN_LAYERNORM_NOSPLIT=1` keeps the old kernel. |
| `[Vulkan:Perf] Conv3D: 3D work-groups spanning OC blocks, H tile` | The flat 1D launch shared one output-channel block per work-group, so each input voxel was re-fetched per block. 3D launch with host-chosen group shape (groups span 8 OC blocks) and an H tile: 0.272 -> 0.230 s per tile (fp32). Slower on a TITAN V with fp16 (0.76 -> 0.90 s), where the flat mapping suited Volta better. |
| `[Vulkan:Perf] LayerNorm: split few very long rows across work-groups` | Same split for `norm_opt` (one 64-invocation group per row): `glsl/norm_split.comp`, three dispatches with buffer barriers. 0.231 -> 0.218 s per tile. |

Every commit builds on its own. All 529 op outputs stay within 2e-3 of the CPU
backend on both GPU backends; end-to-end on `sub-01_T1w` the labels match the
previous kernels on 99.9999 % (fp32) of voxels. One fold on the RTX 5090:
OpenCL 13.6 -> 10.7 s, Vulkan 14.6 -> 13.2 s (fp32) / 13.2 -> 12.6 s (fp16).
