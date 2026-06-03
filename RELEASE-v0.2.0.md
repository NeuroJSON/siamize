# siamize v0.2.0 — initial release

*native C++/ONNX port of SIAM v0.3 brain/head MRI segmentation*

* **Copyright**: (C) Qianqian Fang (2026) \<q.fang at neu.edu>
* **License**: Apache License, Version 2.0
* **GitHub**: https://github.com/NeuroJSON/siamize
* **Docker**: https://hub.docker.com/r/openjdata/siamize
* **Upstream**: [SIAM v0.3](https://github.com/romainVala/SIAM) by Valabregue, Khemir, Bardinet, Rousseau, Auzias & Dorent (2026), [arXiv:2605.02737](https://arxiv.org/abs/2605.02737)

---

**siamize** is a native, vendor-neutral C++ port of
[SIAM v0.3](https://github.com/romainVala/SIAM) — the *Segment It All Model* for
head/brain tissue segmentation — that runs **without PyTorch, nnU-Net, or
torchio** at deployment time. This is the first public release: the same model
runs on **NVIDIA, AMD, and Intel GPUs** (or any CPU), ships as a prebuilt
**Docker image**, and includes **MATLAB / GNU Octave** bindings.

## What's in the box

- **Slim Python reference** (`py/siam_ref.py`) — reproduces SIAM v0.3 inference
  using only PyTorch + numpy + nibabel + scipy + `dynamic_network_architectures`
  (no nnU-Net, no torchio, no SimpleITK).
- **ONNX export pipeline** (`tools/onnx_export/`) — converts each fold of the
  SIAM v0.3 ResEnc-UNet to fp16 `.onnx`, validated against the Python reference.
- **C++ standalone binary** (`src/`) — drop-in for `siam-pred` with no Python at
  runtime, with two interchangeable inference backends (below).

## Two inference backends

| Binary (entrypoint) | Backend | Hardware |
|---|---|---|
| `siamize` *(default)* | ONNX Runtime + CUDA EP | NVIDIA GPU, or CPU |
| `siamize-opencl` | MNN + OpenCL | NVIDIA / AMD / Intel GPU, or CPU |

- **ONNX Runtime / CUDA** — 232 KB executable + `libonnxruntime.so` + per-fold
  fp16 `.onnx` weights; CUDA EP on NVIDIA, CPU fallback otherwise. A CoreML
  variant is built for macOS arm64.
- **MNN / OpenCL** (`-DSIAMIZE_BACKEND=mnn`) — vendor-neutral GPU inference via
  OpenCL (and Vulkan / Metal). Statically linked (~9 MB, no `libMNN.so`),
  dlopens the OpenCL ICD at runtime; per-fold fp32 native-Conv3D `.mnn` weights.

## Features

- **5-fold ensemble, 18 SIAM classes**, sliding-window inference at a target
  isotropic spacing (`-u`, default 0.75 mm).
- **Flexible I/O** — input NIfTI (`.nii`/`.nii.gz`) or JNIfTI (`.jnii`/`.bnii`);
  output a uint8 labelmap, a 4D float32 tissue-probability map (`--tpm`), an
  SPM12-style 6-class map (`-C spm`), or JNIfTI (`-F jnii`).
- **Auto-downloaded weights** — fold weights fetch from NeuroJSON on first run
  (mount a volume at `/cache` to keep them).
- **GPU device selection** — flat, 1-based `-G N` index plus a `--list-gpu`
  listing of all OpenCL platforms/devices (mapped through MNN's platform
  reordering).
- **Auto-tuning & memory adaptivity** — per-GPU OpenCL kernel-tuning cache,
  automatic sliding-window patch shrink on VRAM-tight hosts, and a `--lowmem`
  preset.
- **Docker image** — [`openjdata/siamize`](https://hub.docker.com/r/openjdata/siamize),
  CUDA 12 + cuDNN 9, bundling **both** backends. Calendar-versioned `vYYYY.M`
  (this release: `v2026.6`).
- **MATLAB / GNU Octave bindings** — MEX for both backends, with portable Linux
  MEX (pinned `condition_variable::wait` to clear the `GLIBCXX_3.4.30` load
  failure against bundled MATLAB libstdc++).

## Why siamize (vs. upstream SIAM Python)

Same trained SIAM v0.3 weights, but packaged for real-world deployment.

**Deployment & dependencies**

| | Upstream `siam-pred` | siamize |
|---|---|---|
| Runtime | Python ≥3.10 + `torch` + `nnunetv2` + `torchio` | self-contained C++17 binary |
| Dependency footprint | ~5–6 GB | ~50 MB binary + ~70 MB ORT lib |
| Bindings | Python CLI only | CLI + **MATLAB** + **GNU Octave** (MEX) |
| Container/cluster use | full Python + CUDA stack | single binary + weights; published Docker image |
| OS coverage | Linux (macOS via `mps`) | Linux / macOS / Windows (all CI-tested) |

**Hardware**

| Backend | Upstream | siamize |
|---|---|---|
| NVIDIA CUDA | ✅ (PyTorch) | ✅ (ORT CUDA EP) |
| NVIDIA TensorRT | ❌ | ✅ (opt-in) |
| AMD / Intel GPU | ❌ | ✅ (**MNN + OpenCL**) |
| Apple Silicon | partial (Metal GPU, **no ANE**) | ✅ CoreML incl. **ANE** |
| CPU | ✅ | ✅ |
| Auto-fallback | cuda → mps → cpu | auto → tensorrt → cuda → coreml → cpu |

**Weights & performance**

| | Upstream | siamize |
|---|---|---|
| Format / precision | nnU-Net `.pth`, fp32 | `.onnx`, **fp16** (same trained values) |
| Size per fold | 1.14 GB | ~270 MB raw / **~90 MB gzipped** |
| 5-fold total | ~5.7 GB | **~450 MB gzipped** |
| Single fold, A100 | not benchmarked | **9.8 s** |
| Single fold, RTX 2080S | — | 13.3 s |
| CPU 5-fold ensemble | ~25 min | **~10.5 min** |
| Determinism | run-to-run cuDNN algo search | deterministic fp16 graph |

**Output & I/O features**

| Capability | Upstream | siamize |
|---|---|---|
| Output containers | NIfTI-1 `.nii.gz` only | `.nii` / `.nii.gz` / **`.jnii`** / **`.bnii`** |
| Probability map | ❌ | ✅ `--tpm` (4D float32, + temperature) |
| Class remap | ❌ | ✅ `-C spm` (SIAM-18 → SPM12-6) |
| Embedded label table | ❌ | ✅ JGIFTI LabelTable |
| Fold selection | always 5-fold | single or N-fold via `-M` |
| Save at inference resolution | ❌ | ✅ `--upsample` |

**Trade-offs / not (yet) supported**

| Item | Notes |
|---|---|
| ~0.3% boundary-voxel gap vs Python | almost entirely the resampling kernel (Catmull-Rom vs scipy cubic B-spline) — not the network or the fp16 cast |
| Batch-folder input | one file per invocation (bash-loop workaround) |
| 4D-volume input | split first (same as upstream) |
| Test-time augmentation | neither tool uses it |
| AMD ROCm EP | not wired (OpenCL covers AMD instead) |

Weights are unchanged from SIAM v0.3 — no pruning, no quantization beyond fp16,
no architecture trimming.

## Patched MNN included

The MNN backend uses a patched MNN fork (`v3.5-opencl-conv3d`):
`CPUConvolution3D` / `CPUDeconvolution3D` now correctly handle MNN's NC4HW4
activation layout (unpack → NCDHW GEMM → repack), fixing incorrect CPU-backend
results. Verified at 667/667 ops and 99.59% end-to-end voxel agreement.

## Accuracy

Reproduces the original SIAM output on the bundled `sub-01_T1w.nii.gz` (5-fold
ensemble, 18 classes) within a known ~0.3% precision gap vs the Python SIAM
pipeline; see the README's
[Known precision gap](https://github.com/NeuroJSON/siamize#known-precision-gap-03-vs-python)
section.

## Quick start (Docker)

```bash
# NVIDIA GPU, ONNX Runtime / CUDA -- 5-fold ensemble (default entrypoint)
docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
    openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/labels.nii.gz -M 0,1,2,3,4 -c cuda

# Vendor-neutral GPU via MNN / OpenCL (NVIDIA / AMD / Intel)
docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
    --entrypoint siamize-opencl openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/labels.nii.gz -M 0 -c opencl
```

See the [README](https://github.com/NeuroJSON/siamize#docker) for
build-from-source, CPU-only, `--tpm`, `-C spm`, and JNIfTI examples.

## Citation

If you use this in your work, please cite the upstream SIAM paper
([arXiv:2605.02737](https://arxiv.org/abs/2605.02737)).

**Acknowledgement:** This project uses resources and data formats developed as
part of the [NeuroJSON project](https://neurojson.org), supported by US National
Institute of Health (NIH) grant
[U24-NS124027](https://reporter.nih.gov/project-details/10308329).
