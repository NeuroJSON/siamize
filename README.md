# siamize - native C++/ONNX port of SIAM v0.3 brain segmentation

[![ci](https://github.com/NeuroJSON/siamize/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuroJSON/siamize/actions/workflows/ci.yml)

* **Copyright**: (C) Qianqian Fang (2026) \<q.fang at neu.edu>
* **License**: Apache License, Version 2.0
* **Version**: 0.1.0
* **GitHub**: [https://github.com/NeuroJSON/siamize](https://github.com/NeuroJSON/siamize)
* **Upstream**: [https://github.com/romainVala/SIAM](https://github.com/romainVala/SIAM) — SIAM v0.3 by Valabregue, Khemir, Bardinet, Rousseau, Auzias & Dorent (2026), [arXiv:2605.02737](https://arxiv.org/abs/2605.02737)

---

## Table of Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Layers, in dependency order](#layers-in-dependency-order)
- [Footprint](#footprint)
- [Platforms](#platforms)
- [Performance](#performance)
- [Engine choice / GPU portability](#engine-choice--gpu-portability)
- [Known precision gap (~0.3% vs Python)](#known-precision-gap-03-vs-python)
- [Citation](#citation)
- [Credits](#credits)

---

## Overview

A native, vendor-neutral port of [SIAM v0.3](https://github.com/romainVala/SIAM)
— the *Segment It All Model* for head/brain tissue segmentation — that runs
without PyTorch, nnU-Net, or torchio at deployment time.

`siamize` ships:

1. A **slim Python reference** (`py/siam_ref.py`) that reproduces SIAM's
   inference using only PyTorch + numpy + nibabel + scipy +
   `dynamic_network_architectures`. No nnU-Net, no torchio, no SimpleITK.
2. An **ONNX export pipeline** (`tools/onnx_export/`) that converts each fold
   of the SIAM v0.3 ResEnc-UNet to fp16 `.onnx`, validating against (1).
3. A **C++ standalone binary** (`src/`) — 232 KB executable + 23 MB
   `libonnxruntime.so` + per-fold 270 MB `.onnx` — drop-in for `siam-pred`
   with no Python at runtime.

Accuracy vs original SIAM on the bundled `sub-01_T1w.nii.gz`
(5-fold ensemble, 18 classes):

| Pipeline | Voxel agreement | Worst per-class Dice |
|---|---|---|
| `py/siam_ref.py`                    | 99.989%  | 0.9990 |
| `tools/onnx_export/siam_ort.py`     | 99.989%  | 0.9990 |
| C++ binary (`build/siamize`)        | 99.715%  | 0.9697 (Anomalies, 17 voxels) |

## Quickstart

### 1. Fetch dependencies

```bash
scripts/fetch_deps.sh        # downloads ORT prebuilt + clones nifti_clib into third_party/
scripts/fetch_weights.sh     # downloads the 5 fp16 .onnx folds into models/ (~1.35 GB)
```

The fetch script auto-detects the host (Linux x64, Linux aarch64, macOS
x86_64, macOS arm64, Windows x64) and pulls the right ORT prebuilt. On
Windows, run it from Git Bash (or any POSIX shell — Git for Windows ships
bash, curl, and tar with `.zip` support out of the box).

For native Windows users without a POSIX shell, the equivalent PowerShell
script is also provided:

```powershell
scripts\fetch_deps.ps1
```

### 2. Build the C++ binary

```bash
cmake -S . -B build
cmake --build build -j
```

This produces `build/siamize`. `libonnxruntime.so.1` is located by RPATH:
the binary looks first in `$ORIGIN` (next to itself) and then in
`third_party/onnxruntime/lib/` (the development tree), so you can either
drop the .so next to the binary for distribution or run from a fresh
checkout without setting `LD_LIBRARY_PATH`.

### 3. Run

```bash
build/siamize \
    -i input.nii.gz \
    -o output.nii.gz \
    --models models/fold_0_fp16.onnx,models/fold_1_fp16.onnx,models/fold_2_fp16.onnx,models/fold_3_fp16.onnx,models/fold_4_fp16.onnx \
    --threads 8 -v
```

Single-fold prediction is also supported (just pass one `.onnx`).

### 4. Regression test (optional)

```bash
tests/run_regression.sh
```

Runs the bundled sample through `build/siamize` and reports voxel
agreement vs `tests/pred_ref_allfolds.nii.gz`.

## Layers, in dependency order

```
py/siam_ref.py          # slim PyTorch reference, used to validate (2)
                              │
                              v
tools/onnx_export/      # PyTorch → fp16 .onnx; uses py/siam_ref to verify
                              │
                              v
src/  +  CMakeLists.txt # C++ standalone with ONNX Runtime, uses .onnx from (2)
```

## Footprint

| Artifact | Size |
|---|---|
| `siamize` binary (static-linked C++/zlib/OpenMP) | 2.2 MB |
| `libonnxruntime.so.1.26.0` | 23 MB |
| One fold `.onnx` (fp16) | 270 MB |
| Five folds | 1.35 GB |
| **Single-fold deployable bundle** | **≈295 MB** |

vs. the original SIAM stack: multi-GB PyTorch + nnU-Net + torchio install,
plus 5.4 GB checkpoints.

### Runtime shared-library dependencies of `siamize`

```
libonnxruntime.so.1    # the only non-glibc dep; bundled with the binary
libm.so.6              # glibc
libc.so.6              # glibc
ld-linux-x86-64.so.2   # glibc
```

`libstdc++`, `libgcc`, `libgomp`, `libz` and `nifti_clib` are all statically
linked into the binary (CMake option `SIAMIZE_STATIC_LINK=ON`, the default).
The `libstdc++` / `libgcc` / `libpthread` etc. that show up in `ldd` output
are transitive deps of `libonnxruntime.so.1`, not of `siamize` itself —
verifiable via `readelf -d build/siamize | grep NEEDED`. ONNX Runtime
ships only as a `.so` (Microsoft does not provide a static `.a`); building
ORT from source statically is possible but a substantial undertaking and
not done by default here.

To toggle the static linking, pass `-DSIAMIZE_STATIC_LINK=OFF` to CMake.

## Platforms

The C++ code is portable C++17 and the build is CMake-driven. CI builds
the binary on all three:

| Host | Toolchain | Static-linked C/C++ runtime? |
|---|---|---|
| Linux x86_64 / aarch64 | GCC (Apt) | yes (`-static-libstdc++ -static-libgcc`, static `libgomp.a`, static `libz.a`) |
| macOS x86_64 / arm64   | Apple clang | partial: relies on libc++ (ABI-stable on macOS); OpenMP via Homebrew `libomp` |
| Windows x64            | MSVC | yes (`/MT` static CRT); `onnxruntime.dll` copied next to `siamize.exe` |

On every platform the binary ships with `libonnxruntime` (`.so` / `.dylib` /
`.dll`) sitting next to it; everything else statically linkable is statically
linked by default. Set `-DSIAMIZE_STATIC_LINK=OFF` to keep things dynamic.

Locally tested: Linux x86_64. macOS / Windows are exercised by CI (see
`.github/workflows/ci.yml`); please open an issue if a host setup breaks.

## Performance

Measured on a TITAN V workstation, CPU only:

| Run | Time |
|---|---|
| C++ 5-fold ensemble | 634 s (10.5 min) |
| Python ORT 5-fold | 781 s (13 min) |
| Original `siam-pred` 5-fold CPU (per upstream README) | ~25 min |

## Engine choice / GPU portability

The C++ binary uses **ONNX Runtime** with the CPU execution provider. Adding
optional GPU providers later is a build-flag change, not a code change:

- **CUDA EP** for NVIDIA — drop in `libonnxruntime-gpu.so` from the same ORT
  release.
- **DirectML EP** for any DX12 GPU on Windows.
- **OpenVINO EP** for Intel CPU/GPU on Linux/Windows.

A vendor-neutral GPU path on Linux (Vulkan/OpenCL) is not provided by ORT
itself; for that, the same `.onnx` files can feed [MNN](https://github.com/alibaba/MNN)
(OpenCL) or [TVM](https://tvm.apache.org/) (Vulkan/OpenCL/SPIR-V). Initial
exploration of [ncnn](https://github.com/Tencent/ncnn) found its Vulkan
backend lacks 3D conv kernels for this model.

## Known precision gap (~0.3% vs Python)

The C++ pipeline uses cubic Catmull-Rom (3rd-order Hermite) for the forward
image resample. scipy/skimage use cubic B-spline (also 3rd-order, different
basis with a pre-filter step). The two give visually identical output but
differ at fp32-noise level on the network input, which propagates into
~0.27% boundary voxel disagreements after argmax. If sub-percent precision
matters, a scipy-compatible cubic B-spline resampler is the obvious next
upgrade (~150 lines of standard code).

## Citation

If you use `siamize` in your work, **please cite the original SIAM paper**:

> Valabregue, R., Khemir, I., Bardinet, E., Rousseau, F., Auzias, G., & Dorent, R. (2026).
> *SIAM: Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training.*
> arXiv:2605.02737. https://arxiv.org/abs/2605.02737

BibTeX:

```bibtex
@article{valabregue2026siam,
  title   = {SIAM: Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training},
  author  = {Valabregue, Romain and Khemir, Ikram and Bardinet, Eric and Rousseau, Francois and Auzias, Guillaume and Dorent, Reuben},
  year    = {2026},
  journal = {arXiv preprint arXiv:2605.02737},
  url     = {https://arxiv.org/abs/2605.02737}
}
```

### Optional: citing this port

If you are required to also cite the specific software port (e.g., a
journal that asks for the inference tool you used), you may additionally
reference `siamize`:

```bibtex
@software{siamize,
  title  = {siamize: native C++/ONNX port of SIAM v0.3 brain segmentation},
  author = {Fang, Qianqian},
  year   = {2026},
  url    = {https://github.com/NeuroJSON/siamize}
}
```

This is **secondary** — please always cite the SIAM paper above first.

## Credits

`siamize` is a port of [**SIAM v0.3**](https://github.com/romainVala/SIAM)
by Valabregue, Khemir, Bardinet, Rousseau, Auzias & Dorent (2026), and reuses
the published SIAM v0.3 weights without modification.

### Bundled third-party code

- **[zmat](https://github.com/NeuroJSON/zmat)** by Qianqian Fang — the
  single-header amalgamation `src/zmat/zmat.h` provides all `.nii.gz`
  compression and decompression. zmat is part of the
  [NeuroJSON project](https://neurojson.org), supported by US NIH grant
  [U24-NS124027](https://reporter.nih.gov/project-details/10308329).
  Upstream zmat is GPL-3.0; this single file has been **dual-licensed
  under Apache-2.0 for siamize** by the zmat author, as documented in
  the file's header. Inside zmat:
  - **[miniz](https://github.com/richgel999/miniz)** by Rich Geldreich
    — public-domain (Unlicense) zlib-subset deflate/inflate.

### Test data

The bundled test image `tests/sub-01_T1w.nii.gz` is the `sub-01` anatomical
T1-weighted scan from [OpenNeuro ds000001 v1.0.0](https://openneuro.org/datasets/ds000001/versions/1.0.0),
redistributed here under its original **CC0** public-domain dedication.
See `tests/README.md` for details.
