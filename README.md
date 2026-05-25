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
- [MATLAB / GNU Octave bindings](#matlab--gnu-octave-bindings)
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
git submodule update --init  # pulls the bundled jsonlab under matlab/jsonlab (only needed for the MATLAB/Octave wrapper; see below)
```

The fetch script auto-detects the host (Linux x64, Linux aarch64, macOS
x86_64, macOS arm64, Windows x64) and pulls the right ORT prebuilt. On
Windows, run it from Git Bash (or any POSIX shell — Git for Windows ships
bash, curl, and tar with `.zip` support out of the box). For native
Windows users without a POSIX shell, the equivalent PowerShell script is
also provided:

```powershell
scripts\fetch_deps.ps1
```

The fp16 ONNX fold weights are **not** fetched up-front: `siamize` and
its MATLAB/Octave wrapper auto-download any missing fold from
NeuroJSON (URL prefix
`https://neurojson.org/io/stat.cgi?action=get&db=siam_v03&doc=dynshape&size=95360591&file=`,
overridable via `SIAMIZE_WEIGHTS_BASE_URL`) into a shared cache (`$SIAMIZE_CACHE_DIR`,
default `$HOME/.cache/siamize/models/` on POSIX or
`%LOCALAPPDATA%/siamize/models/` on Windows). One download serves both
the CLI binary and the MEX. If you want all five folds pre-staged
before going offline, run:

```bash
scripts/fetch_weights.sh     # downloads the 5 fp16 .onnx folds (~1.35 GB) into models/
```

### 2. Build the C++ binary

CPU-only (default):

```bash
make            # convenience target -- wraps cmake configure + build
# or, equivalently:
cmake -S . -B build && cmake --build build -j
```

This produces `build/siamize`. `libonnxruntime.so.1` is located by RPATH:
the binary looks first in `$ORIGIN` (next to itself) and then in
`third_party/onnxruntime/lib/` (the development tree), so you can either
drop the .so next to the binary for distribution or run from a fresh
checkout without setting `LD_LIBRARY_PATH`.

#### Optional: NVIDIA GPU build (CUDA Execution Provider)

```bash
make cuda       # re-fetches GPU ORT prebuilt (only if needed) + configures + builds
```

That's the convenience shortcut. The equivalent explicit form:

```bash
rm -rf third_party/onnxruntime build
ORT_BUILD=gpu scripts/fetch_deps.sh                  # default = CUDA 12.x build
# or, if your NVIDIA driver is CUDA 13:
# ORT_BUILD=gpu ORT_CUDA=13 scripts/fetch_deps.sh
cmake -S . -B build -DSIAMIZE_GPU=cuda
cmake --build build -j
```

The binary then accepts `-c {auto,cpu,cuda}` (default `auto`). On
`auto` it tries to register the CUDA Execution Provider and falls back to
CPU if the runtime libraries (`libcudart`, `libcudnn`, `libcublasLt`) can't
be loaded. Pass `-c cuda` to force GPU and fail loudly if it isn't
available; pass `-c cpu` to skip GPU even when compiled in.

CUDA runtime libraries are loaded via `dlopen`, so you may need to set
`LD_LIBRARY_PATH` to include their location. With PyTorch-managed CUDA
(pip's `nvidia-*` packages):

```bash
NV=$(python3 -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib:$LD_LIBRARY_PATH"
build/siamize -i ... -c cuda ...
```

With a system CUDA install, point at it via the standard `CUDA_HOME` env
var (set by the NVIDIA installer on most distros, otherwise default
`/usr/local/cuda`):

```bash
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
build/siamize -i ... -c cuda ...
```

ORT 1.26 requires **cuDNN 9** with a kernel image for your GPU's compute
capability — older GPUs (e.g., Volta sm_70) may need a cuDNN build that
explicitly includes those kernels. If cuDNN was installed separately (the
typical NVIDIA flow), make sure its `lib64/` is on `LD_LIBRARY_PATH` too;
the official installer drops it next to `$CUDA_HOME/lib64/` so the line
above usually covers it.

##### What the prebuilt CUDA bundle ships vs. what you must supply

The `siamize-*-cuda.zip` artifact produced by CI / `make package-cuda`
contains only what's redistributable: the `siamize` binary, ORT 1.26
core, and ORT's CUDA EP plugin DLLs. CUDA/cuDNN themselves are *not*
bundled — they're large (~1 GB combined for cuDNN 9 + cuBLAS + cuFFT),
and cuDNN's license forbids third-party redistribution.

| Component | In the zip | You install |
|---|---|---|
| `siamize` / `siamize.exe` | ✅ | — |
| `libonnxruntime.so.1` / `onnxruntime.dll` | ✅ | — |
| `libonnxruntime_providers_shared.so` / `.dll` | ✅ | — |
| `libonnxruntime_providers_cuda.so` / `.dll` | ✅ | — |
| `libcudart` (CUDA runtime) | ❌ | CUDA Toolkit, or `pip install nvidia-cuda-runtime-cu12` |
| `libcublas` + `libcublasLt` | ❌ | CUDA Toolkit, or `pip install nvidia-cublas-cu12` |
| `libcudnn` (cuDNN 9 for ORT 1.26) | ❌ | NVIDIA cuDNN 9 installer, or `pip install "nvidia-cudnn-cu12==9.*"` |
| `libcufft`, `libcurand`, `cuda_nvrtc`, `nvjitlink` | ❌ | CUDA Toolkit, or matching `nvidia-*-cu12` pip wheels |

##### Required shared libraries by exact filename

ORT 1.26's CUDA EP plugin (`libonnxruntime_providers_cuda.so`) is
dlopen'd at runtime. It in turn pulls in the following dependencies
*at the exact major version*:

| Library (Linux) | Library (Windows) | Provided by |
|---|---|---|
| `libcudart.so.12` | `cudart64_12.dll` | CUDA 12 runtime |
| `libcublas.so.12` | `cublas64_12.dll` | CUDA 12 |
| `libcublasLt.so.12` | `cublasLt64_12.dll` | CUDA 12 |
| `libcufft.so.11` | `cufft64_11.dll` | CUDA 12 (cuFFT is independently versioned) |
| `libcurand.so.10` | `curand64_10.dll` | CUDA 12 (cuRAND ditto) |
| `libcudnn.so.9` | `cudnn64_9.dll` | cuDNN 9 |
| `libcudnn_ops.so.9` | `cudnn_ops64_9.dll` | cuDNN 9 |
| `libcudnn_cnn.so.9` | `cudnn_cnn64_9.dll` | cuDNN 9 |
| `libcudnn_engines_precompiled.so.9` | `cudnn_engines_precompiled64_9.dll` | cuDNN 9 |
| `libnvrtc.so.12` | `nvrtc64_120_0.dll` | CUDA 12 nvrtc |
| `libnvJitLink.so.12` | `nvJitLink_120_0.dll` | CUDA 12 nvjitlink |

If any one of these is missing or its loader path is wrong, siamize's
`-c auto` probe falls back to CPU and the `[warn] ORT:` line names
the *first* missing file — e.g.:

```
[warn]     ORT: Failed to load library .../libonnxruntime_providers_cuda.so
           with error: libcublasLt.so.12: cannot open shared object file
[cuda]     unavailable (...); using CPU
```

Means cuBLAS isn't on `LD_LIBRARY_PATH`. The CUDA EP plugin loaded;
the *transitive* cuBLAS dependency didn't. Add cuBLAS's directory to
`LD_LIBRARY_PATH` (per the pip-wheels or `CUDA_HOME` one-liners
above) and re-run. The probe will then surface the *next* missing
library, if any — fix iteratively until `-c auto` reports
`[cuda]     enabled (gpuid=0)`.

To preflight before running siamize, `ldd` the CUDA EP plugin:

```bash
ldd third_party/onnxruntime/lib/libonnxruntime_providers_cuda.so | grep -E "not found|=>"
# every line should resolve; any "not found" entry names a missing lib
```

On Windows, use `dumpbin /dependents` (Visual Studio Developer Prompt):

```cmd
dumpbin /dependents third_party\onnxruntime\lib\onnxruntime_providers_cuda.dll
```

##### Windows: pointing siamize.exe at the CUDA runtime DLLs

On Windows the loader uses `PATH` (not `LD_LIBRARY_PATH`) to find DLLs.
The CUDA Toolkit installer sets the `CUDA_PATH` env var (e.g.
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`) and usually
prepends `%CUDA_PATH%\bin` to `PATH` itself. If `siamize.exe` reports
"cannot find cudart64_12.dll" after unzipping the bundle, force-add it:

```powershell
# PowerShell
$env:PATH = "$env:CUDA_PATH\bin;" + $env:PATH

# cmd.exe equivalent:
# set PATH=%CUDA_PATH%\bin;%PATH%

.\siamize.exe -i input.nii.gz -o pred.nii.gz -M 0 -c cuda
```

cuDNN's Windows installer copies its DLLs into `%CUDA_PATH%\bin` (the
default checkbox in the cuDNN MSI), so the same one-liner usually
covers cuDNN too.

For a lighter-weight install via pip wheels (no CUDA Toolkit needed):

```powershell
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 `
            "nvidia-cudnn-cu12==9.*"   `
            nvidia-cufft-cu12 nvidia-curand-cu12 `
            nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12

# Prepend the wheel DLL dirs to PATH (Windows equivalent of the
# LD_LIBRARY_PATH one-liner shown above for Linux).
$NV = (python -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))")
$env:PATH = "$NV\cublas\bin;$NV\cuda_runtime\bin;$NV\cudnn\bin;" `
          + "$NV\cufft\bin;$NV\curand\bin;$NV\cuda_nvrtc\bin;" `
          + "$NV\nvjitlink\bin;" + $env:PATH

.\siamize.exe -i input.nii.gz -o pred.nii.gz -M 0 -c cuda
```

Note: the pip wheels put their DLLs under `bin\` on Windows (vs. `lib\`
on Linux). Hardware-compatibility caveat is the same as Linux — the pip
cuDNN/cuBLAS wheels target sm_75+; older GPUs need the official NVIDIA
installer.

#### Optional: TensorRT EP (advanced, batch-processing workloads only)

For workloads that process hundreds of volumes with the same model/GPU
combo, the TensorRT Execution Provider can shave ~35% off CUDA EP wall
time. It's an opt-in build:

```bash
# Build with TRT enabled (gpu ORT prebuilt also has the TRT provider plugin).
make tensorrt
# equivalent explicit form:
# cmake -S . -B build -DSIAMIZE_GPU=tensorrt && cmake --build build -j

# Install the matching TensorRT Python wheel (ships libnvinfer + per-SM
# builder resources). Pin it to your CUDA runtime version.
pip install --user "tensorrt~=10.0"

# Make TRT libs visible alongside the CUDA libs.
TRT=$(python3 -c "import os, tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))")
export LD_LIBRARY_PATH="$TRT:$LD_LIBRARY_PATH"

build/siamize -i input.nii.gz -o output.nii.gz \
    -M models/fold_0_fp16.onnx \
    -c tensorrt \
    --trt-cache-dir $HOME/.cache/siamize/trt
```

**Cost model on a Turing RTX 2080 SUPER (single fold):**

| Mode | Wall time | Notes |
|---|---|---|
| CUDA EP | 13.3 s | warm |
| TRT EP, first run | **962 s** | one-time engine build per fold/GPU/TRT-version |
| TRT EP, cached | **8.7 s** | ~35 % faster than CUDA, every subsequent run |

Correctness: TRT vs CUDA output → 99.97 % voxel agreement, worst per-class
Dice 0.997 (fused-kernel rounding only).

**Breakeven**: amortizing one cold engine build (962 s) against the per-run
savings (13.3 − 8.7 = 4.6 s) takes **~209 inferences per fold**. For a
5-fold ensemble that's ~209 full-volume runs end-to-end.

**Hidden costs:**

- TensorRT Python wheel: ~1 GB on disk (libnvinfer + per-arch builder
  resources for sm_75…sm_120).
- Engine cache: 274 MB per fold (1.37 GB for the 5-fold ensemble).
- Cache invalidation: any change to the ONNX model, the GPU compute
  capability, or the TRT minor version forces a fresh ~16 min/fold rebuild.

If you're not deploying to a batch server, **stick with the default CUDA
EP**. The TRT path stays available for the lab that needs it.

### 3. Run

```bash
# Full 5-fold ensemble (the digit shortcut expands to fold_<N>_fp16.onnx;
# any missing weight auto-downloads into the shared cache).
build/siamize -i input.nii.gz -o output.nii.gz -M 0,1,2,3,4 -v

# Single-fold prediction is also supported:
build/siamize -i input.nii.gz -o output.nii.gz -M 0 -v

# Explicit paths still work alongside shortcuts:
build/siamize -i input.nii.gz -o output.nii.gz \
    -M models/fold_0_fp16.onnx,models/fold_1_fp16.onnx
```

`-t/--thread` defaults to `0` (all available cores via
`std::thread::hardware_concurrency()`); set it explicitly only if you
want to throttle CPU use.

#### JNIfTI containers (`.jnii` / `.bnii`)

In addition to NIfTI-1, `siamize` reads and writes
[JNIfTI](https://neurojson.org/jnifti) — the JData/BJData wrapping
of NIfTI metadata + voxel data used by the NeuroJSON ecosystem.
Input format is inferred from the file extension; output format is
selected via `-F/--format`:

```bash
# Text-JSON JNIfTI labelmap (human-readable; zlib + base64 payload)
build/siamize -i input.nii.gz -o labels.jnii -M 0 -F jnii

# BJData binary JNIfTI labelmap (compact binary; zlib raw bytes)
build/siamize -i input.nii.gz -o labels.bnii -M 0 -F bnii

# JNIfTI input (e.g. produced by jsonlab's savejd):
build/siamize -i preproc.bnii -o labels.bnii -M 0 -F bnii
```

| `-F` | Output container | Payload | Notes |
|---|---|---|---|
| `nii` (default) | NIfTI-1 (`.nii`, gzipped if `.gz`) | raw bytes | Smallest for sparse uint8 labelmaps. |
| `jnii` | JSON-text JNIfTI | zlib + base64 | Human-readable; pairs with `loadjd` / `jsonlab`. |
| `bnii` | BJData binary JNIfTI | zlib (raw bytes) | Compact binary; pairs with the NeuroJSON ecosystem. |

For label volumes specifically, `.nii.gz` is usually the tightest on
disk — raw gzip already exploits per-voxel value redundancy
efficiently. JNIfTI's value here is interoperability with JSON / JData
tooling (jsonlab, `jdataencode` / `jdatadecode`, browser-based viewers,
the NeuroJSON database), not raw compression. Voxel-identical
round-tripping has been verified for `.nii.gz` ↔ `.jnii` ↔ `.bnii`
inputs on the bundled `sub-01_T1w.nii.gz` (100 % agreement, 5-fold
ensemble).

### 4. Regression test (optional)

```bash
tests/run_regression.sh
```

Runs the bundled sample through `build/siamize` and reports voxel
agreement vs `tests/pred_ref_allfolds.nii.gz`.

## MATLAB / GNU Octave bindings

The same inference pipeline is callable from MATLAB and Octave through a
thin MEX (`siamex.mex*`) wrapped by a pure-MATLAB dispatcher
(`matlab/siamize.m`). MEX and CLI predictions are bit-identical (they
share the `siamize_core` C++ sources).

### Build

```bash
# Octave (Linux/macOS):
make mex-octave              # -> build/siamex.mex

# MATLAB (Linux/macOS/Windows):
make mex-matlab              # -> build/siamex.mexa64 / .mexmaca64 / .mexw64

# Equivalent explicit forms:
# cmake -S . -B build -DSIAMIZE_BUILD_OCTAVE_MEX=ON  && cmake --build build -j
# cmake -S . -B build -DSIAMIZE_BUILD_MATLAB_MEX=ON  && cmake --build build -j
```

The bundled jsonlab submodule (`matlab/jsonlab/`) provides
`loadjd` / `savejd` / `loadnifti` / `jnii2nii` / `savejnifti` / etc.;
`siamize.m` adds it to the path automatically if it isn't already
visible.

### Calling format

`siamize.m` accepts flexible inputs and returns a single jnifti
struct (`nii.NIFTIHeader` + `nii.NIFTIData`). The same struct is
written to disk when an outputfile is supplied.

```matlab
% one-shot file -> file (defaults: single-fold fold_0, auto-downloaded)
siamize('input.nii.gz', 'labels.nii.gz');

% cross-format: read .nii.gz, write binary JNIfTI, full 5-fold ensemble
siamize('input.nii.gz', 'labels.bnii', 0:4);

% struct input (jnifti or readnifti-style), in-memory labels
nii_in  = loadnifti('input.nii.gz');
nii_out = siamize(nii_in);        % nii_out.NIFTIData is uint8 3D labels

% bare 3D array, default centered affine inferred
nii_out = siamize(my_volume);
nii_out = siamize(my_volume, 0);                       % single fold by shortcut
nii_out = siamize(my_volume, '0,2,4', 'verbose', true);

% explicit affine + output file + ensemble + opts
siamize(my_volume, A, 'labels.nii.gz', 0:4, 'compute', 'cuda');

% TPM mode: nii_out.NIFTIData becomes 4D single (float32) [X, Y, Z, 18]:
nii_tpm = siamize('input.nii.gz', 0:4, 'tpm', true, 'tpm_t', 1.5);
siamize('input.nii.gz', 'tpm.nii.gz', 0:4, 'tpm', true);  % save TPM to disk
```

| First arg | Interpretation |
|---|---|
| `'file.{nii,nii.gz,jnii,bnii}'` | read via `loadjd`; affine taken from header |
| jnifti struct (`.NIFTIData` + `.NIFTIHeader.Affine`) | passthrough |
| readnifti struct (`.img` + `.hdr.srow_*`) | passthrough; affine from sform |
| 3D numeric array | identity rotation + centered translation synthesized when no affine follows |

The `models` argument accepts numeric indices, char shortcuts, full
paths, or mixes thereof: `0`, `0:4`, `'0,2,4'`, `{'0','fold_3_fp16.onnx'}`.
Output extension picks the writer (`.nii[.gz]` → `jnii2nii`,
`.jnii`/`.bnii` → `savejnifti`). The shared weight cache
(`$SIAMIZE_CACHE_DIR`) is reused so a single download serves both the
MEX and the CLI binary. Full reference: [`matlab/README.md`](matlab/README.md).

### Tests

```bash
make mex-test
# equivalent: octave-cli --no-gui --eval "cd matlab/tests; run_tests('--exit')"
```

30 unit tests that stub the underlying MEX so they run in under a
second and require no ORT or weight files. Covers argument-form
dispatch, default-affine math, model-spec parsing, file-in/file-out
across the four extensions, source-header preservation, and the error
paths. CI runs the same suite on both Octave and MATLAB legs.

## Layers, in dependency order

```
py/siam_ref.py          # slim PyTorch reference, used to validate (2)
                              │
                              v
tools/onnx_export/      # PyTorch → fp16 .onnx; uses py/siam_ref to verify
                              │
                              v
src/  +  CMakeLists.txt # C++ standalone with ONNX Runtime, uses .onnx from (2)
        │       │
        │       └───>  build/siamize          # CLI binary
        │
        └─────────>    build/siamex.mex*      # MATLAB / Octave MEX
                              │               (shares siamize_core sources)
                              v
                       matlab/siamize.m       # pure-MATLAB dispatcher
                       matlab/jsonlab/        # bundled NeuroJSON jsonlab (submodule)
                       matlab/tests/          # Octave + MATLAB unit tests
```

## Footprint

| Artifact | Size |
|---|---|
| `siamize` binary (static-linked C++/zlib/OpenMP) | 2.2 MB |
| `siamex.mex*` (Octave MEX, dynamic libstdc++) | 180 KB |
| `siamex.mexa64` (MATLAB MEX, static libstdc++) | ~3 MB |
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

The MATLAB / Octave MEX (`siamex.mex*`) is exercised by CI on
`linux-octave`, `linux-matlab`, and `windows-matlab` matrix legs; on
Linux the MATLAB MEX statically embeds libstdc++ (to escape MATLAB's
older bundled `libstdc++.so.6`) while the Octave MEX stays dynamic
(static-linking would conflict with Octave's already-loaded C++
runtime).

Locally tested: Linux x86_64. macOS / Windows are exercised by CI (see
`.github/workflows/ci.yml`); please open an issue if a host setup breaks.

## Performance

All measurements use the bundled `tests/sub-01_T1w.nii.gz` (160×192×192,
1.0/1.333/1.333 mm) running siamize's 5-fold ensemble with `models/fold_*_fp16.onnx`.

### CPU (TITAN V workstation, x86, 8 threads)

| Run | Time |
|---|---|
| C++ 5-fold ensemble (`siamize -c cpu`) | 634 s (10.5 min) |
| C++ single fold | 126 s |
| Python ORT 5-fold | 781 s (13 min) |
| Original `siam-pred` 5-fold CPU (per upstream README) | ~25 min |

#### CPU thread tuning on many-core hosts (Zen2 / EPYC / Threadripper)

ORT's CPU Execution Provider does **not** scale linearly past ~16
threads on this workload. Profiling on an AMD Threadripper 3990X
(Zen2, 64C/128T, single NUMA node) gives a clear shape:

| `-t` | Wall (s) | CPU% | Avg cores busy |
|---|---|---|---|
| 4 (no auto) | 300.2 | 331 % | 3.3 |
| 8 | 206.4 | 540 % | 5.4 |
| 12 | 171.6 | 694 % | 6.9 |
| **16** | **158.9** | 840 % | **8.4** ← optimum |
| 32 | 160.4 | 1098 % | 11.0 |
| 48 | 162.2 | 1372 % | 13.7 |
| 64 | 165.1 | 1632 % | 16.3 |
| 0 (=128) | 178.4 | 2674 % | 26.7 |

Zen2's 16-cores-per-CCD topology is the reason: `-t 16` keeps the
active hot set inside one CCD's 64 MB L3 and one memory controller.
At 32+ threads the active set spills across CCDs through the I/O
die, which costs roughly what the extra parallelism gains. At 128
the contention wins outright.

To match this, `siamize -t 0` (the default) now resolves to
`min(hardware_concurrency, 16)` rather than full
`hardware_concurrency`. On laptops / small workstations (≤16 cores)
this is a no-op; on big servers it stops siamize from oversubscribing
ORT's thread pool. Pass `-t N` explicitly to override.

#### Memory-arena trade-off

`siamize -c cpu` keeps ORT's CPU memory arena + memory-pattern
optimizer **on** by default. On the same Threadripper run this
buys 1.5× wall-time (273.6 → 178.3 s on `-t 0`) at the cost of
peak RSS growing from 12.4 GB → 28.3 GB for the 18-class network.
Profiling traced the off-arena path's slowdown to 75 M minor page
faults and a 43 % dTLB miss rate from per-op `mmap`/`munmap` churn.

If you'd rather trade speed for memory, pass `--no-arena` on the
CLI (or `engine_tuning.cpu_arena = false` from the MEX). The
`-v` header surfaces the choice (`--no-arena` is appended when
the arena is disabled).

### NVIDIA GPU (siamize built with `-DSIAMIZE_GPU=cuda`)

| Run | GPU | Time | vs CPU C++ |
|---|---|---|---|
| Single fold | RTX 2080 Super (Turing sm_75, 8 GB) | 13.3 s (±0.04 s, n=3) | **9.5×** |
| 5-fold ensemble | RTX 2080 Super (Turing sm_75, 8 GB) | 58.5 s | **~11×** |
| Single fold | A100 (Ampere sm_80, 40 GB) | pending — device contended at benchmark time |

Correctness: the Turing 5-fold output matches the Phase-1 PyTorch reference
at 99.7167% voxel agreement — identical to the CPU C++ result. Switching to
the CUDA Execution Provider does not introduce additional numerical drift on
top of fp16 ONNX + cubic-Hermite resampling.

GPU memory: the full 5-fold run fits on the 8 GB RTX 2080 Super with no OOM.
Estimated peak total ≈ 4–6 GB (model weights + held activations + cuDNN
workspace + output). For low-VRAM cards a `gpu_mem_limit` knob can be wired
in; in practice 8 GB has been sufficient.

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
- **[nlohmann/json](https://github.com/NeuroJSON/json)** (NeuroJSON
  fork) — the single-header `src/nlohmann/json.hpp` provides the
  JSON / BJData parser and serializer used for `.jnii` / `.bnii` I/O.
  MIT-licensed. The NeuroJSON fork extends upstream nlohmann/json with
  BJData Draft 4 SOA (structure-of-arrays) support; the JData
  annotated-array spec (`_ArrayType_`, `_ArrayZipData_`, etc.) is
  implemented by `src/jnifti_io.cpp` on top of it. The header is
  vendored from the **[umcx](https://github.com/fangq/umcx)** project
  by Qianqian Fang — a single-source OpenMP / OpenACC Monte Carlo
  photon simulator — where the same header serves the same JData /
  BJData container role; keeping a single vendored copy across
  NeuroJSON-family C/C++ tools (umcx, siamize, …) avoids divergence.

### Standards implemented

- **[JNIfTI specification](https://neurojson.org/jnifti)** by Qianqian
  Fang — defines the JSON / BJData wrapping of NIfTI-1/2 metadata and
  voxel data used by siamize for `.jnii` / `.bnii` containers. JNIfTI
  is part of the [NeuroJSON project](https://neurojson.org); the
  underlying [JData](https://neurojson.org/jdata) annotated-array
  encoding (`_ArrayType_`, `_ArraySize_`, `_ArrayZipData_`, …) and
  [BJData](https://neurojson.org/bjdata) binary container are
  implemented in `src/jnifti_io.cpp` on top of the vendored
  nlohmann/json header described above.

### Test data

The bundled test image `tests/sub-01_T1w.nii.gz` is the `sub-01` anatomical
T1-weighted scan from [OpenNeuro ds000001 v1.0.0](https://openneuro.org/datasets/ds000001/versions/1.0.0),
redistributed here under its original **CC0** public-domain dedication.
See `tests/README.md` for details.
