# siamize/mex - MATLAB / GNU Octave bindings for siamize

[![ci](https://github.com/NeuroJSON/siamize/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuroJSON/siamize/actions/workflows/ci.yml)

* **Copyright**: (C) Qianqian Fang (2026) \<q.fang at neu.edu>
* **License**: Apache License, Version 2.0
* **Version**: 0.1.0
* **GitHub**: [https://github.com/NeuroJSON/siamize](https://github.com/NeuroJSON/siamize)
* **Parent toolbox**: [siamize](../README.md) — native C++/ONNX port of SIAM v0.3

---

## Table of Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Calling forms](#calling-forms)
- [File-in / file-out](#file-in--file-out)
- [Model selection](#model-selection)
- [Options](#options)
- [Weight cache](#weight-cache)
- [Layout](#layout)
- [Platforms](#platforms)
- [Bundled dependencies](#bundled-dependencies)
- [Citation](#citation)

---

## Overview

`matlab/` ships the MATLAB / GNU Octave interface to siamize:

1. **`siamex.mex*`** — a thin C++ MEX entry point that wraps the same
   preprocess → ORT inference → postprocess pipeline as the standalone CLI
   binary. Built against the bundled `siamize_core` sources, so MEX-side and
   CLI predictions are bit-identical.
2. **`siamize.m`** — a pure MATLAB/Octave dispatcher. Handles model name
   shortcuts, weight auto-download, jsonlab path injection, and accepts
   inputs/outputs as files, jnifti structs, readnifti-style structs, or bare
   arrays. It forwards the canonicalized call to `siamex`.
3. **`jsonlab/`** (git submodule) — bundled NeuroJSON
   [jsonlab](https://github.com/NeuroJSON/jsonlab), providing `loadjd`,
   `savejd`, `loadnifti`, `jnii2nii`, `savejnifti`, `jnifticreate`, the
   `zlib*` / `gzip*` / `base64*` codecs, and the JSON/BJData parsers. After a
   plain `git clone`, run `git submodule update --init` (or clone with
   `--recurse-submodules`) so this directory is populated.

The MEX uses the same shared weight cache as the CLI binary (see
[Weight cache](#weight-cache)), so one download serves both. The
`.mex*` and the `.m` wrapper are versioned together with the rest of the
repo.

## Quickstart

### 1. Fetch C++ deps (only needed once)

```bash
scripts/fetch_deps.sh        # downloads ORT prebuilt + clones nifti_clib
git submodule update --init  # populates matlab/jsonlab
```

### 2. Build the MEX

GNU Octave (Linux/macOS):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSIAMIZE_BUILD_OCTAVE_MEX=ON
cmake --build build -j
# -> build/siamex.mex
```

MATLAB (Linux/macOS/Windows):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSIAMIZE_BUILD_MATLAB_MEX=ON
cmake --build build -j
# -> build/siamex.mexa64  (Linux)
#    build/siamex.mexmaca64 (macOS arm64)
#    build/Release/siamex.mexw64 (Windows)
```

The MEX dlopens `libonnxruntime` from the ORT prebuilt under
`third_party/onnxruntime/lib/` (Linux/macOS) or `bin/` (Windows). For
deployment, drop `siamex.mex*`, `siamize.m`, the `jsonlab/` directory, and
the matching `libonnxruntime.{so,dylib,dll}` into a single folder; users
just `addpath()` it.

### 3. Run

```matlab
% one-shot file -> file
siamize('input.nii.gz', 'labels.nii.gz');

% read .nii.gz, write binary JNIfTI, full 5-fold ensemble
siamize('input.nii.gz', 'labels.bnii', 0:4);

% pre-loaded jnifti struct, get a jnifti struct back in memory
nii_in  = loadnifti('input.nii.gz');
nii_out = siamize(nii_in);
% nii_out.NIFTIData is uint8 [X, Y, Z] (the labelmap).
% nii_out.NIFTIHeader is cloned from nii_in's header.

% pure array, default centered affine inferred
nii_out = siamize(my_volume);

% TPM mode: nii_out.NIFTIData becomes 4D single (float32) [X, Y, Z, 18]
nii_tpm = siamize('input.nii.gz', 0:4, 'tpm', true, 'tpm_t', 1.5);
```

`siamize` always returns a single jnifti struct (`nii.NIFTIHeader` +
`nii.NIFTIData`). The dtype/rank of `nii.NIFTIData` depends on
`opts.tpm`:

| opts.tpm | nii.NIFTIData class | nii.NIFTIData size |
|---|---|---|
| `false` (default) | `uint8` | `[X Y Z]` (the labelmap) |
| `true` | `single` | `[X Y Z num_classes]` (softmax probs, sum = 1 per voxel) |

The wrapper auto-downloads each missing fold from the NeuroJSON URL
listed under [Weight cache](#weight-cache) (overridable via
`SIAMIZE_WEIGHTS_BASE_URL`).

## Calling forms

`siamize.m` accepts a flexible first argument:

| First arg | Interpretation | Affine source |
|---|---|---|
| `'file.nii'` / `'file.nii.gz'` / `'file.jnii'` / `'file.bnii'` | path read via `loadjd` | jsonlab parser |
| jnifti struct (`.NIFTIData` + `.NIFTIHeader.Affine`) | passthrough | `.NIFTIHeader.Affine` |
| jnifti struct without `.Affine` | passthrough | centered identity |
| readnifti struct (`.img` + `.hdr` with `srow_x/y/z`) | from `loadnifti(file,'nii')` | stacked sform rows |
| 3D numeric array (no following affine arg) | passthrough | centered identity |
| 3D numeric array + 3x4 / 4x4 matrix | passthrough | matrix arg |

The centered-identity default places the world origin on the volume's
centre voxel:

```
A = [1 0 0 -(Nx-1)/2;
     0 1 0 -(Ny-1)/2;
     0 0 1 -(Nz-1)/2]
```

The translation does not affect predicted labels — siamize only consumes
axes orientation + voxel spacing — so the default is cosmetic for header
round-tripping when the user supplies a bare array.

## File-in / file-out

```matlab
siamize(inputfile, outputfile)
siamize(inputfile, outputfile, models)
siamize(inputfile, outputfile, models, opts...)         % name/value pairs
siamize(img, affine, outputfile, models, opts...)
siamize(array, outputfile, models, opts...)             % synth header

% TPM output works the same way -- writer infers dtype/rank from
% nii.NIFTIData, so the same call writes a 4D file when opts.tpm is true:
siamize('input.nii.gz', 'tpm.nii.gz', 0:4, 'tpm', true);
```

Output extension picks the writer (works equally for 3D uint8 labels
and 4D float32 TPMs):

| Extension | Writer |
|---|---|
| `.nii`, `.nii.gz` | `jnii2nii(nii_out, file)` |
| `.jnii` | `savejnifti(nii_out, file)` (text JNIfTI) |
| `.bnii` | `savejnifti(nii_out, file)` (binary JNIfTI) |

When the input is a file / jnifti struct / readnifti struct, the source
`NIFTIHeader` is preserved in the output (only `NIFTIData` is swapped to
the segmentation result and `Affine` is overwritten with the working
affine). When the input is a bare array, `jnifticreate` builds a
minimal header.

The 2-arg ambiguity (output filename vs model spec) resolves by extension:
`siamize(in, 'out.nii.gz')` writes a file, `siamize(in, '0')` runs fold 0.

## Model selection

`models` accepts numeric indices, char shortcuts, full filenames, or
mixes thereof:

```matlab
siamize(in, out, 0)              % single-fold fold_0
siamize(in, out, 0:4)            % full 5-fold ensemble
siamize(in, out, '0,2,4')        % comma string shortcut
siamize(in, out, {'0','2'})      % cellstr shortcut
siamize(in, out, 'fold_0_fp16.onnx,fold_1_fp16.onnx')
siamize(in, out, {'fold_0_fp16.onnx','/abs/path/fold_1_fp16.onnx'})
```

Single-digit `0..9` expands to `fold_<N>_fp16.onnx`; anything else is
treated as a filename / path.

## Options

Options live in `opts`, which is the variadic tail after `models`.
You can pass either a struct, a sequence of `'name', value` pairs, or
any mix thereof — jsonlab's `varargin2struct` merges them into a
single lowercase-keyed struct. The recognized keys mirror the
siamize CLI long flags (hyphens become underscores in MATLAB):

| Key | CLI counterpart | Value |
|---|---|---|
| `'compute'` | `-c / --compute` | `'auto'` (default), `'cpu'`, `'cuda'`, `'tensorrt'` |
| `'thread'` | `-t / --thread` | int (default 0 = all available cores) |
| `'gpu'` | `-G / --gpu` | int (0-based CUDA device id) |
| `'patch'` | `-P / --patch` | `[pz, py, px]` (default `[256 256 192]`) |
| `'spacing'` | `-u / --spacing` | double mm (default 0.75) |
| `'classes'` | `-C / --classes` | int (default 18, matches SIAM v0.3) |
| `'trt_cache'` | `--trt-cache-dir` | char (default `~/.cache/siamize/trt`) |
| `'verbose'` | `-v / --verbose` | logical (default false) |
| `'cudnn_max_workspace'` | `--cudnn-max-workspace` | 0 or 1 (default 1) |
| `'arena_extend'` | `--arena-extend` | `'power'` (default) or `'same'` |
| `'cudnn_algo'` | `--cudnn-algo` | `'default'` / `'heuristic'` / `'exhaustive'` |
| `'gpu_mem_limit'` | `--gpu-mem-limit` | bytes (double, e.g. `6*1024^3` for 6 GB) |
| `'tpm'` | `--tpm` | logical (default false). When true, `nii.NIFTIData` is a 4D single TPM instead of 3D uint8 labels. |
| `'tpm_t'` | `--tpm-t` | softmax temperature (default 1.0). Only meaningful when `tpm` is true. |

Examples mixing the three opts-passing styles:

```matlab
% struct
siamize('in.nii.gz', 'lab.nii.gz', 0:4, struct('compute', 'cuda', 'verbose', 1));

% name/value pairs
siamize('in.nii.gz', 'lab.nii.gz', 0:4, 'compute', 'cuda', 'verbose', 1);

% struct + override
defs = struct('compute', 'cuda', 'tpm_t', 1.5);
siamize('in.nii.gz', 'tpm.nii.gz', 0:4, defs, 'tpm', true);

% tight-VRAM GPU recipe (8 GB consumer laptop card):
siamize('in.nii.gz', 'lab.nii.gz', 0:4, ...
        'compute', 'cuda', ...
        'cudnn_max_workspace', 0, ...
        'arena_extend', 'same', ...
        'patch', [192 192 128]);

% multi-GPU box, pick GPU 1 of N:
siamize('in.nii.gz', 'lab.nii.gz', 0:4, 'compute', 'cuda', 'gpu', 1);
```

## Weight cache

The MEX wrapper and the CLI binary share one cache, so a download serves
both. The wrapper looks up each missing fold in this order:

1. **`$SIAMIZE_CACHE_DIR/<basename>`** — if the file already exists, used
   verbatim.
2. **Default cache path** if the env var is unset:
   - POSIX: `$HOME/.cache/siamize/models/`
   - Windows: `%LOCALAPPDATA%/siamize/models/`
3. **`$SIAMIZE_WEIGHTS_BASE_URL` + `<basename>.gz`** (the basename is
   appended directly with no `/` separator since the prefix is expected
   to end with the parameter that takes the filename, e.g. NeuroJSON's
   `&file=`). Fetched via MATLAB/Octave's `urlwrite`, with a
   gzip-magic-bytes sanity check before `gunzip`.
4. **`$SIAMIZE_WEIGHTS_BASE_URL` + `<basename>`** (raw .onnx) as a
   fallback if the .gz form 404s.

Default URL prefix:

```
https://neurojson.org/io/stat.cgi?action=get&db=siam_v03&doc=dynshape&size=95360591&file=
```

(`size=` is informational and not validated by the server, so the same
constant is reused for every fold.)

## Layout

```
matlab/
├── README.md            # this file
├── siamize.m            # public dispatcher (loadjd, shortcuts, cache, save)
├── jsonlab/             # submodule: NeuroJSON jsonlab (loadjd/savejd/...)
└── tests/               # unit tests for siamize.m (Octave + MATLAB)
```

The MEX C++ entry point (`src/siamize_mex.cpp`) lives next to the rest of
the C++ sources at `src/`; CMake's `matlab_add_mex` / `mkoctfile` rules
compile it together with the shared `siamize_core` translation units, so
MEX-side and CLI predictions stay bit-identical.

## Platforms

Built and smoke-tested by CI on:

| Host | Octave | MATLAB |
|---|---|---|
| Linux x86_64 | apt `octave` + `liboctave-dev` | `R2024a` via `matlab-actions/setup-matlab@v2` |
| Windows x64 | — | `R2024a` |

macOS support follows the same CMake recipe but isn't pinned in CI yet —
please open an issue if a host setup breaks. The MEX is portable C++17 +
MEX C API + OpenMP, with no Octave-specific or MATLAB-specific glue beyond
the standard `mexFunction` entry point.

### Linux MATLAB: GLIBCXX symbol mismatch

MATLAB R2024a on Linux bundles a `libstdc++.so.6` that supports up to
`GLIBCXX_3.4.28` (GCC 9 / 10). If you build the MEX with a newer GCC
(11+), the resulting binary references `GLIBCXX_3.4.29` and MATLAB
refuses to load it:

```
Invalid MEX-file '.../siamex.mexa64':
  /opt/MATLAB/.../sys/os/glnxa64/libstdc++.so.6:
  version `GLIBCXX_3.4.29' not found (required by siamex.mexa64)
```

The right fix is to build with the MathWorks-supported GCC version
(GCC 10.2.1 for R2024a):

```bash
sudo apt-get install -y g++-10 gcc-10
CC=gcc-10 CXX=g++-10 cmake -S . -B build -DSIAMIZE_BUILD_MATLAB_MEX=ON
cmake --build build -j
```

The resulting MEX only references GLIBCXX symbols MATLAB's bundled
libstdc++ already has, so it loads with no environment tweaks. CI
follows this recipe on the linux-matlab leg.

Two approaches that **don't** work and are worth not wasting time on:

* **Statically linking libstdc++ into the MEX** (`-static-libstdc++`).
  A second C++ runtime inside a MEX fights MATLAB's already-loaded
  libstdc++ over `type_info`, vtables, and `std::cout`, and aborts on
  MEX load with "MATLAB is exiting because of fatal error".
* **LD_PRELOAD-ing the system libstdc++.** Works for a smoke load but
  forces every MATLAB launch into a non-default config and breaks any
  MathWorks toolbox compiled against the bundled libstdc++.

## Bundled dependencies

- **[jsonlab](https://github.com/NeuroJSON/jsonlab)** by Qianqian Fang —
  pulled in as a git submodule under `matlab/jsonlab/`. Provides
  `loadjd` / `savejd` (extension-dispatched I/O), `loadnifti`,
  `jnii2nii`, `savejnifti`, `jnifticreate`, `nifticreate`, `niiheader2jnii`,
  `niicodemap`, the `loadjson` / `savejson` / `loadbj` / `savebj`
  JSON/BJData parsers, and the `zlib*` / `gzip*` / `base64*` codecs that
  back the JNIfTI binary container. jsonlab is part of the
  [NeuroJSON project](https://neurojson.org), supported by US NIH grant
  [U24-NS124027](https://reporter.nih.gov/project-details/10308329).
  Apache-2.0 licensed (per its `LICENSE_Apache-2.0.txt`).

## Citation

If you use `siamize` (any binding) in your work, **please cite the
original SIAM paper** — see the [top-level README](../README.md#citation)
for the BibTeX. The MATLAB/Octave bindings are part of the same `siamize`
software project; the optional secondary citation in the parent README
covers them.
