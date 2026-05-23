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
- [Weight cache](#weight-cache)
- [Layout](#layout)
- [Platforms](#platforms)
- [Bundled dependencies](#bundled-dependencies)
- [Citation](#citation)

---

## Overview

`mex/` ships the MATLAB / GNU Octave interface to siamize:

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
git submodule update --init  # populates mex/jsonlab
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

% pre-loaded jnifti struct, return labels in-memory
nii = loadnifti('input.nii.gz');
lab = siamize(nii);

% pure array, default centered affine inferred
lab = siamize(my_volume);
```

The wrapper auto-downloads each missing fold from
`https://neurojson.org/siamize/weights/siam_v03/` (overridable, see
[Weight cache](#weight-cache)).

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
siamize(inputfile, outputfile, models, opts)
siamize(img, affine, outputfile, models, opts)
siamize(array, outputfile, models, opts)   % synth header
```

Output extension picks the writer:

| Extension | Writer |
|---|---|
| `.nii`, `.nii.gz` | `jnii2nii(jnii_out, file)` |
| `.jnii` | `savejnifti(jnii_out, file)` (text JNIfTI) |
| `.bnii` | `savejnifti(jnii_out, file)` (binary JNIfTI) |

When the input is a file / jnifti struct / readnifti struct, the source
`NIFTIHeader` is preserved in the output (only `NIFTIData` is swapped to
the labels and `Affine` is overwritten with the working affine). When the
input is a bare array, `jnifticreate` builds a minimal header.

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

## Weight cache

The MEX wrapper and the CLI binary share one cache, so a download serves
both. The wrapper looks up each missing fold in this order:

1. **`$SIAMIZE_CACHE_DIR/<basename>`** — if the file already exists, used
   verbatim.
2. **Default cache path** if the env var is unset:
   - POSIX: `$HOME/.cache/siamize/models/`
   - Windows: `%LOCALAPPDATA%/siamize/models/`
3. **`$SIAMIZE_WEIGHTS_BASE_URL/<basename>.gz`** via MATLAB/Octave's
   `urlwrite`, with a gzip-magic-bytes sanity check before `gunzip`.
4. **`$SIAMIZE_WEIGHTS_BASE_URL/<basename>`** (raw .onnx) as a fallback if
   the .gz form 404s.

Default URL base: `https://neurojson.org/siamize/weights/siam_v03/`.

## Layout

```
mex/
├── README.md            # this file
├── siamize.m            # public dispatcher (loadjd, shortcuts, cache, save)
├── siamize_mex.cpp      # MEX entry point -> siamex.mex*
└── jsonlab/             # submodule: NeuroJSON jsonlab (loadjd/savejd/...)
```

The MEX C++ links the same `siamize_core` library as the CLI binary;
shared headers live under `src/` at the repo root.

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

## Bundled dependencies

- **[jsonlab](https://github.com/NeuroJSON/jsonlab)** by Qianqian Fang —
  pulled in as a git submodule under `mex/jsonlab/`. Provides
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
