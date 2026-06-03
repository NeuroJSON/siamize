![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# siamize — native C++ port of SIAM v0.3 brain/head MRI segmentation

* **Copyright**: (C) Qianqian Fang (2026) \<q.fang at neu.edu>
* **License**: Apache License, Version 2.0
* **Version**: 0.2.0
* **GitHub**: [https://github.com/NeuroJSON/siamize](https://github.com/NeuroJSON/siamize)
* **Upstream**: [https://github.com/romainVala/SIAM](https://github.com/romainVala/SIAM) — SIAM v0.3 by Valabregue, Khemir, Bardinet, Rousseau, Auzias & Dorent (2026), [arXiv:2605.02737](https://arxiv.org/abs/2605.02737)
* **Acknowledgement:** This project uses resources and data formats developed as part of the [NeuroJSON project](https://neurojson.org) supported by US National Institute of Health (NIH) grant [U24-NS124027](https://reporter.nih.gov/project-details/10308329).

Native C++ port of [**SIAM v0.3**](https://github.com/romainVala/SIAM) (*Segment
It All Model*) for head/brain tissue segmentation — runs with **no PyTorch /
nnU-Net** at inference time. One image, two GPU backends:

| Binary (entrypoint) | Backend | Hardware |
|---|---|---|
| `siamize` *(default)* | ONNX Runtime + CUDA EP | NVIDIA GPU, or CPU |
| `siamize-opencl` | MNN + OpenCL | NVIDIA / AMD / Intel GPU, or CPU |

Input: NIfTI (`.nii`/`.nii.gz`) or JNIfTI (`.jnii`/`.bnii`). Output: a uint8
labelmap (18 SIAM classes) or a 4D tissue-probability map. Fold weights
**auto-download** from NeuroJSON on first run — mount a volume at `/cache` to
keep them.

## Quick start

`--gpus all` exposes the GPU (and the NVIDIA OpenCL driver). Mount your data at
`/data` and a named cache volume at `/cache`.

```bash
# NVIDIA GPU, ONNX Runtime / CUDA — 5-fold ensemble (default entrypoint)
docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
    openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/labels.nii.gz -M 0,1,2,3,4 -c cuda

# Single fold (faster, lighter)
docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
    openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/labels.nii.gz -M 0 -c cuda

# Vendor-neutral GPU via MNN / OpenCL (NVIDIA / AMD / Intel) — override entrypoint
docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
    --entrypoint siamize-opencl openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/labels.nii.gz -M 0 -c opencl

# CPU only (no GPU needed — drop --gpus)
docker run --rm -v "$PWD":/data -v siamize-cache:/cache \
    openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/labels.nii.gz -M 0 -c cpu

# List the OpenCL devices the container sees (then pick one with -G N)
docker run --rm --gpus all --entrypoint siamize-opencl \
    openjdata/siamize:v2026.6 --list-gpu

# 4D tissue-probability map (float32 softmax) instead of a labelmap
docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
    openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/tpm.nii.gz -M 0,1,2,3,4 -c cuda --tpm

# SPM12-style 6-class output (GM, WM, CSF, Bone, Soft, Air)
docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
    openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/spm.nii.gz -M 0 -c cuda -C spm

# JNIfTI output (JSON / binary, for the NeuroJSON ecosystem)
docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
    openjdata/siamize:v2026.6 \
    -i /data/in.nii.gz -o /data/labels.jnii -M 0 -c cuda -F jnii
```

## Common options

| Flag | Meaning |
|---|---|
| `-i FILE` | input volume (`.nii[.gz]` / `.jnii` / `.bnii`) |
| `-o FILE` | output file (extension picks the format, or use `-F`) |
| `-M 0,1,2,3,4` | fold weights (digits 0–4 = the 5-fold ensemble; auto-downloaded) |
| `-c {auto\|cuda\|cpu}` (ORT) / `{auto\|opencl\|cpu}` (`siamize-opencl`) | backend; `auto` picks GPU then falls back to CPU |
| `-G N` | GPU device. ORT: CUDA index. `siamize-opencl`: 1-based device from `--list-gpu` |
| `--list-gpu` | (`siamize-opencl`) list OpenCL devices and exit |
| `-t N` | CPU threads (default auto) |
| `-P ZxYxX` | sliding-window patch (default `256x256x192`; smaller = less memory) |
| `-u S` | target isotropic spacing in mm (default `0.75`) |
| `-C N\|spm` | output classes: `18` (default) or `spm` (6 TPM channels) |
| `--tpm` | write a 4D float32 tissue-probability map instead of a labelmap |
| `-F nii\|jnii\|bnii` | output container |
| `--lowmem` | force the low-memory preset (smaller patch + tighter knobs) |

Run `--help` (default entrypoint) for the full list:
```bash
docker run --rm openjdata/siamize:v2026.6 --help
```

## Volumes & environment

| Mount / var | Purpose |
|---|---|
| `-v "$PWD":/data` | your input/output files |
| `-v siamize-cache:/cache` | persists auto-downloaded fold weights (+ OpenCL tuning cache) across runs |
| `--gpus all` | **required for any GPU run** — exposes the GPU and the NVIDIA OpenCL driver |

To confirm an OpenCL run is on the GPU (not a silent CPU fallback): the log
shows `device=opencl` with **no** `OpenCL init error`, and `nvidia-smi` shows
GPU activity. CPU-only hosts work with `-c cpu` and no `--gpus`.

## Tags

Calendar-versioned `vYYYY.M` (e.g. `v2026.6`). The image is CUDA 12 + cuDNN 9
based; the MNN/OpenCL binary is static and vendor-neutral.

## Links

- **Source / docs:** https://github.com/NeuroJSON/siamize
- **Upstream SIAM v0.3:** https://github.com/romainVala/SIAM — [arXiv:2605.02737](https://arxiv.org/abs/2605.02737)
- **Please cite the SIAM paper** if you use this in your work.

**Acknowledgement:** uses resources and data formats from the
[NeuroJSON project](https://neurojson.org), supported by US NIH grant
[U24-NS124027](https://reporter.nih.gov/project-details/10308329).
