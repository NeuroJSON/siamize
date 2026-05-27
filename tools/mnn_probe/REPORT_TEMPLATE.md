# MNN 3.5+ SIAM Probe Report

Copy this template to `tools/mnn_probe/reports/<YYYY-MM-DD>-<hostname>.md`
and fill in the blanks after running `./run_all.sh`.

## Run metadata

- **Date**: YYYY-MM-DD
- **Host**: hostname
- **OS**: e.g. Ubuntu 22.04 / macOS 14.5
- **CPU**: model + core count
- **GPU**: model + VRAM (or "N/A — CPU-only host")
- **MNN version**: `MNNConvert --version` output
- **siamize commit**: `git rev-parse HEAD` from the siamize repo
- **ONNX source**: `fold_0_fp16.onnx` from `doc=dynshape` / `doc=fixshape` / local
- **MNN install method**: `pip install MNN==3.5.0` or source build (give cmake flags)

## Stage 1 — Conversion

- **Verdict**: PASS / PARTIAL / FAIL
- **Output `.mnn` size**: NN MB
- **Conversion wall**: N s
- **Notable warnings** (paste relevant `grep` output from `convert.log`):

```
(paste here)
```

## Stage 2 — Op histogram

- **Verdict**: PASS / PARTIAL / FAIL
- **Total runtime op count**: N
- **3D ops surviving**: (list — should be empty)

**Top-10 op-type histogram:**

```
(paste output of: jq -r '.oplists[].type' fold_0.json | sort | uniq -c | sort -rn | head -10)
```

**Notes** (e.g. InstanceNorm3D survived as `BatchInstanceNorm`, Resize3D decomposed to three 1D Interps, etc.):

## Stage 3 — CPU parity (ORT vs MNN CPU)

- **Verdict**: PASS / PARTIAL / FAIL
- **max\|diff\|**:
- **mean\|diff\|**:
- **argmax agreement**: %
- **ORT CPU wall**: N s
- **MNN CPU wall**: N s

**Notes**:

## Stage 4 — OpenCL parity (MNN CPU vs MNN OpenCL)

- **Verdict**: PASS / FAIL / SILENT-FALLBACK
- **OpenCL JIT-compile time** (first run): N s
- **OpenCL warm wall**: N s
- **max\|diff\| (vs MNN CPU)**:
- **argmax agreement**: %
- **GPU utility observed during run**:
  - peak GPU%: (per `nvidia-smi dmon` / `intel_gpu_top` / `radeontop`)
  - sustained GPU%:
  - clearly using GPU? yes / no / unclear

**Notes**:

## Stage 5 — End-to-end performance

| Backend | Wall (s) | Ratio vs ORT CPU |
|---|---|---|
| ORT CPU | | 1.00 (baseline) |
| MNN CPU | | |
| MNN OpenCL | | |

- **MNN OpenCL speedup vs MNN CPU**: Nx
- **Verdict**: STRONG PASS / MARGINAL / FAIL / SIDE-FAIL
- **End-to-end label-volume agreement (ORT CPU vs MNN OpenCL)**: %

**Notes**:

## Final verdict

- [ ] GO — integrate MNN into siamize as a vendor-neutral GPU backend (~3-4 days)
- [ ] NO-GO — MNN doesn't meet siamize's requirements at this MNN version
- [ ] CONDITIONAL — specific gaps to fix upstream; revisit when MNN N.M lands

**Recommendation** (1-2 paragraphs):

## Followups

- [ ] File MNN GitHub issue: (link)
- [ ] Re-probe at MNN version Y.Z when (specific op) lands OpenCL kernel
- [ ] Test on (other host class) — e.g. AMD RDNA3 / Intel Arc / Apple Silicon iGPU
