# MNN 3.5+ probe for SIAM v0.3

This directory holds the scripts for a **read-only, decision-gated probe**
that determines whether [MNN](https://github.com/alibaba/MNN) is a viable
alternative inference backend for siamize before committing to any C++
integration work.

The probe answers four questions in order, stopping at the first NO:

1. Does `MNNConvert` successfully ingest SIAM's ONNX?
2. Does the converted `.mnn` produce numerically-correct output on MNN's
   CPU backend? (compared to ORT as ground truth)
3. Does it run correctly on MNN's OpenCL backend without silent CPU
   fallback?
4. Is OpenCL acceleration meaningful (≥ 3× MNN CPU) on a vendor-neutral
   host?

If all four PASS, MNN becomes a viable second backend for siamize and
the ~3-4 days of integration work is justified. If any stage FAILs, we
know exactly which capability gap killed it, and the corresponding MNN
upstream issue is the natural follow-up.

## What this probe does NOT do

- No C++ integration into siamize.
- No multi-fold ensembling.
- No benchmarking across hardware classes (just whichever host runs the
  probe).
- No accuracy comparison against ground-truth labels — uses ORT as the
  per-fold logit oracle.

## Why the probe matters

MNN's stack reportedly went through a major architectural shift around
version 3.0: `Conv3D` and `ConvTranspose3D` got **deprecated as runtime
operators** and are now decomposed into 2D primitives by `MNNConvert` at
model-conversion time (per the `MNN_SUPPORT_DEPRECATED_OPV2` cmake-flag
docs). If that decomposition fires cleanly for SIAM's specific Conv3D
patterns, MNN's OpenCL backend becomes a credible vendor-neutral GPU
target despite its OpenCL kernels not natively implementing 3D conv.

The probe is the cheapest way to find out whether the decomposition
actually fires correctly for SIAM and what (if anything) is still
required at runtime as a real 3D op.

## Setup

A self-contained venv to keep MNN's torch/numpy from colliding with
siamize's pinned versions:

```bash
mkdir -p /tmp/mnn-probe && cd /tmp/mnn-probe
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install MNN==3.5.0 onnx onnxruntime numpy nibabel

# Confirm the CLI tools came with the pip wheel.
which MNNConvert MNNDump2Json || {
    echo "FALL BACK: pip wheel lacks the CLI tools."
    echo "Build MNN from source instead:"
    cat <<'BUILD'
        git clone --depth 1 --branch 3.5.0 \
            https://github.com/alibaba/MNN.git
        cd MNN && mkdir build && cd build
        cmake .. -DMNN_BUILD_CONVERTER=ON -DMNN_OPENCL=ON \
                 -DMNN_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
        make -j8
        export PATH="$PWD:$PATH"
BUILD
}

# Copy SIAM weights. Use the fixshape variant first (no dynamic-axes
# complication for the initial conversion test).
cp <siamize-repo>/models/fold_0_fp16.onnx /tmp/mnn-probe/fold_0.onnx
```

## Probe stages

Each stage is a separate script with a clear PASS / FAIL gate. Run them
in order; stop at the first FAIL. See `run_all.sh` for the wrapper that
chains them with the right environment variables.

### Stage 1 — `01_convert.sh` (~15 min)

Convert the ONNX to MNN format with verbose logging; grep the conversion
log for warning patterns that would signal unsupported ops.

**Gate:**

| Outcome | Verdict | Next |
|---|---|---|
| Clean conversion, no "unsupported op" warnings | PASS | Stage 2 |
| Specific 3D ops warned | PARTIAL | Stage 2 with caveat |
| Fatal "Convolution3D not supported" | FAIL | STOP |

### Stage 2 — `02_op_histogram.sh` (~5 min)

Dump the resulting `.mnn` to JSON and count op types; verify the 3D-to-2D
decomposition actually fired (no `Conv3D` / `ConvTranspose3D` should
survive at runtime).

**Gate:**

| Outcome | Verdict |
|---|---|
| Pure 2D Conv + Add + Reshape + Reduce + elementwise + 1D Interp | PASS |
| InstanceNorm3D / Resize3D survive | PARTIAL (will fall back to CPU on OpenCL) |
| Conv3D / ConvTranspose3D survive | FAIL — deprecation didn't fire |

### Stage 3 — `03_cpu_parity.py` (~15 min)

Run a single (1, 1, 256, 256, 192) tile through both ORT (CPU EP) and
MNN (CPU backend), compare logits and argmax.

**Gate:**

| Outcome | Verdict |
|---|---|
| `max|diff| < 1e-3`, argmax agreement ≥ 99.9% | PASS |
| Mild logit drift (~1-5%) but argmax ≥ 99% | PARTIAL |
| Argmax agreement < 99% | FAIL — conversion math diverges |

### Stage 4 — `04_opencl_parity.py` (~10 min)

Same script, OpenCL backend. While it runs, monitor GPU utilization in
another terminal (`nvidia-smi dmon -s u` / `intel_gpu_top -L` /
`radeontop -d -`) to verify the GPU is actually engaged and MNN isn't
silently falling back to CPU for some ops.

**Gate:**

| Outcome | Verdict |
|---|---|
| Output matches CPU MNN within fp16 noise; GPU utilization visibly active | PASS |
| Output diverges from CPU MNN | FAIL (OpenCL kernel bug for some decomposed op) |
| GPU shows 0% utilization | SILENT FALLBACK — investigate per-op |

### Stage 5 — `05_e2e_perf.py` (~20 min)

End-to-end timing on the bundled `sub-01_T1w.nii.gz` driven through
the siamize preprocessing → sliding-window → postprocessing pipeline,
on CPU vs OpenCL MNN backends. ORT CPU timing is captured for
reference.

**Gate:**

| MNN OpenCL vs MNN CPU speedup | Verdict | Recommendation |
|---|---|---|
| ≥ 5× | STRONG PASS | Worth integrating |
| 2-5× | MARGINAL | Useful only where CUDA isn't available |
| < 2× | FAIL | Decomposed graph too dispatch-heavy |
| (MNN CPU slower than ORT CPU by ≥ 1.5×) | side-fail | Stop before Stage 5 |

## Reporting

After running all relevant stages, fill in `REPORT_TEMPLATE.md` with the
host details, MNN version, decision-gate outcomes per stage, and a
GO / NO-GO verdict. Commit the filled-in report under
`tools/mnn_probe/reports/<YYYY-MM-DD>-<hostname>.md` so multiple-host
probe results accumulate as evidence.

## File layout

```
tools/mnn_probe/
├── README.md             # this file
├── 01_convert.sh         # Stage 1: MNNConvert + log grep
├── 02_op_histogram.sh    # Stage 2: dump-to-json + jq
├── 03_cpu_parity.py      # Stage 3: ORT vs MNN CPU on a synthetic tile
├── 04_opencl_parity.py   # Stage 4: MNN CPU vs MNN OpenCL on a tile
├── 05_e2e_perf.py        # Stage 5: full siamize pipeline through MNN
├── run_all.sh            # convenience wrapper for stages 1-5
├── REPORT_TEMPLATE.md    # fill in per host
└── reports/              # accumulated probe results, gitignored?
```

## Time budget

| Stage | Time |
|---|---|
| Setup (one-time per host) | 10 min |
| 1 Conversion | 15 min |
| 2 Op histogram | 5 min |
| 3 CPU parity | 15 min |
| 4 OpenCL parity | 10 min |
| 5 Performance | 20 min |
| **Total** | **~75 min** to a GO / NO-GO verdict |
