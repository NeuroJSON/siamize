# siamize — two GPU backends in one image, ready to run out of the box:
#   siamize         ONNX Runtime / CUDA EP   (default ENTRYPOINT)
#   siamize-opencl  MNN / OpenCL backend     (vendor-neutral; NVIDIA/AMD/Intel)
#
# Base: CUDA 12 + cuDNN 9 (the `-cudnn-` runtime tag bundles cuDNN 9 on
# CUDA 12.4+). ORT 1.26 GPU links libcudart.so.12 / libcublas*.so.12 /
# libcufft.so.11 / libcurand.so.10 / libcudnn.so.9 — all provided by that
# image. The MNN/OpenCL binary is statically linked (~9 MB, no libMNN.so)
# and needs neither CUDA nor cuDNN; it dlopen's the OpenCL ICD at runtime.
#
# Build (run from the repo root; the build context is the repo, this Dockerfile
# lives in docker/). First build compiles MNN from source -> ~15-20 min in the
# builder stage; that stage is discarded, so it doesn't affect final image size:
#   make dockerimg                       # convenience wrapper (DOCKER_IMG=... to tag)
#   # or directly:
#   docker build -f docker/Dockerfile -t <user>/siamize:cuda12 .
#   # pin a different CUDA minor if your driver is older:
#   docker build -f docker/Dockerfile --build-arg CUDA_VERSION=12.4.1 -t <user>/siamize:cuda12 .
#
# Run (needs the NVIDIA Container Toolkit; --gpus all exposes the GPU):
#   # ONNX Runtime / CUDA (default entrypoint):
#   docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
#       <user>/siamize:cuda12 \
#       -i /data/in.nii.gz -o /data/out.nii.gz -M 0,1,2,3,4 -c cuda
#   # MNN / OpenCL (override the entrypoint):
#   docker run --rm --gpus all -v "$PWD":/data -v siamize-cache:/cache \
#       --entrypoint siamize-opencl <user>/siamize:cuda12 \
#       -i /data/in.nii.gz -o /data/out.nii.gz -M 0,1,2,3,4 -c opencl
#   # fold weights auto-download from NeuroJSON into /cache on first run;
#   # mount a named volume (as above) to persist them across runs.
#
# Push to Docker Hub:
#   docker login
#   docker push <user>/siamize:cuda12
#
# CPU-only hosts still work: drop --gpus all and pass `-c cpu` (the CUDA EP
# reports unavailable and falls back; MNN likewise falls back to its CPU
# backend).

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04

###############################################################################
# Stage 1 — build siamize against the prebuilt ONNX Runtime GPU package.
# siamize links only libonnxruntime.so (ORT dlopen's the CUDA provider at
# runtime), so the build needs no CUDA toolkit — a plain Ubuntu builder with
# the same glibc as the runtime image suffices.
###############################################################################
FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates tar unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Pull the ONNX Runtime GPU prebuilt (CUDA 12 build, version pinned in
# scripts/fetch_deps.sh -> ORT_VERSION, currently 1.26.0). zmat is vendored.
RUN ORT_BUILD=gpu scripts/fetch_deps.sh

# SIAMIZE_BACKEND=ort + SIAMIZE_GPU=cuda -> ONNX Runtime CUDA Execution
# Provider. STATIC_LINK folds siamize's own deps in so only the ORT shared
# libs need shipping alongside the binary.
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSIAMIZE_BACKEND=ort \
        -DSIAMIZE_GPU=cuda \
        -DSIAMIZE_STATIC_LINK=ON \
 && cmake --build build --config Release --parallel "$(nproc)"

# Also build the MNN/OpenCL backend (siamize-opencl). fetch_mnn.sh downloads
# and compiles the patched MNN (NeuroJSON/MNN, default ref v3.5-opencl-conv3d)
# as a static libMNN.a; MNN bundles its own OpenCL headers and dlopen's the
# ICD at runtime (MNN_USE_SYSTEM_LIB=OFF), so no OpenCL dev package is needed
# to build. STATIC_LINK folds libMNN.a in -> a self-contained ~9 MB binary
# with no libMNN.so to ship. This MNN compile is the slow part of the build.
RUN MNN_STATIC=1 scripts/fetch_mnn.sh
RUN cmake -S . -B build-mnn \
        -DCMAKE_BUILD_TYPE=Release \
        -DSIAMIZE_BACKEND=mnn \
        -DSIAMIZE_STATIC_LINK=ON \
 && cmake --build build-mnn --config Release --parallel "$(nproc)"

###############################################################################
# Stage 2 — minimal runtime: CUDA 12 + cuDNN 9, plus the binary and the ORT
# shared libs it loads (core + CUDA EP). The TensorRT provider is intentionally
# omitted (SIAMIZE_GPU=cuda doesn't load it).
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# curl + CA certs: siamize shells out to `curl` to auto-download fold weights
# from NeuroJSON on first run.
# ocl-icd-libopencl1: the OpenCL ICD loader (libOpenCL.so.1) that the MNN
# backend dlopen's.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

# Register the NVIDIA OpenCL ICD. With `--gpus` the NVIDIA Container Toolkit
# injects the driver itself (libnvidia-opencl.so.1) but NOT this registration
# file, so without it the ICD loader finds zero platforms and MNN falls back
# to CPU. The file just names the driver lib; libnvidia-opencl.so.1 is present
# only at runtime under --gpus, so this is a no-op (no platform) on CPU-only
# runs. Verified: with this file, clinfo enumerates the NVIDIA GPU in-container.
#
# Also symlink the UNVERSIONED libOpenCL.so -> libOpenCL.so.1. The
# ocl-icd-libopencl1 runtime package ships only the versioned .so.1, but MNN's
# loader (OpenCLWrapper.cpp) dlopen's the bare "libOpenCL.so" name. Without the
# symlink MNN can't load the ICD and `siamize-opencl -c opencl` silently falls
# back to CPU even though the GPU enumerates fine. (Equivalent to installing
# ocl-icd-opencl-dev, but without pulling the headers.)
RUN mkdir -p /etc/OpenCL/vendors \
 && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
 && ln -sf libOpenCL.so.1 /usr/lib/x86_64-linux-gnu/libOpenCL.so

# ONNX Runtime / CUDA binary + the ORT libs it loads (core + CUDA EP).
COPY --from=builder /src/build/siamize                                          /opt/siamize/bin/siamize
COPY --from=builder /src/third_party/onnxruntime/lib/libonnxruntime.so*         /opt/siamize/lib/
COPY --from=builder /src/third_party/onnxruntime/lib/libonnxruntime_providers_shared.so /opt/siamize/lib/
COPY --from=builder /src/third_party/onnxruntime/lib/libonnxruntime_providers_cuda.so   /opt/siamize/lib/
# MNN / OpenCL binary (static; no extra libs).
COPY --from=builder /src/build-mnn/siamize                                      /opt/siamize/bin/siamize-opencl

ENV PATH=/opt/siamize/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/siamize/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
# Ensure the NVIDIA Container Toolkit exposes the OpenCL ICD (covered by
# `compute`) in addition to CUDA. This is the default on nvidia/cuda images;
# set explicitly so the MNN/OpenCL path works regardless of base defaults.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Default weight cache; mount a volume here to persist downloads.
ENV SIAMIZE_CACHE_DIR=/cache
RUN mkdir -p /cache
VOLUME ["/cache"]

ENTRYPOINT ["siamize"]
CMD ["--help"]
