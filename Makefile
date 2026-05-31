# Convenience top-level Makefile for siamize. Wraps the CMake build and adds
# code-formatting / cleanup / packaging targets. CMake is still the primary
# build system (this Makefile just shells out to it).
#
# Targets:
#
#   make                  configure + build CPU CLI (Release) under build/
#   make cuda             configure + build CUDA CLI (re-fetches GPU ORT if needed)
#   make tensorrt         configure + build TensorRT CLI (re-fetches GPU ORT if needed)
#
#   make opencl           configure + build the MNN-backed CLI with the OpenCL
#                         backend enabled (NVIDIA via ICD, AMD, Intel iGPU,
#                         Mali, Adreno). First run also builds libMNN via
#                         scripts/fetch_mnn.sh; pass MNN_STATIC=1 to get a
#                         self-contained binary (no libMNN.so to ship).
#   make openclmex        MATLAB MEX, MNN backend (siamex.mex<a64|maca64|w64>)
#   make opencloct        Octave  MEX, MNN backend (siamex.mex)
#
#   make mex-octave       build the Octave MEX (siamex.mex)
#   make mex-matlab       build the MATLAB MEX (siamex.mex{a64,maca64,w64})
#   make mex-test         run matlab/tests/run_tests.m in Octave (30 unit tests)
#
#   make package          stage + zip the CPU CLI bundle
#   make package-cuda     stage + zip the CUDA CLI bundle (needs `make cuda` first)
#   make package-tensorrt stage + zip the TRT  CLI bundle (needs `make tensorrt`)
#   make package-mex      stage + zip the MEX bundle      (needs `make mex-*`)
#
#   make test             run tests/run_regression.sh (needs models/ populated)
#
#   make doc              build doxygen HTML docs -> build/doc/html/index.html
#   make doc-clean        rm -rf build/doc/
#
#   make pretty           astyle on C++, black on Python
#   make pretty-cpp / pretty-py
#
#   make clean            rm -rf build/   (keeps third_party/)
#   make distclean        rm -rf build/ third_party/onnxruntime/
#
# Notes:
#   * The CPU and GPU ORT prebuilts share `third_party/onnxruntime/`, so
#     switching between `make` and `make cuda` re-fetches ORT only if the
#     current install doesn't have the right provider plugins.
#   * `make package*` runs `scripts/package.sh` which uses 7z; on macOS
#     you may need `brew install p7zip` if 7z isn't already present.

BUILD_DIR  ?= build
BUILD_TYPE ?= Release
ORT_DIR    := third_party/onnxruntime
MNN_DIR    := third_party/mnn

# Filenames used to detect which ORT prebuilt (CPU vs GPU) is currently
# installed under third_party/onnxruntime/.
ORT_GPU_MARKER := $(ORT_DIR)/lib/libonnxruntime_providers_cuda.so
ORT_GPU_MARKER_DLL := $(ORT_DIR)/lib/onnxruntime_providers_cuda.dll

# MNN stage marker. fetch_mnn.sh writes either libMNN.a (MNN_STATIC=1)
# or libMNN.{so,dylib,dll}; checking the include header is the cheapest
# stable signal that the stage is populated.
MNN_MARKER := $(MNN_DIR)/include/MNN/Interpreter.hpp

.PHONY: all build cuda tensorrt mex-octave mex-matlab mex-test \
        package package-cuda package-tensorrt package-mex \
        package-opencl package-openclmex \
        cudaoct cudamex coreml coremloct coremlmex \
        opencl openclmex opencloct mnn-deps \
        ort-cpu ort-gpu clean distclean pretty pretty-cpp pretty-py test \
        doc doc-clean

# ---- CLI builds -------------------------------------------------------------

all: build

build: ort-cpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel

cuda: ort-gpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DSIAMIZE_GPU=cuda
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo
	@echo "[make cuda] built $(BUILD_DIR)/siamize with CUDA EP."
	@echo "Runtime: see README.md > 'Optional: NVIDIA GPU build' for the"
	@echo "LD_LIBRARY_PATH one-liner that exposes CUDA/cuDNN libs to ORT."

tensorrt: ort-gpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DSIAMIZE_GPU=tensorrt
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo
	@echo "[make tensorrt] built $(BUILD_DIR)/siamize with TensorRT + CUDA EPs."

# macOS-only. CoreML EP is statically baked into ORT 1.26's macOS dylib
# (no separate provider plugin to fetch), so we use the standard CPU
# ORT bundle. -DSIAMIZE_GPU=coreml turns on the SIAMIZE_HAS_COREML
# define and links the CoreML / Foundation frameworks. Runtime selects
# CPU / GPU / Neural Engine via --coreml-units (default 'all').
coreml: ort-cpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DSIAMIZE_GPU=coreml
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo
	@echo "[make coreml] built $(BUILD_DIR)/siamize with CoreML EP."
	@echo "First run compiles the ONNX -> .mlmodelc (~10-30 s, cached)."

# ---- MNN-backed builds ------------------------------------------------------
# `opencl` is the user-facing name for the MNN backend's headline GPU
# path (also covers Vulkan / Metal at runtime via -c). Build-time, this
# is `-DSIAMIZE_BACKEND=mnn` -- MNN's OpenCL backend gets enabled at
# fetch_mnn.sh time (MNN_OPENCL=1, default). The CLI's -c flag then
# picks the actual runtime (cpu | opencl | vulkan | metal).
#
# MNN_STATIC=1 builds libMNN.a and produces a self-contained binary.
# Pass it on the make line and it propagates to scripts/fetch_mnn.sh.
opencl: mnn-deps
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DSIAMIZE_BACKEND=mnn
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo
	@echo "[make opencl] built $(BUILD_DIR)/siamize with MNN backend."
	@echo "Runtime: pass -c {cpu|opencl|vulkan|metal}. Default auto picks"
	@echo "OpenCL when MNN was built with MNN_OPENCL=ON, else CPU."

# ---- MNN prebuilt management ------------------------------------------------

# Build (or skip if cached) the MNN runtime under third_party/mnn/.
# fetch_mnn.sh is itself idempotent (skips when the stage is populated
# unless FORCE=1), but having a Make-level marker dep is faster than
# spawning bash every invocation.
mnn-deps:
	@if [ -f $(MNN_MARKER) ]; then \
	    echo "[mnn] already staged under $(MNN_DIR)"; \
	else \
	    echo "[mnn] building libMNN (this takes ~15-20 min first time)"; \
	    scripts/fetch_mnn.sh; \
	fi

# ---- ORT prebuilt management ------------------------------------------------

# CPU ORT: fetch only if no CPU-only install is present. If a GPU build is
# currently installed, wipe build/ + ORT and refetch.
ort-cpu:
	@if [ -d $(ORT_DIR) ] && [ ! -e $(ORT_GPU_MARKER) ] && [ ! -e $(ORT_GPU_MARKER_DLL) ]; then \
	    echo "[ort] CPU ORT already installed under $(ORT_DIR)"; \
	else \
	    echo "[ort] (re)installing CPU ORT prebuilt"; \
	    rm -rf $(ORT_DIR) $(BUILD_DIR); \
	    scripts/fetch_deps.sh; \
	fi

# GPU ORT: fetch only if the CUDA provider plugin isn't already present.
ort-gpu:
	@if [ -e $(ORT_GPU_MARKER) ] || [ -e $(ORT_GPU_MARKER_DLL) ]; then \
	    echo "[ort] GPU ORT already installed under $(ORT_DIR)"; \
	else \
	    echo "[ort] (re)installing GPU ORT prebuilt"; \
	    rm -rf $(ORT_DIR) $(BUILD_DIR); \
	    ORT_BUILD=gpu scripts/fetch_deps.sh; \
	fi

# ---- MEX builds -------------------------------------------------------------

mex-octave: ort-cpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DSIAMIZE_BUILD_OCTAVE_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make mex-octave] built matlab/siamex.mex"

mex-matlab: ort-cpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DSIAMIZE_BUILD_MATLAB_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make mex-matlab] built matlab/siamex.mex<a64|maca64|w64>"

# CUDA-enabled MEX variants: same as mex-octave / mex-matlab but
# fetch the GPU-flavor ORT (libonnxruntime_providers_cuda.so) and
# pass -DSIAMIZE_GPU=cuda so sliding.cpp's CUDA EP probe is compiled
# in. At MATLAB / Octave runtime the user still has to set
# LD_LIBRARY_PATH to include CUDA + cuDNN (same recipe as the CLI;
# see README "Required shared libraries by exact filename").
cudaoct: ort-gpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DSIAMIZE_GPU=cuda -DSIAMIZE_BUILD_OCTAVE_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make cudaoct] built matlab/siamex.mex (CUDA-enabled Octave MEX)"

cudamex: ort-gpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DSIAMIZE_GPU=cuda -DSIAMIZE_BUILD_MATLAB_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make cudamex] built matlab/siamex.mex<a64|maca64|w64> (CUDA-enabled MATLAB MEX)"

# macOS-only. Like cudaoct / cudamex but builds the CoreML MEX
# (Apple Silicon CPU + GPU + ANE). Uses the standard CPU ORT bundle
# since CoreML EP is statically baked into ORT's macOS dylib.
coremloct: ort-cpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DSIAMIZE_GPU=coreml -DSIAMIZE_BUILD_OCTAVE_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make coremloct] built matlab/siamex.mex (CoreML-enabled Octave MEX)"

coremlmex: ort-cpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DSIAMIZE_GPU=coreml -DSIAMIZE_BUILD_MATLAB_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make coremlmex] built matlab/siamex.mex<maca64|maci64> (CoreML-enabled MATLAB MEX)"

# MNN-backed MEX variants. siamex.mex (Octave) or siamex.mex<a64|w64|...>
# (MATLAB) gets a tiny libMNN-static link line if MNN_STATIC=1 was set
# at scripts/fetch_mnn.sh time. The MATLAB-side wrapper queries
# siamex('backend') at runtime to pick the right fold filename
# (_fp32.mnn) and the matching NeuroJSON doc=mnn_n3d URL, so callers
# don't need backend-specific changes to their .m scripts.
opencloct: mnn-deps
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DSIAMIZE_BACKEND=mnn -DSIAMIZE_BUILD_OCTAVE_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make opencloct] built matlab/siamex.mex (MNN-backed Octave MEX)"

openclmex: mnn-deps
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DSIAMIZE_BACKEND=mnn -DSIAMIZE_BUILD_MATLAB_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make openclmex] built matlab/siamex.mex<a64|maca64|w64> (MNN-backed MATLAB MEX)"

mex-test:
	octave-cli --no-gui --eval "cd matlab/tests; run_tests('--exit')"

# ---- Packaging --------------------------------------------------------------

# Each package-* target stages files under dist/<name>/ and zips to
# <name>.zip in the repo root via scripts/package.sh.

package:
	scripts/package.sh cpu      siamize-cpu

package-cuda:
	scripts/package.sh cuda     siamize-cuda

package-tensorrt:
	scripts/package.sh tensorrt siamize-tensorrt

package-mex:
	scripts/package.sh mex      siamex

package-opencl:
	scripts/package.sh opencl   siamize-opencl

package-openclmex:
	scripts/package.sh openclmex siamex-opencl

# ---- Documentation ----------------------------------------------------------

# Build doxygen HTML docs from the in-source doxygen blocks. Output lands
# under build/doc/html/. Requires the `doxygen` binary on PATH
# (apt install doxygen / brew install doxygen).
doc:
	@command -v doxygen >/dev/null 2>&1 || { \
	    echo "doxygen not found. Install it:"; \
	    echo "    Debian/Ubuntu: sudo apt install doxygen"; \
	    echo "    macOS:         brew install doxygen"; \
	    exit 1; \
	}
	doxygen Doxyfile
	@echo
	@echo "[make doc] generated $(BUILD_DIR)/doc/html/index.html"

doc-clean:
	rm -rf $(BUILD_DIR)/doc

# ---- Misc -------------------------------------------------------------------

test:
	tests/run_regression.sh

clean:
	rm -rf $(BUILD_DIR) dist/ \
	    siamize-cpu.zip siamize-cuda.zip siamize-tensorrt.zip \
	    siamize-opencl.zip siamex.zip siamex-opencl.zip

distclean: clean
	rm -rf $(ORT_DIR) $(MNN_DIR) third_party/mnn-build

pretty: pretty-cpp pretty-py

# astyle settings borrowed from MCX (https://github.com/fangq/mcx),
# itself derived from https://github.com/nlohmann/json.
pretty-cpp:
	astyle \
	    --style=attach \
	    --indent=spaces=4 \
	    --indent-modifiers \
	    --indent-switches \
	    --indent-preproc-block \
	    --indent-preproc-define \
	    --indent-col1-comments \
	    --pad-oper \
	    --pad-header \
	    --align-pointer=type \
	    --align-reference=type \
	    --add-brackets \
	    --convert-tabs \
	    --close-templates \
	    --lineend=linux \
	    --preserve-date \
	    --suffix=none \
	    --formatted \
	    --break-blocks \
	    "src/*.cpp" "src/*.h"

# Python formatting via black (PEP 8 conformant, default 88-col line length).
pretty-py:
	black py/ tools/
