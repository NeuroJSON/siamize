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

# Filenames used to detect which ORT prebuilt (CPU vs GPU) is currently
# installed under third_party/onnxruntime/.
ORT_GPU_MARKER := $(ORT_DIR)/lib/libonnxruntime_providers_cuda.so
ORT_GPU_MARKER_DLL := $(ORT_DIR)/lib/onnxruntime_providers_cuda.dll

.PHONY: all build cuda tensorrt mex-octave mex-matlab mex-test \
        package package-cuda package-tensorrt package-mex \
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
	@echo "[make mex-octave] built $(BUILD_DIR)/siamex.mex"

mex-matlab: ort-cpu
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DSIAMIZE_BUILD_MATLAB_MEX=ON
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel
	@echo "[make mex-matlab] built $(BUILD_DIR)/siamex.mex<a64|maca64|w64>"

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
	rm -rf $(BUILD_DIR) dist/ siamize-cpu.zip siamize-cuda.zip siamize-tensorrt.zip siamex.zip

distclean: clean
	rm -rf $(ORT_DIR)

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
