#!/usr/bin/env bash
# scripts/package.sh -- create a release zip bundle for siamize.
#
# Usage:
#   scripts/package.sh <mode> <output_name>
#
# Modes:
#   cpu       CLI binary + libonnxruntime
#   cuda      CLI binary + libonnxruntime + ORT CUDA provider plugins
#   tensorrt  CLI binary + libonnxruntime + ORT CUDA + TensorRT plugins
#   opencl    CLI binary, MNN backend (self-contained, no runtime .so)
#   mex       siamex.mex* + siamize.m + matlab/jsonlab + libonnxruntime
#   openclmex siamex.mex* + siamize.m + matlab/jsonlab (MNN, no runtime .so)
#
# Output: ./${output_name}.zip
#
# Runs cross-platform on Linux / macOS / Windows runners (relies only on
# bash, 7z, and a populated third_party/onnxruntime + build directory).

set -e

mode="${1:?usage: $0 <mode> <output_name>}"
out="${2:?usage: $0 <mode> <output_name>}"
stagedir="dist/${out}"

case "$(uname -s)" in
    Linux)               os=linux ;;
    Darwin)              os=macos ;;
    MINGW*|MSYS*|CYGWIN*) os=windows ;;
    *) echo "[package] unknown OS: $(uname -s)"; exit 1 ;;
esac

ort_lib_dir="third_party/onnxruntime/lib"

rm -rf "${stagedir}"
mkdir -p "${stagedir}"

copy_cli() {
    case "${os}" in
        linux)
            cp build/siamize "${stagedir}/"
            # Bundle the real lib under its soname so the loader resolves
            # 'libonnxruntime.so.1' directly without a symlink in the zip.
            cp "${ort_lib_dir}/libonnxruntime.so.1.26.0" \
               "${stagedir}/libonnxruntime.so.1"
            ;;
        macos)
            cp build/siamize "${stagedir}/"
            cp "${ort_lib_dir}/libonnxruntime.1.26.0.dylib" \
               "${stagedir}/libonnxruntime.1.dylib"
            ;;
        windows)
            cp build/Release/siamize.exe "${stagedir}/"
            cp build/Release/onnxruntime.dll "${stagedir}/"
            ;;
    esac
}

copy_cuda_eps() {
    if [ "${os}" = "windows" ]; then
        cp build/Release/onnxruntime_providers_cuda.dll   "${stagedir}/"
        cp build/Release/onnxruntime_providers_shared.dll "${stagedir}/"
    else
        cp "${ort_lib_dir}/libonnxruntime_providers_cuda.so"   "${stagedir}/"
        cp "${ort_lib_dir}/libonnxruntime_providers_shared.so" "${stagedir}/"
    fi
}

copy_trt_eps() {
    if [ "${os}" = "windows" ]; then
        cp build/Release/onnxruntime_providers_tensorrt.dll "${stagedir}/"
    else
        cp "${ort_lib_dir}/libonnxruntime_providers_tensorrt.so" "${stagedir}/"
    fi
}

copy_cli_opencl() {
    # MNN backend, static-linked libMNN.a. The binary is self-contained
    # -- no runtime .so to bundle alongside it. We do copy libMNN.so /
    # .dylib / .dll if it's present (i.e. the user built the shared-MNN
    # variant) so the same bundle name covers both link modes.
    case "${os}" in
        linux)
            cp build/siamize "${stagedir}/"
            if [ -f third_party/mnn/lib/libMNN.so ]; then
                cp third_party/mnn/lib/libMNN.so "${stagedir}/"
            fi
            ;;
        macos)
            cp build/siamize "${stagedir}/"
            if [ -f third_party/mnn/lib/libMNN.dylib ]; then
                cp third_party/mnn/lib/libMNN.dylib "${stagedir}/"
            fi
            ;;
        windows)
            cp build/Release/siamize.exe "${stagedir}/"
            if [ -f build/Release/MNN.dll ]; then
                cp build/Release/MNN.dll "${stagedir}/"
            fi
            ;;
    esac
}

copy_mex_opencl() {
    # Same MEX-discovery rules as copy_mex() but no libonnxruntime to
    # bundle. libMNN.{so,dylib,dll} only ships if the MEX was linked
    # against the shared variant; the static MEX is self-contained.
    case "${os}" in
        linux)
            if   [ -f matlab/siamex.mexa64 ]; then cp matlab/siamex.mexa64 "${stagedir}/"
            elif [ -f matlab/siamex.mex   ]; then cp matlab/siamex.mex   "${stagedir}/"
            else echo "[package] no MEX file found in matlab/"; exit 1
            fi
            if [ -f third_party/mnn/lib/libMNN.so ]; then
                cp third_party/mnn/lib/libMNN.so "${stagedir}/"
            fi
            ;;
        macos)
            if   [ -f matlab/siamex.mexmaca64 ]; then cp matlab/siamex.mexmaca64 "${stagedir}/"
            elif [ -f matlab/siamex.mexmaci64 ]; then cp matlab/siamex.mexmaci64 "${stagedir}/"
            elif [ -f matlab/siamex.mex       ]; then cp matlab/siamex.mex       "${stagedir}/"
            else echo "[package] no MEX file found in matlab/"; exit 1
            fi
            if [ -f third_party/mnn/lib/libMNN.dylib ]; then
                cp third_party/mnn/lib/libMNN.dylib "${stagedir}/"
            fi
            ;;
        windows)
            cp matlab/siamex.mexw64 "${stagedir}/"
            if [ -f matlab/MNN.dll ]; then
                cp matlab/MNN.dll "${stagedir}/"
            fi
            ;;
    esac
    cp matlab/siamize.m "${stagedir}/"
    cp -r matlab/jsonlab "${stagedir}/jsonlab"
    rm -f "${stagedir}/jsonlab/.git"
    find "${stagedir}/jsonlab" -name '.git*' -prune -exec rm -rf {} + 2>/dev/null || true
    cp matlab/README.md "${stagedir}/README.md"
}

copy_mex() {
    # MEX outputs land in matlab/ next to siamize.m since 869ddc5
    # (CMakeLists's RUNTIME/LIBRARY_OUTPUT_DIRECTORY override). Pick
    # whichever extension the current build produced.
    case "${os}" in
        linux)
            if   [ -f matlab/siamex.mexa64 ]; then cp matlab/siamex.mexa64 "${stagedir}/"
            elif [ -f matlab/siamex.mex   ]; then cp matlab/siamex.mex   "${stagedir}/"
            else echo "[package] no MEX file found in matlab/"; exit 1
            fi
            cp "${ort_lib_dir}/libonnxruntime.so.1.26.0" \
               "${stagedir}/libonnxruntime.so.1"
            ;;
        macos)
            if   [ -f matlab/siamex.mexmaca64 ]; then cp matlab/siamex.mexmaca64 "${stagedir}/"
            elif [ -f matlab/siamex.mexmaci64 ]; then cp matlab/siamex.mexmaci64 "${stagedir}/"
            elif [ -f matlab/siamex.mex       ]; then cp matlab/siamex.mex       "${stagedir}/"
            else echo "[package] no MEX file found in matlab/"; exit 1
            fi
            cp "${ort_lib_dir}/libonnxruntime.1.26.0.dylib" \
               "${stagedir}/libonnxruntime.1.dylib"
            ;;
        windows)
            cp matlab/siamex.mexw64 "${stagedir}/"
            # The MEX POST_BUILD step copies onnxruntime.dll next to the
            # .mexw64, so it lives in matlab/ now (not build/Release/).
            cp matlab/onnxruntime.dll "${stagedir}/"
            ;;
    esac
    cp matlab/siamize.m "${stagedir}/"
    # Whole jsonlab dir minus its submodule gitlink (otherwise users get
    # a broken .git pointer when they unzip).
    cp -r matlab/jsonlab "${stagedir}/jsonlab"
    rm -f "${stagedir}/jsonlab/.git"
    find "${stagedir}/jsonlab" -name '.git*' -prune -exec rm -rf {} + 2>/dev/null || true
    cp matlab/README.md "${stagedir}/README.md"

    # If a GPU ORT was fetched (ORT_BUILD=gpu), the CUDA provider plugins are
    # present -- ship them next to the MEX so a GPU host uses the CUDA EP. ORT
    # loads them lazily at session creation (and falls back to CPU otherwise),
    # so the bundle still works on CPU-only hosts. The CUDA 12 + cuDNN 9 runtime
    # come from the host, same as the siamize-*-cuda CLI bundle. A CPU ORT build
    # has no such libs, so this is skipped (the MEX stays CPU-only).
    case "${os}" in
        windows)
            if [ -f "${ort_lib_dir}/onnxruntime_providers_cuda.dll" ]; then
                cp "${ort_lib_dir}/onnxruntime_providers_cuda.dll"   "${stagedir}/"
                cp "${ort_lib_dir}/onnxruntime_providers_shared.dll" "${stagedir}/"
            fi
            ;;
        linux)
            if [ -f "${ort_lib_dir}/libonnxruntime_providers_cuda.so" ]; then
                cp "${ort_lib_dir}/libonnxruntime_providers_cuda.so"   "${stagedir}/"
                cp "${ort_lib_dir}/libonnxruntime_providers_shared.so" "${stagedir}/"
            fi
            ;;
    esac
}

case "${mode}" in
    cpu)       copy_cli ;;
    cuda)      copy_cli; copy_cuda_eps ;;
    tensorrt)  copy_cli; copy_cuda_eps; copy_trt_eps ;;
    opencl)    copy_cli_opencl ;;
    mex)       copy_mex ;;
    openclmex) copy_mex_opencl ;;
    *) echo "[package] unknown mode: ${mode}"; exit 1 ;;
esac

cp LICENSE "${stagedir}/"
if [ "${mode}" != "mex" ] && [ "${mode}" != "openclmex" ]; then
    cp README.md "${stagedir}/"
fi

zipfile="${out}.zip"
rm -f "${zipfile}"
( cd dist && 7z a -tzip "../${zipfile}" "${out}" >/dev/null )

echo "[package] created ${zipfile}:"
ls -l "${zipfile}"
echo "[package] contents:"
7z l "${zipfile}" | tail -n +14
