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
#   mex       siamex.mex* + siamize.m + matlab/jsonlab + libonnxruntime
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
}

case "${mode}" in
    cpu)      copy_cli ;;
    cuda)     copy_cli; copy_cuda_eps ;;
    tensorrt) copy_cli; copy_cuda_eps; copy_trt_eps ;;
    mex)      copy_mex ;;
    *) echo "[package] unknown mode: ${mode}"; exit 1 ;;
esac

cp LICENSE "${stagedir}/"
if [ "${mode}" != "mex" ]; then
    cp README.md "${stagedir}/"
fi

zipfile="${out}.zip"
rm -f "${zipfile}"
( cd dist && 7z a -tzip "../${zipfile}" "${out}" >/dev/null )

echo "[package] created ${zipfile}:"
ls -l "${zipfile}"
echo "[package] contents:"
7z l "${zipfile}" | tail -n +14
