// JNIfTI (.jnii text / .bnii BJData) container I/O. Same NIfTI metadata
// + voxel data the NIfTI-1 path handles, but wrapped in a JSON-Data
// container per https://neurojson.org/jnifti. Voxel data is
// zlib-compressed and stored as a JData annotated array
// (_ArrayZipData_).

#include "jnifti_io.h"
#include "orient.h"
#include "siam.h"

#include "nlohmann/json.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Forward decls of miniz's zlib (RFC 1950) compress/uncompress entry
// points. Definitions are in zmat/zmat.h under ZMAT_IMPLEMENTATION,
// which nifti_io.cpp instantiates. zmat.h wraps these declarations in
// `extern "C"` so the symbols have C linkage -- match that here.
extern "C" {
    typedef unsigned long mz_ulong;
    int mz_compress2(unsigned char* pDest, mz_ulong* pDest_len,
                     const unsigned char* pSource, mz_ulong source_len, int level);
    int mz_uncompress(unsigned char* pDest, mz_ulong* pDest_len,
                      const unsigned char* pSource, mz_ulong source_len);
    mz_ulong mz_compressBound(mz_ulong source_len);
}

namespace siam {

using json = nlohmann::ordered_json;

namespace {

std::vector<uint8_t> zlib_compress(const uint8_t* in, size_t n, int level = 6) {
    mz_ulong cap = mz_compressBound(static_cast<mz_ulong>(n));
    std::vector<uint8_t> out(cap);
    mz_ulong dest_len = cap;
    int rc = mz_compress2(out.data(), &dest_len, in, static_cast<mz_ulong>(n), level);

    if (rc != 0) {
        throw std::runtime_error("mz_compress2 failed (rc=" + std::to_string(rc) + ")");
    }

    out.resize(static_cast<size_t>(dest_len));
    return out;
}

std::vector<uint8_t> zlib_decompress(const uint8_t* in, size_t n, size_t expected) {
    std::vector<uint8_t> out(expected);
    mz_ulong dest_len = static_cast<mz_ulong>(expected);
    int rc = mz_uncompress(out.data(), &dest_len, in, static_cast<mz_ulong>(n));

    if (rc != 0) {
        throw std::runtime_error("mz_uncompress failed (rc=" + std::to_string(rc) + ")");
    }

    out.resize(static_cast<size_t>(dest_len));
    return out;
}

// Standard RFC 4648 base64 encoder.
std::string base64_encode(const uint8_t* in, size_t n) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((n + 2) / 3) * 4);
    size_t i = 0;

    for (; i + 3 <= n; i += 3) {
        uint32_t v = (uint32_t(in[i]) << 16) | (uint32_t(in[i + 1]) << 8) | in[i + 2];
        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back(tbl[(v >> 6) & 63]);
        out.push_back(tbl[v & 63]);
    }

    if (i < n) {
        uint32_t v = uint32_t(in[i]) << 16;

        if (i + 1 < n) {
            v |= uint32_t(in[i + 1]) << 8;
        }

        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back(i + 1 < n ? tbl[(v >> 6) & 63] : '=');
        out.push_back('=');
    }

    return out;
}

std::vector<uint8_t> base64_decode(const std::string& s) {
    static int8_t lut[128];
    static bool init = false;

    if (!init) {
        for (int i = 0; i < 128; ++i) {
            lut[i] = -1;
        }

        const char* tbl =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        for (int i = 0; i < 64; ++i) {
            lut[static_cast<unsigned char>(tbl[i])] = i;
        }

        init = true;
    }

    std::vector<uint8_t> out;
    out.reserve((s.size() / 4) * 3);
    uint32_t v = 0;
    int bits = 0;

    for (char c : s) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            continue;
        }

        if (static_cast<unsigned char>(c) >= 128 || lut[(int)c] < 0) {
            throw std::runtime_error("invalid base64 character");
        }

        v = (v << 6) | static_cast<uint32_t>(lut[(int)c]);
        bits += 6;

        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((v >> bits) & 0xFFu));
        }
    }

    return out;
}

// Column-major (X-fastest) -> row-major (last-dim-fastest) transpose.
// Required for _ArrayZipData_ output because jsonlab's jdataencode does
// the same reshape+permute internally before compression, so the on-
// disk byte order is row-major regardless of any _ArrayOrder_ tag.
// Matches jsonlab/jdataencode.m lines 475-480.
template <typename T>
std::vector<T> col_to_row_major(const T* col, const std::vector<int64_t>& shape) {
    size_t n = 1;

    for (auto d : shape) {
        n *= static_cast<size_t>(d);
    }

    std::vector<T> row(n);
    int ndim = static_cast<int>(shape.size());
    std::vector<int64_t> col_stride(ndim), row_stride(ndim);
    col_stride[0] = 1;

    for (int d = 1; d < ndim; ++d) {
        col_stride[d] = col_stride[d - 1] * shape[d - 1];
    }

    row_stride[ndim - 1] = 1;

    for (int d = ndim - 2; d >= 0; --d) {
        row_stride[d] = row_stride[d + 1] * shape[d + 1];
    }

    for (size_t p = 0; p < n; ++p) {
        size_t rem = p;
        size_t col_idx = 0;

        for (int d = 0; d < ndim; ++d) {
            int64_t i = static_cast<int64_t>(rem / static_cast<size_t>(row_stride[d]));
            rem -= static_cast<size_t>(i) * static_cast<size_t>(row_stride[d]);
            col_idx += static_cast<size_t>(i) * static_cast<size_t>(col_stride[d]);
        }

        row[p] = col[col_idx];
    }

    return row;
}

// JData dtype string for the C++ scalar type.
template <typename T> std::string jdata_dtype();
template <> std::string jdata_dtype<uint8_t>()  { return "uint8";  }
template <> std::string jdata_dtype<int16_t>()  { return "int16";  }
template <> std::string jdata_dtype<int32_t>()  { return "int32";  }
template <> std::string jdata_dtype<float>()    { return "single"; }
template <> std::string jdata_dtype<double>()   { return "double"; }

// Build the JData annotated-array sub-object for `data` of dtype T,
// dimensions `shape`. Compresses with zlib; encodes the result as
// base64 string for text format, raw bytes for binary format.
template <typename T>
json jdata_annotated(const T* data, const std::vector<int64_t>& shape, bool binary_format) {
    size_t n_elem = 1;

    for (auto d : shape) {
        n_elem *= static_cast<size_t>(d);
    }

    // jsonlab/jdataencode.m:450 permutes the source array's axis order
    // BEFORE flattening, producing bytes in last-dim-fastest order. For
    // shape [X,Y,Z,C], that's C-fastest, then Z, Y, X. We mirror that
    // transposition here so the resulting .jnii/.bnii rounds-trips
    // through jdatadecode without channel scrambling. _ArrayZipSize_
    // is then [1, N] (jsonlab convention for compressed arrays); the
    // actual N-D shape lives in _ArraySize_, where jdatadecode's
    // format>1.9 reshape+permute pair recovers it correctly.
    std::vector<T> row = col_to_row_major(data, shape);
    const size_t raw_bytes = n_elem * sizeof(T);
    auto comp = zlib_compress(reinterpret_cast<const uint8_t*>(row.data()), raw_bytes);

    json arr = json::object();
    arr["_ArrayType_"]    = jdata_dtype<T>();
    arr["_ArraySize_"]    = shape;
    arr["_ArrayZipType_"] = "zlib";
    arr["_ArrayZipSize_"] = std::vector<int64_t>{1, static_cast<int64_t>(n_elem)};

    if (binary_format) {
        // BJData: wrap in json::binary so to_bjdata emits a true byte
        // string ('[$U#<len><bytes>' typed array) instead of a generic
        // numeric array of per-byte numbers. jsonlab's loadbj reads this
        // back as a uint8 vector directly, which is what _ArrayZipData_
        // should be on the .bnii wire.
        arr["_ArrayZipData_"] = json::binary(comp);
    } else {
        arr["_ArrayZipData_"] = base64_encode(comp.data(), comp.size());
    }

    return arr;
}

json build_header(const NiftiImage& src, const std::vector<int64_t>& dim) {
    json h = json::object();
    h["NIIHeaderSize"] = 348;
    h["Dim"]           = dim;

    json aff = json::array();

    for (int r = 0; r < 3; ++r) {
        json row = json::array();

        for (int c = 0; c < 4; ++c) {
            row.push_back(src.affine_orig[r * 4 + c]);
        }

        aff.push_back(row);
    }

    h["Affine"] = aff;

    auto col_norm = [&](int c) {
        const auto& A = src.affine_orig;
        return std::sqrt(A[0 * 4 + c] * A[0 * 4 + c]
                         + A[1 * 4 + c] * A[1 * 4 + c]
                         + A[2 * 4 + c] * A[2 * 4 + c]);
    };
    h["VoxelSize"] = std::vector<float>{
        static_cast<float>(col_norm(0)),
        static_cast<float>(col_norm(1)),
        static_cast<float>(col_norm(2))
    };
    h["DataType"] = (dim.size() == 4) ? std::string("single") : std::string("uint8");
    h["BitDepth"] = (dim.size() == 4) ? 32 : 8;
    return h;
}

void write_jnifti_root(const std::string& path, const json& j, bool binary) {
    std::ofstream f(path, std::ios::binary);

    if (!f) {
        throw std::runtime_error("failed to open " + path + " for writing");
    }

    if (binary) {
        std::vector<uint8_t> bj;
        json::to_bjdata(j, bj, true, true);  // use_count=true, use_type=true (optimized BJData)
        f.write(reinterpret_cast<const char*>(bj.data()), static_cast<std::streamsize>(bj.size()));
    } else {
        std::string txt = j.dump();
        f.write(txt.data(), static_cast<std::streamsize>(txt.size()));
    }

    if (!f) {
        throw std::runtime_error("failed to write " + path);
    }
}

}  // anonymous namespace


void save_jnifti_labels(const std::string& path,
                        const NiftiImage& src,
                        const uint8_t* labels_zyx,
                        const std::string& format) {
    bool binary = (format == "bnii");

    if (format != "jnii" && format != "bnii") {
        throw std::runtime_error(
            "save_jnifti_labels: format must be 'jnii' or 'bnii' (got '"
            + format + "')");
    }

    const int64_t X = src.shape_orig[0];
    const int64_t Y = src.shape_orig[1];
    const int64_t Z = src.shape_orig[2];
    std::array<int, 3> dst{}, sgn{};

    for (int i = 0; i < 3; ++i) {
        dst[i] = src.perm_canon_to_orig[i];
        sgn[i] = src.flip_canon[dst[i]];
    }

    std::vector<uint8_t> data_xyz(static_cast<size_t>(X) * Y * Z, 0);
    copy_reorient_from_canonical<uint8_t, uint8_t>(
        labels_zyx,
        src.volume.shape[0], src.volume.shape[1], src.volume.shape[2],
        X, Y, Z, dst, sgn,
        data_xyz.data());

    json root;
    root["NIFTIHeader"] = build_header(src, {X, Y, Z});
    root["NIFTIData"]   = jdata_annotated<uint8_t>(
                              data_xyz.data(), {X, Y, Z}, binary);
    write_jnifti_root(path, root, binary);
}


void save_jnifti_tpm(const std::string& path,
                     const NiftiImage& src,
                     const float* tpm_canon_czyx,
                     int64_t num_classes,
                     const std::string& format) {
    bool binary = (format == "bnii");

    if (format != "jnii" && format != "bnii") {
        throw std::runtime_error(
            "save_jnifti_tpm: format must be 'jnii' or 'bnii' (got '"
            + format + "')");
    }

    const int64_t X = src.shape_orig[0];
    const int64_t Y = src.shape_orig[1];
    const int64_t Z = src.shape_orig[2];
    const int64_t cZ = src.volume.shape[0];
    const int64_t cY = src.volume.shape[1];
    const int64_t cX = src.volume.shape[2];
    const int64_t per_channel = X * Y * Z;
    const int64_t canon_per_channel = cZ * cY * cX;

    std::array<int, 3> dst{}, sgn{};

    for (int i = 0; i < 3; ++i) {
        dst[i] = src.perm_canon_to_orig[i];
        sgn[i] = src.flip_canon[dst[i]];
    }

    std::vector<float> data_xyzc(static_cast<size_t>(per_channel * num_classes), 0.0f);

    for (int64_t c = 0; c < num_classes; ++c) {
        copy_reorient_from_canonical<float, float>(
            tpm_canon_czyx + c * canon_per_channel,
            cZ, cY, cX,
            X, Y, Z, dst, sgn,
            data_xyzc.data() + c * per_channel);
    }

    json root;
    root["NIFTIHeader"] = build_header(src, {X, Y, Z, num_classes});
    root["NIFTIData"]   = jdata_annotated<float>(
                              data_xyzc.data(),
                              {X, Y, Z, num_classes}, binary);
    write_jnifti_root(path, root, binary);
}


NiftiImage load_jnifti_ras(const std::string& path) {
    (void)path;
    throw std::runtime_error("load_jnifti_ras: not yet implemented (Phase 3)");
}

}  // namespace siam
