// NIfTI-1 read/write with RAS canonicalization, no external dependencies
// beyond zmat (bundled single-header). Drops nifti_clib + system zlib.
//
// Strategy:
//   - Read the whole file into memory.
//   - If gzipped (magic bytes 0x1F 0x8B), zmat_decode to a fresh buffer.
//   - Parse the 348-byte nifti_1_header from the buffer directly.
//   - Recover the sform/qform affine, derive permutation + flip to RAS,
//     copy data into a contiguous (Z, Y, X) float32 Volume.
//   - For write: build an in-memory .nii image (header + uint8 data),
//     zmat_encode to gzip if the output path ends in .gz, write to file.
//
// Buffer-oriented because the whole brain volume already fits comfortably
// in RAM for our use case (typical ~5-200 MB uncompressed).

#include "nifti_io.h"
#include "siam.h"

// zmat: define the implementation in this TU only.
#define ZMAT_IMPLEMENTATION
#include "zmat.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace siam {

namespace {

// ---------- NIfTI-1 header (wire-format, 348 bytes) ------------------------

#pragma pack(push, 1)
struct Nifti1Header {
    int32_t  sizeof_hdr;          // MUST be 348
    char     data_type[10];
    char     db_name[18];
    int32_t  extents;
    int16_t  session_error;
    char     regular;
    char     dim_info;
    int16_t  dim[8];
    float    intent_p1;
    float    intent_p2;
    float    intent_p3;
    int16_t  intent_code;
    int16_t  datatype;
    int16_t  bitpix;
    int16_t  slice_start;
    float    pixdim[8];
    float    vox_offset;
    float    scl_slope;
    float    scl_inter;
    int16_t  slice_end;
    char     slice_code;
    char     xyzt_units;
    float    cal_max;
    float    cal_min;
    float    slice_duration;
    float    toffset;
    int32_t  glmax;
    int32_t  glmin;
    char     descrip[80];
    char     aux_file[24];
    int16_t  qform_code;
    int16_t  sform_code;
    float    quatern_b;
    float    quatern_c;
    float    quatern_d;
    float    qoffset_x;
    float    qoffset_y;
    float    qoffset_z;
    float    srow_x[4];
    float    srow_y[4];
    float    srow_z[4];
    char     intent_name[16];
    char     magic[4];
};
#pragma pack(pop)
static_assert(sizeof(Nifti1Header) == 348, "Nifti1Header must be exactly 348 bytes");

// NIfTI-1 datatype codes
enum NiftiDT : int16_t {
    DT_UINT8   = 2,
    DT_INT16   = 4,
    DT_INT32   = 8,
    DT_FLOAT32 = 16,
    DT_FLOAT64 = 64,
    DT_INT8    = 256,
    DT_UINT16  = 512,
    DT_UINT32  = 768,
};

// ---------- IO + gzip ------------------------------------------------------

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);

    if (!f) {
        throw std::runtime_error("failed to open file: " + path);
    }

    auto sz = static_cast<std::streamsize>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));

    if (sz > 0 && !f.read(reinterpret_cast<char*>(buf.data()), sz)) {
        throw std::runtime_error("failed to read file: " + path);
    }

    return buf;
}

void write_file_bytes(const std::string& path, const uint8_t* data, size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);

    if (!f) {
        throw std::runtime_error("failed to open file for writing: " + path);
    }

    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));

    if (!f) {
        throw std::runtime_error("failed to write file: " + path);
    }
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size()
           && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

// Inflate a gzipped buffer using zmat's direct miniz helper. The returned
// buffer is malloc'd by zmat; we copy it into a vector and free the original.
std::vector<uint8_t> gunzip(const uint8_t* in, size_t n) {
    void* out = nullptr;
    size_t outlen = 0;
    int rc = miniz_gzip_uncompress(const_cast<uint8_t*>(in), n, &out, &outlen);

    if (rc != 0 || out == nullptr) {
        if (out) {
            free(out);
        }

        throw std::runtime_error("gzip decode failed (rc=" + std::to_string(rc) + ")");
    }

    std::vector<uint8_t> result(static_cast<uint8_t*>(out),
                                static_cast<uint8_t*>(out) + outlen);
    free(out);
    return result;
}

// Deflate a buffer into gzip via zmat. zmat doesn't expose a standalone
// miniz_gzip_compress helper (compression is inlined inside zmat_encode),
// so we use the zmat_encode path here. The returned buffer is also malloc'd
// internally and must be released via zmat_free.
std::vector<uint8_t> gzip_compress(const uint8_t* in, size_t n) {
    unsigned char* out = nullptr;
    size_t outlen = 0;
    int ret = 0;
    int rc = zmat_encode(n, const_cast<unsigned char*>(in), &outlen, &out, zmGzip, &ret);

    if (rc != 0 || out == nullptr) {
        if (out) {
            zmat_free(&out);
        }

        throw std::runtime_error("gzip encode failed (rc=" + std::to_string(rc)
                                 + ", ret=" + std::to_string(ret) + ")");
    }

    std::vector<uint8_t> result(out, out + outlen);
    zmat_free(&out);
    return result;
}

// ---------- Affine + canonical reorient ------------------------------------

// Convert NIfTI quaternion (b, c, d) + offset + pixdim into a 4x4 affine.
// Reference: nifti1_io.c::nifti_quatern_to_mat44.
void quatern_to_mat44(float qb, float qc, float qd,
                      float qx, float qy, float qz,
                      float dx, float dy, float dz,
                      float qfac,
                      std::array<float, 16>& m) {
    double b = qb, c = qc, d = qd;
    double a = 1.0 - (b * b + c * c + d * d);

    if (a < 1e-7) {
        a = 1.0 / std::sqrt(b * b + c * c + d * d);
        b *= a;
        c *= a;
        d *= a;
        a = 0.0;
    } else {
        a = std::sqrt(a);
    }

    double xd = (dx > 0) ? dx : 1.0;
    double yd = (dy > 0) ? dy : 1.0;
    double zd = (dz > 0) ? dz : 1.0;

    if (qfac < 0.0f) {
        zd = -zd;
    }

    m[0]  = static_cast<float>((a * a + b * b - c * c - d * d) * xd);
    m[1]  = static_cast<float>(2.0 * (b * c - a * d) * yd);
    m[2]  = static_cast<float>(2.0 * (b * d + a * c) * zd);
    m[3]  = qx;
    m[4]  = static_cast<float>(2.0 * (b * c + a * d) * xd);
    m[5]  = static_cast<float>((a * a + c * c - b * b - d * d) * yd);
    m[6]  = static_cast<float>(2.0 * (c * d - a * b) * zd);
    m[7]  = qy;
    m[8]  = static_cast<float>(2.0 * (b * d - a * c) * xd);
    m[9]  = static_cast<float>(2.0 * (c * d + a * b) * yd);
    m[10] = static_cast<float>((a * a + d * d - c * c - b * b) * zd);
    m[11] = qz;
    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;
}

// Build the affine to use for canonicalization, preferring sform when valid.
std::array<float, 16> extract_affine(const Nifti1Header& h) {
    std::array<float, 16> m{};

    if (h.sform_code > 0) {
        m[0]  = h.srow_x[0];
        m[1]  = h.srow_x[1];
        m[2]  = h.srow_x[2];
        m[3]  = h.srow_x[3];
        m[4]  = h.srow_y[0];
        m[5]  = h.srow_y[1];
        m[6]  = h.srow_y[2];
        m[7]  = h.srow_y[3];
        m[8]  = h.srow_z[0];
        m[9]  = h.srow_z[1];
        m[10] = h.srow_z[2];
        m[11] = h.srow_z[3];
        m[12] = 0.0f;
        m[13] = 0.0f;
        m[14] = 0.0f;
        m[15] = 1.0f;
    } else if (h.qform_code > 0) {
        float qfac = (h.pixdim[0] < 0.0f) ? -1.0f : 1.0f;
        quatern_to_mat44(h.quatern_b, h.quatern_c, h.quatern_d,
                         h.qoffset_x, h.qoffset_y, h.qoffset_z,
                         h.pixdim[1], h.pixdim[2], h.pixdim[3],
                         qfac, m);
    } else {
        // Fallback: identity scaled by pixdim.
        m[0]  = h.pixdim[1];
        m[5]  = h.pixdim[2];
        m[10] = h.pixdim[3];
        m[15] = 1.0f;
    }

    return m;
}

// For each input axis, find which canonical axis (R=0, A=1, S=2) it most
// closely maps to, and the sign.
void axes_to_canonical(const std::array<float, 16>& affine,
                       std::array<int, 3>& dst,
                       std::array<int, 3>& sgn) {
    float col[3][3];

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            col[i][j] = affine[j * 4 + i];
        }
    }

    std::array<bool, 3> used = {false, false, false};

    for (int i = 0; i < 3; ++i) {
        int best = -1;
        float best_abs = -1.0f;

        for (int j = 0; j < 3; ++j) {
            if (used[j]) {
                continue;
            }

            float v = std::fabs(col[i][j]);

            if (v > best_abs) {
                best_abs = v;
                best = j;
            }
        }

        dst[i] = best;
        sgn[i] = (col[i][best] >= 0.0f) ? +1 : -1;
        used[best] = true;
    }
}

template <typename SrcT>
void copy_reorient_to_canonical(const SrcT* src,
                                int64_t X, int64_t Y, int64_t Z,
                                const std::array<int, 3>& dst,
                                const std::array<int, 3>& sgn,
                                Volume& out) {
    std::array<int64_t, 3> in_shape{X, Y, Z};
    std::array<int64_t, 3> canon_shape{0, 0, 0};

    for (int i = 0; i < 3; ++i) {
        canon_shape[dst[i]] = in_shape[i];
    }

    out.resize(canon_shape[2], canon_shape[1], canon_shape[0]);
    const int64_t in_sy = X;
    const int64_t in_sz = X * Y;

    for (int64_t iz = 0; iz < Z; ++iz) {
        for (int64_t iy = 0; iy < Y; ++iy) {
            for (int64_t ix = 0; ix < X; ++ix) {
                int64_t idx_in[3] = {ix, iy, iz};
                int64_t cidx[3] = {0, 0, 0};

                for (int i = 0; i < 3; ++i) {
                    cidx[dst[i]] = (sgn[i] == +1) ? idx_in[i] : (in_shape[i] - 1 - idx_in[i]);
                }

                int64_t out_idx = cidx[2] * out.stride(0) + cidx[1] * out.stride(1) + cidx[0];
                out.data[out_idx] = static_cast<float>(src[iz * in_sz + iy * in_sy + ix]);
            }
        }
    }
}

template <typename DstT>
void copy_reorient_from_canonical(const uint8_t* labels_canon_zyx,
                                  const Volume& canon_shape_template,
                                  int64_t X, int64_t Y, int64_t Z,
                                  const std::array<int, 3>& dst,
                                  const std::array<int, 3>& sgn,
                                  DstT* out_xyz) {
    std::array<int64_t, 3> in_shape{X, Y, Z};

    for (int64_t iz = 0; iz < Z; ++iz) {
        for (int64_t iy = 0; iy < Y; ++iy) {
            for (int64_t ix = 0; ix < X; ++ix) {
                int64_t idx_in[3] = {ix, iy, iz};
                int64_t cidx[3] = {0, 0, 0};

                for (int i = 0; i < 3; ++i) {
                    cidx[dst[i]] = (sgn[i] == +1) ? idx_in[i] : (in_shape[i] - 1 - idx_in[i]);
                }

                int64_t in_idx = cidx[2] * (canon_shape_template.shape[1] * canon_shape_template.shape[2])
                                 + cidx[1] * canon_shape_template.shape[2]
                                 + cidx[0];
                out_xyz[iz * X * Y + iy * X + ix] = static_cast<DstT>(labels_canon_zyx[in_idx]);
            }
        }
    }
}

}  // namespace

NiftiImage load_nifti_ras(const std::string& path) {
    auto raw = read_file_bytes(path);

    if (raw.size() >= 2 && raw[0] == 0x1F && raw[1] == static_cast<uint8_t>(0x8B)) {
        raw = gunzip(raw.data(), raw.size());
    }

    if (raw.size() < sizeof(Nifti1Header)) {
        throw std::runtime_error("file too short to contain a NIfTI-1 header: " + path);
    }

    Nifti1Header h{};
    std::memcpy(&h, raw.data(), sizeof(h));

    if (h.sizeof_hdr != 348) {
        throw std::runtime_error(
            "byte-swapped NIfTI not supported (sizeof_hdr=" + std::to_string(h.sizeof_hdr) + "): " + path);
    }

    if (std::memcmp(h.magic, "n+1", 3) != 0 && std::memcmp(h.magic, "ni1", 3) != 0) {
        throw std::runtime_error("not a NIfTI-1 image (bad magic): " + path);
    }

    if (h.dim[0] < 3) {
        throw std::runtime_error("NIfTI must be at least 3D: " + path);
    }

    if (h.dim[0] > 3 && h.dim[4] > 1) {
        throw std::runtime_error("4D NIfTI not supported; use a 3D volume");
    }

    NiftiImage out;
    out.shape_orig = {h.dim[1], h.dim[2], h.dim[3]};
    out.datatype_orig = h.datatype;
    out.affine_orig = extract_affine(h);

    std::array<int, 3> dst{}, sgn{};
    axes_to_canonical(out.affine_orig, dst, sgn);

    const int64_t X = h.dim[1], Y = h.dim[2], Z = h.dim[3];
    const size_t nvox = static_cast<size_t>(X) * Y * Z;

    size_t data_offset = static_cast<size_t>(h.vox_offset);

    if (data_offset == 0) {
        data_offset = sizeof(Nifti1Header) + 4;
    }

    if (raw.size() < data_offset + nvox * static_cast<size_t>(h.bitpix / 8)) {
        throw std::runtime_error("NIfTI file truncated: data extends past EOF");
    }

    const uint8_t* data_ptr = raw.data() + data_offset;

    switch (h.datatype) {
        case DT_INT8:
            copy_reorient_to_canonical(reinterpret_cast<const int8_t*>(data_ptr),   X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_UINT8:
            copy_reorient_to_canonical(reinterpret_cast<const uint8_t*>(data_ptr),  X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_INT16:
            copy_reorient_to_canonical(reinterpret_cast<const int16_t*>(data_ptr),  X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_UINT16:
            copy_reorient_to_canonical(reinterpret_cast<const uint16_t*>(data_ptr), X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_INT32:
            copy_reorient_to_canonical(reinterpret_cast<const int32_t*>(data_ptr),  X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_UINT32:
            copy_reorient_to_canonical(reinterpret_cast<const uint32_t*>(data_ptr), X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_FLOAT32:
            copy_reorient_to_canonical(reinterpret_cast<const float*>(data_ptr),    X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_FLOAT64:
            copy_reorient_to_canonical(reinterpret_cast<const double*>(data_ptr),   X, Y, Z, dst, sgn, out.volume);
            break;

        default:
            throw std::runtime_error("unsupported NIfTI datatype: " + std::to_string(h.datatype));
    }

    if (h.scl_slope != 0.0f && (h.scl_slope != 1.0f || h.scl_inter != 0.0f)) {
        for (auto& v : out.volume.data) {
            v = v * h.scl_slope + h.scl_inter;
        }
    }

    // Canonical affine: place each input column at its dst position with sgn[i] sign,
    // and translate origin if axis was flipped.
    const auto& A = out.affine_orig;
    std::array<float, 16> canon{};
    canon[15] = 1.0f;

    for (int i = 0; i < 3; ++i) {
        int k = dst[i];
        float s = static_cast<float>(sgn[i]);

        for (int r = 0; r < 3; ++r) {
            canon[r * 4 + k] = s * A[r * 4 + i];
        }

        if (sgn[i] == -1) {
            for (int r = 0; r < 3; ++r) {
                canon[r * 4 + 3] += A[r * 4 + i] * static_cast<float>(out.shape_orig[i] - 1);
            }
        }
    }

    for (int r = 0; r < 3; ++r) {
        canon[r * 4 + 3] += A[r * 4 + 3];
    }

    out.affine_canon = canon;
    out.zooms_canon = {std::fabs(canon[0]), std::fabs(canon[5]), std::fabs(canon[10])};

    for (int i = 0; i < 3; ++i) {
        out.perm_canon_to_orig[i] = dst[i];
    }

    for (int i = 0; i < 3; ++i) {
        out.flip_canon[dst[i]] = sgn[i];
    }

    return out;
}

void save_nifti_labels(const std::string& path,
                       const NiftiImage& src,
                       const uint8_t* labels_zyx) {
    Nifti1Header h{};
    h.sizeof_hdr = 348;
    h.dim[0] = 3;
    h.dim[1] = static_cast<int16_t>(src.shape_orig[0]);
    h.dim[2] = static_cast<int16_t>(src.shape_orig[1]);
    h.dim[3] = static_cast<int16_t>(src.shape_orig[2]);
    h.dim[4] = 1;
    h.dim[5] = 1;
    h.dim[6] = 1;
    h.dim[7] = 1;
    h.datatype = DT_UINT8;
    h.bitpix = 8;

    auto col_norm = [&](int c) {
        const auto& A = src.affine_orig;
        float v0 = A[0 * 4 + c], v1 = A[1 * 4 + c], v2 = A[2 * 4 + c];
        return std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    };
    h.pixdim[0] = -1.0f;
    h.pixdim[1] = col_norm(0);
    h.pixdim[2] = col_norm(1);
    h.pixdim[3] = col_norm(2);

    h.vox_offset = static_cast<float>(sizeof(Nifti1Header) + 4);
    h.scl_slope = 0.0f;
    h.scl_inter = 0.0f;

    // sform from src.affine_orig.
    h.sform_code = 2;
    h.qform_code = 0;
    const auto& A = src.affine_orig;

    for (int c = 0; c < 4; ++c) {
        h.srow_x[c] = A[0 * 4 + c];
    }

    for (int c = 0; c < 4; ++c) {
        h.srow_y[c] = A[1 * 4 + c];
    }

    for (int c = 0; c < 4; ++c) {
        h.srow_z[c] = A[2 * 4 + c];
    }

    std::memcpy(h.magic, "n+1\0", 4);

    // De-canonicalize the label volume back into (X, Y, Z) input axis order.
    const int64_t X = src.shape_orig[0];
    const int64_t Y = src.shape_orig[1];
    const int64_t Z = src.shape_orig[2];
    std::array<int, 3> dst{};

    for (int i = 0; i < 3; ++i) {
        dst[i] = src.perm_canon_to_orig[i];
    }

    std::array<int, 3> sgn{};

    for (int i = 0; i < 3; ++i) {
        sgn[i] = src.flip_canon[dst[i]];
    }

    std::vector<uint8_t> labels_xyz(static_cast<size_t>(X) * Y * Z, 0);
    copy_reorient_from_canonical(labels_zyx, src.volume, X, Y, Z, dst, sgn, labels_xyz.data());

    // Assemble the on-disk image: header + 4 bytes padding + data.
    const size_t total_size = static_cast<size_t>(h.vox_offset) + labels_xyz.size();
    std::vector<uint8_t> nii(total_size, 0);
    std::memcpy(nii.data(), &h, sizeof(Nifti1Header));
    std::memcpy(nii.data() + static_cast<size_t>(h.vox_offset),
                labels_xyz.data(),
                labels_xyz.size());

    if (ends_with(path, ".gz") || ends_with(path, ".GZ")) {
        auto gz = gzip_compress(nii.data(), nii.size());
        write_file_bytes(path, gz.data(), gz.size());
    } else {
        write_file_bytes(path, nii.data(), nii.size());
    }
}

}  // namespace siam
