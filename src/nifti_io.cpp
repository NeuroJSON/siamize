/***************************************************************************//**
**  \mainpage siamize - native C++/ONNX port of SIAM v0.3 brain segmentation
**
**  \author    Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2026
**
**  \section sref Reference:
**  \li \c (\b Valabregue2026) Romain Valabregue, Ikram Khemir, Eric Bardinet,
**         Francois Rousseau, Guillaume Auzias, Reuben Dorent, "SIAM: Head and
**         Brain MRI Segmentation from Few High-Quality Templates via Synthetic
**         Training," arXiv:2605.02737 (2026).
**         <a href="https://arxiv.org/abs/2605.02737">arxiv.org/abs/2605.02737</a>
**  \li \c (\b NIfTI1) NIfTI-1 Data Format,
**         <a href="https://nifti.nimh.nih.gov/nifti-1">nifti.nimh.nih.gov/nifti-1</a>
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    nifti_io.cpp
\brief   NIfTI-1 (.nii / .nii.gz) reader/writer implementation

This translation unit reads and writes plain NIfTI-1 files without
pulling in nifti_clib or a system zlib -- the only compression
dependency is the bundled zmat single-header (src/zmat/zmat.h),
which also instantiates miniz in this TU (the only place
ZMAT_IMPLEMENTATION is defined in the project).

Strategy:

  - Read the whole file into memory.
  - If gzipped (magic bytes 0x1F 0x8B), inflate via zmat/miniz to a
    fresh buffer.
  - Parse the 348-byte `nifti_1_header` from the buffer directly.
  - Recover the affine, preferring `sform` when set and otherwise
    decoding `qform`'s quaternion. Derive the axis permutation and
    flip that bring the input voxel axes into canonical (Z, Y, X) RAS,
    then copy the data into a contiguous float32 Volume via the
    typed copy_reorient_to_canonical templates.
  - For writes, build the on-disk image in memory (header + 4 byte
    padding + data), gzip-encode via zmat if the output path ends in
    `.gz`, then write the buffer to disk in a single I/O.

The buffer-oriented strategy is deliberate: typical brain volumes
fit comfortably in RAM (~5-200 MB uncompressed), so streaming
compression isn't worth the added complexity.
*******************************************************************************/

#include "nifti_io.h"
#include "orient.h"
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

/* ============================================================================ */
/*                NIfTI-1 wire-format header and datatype codes                */
/* ============================================================================ */

/**
 * \struct Nifti1Header
 * \brief NIfTI-1 on-disk header (348 bytes), packed for byte-exact layout
 *
 * Field order and types match the canonical `nifti_1_header` from
 * nifti_clib's nifti1.h. Only the fields siamize actually consumes
 * or emits are documented inline; the rest are present for header
 * size + offset correctness.
 */
#pragma pack(push, 1)
struct Nifti1Header {
    int32_t  sizeof_hdr;          /**< must be 348 (also serves as endianness check) */
    char     data_type[10];       /**< unused */
    char     db_name[18];         /**< unused */
    int32_t  extents;             /**< unused */
    int16_t  session_error;       /**< unused */
    char     regular;             /**< unused */
    char     dim_info;            /**< MRI slice ordering (unused) */
    int16_t  dim[8];              /**< dim[0]=ndim, dim[1..7]=axis sizes */
    float    intent_p1;
    float    intent_p2;
    float    intent_p3;
    int16_t  intent_code;
    int16_t  datatype;            /**< NIfTI datatype code (see NiftiDT) */
    int16_t  bitpix;              /**< bits per voxel */
    int16_t  slice_start;
    float    pixdim[8];           /**< pixdim[0]=qfac, pixdim[1..3]=mm spacing */
    float    vox_offset;          /**< byte offset of first voxel (typ. 352 = sizeof header + 4) */
    float    scl_slope;           /**< scaling slope (0 means no scaling) */
    float    scl_inter;           /**< scaling intercept */
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
    int16_t  qform_code;          /**< nonzero => quatern_* + qoffset_* define the affine */
    int16_t  sform_code;          /**< nonzero => srow_x/y/z directly define the affine (preferred) */
    float    quatern_b;           /**< qform rotation: quaternion b */
    float    quatern_c;           /**< qform rotation: quaternion c */
    float    quatern_d;           /**< qform rotation: quaternion d */
    float    qoffset_x;           /**< qform translation X */
    float    qoffset_y;           /**< qform translation Y */
    float    qoffset_z;           /**< qform translation Z */
    float    srow_x[4];           /**< sform row 0 (Rx Ry Rz Tx) */
    float    srow_y[4];           /**< sform row 1 */
    float    srow_z[4];           /**< sform row 2 */
    char     intent_name[16];
    char     magic[4];            /**< "n+1\0" for single-file NIfTI, "ni1\0" for hdr+img pair */
};
#pragma pack(pop)
static_assert(sizeof(Nifti1Header) == 348, "Nifti1Header must be exactly 348 bytes");

/**
 * \enum NiftiDT
 * \brief Subset of NIfTI-1 datatype codes that siamize reads/writes
 */
enum NiftiDT : int16_t {
    DT_UINT8   = 2,    /**< uint8 */
    DT_INT16   = 4,    /**< int16 (signed short) */
    DT_INT32   = 8,    /**< int32 */
    DT_FLOAT32 = 16,   /**< float32 (the inference path's working type) */
    DT_FLOAT64 = 64,   /**< float64 */
    DT_INT8    = 256,  /**< int8 */
    DT_UINT16  = 512,  /**< uint16 */
    DT_UINT32  = 768,  /**< uint32 */
};

/* ============================================================================ */
/*                              File I/O + gzip                                */
/* ============================================================================ */

/*******************************************************************************/
/*! \fn    std::vector<uint8_t> read_file_bytes(const std::string& path)
    \brief Slurp the entire contents of a file into memory
    \param  path  source file path
    \return       a std::vector<uint8_t> holding the raw file bytes
*/
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

/*******************************************************************************/
/*! \fn    void write_file_bytes(const std::string& path,
                                 const uint8_t* data, size_t n)
    \brief Write \a n bytes of \a data to \a path, truncating any existing file
    \param  path  destination file path
    \param  data  byte buffer to write
    \param  n     number of bytes
*/
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

/*******************************************************************************/
/*! \fn    bool ends_with(const std::string& s, const std::string& suffix)
    \brief Case-sensitive suffix test on a string
    \param s       the haystack string (typically a file path)
    \param suffix  the needle suffix
    \return        true if \a s ends with \a suffix, false otherwise
*/
bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size()
           && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

/*******************************************************************************/
/*! \fn    std::vector<uint8_t> gunzip(const uint8_t* in, size_t n)
    \brief Inflate a gzipped buffer using zmat's direct miniz helper

    Calls the lower-level `miniz_gzip_uncompress` exposed by zmat,
    which uses a streaming inflate sized for typical NIfTI volumes.
    The returned buffer is malloc'd internally; we copy into a
    std::vector and free the malloc'd buffer before returning so
    the caller never has to worry about manual cleanup.

    \param  in  gzip-encoded buffer (must start with magic 0x1F 0x8B)
    \param  n   number of bytes in \a in
    \return     decompressed payload bytes
*/
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

/*******************************************************************************/
/*! \fn    std::vector<uint8_t> gzip_compress(const uint8_t* in, size_t n)
    \brief Deflate a buffer into gzip format via zmat_encode

    zmat does not expose a standalone `miniz_gzip_compress` helper
    (compression lives inside `zmat_encode`), so we go through
    `zmat_encode(... zmGzip ...)` here. The returned buffer is
    malloc'd by zmat and released via `zmat_free` after copying.

    \param  in  raw buffer to compress
    \param  n   number of bytes in \a in
    \return     gzip-encoded bytes
*/
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

/* ============================================================================ */
/*                       Affine + canonical reorient                          */
/* ============================================================================ */

/*******************************************************************************/
/*! \fn    void quatern_to_mat44(float qb, float qc, float qd,
                                 float qx, float qy, float qz,
                                 float dx, float dy, float dz,
                                 float qfac,
                                 std::array<float, 16>& m)
    \brief Convert NIfTI qform quaternion + offset + pixdim into a 4x4 affine

    Implements the standard NIfTI-1 quaternion-to-matrix conversion,
    matching `nifti_quatern_to_mat44` from nifti_clib's nifti1_io.c.
    The first three rows hold the rotation/scale; the fourth row is
    always (0, 0, 0, 1).

    \param  qb,qc,qd  the three quaternion components stored in the header
    \param  qx,qy,qz  translation offsets (qoffset_x/y/z)
    \param  dx,dy,dz  voxel sizes (pixdim[1..3])
    \param  qfac      qfac flag (+1 standard, -1 indicates Z-flip)
    \param  m         output: 4x4 row-major affine
*/
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

/*******************************************************************************/
/*! \fn    std::array<float, 16> extract_affine(const Nifti1Header& h)
    \brief Choose and decode the file's affine, preferring sform over qform

    Tries the affine sources in NIfTI-1's documented order of priority:
    `sform` (direct 3x4 matrix) if `sform_code` > 0, otherwise the qform
    quaternion if `qform_code` > 0, otherwise a defensive identity
    matrix scaled by `pixdim`. The returned matrix is row-major with
    the standard `(0, 0, 0, 1)` homogeneous bottom row.

    \param  h  the parsed NIfTI-1 header
    \return    a 4x4 row-major affine as a 16-element std::array
*/
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

}  // anonymous namespace

/*******************************************************************************/
/*! \fn    NiftiImage load_nifti_ras(const std::string& path)
    \brief Load a NIfTI-1 file and convert to canonical (Z, Y, X) RAS float32

    Pipeline:

      -# Slurp the file into memory.
      -# Auto-detect gzip via the magic bytes (0x1F 0x8B) and inflate
         if needed.
      -# Validate header (sizeof_hdr=348, magic="n+1"/"ni1", dim[0]>=3,
         no 4D inputs).
      -# Recover the affine via extract_affine() and the axis
         permutation+flip into RAS via axes_to_canonical().
      -# Dispatch on `datatype` to the matching
         copy_reorient_to_canonical<SrcT>() instantiation, producing
         the canonical float32 volume.
      -# Apply `scl_slope` / `scl_inter` scaling if requested.
      -# Compute the canonical affine and per-axis voxel zooms.

    \param  path  source `.nii` or `.nii.gz`
    \return       canonical NiftiImage (volume + affine + perm metadata)
*/
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
    auto canon = canonicalize_affine(out.affine_orig, dst, sgn, out.shape_orig);
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

/*******************************************************************************/
/*! \fn    void save_nifti_labels(const std::string& path,
                                  const NiftiImage& src,
                                  const uint8_t* labels_zyx)
    \brief Save a 3D uint8 labelmap as a NIfTI-1 file

    Builds an in-memory NIfTI-1 image (header + 4 byte padding +
    voxel data), reorients the canonical labelmap back into the
    input file's native voxel axis order via
    copy_reorient_from_canonical(), and writes the buffer to disk
    (auto-gzipped when the path ends in `.gz` / `.GZ`).

    The header carries the input file's sform affine and per-axis
    voxel sizes so the output is spatially coregistered with the
    input.

    \param  path        destination `.nii` or `.nii.gz`
    \param  src         source NiftiImage (provides affine + axis order)
    \param  labels_zyx  canonical (Z, Y, X) labelmap, cZ*cY*cX bytes
*/
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
    copy_reorient_from_canonical(labels_zyx,
                                 src.volume.shape[0], src.volume.shape[1], src.volume.shape[2],
                                 X, Y, Z, dst, sgn, labels_xyz.data());

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


/*******************************************************************************/
/*! \fn    void save_nifti_tpm(const std::string& path,
                               const NiftiImage& src,
                               const float* tpm_canon_czyx,
                               int64_t num_classes)
    \brief Save a 4D float32 tissue probability map as a NIfTI-1 file

    Same machinery as save_nifti_labels but for the per-class softmax
    output: emits a 4D NIfTI with `dim[1..3] = (X, Y, Z)` and
    `dim[4] = num_classes`. The input layout is channel-slowest
    `(C, cZ, cY, cX)` in canonical orientation; each channel is
    independently reoriented to the input axis order, then the
    channel slabs are concatenated back-to-back in memory before
    serialization.

    \param  path             destination file path
    \param  src              source NiftiImage whose header is reused
    \param  tpm_canon_czyx   channel-slowest TPM in canonical orientation
    \param  num_classes      number of channels (NIfTI dim[4])
*/
void save_nifti_tpm(const std::string& path,
                    const NiftiImage& src,
                    const float* tpm_canon_czyx,
                    int64_t num_classes) {
    Nifti1Header h{};
    h.sizeof_hdr = 348;
    h.dim[0] = 4;
    h.dim[1] = static_cast<int16_t>(src.shape_orig[0]);
    h.dim[2] = static_cast<int16_t>(src.shape_orig[1]);
    h.dim[3] = static_cast<int16_t>(src.shape_orig[2]);
    h.dim[4] = static_cast<int16_t>(num_classes);
    h.dim[5] = 1;
    h.dim[6] = 1;
    h.dim[7] = 1;
    h.datatype = DT_FLOAT32;
    h.bitpix = 32;

    auto col_norm = [&](int c) {
        const auto& A = src.affine_orig;
        float v0 = A[0 * 4 + c], v1 = A[1 * 4 + c], v2 = A[2 * 4 + c];
        return std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    };
    h.pixdim[0] = -1.0f;
    h.pixdim[1] = col_norm(0);
    h.pixdim[2] = col_norm(1);
    h.pixdim[3] = col_norm(2);
    h.pixdim[4] = 1.0f;   // unit "step" between channels (semantically meaningless)

    h.vox_offset = static_cast<float>(sizeof(Nifti1Header) + 4);
    h.scl_slope  = 0.0f;
    h.scl_inter  = 0.0f;

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

    // De-canonicalize each channel back to (X, Y, Z) input-axis order.
    // NIfTI 4D layout: dim[1..4] = X, Y, Z, T, with X fastest in memory
    // and channel T slowest. So we concatenate per-channel 3D blocks
    // back-to-back.
    const int64_t X = src.shape_orig[0];
    const int64_t Y = src.shape_orig[1];
    const int64_t Z = src.shape_orig[2];
    const int64_t cZ = src.volume.shape[0];
    const int64_t cY = src.volume.shape[1];
    const int64_t cX = src.volume.shape[2];
    const int64_t per_channel = X * Y * Z;
    const int64_t canon_per_channel = cZ * cY * cX;

    std::array<int, 3> dst{};

    for (int i = 0; i < 3; ++i) {
        dst[i] = src.perm_canon_to_orig[i];
    }

    std::array<int, 3> sgn{};

    for (int i = 0; i < 3; ++i) {
        sgn[i] = src.flip_canon[dst[i]];
    }

    std::vector<float> tpm_xyzc(static_cast<size_t>(per_channel * num_classes), 0.0f);

    for (int64_t c = 0; c < num_classes; ++c) {
        copy_reorient_from_canonical(
            tpm_canon_czyx + c * canon_per_channel,
            cZ, cY, cX,
            X, Y, Z, dst, sgn,
            tpm_xyzc.data() + c * per_channel);
    }

    // Assemble the on-disk image: header + 4 bytes padding + data.
    const size_t data_bytes = static_cast<size_t>(per_channel * num_classes) * sizeof(float);
    const size_t total_size = static_cast<size_t>(h.vox_offset) + data_bytes;
    std::vector<uint8_t> nii(total_size, 0);
    std::memcpy(nii.data(), &h, sizeof(Nifti1Header));
    std::memcpy(nii.data() + static_cast<size_t>(h.vox_offset),
                tpm_xyzc.data(),
                data_bytes);

    if (ends_with(path, ".gz") || ends_with(path, ".GZ")) {
        auto gz = gzip_compress(nii.data(), nii.size());
        write_file_bytes(path, gz.data(), gz.size());
    } else {
        write_file_bytes(path, nii.data(), nii.size());
    }
}

}  // namespace siam
