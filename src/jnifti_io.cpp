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
**  \li \c (\b JGIFTI) The JGIFTI specification, Q. Fang, NeuroJSON project.
**         Defines the LabelTable schema attached at
**         NIFTIHeader._DataInfo_.LabelTable for labelmap output.
**         <a href="https://github.com/NeuroJSON/jgifti">https://github.com/NeuroJSON/jgifti</a>
**  \li \c (\b JNIfTI) The JNIfTI specification, Q. Fang, NeuroJSON project.
**         <a href="https://neurojson.org/jnifti">https://neurojson.org/jnifti</a>
**  \li \c (\b JData) The JData specification, Q. Fang, NeuroJSON project.
**         <a href="https://neurojson.org/jdata">https://neurojson.org/jdata</a>
**  \li \c (\b BJData) The Binary JData specification, Q. Fang, NeuroJSON project.
**         <a href="https://neurojson.org/bjdata">https://neurojson.org/bjdata</a>
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    jnifti_io.cpp
\brief   JNIfTI (.jnii / .bnii) container reader / writer implementation

This translation unit implements siamize's JNIfTI I/O path. It mirrors
the NIfTI-1 functionality in nifti_io.cpp but emits / consumes JNIfTI
containers per https://neurojson.org/jnifti -- JData-annotated JSON
(.jnii, text) or BJData (.bnii, binary JSON). Voxel data is always
stored compressed via the `_ArrayZipData_` field; compression and
base64 encoding/decoding are delegated to zmat (src/zmat/zmat.h),
matching the codec siamize already uses for `.nii.gz` gzip I/O.

The reader handles the dtype variety that real-world NIfTI volumes
contain (uint8 / int8 / int16 / uint16 / int32 / uint32 / float32 /
float64) and converts to a column-major (NIfTI X-fastest) float32
volume, which copy_reorient_to_canonical() then turns into the
(Z, Y, X) RAS layout the rest of siamize expects. The writer mirrors
jsonlab's jdataencode convention: it permutes axes BEFORE flattening
so the on-disk byte order is row-major (last-dim-fastest), and emits
`_ArrayZipSize_ = [1, N]` (the jsonlab "flat-shape" convention for
compressed arrays).
*******************************************************************************/

#include "jnifti_io.h"
#include "orient.h"
#include "siam.h"

#include "nlohmann/json.hpp"
#include "zmat.h"   // declarations only; impl lives in nifti_io.cpp's TU

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace siam {

using json = nlohmann::ordered_json;  ///< preserve key order in emitted JSON

namespace {

/*******************************************************************************/
/*! \fn    bool ends_with(const std::string& s, const std::string& suffix)
    \brief Case-sensitive suffix test on a path string
    \param s       the haystack string (typically a file path)
    \param suffix  the needle suffix (e.g. ".bnii")
    \return true if \a s ends with \a suffix, false otherwise
*/
bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size()
           && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/*******************************************************************************/
/*! \fn    std::vector<uint8_t> zmat_xform(const uint8_t* in, size_t n,
                                           int zipid, int iscompress)
    \brief Thin C++ wrapper around zmat's all-in-one driver zmat_run

    Returns the output buffer in a std::vector and throws on error. zmat
    owns the raw output buffer via malloc; this wrapper copies into a
    vector and calls zmat_free before returning so the caller never has
    to worry about manual cleanup. JNIfTI only mandates zlib for the
    `_ArrayZipType_` field, but zmat itself supports zmGzip, zmLzma,
    zmZstd, zmBlosc2*, etc. -- the same wrapper handles them all.

    \param  in          input buffer (read-only)
    \param  n           number of bytes in \a in
    \param  zipid       codec ID: zmZlib / zmGzip / zmBase64 / ...
    \param  iscompress  1 = encode/compress, 0 = decode/decompress
    \return             a std::vector<uint8_t> holding the transformed bytes
*/
std::vector<uint8_t> zmat_xform(const uint8_t* in, size_t n, int zipid, int iscompress) {
    unsigned char* out = nullptr;
    size_t outlen = 0;
    int zret = 0;
    int rc = zmat_run(n, const_cast<unsigned char*>(in), &outlen, &out,
                      zipid, &zret, iscompress);

    if (rc != 0 || out == nullptr) {
        if (out) {
            zmat_free(&out);
        }

        throw std::runtime_error(
            std::string("zmat_run failed (zipid=") + std::to_string(zipid)
            + ", iscompress=" + std::to_string(iscompress)
            + ", rc=" + std::to_string(rc)
            + ", zret=" + std::to_string(zret) + ")");
    }

    std::vector<uint8_t> result(out, out + outlen);
    zmat_free(&out);
    return result;
}

/*******************************************************************************/
/*! \fn    template <typename T>
           std::vector<T> col_to_row_major(const T* col,
                                           const std::vector<int64_t>& shape)
    \brief Transpose a flat buffer from column-major to row-major layout

    Column-major == NIfTI X-fastest / MATLAB native; row-major ==
    last-dim-fastest / numpy C-order. Required before writing
    `_ArrayZipData_` because jsonlab's jdataencode internally
    permutes axes via `permute(item, ndims:-1:1)` before flattening
    (see jsonlab/jdataencode.m lines 475-480), so the on-disk byte
    order is last-dim-fastest regardless of any `_ArrayOrder_` tag.
    Mirroring that here ensures round-trip identity through jsonlab's
    loadjd / jdatadecode without channel scrambling.

    \tparam T      element type (uint8_t, float, etc.)
    \param  col    column-major flat buffer of length prod(shape)
    \param  shape  N-D shape (X, Y, Z[, C])
    \return        row-major flat buffer of the same length
*/
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

/*******************************************************************************/
/*! \fn    template <typename T> std::string jdata_dtype()
    \brief Map a C++ scalar type to the JData `_ArrayType_` string

    JData uses MATLAB-style dtype names ("single" / "double" instead of
    "float32" / "float64"). Specializations are provided for the dtypes
    siamize actually emits (writer side). The reader supports a wider
    set (uint16, uint32, etc.) via decode_nifti_data().
*/
template <typename T> std::string jdata_dtype();
template <> std::string jdata_dtype<uint8_t>()  { return "uint8";  }
template <> std::string jdata_dtype<int16_t>()  { return "int16";  }
template <> std::string jdata_dtype<int32_t>()  { return "int32";  }
template <> std::string jdata_dtype<float>()    { return "single"; }
template <> std::string jdata_dtype<double>()   { return "double"; }

/*******************************************************************************/
/*! \fn    template <typename T>
           json jdata_annotated(const T* data,
                                const std::vector<int64_t>& shape,
                                bool binary_format)
    \brief Build a JData annotated-array sub-object for one volume buffer

    Permutes \a data from column-major to row-major (jsonlab convention),
    zlib-compresses the result, and packages it as a JData annotated
    array with `_ArrayType_`, `_ArraySize_`, `_ArrayZipType_`,
    `_ArrayZipSize_`, and `_ArrayZipData_` fields. `_ArrayZipSize_` is
    always `[1, N]` (jsonlab's flat-shape convention for compressed
    arrays); the actual N-D shape is recovered by jsonlab from
    `_ArraySize_` via its format>1.9 reshape+permute pair.

    \tparam T              scalar element type (uint8_t, float, ...)
    \param  data           column-major flat input, length prod(shape)
    \param  shape          N-D shape, in NIfTI-native (X-fastest) order
    \param  binary_format  true for .bnii (raw bytes in BJData binary string),
                           false for .jnii (base64-encoded ASCII string)
    \return                a JSON object holding the annotated array
*/
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
    auto comp = zmat_xform(reinterpret_cast<const uint8_t*>(row.data()), raw_bytes,
                         zmZlib, /*iscompress=*/1);

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
        auto b64 = zmat_xform(comp.data(), comp.size(), zmBase64, /*iscompress=*/1);
        arr["_ArrayZipData_"] = std::string(b64.begin(), b64.end());
    }

    return arr;
}

/*******************************************************************************/
/*! \fn    json build_header(const NiftiImage& src,
                             const std::vector<int64_t>& dim)
    \brief Build the JNIfTI `NIFTIHeader` sub-object for the output file

    Carries the input image's affine and voxel size into the output
    header so downstream readers see the same spatial metadata the
    original NIfTI carried. Affine is written as a 3x4 nested JSON
    array (the form jsonlab's text path emits for small matrices; the
    reader also accepts the JData annotated form for BJData files).

    \param  src  source NiftiImage (provides affine_orig)
    \param  dim  output volume shape (3D for labelmaps, 4D for TPM)
    \return      a NIFTIHeader JSON object
*/
/*******************************************************************************/
/*! \fn    json build_label_table(ClassSet class_set, int num_classes)
    \brief Build a JGIFTI-style LabelTable for the labelmap output

    Constructs the per-class anatomical name + RGBA color dictionary
    that goes into `NIFTIHeader._DataInfo_.LabelTable`. Follows the
    JGIFTI specification's object form (keyed by stringified integer
    label IDs) -- see
    https://github.com/NeuroJSON/jgifti/blob/main/JGIFTI_specification.md

    Two presets are emitted today:

      - SIAM v0.3 (18 classes, when class_set == CUSTOM_N AND
        num_classes == 18): the SIAM tissue dictionary from
        label_siamV03_.json (background, GM, WM, CSF, CSFv, cerGM,
        Thal, Pal, Put, Caud, Accu, Amyg, Hippo, Dura, vascular,
        Skull, Head, Anomalies).
      - SPM (6 classes, when class_set == SPM): GM, WM, CSF, Bone,
        Soft, Air in spm12/tpm/TPM.nii channel order.

    For other class sets we return an empty json::object() and the
    caller omits LabelTable entirely; viewers will then fall back to
    their default colormap. RGBA components are normalized floats in
    [0, 1]; "background"/"Air" entries use alpha=0 (transparent) per
    the JGIFTI rule for unassigned regions.

    \param  class_set     semantic label set selector
    \param  num_classes   total class count (only used to decide
                          whether to emit SIAM v0.3 names)
    \return  LabelTable JSON object, or empty if the class set is
             unknown
*/
json build_label_table(ClassSet class_set, int num_classes) {
    // {label-id -> {"Label": name, "RGBA": [r,g,b,a]}}
    json tbl = json::object();

    auto add = [&](int id, const char* name,
                   float r, float g, float b, float a) {
        json entry = json::object();
        entry["Label"] = name;
        entry["RGBA"]  = std::vector<float>{r, g, b, a};
        tbl[std::to_string(id)] = entry;
    };

    if (class_set == ClassSet::SPM) {
        // SPM12 6-class TPM order: GM, WM, CSF, Bone, Soft, Air.
        // Air keeps alpha=0 per JGIFTI "unassigned -> transparent".
        add(0, "GM",   0.700f, 0.700f, 0.700f, 1.0f);
        add(1, "WM",   0.950f, 0.950f, 0.950f, 1.0f);
        add(2, "CSF",  0.000f, 0.600f, 0.900f, 1.0f);
        add(3, "Bone", 0.900f, 0.850f, 0.550f, 1.0f);
        add(4, "Soft", 0.950f, 0.750f, 0.650f, 1.0f);
        add(5, "Air",  0.000f, 0.000f, 0.000f, 0.0f);
        return tbl;
    }

    if (class_set == ClassSet::CUSTOM_N && num_classes == 18) {
        // SIAM v0.3 anatomical dictionary, from label_siamV03_.json.
        // Background gets alpha=0; deep-gray nuclei get warm hues to
        // visually separate from cortical GM; CSF / CSFv get blue
        // variants; skull/head get bone/flesh tones; anomalies get
        // magenta for high-visibility QC.
        add( 0, "background", 0.000f, 0.000f, 0.000f, 0.0f);
        add( 1, "GM",         0.700f, 0.700f, 0.700f, 1.0f);
        add( 2, "WM",         0.950f, 0.950f, 0.950f, 1.0f);
        add( 3, "CSF",        0.000f, 0.600f, 0.900f, 1.0f);
        add( 4, "CSFv",       0.000f, 0.300f, 0.700f, 1.0f);
        add( 5, "cerGM",      0.550f, 0.550f, 0.550f, 1.0f);
        add( 6, "Thal",       0.800f, 0.400f, 0.300f, 1.0f);
        add( 7, "Pal",        0.950f, 0.600f, 0.100f, 1.0f);
        add( 8, "Put",        0.700f, 0.200f, 0.200f, 1.0f);
        add( 9, "Caud",       0.950f, 0.850f, 0.200f, 1.0f);
        add(10, "Accu",       0.550f, 0.350f, 0.200f, 1.0f);
        add(11, "Amyg",       0.450f, 0.850f, 0.450f, 1.0f);
        add(12, "Hippo",      0.150f, 0.550f, 0.200f, 1.0f);
        add(13, "Dura",       0.700f, 0.600f, 0.400f, 1.0f);
        add(14, "vascular",   0.800f, 0.050f, 0.050f, 1.0f);
        add(15, "Skull",      0.900f, 0.850f, 0.550f, 1.0f);
        add(16, "Head",       0.950f, 0.750f, 0.650f, 1.0f);
        add(17, "Anomalies",  1.000f, 0.000f, 1.000f, 1.0f);
        return tbl;
    }

    // Unknown class set / count -> caller omits LabelTable.
    return json::object();
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

/*******************************************************************************/
/*! \fn    void write_jnifti_root(const std::string& path,
                                  const json& j, bool binary)
    \brief Serialize the root JSON object and write it to disk

    For text output, dumps the JSON tree with nlohmann/json's default
    compact formatter. For binary output, uses BJData's optimized
    "use_count=true, use_type=true" mode which produces typed `[$T#`
    array headers, matching what jsonlab's saveubjson / savebj emits.

    \param  path    destination file path
    \param  j       fully-populated JNIfTI root object
    \param  binary  true to emit BJData (.bnii), false for JSON text (.jnii)
*/
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

/*******************************************************************************/
/*! \fn    template <typename T>
           std::vector<T> row_to_col_major(const T* row,
                                           const std::vector<int64_t>& shape)
    \brief Transpose a flat buffer from row-major to column-major

    Inverse of col_to_row_major: takes a buffer flat-stored in
    last-dim-fastest (row-major / numpy C-order) layout and produces
    a buffer flat-stored in first-dim-fastest (column-major /
    MATLAB-native / NIfTI X-fastest) layout, of the same N-D shape.
    Used on the read path to recover NIfTI-native voxel order from
    the wire-format byte stream.

    \tparam T      element type
    \param  row    row-major flat buffer of length prod(shape)
    \param  shape  N-D shape
    \return        column-major flat buffer of the same length
*/
template <typename T>
std::vector<T> row_to_col_major(const T* row, const std::vector<int64_t>& shape) {
    size_t n = 1;

    for (auto d : shape) {
        n *= static_cast<size_t>(d);
    }

    std::vector<T> col(n);
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
        // Decode p as row-major linear -> per-axis indices -> col-major linear.
        size_t rem = p;
        size_t col_idx = 0;

        for (int d = 0; d < ndim; ++d) {
            int64_t i = static_cast<int64_t>(rem / static_cast<size_t>(row_stride[d]));
            rem -= static_cast<size_t>(i) * static_cast<size_t>(row_stride[d]);
            col_idx += static_cast<size_t>(i) * static_cast<size_t>(col_stride[d]);
        }

        col[col_idx] = row[p];
    }

    return col;
}

/*******************************************************************************/
/*! \fn    std::vector<uint8_t> read_file_bytes(const std::string& path)
    \brief Slurp the entire contents of a file into memory

    \param  path  source file path
    \return       a std::vector<uint8_t> holding the raw file bytes
*/
std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);

    if (!f) {
        throw std::runtime_error("failed to open " + path + " for reading");
    }

    auto sz = static_cast<std::streamsize>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));

    if (sz > 0 && !f.read(reinterpret_cast<char*>(buf.data()), sz)) {
        throw std::runtime_error("failed to read " + path);
    }

    return buf;
}

/*******************************************************************************/
/*! \fn    json parse_jnifti(const std::string& path, bool& is_binary)
    \brief Parse a JNIfTI file (.jnii or .bnii) into a nlohmann::json tree

    The container format is inferred from the file extension
    (case-insensitive): `.bnii` / `.BNII` -> BJData binary, anything
    else -> JSON text.

    \param  path       source file path
    \param  is_binary  output: set true if the file was parsed as BJData
    \return            parsed JSON tree (root object)
*/
json parse_jnifti(const std::string& path, bool& is_binary) {
    is_binary = ends_with(path, ".bnii") || ends_with(path, ".BNII");
    auto bytes = read_file_bytes(path);

    if (is_binary) {
        return json::from_bjdata(bytes);
    }

    return json::parse(bytes.begin(), bytes.end());
}

/*******************************************************************************/
/*! \fn    std::array<float, 16> extract_affine(const json& nii_header)
    \brief Extract a 4x4 row-major affine from `NIFTIHeader.Affine`

    JNIfTI files written by different tools encode the affine in
    different ways:

      1. Plain nested JSON array `[[r0c0..r0c3], [r1...], [r2...]]` --
         emitted by jsonlab's text-JSON path for small matrices, and by
         siamize's own writer.

      2. JData annotated array
         `{_ArrayType_, _ArraySize_=[rows, 4], _ArrayData_=[12-or-16 values]}`
         where the data is row-major after jsonlab's permute-reverse
         step -- emitted by jsonlab's BJData path.

    Both forms are accepted. Missing rows (a 3x4 affine) are interpreted
    as having an implicit `[0, 0, 0, 1]` bottom row.

    \param  nii_header  the parsed `NIFTIHeader` JSON object
    \return             the affine as a 16-element row-major std::array
*/
std::array<float, 16> extract_affine(const json& nii_header) {
    std::array<float, 16> a{};
    a[0] = a[5] = a[10] = a[15] = 1.0f;

    if (!nii_header.is_object() || !nii_header.contains("Affine")) {
        throw std::runtime_error("NIFTIHeader.Affine missing in JNIfTI input");
    }

    const json& A = nii_header["Affine"];

    if (A.is_object() && A.contains("_ArrayData_") && A.contains("_ArraySize_")) {
        // JData annotated 2D matrix.
        const json& sz = A["_ArraySize_"];

        if (!sz.is_array() || sz.size() != 2) {
            throw std::runtime_error(
                "NIFTIHeader.Affine: annotated _ArraySize_ must be 2D");
        }

        size_t rows = sz[0].get<size_t>();
        size_t cols = sz[1].get<size_t>();

        if ((rows != 3 && rows != 4) || cols != 4) {
            throw std::runtime_error(
                "NIFTIHeader.Affine annotated shape must be 3x4 or 4x4");
        }

        const json& ad = A["_ArrayData_"];

        if (!ad.is_array() || ad.size() != rows * cols) {
            throw std::runtime_error(
                "NIFTIHeader.Affine: _ArrayData_ length mismatch");
        }

        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                a[r * 4 + c] = ad[r * cols + c].get<float>();
            }
        }

        return a;
    }

    if (!A.is_array()) {
        throw std::runtime_error(
            "NIFTIHeader.Affine must be a nested JSON array or a JData "
            "annotated matrix");
    }

    size_t rows = A.size();

    if (rows != 3 && rows != 4) {
        throw std::runtime_error("NIFTIHeader.Affine must have 3 or 4 rows");
    }

    for (size_t r = 0; r < rows; ++r) {
        if (!A[r].is_array() || A[r].size() != 4) {
            throw std::runtime_error("NIFTIHeader.Affine row must be length-4 array");
        }

        for (size_t c = 0; c < 4; ++c) {
            a[r * 4 + c] = A[r][c].get<float>();
        }
    }

    return a;
}

/*******************************************************************************/
/*! \fn    std::vector<uint8_t> decode_nifti_data(const json& nd,
                                                  bool is_binary,
                                                  std::vector<int64_t>& shape,
                                                  std::string& dtype)
    \brief Decode `NIFTIData` into a flat row-major byte buffer

    Handles both wire-format paths:

      - **Compressed**: `_ArrayZipData_` (zlib bytes, in either a BJData
        binary string, base64-encoded JSON string, or a defensive
        numeric-array fallback). Decompressed via zmat.
      - **Uncompressed**: `_ArrayData_` as a flat numeric array (or a
        BJData binary string for uint8). Each element is marshalled
        into its dtype's wire representation via std::memcpy.

    The output buffer is in row-major (last-dim-fastest) order with
    the dtype reported via the \a dtype out-parameter; downstream
    code converts to column-major + float32 via to_float_col_major().

    \param  nd         the parsed `NIFTIData` JSON object
    \param  is_binary  whether the source file was BJData (currently
                       unused; the in-payload encoding is self-describing)
    \param  shape      output: the N-D shape from `_ArraySize_`
    \param  dtype      output: the JData dtype string from `_ArrayType_`
    \return            flat row-major byte buffer of length prod(shape)*sizeof(dtype)
*/
std::vector<uint8_t> decode_nifti_data(const json& nd,
                                       bool is_binary,
                                       std::vector<int64_t>& shape,
                                       std::string& dtype) {
    if (!nd.is_object() || !nd.contains("_ArraySize_") || !nd.contains("_ArrayType_")) {
        throw std::runtime_error(
            "NIFTIData must be a JData annotated array with "
            "_ArrayType_ + _ArraySize_ (got non-struct or missing fields)");
    }

    dtype = nd["_ArrayType_"].get<std::string>();
    shape.clear();

    for (const auto& d : nd["_ArraySize_"]) {
        shape.push_back(d.get<int64_t>());
    }

    size_t n_elem = 1;

    for (auto d : shape) {
        n_elem *= static_cast<size_t>(d);
    }

    auto type_bytes = [&]() -> size_t {
        if (dtype == "uint8"  || dtype == "int8")    { return 1; }

        if (dtype == "int16"  || dtype == "uint16")  { return 2; }

        if (dtype == "int32"  || dtype == "uint32"
                || dtype == "single" || dtype == "float32") {
            return 4;
        }

        if (dtype == "int64"  || dtype == "uint64"
                || dtype == "double" || dtype == "float64") {
            return 8;
        }

        throw std::runtime_error("unsupported _ArrayType_: " + dtype);
    };
    const size_t raw_bytes = n_elem * type_bytes();

    // Compressed path.
    if (nd.contains("_ArrayZipData_")) {
        std::string ziptype =
            nd.contains("_ArrayZipType_") ? nd["_ArrayZipType_"].get<std::string>() : "zlib";

        if (ziptype != "zlib") {
            throw std::runtime_error(
                "unsupported _ArrayZipType_: " + ziptype + " (only 'zlib')");
        }

        const json& zd = nd["_ArrayZipData_"];
        std::vector<uint8_t> comp;

        if (zd.is_binary()) {
            // .bnii: raw zlib bytes already in a BJData byte string.
            const auto& b = zd.get_binary();
            comp.assign(b.begin(), b.end());
        } else if (zd.is_string()) {
            // .jnii: base64 ASCII -> raw zlib bytes via zmat.
            const std::string& s = zd.get<std::string>();
            comp = zmat_xform(reinterpret_cast<const uint8_t*>(s.data()), s.size(),
                            zmBase64, /*iscompress=*/0);
        } else if (zd.is_array()) {
            // Defensive: a few encoders emit byte payloads as numeric arrays.
            comp.reserve(zd.size());

            for (const auto& b : zd) {
                comp.push_back(static_cast<uint8_t>(b.get<int>()));
            }
        } else {
            throw std::runtime_error("_ArrayZipData_ has unexpected type");
        }

        auto raw = zmat_xform(comp.data(), comp.size(), zmZlib, /*iscompress=*/0);

        if (raw.size() != raw_bytes) {
            throw std::runtime_error(
                "_ArrayZipData_ decompressed to " + std::to_string(raw.size())
                + " bytes, expected " + std::to_string(raw_bytes));
        }

        return raw;
    }

    // Uncompressed path: _ArrayData_ is a flat numeric array.
    if (!nd.contains("_ArrayData_")) {
        throw std::runtime_error(
            "NIFTIData has neither _ArrayZipData_ nor _ArrayData_");
    }

    const json& ad = nd["_ArrayData_"];
    std::vector<uint8_t> raw(raw_bytes);
    // Walk the JSON array and write per-element values into `raw`. Handle
    // each supported dtype explicitly so we get exact wire-format
    // marshalling (no implicit type promotion through json::get<double>()).
    auto write_one = [&](size_t i, const json& v) {
        if (dtype == "uint8") {
            raw[i] = v.get<uint8_t>();
        } else if (dtype == "int8") {
            int8_t x = v.get<int8_t>();
            std::memcpy(&raw[i], &x, 1);
        } else if (dtype == "int16") {
            int16_t x = v.get<int16_t>();
            std::memcpy(&raw[i * 2], &x, 2);
        } else if (dtype == "uint16") {
            uint16_t x = v.get<uint16_t>();
            std::memcpy(&raw[i * 2], &x, 2);
        } else if (dtype == "int32") {
            int32_t x = v.get<int32_t>();
            std::memcpy(&raw[i * 4], &x, 4);
        } else if (dtype == "uint32") {
            uint32_t x = v.get<uint32_t>();
            std::memcpy(&raw[i * 4], &x, 4);
        } else if (dtype == "single" || dtype == "float32") {
            float x = v.get<float>();
            std::memcpy(&raw[i * 4], &x, 4);
        } else if (dtype == "double" || dtype == "float64") {
            double x = v.get<double>();
            std::memcpy(&raw[i * 8], &x, 8);
        } else {
            throw std::runtime_error("unsupported _ArrayType_ for direct read: " + dtype);
        }
    };

    if (ad.is_array()) {
        if (ad.size() != n_elem) {
            throw std::runtime_error(
                "_ArrayData_ length " + std::to_string(ad.size())
                + " != expected " + std::to_string(n_elem));
        }

        for (size_t i = 0; i < n_elem; ++i) {
            write_one(i, ad[i]);
        }
    } else if (ad.is_binary() && dtype == "uint8") {
        // BJData binary uint8 array.
        const auto& b = ad.get_binary();

        if (b.size() != raw_bytes) {
            throw std::runtime_error("_ArrayData_ binary length mismatch");
        }

        std::copy(b.begin(), b.end(), raw.begin());
        (void)is_binary;
    } else {
        throw std::runtime_error("_ArrayData_ has unexpected type");
    }

    (void)is_binary;
    return raw;
}

/*******************************************************************************/
/*! \fn    template <typename SrcT>
           std::vector<float> to_float_col_major(const uint8_t* row_bytes,
                                                 const std::vector<int64_t>& shape)
    \brief Convert a typed row-major byte buffer to a column-major float32 volume

    Reinterprets \a row_bytes as `SrcT*`, casts each element to float,
    then transposes to column-major via row_to_col_major(). Matches
    the dtype handling of NIfTI-1: the inference pipeline only ever
    sees float32 with NIfTI-native X-fastest axis order.

    \tparam SrcT       source element type (uint8_t, int16_t, float, ...)
    \param  row_bytes  row-major byte buffer
    \param  shape      N-D shape (X, Y, Z)
    \return            column-major float32 buffer of length prod(shape)
*/
template <typename SrcT>
std::vector<float> to_float_col_major(const uint8_t* row_bytes, const std::vector<int64_t>& shape) {
    size_t n = 1;

    for (auto d : shape) {
        n *= static_cast<size_t>(d);
    }

    const SrcT* src = reinterpret_cast<const SrcT*>(row_bytes);
    std::vector<float> row_f(n);

    for (size_t i = 0; i < n; ++i) {
        row_f[i] = static_cast<float>(src[i]);
    }

    return row_to_col_major<float>(row_f.data(), shape);
}

}  // anonymous namespace


/*******************************************************************************/
/*! \fn    void save_jnifti_labels(const std::string& path,
                                   const NiftiImage& src,
                                   const uint8_t* labels_zyx,
                                   const std::string& format)
    \brief Save a 3D uint8 labelmap as a JNIfTI container

    Reorients the canonical (Z, Y, X) labelmap back to the input
    file's native voxel axis order via copy_reorient_from_canonical,
    builds the NIFTIHeader/NIFTIData JSON root, and writes it as
    either text JSON (.jnii) or BJData binary (.bnii).

    \param  path        destination file path
    \param  src         input NiftiImage providing affine + axis order
    \param  labels_zyx  canonical (Z, Y, X) labelmap, length cZ*cY*cX bytes
    \param  format      "jnii" (text JSON) or "bnii" (BJData binary)
*/
void save_jnifti_labels(const std::string& path,
                        const NiftiImage& src,
                        const uint8_t* labels_zyx,
                        const std::string& format,
                        ClassSet class_set,
                        int num_classes) {
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

    // Attach JGIFTI-style LabelTable at NIFTIHeader._DataInfo_.LabelTable
    // when the class set is recognized (SIAM v0.3 18-class or SPM 6-class).
    // Viewers that honour the JGIFTI LabelTable spec will then render
    // anatomical names + per-tissue colors instead of a default colormap.
    json label_table = build_label_table(class_set, num_classes);

    if (!label_table.empty()) {
        json data_info = json::object();
        data_info["LabelTable"] = label_table;
        root["NIFTIHeader"]["_DataInfo_"] = data_info;
    }

    root["NIFTIData"] = jdata_annotated<uint8_t>(
                            data_xyz.data(), {X, Y, Z}, binary);
    write_jnifti_root(path, root, binary);
}


/*******************************************************************************/
/*! \fn    void save_jnifti_tpm(const std::string& path,
                                const NiftiImage& src,
                                const float* tpm_canon_czyx,
                                int64_t num_classes,
                                const std::string& format)
    \brief Save a 4D float32 tissue probability map as a JNIfTI container

    Same machinery as save_jnifti_labels but for the per-class softmax
    output. The source layout is channel-slowest (C, cZ, cY, cX) in
    canonical orientation; each channel is independently reoriented
    via copy_reorient_from_canonical and interleaved into a
    (X, Y, Z, C) volume before serialization.

    \param  path             destination file path
    \param  src              input NiftiImage whose header is reused
    \param  tpm_canon_czyx   channel-slowest TPM in canonical orientation
    \param  num_classes      number of channels (NIfTI dim[4])
    \param  format           "jnii" (text JSON) or "bnii" (BJData binary)
*/
void save_jnifti_tpm(const std::string& path,
                     const NiftiImage& src,
                     const float* tpm_canon_czyx,
                     int64_t num_classes,
                     const std::string& format,
                     ClassSet class_set) {
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

    // Attach JGIFTI-style LabelTable for the 4D TPM as well: each
    // channel of the (X, Y, Z, C) volume corresponds to one entry in
    // the table. The integer key matches the channel index. Same two
    // presets as the labelmap path -- SIAM v0.3 18-channel default
    // and SPM 6-channel for class_set==SPM.
    json label_table = build_label_table(class_set,
                                         static_cast<int>(num_classes));

    if (!label_table.empty()) {
        json data_info = json::object();
        data_info["LabelTable"] = label_table;
        root["NIFTIHeader"]["_DataInfo_"] = data_info;
    }

    root["NIFTIData"] = jdata_annotated<float>(
                            data_xyzc.data(),
                            {X, Y, Z, num_classes}, binary);
    write_jnifti_root(path, root, binary);
}


/*******************************************************************************/
/*! \fn    NiftiImage load_jnifti_ras(const std::string& path)
    \brief Load a JNIfTI file (.jnii or .bnii) into a canonical NiftiImage

    Pipeline: read file bytes -> parse JSON / BJData -> extract affine
    from `NIFTIHeader.Affine` -> decode `NIFTIData` to a row-major byte
    buffer of the appropriate dtype -> cast to float32, transpose to
    column-major -> compute axis permutation to RAS via
    axes_to_canonical -> copy into canonical (Z, Y, X) order via
    copy_reorient_to_canonical -> compute canonical affine + zooms.

    The returned NiftiImage is bit-compatible with what load_nifti_ras()
    produces, so the downstream pipeline does not need to know which
    container fed it.

    \param  path  source file path (.jnii or .bnii, case-insensitive)
    \return       canonical NiftiImage in (Z, Y, X) RAS orientation
*/
NiftiImage load_jnifti_ras(const std::string& path) {
    bool is_binary = false;
    json root = parse_jnifti(path, is_binary);

    if (!root.is_object() || !root.contains("NIFTIHeader") || !root.contains("NIFTIData")) {
        throw std::runtime_error(
            "JNIfTI file must have NIFTIHeader and NIFTIData top-level fields");
    }

    auto affine = extract_affine(root["NIFTIHeader"]);

    std::vector<int64_t> shape;
    std::string dtype;
    auto row_bytes = decode_nifti_data(root["NIFTIData"], is_binary, shape, dtype);

    if (shape.size() != 3) {
        throw std::runtime_error(
            "siamize JNIfTI input must be a 3D scalar volume (got rank "
            + std::to_string(shape.size()) + ")");
    }

    const int64_t X = shape[0];
    const int64_t Y = shape[1];
    const int64_t Z = shape[2];

    std::vector<float> col_f;

    if      (dtype == "uint8")            { col_f = to_float_col_major<uint8_t>(row_bytes.data(), shape); }
    else if (dtype == "int8")             { col_f = to_float_col_major<int8_t>(row_bytes.data(), shape); }
    else if (dtype == "int16")            { col_f = to_float_col_major<int16_t>(row_bytes.data(), shape); }
    else if (dtype == "uint16")           { col_f = to_float_col_major<uint16_t>(row_bytes.data(), shape); }
    else if (dtype == "int32")            { col_f = to_float_col_major<int32_t>(row_bytes.data(), shape); }
    else if (dtype == "uint32")           { col_f = to_float_col_major<uint32_t>(row_bytes.data(), shape); }
    else if (dtype == "single" || dtype == "float32") { col_f = to_float_col_major<float>(row_bytes.data(), shape); }
    else if (dtype == "double" || dtype == "float64") { col_f = to_float_col_major<double>(row_bytes.data(), shape); }
    else {
        throw std::runtime_error("unsupported _ArrayType_ in JNIfTI input: " + dtype);
    }

    NiftiImage out;
    out.affine_orig = affine;
    out.shape_orig  = {X, Y, Z};

    std::array<int, 3> dst{}, sgn{};
    axes_to_canonical(affine, dst, sgn);
    copy_reorient_to_canonical<float>(col_f.data(), X, Y, Z, dst, sgn, out.volume);
    out.affine_canon = canonicalize_affine(affine, dst, sgn, {X, Y, Z});

    // perm_canon_to_orig[i] = the input-axis index that ends up at canonical axis i.
    // dst[i] tells us "canonical axis dst[i] receives input axis i". Invert.
    for (int i = 0; i < 3; ++i) {
        out.perm_canon_to_orig[dst[i]] = i;
    }

    for (int i = 0; i < 3; ++i) {
        out.flip_canon[dst[i]] = sgn[i];
    }

    // Voxel size (X, Y, Z) in canonical: column norms of the canonical affine.
    auto col_norm = [&](int c) {
        const auto& A = out.affine_canon;
        return std::sqrt(A[0 * 4 + c] * A[0 * 4 + c]
                         + A[1 * 4 + c] * A[1 * 4 + c]
                         + A[2 * 4 + c] * A[2 * 4 + c]);
    };
    out.zooms_canon = {
        static_cast<float>(col_norm(0)),
        static_cast<float>(col_norm(1)),
        static_cast<float>(col_norm(2))
    };

    return out;
}

}  // namespace siam
