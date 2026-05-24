#pragma once
#include "siam.h"
#include <cstdint>
#include <string>

namespace siam {

// JNIfTI (NeuroJSON) container I/O. Writes the same NIfTI metadata +
// voxel data the existing nifti_io.cpp handles, but wrapped in a JSON
// (.jnii) or BJData (.bnii) container per
// https://neurojson.org/jnifti . Voxel data is zlib-compressed and
// stored as a JData annotated array (_ArrayZipData_) -- typically
// 3-5x smaller on disk than gzipped raw NIfTI for label/TPM volumes
// thanks to per-voxel value redundancy.

// Save a 3D uint8 labelmap as JNIfTI. format is "jnii" (text JSON,
// _ArrayZipData_ holds base64 of the compressed bytes) or "bnii"
// (BJData binary, _ArrayZipData_ holds the raw compressed bytes).
void save_jnifti_labels(const std::string& path,
                        const NiftiImage& src,
                        const uint8_t* labels_zyx,
                        const std::string& format);

// Save a 4D float32 tissue probability map as JNIfTI. tpm_canon_czyx
// has channel-slowest layout (matching save_nifti_tpm); shape is
// (num_classes, canon_Z, canon_Y, canon_X). Reoriented per-channel to
// the input file's voxel axis order before compression.
void save_jnifti_tpm(const std::string& path,
                     const NiftiImage& src,
                     const float* tpm_canon_czyx,
                     int64_t num_classes,
                     const std::string& format);

// Load a JNIfTI file (.jnii or .bnii) and return it as a NiftiImage
// in canonical (Z, Y, X) RAS layout. Handles _ArrayZipData_ via zlib
// decompression and _ArrayData_ as a raw passthrough.
NiftiImage load_jnifti_ras(const std::string& path);

}  // namespace siam
