#pragma once
#include "siam.h"
#include <string>

namespace siam {

// Load a NIfTI volume, reorient to RAS canonical (Z, Y, X), return as float32.
NiftiImage load_nifti_ras(const std::string& path);

// Save a uint8 label volume that occupies the same grid as `src`. The label
// data is in canonical (Z, Y, X) order and will be reoriented back to match
// the original file before writing.
void save_nifti_labels(const std::string& path,
                       const NiftiImage& src,
                       const uint8_t* labels_zyx);

// Save a 4D float32 tissue probability map (e.g. softmax over the SIAM
// classes). Data is provided in canonical (C, Z, Y, X) layout with the
// channel axis slowest, and is reoriented per-channel back into the
// original file's voxel axis order before writing. The output is a 4D
// NIfTI-1 with dim[1..3] = the input X/Y/Z and dim[4] = num_classes.
void save_nifti_tpm(const std::string& path,
                    const NiftiImage& src,
                    const float* tpm_canon_czyx,
                    int64_t num_classes);

}  // namespace siam
