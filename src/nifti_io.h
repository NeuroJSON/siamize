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

}  // namespace siam
