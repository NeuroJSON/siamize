#pragma once
#include "siam.h"
#include <array>

namespace siam {

// Crop to the bounding box of nonzero voxels.
// Returns the cropped data + bbox (Z, Y, X) = {{z_lo, z_hi}, {y_lo, y_hi}, {x_lo, x_hi}}.
struct CropResult {
    Volume cropped;
    std::array<std::array<int64_t, 2>, 3> bbox;   // [axis][{lo, hi}], hi exclusive
};
CropResult crop_to_nonzero(const Volume& vol);

// In-place z-score normalization (mean/std over the whole volume).
void zscore_inplace(Volume& vol);

// Trilinear resample to a target shape with mode='edge' (skimage parity).
Volume resample_trilinear(const Volume& src,
                          int64_t Z, int64_t Y, int64_t X);

// Cubic Catmull-Rom (3rd-order Hermite) resample with edge clamping.
// Matches skimage `resize(order=3, mode='edge', anti_aliasing=False)` to
// ~99.9% argmax agreement on real images (not bit-exact to scipy's B-spline
// but close enough that segmentation differences are at noise level).
Volume resample_cubic(const Volume& src,
                      int64_t Z, int64_t Y, int64_t X);

// Compute the new shape after spacing change: round(old_spacing/new_spacing * old_shape).
std::array<int64_t, 3> compute_new_shape(std::array<int64_t, 3> old_shape,
        std::array<float, 3> old_spacing,
        std::array<float, 3> new_spacing);

}  // namespace siam
