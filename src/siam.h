// SIAM v0.3 native C++ inference — common types and helpers.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace siam {

// Spatial axis order is (Z, Y, X) internally — same as nnUNet's runtime.
// In-memory layout is contiguous fastest-varying X.
struct Volume {
    std::vector<float> data;
    std::array<int64_t, 3> shape{0, 0, 0};       // (Z, Y, X)

    int64_t numel() const {
        return shape[0] * shape[1] * shape[2];
    }
    int64_t stride(int d) const {
        // contiguous (Z, Y, X)
        if (d == 0) {
            return shape[1] * shape[2];
        }

        if (d == 1) {
            return shape[2];
        }

        return 1;
    }
    float&       at(int64_t z, int64_t y, int64_t x)       {
        return data[z * stride(0) + y * stride(1) + x];
    }
    float        at(int64_t z, int64_t y, int64_t x) const {
        return data[z * stride(0) + y * stride(1) + x];
    }
    void resize(int64_t Z, int64_t Y, int64_t X) {
        shape = {Z, Y, X};
        data.assign(static_cast<size_t>(Z) * Y * X, 0.0f);
    }
};

// 4-channel logits volume used as accumulator. Channel-major: (C, Z, Y, X)
struct LogitsVolume {
    std::vector<float> data;
    int64_t C = 0;
    std::array<int64_t, 3> spat{0, 0, 0};

    int64_t numel() const {
        return C * spat[0] * spat[1] * spat[2];
    }
    int64_t cstride() const {
        return spat[0] * spat[1] * spat[2];
    }
    void resize(int64_t Cn, int64_t Z, int64_t Y, int64_t X) {
        C = Cn;
        spat = {Z, Y, X};
        data.assign(static_cast<size_t>(numel()), 0.0f);
    }
    float* channel_ptr(int64_t c) {
        return data.data() + c * cstride();
    }
    const float* channel_ptr(int64_t c) const {
        return data.data() + c * cstride();
    }
};

// NIfTI image with its affine + canonical reorientation info, so we can
// (a) load → reorient to RAS → process → (b) restore the original axis order
// and (c) write a NIfTI that overlays the input exactly.
struct NiftiImage {
    Volume volume;                                // RAS-canonical (Z, Y, X) data, float32
    std::array<float, 16> affine_orig{};          // input file affine, row-major 4x4
    std::array<float, 16> affine_canon{};         // affine after to-canonical reorient
    std::array<float, 3> zooms_canon{};           // voxel sizes in canonical (X, Y, Z)
    // Reorient: how to go from canonical (Z, Y, X) back to original storage axes.
    // perm_canon_to_orig[i] in {0,1,2} is the canonical axis that maps to the i-th original axis.
    // flip_canon[i] in {-1, +1} per canonical axis is the flip applied during canonicalization.
    std::array<int, 3> perm_canon_to_orig{0, 1, 2};   // (X-orig, Y-orig, Z-orig) source axes in canon
    std::array<int, 3> flip_canon{1, 1, 1};            // per canonical (X, Y, Z) axis
    std::array<int64_t, 3> shape_orig{0, 0, 0};        // (X, Y, Z) in original file order
    int  datatype_orig = 0;                            // NIFTI datatype code of input
};

}  // namespace siam
