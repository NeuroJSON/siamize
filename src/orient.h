// RAS canonicalization primitives, shared between nifti_io and the MEX
// wrapper. Given an arbitrary input voxel grid + 3x4 (or 4x4) affine
// describing how voxel axes map to world (R, A, S) coordinates, this
// computes the per-axis permutation + flip that brings the data to
// "closest canonical RAS" (nibabel's `as_closest_canonical` equivalent)
// and provides copy helpers to apply / undo the reorientation.
//
// Affine convention: row-major std::array<float, 16> for a 4x4 matrix
// where row r, column c is affine[r * 4 + c]. The last row is 0,0,0,1.

#pragma once
#include "siam.h"
#include <array>
#include <cstdint>

namespace siam {

// For each input voxel axis i in 0..2:
//   dst[i] in {0,1,2}  -- which canonical world axis (R=0, A=1, S=2) this
//                        input axis maps to most strongly.
//   sgn[i] in {-1, +1} -- sign of the mapping.
void axes_to_canonical(const std::array<float, 16>& affine,
                       std::array<int, 3>& dst,
                       std::array<int, 3>& sgn);

// Apply axes_to_canonical (dst + sgn) to a raw input buffer to fill
// `out` in canonical (Z, Y, X) layout with X-fastest.
//
//   src        : pointer to (X * Y * Z) values, X-fastest in memory (NIfTI / column-major MATLAB convention)
//   X, Y, Z    : input shape (in input-axis order, not canonical)
//   dst, sgn   : from axes_to_canonical
//   out        : pre-allocated Volume; this routine calls out.resize(canonZ, canonY, canonX).
template <typename SrcT>
void copy_reorient_to_canonical(const SrcT* src,
                                int64_t X, int64_t Y, int64_t Z,
                                const std::array<int, 3>& dst,
                                const std::array<int, 3>& sgn,
                                Volume& out);

// Inverse of `copy_reorient_to_canonical`. Takes labels already in
// canonical (Z, Y, X) order plus the original input shape and writes
// them back into input-axis order with X-fastest layout.
template <typename DstT>
void copy_reorient_from_canonical(const uint8_t* labels_canon_zyx,
                                  int64_t canonZ, int64_t canonY, int64_t canonX,
                                  int64_t X, int64_t Y, int64_t Z,
                                  const std::array<int, 3>& dst,
                                  const std::array<int, 3>& sgn,
                                  DstT* out_xyz);

// Given an input affine and its dst/sgn, build the canonical-frame affine
// (the affine that describes the data after reorientation).
std::array<float, 16> canonicalize_affine(const std::array<float, 16>& input_affine,
        const std::array<int, 3>& dst,
        const std::array<int, 3>& sgn,
        const std::array<int64_t, 3>& input_shape_xyz);

}  // namespace siam
