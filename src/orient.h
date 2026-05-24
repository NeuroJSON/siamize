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
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    orient.h

@brief   RAS canonicalization primitives shared by nifti_io / jnifti_io / MEX

Given an arbitrary input voxel grid and a 3x4 / 4x4 affine that
describes how the voxel axes map to world (Right, Anterior, Superior)
coordinates, this module computes the per-axis permutation + flip
that brings the data to "closest canonical RAS" -- the same outcome
nibabel produces with its `as_closest_canonical` helper. It also
exposes copy helpers that apply or undo the reorientation in-place
on raw buffers, and a small helper that produces the affine of the
canonicalized frame.

Affine convention: row-major std::array<float, 16> for a 4x4 matrix
where row \a r, column \a c is `affine[r * 4 + c]`. The last row is
always `(0, 0, 0, 1)`.
*******************************************************************************/

#ifndef SIAMIZE_ORIENT_H
#define SIAMIZE_ORIENT_H

#include "siam.h"
#include <array>
#include <cstdint>

namespace siam {

/**
 * @brief Compute the permutation + flip that brings input voxel axes to RAS
 *
 * For each input voxel axis \a i in {0, 1, 2}:
 *
 *   - `dst[i]` is the canonical world axis (R=0, A=1, S=2) this input
 *     axis maps to most strongly (i.e. the column of the 3x3 rotation
 *     submatrix with the largest absolute value in row \a i).
 *   - `sgn[i]` is +1 or -1, capturing whether the mapping is in the
 *     positive or negative canonical direction.
 *
 * Together, (dst, sgn) describe a 24-element discrete group element
 * (the proper-and-improper rotations of the cube).
 *
 * @param  affine  the 4x4 row-major affine
 * @param  dst     output: per-input-axis canonical-axis index
 * @param  sgn     output: per-input-axis sign in {-1, +1}
 */
void axes_to_canonical(const std::array<float, 16>& affine,
                       std::array<int, 3>& dst,
                       std::array<int, 3>& sgn);

/**
 * @brief Reorient an input voxel buffer into canonical (Z, Y, X) layout
 *
 * Applies the (dst, sgn) returned by axes_to_canonical to copy \a src
 * (laid out X-fastest in NIfTI / column-major-MATLAB order) into
 * \a out as a contiguous (Z, Y, X) float32 volume, casting elements
 * to float32 as needed. The output volume is resized via Volume::resize
 * before writing.
 *
 * @tparam SrcT  source element type
 * @param  src   pointer to X*Y*Z values, X-fastest in memory
 * @param  X,Y,Z input shape in input-axis order (NOT canonical)
 * @param  dst   per-input-axis canonical-axis index
 * @param  sgn   per-input-axis sign
 * @param  out   output volume (resized + filled)
 */
template <typename SrcT>
void copy_reorient_to_canonical(const SrcT* src,
                                int64_t X, int64_t Y, int64_t Z,
                                const std::array<int, 3>& dst,
                                const std::array<int, 3>& sgn,
                                Volume& out);

/**
 * @brief Inverse of copy_reorient_to_canonical
 *
 * Takes data already in canonical (Z, Y, X) order plus the original
 * input shape and the same (dst, sgn) that brought it canonical, and
 * writes the values back into input-axis order with X fastest.
 *
 * @tparam SrcT       canonical buffer element type
 * @tparam DstT       output buffer element type (typically same as SrcT)
 * @param  canon_zyx  input canonical (Z, Y, X) buffer
 * @param  canonZ,canonY,canonX   canonical extents
 * @param  X,Y,Z      original input shape
 * @param  dst,sgn    from axes_to_canonical
 * @param  out_xyz    output buffer, must have at least X*Y*Z elements
 */
template <typename SrcT, typename DstT>
void copy_reorient_from_canonical(const SrcT* canon_zyx,
                                  int64_t canonZ, int64_t canonY, int64_t canonX,
                                  int64_t X, int64_t Y, int64_t Z,
                                  const std::array<int, 3>& dst,
                                  const std::array<int, 3>& sgn,
                                  DstT* out_xyz);

/**
 * @brief Build the affine describing the canonicalized frame
 *
 * Given the input file's affine and the (dst, sgn) that bring its
 * voxel axes to RAS, returns the affine of the data after
 * canonicalization. The rotation/scale columns are permuted and
 * sign-flipped accordingly, and the translation is adjusted so the
 * canonical frame still places the corner voxel at the same world
 * coordinate as before.
 *
 * @param  input_affine     the input file's 4x4 row-major affine
 * @param  dst              per-input-axis canonical-axis index
 * @param  sgn              per-input-axis sign
 * @param  input_shape_xyz  the input shape (X, Y, Z) in input-axis order
 * @return                  the affine of the canonicalized data
 */
std::array<float, 16> canonicalize_affine(const std::array<float, 16>& input_affine,
        const std::array<int, 3>& dst,
        const std::array<int, 3>& sgn,
        const std::array<int64_t, 3>& input_shape_xyz);

}  // namespace siam

#endif  // SIAMIZE_ORIENT_H
