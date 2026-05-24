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
\file    orient.cpp
\brief   RAS canonicalization: axis permutation, copy reorient, canonical affine

Implements the per-axis permutation+flip computation that brings an
arbitrary input voxel grid to closest canonical RAS (matching
nibabel's `as_closest_canonical`), and the buffer-copy helpers that
apply / undo the reorientation. Template instantiations cover all
the source element types siamize loads from NIfTI-1 / JNIfTI:
int8 / uint8 / int16 / uint16 / int32 / uint32 / float32 / float64
into a canonical float32 Volume, and uint8/float32 round-trip back
to the on-disk axis order for the writers.
*******************************************************************************/

#include "orient.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace siam {

/*******************************************************************************/
/*! \fn    void axes_to_canonical(const std::array<float, 16>& affine,
                                  std::array<int, 3>& dst,
                                  std::array<int, 3>& sgn)
    \brief Compute the input-axis to canonical-axis permutation + flip

    For each input voxel axis \a i in {0, 1, 2}, looks at how column
    \a i of the 3x3 rotation submatrix projects into the world (R, A, S)
    axes and picks the world axis with the largest |projection| as
    `dst[i]`. The sign of that projection becomes `sgn[i]`. The result
    is one of the 24 proper-or-improper rotations of the cube -- the
    same group nibabel's `aff2axcodes` selects from.

    Greedy assignment with a `used[]` mask ensures dst[] is a valid
    permutation even when two input axes have very similar projections
    onto the same world axis (an extremely oblique affine).

    \param  affine  row-major 4x4 input affine
    \param  dst     output: per-input-axis canonical-axis index
    \param  sgn     output: per-input-axis sign (+1 or -1)
*/
void axes_to_canonical(const std::array<float, 16>& affine,
                       std::array<int, 3>& dst,
                       std::array<int, 3>& sgn) {
    // affine is row-major 4x4. The columns 0..2 are how each input voxel
    // axis projects into world (R, A, S).
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

/*******************************************************************************/
/*! \fn    template <typename SrcT>
           void copy_reorient_to_canonical(const SrcT* src,
                                           int64_t X, int64_t Y, int64_t Z,
                                           const std::array<int, 3>& dst,
                                           const std::array<int, 3>& sgn,
                                           Volume& out)
    \brief Apply (dst, sgn) to copy an X-fastest input buffer into canonical RAS

    Walks the input voxel grid in (Z, Y, X) order, computes the
    canonical (R, A, S) indices via the (dst, sgn) permutation, and
    stores the cast-to-float value into the canonical Volume at the
    contiguous (Z, Y, X) X-fastest position. The output Volume is
    resized inside this routine.

    \tparam SrcT    source element type
    \param  src     pointer to X*Y*Z values, X-fastest in memory
    \param  X,Y,Z   input shape in input-axis order
    \param  dst,sgn from axes_to_canonical
    \param  out     output Volume (resized + filled)
*/
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

/*******************************************************************************/
/*! \fn    template <typename SrcT, typename DstT>
           void copy_reorient_from_canonical(const SrcT* canon_zyx,
                                             int64_t canonZ, int64_t canonY, int64_t canonX,
                                             int64_t X, int64_t Y, int64_t Z,
                                             const std::array<int, 3>& dst,
                                             const std::array<int, 3>& sgn,
                                             DstT* out_xyz)
    \brief Inverse reorient: write canonical data back to input-axis order

    Used by the writers to restore the on-disk axis order before
    serialization. Walks the input voxel grid in (Z, Y, X) order,
    computes the matching canonical position, and pulls the value
    from the canonical buffer into the X-fastest output buffer.

    \tparam SrcT      canonical buffer element type
    \tparam DstT      output buffer element type
    \param  canon_zyx canonical (Z, Y, X) input buffer
    \param  canonZ,canonY,canonX  canonical extents
    \param  X,Y,Z     original input shape
    \param  dst,sgn   from axes_to_canonical
    \param  out_xyz   output buffer (X*Y*Z elements, X-fastest)
*/
template <typename SrcT, typename DstT>
void copy_reorient_from_canonical(const SrcT* canon_zyx,
                                  int64_t canonZ, int64_t canonY, int64_t canonX,
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

                int64_t in_idx = cidx[2] * (canonY * canonX) + cidx[1] * canonX + cidx[0];
                out_xyz[iz * X * Y + iy * X + ix] = static_cast<DstT>(canon_zyx[in_idx]);
            }
        }
    }
}

/*******************************************************************************/
/*! \fn    std::array<float, 16> canonicalize_affine(
              const std::array<float, 16>& A,
              const std::array<int, 3>& dst,
              const std::array<int, 3>& sgn,
              const std::array<int64_t, 3>& shape_xyz)
    \brief Build the affine of the canonicalized data

    Permutes the input affine's rotation/scale columns by (dst, sgn).
    When a sign flip is applied to input axis \a i, the origin column
    of the affine has to be shifted by `(shape[i] - 1) * A[:, i]`
    so the canonical frame still places the corner voxel at the same
    world coordinate.

    \param  A          row-major 4x4 input affine
    \param  dst,sgn    from axes_to_canonical
    \param  shape_xyz  input shape (X, Y, Z) in input-axis order
    \return            row-major 4x4 canonical affine
*/
std::array<float, 16> canonicalize_affine(const std::array<float, 16>& A,
        const std::array<int, 3>& dst,
        const std::array<int, 3>& sgn,
        const std::array<int64_t, 3>& shape_xyz) {
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
                canon[r * 4 + 3] += A[r * 4 + i] * static_cast<float>(shape_xyz[i] - 1);
            }
        }
    }

    for (int r = 0; r < 3; ++r) {
        canon[r * 4 + 3] += A[r * 4 + 3];
    }

    return canon;
}

/* ============================================================================ */
/*  Explicit template instantiations for the dtypes nifti_io / jnifti_io / MEX */
/*  consume on the read path, and for the round-trip back to disk on the      */
/*  write path. Keeping these in the same TU as the templates means the       */
/*  callers do not need to see the template definitions in a header.          */
/* ============================================================================ */
template void copy_reorient_to_canonical(const int8_t*,    int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, Volume&);
template void copy_reorient_to_canonical(const uint8_t*,   int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, Volume&);
template void copy_reorient_to_canonical(const int16_t*,   int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, Volume&);
template void copy_reorient_to_canonical(const uint16_t*,  int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, Volume&);
template void copy_reorient_to_canonical(const int32_t*,   int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, Volume&);
template void copy_reorient_to_canonical(const uint32_t*,  int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, Volume&);
template void copy_reorient_to_canonical(const float*,     int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, Volume&);
template void copy_reorient_to_canonical(const double*,    int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, Volume&);

template void copy_reorient_from_canonical<uint8_t, uint8_t>(const uint8_t*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, uint8_t*);
template void copy_reorient_from_canonical<float,   float  >(const float*,   int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, const std::array<int, 3>&, const std::array<int, 3>&, float*);

}  // namespace siam
