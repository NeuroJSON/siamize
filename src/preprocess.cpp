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
\file    preprocess.cpp
\brief   Pre-network preprocessing primitives (crop, z-score, resample)

Implements the volume-level preprocessing that mirrors the SIAM
reference pipeline (py/siam_ref.py / tools/onnx_export/siam_ort.py):

  - crop_to_nonzero:    nnUNet-style tight bounding box around the
                        nonzero region. We omit the
                        `binary_fill_holes` step found in the
                        reference; on realistic head MRI the head is
                        convex enough that the bbox is identical
                        with or without it.

  - zscore_inplace:     whole-volume z-score normalization,
                        `(x - mean) / max(std, 1e-8)`. No masking.

  - resample_trilinear: first-order resampler, matching skimage's
                        `resize(order=1, mode='edge', anti_aliasing=False)`
                        using the
                        `input_coord = (i + 0.5) * scale - 0.5` mapping
                        with edge clamping at the borders. Used as a
                        fast path.

  - resample_cubic:     three-pass separable Catmull-Rom cubic
                        Hermite resampler, matching skimage's
                        `resize(order=3, mode='edge', anti_aliasing=False)`
                        to ~99.9 % argmax agreement on real images.
*******************************************************************************/

#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace siam {

/*******************************************************************************/
/*! \fn    CropResult crop_to_nonzero(const Volume& vol)
    \brief Crop a volume to the bounding box of its nonzero voxels

    Single triple-nested pass over the volume that tracks the
    per-axis min/max indices where a voxel exceeds zero. When no
    nonzero voxels exist, returns the full volume with an
    [0, shape) bbox (degenerate but well-defined).

    \param  vol  input canonical volume
    \return      cropped volume + axis-aligned bbox
*/
CropResult crop_to_nonzero(const Volume& vol) {
    int64_t Z = vol.shape[0], Y = vol.shape[1], X = vol.shape[2];
    int64_t zlo = Z, zhi = -1, ylo = Y, yhi = -1, xlo = X, xhi = -1;

    for (int64_t z = 0; z < Z; ++z) {
        for (int64_t y = 0; y < Y; ++y) {
            for (int64_t x = 0; x < X; ++x) {
                if (vol.at(z, y, x) != 0.0f) {
                    if (z < zlo) {
                        zlo = z;
                    }

                    if (z > zhi) {
                        zhi = z;
                    }

                    if (y < ylo) {
                        ylo = y;
                    }

                    if (y > yhi) {
                        yhi = y;
                    }

                    if (x < xlo) {
                        xlo = x;
                    }

                    if (x > xhi) {
                        xhi = x;
                    }
                }
            }
        }
    }

    CropResult r;

    if (zhi < 0) {
        // No nonzero voxels: return whole volume (degenerate).
        r.cropped = vol;
        r.bbox = {{{0, Z}, {0, Y}, {0, X}}};
        return r;
    }

    r.bbox = {{{zlo, zhi + 1}, {ylo, yhi + 1}, {xlo, xhi + 1}}};
    int64_t cZ = zhi - zlo + 1, cY = yhi - ylo + 1, cX = xhi - xlo + 1;
    r.cropped.resize(cZ, cY, cX);

    for (int64_t z = 0; z < cZ; ++z) {
        for (int64_t y = 0; y < cY; ++y) {
            const float* src = &vol.data[(z + zlo) * vol.stride(0) + (y + ylo) * vol.stride(1) + xlo];
            float* dst = &r.cropped.data[z * r.cropped.stride(0) + y * r.cropped.stride(1)];
            std::copy_n(src, cX, dst);
        }
    }

    return r;
}

/*******************************************************************************/
/*! \fn    void zscore_inplace(Volume& vol)
    \brief Whole-volume z-score normalization, in place

    Two passes over the data: pass 1 accumulates sum and sum-of-squares
    (in double precision to keep the variance stable for large volumes);
    pass 2 applies `(x - mean) / max(std, 1e-8)`. No masking and no
    per-channel split because the input is single-channel.

    \param  vol  volume modified in place
*/
void zscore_inplace(Volume& vol) {
    double sum = 0.0;
    double sum_sq = 0.0;
    int64_t n = vol.numel();

    for (int64_t i = 0; i < n; ++i) {
        double v = vol.data[i];
        sum += v;
        sum_sq += v * v;
    }

    double mean = sum / static_cast<double>(n);
    double var = sum_sq / static_cast<double>(n) - mean * mean;

    if (var < 0.0) {
        var = 0.0;
    }

    double std = std::sqrt(var);

    if (std < 1e-8) {
        std = 1e-8;
    }

    float fm = static_cast<float>(mean);
    float fs = static_cast<float>(std);

    for (int64_t i = 0; i < n; ++i) {
        vol.data[i] = (vol.data[i] - fm) / fs;
    }
}

/*******************************************************************************/
/*! \fn    std::array<int64_t, 3> compute_new_shape(
              std::array<int64_t, 3> old_shape,
              std::array<float, 3> old_spacing,
              std::array<float, 3> new_spacing)
    \brief Round-to-nearest shape after a spacing change

    Per-axis: `new = round(old * old_spacing / new_spacing)`. Matches
    nnUNet's preprocessing rule.

    \param  old_shape    current shape in voxels
    \param  old_spacing  current voxel spacing (mm) per axis
    \param  new_spacing  target voxel spacing (mm) per axis
    \return              the new shape after resampling
*/
std::array<int64_t, 3> compute_new_shape(std::array<int64_t, 3> old_shape,
        std::array<float, 3> old_spacing,
        std::array<float, 3> new_spacing) {
    std::array<int64_t, 3> out{};

    for (int i = 0; i < 3; ++i) {
        double r = static_cast<double>(old_spacing[i]) / new_spacing[i];
        out[i] = static_cast<int64_t>(std::llround(r * old_shape[i]));
    }

    return out;
}

/* ============================================================================ */
/*                                 Resampling                                  */
/* ============================================================================ */

namespace {

/**
 * @brief Clamp a sample index to `[0, n)` for edge-mode boundary handling
 */
inline int64_t clamp_idx(int64_t i, int64_t n) {
    if (i < 0) {
        return 0;
    }

    if (i >= n) {
        return n - 1;
    }

    return i;
}
}  // namespace

/*******************************************************************************/
/*! \fn    Volume resample_trilinear(const Volume& src,
                                     int64_t outZ, int64_t outY, int64_t outX)
    \brief Trilinear resample with edge-clamped boundary handling

    Pre-computes per-axis sample indices and weights once, then runs
    a triple-nested loop (parallelized over Z when the workload is
    non-trivial). Coordinate mapping is
    `t = (k + 0.5) * (in / out) - 0.5` to match skimage's half-pixel
    center convention with `mode='edge'`.

    \param  src             source volume (any shape)
    \param  outZ,outY,outX  target shape
    \return                 new volume of shape (outZ, outY, outX)
*/
Volume resample_trilinear(const Volume& src,
                          int64_t outZ, int64_t outY, int64_t outX) {
    Volume out;
    out.resize(outZ, outY, outX);

    if (outZ <= 0 || outY <= 0 || outX <= 0) {
        return out;
    }

    const int64_t inZ = src.shape[0], inY = src.shape[1], inX = src.shape[2];

    const double sZ = static_cast<double>(inZ) / outZ;
    const double sY = static_cast<double>(inY) / outY;
    const double sX = static_cast<double>(inX) / outX;

    // Precompute per-axis sample indices and weights.
    std::vector<int64_t> z0(outZ), z1(outZ);
    std::vector<float> wz(outZ);

    for (int64_t k = 0; k < outZ; ++k) {
        double t = (k + 0.5) * sZ - 0.5;
        int64_t i0 = static_cast<int64_t>(std::floor(t));
        double a = t - i0;
        z0[k] = clamp_idx(i0,     inZ);
        z1[k] = clamp_idx(i0 + 1, inZ);
        wz[k] = static_cast<float>(a);
    }

    std::vector<int64_t> y0(outY), y1(outY);
    std::vector<float> wy(outY);

    for (int64_t k = 0; k < outY; ++k) {
        double t = (k + 0.5) * sY - 0.5;
        int64_t i0 = static_cast<int64_t>(std::floor(t));
        double a = t - i0;
        y0[k] = clamp_idx(i0,     inY);
        y1[k] = clamp_idx(i0 + 1, inY);
        wy[k] = static_cast<float>(a);
    }

    std::vector<int64_t> x0(outX), x1(outX);
    std::vector<float> wx(outX);

    for (int64_t k = 0; k < outX; ++k) {
        double t = (k + 0.5) * sX - 0.5;
        int64_t i0 = static_cast<int64_t>(std::floor(t));
        double a = t - i0;
        x0[k] = clamp_idx(i0,     inX);
        x1[k] = clamp_idx(i0 + 1, inX);
        wx[k] = static_cast<float>(a);
    }

    const int64_t in_sz = src.stride(0);
    const int64_t in_sy = src.stride(1);
    const int64_t out_sz = out.stride(0);
    const int64_t out_sy = out.stride(1);

    #pragma omp parallel for if(outZ * outY > 4096) schedule(static)

    for (int64_t kz = 0; kz < outZ; ++kz) {
        const float wzv = wz[kz], wzv1 = 1.0f - wzv;
        const float* sz0 = &src.data[z0[kz] * in_sz];
        const float* sz1 = &src.data[z1[kz] * in_sz];

        for (int64_t ky = 0; ky < outY; ++ky) {
            const float wyv = wy[ky], wyv1 = 1.0f - wyv;
            const float* sz0y0 = sz0 + y0[ky] * in_sy;
            const float* sz0y1 = sz0 + y1[ky] * in_sy;
            const float* sz1y0 = sz1 + y0[ky] * in_sy;
            const float* sz1y1 = sz1 + y1[ky] * in_sy;
            float* dst = &out.data[kz * out_sz + ky * out_sy];

            for (int64_t kx = 0; kx < outX; ++kx) {
                const float wxv = wx[kx], wxv1 = 1.0f - wxv;
                int64_t ix0 = x0[kx], ix1 = x1[kx];
                float v000 = sz0y0[ix0], v001 = sz0y0[ix1];
                float v010 = sz0y1[ix0], v011 = sz0y1[ix1];
                float v100 = sz1y0[ix0], v101 = sz1y0[ix1];
                float v110 = sz1y1[ix0], v111 = sz1y1[ix1];
                float c00 = v000 * wxv1 + v001 * wxv;
                float c01 = v010 * wxv1 + v011 * wxv;
                float c10 = v100 * wxv1 + v101 * wxv;
                float c11 = v110 * wxv1 + v111 * wxv;
                float c0  = c00  * wyv1 + c01  * wyv;
                float c1  = c10  * wyv1 + c11  * wyv;
                dst[kx]   = c0   * wzv1 + c1   * wzv;
            }
        }
    }

    return out;
}

/* ============================================================================ */
/*                          Cubic Catmull-Rom resample                         */
/* ============================================================================ */

namespace {

/**
 * @brief One-dimensional Catmull-Rom cubic Hermite interpolation
 *
 * Standard Catmull-Rom basis (tension = 0); evaluates the cubic
 * Hermite polynomial that interpolates four equispaced samples
 * v(-1), v(0), v(+1), v(+2) at position `t` in [0, 1].
 *
 * @param  v_m1, v_0, v_p1, v_p2  four equispaced samples
 * @param  t   fractional offset in [0, 1] from v_0 toward v_p1
 * @return     interpolated value
 */
inline float catmull_rom(float v_m1, float v_0, float v_p1, float v_p2, float t) {
    // Standard Catmull-Rom (tension = 0) cubic Hermite.
    const float t2 = t * t;
    const float t3 = t2 * t;
    return 0.5f * (
               (2.0f * v_0) +
               (-v_m1 + v_p1) * t +
               (2.0f * v_m1 - 5.0f * v_0 + 4.0f * v_p1 - v_p2) * t2 +
               (-v_m1 + 3.0f * v_0 - 3.0f * v_p1 + v_p2) * t3
           );
}

/**
 * \struct CubicCoeffs
 * \brief  Per-output-pixel sample indices + fractional offset for cubic resample
 *
 * Used by resample_cubic to precompute, for each output pixel along
 * one axis, the four input sample indices (with edge clamping
 * already applied) and the [0, 1) fractional offset that feed
 * catmull_rom().
 */
struct CubicCoeffs {
    std::vector<int64_t> i_m1;   /**< clamped index of v(-1) for each output pixel */
    std::vector<int64_t> i_0;    /**< clamped index of v(0) */
    std::vector<int64_t> i_p1;   /**< clamped index of v(+1) */
    std::vector<int64_t> i_p2;   /**< clamped index of v(+2) */
    std::vector<float> t;        /**< fractional offset in [0, 1) */
};

/**
 * @brief Pre-compute per-output-pixel Catmull-Rom coefficients along one axis
 *
 * @param  out    target axis length
 * @param  in_n   input axis length
 * @return        4 index vectors + a fractional-offset vector
 */
CubicCoeffs make_cubic_coeffs(int64_t out, int64_t in_n) {
    CubicCoeffs c;
    c.i_m1.resize(out);
    c.i_0.resize(out);
    c.i_p1.resize(out);
    c.i_p2.resize(out);
    c.t.resize(out);
    const double s = static_cast<double>(in_n) / out;

    for (int64_t k = 0; k < out; ++k) {
        double pos = (k + 0.5) * s - 0.5;
        int64_t i0 = static_cast<int64_t>(std::floor(pos));
        double t = pos - i0;
        c.i_m1[k] = clamp_idx(i0 - 1, in_n);
        c.i_0[k]  = clamp_idx(i0,     in_n);
        c.i_p1[k] = clamp_idx(i0 + 1, in_n);
        c.i_p2[k] = clamp_idx(i0 + 2, in_n);
        c.t[k]    = static_cast<float>(t);
    }

    return c;
}

}  // namespace

/*******************************************************************************/
/*! \fn    Volume resample_cubic(const Volume& src,
                                 int64_t outZ, int64_t outY, int64_t outX)
    \brief Three-pass separable Catmull-Rom cubic resample

    Three-pass separable design (along X, Y, Z) so each pass is a
    simple 4-tap stencil. Two intermediate buffers hold the partial
    products; the X-result buffer is freed before the Z pass to keep
    peak memory bounded.

    \param  src             source volume (any shape)
    \param  outZ,outY,outX  target shape
    \return                 new volume of shape (outZ, outY, outX)
*/
Volume resample_cubic(const Volume& src,
                      int64_t outZ, int64_t outY, int64_t outX) {
    Volume out;
    out.resize(outZ, outY, outX);

    if (outZ <= 0 || outY <= 0 || outX <= 0) {
        return out;
    }

    const int64_t inZ = src.shape[0], inY = src.shape[1], inX = src.shape[2];

    auto cz = make_cubic_coeffs(outZ, inZ);
    auto cy = make_cubic_coeffs(outY, inY);
    auto cx = make_cubic_coeffs(outX, inX);

    // Strategy: two-pass separable. Pass 1: cubic along X for each (z, y) in input
    // → tmp1 shape (inZ, inY, outX). Pass 2: cubic along Y for each (z, x) → tmp2
    // shape (inZ, outY, outX). Pass 3: cubic along Z for each (y, x) → out.
    //
    // This is 3 * inZ*inY*outX + ... etc. operations, roughly comparable to
    // trilinear but with 4-tap stencils.

    // Pass 1: X
    std::vector<float> tmp1(static_cast<size_t>(inZ) * inY * outX);
    #pragma omp parallel for if(inZ * inY > 4096) schedule(static)

    for (int64_t z = 0; z < inZ; ++z) {
        const float* zp = &src.data[z * src.stride(0)];

        for (int64_t y = 0; y < inY; ++y) {
            const float* row = zp + y * src.stride(1);
            float* dst = &tmp1[z * inY * outX + y * outX];

            for (int64_t kx = 0; kx < outX; ++kx) {
                dst[kx] = catmull_rom(row[cx.i_m1[kx]], row[cx.i_0[kx]],
                                      row[cx.i_p1[kx]], row[cx.i_p2[kx]],
                                      cx.t[kx]);
            }
        }
    }

    // Pass 2: Y
    std::vector<float> tmp2(static_cast<size_t>(inZ) * outY * outX);
    #pragma omp parallel for if(inZ * outX > 4096) schedule(static)

    for (int64_t z = 0; z < inZ; ++z) {
        const float* zp = &tmp1[z * inY * outX];

        for (int64_t ky = 0; ky < outY; ++ky) {
            const float* rm1 = zp + cy.i_m1[ky] * outX;
            const float* r0  = zp + cy.i_0[ky]  * outX;
            const float* rp1 = zp + cy.i_p1[ky] * outX;
            const float* rp2 = zp + cy.i_p2[ky] * outX;
            float* dst = &tmp2[z * outY * outX + ky * outX];
            float t = cy.t[ky];

            for (int64_t kx = 0; kx < outX; ++kx) {
                dst[kx] = catmull_rom(rm1[kx], r0[kx], rp1[kx], rp2[kx], t);
            }
        }
    }

    tmp1.clear();
    tmp1.shrink_to_fit();

    // Pass 3: Z
    #pragma omp parallel for if(outY * outX > 4096) schedule(static)

    for (int64_t kz = 0; kz < outZ; ++kz) {
        const float* zm1 = &tmp2[cz.i_m1[kz] * outY * outX];
        const float* z0  = &tmp2[cz.i_0[kz]  * outY * outX];
        const float* zp1 = &tmp2[cz.i_p1[kz] * outY * outX];
        const float* zp2 = &tmp2[cz.i_p2[kz] * outY * outX];
        float t = cz.t[kz];
        float* dst = &out.data[kz * out.stride(0)];

        for (int64_t i = 0; i < outY * outX; ++i) {
            dst[i] = catmull_rom(zm1[i], z0[i], zp1[i], zp2[i], t);
        }
    }

    return out;
}

}  // namespace siam
