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
\file    siam.h

@brief   Core siamize types: Volume, LogitsVolume, NiftiImage

This header defines the data types every other siamize translation
unit consumes:

  - Volume:       a single-channel float32 volume in canonical
                  (Z, Y, X) RAS layout, X-fastest in memory.
  - LogitsVolume: a multi-channel float32 accumulator with the same
                  spatial layout but channel-slowest, used to hold
                  per-fold logits during sliding-window inference.
  - NiftiImage:   bundle of (canonical Volume + original/canonical
                  affines + reorient metadata + on-disk shape and
                  dtype) so the writers can restore the input axis
                  order exactly.

The (Z, Y, X) canonical layout matches nnUNet's runtime convention;
the writers (nifti_io / jnifti_io) flip back to NIfTI-native
(X, Y, Z) X-fastest order via the perm_canon_to_orig + flip_canon
metadata.
*******************************************************************************/

#ifndef SIAMIZE_SIAM_H
#define SIAMIZE_SIAM_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace siam {

/**
 * \struct Volume
 * \brief  Single-channel float32 volume in canonical (Z, Y, X) RAS layout
 *
 * In-memory layout is contiguous with X fastest-varying. The volume
 * owns its data via std::vector; resize() reallocates and zero-fills.
 */
struct Volume {
    std::vector<float> data;                /**< raw voxel data, length = prod(shape) */
    std::array<int64_t, 3> shape{0, 0, 0};  /**< (Z, Y, X) extents */

    /**
     * @brief Total number of voxels in the volume
     * @return Z*Y*X
     */
    int64_t numel() const {
        return shape[0] * shape[1] * shape[2];
    }

    /**
     * @brief Stride (in voxels) along axis \a d for the contiguous (Z, Y, X) layout
     * @param  d  axis index in {0=Z, 1=Y, 2=X}
     * @return    stride = Y*X if d==0, X if d==1, 1 if d==2
     */
    int64_t stride(int d) const {
        if (d == 0) {
            return shape[1] * shape[2];
        }

        if (d == 1) {
            return shape[2];
        }

        return 1;
    }

    /**
     * @brief Mutable accessor for voxel at canonical position (z, y, x)
     */
    float& at(int64_t z, int64_t y, int64_t x) {
        return data[z * stride(0) + y * stride(1) + x];
    }

    /**
     * @brief Read-only accessor for voxel at canonical position (z, y, x)
     */
    float  at(int64_t z, int64_t y, int64_t x) const {
        return data[z * stride(0) + y * stride(1) + x];
    }

    /**
     * @brief Reshape and zero-fill the volume
     * @param  Z, Y, X  canonical extents
     */
    void resize(int64_t Z, int64_t Y, int64_t X) {
        shape = {Z, Y, X};
        data.assign(static_cast<size_t>(Z) * Y * X, 0.0f);
    }
};

/**
 * \struct LogitsVolume
 * \brief  Multi-channel float32 accumulator, channel-slowest (C, Z, Y, X)
 *
 * Used to accumulate per-fold network outputs during sliding-window
 * inference before softmax / argmax. Channel-major layout means
 * channel-wise data lives in a contiguous slab, which makes
 * per-channel reorientation and softmax straightforward.
 */
struct LogitsVolume {
    std::vector<float> data;                /**< raw data, length = C * prod(spat) */
    int64_t C = 0;                          /**< number of channels */
    std::array<int64_t, 3> spat{0, 0, 0};   /**< spatial extents (Z, Y, X) */

    /**
     * @brief Total element count = C * Z * Y * X
     */
    int64_t numel() const {
        return C * spat[0] * spat[1] * spat[2];
    }

    /**
     * @brief Stride (in voxels) between consecutive channels
     * @return Z * Y * X
     */
    int64_t cstride() const {
        return spat[0] * spat[1] * spat[2];
    }

    /**
     * @brief Reshape and zero-fill the accumulator
     * @param  Cn      number of channels
     * @param  Z, Y, X canonical spatial extents
     */
    void resize(int64_t Cn, int64_t Z, int64_t Y, int64_t X) {
        C = Cn;
        spat = {Z, Y, X};
        data.assign(static_cast<size_t>(numel()), 0.0f);
    }

    /**
     * @brief Mutable pointer to the start of channel \a c
     */
    float* channel_ptr(int64_t c) {
        return data.data() + c * cstride();
    }

    /**
     * @brief Read-only pointer to the start of channel \a c
     */
    const float* channel_ptr(int64_t c) const {
        return data.data() + c * cstride();
    }
};

/**
 * \struct NiftiImage
 * \brief  A NIfTI volume + affine + canonical reorientation metadata
 *
 * Carries everything required to (a) load a NIfTI / JNIfTI file and
 * reorient it to canonical (Z, Y, X) RAS for processing, then (b)
 * restore the original axis order and write a NIfTI / JNIfTI that
 * overlays the input exactly.
 *
 * The `perm_canon_to_orig` + `flip_canon` pair is the inverse of the
 * axes_to_canonical permutation/flip: `perm_canon_to_orig[i]` names
 * the canonical axis that ends up at original-storage axis \a i, and
 * `flip_canon[i]` is \pm 1 telling whether canonical axis \a i was
 * flipped during canonicalization.
 */
struct NiftiImage {
    Volume volume;                                  /**< RAS-canonical (Z, Y, X) float32 data */
    std::array<float, 16> affine_orig{};            /**< input file affine (4x4 row-major) */
    std::array<float, 16> affine_canon{};           /**< affine after to-canonical reorient */
    std::array<float, 3> zooms_canon{};             /**< voxel sizes (X, Y, Z) in canonical */
    std::array<int, 3> perm_canon_to_orig{0, 1, 2}; /**< canonical-axis-index per original axis */
    std::array<int, 3> flip_canon{1, 1, 1};         /**< per-canonical-axis flip (+1 or -1) */
    std::array<int64_t, 3> shape_orig{0, 0, 0};     /**< (X, Y, Z) in original file order */
    int datatype_orig = 0;                          /**< NIfTI datatype code of the input file */
};

}  // namespace siam

#endif  // SIAMIZE_SIAM_H
