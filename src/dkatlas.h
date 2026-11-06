/***************************************************************************//**
\file    dkatlas.h
@brief   Desikan-Killiany cortical parcellation of SIAM's gray matter

Runs brainchop's DKatlas network (MeshNet: 13 dilated 3x3x3 convolutions,
24 channels, 104 classes; Masoud & Plis, https://github.com/neuroneural/brainchop)
on the input T1, conformed to 256^3 at 1 mm as brainchop does, and transfers
its 68 cortical labels onto SIAM's cerebral GM: a GM voxel inside a DK cortical
label keeps it; the rest of the GM takes the nearest one, grown through the GM
only (so a label cannot jump a sulcus) and only within its own hemisphere.

The network is small (198 k parameters, 0.79 MB): its weights are compiled in
(src/dkatlas_weights.inc; MIT, src/dkatlas_LICENSE.txt) and it runs natively on
the CPU here, not through the ORT / MNN engine (MNN's native Conv3D has no
dilation).
*******************************************************************************/

#ifndef SIAMIZE_DKATLAS_H
#define SIAMIZE_DKATLAS_H

#include "dkatlas_labels.h"
#include "siam.h"

#include <cstdint>
#include <string>
#include <vector>

namespace siam {

/**
 * \struct DKStats
 * \brief  How the parcellation went (for the log)
 */
struct DKStats {
    int64_t gm = 0;        /**< SIAM GM voxels */
    int64_t direct = 0;    /**< labelled by the DK cortical label at the voxel */
    int64_t grown = 0;     /**< labelled by growth through the GM */
    int64_t unreached = 0; /**< GM voxels left unlabelled (no same-side label connected) */
    int classes = 0;       /**< DK classes the network produced */
    double seconds_net = 0, seconds_transfer = 0;
};

/**
 * @brief  DK parcellation of SIAM's cerebral GM
 *
 * @param  t1          the input image (canonical; its raw intensities)
 * @param  grid        the header of the output grid (the labels' image)
 * @param  labels_zyx  SIAM labels on `grid`, canonical (Z, Y, X)
 * @param  gm_label    SIAM's cerebral GM label (1 for SIAM v0.3's 18 classes)
 * @param  weights     a DKatlas weights file (brainchop's model.bin, raw fp32), or
 *                     empty for the compiled-in copy
 * @param  st          filled with statistics
 * @return canonical (Z, Y, X) labels on `grid`: the DK index (1..68) on GM, else 0
 */
std::vector<uint8_t> dk_parcellate(const NiftiImage& t1, const NiftiImage& grid,
                                   const uint8_t* labels_zyx, uint8_t gm_label,
                                   const std::string& weights, DKStats& st);

}  // namespace siam

#endif  // SIAMIZE_DKATLAS_H
