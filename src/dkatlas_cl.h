/***************************************************************************//**
\file    dkatlas_cl.h
@brief   The DKatlas network (dkatlas.cpp) on an OpenCL device
*******************************************************************************/

#ifndef SIAMIZE_DKATLAS_CL_H
#define SIAMIZE_DKATLAS_CL_H

#include <cstdint>
#include <string>

namespace siam {

/**
 * \struct DKNetWeights
 * \brief  The DKatlas weights, as dkatlas.cpp holds them (not owned)
 */
struct DKNetWeights {
    const float* conv[13];       /**< per layer [tap 27][cin][24], cin 1 then 24 */
    const float (*scale)[24];    /**< per layer, the channel affine after the norm */
    const float (*shift)[24];
    const float* fk;             /**< [24][104] */
    const float* fb;             /**< [104] */
    const int* dil;              /**< per layer, the dilation */
};

/**
 * @brief  the network on the 256^3 conformed input, on an OpenCL device
 *
 * @param  in      the conformed, normalized input, 256^3 (i slowest)
 * @param  W       the weights
 * @param  device  1-based OpenCL device in clinfo order (siamize -G), or 0 for the first GPU
 * @param  lab     256^3 output: the class (0..103) per voxel
 * @param  info    the device's name on success, else why it could not run
 * @return false when it could not run (the caller then uses the CPU)
 */
bool dk_run_cl(const float* in, const DKNetWeights& W, int device, uint8_t* lab, std::string& info);

}  // namespace siam

#endif  // SIAMIZE_DKATLAS_CL_H
