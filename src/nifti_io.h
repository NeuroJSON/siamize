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
**  \li \c (\b NIfTI1) NIfTI-1 Data Format,
**         <a href="https://nifti.nimh.nih.gov/nifti-1">nifti.nimh.nih.gov/nifti-1</a>
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    nifti_io.h

@brief   NIfTI-1 (.nii / .nii.gz) reader/writer public interface

Declares the entry points used by siamize to read and write NIfTI-1
volumes. The reader normalizes input to siamize's canonical (Z, Y, X)
RAS float32 layout; the writers reorient back to the original voxel
axis order before serialization so output spatial metadata is
round-trip compatible with the input.

For the JNIfTI (.jnii / .bnii) container variants, see jnifti_io.h.
*******************************************************************************/

#ifndef SIAMIZE_NIFTI_IO_H
#define SIAMIZE_NIFTI_IO_H

#include "siam.h"
#include <string>

namespace siam {

/**
 * @brief Load a NIfTI-1 file and convert to canonical RAS (Z, Y, X) float32
 *
 * Reads `.nii` or `.nii.gz` (gzip-detected by magic bytes 1F 8B and
 * inflated via zmat/miniz), parses the 348-byte NIfTI-1 header,
 * recovers the affine from sform when set otherwise from qform, then
 * uses axes_to_canonical + copy_reorient_to_canonical to produce a
 * (Z, Y, X) RAS volume regardless of the on-disk axis order.
 *
 * @param  path  source file path (`.nii` or `.nii.gz`)
 * @return       canonical NiftiImage (volume + affine + permutation metadata)
 */
NiftiImage load_nifti_ras(const std::string& path);

/**
 * @brief Save a 3D uint8 labelmap as a NIfTI-1 file
 *
 * Reorients the canonical (Z, Y, X) labelmap back to the input file's
 * native voxel axis order via copy_reorient_from_canonical, then
 * emits a NIfTI-1 file (auto-gzipped if the path ends in `.gz`).
 *
 * @param  path        destination file path (`.nii` or `.nii.gz`)
 * @param  src         the input NiftiImage that supplied the affine and axis order
 * @param  labels_zyx  canonical-axis (Z, Y, X) labelmap, length cZ*cY*cX bytes
 */
void save_nifti_labels(const std::string& path,
                       const NiftiImage& src,
                       const uint8_t* labels_zyx);

/**
 * @brief Save a 4D float32 tissue probability map (TPM) as NIfTI-1
 *
 * Reorients each of the per-channel volumes back into the input
 * file's voxel axis order before emitting a 4D NIfTI-1 file with
 * `dim[1..3] = (X, Y, Z)` and `dim[4] = num_classes`. The input
 * layout is channel-slowest (C, Z, Y, X) in canonical orientation.
 *
 * @param  path             destination file path
 * @param  src              the input NiftiImage whose header is reused
 * @param  tpm_canon_czyx   channel-slowest TPM in canonical orientation
 * @param  num_classes      number of channels (NIfTI dim[4])
 */
void save_nifti_tpm(const std::string& path,
                    const NiftiImage& src,
                    const float* tpm_canon_czyx,
                    int64_t num_classes);

}  // namespace siam

#endif  // SIAMIZE_NIFTI_IO_H
