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
**  \li \c (\b JNIfTI) The JNIfTI specification, Q. Fang, NeuroJSON project.
**         <a href="https://neurojson.org/jnifti">https://neurojson.org/jnifti</a>
**  \li \c (\b JData) The JData specification, Q. Fang, NeuroJSON project.
**         <a href="https://neurojson.org/jdata">https://neurojson.org/jdata</a>
**  \li \c (\b BJData) The Binary JData specification, Q. Fang, NeuroJSON project.
**         <a href="https://neurojson.org/bjdata">https://neurojson.org/bjdata</a>
**
**  \section slicense License
**          Apache License 2.0, see LICENSE for details
*******************************************************************************/

/***************************************************************************//**
\file    jnifti_io.h

@brief   JNIfTI (.jnii / .bnii) container I/O public interface

This header declares the entry points used by siamize to read and write
JNIfTI containers in place of (or alongside) classic NIfTI-1 files.
A JNIfTI file wraps NIfTI-1/2 metadata plus voxel data in either a
JData annotated JSON object (.jnii, text JSON) or a BJData binary
JSON object (.bnii); the voxel payload is zlib-compressed inside an
`_ArrayZipData_` field. See https://neurojson.org/jnifti for the spec.
*******************************************************************************/

#ifndef SIAMIZE_JNIFTI_IO_H
#define SIAMIZE_JNIFTI_IO_H

#include "siam.h"
#include <cstdint>
#include <string>

namespace siam {

/**
 * @brief Save a 3D uint8 labelmap as a JNIfTI container
 *
 * Reorients the canonical (Z, Y, X) labelmap back to the input file's
 * native voxel-axis order before serialization, zlib-compresses the
 * byte stream, and writes either a text-JSON (.jnii, base64-encoded
 * payload) or BJData binary (.bnii, raw-byte payload) container.
 *
 * @param  path        destination file path
 * @param  src         the input NiftiImage that supplied the affine and axis order
 * @param  labels_zyx  canonical-axis (Z, Y, X) labelmap, length cZ*cY*cX bytes
 * @param  format      "jnii" for text JSON or "bnii" for binary BJData
 */
void save_jnifti_labels(const std::string& path,
                        const NiftiImage& src,
                        const uint8_t* labels_zyx,
                        const std::string& format);

/**
 * @brief Save a 4D float32 tissue probability map (TPM) as JNIfTI
 *
 * Like save_jnifti_labels but for the per-class softmax output. The
 * source layout is channel-slowest (C, cZ, cY, cX) in canonical
 * orientation; each channel is independently reoriented to the input
 * file's voxel axis order before being interleaved into the (X, Y, Z, C)
 * NIfTI volume.
 *
 * @param  path             destination file path
 * @param  src              the input NiftiImage whose header is reused
 * @param  tpm_canon_czyx   channel-slowest TPM in canonical orientation
 * @param  num_classes      number of channels (NIfTI dim[4])
 * @param  format           "jnii" or "bnii"
 */
void save_jnifti_tpm(const std::string& path,
                     const NiftiImage& src,
                     const float* tpm_canon_czyx,
                     int64_t num_classes,
                     const std::string& format);

/**
 * @brief Load a JNIfTI file (.jnii or .bnii) into a canonical NiftiImage
 *
 * Parses either a JSON-text (.jnii) or BJData (.bnii) container,
 * decodes the `_ArrayZipData_` / `_ArrayData_` payload (zlib +
 * optionally base64) into a contiguous voxel buffer, then reorients
 * the volume into siamize's canonical (Z, Y, X) RAS orientation. The
 * returned NiftiImage is bit-compatible with what siam::load_nifti_ras()
 * produces so downstream code does not care which container fed it.
 *
 * @param  path  source file path; `.jnii` / `.bnii` inferred from extension
 * @return       canonical NiftiImage (volume + affine + permutation metadata)
 */
NiftiImage load_jnifti_ras(const std::string& path);

}  // namespace siam

#endif  // SIAMIZE_JNIFTI_IO_H
