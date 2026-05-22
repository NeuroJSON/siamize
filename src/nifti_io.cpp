// NIfTI I/O with RAS canonicalization. nibabel's `as_closest_canonical`
// equivalent: find the permutation + flip from input voxel axes (i, j, k) to
// canonical (R, A, S), apply to the data, remember the inverse so we can
// write labels back into the original grid.

#include "nifti_io.h"
#include "siam.h"

#include <nifti1_io.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace siam {

namespace {

// Pack a mat44 (column-major in nifti_clib) into row-major 4x4 floats.
std::array<float, 16> pack_mat44(const mat44& m) {
    std::array<float, 16> a{};

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            a[r * 4 + c] = m.m[r][c];
        }
    }

    return a;
}

// Return permutation + sign per input axis so that:
//   canon_axis = (input_axis goes to canon_axis dst[input_axis], with sign sgn[input_axis])
// canon order is (R, A, S) = (+X, +Y, +Z).
// dst[i] in {0,1,2} = the canonical axis that input axis i most closely matches.
// sgn[i] in {-1, +1} = sign of that mapping.
void axes_to_canonical(const mat44& affine,
                       std::array<int, 3>& dst,
                       std::array<int, 3>& sgn) {
    // Pull columns 0..2 of the affine (column vectors describe how each
    // voxel axis projects into world (R, A, S)).
    float col[3][3];

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            col[i][j] = affine.m[j][i];
        }
    }

    // For each input axis, find its dominant canonical world axis.
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

template <typename SrcT>
void copy_reorient_to_canonical(const SrcT* src,
                                int64_t X, int64_t Y, int64_t Z,
                                const std::array<int, 3>& dst,
                                const std::array<int, 3>& sgn,
                                Volume& out) {
    // src is contiguous in (X, Y, Z) with X fastest, as nifti returns.
    // After reorientation, we write into out which is (canonZ, canonY, canonX)
    // — i.e. canonical X is the FASTEST axis, canonical Z is slowest.
    std::array<int64_t, 3> in_shape{X, Y, Z};
    std::array<int64_t, 3> in_stride{1, X, X * Y};   // i_x, i_y, i_z

    // Compute canonical shape
    std::array<int64_t, 3> canon_shape{0, 0, 0};

    for (int i = 0; i < 3; ++i) {
        canon_shape[dst[i]] = in_shape[i];
    }

    out.resize(canon_shape[2], canon_shape[1], canon_shape[0]);
    // out indexed as out.at(canon_z, canon_y, canon_x), but we store data in
    // canonical-(Z,Y,X) order with X fastest.

    // For each canonical (x, y, z) compute the corresponding (ix, iy, iz):
    //   in_axis i maps to canon_axis dst[i]. Inverse: canon_axis k -> input axis inv[k].
    std::array<int, 3> inv{-1, -1, -1};

    for (int i = 0; i < 3; ++i) {
        inv[dst[i]] = i;
    }

    std::array<int, 3> sgn_inv{0, 0, 0};

    for (int k = 0; k < 3; ++k) {
        sgn_inv[k] = sgn[inv[k]];
    }

    // Loop over input space (fastest path) and scatter into canonical output.
    // For each input (ix, iy, iz), compute canonical (cx, cy, cz):
    //   c[dst[i]] = (sgn[i] == +1) ? in_idx_i : (in_shape[i] - 1 - in_idx_i)
    const int64_t CZ = canon_shape[2];
    const int64_t CY = canon_shape[1];
    const int64_t CX = canon_shape[0];
    (void)CZ;
    (void)CY;
    (void)CX;

    for (int64_t iz = 0; iz < Z; ++iz) {
        for (int64_t iy = 0; iy < Y; ++iy) {
            for (int64_t ix = 0; ix < X; ++ix) {
                int64_t idx_in[3] = {ix, iy, iz};
                int64_t cidx[3] = {0, 0, 0};

                for (int i = 0; i < 3; ++i) {
                    cidx[dst[i]] = (sgn[i] == +1) ? idx_in[i] : (in_shape[i] - 1 - idx_in[i]);
                }

                int64_t out_idx = cidx[2] * out.stride(0) + cidx[1] * out.stride(1) + cidx[0];
                out.data[out_idx] = static_cast<float>(src[iz * in_stride[2] + iy * in_stride[1] + ix]);
            }
        }
    }
}

template <typename DstT>
void copy_reorient_from_canonical(const uint8_t* labels_canon_zyx,
                                  const Volume& canon_shape_template,
                                  int64_t X, int64_t Y, int64_t Z,
                                  const std::array<int, 3>& dst,
                                  const std::array<int, 3>& sgn,
                                  DstT* out_xyz) {
    // Inverse of the reorient: read canonical (z, y, x) and write back into
    // the input axis order (X fastest, then Y, then Z).
    (void)canon_shape_template;
    std::array<int64_t, 3> in_shape{X, Y, Z};

    for (int64_t iz = 0; iz < Z; ++iz) {
        for (int64_t iy = 0; iy < Y; ++iy) {
            for (int64_t ix = 0; ix < X; ++ix) {
                int64_t idx_in[3] = {ix, iy, iz};
                int64_t cidx[3] = {0, 0, 0};

                for (int i = 0; i < 3; ++i) {
                    cidx[dst[i]] = (sgn[i] == +1) ? idx_in[i] : (in_shape[i] - 1 - idx_in[i]);
                }

                int64_t in_idx = cidx[2] * (canon_shape_template.shape[1] * canon_shape_template.shape[2])
                                 + cidx[1] * canon_shape_template.shape[2]
                                 + cidx[0];
                out_xyz[iz * X * Y + iy * X + ix] = static_cast<DstT>(labels_canon_zyx[in_idx]);
            }
        }
    }
}

}  // namespace

NiftiImage load_nifti_ras(const std::string& path) {
    nifti_image* nim = nifti_image_read(path.c_str(), 1);

    if (!nim) {
        throw std::runtime_error("failed to load NIfTI: " + path);
    }

    if (nim->ndim < 3) {
        nifti_image_free(nim);
        throw std::runtime_error("NIfTI must be at least 3D: " + path);
    }

    if (nim->ndim > 3 && nim->nt > 1) {
        nifti_image_free(nim);
        throw std::runtime_error("4D NIfTI not supported; use a 3D volume");
    }

    NiftiImage out;
    out.shape_orig = {nim->nx, nim->ny, nim->nz};
    out.datatype_orig = nim->datatype;

    // Choose sform if valid, else qform
    mat44 affine;

    if (nim->sform_code > 0) {
        affine = nim->sto_xyz;
    } else if (nim->qform_code > 0) {
        affine = nim->qto_xyz;
    } else {
        // Fallback: identity scaled by pixdim
        std::memset(&affine, 0, sizeof(affine));
        affine.m[0][0] = nim->dx;
        affine.m[1][1] = nim->dy;
        affine.m[2][2] = nim->dz;
        affine.m[3][3] = 1.0f;
    }

    out.affine_orig = pack_mat44(affine);

    std::array<int, 3> dst{}, sgn{};
    axes_to_canonical(affine, dst, sgn);

    // Dispatch by datatype to a templated copier (covers the common ones).
    const int64_t X = nim->nx, Y = nim->ny, Z = nim->nz;

    switch (nim->datatype) {
        case DT_INT8:
            copy_reorient_to_canonical(static_cast<int8_t*>(nim->data),   X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_UINT8:
            copy_reorient_to_canonical(static_cast<uint8_t*>(nim->data),  X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_INT16:
            copy_reorient_to_canonical(static_cast<int16_t*>(nim->data),  X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_UINT16:
            copy_reorient_to_canonical(static_cast<uint16_t*>(nim->data), X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_INT32:
            copy_reorient_to_canonical(static_cast<int32_t*>(nim->data),  X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_UINT32:
            copy_reorient_to_canonical(static_cast<uint32_t*>(nim->data), X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_FLOAT32:
            copy_reorient_to_canonical(static_cast<float*>(nim->data),    X, Y, Z, dst, sgn, out.volume);
            break;

        case DT_FLOAT64:
            copy_reorient_to_canonical(static_cast<double*>(nim->data),   X, Y, Z, dst, sgn, out.volume);
            break;

        default:
            nifti_image_free(nim);
            throw std::runtime_error("unsupported NIfTI datatype: " + std::to_string(nim->datatype));
    }

    // Compute canonical affine: permute and flip the input affine columns.
    mat44 canon_aff{};
    canon_aff.m[3][3] = 1.0f;

    // For each input axis i, place its column into canon position dst[i] with sign sgn[i].
    // If we flipped that axis, we also need to translate the origin to the flipped end.
    for (int i = 0; i < 3; ++i) {
        int k = dst[i];
        float s = static_cast<float>(sgn[i]);

        for (int r = 0; r < 3; ++r) {
            canon_aff.m[r][k] = s * affine.m[r][i];
        }

        if (sgn[i] == -1) {
            // Translate origin by (in_shape[i] - 1) * original column to account for flip.
            for (int r = 0; r < 3; ++r) {
                canon_aff.m[r][3] += affine.m[r][i] * static_cast<float>(out.shape_orig[i] - 1);
            }
        }
    }

    // Add input affine's translation column.
    for (int r = 0; r < 3; ++r) {
        canon_aff.m[r][3] += affine.m[r][3];
    }

    out.affine_canon = pack_mat44(canon_aff);
    out.zooms_canon = {std::fabs(canon_aff.m[0][0]),
                       std::fabs(canon_aff.m[1][1]),
                       std::fabs(canon_aff.m[2][2])
                      };

    // Remember the perm/flip to invert during save.
    for (int i = 0; i < 3; ++i) {
        out.perm_canon_to_orig[i] = dst[i];
    }

    for (int i = 0; i < 3; ++i) {
        out.flip_canon[dst[i]] = sgn[i];    // canonical-axis-keyed
    }

    nifti_image_free(nim);
    return out;
}

void save_nifti_labels(const std::string& path,
                       const NiftiImage& src,
                       const uint8_t* labels_zyx) {
    // Build a uint8 nifti with the original geometry, and de-canonicalize the
    // labels back into (X, Y, Z) input axis order.
    nifti_image* nim = nifti_simple_init_nim();
    nim->datatype = DT_UINT8;
    nim->nbyper = 1;
    nim->ndim = 3;
    nim->nx = static_cast<int>(src.shape_orig[0]);
    nim->ny = static_cast<int>(src.shape_orig[1]);
    nim->nz = static_cast<int>(src.shape_orig[2]);
    nim->dim[0] = 3;
    nim->dim[1] = nim->nx;
    nim->dim[2] = nim->ny;
    nim->dim[3] = nim->nz;
    nim->dim[4] = 1;
    nim->dim[5] = 1;
    nim->dim[6] = 1;
    nim->dim[7] = 1;
    nim->nvox = static_cast<size_t>(nim->nx) * nim->ny * nim->nz;

    // Restore original sform/qform from src.affine_orig (row-major).
    mat44 affine{};

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            affine.m[r][c] = src.affine_orig[r * 4 + c];
        }
    }

    nim->sto_xyz = affine;
    nim->sform_code = 2;  // aligned
    nim->qform_code = 0;
    // Set voxel sizes from the affine.
    nim->dx = std::sqrt(affine.m[0][0] * affine.m[0][0] + affine.m[1][0] * affine.m[1][0] + affine.m[2][0] * affine.m[2][0]);
    nim->dy = std::sqrt(affine.m[0][1] * affine.m[0][1] + affine.m[1][1] * affine.m[1][1] + affine.m[2][1] * affine.m[2][1]);
    nim->dz = std::sqrt(affine.m[0][2] * affine.m[0][2] + affine.m[1][2] * affine.m[1][2] + affine.m[2][2] * affine.m[2][2]);
    nim->pixdim[0] = -1.0f;
    nim->pixdim[1] = nim->dx;
    nim->pixdim[2] = nim->dy;
    nim->pixdim[3] = nim->dz;

    nim->data = std::calloc(nim->nvox, 1);

    if (!nim->data) {
        nifti_image_free(nim);
        throw std::runtime_error("calloc failed for output label volume");
    }

    // De-canonicalize: invert perm+flip.
    std::array<int, 3> dst{};

    for (int i = 0; i < 3; ++i) {
        dst[i] = src.perm_canon_to_orig[i];
    }

    std::array<int, 3> sgn{};

    for (int i = 0; i < 3; ++i) {
        sgn[i] = src.flip_canon[dst[i]];
    }

    copy_reorient_from_canonical(labels_zyx,
                                 src.volume,
                                 static_cast<int64_t>(nim->nx),
                                 static_cast<int64_t>(nim->ny),
                                 static_cast<int64_t>(nim->nz),
                                 dst, sgn,
                                 static_cast<uint8_t*>(nim->data));

    nifti_set_filenames(nim, path.c_str(), 0, 1);
    nifti_image_write(nim);
    nifti_image_free(nim);
}

}  // namespace siam
