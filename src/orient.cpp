#include "orient.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace siam {

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

// ---- explicit instantiations for the types nifti_io / MEX use --------------
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
