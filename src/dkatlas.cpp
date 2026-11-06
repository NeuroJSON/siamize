/***************************************************************************//**
\file    dkatlas.cpp
@brief   Desikan-Killiany cortical parcellation of SIAM's gray matter (see dkatlas.h)
*******************************************************************************/

#include "dkatlas.h"


#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace siam {

namespace {

constexpr int kN = 256;        // the conformed grid: 256^3 at 1 mm, LIA
constexpr int kC = 24;         // hidden channels
constexpr int kClasses = 104;
constexpr int kLayers = 13;
constexpr int kDil[kLayers] = {1, 3, 5, 7, 13, 19, 31, 19, 13, 7, 5, 3, 1};
constexpr int64_t kVox = static_cast<int64_t>(kN) * kN * kN;

using Mat4 = std::array<double, 16>;  // row-major

struct Weights {
    std::vector<float> conv[kLayers];      // [tap][cin][cout] (TF.js order)
    float scale[kLayers][kC], shift[kLayers][kC];
    std::vector<float> fk;                  // [cin][class]
    float fb[kClasses];
};

/* the weights compiled in: brainchop's model.bin as float32 bit patterns
   (tools/dkatlas/gen_weights.py; MIT, src/dkatlas_LICENSE.txt) */
const uint32_t kWeightBits[] = {
#include "dkatlas_weights.inc"
};

/* brainchop's model.bin: raw little-endian float32, in the TF.js manifest's
   order -- per layer the conv kernel [3,3,3,cin,24], the per-channel affine
   as a 1x1 kernel [24,24] (diagonal) and its bias [24]; then the classifier
   [24,104] and its bias [104]. From `path`, else the compiled-in copy. */
Weights load_weights(const std::string& path) {
    const int64_t expect = 27LL * 1 * kC + 12LL * 27 * kC * kC + kLayers * (kC * kC + kC) + kC * kClasses + kClasses;
    std::vector<float> w(static_cast<size_t>(expect));

    if (path.empty()) {
        static_assert(sizeof(kWeightBits) == sizeof(float) * (27 * kC + 12 * 27 * kC * kC + kLayers * (kC * kC + kC)
                      + kC * kClasses + kClasses), "dkatlas_weights.inc: wrong size");
        std::memcpy(w.data(), kWeightBits, sizeof(kWeightBits));
    } else {
        std::ifstream f(path, std::ios::binary);

        if (!f) {
            throw std::runtime_error("cannot open DKatlas weights: " + path);
        }

        f.seekg(0, std::ios::end);
        const std::streamoff bytes = f.tellg();

        if (bytes != expect * 4) {
            throw std::runtime_error("DKatlas weights " + path + ": " + std::to_string(bytes) + " bytes, expected "
                                     + std::to_string(expect * 4) + " (brainchop model24chan104cls_synth model.bin)");
        }

        f.seekg(0);
        f.read(reinterpret_cast<char*>(w.data()), bytes);
    }

    Weights W;
    size_t o = 0;

    for (int l = 0; l < kLayers; ++l) {
        const size_t n = 27 * static_cast<size_t>(l == 0 ? 1 : kC) * kC;
        W.conv[l].assign(w.begin() + o, w.begin() + o + n);
        o += n;

        for (int c = 0; c < kC; ++c) {
            for (int c2 = 0; c2 < kC; ++c2) {
                if (c2 != c && w[o + c * kC + c2] != 0.0f) {
                    throw std::runtime_error("DKatlas weights: affine " + std::to_string(l) + " is not diagonal");
                }
            }

            W.scale[l][c] = w[o + c * kC + c];
        }

        o += kC * kC;
        std::memcpy(W.shift[l], &w[o], kC * sizeof(float));
        o += kC;
    }

    W.fk.assign(w.begin() + o, w.begin() + o + kC * kClasses);
    o += kC * kClasses;
    std::memcpy(W.fb, &w[o], kClasses * sizeof(float));
    return W;
}

Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) {
                r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }

    return r;
}

Mat4 inv(const Mat4& m) {  // affine: [R t; 0 1]
    const double* a = m.data();
    const double det = a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8])
                       + a[2] * (a[4] * a[9] - a[5] * a[8]);
    Mat4 r{};
    r[0] = (a[5] * a[10] - a[6] * a[9]) / det;
    r[1] = (a[2] * a[9] - a[1] * a[10]) / det;
    r[2] = (a[1] * a[6] - a[2] * a[5]) / det;
    r[4] = (a[6] * a[8] - a[4] * a[10]) / det;
    r[5] = (a[0] * a[10] - a[2] * a[8]) / det;
    r[6] = (a[2] * a[4] - a[0] * a[6]) / det;
    r[8] = (a[4] * a[9] - a[5] * a[8]) / det;
    r[9] = (a[1] * a[8] - a[0] * a[9]) / det;
    r[10] = (a[0] * a[5] - a[1] * a[4]) / det;

    for (int i = 0; i < 3; ++i) {
        r[i * 4 + 3] = -(r[i * 4] * a[3] + r[i * 4 + 1] * a[7] + r[i * 4 + 2] * a[11]);
    }

    r[15] = 1;
    return r;
}

Mat4 to_mat(const std::array<float, 16>& a) {
    Mat4 r{};

    for (int i = 0; i < 16; ++i) {
        r[i] = a[i];
    }

    r[12] = r[13] = r[14] = 0;
    r[15] = 1;
    return r;
}

/* the conformed grid's affine (voxel (i,j,k) -> world), as nibabel's
   conform(out_shape=256^3, voxel_size=1, orientation='LIA') -- the input
   reoriented to LIA (i toward L, j toward I, k toward A), its axes scaled to
   1 mm (an oblique input stays oblique), centred on its voxel (n-1)//2 */
Mat4 conform_affine(const NiftiImage& t1) {
    const int64_t X = t1.volume.shape[2], Y = t1.volume.shape[1], Z = t1.volume.shape[0];
    // canonical (x,y,z) from LIA (i,j,k): x = X-1-i, z = Z-1-j, y = k
    Mat4 P{};
    P[0 * 4 + 0] = -1;
    P[0 * 4 + 3] = static_cast<double>(X - 1);
    P[2 * 4 + 1] = -1;
    P[2 * 4 + 3] = static_cast<double>(Z - 1);
    P[1 * 4 + 2] = 1;
    P[15] = 1;
    const Mat4 R = mul(to_mat(t1.affine_canon), P);  // reoriented (shape X, Z, Y)
    Mat4 out = R;

    for (int c = 0; c < 3; ++c) {
        const double s = std::sqrt(R[c] * R[c] + R[4 + c] * R[4 + c] + R[8 + c] * R[8 + c]);

        for (int r = 0; r < 3; ++r) {
            out[r * 4 + c] = R[r * 4 + c] / s;
        }
    }

    const double ci[3] = {static_cast<double>((X - 1) / 2), static_cast<double>((Z - 1) / 2),
                          static_cast<double>((Y - 1) / 2)
                         };
    const double h = (kN - 1) / 2;  // 127

    for (int r = 0; r < 3; ++r) {
        const double cen = R[r * 4 + 0] * ci[0] + R[r * 4 + 1] * ci[1] + R[r * 4 + 2] * ci[2] + R[r * 4 + 3];
        out[r * 4 + 3] = cen - (out[r * 4] + out[r * 4 + 1] + out[r * 4 + 2]) * h;
    }

    return out;
}

/* the T1 on the conformed grid (trilinear, 0 outside), quantile-normalized to
   [0, 1] between its 2% and 98% values (brainchop's qnormalize) */
std::vector<float> conform_input(const NiftiImage& t1, const Mat4& C) {
    const Mat4 M = mul(inv(to_mat(t1.affine_canon)), C);  // conformed voxel -> canonical voxel
    const Volume& v = t1.volume;
    const int64_t Z = v.shape[0], Y = v.shape[1], X = v.shape[2];
    std::vector<float> in(static_cast<size_t>(kVox));
    #pragma omp parallel for schedule(static)

    for (int i = 0; i < kN; ++i) {
        for (int j = 0; j < kN; ++j) {
            for (int k = 0; k < kN; ++k) {
                const double p[3] = {M[0]* i + M[1]* j + M[2]* k + M[3], M[4]* i + M[5]* j + M[6]* k + M[7],
                                     M[8]* i + M[9]* j + M[10]* k + M[11]
                                    };
                const int64_t x0 = static_cast<int64_t>(std::floor(p[0])), y0 = static_cast<int64_t>(std::floor(p[1])),
                              z0 = static_cast<int64_t>(std::floor(p[2]));
                const double fx = p[0] - x0, fy = p[1] - y0, fz = p[2] - z0;
                double s = 0;

                for (int dz = 0; dz < 2; ++dz)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            const int64_t xx = x0 + dx, yy = y0 + dy, zz = z0 + dz;

                            if (xx < 0 || yy < 0 || zz < 0 || xx >= X || yy >= Y || zz >= Z) {
                                continue;
                            }

                            s += (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz) * v.at(zz, yy, xx);
                        }

                in[(static_cast<size_t>(i) * kN + j) * kN + k] = static_cast<float>(s);
            }
        }
    }

    // numpy.quantile (linear): position q * (n - 1)
    std::vector<float> tmp(in);
    auto quant = [&](double q) {
        const double pos = q * static_cast<double>(kVox - 1);
        const int64_t lo = static_cast<int64_t>(pos);
        std::nth_element(tmp.begin(), tmp.begin() + lo, tmp.end());
        const double a = tmp[lo];
        const double b = lo + 1 < kVox ? *std::min_element(tmp.begin() + lo + 1, tmp.end()) : a;
        return a + (pos - lo) * (b - a);
    };
    const double qlo = quant(0.02), qhi = quant(0.98);
    const float sc = static_cast<float>(1.0 / (qhi - qlo + 1e-3));
    #pragma omp parallel for schedule(static)

    for (int64_t n = 0; n < kVox; ++n) {
        in[n] = std::min(1.0f, std::max(0.0f, (in[n] - static_cast<float>(qlo)) * sc));
    }

    return in;
}

/* a 'same' dilated 3x3x3 convolution, channels-last, cin -> 24 (no bias) */
void conv3(const float* in, int cin, float* out, const float* w, int d) {
    #pragma omp parallel for schedule(dynamic, 1)

    for (int i = 0; i < kN; ++i) {
        for (int j = 0; j < kN; ++j) {
            for (int k = 0; k < kN; ++k) {
                float acc[kC] = {0};

                for (int a = 0; a < 3; ++a) {
                    const int ii = i + (a - 1) * d;

                    if (ii < 0 || ii >= kN) {
                        continue;
                    }

                    for (int b = 0; b < 3; ++b) {
                        const int jj = j + (b - 1) * d;

                        if (jj < 0 || jj >= kN) {
                            continue;
                        }

                        for (int c = 0; c < 3; ++c) {
                            const int kk = k + (c - 1) * d;

                            if (kk < 0 || kk >= kN) {
                                continue;
                            }

                            const float* x = in + ((static_cast<size_t>(ii) * kN + jj) * kN + kk) * cin;
                            const float* wt = w + static_cast<size_t>((a * 3 + b) * 3 + c) * cin * kC;

                            for (int ci = 0; ci < cin; ++ci) {
                                const float xv = x[ci];
                                const float* wr = wt + ci * kC;

                                for (int co = 0; co < kC; ++co) {
                                    acc[co] += xv * wr[co];
                                }
                            }
                        }
                    }
                }

                std::memcpy(out + ((static_cast<size_t>(i) * kN + j) * kN + k) * kC, acc, sizeof(acc));
            }
        }
    }
}

/* per-channel normalization over the whole volume (GroupNorm with one group
   per channel, eps 1e-5), the channel's scale and shift, then GELU (tanh) */
void norm_gelu(float* x, const float* scale, const float* shift) {
    double sum[kC] = {0}, sq[kC] = {0};
    #pragma omp parallel
    {
        double s[kC] = {0}, q[kC] = {0};
        #pragma omp for schedule(static)

        for (int64_t n = 0; n < kVox; ++n) {
            const float* v = x + n * kC;

            for (int c = 0; c < kC; ++c) {
                s[c] += v[c];
                q[c] += static_cast<double>(v[c]) * v[c];
            }
        }

        #pragma omp critical
        {
            for (int c = 0; c < kC; ++c) {
                sum[c] += s[c];
                sq[c] += q[c];
            }
        }
    }
    float a[kC], b[kC];

    for (int c = 0; c < kC; ++c) {
        const double mean = sum[c] / kVox, var = std::max(0.0, sq[c] / kVox - mean * mean);
        a[c] = static_cast<float>(scale[c] / std::sqrt(var + 1e-5));
        b[c] = static_cast<float>(shift[c] - mean * a[c]);
    }

    #pragma omp parallel for schedule(static)

    for (int64_t n = 0; n < kVox; ++n) {
        float* v = x + n * kC;

        for (int c = 0; c < kC; ++c) {
            const float y = v[c] * a[c] + b[c];
            v[c] = 0.5f * y * (1.0f + std::tanh(0.7978845608f * (y + 0.044715f * y * y * y)));
        }
    }
}

/* the network on the conformed input: per-voxel DK class (0..103) */
std::vector<uint8_t> run_net(const std::vector<float>& in, const Weights& W) {
    std::vector<float> A(static_cast<size_t>(kVox) * kC), B(static_cast<size_t>(kVox) * kC);
    conv3(in.data(), 1, A.data(), W.conv[0].data(), kDil[0]);
    norm_gelu(A.data(), W.scale[0], W.shift[0]);
    float* cur = A.data();
    float* nxt = B.data();

    for (int l = 1; l < kLayers; ++l) {
        conv3(cur, kC, nxt, W.conv[l].data(), kDil[l]);
        norm_gelu(nxt, W.scale[l], W.shift[l]);
        std::swap(cur, nxt);
    }

    std::vector<uint8_t> lab(static_cast<size_t>(kVox));
    #pragma omp parallel for schedule(static)

    for (int64_t n = 0; n < kVox; ++n) {
        float z[kClasses];
        std::memcpy(z, W.fb, sizeof(z));
        const float* v = cur + n * kC;

        for (int ci = 0; ci < kC; ++ci) {
            const float* wr = W.fk.data() + ci * kClasses;

            for (int c = 0; c < kClasses; ++c) {
                z[c] += v[ci] * wr[c];
            }
        }

        lab[n] = static_cast<uint8_t>(std::max_element(z, z + kClasses) - z);
    }

    return lab;
}

int side_of(int dk) {  // 1 left, 2 right, 0 neither
    const char* n = kDKNames[dk];
    return (std::strstr(n, "-lh-") || std::strncmp(n, "Left-", 5) == 0) ? 1
           : (std::strstr(n, "-rh-") || std::strncmp(n, "Right-", 6) == 0) ? 2 : 0;
}

bool cortical(int dk) {
    return dk >= 1 && dk <= 68;
}

}  // namespace

std::vector<uint8_t> dk_parcellate(const NiftiImage& t1, const NiftiImage& grid, const uint8_t* labels_zyx,
                                   uint8_t gm_label, const std::string& weights, DKStats& st) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const Weights W = load_weights(weights);
    const Mat4 C = conform_affine(t1);
    std::vector<uint8_t> dk = run_net(conform_input(t1, C), W);

    if (const char* dump = std::getenv("SIAMIZE_DK_DUMP")) {  // (debug: the raw 256^3 classes, i slowest)
        std::ofstream(dump, std::ios::binary).write(reinterpret_cast<const char*>(dk.data()),
                static_cast<std::streamsize>(dk.size()));
    }

    {
        bool seen[kClasses] = {false};

        for (uint8_t v : dk) {
            seen[v] = true;
        }

        st.classes = static_cast<int>(std::count(seen + 1, seen + kClasses, true));
    }

    const auto t1c = clk::now();
    st.seconds_net = std::chrono::duration<double>(t1c - t0).count();

    // 1. the DK class at each output voxel (nearest), through world coordinates
    const int64_t Z = grid.volume.shape[0], Y = grid.volume.shape[1], X = grid.volume.shape[2];
    const int64_t nv = Z * Y * X;
    const Mat4 G = mul(inv(C), to_mat(grid.affine_canon));  // output voxel -> conformed voxel
    std::vector<uint8_t> on(static_cast<size_t>(nv), 0);
    #pragma omp parallel for schedule(static)

    for (int64_t z = 0; z < Z; ++z) {
        for (int64_t y = 0; y < Y; ++y) {
            for (int64_t x = 0; x < X; ++x) {
                const int64_t i = std::llround(G[0] * x + G[1] * y + G[2] * z + G[3]);
                const int64_t j = std::llround(G[4] * x + G[5] * y + G[6] * z + G[7]);
                const int64_t k = std::llround(G[8] * x + G[9] * y + G[10] * z + G[11]);

                if (i >= 0 && j >= 0 && k >= 0 && i < kN && j < kN && k < kN) {
                    on[(z * Y + y) * X + x] = dk[(static_cast<size_t>(i) * kN + j) * kN + k];
                }
            }
        }
    }

    dk = std::vector<uint8_t>();
    const int64_t nb[6] = {1, -1, X, -X, X * Y, -X * Y};
    auto inside = [&](int64_t n, int e) {  // the neighbour n + nb[e] exists
        const int64_t x = n % X, y = (n / X) % Y, z = n / (X * Y);
        return e == 0 ? x + 1 < X : e == 1 ? x > 0 : e == 2 ? y + 1 < Y : e == 3 ? y > 0 : e == 4 ? z + 1 < Z : z > 0;
    };

    // 2. each voxel's hemisphere: that of the nearest DK left / right class
    // (breadth-first from them, 6-connected)
    std::vector<uint8_t> side(static_cast<size_t>(nv), 0);
    std::vector<int64_t> q;
    q.reserve(static_cast<size_t>(nv / 4));

    for (int64_t n = 0; n < nv; ++n) {
        if ((side[n] = static_cast<uint8_t>(side_of(on[n]))) != 0) {
            q.push_back(n);
        }
    }

    for (size_t h = 0; h < q.size(); ++h) {
        const int64_t n = q[h];

        for (int e = 0; e < 6; ++e) {
            if (inside(n, e) && side[n + nb[e]] == 0) {
                side[n + nb[e]] = side[n];
                q.push_back(n + nb[e]);
            }
        }
    }

    // 3. SIAM GM inside a DK cortical class keeps it; the rest grows from those,
    // breadth-first through the GM only, a label only into its own hemisphere
    std::vector<uint8_t> par(static_cast<size_t>(nv), 0);
    q.clear();

    for (int64_t n = 0; n < nv; ++n) {
        if (labels_zyx[n] == gm_label) {
            ++st.gm;

            if (cortical(on[n])) {
                par[n] = on[n];
                q.push_back(n);
            }
        }
    }

    st.direct = static_cast<int64_t>(q.size());

    for (size_t h = 0; h < q.size(); ++h) {
        const int64_t n = q[h];

        for (int e = 0; e < 6; ++e) {
            const int64_t m = n + nb[e];

            if (inside(n, e) && par[m] == 0 && labels_zyx[m] == gm_label && side[m] == side_of(par[n])) {
                par[m] = par[n];
                q.push_back(m);
            }
        }
    }

    st.grown = static_cast<int64_t>(q.size()) - st.direct;
    st.unreached = st.gm - st.direct - st.grown;
    st.seconds_transfer = std::chrono::duration<double>(clk::now() - t1c).count();
    return par;
}

}  // namespace siam
