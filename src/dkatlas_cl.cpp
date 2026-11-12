/***************************************************************************//**
\file    dkatlas_cl.cpp
@brief   The DKatlas network on an OpenCL device (see dkatlas_cl.h)

The OpenCL ICD is loaded at run time (dlopen / LoadLibrary) and the few entry
points used are resolved by name, so siamize gains no link-time OpenCL
dependency and the ORT (CUDA) build gets it too. Any failure -- no ICD, no
device, too little memory, a build or launch error -- returns false with the
reason, and the caller runs the CPU path.
*******************************************************************************/

#include "dkatlas_cl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace siam {

namespace {

// ---- the OpenCL 1.2 subset used, declared here (no CL headers needed) ----
using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_handle = void*;   // platform, device, context, queue, program, kernel, mem

#if defined(_WIN32)
    #define DKCL_API __stdcall
#else
    #define DKCL_API
#endif

constexpr cl_ulong CL_DEVICE_TYPE_GPU = 1ULL << 2, CL_DEVICE_TYPE_ALL = 0xFFFFFFFFULL;
constexpr cl_uint CL_DEVICE_TYPE = 0x1000, CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F,
                  CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020,
                  CL_DEVICE_NAME = 0x102B, CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_ulong CL_MEM_READ_WRITE = 1 << 0, CL_MEM_READ_ONLY = 1 << 2, CL_MEM_WRITE_ONLY = 1 << 1,
                   CL_MEM_COPY_HOST_PTR = 1 << 5;
constexpr cl_uint CL_TRUE = 1;

struct Api {
    cl_int (DKCL_API* GetPlatformIDs)(cl_uint, cl_handle*, cl_uint*);
    cl_int (DKCL_API* GetDeviceIDs)(cl_handle, cl_ulong, cl_uint, cl_handle*, cl_uint*);
    cl_int (DKCL_API* GetDeviceInfo)(cl_handle, cl_uint, size_t, void*, size_t*);
    cl_handle (DKCL_API* CreateContext)(const intptr_t*, cl_uint, const cl_handle*, void*, void*, cl_int*);
    cl_handle (DKCL_API* CreateCommandQueue)(cl_handle, cl_handle, cl_ulong, cl_int*);
    cl_handle (DKCL_API* CreateProgramWithSource)(cl_handle, cl_uint, const char**, const size_t*, cl_int*);
    cl_int (DKCL_API* BuildProgram)(cl_handle, cl_uint, const cl_handle*, const char*, void*, void*);
    cl_int (DKCL_API* GetProgramBuildInfo)(cl_handle, cl_handle, cl_uint, size_t, void*, size_t*);
    cl_handle (DKCL_API* CreateKernel)(cl_handle, const char*, cl_int*);
    cl_handle (DKCL_API* CreateBuffer)(cl_handle, cl_ulong, size_t, void*, cl_int*);
    cl_int (DKCL_API* SetKernelArg)(cl_handle, cl_uint, size_t, const void*);
    cl_int (DKCL_API* EnqueueNDRangeKernel)(cl_handle, cl_handle, cl_uint, const size_t*, const size_t*,
                                            const size_t*, cl_uint, const cl_handle*, cl_handle*);
    cl_int (DKCL_API* EnqueueReadBuffer)(cl_handle, cl_handle, cl_uint, size_t, size_t, void*, cl_uint,
                                         const cl_handle*, cl_handle*);
    cl_int (DKCL_API* Finish)(cl_handle);
    cl_int (DKCL_API* ReleaseMemObject)(cl_handle);
    cl_int (DKCL_API* ReleaseKernel)(cl_handle);
    cl_int (DKCL_API* ReleaseProgram)(cl_handle);
    cl_int (DKCL_API* ReleaseCommandQueue)(cl_handle);
    cl_int (DKCL_API* ReleaseContext)(cl_handle);
};

#if defined(_WIN32)
void* lib_open() {
    return reinterpret_cast<void*>(::LoadLibraryA("OpenCL.dll"));
}
void* lib_sym(void* h, const char* s) {
    return reinterpret_cast<void*>(::GetProcAddress(reinterpret_cast<HMODULE>(h), s));
}
#else
void* lib_open() {
    const char* names[] = {
#if defined(__APPLE__)
        "/System/Library/Frameworks/OpenCL.framework/OpenCL", "libOpenCL.dylib",
#else
        "libOpenCL.so.1", "libOpenCL.so",
#endif
    };

    for (const char* n : names) {
        if (void* h = ::dlopen(n, RTLD_LAZY | RTLD_LOCAL)) {
            return h;
        }
    }

    return nullptr;
}
void* lib_sym(void* h, const char* s) {
    return ::dlsym(h, s);
}
#endif

/* the ICD's entry points (loaded once, never unloaded: a driver may keep
   threads that outlive a dlclose) */
const Api* api(std::string& why) {
    static Api a;
    static int state = 0;   // 0 untried, 1 ok, -1 failed
    static std::string fail;

    if (state == 0) {
        state = -1;
        void* h = lib_open();

        if (!h) {
            fail = "no OpenCL runtime (ICD) found";
        } else {
            bool ok = true;
            auto get = [&](auto & f, const char* name) {
                f = reinterpret_cast<std::remove_reference_t<decltype(f)>>(lib_sym(h, name));
                ok = ok && f != nullptr;
            };
            get(a.GetPlatformIDs, "clGetPlatformIDs");
            get(a.GetDeviceIDs, "clGetDeviceIDs");
            get(a.GetDeviceInfo, "clGetDeviceInfo");
            get(a.CreateContext, "clCreateContext");
            get(a.CreateCommandQueue, "clCreateCommandQueue");
            get(a.CreateProgramWithSource, "clCreateProgramWithSource");
            get(a.BuildProgram, "clBuildProgram");
            get(a.GetProgramBuildInfo, "clGetProgramBuildInfo");
            get(a.CreateKernel, "clCreateKernel");
            get(a.CreateBuffer, "clCreateBuffer");
            get(a.SetKernelArg, "clSetKernelArg");
            get(a.EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
            get(a.EnqueueReadBuffer, "clEnqueueReadBuffer");
            get(a.Finish, "clFinish");
            get(a.ReleaseMemObject, "clReleaseMemObject");
            get(a.ReleaseKernel, "clReleaseKernel");
            get(a.ReleaseProgram, "clReleaseProgram");
            get(a.ReleaseCommandQueue, "clReleaseCommandQueue");
            get(a.ReleaseContext, "clReleaseContext");

            if (ok) {
                state = 1;
            } else {
                fail = "the OpenCL runtime lacks an OpenCL 1.2 entry point";
            }
        }
    }

    why = fail;
    return state == 1 ? &a : nullptr;
}

/* the kernels: channels-last [i][j][k][c], i slowest, N^3 voxels, C channels.
   conv: a 'same' dilated 3x3x3 convolution (zero padding), all C outputs per
   voxel; the weights [tap][cin][C] in constant memory, read by every
   work-item at the same address (a broadcast). The 24-channel layers stage
   input rows in local memory (see conv_c24).
   stats: per-channel sum and sum of squares, one partial per work-group.
   norm_gelu: y = x*a[c] + b[c], then GELU (tanh form), in place.
   head: the 1x1 to the classes and its argmax. */
const char* kSource = R"CLC(
#define N 256
#define C 24
#define NCLS 104

#define C4 (C / 4)
#define WG 64            /* work-items per row of k */
#define VK (N / WG)      /* voxels per work-item: k = t, t + WG, .. */
#define DMAX 31          /* the largest dilation (the tile's halo) */

/* layer 0 (one input channel): a work-item per voxel */
__kernel void conv_c1(__global const float* in, __global float* out, __constant float* w, const int d) {
    const int k = get_global_id(0), j = get_global_id(1), i = get_global_id(2);
    float acc[C];
    for (int co = 0; co < C; ++co) acc[co] = 0.0f;
    for (int a = 0; a < 3; ++a) {
        const int ii = i + (a - 1) * d;
        if (ii < 0 || ii >= N) continue;
        for (int b = 0; b < 3; ++b) {
            const int jj = j + (b - 1) * d;
            if (jj < 0 || jj >= N) continue;
            for (int c = 0; c < 3; ++c) {
                const int kk = k + (c - 1) * d;
                if (kk < 0 || kk >= N) continue;
                const float xv = in[(ii * N + jj) * N + kk];
                __constant float* wr = w + ((a * 3 + b) * 3 + c) * C;
                for (int co = 0; co < C; ++co) acc[co] = mad(xv, wr[co], acc[co]);
            }
        }
    }
    __global float* o = out + ((i * N + j) * N + k) * C;
    for (int co = 0; co < C; ++co) o[co] = acc[co];
}

/* the 24-channel layers: a work-group per output row (i, j); for each of the
   9 input rows (ii, jj) it reaches, the whole row is staged in local memory
   channel-major with a zero halo of d, so the three k taps and all 24 input
   channels are read from there; each float4 weight load (a broadcast from
   constant memory) feeds 4 x VK mads */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void conv_c24(__global const float* in, __global float* out, __constant float4* w, const int d) {
    __local float tile[C][N + 2 * DMAX];
    const int t = get_local_id(0), j = get_global_id(1), i = get_global_id(2);
    float4 acc[VK][C4];
    #pragma unroll
    for (int v = 0; v < VK; ++v)
        #pragma unroll
        for (int q = 0; q < C4; ++q) acc[v][q] = (float4)(0.0f);
    for (int e = t; e < C * d; e += WG) {   /* the halo: zero */
        const int ci = e / d, h = e % d;
        tile[ci][h] = 0.0f;
        tile[ci][N + d + h] = 0.0f;
    }
    for (int a = 0; a < 3; ++a) {
        const int ii = i + (a - 1) * d;     /* (the same for the whole group) */
        if (ii < 0 || ii >= N) continue;
        for (int b = 0; b < 3; ++b) {
            const int jj = j + (b - 1) * d;
            if (jj < 0 || jj >= N) continue;
            __global const float* row = in + (ii * N + jj) * N * C;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int e = t; e < N * C; e += WG) tile[e % C][e / C + d] = row[e];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int c = 0; c < 3; ++c) {
                __constant float4* wt = w + ((a * 3 + b) * 3 + c) * C * C4;
                const int off = t + c * d;  /* tile column of k = t: k + (c - 1) d + d */
                for (int ci = 0; ci < C; ++ci) {
                    float xv[VK];
                    #pragma unroll
                    for (int v = 0; v < VK; ++v) xv[v] = tile[ci][off + v * WG];
                    #pragma unroll
                    for (int q = 0; q < C4; ++q) {
                        const float4 wq = wt[ci * C4 + q];
                        #pragma unroll
                        for (int v = 0; v < VK; ++v) acc[v][q] = mad((float4)(xv[v]), wq, acc[v][q]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int v = 0; v < VK; ++v) {
        __global float4* o = (__global float4*)(out + ((i * N + j) * N + t + v * WG) * C);
        #pragma unroll
        for (int q = 0; q < C4; ++q) o[q] = acc[v][q];
    }
}

__kernel void stats(__global const float* x, __global float* part, __local float* red) {
    const int lid = get_local_id(0), lsz = get_local_size(0);
    const int gsz = get_global_size(0);
    float s[C], q[C];
    for (int c = 0; c < C; ++c) { s[c] = 0.0f; q[c] = 0.0f; }
    for (int n = get_global_id(0); n < N * N * N; n += gsz) {
        __global const float* v = x + n * C;
        for (int c = 0; c < C; ++c) { const float t = v[c]; s[c] += t; q[c] = mad(t, t, q[c]); }
    }
    for (int m = 0; m < 2 * C; ++m) {
        red[lid] = m < C ? s[m] : q[m - C];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int h = lsz / 2; h > 0; h >>= 1) {
            if (lid < h) red[lid] += red[lid + h];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) part[get_group_id(0) * 2 * C + m] = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void norm_gelu(__global float* x, __constant float* ab) {
    const int g = get_global_id(0), c = g % C;
    const float y = mad(x[g], ab[c], ab[C + c]);
    x[g] = 0.5f * y * (1.0f + tanh(0.7978845608f * (y + 0.044715f * y * y * y)));
}

__kernel void head(__global const float* x, __constant float* fk, __constant float* fb,
                   __global uchar* lab) {
    const int n = get_global_id(0);
    __global const float* v = x + n * C;
    float xv[C];
    for (int c = 0; c < C; ++c) xv[c] = v[c];
    float best = -INFINITY;
    int arg = 0;
    for (int k = 0; k < NCLS; ++k) {
        float z = fb[k];
        for (int c = 0; c < C; ++c) z = mad(xv[c], fk[c * NCLS + k], z);
        if (z > best) { best = z; arg = k; }
    }
    lab[n] = (uchar)arg;
}
)CLC";

constexpr int kN = 256, kC = 24, kClasses = 104, kLayers = 13;
constexpr size_t kVox = static_cast<size_t>(kN) * kN * kN;
constexpr size_t kStatGroups = 1024, kStatLocal = 256;

/* the device: -G-style 1-based index in clinfo order (all platforms, all
   types), or 0 for the first GPU */
bool pick_device(const Api& cl, int index, cl_handle& dev, std::string& why) {
    cl_uint np = 0;

    if (cl.GetPlatformIDs(0, nullptr, &np) != 0 || np == 0) {
        why = "no OpenCL platform";
        return false;
    }

    std::vector<cl_handle> plats(np);
    cl.GetPlatformIDs(np, plats.data(), nullptr);
    int flat = 0;

    for (cl_handle p : plats) {
        cl_uint nd = 0;

        if (cl.GetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != 0 || nd == 0) {
            continue;
        }

        std::vector<cl_handle> devs(nd);
        cl.GetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);

        for (cl_handle d : devs) {
            ++flat;
            cl_ulong type = 0;
            cl.GetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);

            if ((index > 0 && flat == index) || (index <= 0 && (type & CL_DEVICE_TYPE_GPU))) {
                dev = d;
                return true;
            }
        }
    }

    why = index > 0 ? "OpenCL device " + std::to_string(index) + " not found" : "no OpenCL GPU";
    return false;
}

/* everything created on the device, released on any exit */
struct Session {
    const Api* cl = nullptr;
    cl_handle ctx = nullptr, queue = nullptr, prog = nullptr;
    std::vector<cl_handle> kernels, mems;

    ~Session() {
        for (cl_handle k : kernels) {
            cl->ReleaseKernel(k);
        }

        for (cl_handle m : mems) {
            cl->ReleaseMemObject(m);
        }

        if (prog) {
            cl->ReleaseProgram(prog);
        }

        if (queue) {
            cl->ReleaseCommandQueue(queue);
        }

        if (ctx) {
            cl->ReleaseContext(ctx);
        }
    }
    cl_handle buffer(cl_ulong flags, size_t bytes, const void* host, cl_int& err) {
        cl_handle m = cl->CreateBuffer(ctx, flags | (host ? CL_MEM_COPY_HOST_PTR : 0), bytes,
                                       const_cast<void*>(host), &err);

        if (m) {
            mems.push_back(m);
        }

        return m;
    }
    cl_handle kernel(const char* name, cl_int& err) {
        cl_handle k = cl->CreateKernel(prog, name, &err);

        if (k) {
            kernels.push_back(k);
        }

        return k;
    }
};

}  // namespace

bool dk_run_cl(const float* in, const DKNetWeights& W, int device, uint8_t* lab, std::string& info) {
    const Api* clp = api(info);

    if (!clp) {
        return false;
    }

    const Api& cl = *clp;
    cl_handle dev = nullptr;

    if (!pick_device(cl, device, dev, info)) {
        return false;
    }

    char name[256] = {0};
    cl.GetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
    cl_ulong gmem = 0, maxalloc = 0, cmax = 0;
    cl.GetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
    cl.GetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxalloc), &maxalloc, nullptr);
    cl.GetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cmax), &cmax, nullptr);
    const size_t act = kVox * kC * sizeof(float);   // one activation volume (1.6 GB)
    const size_t wmax = static_cast<size_t>(27) * kC * kC * sizeof(float);

    if (maxalloc < act || gmem < 2 * act + kVox * 5 + (64ull << 20)) {
        info = std::string(name) + ": too little memory (" + std::to_string(gmem >> 20) + " MB, max buffer " +
               std::to_string(maxalloc >> 20) + " MB; needs 2 x " + std::to_string(act >> 20) + " MB)";
        return false;
    }

    if (cmax < wmax) {
        info = std::string(name) + ": constant memory " + std::to_string(cmax) + " B < " + std::to_string(wmax);
        return false;
    }

    Session S;
    S.cl = clp;
    cl_int err = 0;
    S.ctx = cl.CreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    auto fail = [&](const char* what) {
        info = std::string(name) + ": " + what + " failed (error " + std::to_string(err) + ")";
        return false;
    };

    if (!S.ctx) {
        return fail("clCreateContext");
    }

    S.queue = cl.CreateCommandQueue(S.ctx, dev, 0, &err);

    if (!S.queue) {
        return fail("clCreateCommandQueue");
    }

    const char* src = kSource;
    S.prog = cl.CreateProgramWithSource(S.ctx, 1, &src, nullptr, &err);

    if (!S.prog) {
        return fail("clCreateProgramWithSource");
    }

    const std::string opts = "-cl-mad-enable";

    if ((err = cl.BuildProgram(S.prog, 1, &dev, opts.c_str(), nullptr, nullptr)) != 0) {
        size_t n = 0;
        cl.GetProgramBuildInfo(S.prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        cl.GetProgramBuildInfo(S.prog, dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
        info = std::string(name) + ": kernel build failed (error " + std::to_string(err) + "): " + log;
        return false;
    }

    cl_handle kc1 = S.kernel("conv_c1", err), kc24 = S.kernel("conv_c24", err), kst = S.kernel("stats", err),
              kng = S.kernel("norm_gelu", err), khd = S.kernel("head", err);

    if (!kc1 || !kc24 || !kst || !kng || !khd) {
        return fail("clCreateKernel");
    }

    cl_handle bin = S.buffer(CL_MEM_READ_ONLY, kVox * sizeof(float), in, err);
    cl_handle bA = S.buffer(CL_MEM_READ_WRITE, act, nullptr, err);
    cl_handle bB = S.buffer(CL_MEM_READ_WRITE, act, nullptr, err);
    cl_handle bpart = S.buffer(CL_MEM_WRITE_ONLY, kStatGroups * 2 * kC * sizeof(float), nullptr, err);
    cl_handle blab = S.buffer(CL_MEM_WRITE_ONLY, kVox, nullptr, err);
    cl_handle bfk = S.buffer(CL_MEM_READ_ONLY, sizeof(float) * kC * kClasses, W.fk, err);
    cl_handle bfb = S.buffer(CL_MEM_READ_ONLY, sizeof(float) * kClasses, W.fb, err);
    std::vector<cl_handle> bw(kLayers);

    for (int l = 0; l < kLayers; ++l) {
        bw[l] = S.buffer(CL_MEM_READ_ONLY, sizeof(float) * 27 * (l ? kC : 1) * kC, W.conv[l], err);
    }

    bool all = bin && bA && bB && bpart && blab && bfk && bfb;

    for (cl_handle b : bw) {
        all = all && b;
    }

    if (!all) {
        return fail("clCreateBuffer");
    }

    std::vector<float> part(kStatGroups * 2 * kC);
    std::vector<cl_handle> bab;   // per layer: a[C], b[C]
    cl_handle cur = bA, nxt = bB;

    for (int l = 0; l < kLayers; ++l) {
        // the convolution, in slabs of i (a long kernel trips display watchdogs)
        cl_handle k = l ? kc24 : kc1;
        cl_handle src_buf = l ? cur : bin;
        cl_handle dst_buf = l ? nxt : cur;
        const int d = W.dil[l];
        cl.SetKernelArg(k, 0, sizeof(cl_handle), &src_buf);
        cl.SetKernelArg(k, 1, sizeof(cl_handle), &dst_buf);
        cl.SetKernelArg(k, 2, sizeof(cl_handle), &bw[l]);
        cl.SetKernelArg(k, 3, sizeof(int), &d);
        constexpr int kSlab = 32;

        for (int i0 = 0; i0 < kN; i0 += kSlab) {
            const size_t off[3] = {0, 0, static_cast<size_t>(i0)};
            const size_t gws[3] = {l ? 64u : static_cast<size_t>(kN), kN, kSlab}, lws[3] = {64, 1, 1};

            if ((err = cl.EnqueueNDRangeKernel(S.queue, k, 3, off, gws, lws, 0, nullptr, nullptr)) != 0) {
                return fail("the convolution");
            }
        }

        // the channel statistics over the whole volume, summed here in double
        cl.SetKernelArg(kst, 0, sizeof(cl_handle), &dst_buf);
        cl.SetKernelArg(kst, 1, sizeof(cl_handle), &bpart);
        cl.SetKernelArg(kst, 2, kStatLocal * sizeof(float), nullptr);
        {
            const size_t gws = kStatGroups * kStatLocal, lws = kStatLocal;

            if ((err = cl.EnqueueNDRangeKernel(S.queue, kst, 1, nullptr, &gws, &lws, 0, nullptr, nullptr)) != 0) {
                return fail("the statistics");
            }
        }

        if ((err = cl.EnqueueReadBuffer(S.queue, bpart, CL_TRUE, 0, part.size() * sizeof(float), part.data(), 0,
                                        nullptr, nullptr)) != 0) {
            return fail("reading the statistics");
        }

        float ab[2 * kC];

        for (int c = 0; c < kC; ++c) {
            double s = 0, q = 0;

            for (size_t g = 0; g < kStatGroups; ++g) {
                s += part[g * 2 * kC + c];
                q += part[g * 2 * kC + kC + c];
            }

            const double mean = s / kVox, var = std::max(0.0, q / kVox - mean * mean);
            ab[c] = static_cast<float>(W.scale[l][c] / std::sqrt(var + 1e-5));
            ab[kC + c] = static_cast<float>(W.shift[l][c] - mean * ab[c]);
        }

        cl_handle b = S.buffer(CL_MEM_READ_ONLY, sizeof(ab), ab, err);

        if (!b) {
            return fail("clCreateBuffer");
        }

        cl.SetKernelArg(kng, 0, sizeof(cl_handle), &dst_buf);
        cl.SetKernelArg(kng, 1, sizeof(cl_handle), &b);
        {
            const size_t gws = kVox * kC, lws = 256;

            if ((err = cl.EnqueueNDRangeKernel(S.queue, kng, 1, nullptr, &gws, &lws, 0, nullptr, nullptr)) != 0) {
                return fail("the normalization");
            }
        }

        if (l) {
            std::swap(cur, nxt);
        }
    }

    cl.SetKernelArg(khd, 0, sizeof(cl_handle), &cur);
    cl.SetKernelArg(khd, 1, sizeof(cl_handle), &bfk);
    cl.SetKernelArg(khd, 2, sizeof(cl_handle), &bfb);
    cl.SetKernelArg(khd, 3, sizeof(cl_handle), &blab);
    {
        const size_t gws = kVox, lws = 256;

        if ((err = cl.EnqueueNDRangeKernel(S.queue, khd, 1, nullptr, &gws, &lws, 0, nullptr, nullptr)) != 0) {
            return fail("the classifier");
        }
    }

    if ((err = cl.EnqueueReadBuffer(S.queue, blab, CL_TRUE, 0, kVox, lab, 0, nullptr, nullptr)) != 0) {
        return fail("reading the labels");
    }

    cl.Finish(S.queue);
    info = name;
    return true;
}

}  // namespace siam
