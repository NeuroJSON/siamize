// clbench: standalone OpenCL micro-benchmark for the SIAM Conv3D kernels.
//
// Compiles an OpenCL file with MNN's fp32 build options, runs one kernel on
// every SIAM v0.3 Conv3D shape (192x192x128 tile) over a list of work-group
// sizes, checks each result against conv_3d_buf_nc4dhw4 (reference) and
// reports the best time per shape plus a per-tile estimate weighted by how
// often each shape occurs. Optionally dumps the program binary (PTX on NVIDIA)
// for register / load-width inspection (e.g. `ptxas -v`).
//
// Build (from the siamize root, after scripts/fetch_mnn.sh):
//   g++ -O2 -std=c++17 tools/mnn_probe/clbench.cpp
//       -Ithird_party/mnn-build/MNN-<ref>/3rd_party/OpenCLHeaders -lOpenCL -o clbench
// Run next to a copy of the fork's conv_3d_buf.cl (or set CLBENCH_REF=path):
//   clbench conv_3d_buf.cl conv_3d_buf_opt "-DTW=4 -DTH=2 -DTOC=1" 4 2 1
//           'lws=auto;16x1x8;8x2x8;8x4x4' [--ptx out.ptx] [--shapes 0,2]
// The kernel must use the conv_3d_buf_nc4dhw4 argument list; its global size
// is (ceil(Wout/TW), Dout*ceil(Hout/TH), ceil(Cout4/TOC)).
// CLBENCH_BUILD_ONLY=1 only compiles (and dumps) the file.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)

static const char* kFP32 =
    " -DFLOAT=float -DFLOAT2=float2 -DFLOAT3=float3 -DFLOAT4=float4 -DFLOAT8=float8 -DFLOAT16=float16"
    " -DCOMPUTE_FLOAT=float -DCOMPUTE_FLOAT2=float2 -DCOMPUTE_FLOAT3=float3 -DCOMPUTE_FLOAT4=float4"
    " -DCOMPUTE_FLOAT8=float8 -DCOMPUTE_FLOAT16=float16 -DCONVERT_COMPUTE_FLOAT=convert_float"
    " -DCONVERT_COMPUTE_FLOAT2=convert_float2 -DCONVERT_COMPUTE_FLOAT3=convert_float3"
    " -DCONVERT_COMPUTE_FLOAT4=convert_float4 -DCONVERT_COMPUTE_FLOAT8=convert_float8"
    " -DCONVERT_COMPUTE_FLOAT16=convert_float16 -DCONVERT_FLOAT=convert_float -DCONVERT_FLOAT2=convert_float2"
    " -DCONVERT_FLOAT3=convert_float3 -DCONVERT_FLOAT4=convert_float4 -DCONVERT_FLOAT8=convert_float8"
    " -DCONVERT_FLOAT16=convert_float16 -cl-mad-enable -w";

struct Shape { const char* tag; int ci, co, d, h, w, k, s, perTile; };
// SIAM v0.3 Conv3D shapes at a 192x192x128 tile (output size; stride-1 3x3x3, pad 1),
// with how many times each occurs per tile (from MNN_CONV3D_TIMING).
static const Shape kShapes[] = {
    {"32-32@192",   32,  32, 192, 192, 128, 3, 1, 2},
    {"64-32@192",   64,  32, 192, 192, 128, 3, 1, 1},
    {"64-64@96",    64,  64,  96,  96,  64, 3, 1, 5},
    {"128-128@48", 128, 128,  48,  48,  32, 3, 1, 7},
    {"256-256@24", 256, 256,  24,  24,  16, 3, 1, 11},
    {"320-320@12", 320, 320,  12,  12,   8, 3, 1, 11},
    {"320-320@6",  320, 320,   6,   6,   4, 3, 1, 11},
    {"320-320@3",  320, 320,   3,   3,   4, 3, 1, 11},
};

static std::string readFile(const char* p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }

static cl_program build(cl_context ctx, cl_device_id dev, const std::string& src, const std::string& opts, const char* ptxOut) {
    const char* s = src.c_str(); size_t n = src.size(); cl_int e;
    cl_program p = clCreateProgramWithSource(ctx, 1, &s, &n, &e); CK(e);
    if (clBuildProgram(p, 1, &dev, opts.c_str(), nullptr, nullptr) != CL_SUCCESS) {
        size_t ls; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ls);
        std::string log(ls, 0); clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, ls, &log[0], nullptr);
        fprintf(stderr, "build failed:\n%s\n", log.c_str()); exit(1);
    }
    if (ptxOut) {
        size_t bs; CK(clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(bs), &bs, nullptr));
        std::vector<unsigned char> bin(bs); unsigned char* bp = bin.data();
        CK(clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(bp), &bp, nullptr));
        std::ofstream(ptxOut, std::ios::binary).write((const char*)bin.data(), bs);
    }
    return p;
}

static double runKernel(cl_command_queue q, cl_kernel k, const Shape& S, int TW, int TH, int TOC, const size_t* lws,
                        cl_mem in, cl_mem wt, cl_mem bias, cl_mem out, int reps) {
    int cib = (S.ci + 3) / 4, cob = (S.co + 3) / 4;
    int ind = S.d * S.s, inh = S.h * S.s, inw = S.w * S.s;  // same-size input for stride 1
    size_t g[3] = {(size_t)((S.w + TW - 1) / TW), (size_t)(S.d * ((S.h + TH - 1) / TH)), (size_t)((cob + TOC - 1) / TOC)};
    size_t gr[3];
    for (int i = 0; i < 3; ++i) gr[i] = lws ? ((g[i] + lws[i] - 1) / lws[i]) * lws[i] : g[i];
    int a = 0, p = S.k / 2;
    int gs0 = (int)g[0], gs1 = (int)g[1], gs2 = (int)g[2];
    int args[] = {cib, ind, inh, inw, S.d, S.h, S.w, S.k, S.k, S.k, S.s, S.s, S.s, p, p, p};
    CK(clSetKernelArg(k, a++, sizeof(int), &gs0)); CK(clSetKernelArg(k, a++, sizeof(int), &gs1)); CK(clSetKernelArg(k, a++, sizeof(int), &gs2));
    CK(clSetKernelArg(k, a++, sizeof(cl_mem), &in)); CK(clSetKernelArg(k, a++, sizeof(cl_mem), &wt));
    CK(clSetKernelArg(k, a++, sizeof(cl_mem), &bias)); CK(clSetKernelArg(k, a++, sizeof(cl_mem), &out));
    for (int v : args) CK(clSetKernelArg(k, a++, sizeof(int), &v));
    std::vector<double> t;
    for (int r = 0; r < reps + 2; ++r) {
        cl_event ev;
        cl_int e = clEnqueueNDRangeKernel(q, k, 3, nullptr, gr, lws, 0, nullptr, &ev);
        if (e != CL_SUCCESS) return -1;  // e.g. LWS too large for this kernel
        CK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        if (r >= 2) t.push_back((t1 - t0) * 1e-6);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: clbench kernel.cl name \"opts\" TW TH TOC [lws=x,y,z;...] [--ptx f] [--shapes i,j]\n"); return 1; }
    std::string kfile = argv[1], kname = argv[2], extra = argv[3];
    int TW = atoi(argv[4]), TH = atoi(argv[5]), TOC = atoi(argv[6]);
    std::vector<std::vector<size_t>> lwsList = {{}};
    const char* ptx = nullptr; std::vector<int> shapeSel;
    for (int i = 7; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ptx" && i + 1 < argc) { ptx = argv[++i]; continue; }
        if (a == "--shapes" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) shapeSel.push_back(atoi(t.c_str())); continue; }
        if (a.rfind("lws=", 0) == 0) {
            lwsList.clear(); std::stringstream ss(a.substr(4)); std::string item;
            while (std::getline(ss, item, ';')) {
                if (item == "auto") { lwsList.push_back({}); continue; }
                std::vector<size_t> v; std::stringstream s2(item); std::string n;
                while (std::getline(s2, n, 'x')) v.push_back(atoi(n.c_str()));
                lwsList.push_back(v);
            }
        }
    }
    cl_platform_id plats[8]; cl_uint np; CK(clGetPlatformIDs(8, plats, &np));
    cl_device_id dev = nullptr;
    for (cl_uint i = 0; i < np && !dev; ++i) {
        char vend[256]; clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, sizeof(vend), vend, nullptr);
        if (strstr(vend, "NVIDIA")) clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    }
    if (!dev) { fprintf(stderr, "no NVIDIA OpenCL GPU\n"); return 1; }
    char dname[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    cl_int e; cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e); CK(e);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e); CK(e);

    if (getenv("CLBENCH_BUILD_ONLY")) {  // compile + dump PTX, no run
        build(ctx, dev, readFile(kfile.c_str()), std::string(kFP32) + " " + extra, ptx);
        return 0;
    }
    std::string refSrc = readFile(getenv("CLBENCH_REF") ? getenv("CLBENCH_REF") : "conv_3d_buf.cl");
    cl_program refP = build(ctx, dev, refSrc, kFP32, nullptr);
    cl_kernel refK = clCreateKernel(refP, "conv_3d_buf_nc4dhw4", &e); CK(e);
    cl_program p = build(ctx, dev, readFile(kfile.c_str()), std::string(kFP32) + " " + extra, ptx);
    cl_kernel k = clCreateKernel(p, kname.c_str(), &e); CK(e);
    size_t kwg, pm; cl_ulong priv, loc;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kwg), &kwg, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(pm), &pm, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(loc), &loc, nullptr);
    printf("%s | %s %s TW=%d TH=%d TOC=%d | max WG %zu, private %llu B, local %llu B\n", dname, kname.c_str(), extra.c_str(),
           TW, TH, TOC, kwg, (unsigned long long)priv, (unsigned long long)loc);

    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 1.f);
    double tileMs = 0; bool allOk = true;
    int nshapes = sizeof(kShapes) / sizeof(kShapes[0]);
    for (int si = 0; si < nshapes; ++si) {
        if (!shapeSel.empty() && std::find(shapeSel.begin(), shapeSel.end(), si) == shapeSel.end()) continue;
        const Shape& S = kShapes[si];
        int cib = (S.ci + 3) / 4, cob = (S.co + 3) / 4;
        size_t inN = (size_t)cib * S.d * S.h * S.w * 4, outN = (size_t)cob * S.d * S.h * S.w * 4;
        size_t wN = (size_t)cob * S.k * S.k * S.k * cib * 16;
        std::vector<float> hin(inN), hw(wN), hb(cob * 4);
        for (auto& v : hin) v = nd(rng);
        for (auto& v : hw) v = nd(rng) * 0.05f;
        for (auto& v : hb) v = nd(rng);
        cl_mem in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inN * 4, hin.data(), &e); CK(e);
        cl_mem wt = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wN * 4, hw.data(), &e); CK(e);
        cl_mem bs = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hb.size() * 4, hb.data(), &e); CK(e);
        cl_mem ref = clCreateBuffer(ctx, CL_MEM_READ_WRITE, outN * 4, nullptr, &e); CK(e);
        cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, outN * 4, nullptr, &e); CK(e);
        size_t refL[3] = {16, 4, 1};
        double refMs = runKernel(q, refK, S, 1, 1, 1, refL, in, wt, bs, ref, 1);
        std::vector<float> hr(outN), ho(outN);
        CK(clEnqueueReadBuffer(q, ref, CL_TRUE, 0, outN * 4, hr.data(), 0, nullptr, nullptr));
        double best = 1e30; std::string bestL = "?";
        double flop = 2.0 * S.ci * S.co * S.k * S.k * S.k * S.d * S.h * S.w;
        for (auto& L : lwsList) {
            float fill = NAN; CK(clEnqueueFillBuffer(q, out, &fill, 4, 0, outN * 4, 0, nullptr, nullptr));
            double ms = runKernel(q, k, S, TW, TH, TOC, L.empty() ? nullptr : L.data(), in, wt, bs, out, 5);
            if (ms < 0) continue;
            CK(clEnqueueReadBuffer(q, out, CL_TRUE, 0, outN * 4, ho.data(), 0, nullptr, nullptr));
            double md = 0, mr = 0;
            for (size_t i = 0; i < outN; ++i) { md = std::max(md, (double)std::fabs(ho[i] - hr[i])); mr = std::max(mr, (double)std::fabs(hr[i])); }
            bool ok = std::isfinite(md) && md <= 1e-4 * std::max(mr, 1.0);
            if (!ok) { allOk = false; printf("   %-11s lws=%s WRONG maxdiff=%g\n", S.tag, L.empty() ? "auto" : "set", md); continue; }
            if (ms < best) { best = ms; char b[64]; if (L.empty()) snprintf(b, 64, "auto"); else snprintf(b, 64, "%zux%zux%zu", L[0], L[1], L[2]); bestL = b; }
        }
        tileMs += best * S.perTile;
        printf("   %-11s x%-2d ref %7.3f ms | best %7.3f ms (%5.1f TF/s, lws %s) %4.2fx\n", S.tag, S.perTile, refMs, best,
               flop / best / 1e9, bestL.c_str(), refMs / best);
        clReleaseMemObject(in); clReleaseMemObject(wt); clReleaseMemObject(bs); clReleaseMemObject(ref); clReleaseMemObject(out);
    }
    printf("   weighted per-tile Conv3D (these shapes): %.1f ms %s\n", tileMs, allOk ? "" : "(SOME WRONG)");
    return allOk ? 0 : 2;
}
