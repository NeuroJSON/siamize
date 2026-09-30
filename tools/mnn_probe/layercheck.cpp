// layercheck: per-op correctness + timing check of an .mnn model on a GPU
// backend (Vulkan or OpenCL) against MNN's CPU backend, linked against the
// SAME libMNN siamize uses (third_party/mnn). The pip MNN wheel cannot test
// the NeuroJSON/MNN fork's native Conv3D/Deconv3D paths; this can.
//
// Runs the model on CPU and on the GPU with an identical random input (N(0,1),
// seed 0), compares EVERY op output in execution order, and reports the first
// op whose error exceeds the threshold (relative to max(|ref|, 1e-5)), the
// worst error per op type, and argmax agreement of the final logits.
//
// Build (from the siamize root, after scripts/fetch_mnn.sh):
//   g++ -O2 -std=c++17 tools/mnn_probe/layercheck.cpp -Ithird_party/mnn/include
//       -Lthird_party/mnn/lib -lMNN -Wl,-rpath,$PWD/third_party/mnn/lib -o layercheck
//   (one line)
//
// Usage:
//   layercheck model.mnn D H W [vulkan|opencl] [high|low|normal] [threshold]
//   D/H/W must be multiples of 64 for SIAM v0.3 (6 downsampling stages); use
//   >= 128 so the deepest InstanceNorms are not over 1-8 voxels.
//
// Environment:
//   LC_TIMEONLY=1      skip the CPU reference; time two GPU runs, split into
//                      compute and output readback
//   LC_MODE=0x..       GPU mode bits (ScheduleConfig::numThread); Vulkan accepts
//                      0x1 NONE / 0x2 HEAVY / 0x4 WIDE (| 0x200 RECORD_BATCH)
//   MNN_VULKAN_DEVICE  Vulkan physical-device index (vulkaninfo --summary)
//   MNN_VK_CONV3D_WTILE / _OCTILE / _LOCAL   Vulkan Conv3D tiling overrides
//
// Build MNN with -DMNN_GPU_TIME_PROFILE=ON to get per-layer GPU times
// (Conv3D layers are labelled with channels / size / layouts).
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace MNN;

struct Rec {
    std::string name, type;
    std::vector<int> shape;
    std::vector<float> data;
};

static std::vector<Rec> run(Interpreter* net, MNNForwardType type, BackendConfig::PrecisionMode prec,
                            const std::vector<int>& shape, const std::vector<float>& input, bool keep,
                            const std::vector<Rec>* ref, double thr, double* seconds) {
    ScheduleConfig cfg;
    cfg.type = type;
    cfg.numThread = (type == MNN_FORWARD_CPU) ? 16 : MNN_GPU_TUNING_NONE;
    if (type == MNN_FORWARD_OPENCL) cfg.numThread = MNN_GPU_TUNING_FAST | MNN_GPU_MEMORY_BUFFER;
    if (type != MNN_FORWARD_CPU && getenv("LC_MODE")) cfg.numThread = (int)strtol(getenv("LC_MODE"), nullptr, 0);
    BackendConfig bc;
    bc.precision = prec;
    cfg.backendConfig = &bc;
    auto session = net->createSession(cfg);
    auto in = net->getSessionInput(session, nullptr);
    net->resizeTensor(in, shape);
    net->resizeSession(session);
    {
        int status = 0;
        net->getSessionInfo(session, Interpreter::RESIZE_STATUS, &status);
        if (status != 0) {
            printf("  resizeSession failed (RESIZE_STATUS=%d)\n", status);
            exit(3);
        }
    }
    Tensor host(in, Tensor::CAFFE);
    ::memcpy(host.host<float>(), input.data(), input.size() * sizeof(float));
    in->copyFromHostTensor(&host);

    std::vector<Rec> recs;
    size_t idx = 0;
    bool reported = false;
    std::map<std::string, std::pair<int, double>> worstByType;
    auto before = [](const std::vector<Tensor*>&, const OperatorInfo*) { return true; };
    auto after = [&](const std::vector<Tensor*>& outs, const OperatorInfo* info) {
        for (size_t o = 0; o < outs.size(); ++o) {
            auto t = outs[o];
            if (t->getType().code != halide_type_float || t->elementSize() <= 0) continue;
            Tensor h(t, Tensor::CAFFE);
            t->copyToHostTensor(&h);
            Rec r;
            r.name = info->name() + "#" + std::to_string(o);
            r.type = info->type();
            r.shape = t->shape();
            r.data.assign(h.host<float>(), h.host<float>() + h.elementSize());
            if (ref) {
                if (idx >= ref->size() || (*ref)[idx].name != r.name) {
                    if (!reported) printf("  [order mismatch at %zu: %s]\n", idx, r.name.c_str());
                    reported = true;
                    ++idx;
                    continue;
                }
                const Rec& e = (*ref)[idx];
                double maxRef = 0, maxDiff = 0;
                size_t nan = 0;
                for (size_t i = 0; i < e.data.size() && i < r.data.size(); ++i) {
                    maxRef = std::max(maxRef, (double)std::fabs(e.data[i]));
                    if (!std::isfinite(r.data[i])) { ++nan; continue; }
                    maxDiff = std::max(maxDiff, (double)std::fabs(e.data[i] - r.data[i]));
                }
                double rel = (e.data.size() != r.data.size() || nan) ? 1e9 : maxDiff / std::max(maxRef, 1e-5);
                auto& w = worstByType[r.type];
                w.first++;
                w.second = std::max(w.second, rel);
                if (rel > thr && !reported) {
                    printf("  FIRST DIVERGENCE op %zu %s [%s] shape(", idx, r.name.c_str(), r.type.c_str());
                    for (auto s : r.shape) printf("%d,", s);
                    printf(") rel=%.3e maxDiff=%.3e maxRef=%.3e nonfinite=%zu\n", rel, maxDiff, maxRef, nan);
                    reported = true;
                }
            }
            if (keep) recs.emplace_back(std::move(r));
            ++idx;
        }
        return true;
    };
    if (!getenv("LC_TIMEONLY")) net->runSessionWithCallBackInfo(session, before, after);  // warm-up + record
    if (ref && !ref->empty()) {
        auto out = net->getSessionOutput(session, nullptr);
        Tensor oh(out, Tensor::CAFFE);
        out->copyToHostTensor(&oh);
        const Rec& e = ref->back();
        int C = out->length(1);
        size_t S = oh.elementSize() / C, agree = 0;
        double maxDiff = 0;
        for (size_t s = 0; s < S; ++s) {
            int a = 0, b = 0;
            for (int c = 1; c < C; ++c) {
                if (e.data[c * S + s] > e.data[a * S + s]) a = c;
                if (oh.host<float>()[c * S + s] > oh.host<float>()[b * S + s]) b = c;
            }
            agree += (a == b);
        }
        for (size_t i = 0; i < e.data.size(); ++i) maxDiff = std::max(maxDiff, (double)std::fabs(e.data[i] - oh.host<float>()[i]));
        printf("  final logits: max|diff|=%.3e  argmax agreement %.4f%% (%zu voxels)\n", maxDiff, 100.0 * agree / S, S);
    }
    if (ref) {
        printf("  per-op-type worst relative error:\n");
        for (auto& kv : worstByType) printf("    %-18s n=%-4d worst=%.3e\n", kv.first.c_str(), kv.second.first, kv.second.second);
        if (!reported) printf("  all %zu op outputs within rel %.1e\n", idx, thr);
    }
    // Timed plain run (no callbacks, no per-op readback).
    if (seconds) {
        net->runSession(session);
        auto out = net->getSessionOutput(session, nullptr);
        Tensor oh(out, Tensor::CAFFE);
        out->copyToHostTensor(&oh);
        auto t0 = std::chrono::steady_clock::now();
        net->runSession(session);
        out->wait(Tensor::MAP_TENSOR_READ, true);
        auto t1 = std::chrono::steady_clock::now();
        out->copyToHostTensor(&oh);
        auto t2 = std::chrono::steady_clock::now();
        *seconds = std::chrono::duration<double>(t2 - t0).count();
        printf("  [split] compute %.3f s  readback %.3f s (%.1f MB)\n", std::chrono::duration<double>(t1 - t0).count(),
               std::chrono::duration<double>(t2 - t1).count(), oh.size() / 1e6);
    }
    net->releaseSession(session);
    return recs;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.mnn D H W [vulkan|opencl] [high|low|normal] [thr]\n", argv[0]);
        return 1;
    }
    setenv("SIAM_DISABLE_GEOM_CONV3D", "1", 0);
    setenv("SIAM_DISABLE_GEOM_DECONV3D", "1", 0);
    std::vector<int> shape{1, 1, atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
    MNNForwardType gpu = (argc > 5 && !strcmp(argv[5], "opencl")) ? MNN_FORWARD_OPENCL : MNN_FORWARD_VULKAN;
    auto prec = BackendConfig::Precision_High;
    if (argc > 6 && !strcmp(argv[6], "low")) prec = BackendConfig::Precision_Low;
    if (argc > 6 && !strcmp(argv[6], "normal")) prec = BackendConfig::Precision_Normal;
    double thr = argc > 7 ? atof(argv[7]) : 1e-3;

    std::unique_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    net->setSessionMode(Interpreter::Session_Debug);
    size_t n = (size_t)shape[2] * shape[3] * shape[4];
    std::vector<float> input(n);
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : input) v = nd(rng);

    double tc = 0, tg = 0;
    if (getenv("LC_TIMEONLY")) {
        for (int rep = 0; rep < 2; ++rep) {
            run(net.get(), gpu, prec, shape, input, false, nullptr, thr, &tg);
            printf("timed %.3f s\n", tg);
        }
        return 0;
    }
    printf("CPU reference...\n");
    auto ref = run(net.get(), MNN_FORWARD_CPU, BackendConfig::Precision_High, shape, input, true, nullptr, thr, &tc);
    printf("  %zu op outputs recorded, timed run %.3f s\n", ref.size(), tc);
    printf("%s (%s)...\n", gpu == MNN_FORWARD_VULKAN ? "Vulkan" : "OpenCL", argc > 6 ? argv[6] : "high");
    run(net.get(), gpu, prec, shape, input, false, &ref, thr, &tg);
    printf("  timed run %.3f s (CPU %.3f s)\n", tg, tc);
    return 0;
}
