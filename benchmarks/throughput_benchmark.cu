#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"

using namespace rocketsim_cuda;

int main(int argc, char** argv) {
    uint32_t envs = 32768;
    uint32_t steps = 1000;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--envs" && i + 1 < argc) {
            envs = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        }
    }

    std::cout << "======================================================================\n"
              << "                 ROCKETSIM-CUDA THROUGHPUT BENCHMARK                  \n"
              << "======================================================================\n"
              << "  Environments: " << envs << "\n"
              << "  Steps:        " << steps << "\n"
              << "======================================================================\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    SimContext ctx(envs, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[+] SimContext allocation & init: " << init_ms << " ms ("
              << (ctx.GetAllocatedBytes() / (1024.0 * 1024.0)) << " MB)\n";

    // Warm-up
    ctx.ResetToDefault();
    cudaDeviceSynchronize();

    std::cout << "[+] Benchmark ready (Milestone 1 memory pipeline verified)\n";
    return 0;
}
