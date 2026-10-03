#include <iostream>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"

using namespace rocketsim_cuda;

struct SdfTestResult {
    float dist_floor;
    Vec3 normal_floor;

    float dist_wall_x;
    Vec3 normal_wall_x;

    float dist_wall_y;
    Vec3 normal_wall_y;

    float dist_ceil;
    Vec3 normal_ceil;

    float dist_chamfer;
    Vec3 normal_chamfer;

    float dist_goal;
    Vec3 normal_goal;

    bool ray_hit_floor;
    float ray_dist_floor;
    Vec3 ray_normal_floor;

    bool ray_hit_ramp;
    float ray_dist_ramp;
    Vec3 ray_normal_ramp;
};

__global__ void TestSdfKernel(SdfTestResult* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // 1. Floor center test (0, 0, 100)
    arena_sdf_and_normal(Vec3(0.0f, 0.0f, 100.0f), result->dist_floor, result->normal_floor);

    // 2. Side wall test (4000, 0, 1000)
    arena_sdf_and_normal(Vec3(4000.0f, 0.0f, 1000.0f), result->dist_wall_x, result->normal_wall_x);

    // 3. Back wall test (0, 5000, 1000)
    arena_sdf_and_normal(Vec3(0.0f, 5000.0f, 1000.0f), result->dist_wall_y, result->normal_wall_y);

    // 4. Ceiling test (0, 0, 2000)
    arena_sdf_and_normal(Vec3(0.0f, 0.0f, 2000.0f), result->dist_ceil, result->normal_ceil);

    // 5. Chamfer 45-degree line test: at (3520, 4544, 1000), x + y = 8064
    // Test point 10 UU inside: (3520 - 5*sqrt(2), 4544 - 5*sqrt(2), 1000)
    arena_sdf_and_normal(Vec3(3520.0f, 4544.0f, 1000.0f), result->dist_chamfer, result->normal_chamfer);

    // 6. Goal cavity test (0, 5500, 300)
    arena_sdf_and_normal(Vec3(0.0f, 5500.0f, 300.0f), result->dist_goal, result->normal_goal);

    // 7. Raycast straight down to flat floor
    result->ray_hit_floor = raycast_arena_sdf(
        Vec3(0.0f, 0.0f, 36.0f),
        Vec3(0.0f, 0.0f, -1.0f),
        50.0f,
        &result->ray_dist_floor,
        &result->ray_normal_floor
    );

    // 8. Raycast down onto corner ramp
    result->ray_hit_ramp = raycast_arena_sdf(
        Vec3(4000.0f, 0.0f, 100.0f),
        Vec3(0.0f, 0.0f, -1.0f),
        150.0f,
        &result->ray_dist_ramp,
        &result->ray_normal_ramp
    );
}

int main() {
    std::cout << "[TestSdf] Running CUDA analytical SDF and raycast unit tests...\n";

    SdfTestResult* d_res = nullptr;
    cudaMalloc(&d_res, sizeof(SdfTestResult));

    TestSdfKernel<<<1, 1>>>(d_res);
    cudaDeviceSynchronize();

    SdfTestResult h_res;
    cudaMemcpy(&h_res, d_res, sizeof(SdfTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_res);

    // 1. Verify Floor Center
    std::cout << "  Floor (0, 0, 100): dist = " << h_res.dist_floor 
              << ", normal = (" << h_res.normal_floor.x << ", " << h_res.normal_floor.y << ", " << h_res.normal_floor.z << ")\n";
    assert(std::fabs(h_res.dist_floor - 100.0f) < 1e-3f);
    assert(std::fabs(h_res.normal_floor.z - 1.0f) < 1e-4f);

    // 2. Verify Side Wall (4000, 0, 1000) -> extent 4096 => dist 96, nx = -1
    std::cout << "  Side Wall (4000, 0, 1000): dist = " << h_res.dist_wall_x 
              << ", normal = (" << h_res.normal_wall_x.x << ", " << h_res.normal_wall_x.y << ", " << h_res.normal_wall_x.z << ")\n";
    assert(std::fabs(h_res.dist_wall_x - 96.0f) < 1e-3f);
    assert(std::fabs(h_res.normal_wall_x.x - (-1.0f)) < 1e-4f);

    // 3. Verify Back Wall (0, 5000, 1000) -> extent 5120 => dist 120, ny = -1
    std::cout << "  Back Wall (0, 5000, 1000): dist = " << h_res.dist_wall_y 
              << ", normal = (" << h_res.normal_wall_y.x << ", " << h_res.normal_wall_y.y << ", " << h_res.normal_wall_y.z << ")\n";
    assert(std::fabs(h_res.dist_wall_y - 120.0f) < 1e-3f);
    assert(std::fabs(h_res.normal_wall_y.y - (-1.0f)) < 1e-4f);

    // 4. Verify Ceiling (0, 0, 2000) -> height 2048 => dist 48, nz = -1
    std::cout << "  Ceiling (0, 0, 2000): dist = " << h_res.dist_ceil 
              << ", normal = (" << h_res.normal_ceil.x << ", " << h_res.normal_ceil.y << ", " << h_res.normal_ceil.z << ")\n";
    assert(std::fabs(h_res.dist_ceil - 48.0f) < 1e-3f);
    assert(std::fabs(h_res.normal_ceil.z - (-1.0f)) < 1e-4f);

    // 5. Verify Chamfer (3520, 4544, 1000) -> x + y = 8064 => dist = 0, nx = -sqrt(0.5), ny = -sqrt(0.5)
    std::cout << "  Chamfer (3520, 4544, 1000): dist = " << h_res.dist_chamfer 
              << ", normal = (" << h_res.normal_chamfer.x << ", " << h_res.normal_chamfer.y << ", " << h_res.normal_chamfer.z << ")\n";
    assert(std::fabs(h_res.dist_chamfer) < 1e-3f);
    assert(std::fabs(h_res.normal_chamfer.x - (-0.70710678f)) < 1e-3f);
    assert(std::fabs(h_res.normal_chamfer.y - (-0.70710678f)) < 1e-3f);

    // 6. Verify Goal Interior
    std::cout << "  Goal (0, 5500, 300): dist = " << h_res.dist_goal 
              << ", normal = (" << h_res.normal_goal.x << ", " << h_res.normal_goal.y << ", " << h_res.normal_goal.z << ")\n";
    assert(h_res.dist_goal > 0.0f);

    // 7. Verify Raycast to Floor
    std::cout << "  Raycast Floor: hit = " << h_res.ray_hit_floor << ", dist = " << h_res.ray_dist_floor << "\n";
    assert(h_res.ray_hit_floor);
    assert(std::fabs(h_res.ray_dist_floor - 36.0f) < 1e-3f);
    assert(std::fabs(h_res.ray_normal_floor.z - 1.0f) < 1e-4f);

    // 8. Verify Raycast to Ramp
    std::cout << "  Raycast Ramp: hit = " << h_res.ray_hit_ramp << ", dist = " << h_res.ray_dist_ramp 
              << ", normal = (" << h_res.ray_normal_ramp.x << ", " << h_res.ray_normal_ramp.y << ", " << h_res.ray_normal_ramp.z << ")\n";
    assert(h_res.ray_hit_ramp);
    assert(h_res.ray_dist_ramp > 0.0f && h_res.ray_dist_ramp <= 100.0f);

    std::cout << "[+] All Analytical SDF & Raycast tests passed successfully!\n";
    return 0;
}
