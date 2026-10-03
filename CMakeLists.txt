cmake_minimum_required(VERSION 3.24)
project(RocketSim-CUDA LANGUAGES CXX CUDA)

# C++20 and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# CUDA Architectures (GEMINI.md: Ampere sm_80, Ada sm_89, Hopper sm_90, Blackwell)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

# Strict IEEE-754 Numerical Precision Invariants (GEMINI.md Section 2.3)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20 /utf-8 /permissive- /Zc:__cplusplus /EHsc")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --prec-div=true --prec-sqrt=true -ftz=false -allow-unsupported-compiler -Xcompiler=/std:c++20,/utf-8,/permissive-,/Zc:__cplusplus,/EHsc")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --prec-div=true --prec-sqrt=true -ftz=false")
endif()

# -----------------------------------------------------------------------------
# Target 1: rocketsim_cpu (Ground Truth Oracle Library)
# -----------------------------------------------------------------------------
set(ROCKETSIM_CPU_SOURCES
    src/BulletLink.cpp
    src/CollisionMeshFile/CollisionMeshFile.cpp
    src/Math/Math.cpp
    src/Math/MathTypes/MathTypes.cpp
    src/RocketSim.cpp
    src/Sim/Arena/Arena.cpp
    src/Sim/Arena/ArenaConfig/ArenaConfig.cpp
    src/Sim/Arena/DropshotTiles/DropshotTiles.cpp
    src/Sim/Ball/Ball.cpp
    src/Sim/BallHitInfo/BallHitInfo.cpp
    src/Sim/BallPredTracker/BallPredTracker.cpp
    src/Sim/BoostPad/BoostPad.cpp
    src/Sim/BoostPad/BoostPadGrid/BoostPadGrid.cpp
    src/Sim/Car/Car.cpp
    src/Sim/Car/CarConfig/CarConfig.cpp
    src/Sim/GameEventTracker/GameEventTracker.cpp
    src/Sim/MutatorConfig/MutatorConfig.cpp
    src/Sim/PhysState/PhysState.cpp
    src/Sim/btVehicleRL/btVehicleRL.cpp
)
file(GLOB_RECURSE BULLET3_SOURCES "libsrc/bullet3-3.24/*.cpp")

add_library(rocketsim_cpu STATIC ${ROCKETSIM_CPU_SOURCES} ${BULLET3_SOURCES})
add_library(RocketSim::CPU ALIAS rocketsim_cpu)

target_include_directories(rocketsim_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libsrc/bullet3-3.24>
)

# -----------------------------------------------------------------------------
# Target 2: rocketsim_cuda (GPU Physics & SoA Memory Engine)
# -----------------------------------------------------------------------------
set(ROCKETSIM_CUDA_SOURCES
    src/cuda/sim_context.cu
)

add_library(rocketsim_cuda STATIC ${ROCKETSIM_CUDA_SOURCES})
add_library(RocketSim::CUDA ALIAS rocketsim_cuda)

target_include_directories(rocketsim_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CUDAToolkit_ROOT}/include>
)
target_link_libraries(rocketsim_cuda PUBLIC
    "${CUDAToolkit_ROOT}/lib/x64/cudart.lib"
)

# -----------------------------------------------------------------------------
# Target 3: differential_harness (Differential Testing & Golden Master)
# -----------------------------------------------------------------------------
add_executable(differential_harness
    tests/differential/harness_main.cpp
    tests/differential/cpu_ref_sim.cpp
    tests/differential/golden_master.cpp
)

target_include_directories(differential_harness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/differential
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/libsrc/bullet3-3.24
)

target_link_libraries(differential_harness PRIVATE
    rocketsim_cpu
    rocketsim_cuda
)

# -----------------------------------------------------------------------------
# Unit Tests & Benchmarks
# -----------------------------------------------------------------------------
add_executable(test_math tests/unit/test_math.cu)
target_include_directories(test_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_math PRIVATE rocketsim_cuda)

add_executable(test_soa tests/unit/test_soa.cu)
target_include_directories(test_soa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_soa PRIVATE rocketsim_cuda)

add_executable(test_sdf tests/unit/test_sdf.cu)
target_include_directories(test_sdf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_sdf PRIVATE rocketsim_cuda)

add_executable(throughput_benchmark benchmarks/throughput_benchmark.cu)
target_include_directories(throughput_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(throughput_benchmark PRIVATE rocketsim_cuda)

# CTest Integration
enable_testing()
add_test(NAME TestMath COMMAND test_math)
add_test(NAME TestSoA COMMAND test_soa)
add_test(NAME TestSdf COMMAND test_sdf)
add_test(NAME DifferentialHarness COMMAND differential_harness --ticks 100 --envs 2)