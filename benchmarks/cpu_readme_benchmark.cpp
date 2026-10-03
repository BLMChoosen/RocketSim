#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <thread>
#include <iomanip>

#include "RocketSim.h"
#include "Sim/Arena/Arena.h"
#include "Sim/Car/Car.h"
#include "Sim/Ball/Ball.h"
#include "RLConst.h"

int main(int argc, char** argv) {
    std::cout << "======================================================================\n";
    std::cout << "          ROCKETSIM ORIGINAL (CPU) - 2v2 REPLICATION BENCHMARK        \n";
    std::cout << "======================================================================\n";

    // Initialize RocketSim
    RocketSim::Init("", true);

    const int total_ticks_target = 100000; // 100k ticks for quick accurate profiling
    
    // Benchmark 1: Single-Thread (exact match to README specification)
    std::cout << "\n[1/2] Running Single-Thread Benchmark (1 Arena, 2v2 - 4 cars)...\n";
    {
        RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::THE_VOID);
        
        // Add 2 cars blue, 2 cars orange (2v2)
        std::vector<RocketSim::Car*> cars;
        cars.push_back(arena->AddCar(RocketSim::Team::BLUE));
        cars.push_back(arena->AddCar(RocketSim::Team::BLUE));
        cars.push_back(arena->AddCar(RocketSim::Team::ORANGE));
        cars.push_back(arena->AddCar(RocketSim::Team::ORANGE));

        for (int i = 0; i < 4; i++) {
            cars[i]->Respawn(RocketSim::GameMode::THE_VOID, i, RocketSim::RLConst::BOOST_SPAWN_AMOUNT);
        }

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist_signed(-1.0f, 1.0f);
        std::uniform_int_distribution<int> dist_interval(2, 60);

        int change_counter[4] = {0, 0, 0, 0};

        // Warm-up 1000 ticks
        for (int t = 0; t < 1000; t++) {
            arena->Step(1);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < total_ticks_target; t++) {
            for (int c = 0; c < 4; c++) {
                if (--change_counter[c] <= 0) {
                    cars[c]->controls.throttle = dist_signed(rng);
                    cars[c]->controls.steer = dist_signed(rng);
                    cars[c]->controls.pitch = dist_signed(rng);
                    cars[c]->controls.yaw = dist_signed(rng);
                    cars[c]->controls.roll = dist_signed(rng);
                    cars[c]->controls.boost = (dist_signed(rng) > 0.4f);
                    cars[c]->controls.jump = (dist_signed(rng) > 0.6f);
                    cars[c]->controls.handbrake = (dist_signed(rng) > 0.8f);
                    cars[c]->controls.ClampFix();
                    change_counter[c] = dist_interval(rng);
                }
            }
            arena->Step(1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double tps = total_ticks_target / elapsed_sec;

        std::cout << "    Ticks simulated: " << total_ticks_target << "\n";
        std::cout << "    Elapsed time:    " << (elapsed_sec * 1000.0) << " ms\n";
        std::cout << "    Single-Thread Throughput: " << std::fixed << std::setprecision(0) << tps << " TPS\n";

        delete arena;
    }

    // Benchmark 2: Multi-Thread (12 threads = 12 logical cores on Ryzen 5 5500)
    const unsigned int num_threads = 12;
    std::cout << "\n[2/2] Running Multi-Threaded Benchmark (" << num_threads << " threads / arenas, 2v2)...\n";
    {
        const int ticks_per_thread = total_ticks_target;
        std::vector<std::thread> workers;
        std::vector<double> thread_tps(num_threads, 0.0);

        auto start = std::chrono::high_resolution_clock::now();

        for (unsigned int th = 0; th < num_threads; th++) {
            workers.emplace_back([th, ticks_per_thread, &thread_tps]() {
                RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::THE_VOID);
                std::vector<RocketSim::Car*> cars;
                cars.push_back(arena->AddCar(RocketSim::Team::BLUE));
                cars.push_back(arena->AddCar(RocketSim::Team::BLUE));
                cars.push_back(arena->AddCar(RocketSim::Team::ORANGE));
                cars.push_back(arena->AddCar(RocketSim::Team::ORANGE));

                for (int i = 0; i < 4; i++) {
                    cars[i]->Respawn(RocketSim::GameMode::THE_VOID, th * 10 + i, RocketSim::RLConst::BOOST_SPAWN_AMOUNT);
                }

                std::mt19937 rng(42 + th);
                std::uniform_real_distribution<float> dist_signed(-1.0f, 1.0f);
                std::uniform_int_distribution<int> dist_interval(2, 60);
                int change_counter[4] = {0, 0, 0, 0};

                for (int t = 0; t < 1000; t++) arena->Step(1);

                auto t_start = std::chrono::high_resolution_clock::now();
                for (int t = 0; t < ticks_per_thread; t++) {
                    for (int c = 0; c < 4; c++) {
                        if (--change_counter[c] <= 0) {
                            cars[c]->controls.throttle = dist_signed(rng);
                            cars[c]->controls.steer = dist_signed(rng);
                            cars[c]->controls.pitch = dist_signed(rng);
                            cars[c]->controls.yaw = dist_signed(rng);
                            cars[c]->controls.roll = dist_signed(rng);
                            cars[c]->controls.boost = (dist_signed(rng) > 0.4f);
                            cars[c]->controls.jump = (dist_signed(rng) > 0.6f);
                            cars[c]->controls.handbrake = (dist_signed(rng) > 0.8f);
                            cars[c]->controls.ClampFix();
                            change_counter[c] = dist_interval(rng);
                        }
                    }
                    arena->Step(1);
                }
                auto t_end = std::chrono::high_resolution_clock::now();
                double el = std::chrono::duration<double>(t_end - t_start).count();
                thread_tps[th] = ticks_per_thread / el;
                delete arena;
            });
        }

        for (auto& w : workers) {
            w.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double total_tps = (ticks_per_thread * num_threads) / elapsed_sec;

        std::cout << "    Total Ticks simulated: " << (ticks_per_thread * num_threads) << "\n";
        std::cout << "    Elapsed time:          " << (elapsed_sec * 1000.0) << " ms\n";
        std::cout << "    Multi-Thread (12 threads) Throughput: " << std::fixed << std::setprecision(0) << total_tps << " TPS\n";
    }

    return 0;
}
