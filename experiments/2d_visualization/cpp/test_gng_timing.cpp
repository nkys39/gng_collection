/**
 * @file test_gng_timing.cpp
 * @brief Measure GNG computation time for different iteration counts
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>
#include "gng.hpp"

/**
 * @brief Sample points from triple ring (3 concentric rings).
 */
std::vector<Eigen::Vector2f> sample_triple_ring(
    int n_samples,
    unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);

    // Three rings with different radii
    std::vector<std::pair<float, float>> rings = {
        {0.15f, 0.20f},  // inner ring
        {0.25f, 0.30f},  // middle ring
        {0.35f, 0.40f},  // outer ring
    };

    std::vector<Eigen::Vector2f> samples;
    samples.reserve(n_samples);

    int samples_per_ring = n_samples / 3;

    for (const auto& [r_inner, r_outer] : rings) {
        std::uniform_real_distribution<float> radius_dist(r_inner, r_outer);
        for (int i = 0; i < samples_per_ring; ++i) {
            float angle = angle_dist(rng);
            float radius = radius_dist(rng);
            float x = 0.5f + radius * std::cos(angle);
            float y = 0.5f + radius * std::sin(angle);
            samples.emplace_back(x, y);
        }
    }

    return samples;
}

int main() {
    unsigned int seed = 42;
    int n_samples = 1500;

    // Generate triple ring samples
    auto samples = sample_triple_ring(n_samples, seed);

    // GNG Parameters
    gng::GNGParams params;
    params.max_nodes = 150;  // Paper: 135 nodes
    params.lambda = 100;
    params.eps_b = 0.08f;
    params.eps_n = 0.008f;
    params.alpha = 0.5f;
    params.beta = 0.005f;
    params.max_age = 100;

    std::cout << "GNG Timing Test (C++)\n";
    std::cout << "=====================\n";
    std::cout << "Samples: " << n_samples << " (triple ring)\n";
    std::cout << "max_nodes: " << params.max_nodes << "\n\n";

    // Test 1: 5,000 iterations
    {
        gng::GNG2f model(params, seed);
        model.init();

        auto start = std::chrono::high_resolution_clock::now();
        model.train(samples, 5000);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "=== 5,000 iterations ===\n";
        std::cout << "Time: " << duration.count() << " ms\n";
        std::cout << "Nodes: " << model.num_nodes() << "\n";
        std::cout << "Edges: " << model.num_edges() << "\n\n";
    }

    // Test 2: 50,000 iterations
    {
        gng::GNG2f model(params, seed);
        model.init();

        auto start = std::chrono::high_resolution_clock::now();
        model.train(samples, 50000);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "=== 50,000 iterations ===\n";
        std::cout << "Time: " << duration.count() << " ms\n";
        std::cout << "Nodes: " << model.num_nodes() << "\n";
        std::cout << "Edges: " << model.num_edges() << "\n\n";
    }

    return 0;
}
