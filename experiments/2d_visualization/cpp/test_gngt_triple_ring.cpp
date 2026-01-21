/**
 * @file test_gngt_triple_ring.cpp
 * @brief Test GNG-T on triple ring data (C++ implementation)
 *
 * Based on Kubota & Satomi (2008) GNG with Triangulation.
 * Generates triple ring sample data, runs GNG-T training with 50,000 iterations,
 * and outputs results to CSV files for visualization.
 */

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include "gng_t.hpp"

namespace fs = std::filesystem;

/**
 * @brief Sample points from triple ring (3 concentric rings).
 */
std::vector<Eigen::Vector2f> sample_triple_ring(
    int n_samples,
    float center_x, float center_y,
    unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);

    std::vector<std::pair<float, float>> rings = {
        {0.15f, 0.20f},
        {0.25f, 0.30f},
        {0.35f, 0.40f},
    };

    std::vector<Eigen::Vector2f> samples;
    samples.reserve(n_samples);

    int samples_per_ring = n_samples / 3;

    for (const auto& [r_inner, r_outer] : rings) {
        std::uniform_real_distribution<float> radius_dist(r_inner, r_outer);
        for (int i = 0; i < samples_per_ring; ++i) {
            float angle = angle_dist(rng);
            float radius = radius_dist(rng);
            float x = center_x + radius * std::cos(angle);
            float y = center_y + radius * std::sin(angle);
            samples.emplace_back(x, y);
        }
    }

    return samples;
}

void save_samples(const std::vector<Eigen::Vector2f>& samples, const std::string& path) {
    std::ofstream file(path);
    file << "x,y\n";
    file << std::fixed << std::setprecision(6);
    for (const auto& p : samples) {
        file << p.x() << "," << p.y() << "\n";
    }
}

void save_graph(const gng_t::GNGT2f& gng, const std::string& nodes_path,
                const std::string& edges_path, const std::string& triangles_path) {
    std::vector<Eigen::Vector2f> nodes;
    std::vector<std::pair<int, int>> edges;
    gng.get_graph(nodes, edges);

    {
        std::ofstream file(nodes_path);
        file << "x,y\n";
        file << std::fixed << std::setprecision(6);
        for (const auto& n : nodes) {
            file << n.x() << "," << n.y() << "\n";
        }
    }

    {
        std::ofstream file(edges_path);
        file << "i,j\n";
        for (const auto& e : edges) {
            file << e.first << "," << e.second << "\n";
        }
    }

    // Save triangles
    {
        auto triangles = gng.get_triangles();
        std::ofstream file(triangles_path);
        file << "i,j,k\n";
        for (const auto& [i, j, k] : triangles) {
            file << i << "," << j << "," << k << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    // Default parameters (paper: 50,000 iterations)
    std::string output_dir = "gngt_triple_ring_output";
    int n_samples = 1500;
    int n_iterations = 50000;
    unsigned int seed = 42;
    int frame_interval = 500;

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    fs::create_directories(output_dir);

    std::cout << "GNG-T Triple Ring Test (C++)\n";
    std::cout << "============================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Iterations: " << n_iterations << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    // Generate triple ring samples
    std::cout << "Generating samples...\n";
    auto samples = sample_triple_ring(n_samples, 0.5f, 0.5f, seed);
    save_samples(samples, output_dir + "/samples.csv");
    std::cout << "Generated " << samples.size() << " samples\n\n";

    // Setup GNG-T (paper parameters)
    gng_t::GNGTParams params;
    params.max_nodes = 150;
    params.lambda = 100;
    params.eps_b = 0.08f;
    params.eps_n = 0.008f;
    params.alpha = 0.5f;
    params.beta = 0.005f;
    params.max_age = 100;

    std::cout << "GNG-T Parameters:\n";
    std::cout << "  max_nodes: " << params.max_nodes << "\n";
    std::cout << "  lambda: " << params.lambda << "\n";
    std::cout << "  eps_b: " << params.eps_b << "\n";
    std::cout << "  eps_n: " << params.eps_n << "\n";
    std::cout << "  alpha: " << params.alpha << "\n";
    std::cout << "  beta: " << params.beta << "\n";
    std::cout << "  max_age: " << params.max_age << "\n\n";

    gng_t::GNGT2f model(params, seed);
    model.init();

    std::cout << "Training...\n";

    std::vector<int> frame_iterations;

    auto start_time = std::chrono::high_resolution_clock::now();

    auto callback = [&](const gng_t::GNGT2f& gng, int iter) {
        if (iter % frame_interval == 0 || iter == n_iterations - 1) {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(5) << iter;
            std::string suffix = ss.str();

            save_graph(gng, output_dir + "/nodes_" + suffix + ".csv",
                       output_dir + "/edges_" + suffix + ".csv",
                       output_dir + "/triangles_" + suffix + ".csv");

            frame_iterations.push_back(iter);

            std::cout << "Iteration " << iter << ": "
                      << gng.num_nodes() << " nodes, "
                      << gng.num_edges() << " edges\n";
        }
    };

    model.train(samples, n_iterations, callback);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save final result
    save_graph(model, output_dir + "/final_nodes.csv",
               output_dir + "/final_edges.csv",
               output_dir + "/final_triangles.csv");

    // Save frame list
    {
        std::ofstream file(output_dir + "/frames.csv");
        file << "iteration\n";
        for (int iter : frame_iterations) {
            file << iter << "\n";
        }
    }

    // Save metadata
    {
        std::ofstream file(output_dir + "/metadata.csv");
        file << "key,value\n";
        file << "algorithm,GNG-T\n";
        file << "n_iterations," << n_iterations << "\n";
        file << "time_ms," << duration.count() << "\n";
        file << "final_nodes," << model.num_nodes() << "\n";
        file << "final_edges," << model.num_edges() << "\n";
    }

    std::cout << "\nTraining complete!\n";
    std::cout << "Time: " << duration.count() << " ms\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, " << model.num_edges() << " edges\n";
    std::cout << "Results saved to: " << output_dir << "\n";

    return 0;
}
