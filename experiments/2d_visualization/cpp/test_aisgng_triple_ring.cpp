/**
 * @file test_aisgng_triple_ring.cpp
 * @brief Test AiS-GNG on triple ring data (C++ implementation)
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
#include "ais_gng.hpp"

namespace fs = std::filesystem;

std::vector<Eigen::Vector2f> sample_triple_ring(int n_samples, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);

    std::vector<std::tuple<float, float, float, float>> rings = {
        {0.50f, 0.23f, 0.06f, 0.14f},
        {0.27f, 0.68f, 0.06f, 0.14f},
        {0.73f, 0.68f, 0.06f, 0.14f},
    };

    std::vector<Eigen::Vector2f> samples;
    samples.reserve(n_samples);

    int samples_per_ring = n_samples / 3;

    for (const auto& [cx, cy, r_inner, r_outer] : rings) {
        std::uniform_real_distribution<float> radius_dist(r_inner, r_outer);
        for (int i = 0; i < samples_per_ring; ++i) {
            float angle = angle_dist(rng);
            float radius = radius_dist(rng);
            float x = cx + radius * std::cos(angle);
            float y = cy + radius * std::sin(angle);
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

void save_graph(const ais_gng::AiSGNG2f& model, const std::string& nodes_path, const std::string& edges_path) {
    std::vector<Eigen::Vector2f> nodes;
    std::vector<std::pair<int, int>> edges;
    model.get_graph(nodes, edges);

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
}

int main(int argc, char* argv[]) {
    std::string output_dir = "aisgng_triple_ring_output";
    int n_samples = 1500;
    int n_iterations = 5000;
    unsigned int seed = 42;
    int frame_interval = 50;

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    fs::create_directories(output_dir);

    std::cout << "AiS-GNG Triple Ring Test (C++)\n";
    std::cout << "==============================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Iterations: " << n_iterations << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    std::cout << "Generating samples...\n";
    auto samples = sample_triple_ring(n_samples, seed);
    save_samples(samples, output_dir + "/samples.csv");
    std::cout << "Generated " << samples.size() << " samples\n\n";

    // Setup AiS-GNG (parameters matching Python version)
    ais_gng::AiSGNGParams params;
    params.max_nodes = 100;
    params.lambda = 100;
    params.eps_b = 0.08f;
    params.eps_n = 0.008f;
    params.alpha = 0.5f;
    params.beta = 0.005f;
    params.chi = 0.005f;
    params.max_age = 88;
    params.utility_k = 1000.0f;
    params.kappa = 10;
    params.theta_ais_min = 0.03f;
    params.theta_ais_max = 0.15f;

    std::cout << "AiS-GNG Parameters:\n";
    std::cout << "  max_nodes: " << params.max_nodes << "\n";
    std::cout << "  lambda: " << params.lambda << "\n";
    std::cout << "  eps_b: " << params.eps_b << "\n";
    std::cout << "  eps_n: " << params.eps_n << "\n";
    std::cout << "  alpha: " << params.alpha << "\n";
    std::cout << "  beta: " << params.beta << "\n";
    std::cout << "  chi: " << params.chi << "\n";
    std::cout << "  max_age: " << params.max_age << "\n";
    std::cout << "  utility_k: " << params.utility_k << "\n";
    std::cout << "  kappa: " << params.kappa << "\n";
    std::cout << "  theta_ais_min: " << params.theta_ais_min << "\n";
    std::cout << "  theta_ais_max: " << params.theta_ais_max << "\n\n";

    ais_gng::AiSGNG2f model(params, seed);
    model.init();

    std::cout << "Training...\n";

    std::vector<int> frame_iterations;

    auto start_time = std::chrono::high_resolution_clock::now();

    auto callback = [&](const ais_gng::AiSGNG2f& m, int iter) {
        if (iter % frame_interval == 0 || iter == n_iterations - 1) {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(5) << iter;
            std::string suffix = ss.str();

            save_graph(m, output_dir + "/nodes_" + suffix + ".csv",
                       output_dir + "/edges_" + suffix + ".csv");

            frame_iterations.push_back(iter);

            std::cout << "Iteration " << iter << ": "
                      << m.num_nodes() << " nodes, "
                      << m.num_edges() << " edges, "
                      << "+" << m.n_ais_additions << " AiS, "
                      << "-" << m.n_utility_removals << " util\n";
        }
    };

    model.train(samples, n_iterations, callback);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    save_graph(model, output_dir + "/final_nodes.csv", output_dir + "/final_edges.csv");

    {
        std::ofstream file(output_dir + "/frames.csv");
        file << "iteration\n";
        for (int iter : frame_iterations) {
            file << iter << "\n";
        }
    }

    {
        std::ofstream file(output_dir + "/metadata.csv");
        file << "key,value\n";
        file << "algorithm,AiS-GNG\n";
        file << "n_iterations," << n_iterations << "\n";
        file << "time_ms," << duration.count() << "\n";
        file << "final_nodes," << model.num_nodes() << "\n";
        file << "final_edges," << model.num_edges() << "\n";
        file << "n_ais_additions," << model.n_ais_additions << "\n";
        file << "n_utility_removals," << model.n_utility_removals << "\n";
    }

    std::cout << "\nTraining complete!\n";
    std::cout << "Time: " << duration.count() << " ms\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, " << model.num_edges() << " edges\n";
    std::cout << "AiS additions: " << model.n_ais_additions << "\n";
    std::cout << "Utility-based removals: " << model.n_utility_removals << "\n";
    std::cout << "Results saved to: " << output_dir << "\n";

    return 0;
}
