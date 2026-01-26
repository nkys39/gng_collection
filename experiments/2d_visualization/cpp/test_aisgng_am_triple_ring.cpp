/**
 * @file test_aisgng_am_triple_ring.cpp
 * @brief Test AiS-GNG-AM (SMC 2023 with Amount of Movement) on triple ring data
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
#include "ais_gng_am.hpp"

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

void save_graph(const ais_gng_am::AiSGNGAM2f& model,
                const std::string& nodes_path, const std::string& edges_path) {
    std::vector<Eigen::Vector2f> nodes;
    std::vector<std::pair<int, int>> edges;
    model.get_graph(nodes, edges);

    // Save nodes with movement values
    auto movements = model.get_node_movements();
    {
        std::ofstream file(nodes_path);
        file << "x,y,movement\n";
        file << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < nodes.size(); ++i) {
            file << nodes[i].x() << "," << nodes[i].y() << "," << movements[i] << "\n";
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
    std::string output_dir = "aisgng_am_triple_ring_output";
    int n_samples = 1500;
    int n_iterations = 5000;
    unsigned int seed = 42;
    int frame_interval = 50;

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    fs::create_directories(output_dir);

    std::cout << "AiS-GNG-AM (SMC 2023) Triple Ring Test (C++)\n";
    std::cout << "=============================================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Iterations: " << n_iterations << "\n\n";

    auto samples = sample_triple_ring(n_samples, seed);
    save_samples(samples, output_dir + "/samples.csv");

    ais_gng_am::AiSGNGAMParams params;
    params.max_nodes = 100;
    params.lambda = 100;
    params.eps_b = 0.08f;
    params.eps_n = 0.008f;
    params.max_age = 88;
    params.utility_k = 1000.0f;
    params.kappa = 10;
    params.theta_ais_min = 0.03f;
    params.theta_ais_max = 0.15f;
    params.am_decay = 0.95f;
    params.am_threshold = 0.005f;

    std::cout << "SMC 2023 (AM) Parameters:\n";
    std::cout << "  theta_ais_min: " << params.theta_ais_min << "\n";
    std::cout << "  theta_ais_max: " << params.theta_ais_max << "\n";
    std::cout << "  am_decay: " << params.am_decay << "\n";
    std::cout << "  am_threshold: " << params.am_threshold << "\n\n";

    ais_gng_am::AiSGNGAM2f model(params, seed);
    model.init();

    std::cout << "Training...\n";
    std::vector<int> frame_iterations;

    auto start_time = std::chrono::high_resolution_clock::now();

    auto callback = [&](const ais_gng_am::AiSGNGAM2f& m, int iter) {
        if (iter % frame_interval == 0 || iter == n_iterations - 1) {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(5) << iter;
            save_graph(m, output_dir + "/nodes_" + ss.str() + ".csv",
                       output_dir + "/edges_" + ss.str() + ".csv");
            frame_iterations.push_back(iter);

            std::cout << "Iteration " << iter << ": "
                      << m.num_nodes() << " nodes, +" << m.n_ais_additions << " AiS, "
                      << m.num_moving_nodes() << " moving\n";
        }
    };

    model.train(samples, n_iterations, callback);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    save_graph(model, output_dir + "/final_nodes.csv", output_dir + "/final_edges.csv");

    {
        std::ofstream file(output_dir + "/frames.csv");
        file << "iteration\n";
        for (int iter : frame_iterations) file << iter << "\n";
    }

    {
        std::ofstream file(output_dir + "/metadata.csv");
        file << "key,value\n";
        file << "algorithm,AiS-GNG-AM\n";
        file << "n_iterations," << n_iterations << "\n";
        file << "time_ms," << duration.count() << "\n";
        file << "final_nodes," << model.num_nodes() << "\n";
        file << "final_edges," << model.num_edges() << "\n";
        file << "n_ais_additions," << model.n_ais_additions << "\n";
        file << "n_moving_nodes," << model.num_moving_nodes() << "\n";
    }

    std::cout << "\nComplete! Time: " << duration.count() << " ms\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, AiS: " << model.n_ais_additions << "\n";
    std::cout << "Moving nodes: " << model.num_moving_nodes() << "\n";

    return 0;
}
