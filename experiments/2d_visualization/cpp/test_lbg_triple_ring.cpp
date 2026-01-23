/**
 * @file test_lbg_triple_ring.cpp
 * @brief Test LBG on triple ring data (C++ implementation)
 *
 * Generates triple ring sample data, runs LBG training,
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
#include "lbg.hpp"

namespace fs = std::filesystem;

/**
 * @brief Sample points from triple ring (3 separate rings in triangle pattern).
 */
std::vector<Eigen::Vector2f> sample_triple_ring(
    int n_samples,
    unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);

    // Three separate rings arranged in triangle pattern
    std::vector<std::tuple<float, float, float, float>> rings = {
        {0.50f, 0.23f, 0.06f, 0.14f},  // top center
        {0.27f, 0.68f, 0.06f, 0.14f},  // bottom left
        {0.73f, 0.68f, 0.06f, 0.14f},  // bottom right
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

void save_graph(const lbg::LBG2f& model, const std::string& nodes_path, const std::string& edges_path) {
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
    // Default parameters
    std::string output_dir = "lbg_triple_ring_output";
    int n_samples = 1500;
    int n_iterations = 100;  // LBG uses epochs, not iterations
    unsigned int seed = 42;
    int frame_interval = 1;  // Save every epoch

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    fs::create_directories(output_dir);

    std::cout << "LBG Triple Ring Test (C++)\n";
    std::cout << "==========================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Epochs: " << n_iterations << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    // Generate triple ring samples
    std::cout << "Generating samples...\n";
    auto samples = sample_triple_ring(n_samples, seed);
    save_samples(samples, output_dir + "/samples.csv");
    std::cout << "Generated " << samples.size() << " samples\n\n";

    // Setup LBG (parameters matching Python version)
    lbg::LBGParams params;
    params.n_nodes = 50;
    params.max_epochs = n_iterations;
    params.convergence_threshold = 1e-6f;
    params.use_utility = false;

    std::cout << "LBG Parameters:\n";
    std::cout << "  n_nodes: " << params.n_nodes << "\n";
    std::cout << "  max_epochs: " << params.max_epochs << "\n";
    std::cout << "  convergence_threshold: " << params.convergence_threshold << "\n";
    std::cout << "  use_utility: " << (params.use_utility ? "true" : "false") << "\n\n";

    lbg::LBG2f model(params, seed);
    model.init_from_data(samples);

    std::cout << "Training...\n";

    std::vector<int> frame_iterations;

    // Measure training time
    auto start_time = std::chrono::high_resolution_clock::now();

    auto callback = [&](const lbg::LBG2f& lbg, int epoch) {
        if (epoch % frame_interval == 0 || epoch == n_iterations - 1) {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(5) << epoch;
            std::string suffix = ss.str();

            save_graph(lbg, output_dir + "/nodes_" + suffix + ".csv",
                       output_dir + "/edges_" + suffix + ".csv");

            frame_iterations.push_back(epoch);

            std::cout << "Epoch " << epoch << ": "
                      << lbg.num_nodes() << " nodes, "
                      << lbg.num_edges() << " edges\n";
        }
    };

    model.train(samples, n_iterations, callback);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save final result
    save_graph(model, output_dir + "/final_nodes.csv", output_dir + "/final_edges.csv");

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
        file << "algorithm,LBG\n";
        file << "n_epochs," << n_iterations << "\n";
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
