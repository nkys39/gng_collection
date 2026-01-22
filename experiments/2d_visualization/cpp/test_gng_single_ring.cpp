/**
 * @file test_gng_single_ring.cpp
 * @brief Test GNG on single ring data (C++ implementation)
 *
 * This program generates ring-shaped sample data, runs GNG training,
 * and outputs results to CSV files for visualization with Python.
 *
 * Output files:
 *   - samples.csv: Sample points (x, y)
 *   - nodes_XXXX.csv: Node positions at iteration XXXX (x, y)
 *   - edges_XXXX.csv: Edge list at iteration XXXX (i, j)
 *   - final_nodes.csv: Final node positions
 *   - final_edges.csv: Final edge list
 *
 * Usage:
 *   ./test_gng_single_ring [output_dir] [n_samples] [n_iterations] [seed]
 */

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
#include "gng.hpp"

namespace fs = std::filesystem;

/**
 * @brief Sample points from a ring (donut) shape.
 *
 * @param n_samples Number of samples to generate
 * @param center_x Ring center x coordinate
 * @param center_y Ring center y coordinate
 * @param r_inner Inner radius
 * @param r_outer Outer radius
 * @param seed Random seed
 * @return Vector of sample points
 */
std::vector<Eigen::Vector2f> sample_ring(
    int n_samples,
    float center_x, float center_y,
    float r_inner, float r_outer,
    unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dist(r_inner, r_outer);

    std::vector<Eigen::Vector2f> samples;
    samples.reserve(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        float angle = angle_dist(rng);
        float radius = radius_dist(rng);
        float x = center_x + radius * std::cos(angle);
        float y = center_y + radius * std::sin(angle);
        samples.emplace_back(x, y);
    }

    return samples;
}

/**
 * @brief Save sample points to CSV.
 */
void save_samples(const std::vector<Eigen::Vector2f>& samples, const std::string& path) {
    std::ofstream file(path);
    file << "x,y\n";
    file << std::fixed << std::setprecision(6);
    for (const auto& p : samples) {
        file << p.x() << "," << p.y() << "\n";
    }
}

/**
 * @brief Save GNG graph to CSV files.
 */
void save_graph(const gng::GNG2f& gng, const std::string& nodes_path, const std::string& edges_path) {
    std::vector<Eigen::Vector2f> nodes;
    std::vector<std::pair<int, int>> edges;
    gng.get_graph(nodes, edges);

    // Save nodes
    {
        std::ofstream file(nodes_path);
        file << "x,y\n";
        file << std::fixed << std::setprecision(6);
        for (const auto& n : nodes) {
            file << n.x() << "," << n.y() << "\n";
        }
    }

    // Save edges
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
    std::string output_dir = ".";
    int n_samples = 1500;
    int n_iterations = 5000;
    unsigned int seed = 42;
    int frame_interval = 50;

    // Parse command line arguments
    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    // Create output directory
    fs::create_directories(output_dir);

    std::cout << "GNG Single Ring Test (C++)\n";
    std::cout << "==========================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Iterations: " << n_iterations << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    // Generate ring samples (normalized to [0,1] range)
    // Ring centered at (0.5, 0.5), inner radius 0.25, outer radius 0.375
    std::cout << "Generating samples...\n";
    auto samples = sample_ring(n_samples, 0.5f, 0.5f, 0.25f, 0.375f, seed);
    save_samples(samples, output_dir + "/samples.csv");
    std::cout << "Generated " << samples.size() << " samples\n\n";

    // Setup GNG
    gng::GNGParams params;
    params.max_nodes = 100;
    params.lambda = 100;
    params.eps_b = 0.08f;
    params.eps_n = 0.008f;
    params.alpha = 0.5f;
    params.beta = 0.005f;
    params.max_age = 100;

    std::cout << "GNG Parameters:\n";
    std::cout << "  max_nodes: " << params.max_nodes << "\n";
    std::cout << "  lambda: " << params.lambda << "\n";
    std::cout << "  eps_b: " << params.eps_b << "\n";
    std::cout << "  eps_n: " << params.eps_n << "\n";
    std::cout << "  alpha: " << params.alpha << "\n";
    std::cout << "  beta: " << params.beta << "\n";
    std::cout << "  max_age: " << params.max_age << "\n\n";

    gng::GNG2f model(params, seed);
    model.init();

    // Train with callback to save intermediate results
    std::cout << "Training...\n";

    // Keep track of saved frames
    std::vector<int> frame_iterations;

    auto callback = [&](const gng::GNG2f& gng, int iter) {
        if (iter % frame_interval == 0 || iter == n_iterations - 1) {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(5) << iter;
            std::string suffix = ss.str();

            save_graph(gng, output_dir + "/nodes_" + suffix + ".csv",
                       output_dir + "/edges_" + suffix + ".csv");

            frame_iterations.push_back(iter);

            std::cout << "Iteration " << iter << ": "
                      << gng.num_nodes() << " nodes, "
                      << gng.num_edges() << " edges\n";
        }
    };

    model.train(samples, n_iterations, callback);

    // Save final result
    save_graph(model, output_dir + "/final_nodes.csv", output_dir + "/final_edges.csv");

    // Save frame list for visualization script
    {
        std::ofstream file(output_dir + "/frames.csv");
        file << "iteration\n";
        for (int iter : frame_iterations) {
            file << iter << "\n";
        }
    }

    // Save metadata for visualization (ring parameters)
    {
        std::ofstream file(output_dir + "/metadata.csv");
        file << "key,value\n";
        file << "ring_center_x,0.5\n";
        file << "ring_center_y,0.5\n";
        file << "ring_r_inner,0.25\n";
        file << "ring_r_outer,0.375\n";
        file << "n_iterations," << n_iterations << "\n";
    }

    std::cout << "\nTraining complete!\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, " << model.num_edges() << " edges\n";
    std::cout << "Results saved to: " << output_dir << "\n";

    return 0;
}
