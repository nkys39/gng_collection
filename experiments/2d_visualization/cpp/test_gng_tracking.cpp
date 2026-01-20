/**
 * @file test_gng_tracking.cpp
 * @brief Test GNG tracking of a moving ring (C++ implementation)
 *
 * This program generates a ring that moves along a circular orbit,
 * and uses GNG with online learning (partial_fit) to track it.
 *
 * Output files:
 *   - frame_XXXX_samples.csv: Sample points at frame XXXX
 *   - frame_XXXX_nodes.csv: Node positions at frame XXXX
 *   - frame_XXXX_edges.csv: Edge list at frame XXXX
 *   - frame_XXXX_center.csv: Ring center position at frame XXXX
 *
 * Usage:
 *   ./test_gng_tracking [output_dir] [n_frames] [samples_per_frame] [seed]
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
 */
std::vector<Eigen::Vector2f> sample_ring(
    int n_samples,
    float center_x, float center_y,
    float r_inner, float r_outer,
    std::mt19937& rng)
{
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

/**
 * @brief Save center position to CSV.
 */
void save_center(float x, float y, const std::string& path) {
    std::ofstream file(path);
    file << "x,y\n";
    file << std::fixed << std::setprecision(6);
    file << x << "," << y << "\n";
}

int main(int argc, char* argv[]) {
    // Default parameters (matching Python version)
    std::string output_dir = "tracking_output";
    int n_frames = 120;          // Match Python version
    int samples_per_frame = 50;
    unsigned int seed = 42;

    // Parse command line arguments
    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_frames = std::stoi(argv[2]);
    if (argc > 3) samples_per_frame = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    // Create output directory
    fs::create_directories(output_dir);

    std::cout << "GNG Tracking Test (C++)\n";
    std::cout << "=======================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Frames: " << n_frames << "\n";
    std::cout << "Samples per frame: " << samples_per_frame << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    // Ring parameters (normalized coordinates)
    const float ring_r_inner = 0.08f;
    const float ring_r_outer = 0.12f;

    // Orbit parameters
    const float orbit_center_x = 0.5f;
    const float orbit_center_y = 0.5f;
    const float orbit_radius = 0.25f;

    // Setup GNG with tracking-optimized parameters (matching Python version)
    gng::GNGParams params;
    params.max_nodes = 50;       // Match Python version
    params.lambda = 20;          // More frequent node insertion for tracking
    params.eps_b = 0.15f;        // Higher learning rate for faster adaptation
    params.eps_n = 0.01f;        // Match Python version
    params.alpha = 0.5f;
    params.beta = 0.01f;         // Faster error decay
    params.max_age = 30;         // Shorter edge lifetime for better adaptation

    std::cout << "GNG Parameters (tracking-optimized):\n";
    std::cout << "  max_nodes: " << params.max_nodes << "\n";
    std::cout << "  lambda: " << params.lambda << "\n";
    std::cout << "  eps_b: " << params.eps_b << "\n";
    std::cout << "  eps_n: " << params.eps_n << "\n";
    std::cout << "  alpha: " << params.alpha << "\n";
    std::cout << "  beta: " << params.beta << "\n";
    std::cout << "  max_age: " << params.max_age << "\n\n";

    gng::GNG2f model(params, seed);
    model.init();

    std::mt19937 rng(seed);

    std::cout << "Running tracking simulation...\n";

    // Run tracking simulation
    for (int frame = 0; frame < n_frames; ++frame) {
        // Calculate ring center position on orbit
        float angle = static_cast<float>(frame) / n_frames * 2.0f * M_PI;
        float center_x = orbit_center_x + orbit_radius * std::cos(angle);
        float center_y = orbit_center_y + orbit_radius * std::sin(angle);

        // Generate samples from current ring position
        auto samples = sample_ring(samples_per_frame, center_x, center_y,
                                   ring_r_inner, ring_r_outer, rng);

        // Online learning: process each sample
        for (const auto& sample : samples) {
            model.partial_fit(sample);
        }

        // Save frame data
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(4) << frame;
        std::string suffix = ss.str();

        save_samples(samples, output_dir + "/frame_" + suffix + "_samples.csv");
        save_graph(model, output_dir + "/frame_" + suffix + "_nodes.csv",
                   output_dir + "/frame_" + suffix + "_edges.csv");
        save_center(center_x, center_y, output_dir + "/frame_" + suffix + "_center.csv");

        if (frame % 20 == 0 || frame == n_frames - 1) {
            std::cout << "Frame " << frame << ": "
                      << model.num_nodes() << " nodes, "
                      << model.num_edges() << " edges, "
                      << "center=(" << center_x << ", " << center_y << ")\n";
        }
    }

    // Save metadata
    {
        std::ofstream file(output_dir + "/metadata.csv");
        file << "key,value\n";
        file << "n_frames," << n_frames << "\n";
        file << "samples_per_frame," << samples_per_frame << "\n";
        file << "orbit_center_x," << orbit_center_x << "\n";
        file << "orbit_center_y," << orbit_center_y << "\n";
        file << "orbit_radius," << orbit_radius << "\n";
        file << "ring_r_inner," << ring_r_inner << "\n";
        file << "ring_r_outer," << ring_r_outer << "\n";
    }

    std::cout << "\nTracking simulation complete!\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, " << model.num_edges() << " edges\n";
    std::cout << "Results saved to: " << output_dir << "\n";

    return 0;
}
