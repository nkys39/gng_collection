/**
 * @file test_gsrm_torus.cpp
 * @brief Test GSRM on torus surface reconstruction (C++ implementation)
 *
 * Generates torus point cloud, runs GSRM training,
 * and outputs results to CSV files for visualization.
 */

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <Eigen/Core>

// Include GSRM from algorithms
#include "../../../algorithms/gsrm/cpp/gsrm.hpp"

// Include helpers from 3d_pointcloud
#include "../../3d_pointcloud/cpp/sampler_3d.hpp"
#include "../../3d_pointcloud/cpp/csv_writer.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Default parameters
    std::string output_dir = "gsrm_torus_output";
    int n_samples = 5000;
    int n_iterations = 15000;
    unsigned int seed = 42;
    int frame_interval = 150;  // 15000/100 = 150 frames

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    csv_writer::ensure_directory(output_dir);

    std::cout << "GSRM Torus Surface Reconstruction Test (C++)\n";
    std::cout << "=============================================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Iterations: " << n_iterations << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    // Generate torus samples
    std::cout << "Generating torus point cloud...\n";
    auto samples = sampler_3d::sample_torus(n_samples, 0.3f, 0.12f, 0.5f, 0.5f, 0.5f, seed);
    csv_writer::save_samples_3d(samples, output_dir + "/samples.csv");
    std::cout << "Generated " << samples.size() << " samples\n\n";

    // Setup GSRM parameters (more nodes for torus topology)
    gsrm::GSRMParams params;
    params.max_nodes = 300;
    params.lambda = 50;
    params.eps_b = 0.1f;
    params.eps_n = 0.01f;
    params.alpha = 0.5f;
    params.beta = 0.005f;
    params.max_age = 50;

    std::cout << "GSRM Parameters:\n";
    std::cout << "  max_nodes: " << params.max_nodes << "\n";
    std::cout << "  lambda: " << params.lambda << "\n";
    std::cout << "  eps_b: " << params.eps_b << "\n";
    std::cout << "  eps_n: " << params.eps_n << "\n";
    std::cout << "  alpha: " << params.alpha << "\n";
    std::cout << "  beta: " << params.beta << "\n";
    std::cout << "  max_age: " << params.max_age << "\n\n";

    gsrm::GSRM3f model(params, seed);

    std::cout << "Training...\n";

    std::vector<int> frame_iterations;
    csv_writer::Timer timer;
    timer.start();

    auto callback = [&](const gsrm::GSRM3f& gsrm, int iter) {
        if (iter % frame_interval == 0 || iter == n_iterations - 1) {
            csv_writer::save_gsrm_mesh_3d(gsrm, output_dir, iter);
            frame_iterations.push_back(iter);

            std::cout << "Iteration " << iter << ": "
                      << gsrm.num_nodes() << " nodes, "
                      << gsrm.num_edges() << " edges, "
                      << gsrm.num_faces() << " faces\n";
        }
    };

    model.train(samples, n_iterations, callback);

    timer.stop();

    // Save final result
    std::vector<Eigen::Vector3f> final_nodes;
    std::vector<std::pair<int, int>> final_edges;
    std::vector<std::array<int, 3>> final_faces;
    model.get_mesh(final_nodes, final_edges, final_faces);

    csv_writer::save_nodes_3d(final_nodes, output_dir + "/final_nodes.csv");
    csv_writer::save_edges(final_edges, output_dir + "/final_edges.csv");
    csv_writer::save_faces(final_faces, output_dir + "/final_faces.csv");

    // Save frame list
    csv_writer::save_frames(frame_iterations, output_dir + "/frames.csv");

    // Save metadata
    csv_writer::MetadataWriter metadata(output_dir + "/metadata.csv");
    metadata.add("algorithm", "GSRM");
    metadata.add("shape", "torus");
    metadata.add("n_iterations", n_iterations);
    metadata.add("n_samples", n_samples);
    metadata.add("max_nodes", params.max_nodes);
    metadata.add("time_ms", static_cast<int>(timer.elapsed_ms()));
    metadata.add("final_nodes", model.num_nodes());
    metadata.add("final_edges", model.num_edges());
    metadata.add("final_faces", model.num_faces());
    metadata.save();

    std::cout << "\nTraining complete!\n";
    std::cout << "Time: " << timer.elapsed_ms() << " ms\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, "
              << model.num_edges() << " edges, "
              << model.num_faces() << " faces\n";
    std::cout << "Results saved to: " << output_dir << "\n";

    return 0;
}
