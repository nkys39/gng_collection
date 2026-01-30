/**
 * @file test_gng_floor_wall.cpp
 * @brief Test GNG on 3D floor and wall data (C++ implementation)
 *
 * Generates floor+wall sample data, runs GNG training,
 * and outputs results to CSV files for visualization.
 */

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <Eigen/Core>
#include "gng.hpp"
#include "sampler_3d.hpp"
#include "csv_writer.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Default parameters
    std::string output_dir = "gng_floor_wall_output";
    int n_samples = 2000;
    int n_iterations = 8000;
    unsigned int seed = 42;
    int frame_interval = 80;  // 8000/100 = 80 for ~100 frames

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    csv_writer::ensure_directory(output_dir);

    std::cout << "GNG 3D Floor+Wall Test (C++)\n";
    std::cout << "============================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Iterations: " << n_iterations << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    // Generate floor+wall samples
    std::cout << "Generating samples...\n";
    auto samples = sampler_3d::sample_floor_and_wall(n_samples, sampler_3d::FloorWallParams(), seed);
    csv_writer::save_samples_3d(samples, output_dir + "/samples.csv");
    std::cout << "Generated " << samples.size() << " samples\n\n";

    // Setup GNG (3D parameters)
    gng::GNGParams params;
    params.max_nodes = 150;
    params.lambda = 100;
    params.eps_b = 0.1f;
    params.eps_n = 0.01f;
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

    gng::GNG3f model(params, seed);
    model.init();

    std::cout << "Training...\n";

    std::vector<int> frame_iterations;
    csv_writer::Timer timer;
    timer.start();

    auto callback = [&](const gng::GNG3f& gng, int iter) {
        if (iter % frame_interval == 0 || iter == n_iterations - 1) {
            csv_writer::save_gng_graph_3d(gng, output_dir, iter);
            frame_iterations.push_back(iter);

            std::cout << "Iteration " << iter << ": "
                      << gng.num_nodes() << " nodes, "
                      << gng.num_edges() << " edges\n";
        }
    };

    model.train(samples, n_iterations, callback);

    timer.stop();

    // Save final result
    std::vector<Eigen::Vector3f> final_nodes;
    std::vector<std::pair<int, int>> final_edges;
    model.get_graph(final_nodes, final_edges);
    csv_writer::save_nodes_3d(final_nodes, output_dir + "/final_nodes.csv");
    csv_writer::save_edges(final_edges, output_dir + "/final_edges.csv");

    // Save frame list
    csv_writer::save_frames(frame_iterations, output_dir + "/frames.csv");

    // Save metadata
    csv_writer::MetadataWriter metadata(output_dir + "/metadata.csv");
    metadata.add("algorithm", "GNG");
    metadata.add("n_iterations", n_iterations);
    metadata.add("n_samples", n_samples);
    metadata.add("max_nodes", params.max_nodes);
    metadata.add("time_ms", static_cast<int>(timer.elapsed_ms()));
    metadata.add("final_nodes", model.num_nodes());
    metadata.add("final_edges", model.num_edges());
    metadata.save();

    std::cout << "\nTraining complete!\n";
    std::cout << "Time: " << timer.elapsed_ms() << " ms\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, " << model.num_edges() << " edges\n";
    std::cout << "Results saved to: " << output_dir << "\n";

    return 0;
}
