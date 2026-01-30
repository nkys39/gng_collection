/**
 * @file test_aisgngdt_floor_wall.cpp
 * @brief Test AiS-GNG-DT on 3D floor and wall data (C++ implementation)
 *
 * Tests AiS-GNG-DT combining GNG-DT's multiple topologies with AiS-GNG's
 * Add-if-Silent rule and utility-based node management.
 */

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <Eigen/Core>
#include "ais_gng_dt.hpp"
#include "sampler_3d.hpp"
#include "csv_writer.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string output_dir = "aisgngdt_floor_wall_output";
    int n_samples = 2000;
    int n_iterations = 8000;
    unsigned int seed = 42;
    int frame_interval = 80;

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    csv_writer::ensure_directory(output_dir);

    std::cout << "AiS-GNG-DT 3D Floor+Wall Test (C++)\n";
    std::cout << "===================================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Iterations: " << n_iterations << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    std::cout << "Generating samples...\n";
    auto samples = sampler_3d::sample_floor_and_wall(n_samples, sampler_3d::FloorWallParams(), seed);
    csv_writer::save_samples_3d(samples, output_dir + "/samples.csv");
    std::cout << "Generated " << samples.size() << " samples\n\n";

    // AiS-GNG-DT parameters
    ais_gng_dt::AiSGNGDTParams params;
    params.max_nodes = 150;
    params.lambda = 100;
    params.eps_b = 0.05f;
    params.eps_n = 0.005f;
    params.alpha = 0.5f;
    params.beta = 0.005f;
    params.max_age = 88;
    // GNG-DT specific
    params.tau_normal = 0.95f;
    // AiS-GNG specific
    params.theta_ais_min = 0.03f;
    params.theta_ais_max = 0.12f;
    params.kappa = 10;
    params.utility_k = 1000.0f;
    params.chi = 0.005f;

    std::cout << "AiS-GNG-DT Parameters:\n";
    std::cout << "  max_nodes: " << params.max_nodes << "\n";
    std::cout << "  lambda: " << params.lambda << "\n";
    std::cout << "  eps_b: " << params.eps_b << "\n";
    std::cout << "  eps_n: " << params.eps_n << "\n";
    std::cout << "  max_age: " << params.max_age << "\n";
    std::cout << "  tau_normal: " << params.tau_normal << "\n";
    std::cout << "  theta_ais: [" << params.theta_ais_min << ", " << params.theta_ais_max << "]\n";
    std::cout << "  kappa: " << params.kappa << "\n";
    std::cout << "  utility_k: " << params.utility_k << "\n\n";

    ais_gng_dt::AiSGNGDT model(params, seed);
    model.init();

    std::cout << "Training...\n";

    std::vector<int> frame_iterations;
    csv_writer::Timer timer;
    timer.start();

    auto callback = [&](const ais_gng_dt::AiSGNGDT& gng, int iter) {
        if (iter % frame_interval == 0 || iter == n_iterations - 1) {
            // Save graph for this frame
            std::vector<Eigen::Vector3f> nodes;
            std::vector<std::pair<int, int>> pos_edges;
            std::vector<std::pair<int, int>> color_edges;
            std::vector<std::pair<int, int>> normal_edges;
            gng.get_multi_graph(nodes, pos_edges, color_edges, normal_edges);
            auto normals = gng.get_node_normals();

            std::ostringstream oss;
            oss << std::setfill('0') << std::setw(5) << iter;
            std::string suffix = oss.str();

            csv_writer::save_nodes_3d(nodes, output_dir + "/nodes_" + suffix + ".csv");
            csv_writer::save_edges(pos_edges, output_dir + "/pos_edges_" + suffix + ".csv");
            csv_writer::save_edges(normal_edges, output_dir + "/normal_edges_" + suffix + ".csv");
            csv_writer::save_normals(normals, output_dir + "/normals_" + suffix + ".csv");

            frame_iterations.push_back(iter);

            std::cout << "Iteration " << iter << ": "
                      << gng.num_nodes() << " nodes, "
                      << gng.num_edges_pos() << " pos-edges, "
                      << gng.num_edges_normal() << " normal-edges, "
                      << "AiS: " << gng.n_ais_additions << "\n";
        }
    };

    model.train(samples, n_iterations, callback);

    timer.stop();

    // Save final result
    std::vector<Eigen::Vector3f> final_nodes;
    std::vector<std::pair<int, int>> final_pos_edges;
    std::vector<std::pair<int, int>> final_color_edges;
    std::vector<std::pair<int, int>> final_normal_edges;
    model.get_multi_graph(final_nodes, final_pos_edges, final_color_edges, final_normal_edges);
    auto final_normals = model.get_node_normals();

    csv_writer::save_nodes_3d(final_nodes, output_dir + "/final_nodes.csv");
    csv_writer::save_edges(final_pos_edges, output_dir + "/final_pos_edges.csv");
    csv_writer::save_edges(final_normal_edges, output_dir + "/final_normal_edges.csv");
    csv_writer::save_normals(final_normals, output_dir + "/final_normals.csv");

    csv_writer::save_frames(frame_iterations, output_dir + "/frames.csv");

    csv_writer::MetadataWriter metadata(output_dir + "/metadata.csv");
    metadata.add("algorithm", "AiS-GNG-DT");
    metadata.add("n_iterations", n_iterations);
    metadata.add("n_samples", n_samples);
    metadata.add("max_nodes", params.max_nodes);
    metadata.add("time_ms", static_cast<int>(timer.elapsed_ms()));
    metadata.add("final_nodes", model.num_nodes());
    metadata.add("final_pos_edges", model.num_edges_pos());
    metadata.add("final_normal_edges", model.num_edges_normal());
    metadata.add("n_ais_additions", model.n_ais_additions);
    metadata.add("n_utility_removals", model.n_utility_removals);
    metadata.save();

    std::cout << "\nTraining complete!\n";
    std::cout << "Time: " << timer.elapsed_ms() << " ms\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, "
              << model.num_edges_pos() << " pos-edges, "
              << model.num_edges_normal() << " normal-edges\n";
    std::cout << "AiS additions: " << model.n_ais_additions << "\n";
    std::cout << "Utility removals: " << model.n_utility_removals << "\n";
    std::cout << "Results saved to: " << output_dir << "\n";

    return 0;
}
