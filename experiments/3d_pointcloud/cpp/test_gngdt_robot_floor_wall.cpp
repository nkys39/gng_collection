/**
 * @file test_gngdt_robot_floor_wall.cpp
 * @brief Test GNG-DT Robot on 3D floor and wall data (C++ implementation)
 *
 * Tests GNG-DT Robot with traversability analysis features.
 */

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <Eigen/Core>
#include "gng_dt_robot.hpp"
#include "sampler_3d.hpp"
#include "csv_writer.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string output_dir = "gngdt_robot_floor_wall_output";
    int n_samples = 2000;
    int n_iterations = 8000;
    unsigned int seed = 42;
    int frame_interval = 80;

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) n_samples = std::stoi(argv[2]);
    if (argc > 3) n_iterations = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<unsigned int>(std::stoi(argv[4]));

    csv_writer::ensure_directory(output_dir);

    std::cout << "GNG-DT Robot 3D Floor+Wall Test (C++)\n";
    std::cout << "=====================================\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Samples: " << n_samples << "\n";
    std::cout << "Iterations: " << n_iterations << "\n";
    std::cout << "Seed: " << seed << "\n\n";

    std::cout << "Generating samples...\n";
    auto samples = sampler_3d::sample_floor_and_wall(n_samples, sampler_3d::FloorWallParams(), seed);
    csv_writer::save_samples_3d(samples, output_dir + "/samples.csv");
    std::cout << "Generated " << samples.size() << " samples\n\n";

    // GNG-DT Robot parameters
    gng_dt::GNGDTRobotParams params;
    params.max_nodes = 150;
    params.lambda = 100;
    params.eps_b = 0.05f;
    params.eps_n = 0.0005f;
    params.alpha = 0.5f;
    params.beta = 0.0005f;
    params.max_age = 88;
    params.tau_normal = 0.95f;
    params.max_angle = 20.0f;
    params.s1thv = 1.0f;

    std::cout << "GNG-DT Robot Parameters:\n";
    std::cout << "  max_nodes: " << params.max_nodes << "\n";
    std::cout << "  lambda: " << params.lambda << "\n";
    std::cout << "  max_age: " << params.max_age << "\n";
    std::cout << "  tau_normal: " << params.tau_normal << "\n";
    std::cout << "  max_angle: " << params.max_angle << "\n";
    std::cout << "  s1thv: " << params.s1thv << "\n\n";

    gng_dt::GrowingNeuralGasDTRobot model(params, seed);
    model.init();

    std::cout << "Training...\n";

    std::vector<int> frame_iterations;
    csv_writer::Timer timer;
    timer.start();

    auto callback = [&](const gng_dt::GrowingNeuralGasDTRobot& gng, int iter) {
        if (iter % frame_interval == 0 || iter == n_iterations - 1) {
            csv_writer::save_gngdt_robot_graph_3d(gng, output_dir, iter);
            frame_iterations.push_back(iter);

            // Count traversable nodes
            auto traversability = gng.get_traversability();
            int n_traversable = 0;
            for (int t : traversability) {
                if (t == 1) ++n_traversable;
            }

            std::cout << "Iteration " << iter << ": "
                      << gng.num_nodes() << " nodes, "
                      << gng.num_edges_pos() << " pos-edges, "
                      << n_traversable << " traversable\n";
        }
    };

    model.train(samples, n_iterations, callback);

    timer.stop();

    // Save final result
    std::vector<Eigen::Vector3f> final_nodes;
    std::vector<std::pair<int, int>> final_pos_edges;
    std::vector<std::pair<int, int>> final_color_edges;
    std::vector<std::pair<int, int>> final_normal_edges;
    std::vector<std::pair<int, int>> final_traversability_edges;
    model.get_multi_graph(final_nodes, final_pos_edges, final_color_edges,
                          final_normal_edges, final_traversability_edges);
    auto final_normals = model.get_node_normals();
    auto final_traversability = model.get_traversability();
    auto final_contour = model.get_contour();

    csv_writer::save_nodes_3d(final_nodes, output_dir + "/final_nodes.csv");
    csv_writer::save_edges(final_pos_edges, output_dir + "/final_pos_edges.csv");
    csv_writer::save_edges(final_normal_edges, output_dir + "/final_normal_edges.csv");
    csv_writer::save_edges(final_traversability_edges, output_dir + "/final_traversability_edges.csv");
    csv_writer::save_normals(final_normals, output_dir + "/final_normals.csv");
    csv_writer::save_traversability(final_traversability, output_dir + "/final_traversability.csv");
    csv_writer::save_contour(final_contour, output_dir + "/final_contour.csv");

    csv_writer::save_frames(frame_iterations, output_dir + "/frames.csv");

    // Count final traversable
    int n_traversable = 0;
    for (int t : final_traversability) {
        if (t == 1) ++n_traversable;
    }
    int n_contour = 0;
    for (int c : final_contour) {
        if (c == 1) ++n_contour;
    }

    csv_writer::MetadataWriter metadata(output_dir + "/metadata.csv");
    metadata.add("algorithm", "GNG-DT-Robot");
    metadata.add("n_iterations", n_iterations);
    metadata.add("n_samples", n_samples);
    metadata.add("max_nodes", params.max_nodes);
    metadata.add("time_ms", static_cast<int>(timer.elapsed_ms()));
    metadata.add("final_nodes", model.num_nodes());
    metadata.add("final_pos_edges", model.num_edges_pos());
    metadata.add("final_traversability_edges", model.num_edges_traversability());
    metadata.add("n_traversable", n_traversable);
    metadata.add("n_contour", n_contour);
    metadata.save();

    std::cout << "\nTraining complete!\n";
    std::cout << "Time: " << timer.elapsed_ms() << " ms\n";
    std::cout << "Final: " << model.num_nodes() << " nodes, "
              << n_traversable << " traversable, "
              << n_contour << " contour\n";
    std::cout << "Results saved to: " << output_dir << "\n";

    return 0;
}
