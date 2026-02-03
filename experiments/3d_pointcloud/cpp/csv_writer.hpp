/**
 * @file csv_writer.hpp
 * @brief CSV output utilities for 3D GNG test results
 */

#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace csv_writer {

namespace fs = std::filesystem;

/**
 * @brief Create output directory if it doesn't exist
 */
inline void ensure_directory(const std::string& path) {
    fs::create_directories(path);
}

/**
 * @brief Format iteration number with zero padding
 */
inline std::string format_iteration(int iteration, int width = 5) {
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(width) << iteration;
    return oss.str();
}

/**
 * @brief Save 3D sample points to CSV
 */
inline void save_samples_3d(
    const std::vector<Eigen::Vector3f>& samples,
    const std::string& path
) {
    std::ofstream file(path);
    file << "x,y,z\n";
    for (const auto& p : samples) {
        file << p.x() << "," << p.y() << "," << p.z() << "\n";
    }
}

/**
 * @brief Save 3D node positions to CSV
 */
inline void save_nodes_3d(
    const std::vector<Eigen::Vector3f>& nodes,
    const std::string& path
) {
    std::ofstream file(path);
    file << "x,y,z\n";
    for (const auto& p : nodes) {
        file << p.x() << "," << p.y() << "," << p.z() << "\n";
    }
}

/**
 * @brief Save edges to CSV
 */
inline void save_edges(
    const std::vector<std::pair<int, int>>& edges,
    const std::string& path
) {
    std::ofstream file(path);
    file << "i,j\n";
    for (const auto& [i, j] : edges) {
        file << i << "," << j << "\n";
    }
}

/**
 * @brief Save normal vectors to CSV
 */
inline void save_normals(
    const std::vector<Eigen::Vector3f>& normals,
    const std::string& path
) {
    std::ofstream file(path);
    file << "nx,ny,nz\n";
    for (const auto& n : normals) {
        file << n.x() << "," << n.y() << "," << n.z() << "\n";
    }
}

/**
 * @brief Save traversability values to CSV
 */
inline void save_traversability(
    const std::vector<int>& traversability,
    const std::string& path
) {
    std::ofstream file(path);
    file << "traversability\n";
    for (int t : traversability) {
        file << t << "\n";
    }
}

/**
 * @brief Save contour values to CSV
 */
inline void save_contour(
    const std::vector<int>& contour,
    const std::string& path
) {
    std::ofstream file(path);
    file << "contour\n";
    for (int c : contour) {
        file << c << "\n";
    }
}

/**
 * @brief Save frame iteration numbers to CSV
 */
inline void save_frames(
    const std::vector<int>& frames,
    const std::string& path
) {
    std::ofstream file(path);
    file << "iteration\n";
    for (int f : frames) {
        file << f << "\n";
    }
}

/**
 * @brief Metadata writer for test results
 */
class MetadataWriter {
public:
    MetadataWriter(const std::string& path) : path_(path) {}

    void add(const std::string& key, const std::string& value) {
        entries_.emplace_back(key, value);
    }

    void add(const std::string& key, int value) {
        add(key, std::to_string(value));
    }

    void add(const std::string& key, float value) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << value;
        add(key, oss.str());
    }

    void add(const std::string& key, double value) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << value;
        add(key, oss.str());
    }

    void save() const {
        std::ofstream file(path_);
        file << "key,value\n";
        for (const auto& [key, value] : entries_) {
            file << key << "," << value << "\n";
        }
    }

private:
    std::string path_;
    std::vector<std::pair<std::string, std::string>> entries_;
};

/**
 * @brief Timer utility for measuring execution time
 */
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
};

/**
 * @brief Helper class for saving GNG graph state
 */
template<typename Model>
void save_gng_graph_3d(
    const Model& model,
    const std::string& output_dir,
    int iteration
) {
    std::vector<Eigen::Vector3f> nodes;
    std::vector<std::pair<int, int>> edges;
    model.get_graph(nodes, edges);

    std::string iter_str = format_iteration(iteration);
    save_nodes_3d(nodes, output_dir + "/nodes_" + iter_str + ".csv");
    save_edges(edges, output_dir + "/edges_" + iter_str + ".csv");
}

/**
 * @brief Helper class for saving GNG-DT multi-graph state
 */
template<typename Model>
void save_gngdt_graph_3d(
    const Model& model,
    const std::string& output_dir,
    int iteration
) {
    std::vector<Eigen::Vector3f> nodes;
    std::vector<std::pair<int, int>> pos_edges;
    std::vector<std::pair<int, int>> color_edges;
    std::vector<std::pair<int, int>> normal_edges;
    model.get_multi_graph(nodes, pos_edges, color_edges, normal_edges);

    auto normals = model.get_node_normals();

    std::string iter_str = format_iteration(iteration);
    save_nodes_3d(nodes, output_dir + "/nodes_" + iter_str + ".csv");
    save_edges(pos_edges, output_dir + "/pos_edges_" + iter_str + ".csv");
    save_edges(normal_edges, output_dir + "/normal_edges_" + iter_str + ".csv");
    save_normals(normals, output_dir + "/normals_" + iter_str + ".csv");
}

/**
 * @brief Helper class for saving GNG-DT Robot multi-graph state
 */
template<typename Model>
void save_gngdt_robot_graph_3d(
    const Model& model,
    const std::string& output_dir,
    int iteration
) {
    std::vector<Eigen::Vector3f> nodes;
    std::vector<std::pair<int, int>> pos_edges;
    std::vector<std::pair<int, int>> color_edges;
    std::vector<std::pair<int, int>> normal_edges;
    std::vector<std::pair<int, int>> traversability_edges;
    model.get_multi_graph(nodes, pos_edges, color_edges, normal_edges, traversability_edges);

    auto normals = model.get_node_normals();
    auto traversability = model.get_traversability();
    auto contour = model.get_contour();

    std::string iter_str = format_iteration(iteration);
    save_nodes_3d(nodes, output_dir + "/nodes_" + iter_str + ".csv");
    save_edges(pos_edges, output_dir + "/pos_edges_" + iter_str + ".csv");
    save_edges(normal_edges, output_dir + "/normal_edges_" + iter_str + ".csv");
    save_edges(traversability_edges, output_dir + "/traversability_edges_" + iter_str + ".csv");
    save_normals(normals, output_dir + "/normals_" + iter_str + ".csv");
    save_traversability(traversability, output_dir + "/traversability_" + iter_str + ".csv");
    save_contour(contour, output_dir + "/contour_" + iter_str + ".csv");
}

}  // namespace csv_writer
