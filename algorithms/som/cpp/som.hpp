/**
 * @file som.hpp
 * @brief Self-Organizing Map (SOM) implementation
 *
 * Based on:
 *   - Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps"
 *   - Kohonen, T. (2001). "Self-Organizing Maps" (3rd ed.)
 *
 * SOM uses a fixed grid topology where neurons are arranged in a 2D lattice.
 * The neighborhood function is based on grid distance, not data space distance.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace som {

/**
 * @brief SOM hyperparameters.
 */
struct SOMParams {
    int grid_height = 10;               // Height of the neuron grid
    int grid_width = 10;                // Width of the neuron grid
    float sigma_initial = 5.0f;         // Initial neighborhood radius
    float sigma_final = 0.5f;           // Final neighborhood radius
    float learning_rate_initial = 0.5f; // Initial learning rate
    float learning_rate_final = 0.01f;  // Final learning rate
};

/**
 * @brief Self-Organizing Map (Kohonen Map) implementation.
 *
 * A neural network with a fixed 2D grid topology that learns to represent
 * the input data distribution while preserving topological properties.
 *
 * @tparam PointT Point type (Eigen vector, e.g., Eigen::Vector2f)
 */
template <typename PointT>
class SelfOrganizingMap {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const SelfOrganizingMap&, int)>;

    SOMParams params;
    std::vector<std::vector<PointT>> weights;  // Grid of weight vectors [height][width]
    int n_learning = 0;

private:
    std::mt19937 rng_;
    int total_iterations_ = 1;

public:
    /**
     * @brief Construct SOM with given parameters.
     *
     * @param params SOM hyperparameters
     * @param seed Random seed (0 for random)
     */
    explicit SelfOrganizingMap(const SOMParams& params = SOMParams(), unsigned int seed = 0)
        : params(params),
          weights(params.grid_height, std::vector<PointT>(params.grid_width)),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        // Initialize weights randomly in [0, 1]
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (int i = 0; i < params.grid_height; ++i) {
            for (int j = 0; j < params.grid_width; ++j) {
                for (int k = 0; k < PointT::RowsAtCompileTime; ++k) {
                    weights[i][j](k) = dist(rng_);
                }
            }
        }
    }

    /**
     * @brief Initialize weights from data range.
     *
     * @param data Training data to determine initialization range
     */
    void init_from_data(const std::vector<PointT>& data) {
        if (data.empty()) return;

        // Find data bounds
        PointT min_val = data[0];
        PointT max_val = data[0];
        for (const auto& p : data) {
            for (int k = 0; k < PointT::RowsAtCompileTime; ++k) {
                min_val(k) = std::min(min_val(k), p(k));
                max_val(k) = std::max(max_val(k), p(k));
            }
        }

        // Initialize randomly within data bounds
        for (int i = 0; i < params.grid_height; ++i) {
            for (int j = 0; j < params.grid_width; ++j) {
                for (int k = 0; k < PointT::RowsAtCompileTime; ++k) {
                    std::uniform_real_distribution<Scalar> dist(min_val(k), max_val(k));
                    weights[i][j](k) = dist(rng_);
                }
            }
        }
    }

    /**
     * @brief Train on data for multiple iterations.
     *
     * @param data Vector of training samples
     * @param n_iterations Number of training iterations
     * @param callback Optional callback(self, iteration)
     */
    void train(const std::vector<PointT>& data, int n_iterations,
               const Callback& callback = nullptr) {
        if (data.empty()) return;

        total_iterations_ = n_iterations;
        std::uniform_int_distribution<int> dist(0, static_cast<int>(data.size()) - 1);

        for (int iter = 0; iter < n_iterations; ++iter) {
            int idx = dist(rng_);
            one_train_update(data[idx], iter, n_iterations);

            if (callback) {
                callback(*this, iter);
            }
        }
    }

    /**
     * @brief Single online learning step (uses final parameter values).
     *
     * @param sample Input sample
     */
    void partial_fit(const PointT& sample) {
        auto [bmu_i, bmu_j] = get_bmu(sample);

        // Use final values for online learning
        Scalar sigma = params.sigma_final;
        Scalar lr = params.learning_rate_final;

        // Update all weights
        for (int i = 0; i < params.grid_height; ++i) {
            for (int j = 0; j < params.grid_width; ++j) {
                Scalar h = get_neighborhood_value(bmu_i, bmu_j, i, j, sigma);
                weights[i][j] += lr * h * (sample - weights[i][j]);
            }
        }

        n_learning++;
    }

    /**
     * @brief Get number of nodes (fixed).
     */
    int num_nodes() const {
        return params.grid_height * params.grid_width;
    }

    /**
     * @brief Get number of grid edges.
     */
    int num_edges() const {
        // Horizontal edges + vertical edges
        int h_edges = params.grid_height * (params.grid_width - 1);
        int v_edges = (params.grid_height - 1) * params.grid_width;
        return h_edges + v_edges;
    }

    /**
     * @brief Get node positions.
     *
     * @return Vector of node positions (flattened grid, row-major)
     */
    std::vector<PointT> get_nodes() const {
        std::vector<PointT> result;
        result.reserve(num_nodes());
        for (int i = 0; i < params.grid_height; ++i) {
            for (int j = 0; j < params.grid_width; ++j) {
                result.push_back(weights[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Get edges based on grid topology.
     *
     * @return Vector of (i, j) pairs
     */
    std::vector<std::pair<int, int>> get_edges() const {
        std::vector<std::pair<int, int>> edges;
        edges.reserve(num_edges());

        for (int i = 0; i < params.grid_height; ++i) {
            for (int j = 0; j < params.grid_width; ++j) {
                int idx = i * params.grid_width + j;

                // Right neighbor
                if (j < params.grid_width - 1) {
                    edges.emplace_back(idx, idx + 1);
                }

                // Bottom neighbor
                if (i < params.grid_height - 1) {
                    edges.emplace_back(idx, idx + params.grid_width);
                }
            }
        }

        return edges;
    }

    /**
     * @brief Get graph for visualization.
     *
     * @param out_nodes Output: node positions array
     * @param out_edges Output: edges as (i, j) pairs
     */
    void get_graph(std::vector<PointT>& out_nodes,
                   std::vector<std::pair<int, int>>& out_edges) const {
        out_nodes = get_nodes();
        out_edges = get_edges();
    }

    /**
     * @brief Find the Best Matching Unit (BMU) for input x.
     *
     * @param x Input vector
     * @return Pair of (row, col) indices of the BMU
     */
    std::pair<int, int> get_bmu(const PointT& x) const {
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        int bmu_i = 0, bmu_j = 0;

        for (int i = 0; i < params.grid_height; ++i) {
            for (int j = 0; j < params.grid_width; ++j) {
                Scalar dist = (x - weights[i][j]).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    bmu_i = i;
                    bmu_j = j;
                }
            }
        }

        return {bmu_i, bmu_j};
    }

private:
    /**
     * @brief Compute neighborhood function value.
     *
     * Uses Manhattan distance on the grid per demogng.de reference.
     *
     * @param bmu_i BMU row
     * @param bmu_j BMU column
     * @param i Target row
     * @param j Target column
     * @param sigma Current neighborhood radius
     * @return Neighborhood value (0 to 1)
     */
    Scalar get_neighborhood_value(int bmu_i, int bmu_j, int i, int j, Scalar sigma) const {
        // Manhattan distance on grid
        Scalar grid_dist = static_cast<Scalar>(std::abs(i - bmu_i) + std::abs(j - bmu_j));

        // Gaussian neighborhood function: exp(-d^2 / (2 * sigma^2))
        return std::exp(-(grid_dist * grid_dist) / (2 * sigma * sigma));
    }

    /**
     * @brief Single training update.
     *
     * @param sample Input sample
     * @param iteration Current iteration
     * @param total_iterations Total iterations for decay calculation
     */
    void one_train_update(const PointT& sample, int iteration, int total_iterations) {
        // Compute decay factor
        Scalar t = static_cast<Scalar>(iteration) / std::max(1, total_iterations - 1);

        // Exponential decay of sigma and learning rate
        Scalar sigma = params.sigma_initial *
            std::pow(params.sigma_final / params.sigma_initial, t);
        Scalar lr = params.learning_rate_initial *
            std::pow(params.learning_rate_final / params.learning_rate_initial, t);

        // Find BMU
        auto [bmu_i, bmu_j] = get_bmu(sample);

        // Update all weights
        for (int i = 0; i < params.grid_height; ++i) {
            for (int j = 0; j < params.grid_width; ++j) {
                Scalar h = get_neighborhood_value(bmu_i, bmu_j, i, j, sigma);
                weights[i][j] += lr * h * (sample - weights[i][j]);
            }
        }

        n_learning++;
    }
};

// Common type aliases
using SOM2f = SelfOrganizingMap<Eigen::Vector2f>;
using SOM3f = SelfOrganizingMap<Eigen::Vector3f>;
using SOM2d = SelfOrganizingMap<Eigen::Vector2d>;
using SOM3d = SelfOrganizingMap<Eigen::Vector3d>;

}  // namespace som
