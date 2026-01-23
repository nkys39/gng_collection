/**
 * @file hcl.hpp
 * @brief Hard Competitive Learning (HCL) implementation
 *
 * Based on:
 *   - Rumelhart, D. E., & Zipser, D. (1985). "Feature discovery by competitive learning"
 *   - demogng.de reference implementation
 *
 * HCL is the simplest competitive learning algorithm where only the winner
 * (Best Matching Unit) is updated for each input signal.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace hcl {

/**
 * @brief HCL hyperparameters.
 */
struct HCLParams {
    int n_nodes = 50;                   // Number of reference vectors (fixed)
    float learning_rate_initial = 0.5f; // Initial learning rate
    float learning_rate_final = 0.01f;  // Final learning rate
};

/**
 * @brief Hard Competitive Learning algorithm.
 *
 * HCL is a winner-take-all competitive learning algorithm where only
 * the closest node (Best Matching Unit) is updated for each input.
 * No neighborhood function is used - only the winner moves.
 *
 * This is the simplest form of competitive learning and forms the
 * basis for more sophisticated algorithms like SOM and Neural Gas.
 *
 * @tparam PointT Point type (Eigen vector, e.g., Eigen::Vector2f)
 */
template <typename PointT>
class HardCompetitiveLearning {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const HardCompetitiveLearning&, int)>;

    HCLParams params;
    std::vector<PointT> weights;  // Reference vectors
    int n_learning = 0;

private:
    std::mt19937 rng_;
    int total_iterations_ = 1;

public:
    /**
     * @brief Construct HCL with given parameters.
     *
     * @param params HCL hyperparameters
     * @param seed Random seed (0 for random)
     */
    explicit HardCompetitiveLearning(const HCLParams& params = HCLParams(), unsigned int seed = 0)
        : params(params),
          weights(params.n_nodes),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        // Initialize reference vectors randomly in [0, 1]
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (int i = 0; i < params.n_nodes; ++i) {
            for (int j = 0; j < PointT::RowsAtCompileTime; ++j) {
                weights[i](j) = dist(rng_);
            }
        }
    }

    /**
     * @brief Initialize reference vectors from data range.
     *
     * @param data Training data to determine initialization range
     */
    void init_from_data(const std::vector<PointT>& data) {
        if (data.empty()) return;

        // Find data bounds
        PointT min_val = data[0];
        PointT max_val = data[0];
        for (const auto& p : data) {
            for (int j = 0; j < PointT::RowsAtCompileTime; ++j) {
                min_val(j) = std::min(min_val(j), p(j));
                max_val(j) = std::max(max_val(j), p(j));
            }
        }

        // Initialize randomly within data bounds
        for (int i = 0; i < params.n_nodes; ++i) {
            for (int j = 0; j < PointT::RowsAtCompileTime; ++j) {
                std::uniform_real_distribution<Scalar> dist(min_val(j), max_val(j));
                weights[i](j) = dist(rng_);
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
     * @brief Single online learning step (uses final learning rate).
     *
     * @param sample Input sample
     */
    void partial_fit(const PointT& sample) {
        int bmu_idx = find_bmu(sample);
        weights[bmu_idx] += params.learning_rate_final * (sample - weights[bmu_idx]);
        n_learning++;
    }

    /**
     * @brief Get number of nodes (fixed).
     */
    int num_nodes() const {
        return params.n_nodes;
    }

    /**
     * @brief Get number of edges (always 0 for HCL - no topology).
     */
    int num_edges() const {
        return 0;
    }

    /**
     * @brief Get reference vector positions.
     *
     * @return Copy of weights vector
     */
    std::vector<PointT> get_nodes() const {
        return weights;
    }

    /**
     * @brief Get edges (always empty for HCL).
     */
    std::vector<std::pair<int, int>> get_edges() const {
        return {};
    }

    /**
     * @brief Get graph for visualization.
     *
     * @param out_nodes Output: node positions array
     * @param out_edges Output: edges (always empty)
     */
    void get_graph(std::vector<PointT>& out_nodes,
                   std::vector<std::pair<int, int>>& out_edges) const {
        out_nodes = weights;
        out_edges.clear();
    }

    /**
     * @brief Compute mean quantization error on data.
     *
     * @param data Data points
     * @return Mean distance from each point to its nearest reference vector
     */
    Scalar get_quantization_error(const std::vector<PointT>& data) const {
        if (data.empty()) return 0;

        Scalar total_error = 0;
        for (const auto& x : data) {
            Scalar min_dist = std::numeric_limits<Scalar>::max();
            for (const auto& w : weights) {
                Scalar dist = (x - w).squaredNorm();
                min_dist = std::min(min_dist, dist);
            }
            total_error += std::sqrt(min_dist);
        }
        return total_error / static_cast<Scalar>(data.size());
    }

private:
    /**
     * @brief Find the Best Matching Unit (BMU) for input x.
     *
     * @param x Input vector
     * @return Index of the BMU
     */
    int find_bmu(const PointT& x) const {
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        int bmu_idx = 0;

        for (int i = 0; i < params.n_nodes; ++i) {
            Scalar dist = (x - weights[i]).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                bmu_idx = i;
            }
        }

        return bmu_idx;
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

        // Exponential decay of learning rate
        Scalar lr = params.learning_rate_initial *
            std::pow(params.learning_rate_final / params.learning_rate_initial, t);

        // Find BMU
        int bmu_idx = find_bmu(sample);

        // Update only the winner (hard competitive learning)
        weights[bmu_idx] += lr * (sample - weights[bmu_idx]);

        n_learning++;
    }
};

// Common type aliases
using HCL2f = HardCompetitiveLearning<Eigen::Vector2f>;
using HCL3f = HardCompetitiveLearning<Eigen::Vector3f>;
using HCL2d = HardCompetitiveLearning<Eigen::Vector2d>;
using HCL3d = HardCompetitiveLearning<Eigen::Vector3d>;

}  // namespace hcl
