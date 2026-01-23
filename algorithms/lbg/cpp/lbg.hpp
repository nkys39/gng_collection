/**
 * @file lbg.hpp
 * @brief Linde-Buzo-Gray (LBG) algorithm implementation
 *
 * Based on:
 *   - Linde, Y., Buzo, A., & Gray, R. (1980). "An Algorithm for Vector Quantizer Design"
 *   - demogng.de reference implementation
 *
 * LBG is a batch learning algorithm for vector quantization that iteratively
 * assigns data points to clusters and moves centroids to cluster centers.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace lbg {

/**
 * @brief LBG hyperparameters.
 */
struct LBGParams {
    int n_nodes = 50;                       // Number of codebook vectors (fixed)
    int max_epochs = 100;                   // Maximum number of epochs
    float convergence_threshold = 1e-6f;    // Stop if distortion change is below this
    bool use_utility = false;               // Whether to use utility-based node management
    float utility_threshold = 0.01f;        // Threshold for removing low-utility nodes
};

/**
 * @brief Linde-Buzo-Gray vector quantization algorithm.
 *
 * LBG is a batch algorithm that works by:
 * 1. Assigning each data point to its nearest codebook vector
 * 2. Moving each codebook vector to the centroid of its assigned points
 * 3. Repeating until convergence
 *
 * When use_utility=true (LBG-U), nodes with low utility (few assignments)
 * can be removed and reinitialized in high-error regions.
 *
 * @tparam PointT Point type (Eigen vector, e.g., Eigen::Vector2f)
 */
template <typename PointT>
class LindeBuzoGray {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const LindeBuzoGray&, int)>;

    LBGParams params;
    std::vector<PointT> weights;        // Codebook vectors
    std::vector<float> utility;         // Assignment counts for each node
    std::vector<float> errors;          // Error accumulator for each node
    int n_learning = 0;

private:
    std::mt19937 rng_;

public:
    /**
     * @brief Construct LBG with given parameters.
     *
     * @param params LBG hyperparameters
     * @param seed Random seed (0 for random)
     */
    explicit LindeBuzoGray(const LBGParams& params = LBGParams(), unsigned int seed = 0)
        : params(params),
          weights(params.n_nodes),
          utility(params.n_nodes, 0.0f),
          errors(params.n_nodes, 0.0f),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        // Initialize codebook vectors randomly in [0, 1]
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (int i = 0; i < params.n_nodes; ++i) {
            for (int j = 0; j < PointT::RowsAtCompileTime; ++j) {
                weights[i](j) = dist(rng_);
            }
        }
    }

    /**
     * @brief Initialize codebook vectors from data range.
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
     * @brief Train on data using batch updates.
     *
     * @param data Vector of training samples
     * @param n_iterations Number of epochs (overrides max_epochs if > 0)
     * @param callback Optional callback(self, epoch)
     */
    void train(const std::vector<PointT>& data, int n_iterations = 0,
               const Callback& callback = nullptr) {
        if (data.empty()) return;

        int max_epochs = n_iterations > 0 ? n_iterations : params.max_epochs;
        Scalar prev_distortion = std::numeric_limits<Scalar>::max();

        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            // Assign data points to nearest codebook vectors
            std::vector<int> assignments = assign_to_nearest(data);

            // Compute distortion
            Scalar distortion = compute_distortion(data, assignments);

            // Update centroids
            update_centroids(data, assignments);

            // Handle utility-based management (LBG-U)
            handle_utility(data);

            n_learning++;

            if (callback) {
                callback(*this, epoch);
            }

            // Check convergence
            if (std::abs(prev_distortion - distortion) < params.convergence_threshold) {
                break;
            }

            prev_distortion = distortion;
        }
    }

    /**
     * @brief Single online learning step (not typical for LBG).
     *
     * @param sample Input sample
     */
    void partial_fit(const PointT& sample) {
        // Find nearest codebook vector
        int bmu_idx = find_bmu(sample);

        // Simple online update
        Scalar lr = 0.01;
        weights[bmu_idx] += lr * (sample - weights[bmu_idx]);

        n_learning++;
    }

    /**
     * @brief Get number of nodes (fixed).
     */
    int num_nodes() const {
        return params.n_nodes;
    }

    /**
     * @brief Get number of edges (always 0 for LBG - no topology).
     */
    int num_edges() const {
        return 0;
    }

    /**
     * @brief Get codebook vector positions.
     *
     * @return Copy of weights vector
     */
    std::vector<PointT> get_nodes() const {
        return weights;
    }

    /**
     * @brief Get edges (always empty for LBG).
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
     * @brief Get utility values for all nodes.
     */
    std::vector<float> get_node_utilities() const {
        return utility;
    }

    /**
     * @brief Get error values for all nodes.
     */
    std::vector<float> get_node_errors() const {
        return errors;
    }

    /**
     * @brief Compute mean quantization error on data.
     *
     * @param data Data points
     * @return Mean distance from each point to its nearest codebook vector
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
     * @brief Assign each data point to its nearest codebook vector.
     */
    std::vector<int> assign_to_nearest(const std::vector<PointT>& data) const {
        std::vector<int> assignments(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            assignments[i] = find_bmu(data[i]);
        }

        return assignments;
    }

    /**
     * @brief Compute total distortion (quantization error).
     */
    Scalar compute_distortion(const std::vector<PointT>& data,
                               const std::vector<int>& assignments) const {
        Scalar distortion = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            distortion += (data[i] - weights[assignments[i]]).squaredNorm();
        }
        return distortion;
    }

    /**
     * @brief Update codebook vectors to cluster centroids.
     */
    void update_centroids(const std::vector<PointT>& data,
                          const std::vector<int>& assignments) {
        // Reset utility and errors
        std::fill(utility.begin(), utility.end(), 0.0f);
        std::fill(errors.begin(), errors.end(), 0.0f);

        // Accumulate sums and counts
        std::vector<PointT> sums(params.n_nodes, PointT::Zero());
        std::vector<int> counts(params.n_nodes, 0);

        for (size_t i = 0; i < data.size(); ++i) {
            int cluster = assignments[i];
            sums[cluster] += data[i];
            counts[cluster]++;
            // Accumulate squared error
            errors[cluster] += static_cast<float>((data[i] - weights[cluster]).squaredNorm());
        }

        // Update utility (normalized assignment counts)
        float total_samples = static_cast<float>(data.size());
        for (int j = 0; j < params.n_nodes; ++j) {
            utility[j] = static_cast<float>(counts[j]) / total_samples;
        }

        // Update centroids (only for non-empty clusters)
        for (int j = 0; j < params.n_nodes; ++j) {
            if (counts[j] > 0) {
                weights[j] = sums[j] / static_cast<Scalar>(counts[j]);
            }
        }
    }

    /**
     * @brief Handle low-utility nodes (LBG-U).
     */
    void handle_utility(const std::vector<PointT>& data) {
        if (!params.use_utility) return;

        // Find max error node
        int max_error_idx = 0;
        float max_error = errors[0];
        for (int i = 1; i < params.n_nodes; ++i) {
            if (errors[i] > max_error) {
                max_error = errors[i];
                max_error_idx = i;
            }
        }

        // Reinitialize low-utility nodes near high-error region
        std::normal_distribution<Scalar> noise(0, 0.1);
        for (int i = 0; i < params.n_nodes; ++i) {
            if (utility[i] < params.utility_threshold) {
                weights[i] = weights[max_error_idx];
                for (int j = 0; j < PointT::RowsAtCompileTime; ++j) {
                    weights[i](j) += noise(rng_);
                }
            }
        }
    }
};

// Common type aliases
using LBG2f = LindeBuzoGray<Eigen::Vector2f>;
using LBG3f = LindeBuzoGray<Eigen::Vector3f>;
using LBG2d = LindeBuzoGray<Eigen::Vector2d>;
using LBG3d = LindeBuzoGray<Eigen::Vector3d>;

}  // namespace lbg
