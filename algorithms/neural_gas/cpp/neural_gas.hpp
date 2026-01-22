/**
 * @file neural_gas.hpp
 * @brief Neural Gas algorithm implementation
 *
 * Based on:
 *   - Martinetz, T. and Schulten, K. (1991). "A Neural-Gas Network Learns Topologies"
 *   - Martinetz, T. and Schulten, K. (1994). "Topology Representing Networks"
 *
 * Neural Gas uses a rank-based neighborhood function where all nodes are
 * updated with strength decreasing exponentially with their distance rank.
 * Combined with Competitive Hebbian Learning (CHL) to learn topology.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace neural_gas {

/**
 * @brief Neural Gas hyperparameters.
 */
struct NGParams {
    int n_nodes = 50;              // Number of reference vectors (fixed)
    float lambda_initial = 10.0f;  // Initial neighborhood range
    float lambda_final = 0.1f;     // Final neighborhood range
    float eps_initial = 0.5f;      // Initial learning rate
    float eps_final = 0.005f;      // Final learning rate
    int max_age = 50;              // Maximum edge age (for CHL)
    bool use_chl = true;           // Use Competitive Hebbian Learning for edges
};

/**
 * @brief Neural Gas algorithm.
 *
 * Neural Gas performs vector quantization with soft competitive learning.
 * All reference vectors are updated for each input, with adaptation strength
 * decreasing exponentially with the rank (distance order).
 *
 * When use_chl=true, edges are created between the two closest nodes
 * (Competitive Hebbian Learning) to learn the data topology.
 *
 * @tparam PointT Point type (Eigen vector, e.g., Eigen::Vector2f)
 */
template <typename PointT>
class NeuralGas {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const NeuralGas&, int)>;

    NGParams params;
    std::vector<PointT> weights;                    // Reference vectors
    Eigen::MatrixXi edges;                          // Edge age matrix (0 = no edge)
    int n_learning = 0;

private:
    std::mt19937 rng_;
    int total_iterations_ = 1;

public:
    /**
     * @brief Construct Neural Gas with given parameters.
     *
     * @param params Neural Gas hyperparameters
     * @param seed Random seed (0 for random)
     */
    explicit NeuralGas(const NGParams& params = NGParams(), unsigned int seed = 0)
        : params(params),
          weights(params.n_nodes),
          edges(Eigen::MatrixXi::Zero(params.n_nodes, params.n_nodes)),
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
     * @brief Single online learning step (uses final parameter values).
     *
     * @param sample Input sample
     */
    void partial_fit(const PointT& sample) {
        // Use final values for online learning
        Scalar lambda_t = params.lambda_final;
        Scalar eps_t = params.eps_final;

        // Get ranks
        std::vector<int> ranks = get_ranks(sample);

        // Compute neighborhood function and update all weights
        for (int i = 0; i < params.n_nodes; ++i) {
            Scalar h = std::exp(-static_cast<Scalar>(ranks[i]) / lambda_t);
            weights[i] += eps_t * h * (sample - weights[i]);
        }

        // CHL
        if (params.use_chl) {
            auto [s1, s2] = find_two_nearest(sample);

            // Age existing edges from winner
            for (int j = 0; j < params.n_nodes; ++j) {
                if (edges(s1, j) > 0) {
                    edges(s1, j)++;
                    edges(j, s1)++;
                }
            }

            // Create/reset edge between s1 and s2
            edges(s1, s2) = 1;
            edges(s2, s1) = 1;

            // Remove old edges
            for (int i = 0; i < params.n_nodes; ++i) {
                for (int j = i + 1; j < params.n_nodes; ++j) {
                    if (edges(i, j) > params.max_age) {
                        edges(i, j) = 0;
                        edges(j, i) = 0;
                    }
                }
            }
        }

        n_learning++;
    }

    /**
     * @brief Get number of nodes (always fixed).
     */
    int num_nodes() const {
        return params.n_nodes;
    }

    /**
     * @brief Get number of edges.
     */
    int num_edges() const {
        int count = 0;
        for (int i = 0; i < params.n_nodes; ++i) {
            for (int j = i + 1; j < params.n_nodes; ++j) {
                if (edges(i, j) > 0) count++;
            }
        }
        return count;
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
     * @brief Get edges as (i, j) pairs.
     */
    std::vector<std::pair<int, int>> get_edges() const {
        std::vector<std::pair<int, int>> result;
        for (int i = 0; i < params.n_nodes; ++i) {
            for (int j = i + 1; j < params.n_nodes; ++j) {
                if (edges(i, j) > 0) {
                    result.emplace_back(i, j);
                }
            }
        }
        return result;
    }

    /**
     * @brief Get graph for visualization (same format as GNG).
     *
     * @param out_nodes Output: node positions array
     * @param out_edges Output: edges as (i, j) pairs
     */
    void get_graph(std::vector<PointT>& out_nodes,
                   std::vector<std::pair<int, int>>& out_edges) const {
        out_nodes = weights;
        out_edges = get_edges();
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
     * @brief Get distance ranks for all nodes given input x.
     *
     * @param x Input vector
     * @return Vector of ranks (0 = closest, 1 = second closest, etc.)
     */
    std::vector<int> get_ranks(const PointT& x) const {
        // Compute distances
        std::vector<std::pair<Scalar, int>> dist_idx(params.n_nodes);
        for (int i = 0; i < params.n_nodes; ++i) {
            dist_idx[i] = {(x - weights[i]).squaredNorm(), i};
        }

        // Sort by distance
        std::sort(dist_idx.begin(), dist_idx.end());

        // Assign ranks
        std::vector<int> ranks(params.n_nodes);
        for (int rank = 0; rank < params.n_nodes; ++rank) {
            ranks[dist_idx[rank].second] = rank;
        }

        return ranks;
    }

    /**
     * @brief Find two nearest nodes.
     *
     * @return (winner_index, second_winner_index)
     */
    std::pair<int, int> find_two_nearest(const PointT& x) const {
        Scalar min_dist1 = std::numeric_limits<Scalar>::max();
        Scalar min_dist2 = std::numeric_limits<Scalar>::max();
        int s1 = 0, s2 = 1;

        for (int i = 0; i < params.n_nodes; ++i) {
            Scalar dist = (x - weights[i]).squaredNorm();

            if (dist < min_dist1) {
                min_dist2 = min_dist1;
                s2 = s1;
                min_dist1 = dist;
                s1 = i;
            } else if (dist < min_dist2) {
                min_dist2 = dist;
                s2 = i;
            }
        }

        return {s1, s2};
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

        // Exponential decay of lambda and epsilon
        Scalar lambda_t = params.lambda_initial *
            std::pow(params.lambda_final / params.lambda_initial, t);
        Scalar eps_t = params.eps_initial *
            std::pow(params.eps_final / params.eps_initial, t);

        // Get ranks
        std::vector<int> ranks = get_ranks(sample);

        // Compute neighborhood function and update all weights
        for (int i = 0; i < params.n_nodes; ++i) {
            Scalar h = std::exp(-static_cast<Scalar>(ranks[i]) / lambda_t);
            weights[i] += eps_t * h * (sample - weights[i]);
        }

        // Competitive Hebbian Learning: connect two closest nodes
        if (params.use_chl) {
            auto [s1, s2] = find_two_nearest(sample);

            // Age existing edges from winner (only where edges exist)
            for (int j = 0; j < params.n_nodes; ++j) {
                if (edges(s1, j) > 0) {
                    edges(s1, j)++;
                    edges(j, s1)++;
                }
            }

            // Create/reset edge between s1 and s2
            edges(s1, s2) = 1;
            edges(s2, s1) = 1;

            // Remove old edges
            for (int i = 0; i < params.n_nodes; ++i) {
                for (int j = i + 1; j < params.n_nodes; ++j) {
                    if (edges(i, j) > params.max_age) {
                        edges(i, j) = 0;
                        edges(j, i) = 0;
                    }
                }
            }
        }

        n_learning++;
    }
};

// Common type aliases
using NG2f = NeuralGas<Eigen::Vector2f>;
using NG3f = NeuralGas<Eigen::Vector3f>;
using NG2d = NeuralGas<Eigen::Vector2d>;
using NG3d = NeuralGas<Eigen::Vector3d>;

}  // namespace neural_gas
