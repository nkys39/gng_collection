/**
 * @file gng_u.hpp
 * @brief Growing Neural Gas with Utility (GNG-U) implementation
 *
 * Based on:
 *   - Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
 *   - Fritzke, B. (1997). "Some Competitive Learning Methods"
 *   - Fritzke, B. (1999). "Be Busy and Unique - or Be History - The Utility Criterion"
 *
 * GNG-U extends GNG with a utility measure that allows tracking non-stationary
 * distributions by removing nodes with low utility.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace gng_u {

/**
 * @brief GNG-U hyperparameters.
 */
struct GNGUParams {
    int max_nodes = 100;    // Maximum number of nodes
    int lambda = 100;       // Node insertion interval
    float eps_b = 0.08f;    // Winner learning rate
    float eps_n = 0.008f;   // Neighbor learning rate
    float alpha = 0.5f;     // Error decay on split
    float beta = 0.005f;    // Global error/utility decay
    int max_age = 100;      // Maximum edge age
    float utility_k = 1.3f; // Utility threshold for node removal
};

/**
 * @brief A neuron node in the GNG-U network.
 *
 * @tparam PointT Point type (Eigen vector)
 */
template <typename PointT>
struct NeuronNodeU {
    int id = -1;            // -1 means invalid/removed
    float error = 1.0f;     // Accumulated error
    float utility = 0.0f;   // Utility measure (GNG-U specific)
    PointT weight;          // Position vector

    NeuronNodeU() = default;
    NeuronNodeU(int id_, const PointT& weight_, float utility_ = 0.0f)
        : id(id_), error(1.0f), utility(utility_), weight(weight_) {}
};

/**
 * @brief Growing Neural Gas with Utility algorithm.
 *
 * GNG-U extends the standard GNG algorithm with a utility measure that
 * tracks how useful each node is for reducing the network error.
 * Nodes with low utility can be removed, allowing the network to
 * track non-stationary distributions.
 *
 * @tparam PointT Point type (Eigen vector, e.g., Eigen::Vector2f)
 */
template <typename PointT>
class GrowingNeuralGasU {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GrowingNeuralGasU&, int)>;

    GNGUParams params;
    std::vector<NeuronNodeU<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edges;  // Edge age matrix (0 = no edge, >=1 = connected)
    int n_learning = 0;
    int n_removals = 0;     // Number of utility-based removals

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    std::mt19937 rng_;

public:
    /**
     * @brief Construct GNG-U with given parameters.
     *
     * @param params GNG-U hyperparameters
     * @param seed Random seed (0 for random)
     */
    explicit GrowingNeuralGasU(const GNGUParams& params = GNGUParams(), unsigned int seed = 0)
        : params(params),
          nodes(params.max_nodes),
          edges(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        // Initialize addable indices queue
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }
    }

    /**
     * @brief Initialize with two random nodes in [0,1] range.
     */
    void init() {
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (int i = 0; i < 2; ++i) {
            PointT weight;
            for (int j = 0; j < weight.size(); ++j) {
                weight(j) = dist(rng_);
            }
            add_node(weight);
        }
    }

    /**
     * @brief Initialize with two specific points.
     *
     * @param p1 First node position
     * @param p2 Second node position
     */
    void init(const PointT& p1, const PointT& p2) {
        add_node(p1);
        add_node(p2);
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

        std::uniform_int_distribution<int> dist(0, static_cast<int>(data.size()) - 1);

        for (int iter = 0; iter < n_iterations; ++iter) {
            int idx = dist(rng_);
            one_train_update(data[idx]);

            if (callback) {
                callback(*this, iter);
            }
        }
    }

    /**
     * @brief Single online learning step.
     *
     * @param sample Input sample
     */
    void partial_fit(const PointT& sample) {
        one_train_update(sample);
    }

    /**
     * @brief Get number of active nodes.
     */
    int num_nodes() const {
        int count = 0;
        for (const auto& node : nodes) {
            if (node.id != -1) ++count;
        }
        return count;
    }

    /**
     * @brief Get number of edges.
     */
    int num_edges() const {
        int count = 0;
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id != -1) {
                count += static_cast<int>(neighbors.size());
            }
        }
        return count / 2;  // Each edge counted twice
    }

    /**
     * @brief Get active node positions.
     *
     * @return Vector of (id, position) pairs
     */
    std::vector<std::pair<int, PointT>> get_nodes() const {
        std::vector<std::pair<int, PointT>> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.emplace_back(node.id, node.weight);
            }
        }
        return result;
    }

    /**
     * @brief Get edges as (id1, id2) pairs.
     */
    std::vector<std::pair<int, int>> get_edges() const {
        std::vector<std::pair<int, int>> result;
        std::unordered_set<int64_t> seen;

        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id == -1) continue;
            for (int neighbor_id : neighbors) {
                if (nodes[neighbor_id].id == -1) continue;
                int64_t key = std::min(node_id, neighbor_id) * 10000LL + std::max(node_id, neighbor_id);
                if (seen.find(key) == seen.end()) {
                    seen.insert(key);
                    result.emplace_back(node_id, neighbor_id);
                }
            }
        }
        return result;
    }

    /**
     * @brief Get graph with sequential indices (for visualization).
     *
     * @param out_nodes Output: node positions array
     * @param out_edges Output: edges with new sequential indices
     */
    void get_graph(std::vector<PointT>& out_nodes,
                   std::vector<std::pair<int, int>>& out_edges) const {
        out_nodes.clear();
        out_edges.clear();

        std::unordered_map<int, int> id_to_idx;

        // Collect active nodes
        for (const auto& node : nodes) {
            if (node.id != -1) {
                id_to_idx[node.id] = static_cast<int>(out_nodes.size());
                out_nodes.push_back(node.weight);
            }
        }

        // Convert edges to new indices
        std::unordered_set<int64_t> seen;
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id == -1) continue;
            for (int neighbor_id : neighbors) {
                if (nodes[neighbor_id].id == -1) continue;
                int64_t key = std::min(node_id, neighbor_id) * 10000LL + std::max(node_id, neighbor_id);
                if (seen.find(key) == seen.end()) {
                    seen.insert(key);
                    out_edges.emplace_back(id_to_idx[node_id], id_to_idx[neighbor_id]);
                }
            }
        }
    }

    /**
     * @brief Get utility values for active nodes.
     *
     * @return Vector of utilities in same order as get_graph() nodes
     */
    std::vector<float> get_node_utilities() const {
        std::vector<float> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.utility);
            }
        }
        return result;
    }

    /**
     * @brief Get error values for active nodes.
     *
     * @return Vector of errors in same order as get_graph() nodes
     */
    std::vector<float> get_node_errors() const {
        std::vector<float> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.error);
            }
        }
        return result;
    }

private:
    /**
     * @brief Add a new node.
     *
     * @param weight Node position
     * @param utility Initial utility value
     * @return Node ID, or -1 if no space
     */
    int add_node(const PointT& weight, float utility = 0.0f) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = NeuronNodeU<PointT>(node_id, weight, utility);
        edges_per_node[node_id] = std::unordered_set<int>();
        return node_id;
    }

    /**
     * @brief Remove a node.
     *
     * @param node_id ID of node to remove
     * @param force If true, remove even if node has edges (for utility removal)
     * @return true if node was removed
     */
    bool remove_node(int node_id, bool force = false) {
        auto it = edges_per_node.find(node_id);
        if (!force && it != edges_per_node.end() && !it->second.empty()) {
            return false;  // Has edges, don't remove
        }

        // Remove all edges connected to this node
        if (it != edges_per_node.end()) {
            for (int neighbor_id : std::vector<int>(it->second.begin(), it->second.end())) {
                remove_edge(node_id, neighbor_id);
            }
        }

        edges_per_node.erase(node_id);
        nodes[node_id].id = -1;
        addable_indices_.push_back(node_id);
        return true;
    }

    /**
     * @brief Add or reset edge between two nodes.
     */
    void add_edge(int node1, int node2) {
        if (edges(node1, node2) > 0) {
            // Reset age
            edges(node1, node2) = 1;
            edges(node2, node1) = 1;
        } else {
            // New edge
            edges_per_node[node1].insert(node2);
            edges_per_node[node2].insert(node1);
            edges(node1, node2) = 1;
            edges(node2, node1) = 1;
        }
    }

    /**
     * @brief Remove edge between two nodes.
     */
    void remove_edge(int node1, int node2) {
        edges_per_node[node1].erase(node2);
        edges_per_node[node2].erase(node1);
        edges(node1, node2) = 0;
        edges(node2, node1) = 0;
    }

    /**
     * @brief Find two nearest nodes with squared distances.
     *
     * @return tuple of (winner_id, second_winner_id, dist1_sq, dist2_sq)
     */
    std::tuple<int, int, Scalar, Scalar> find_two_nearest(const PointT& x) const {
        Scalar min_dist1 = std::numeric_limits<Scalar>::max();
        Scalar min_dist2 = std::numeric_limits<Scalar>::max();
        int s1_id = -1, s2_id = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;

            Scalar dist = (x - node.weight).squaredNorm();

            if (dist < min_dist1) {
                min_dist2 = min_dist1;
                s2_id = s1_id;
                min_dist1 = dist;
                s1_id = node.id;
            } else if (dist < min_dist2) {
                min_dist2 = dist;
                s2_id = node.id;
            }
        }

        return {s1_id, s2_id, min_dist1, min_dist2};
    }

    /**
     * @brief Check and remove node with lowest utility if criterion met.
     *
     * GNG-U specific: Remove node if max_error / min_utility > k
     */
    void check_utility_criterion() {
        float max_error = 0.0f;
        float min_utility = std::numeric_limits<float>::max();
        int min_utility_id = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;
            if (node.error > max_error) {
                max_error = node.error;
            }
            if (node.utility < min_utility) {
                min_utility = node.utility;
                min_utility_id = node.id;
            }
        }

        // Check criterion: max_error / min_utility > k
        if (min_utility_id != -1 && min_utility > 0) {
            if (max_error / min_utility > params.utility_k) {
                // Remove node with minimum utility
                if (remove_node(min_utility_id, true)) {
                    n_removals++;
                }
            }
        }
    }

    /**
     * @brief Single training update with utility.
     */
    void one_train_update(const PointT& sample) {
        // Decay all errors and utilities
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error -= params.beta * node.error;
            node.utility -= params.beta * node.utility;  // GNG-U: decay utility
        }

        // Find two nearest nodes (with squared distances)
        auto [s1_id, s2_id, dist1_sq, dist2_sq] = find_two_nearest(sample);
        if (s1_id == -1 || s2_id == -1) return;

        // Update winner error (squared distance)
        nodes[s1_id].error += dist1_sq;

        // GNG-U: Update winner utility (squared distance difference)
        // Utility represents how much error would increase if this node were removed
        nodes[s1_id].utility += dist2_sq - dist1_sq;

        // Move winner toward sample
        nodes[s1_id].weight += params.eps_b * (sample - nodes[s1_id].weight);

        // Connect s1 and s2
        add_edge(s1_id, s2_id);

        // Update neighbors and age edges
        std::vector<int> edges_to_remove;
        for (int neighbor_id : edges_per_node[s1_id]) {
            if (edges(s1_id, neighbor_id) > params.max_age) {
                edges_to_remove.push_back(neighbor_id);
            } else {
                // Move neighbor toward sample
                nodes[neighbor_id].weight += params.eps_n * (sample - nodes[neighbor_id].weight);
                // Increment edge age
                edges(s1_id, neighbor_id)++;
                edges(neighbor_id, s1_id)++;
            }
        }

        // Remove old edges and isolated nodes
        for (int neighbor_id : edges_to_remove) {
            remove_edge(s1_id, neighbor_id);
            if (edges_per_node[neighbor_id].empty()) {
                remove_node(neighbor_id);
            }
        }

        // Periodically add new node
        n_trial_++;
        if (n_trial_ >= params.lambda) {
            n_trial_ = 0;

            if (!addable_indices_.empty()) {
                // Find node with maximum error
                float max_err_q = 0.0f;
                int q_id = -1;
                for (const auto& node : nodes) {
                    if (node.id == -1) continue;
                    if (node.error > max_err_q) {
                        max_err_q = node.error;
                        q_id = node.id;
                    }
                }

                if (q_id == -1) return;

                // Find neighbor of q with maximum error
                float max_err_f = 0.0f;
                int f_id = -1;
                auto it = edges_per_node.find(q_id);
                if (it != edges_per_node.end()) {
                    for (int neighbor_id : it->second) {
                        if (nodes[neighbor_id].error > max_err_f) {
                            max_err_f = nodes[neighbor_id].error;
                            f_id = neighbor_id;
                        }
                    }
                }

                if (f_id == -1) return;

                // Add new node between q and f
                PointT new_weight = (nodes[q_id].weight + nodes[f_id].weight) * static_cast<Scalar>(0.5);

                // GNG-U: New node inherits averaged utility
                float new_utility = (nodes[q_id].utility + nodes[f_id].utility) * 0.5f;
                int new_id = add_node(new_weight, new_utility);

                if (new_id == -1) return;

                // Update edges
                remove_edge(q_id, f_id);
                add_edge(q_id, new_id);
                add_edge(f_id, new_id);

                // Update errors
                nodes[q_id].error *= (1.0f - params.alpha);
                nodes[f_id].error *= (1.0f - params.alpha);
                nodes[new_id].error = (nodes[q_id].error + nodes[f_id].error) * 0.5f;

                // GNG-U: Check utility criterion after insertion
                check_utility_criterion();
            }
        }

        n_learning++;
    }
};

// Common type aliases
using GNGU2f = GrowingNeuralGasU<Eigen::Vector2f>;
using GNGU3f = GrowingNeuralGasU<Eigen::Vector3f>;
using GNGU2d = GrowingNeuralGasU<Eigen::Vector2d>;
using GNGU3d = GrowingNeuralGasU<Eigen::Vector3d>;

}  // namespace gng_u
