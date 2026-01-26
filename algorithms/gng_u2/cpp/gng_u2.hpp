/**
 * @file gng_u2.hpp
 * @brief GNG-U2 (GNG with Utility V2) implementation
 *
 * Based on:
 *   - Toda, Y., & Kubota, N. (2016). "Self-Localization Based on Multiresolution
 *     Map for Remote Control of Multiple Mobile Robots"
 *
 * Key differences from GNG-U:
 * 1. Uses Euclidean distance (not squared) for error and utility
 * 2. Utility check at κ-interval (not λ-interval)
 * 3. Separate decay rate χ for utility (optional, defaults to β)
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

namespace gng_u2 {

/**
 * @brief GNG-U2 hyperparameters.
 */
struct GNGU2Params {
    int max_nodes = 100;    // Maximum number of nodes
    int lambda = 100;       // Node insertion interval
    float eps_b = 0.08f;    // Winner learning rate
    float eps_n = 0.008f;   // Neighbor learning rate
    float alpha = 0.5f;     // Error decay on split
    float beta = 0.005f;    // Global error decay
    float chi = 0.005f;     // Utility decay rate (GNG-U2 specific)
    int max_age = 88;       // Maximum edge age
    float utility_k = 1000.0f; // Utility threshold (higher for Euclidean distance)
    int kappa = 10;         // Utility check interval (GNG-U2 specific)
};

/**
 * @brief A neuron node in the GNG-U2 network.
 */
template <typename PointT>
struct NeuronNodeU2 {
    int id = -1;
    float error = 1.0f;
    float utility = 0.0f;
    PointT weight;

    NeuronNodeU2() = default;
    NeuronNodeU2(int id_, const PointT& weight_, float error_ = 1.0f, float utility_ = 0.0f)
        : id(id_), error(error_), utility(utility_), weight(weight_) {}
};

/**
 * @brief GNG-U2 algorithm implementation.
 *
 * @tparam PointT Point type (Eigen vector)
 */
template <typename PointT>
class GrowingNeuralGasU2 {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GrowingNeuralGasU2&, int)>;

    GNGU2Params params;
    std::vector<NeuronNodeU2<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edges;
    int n_learning = 0;
    int n_removals = 0;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    std::mt19937 rng_;

public:
    explicit GrowingNeuralGasU2(const GNGU2Params& params = GNGU2Params(), unsigned int seed = 0)
        : params(params),
          nodes(params.max_nodes),
          edges(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }
    }

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

    void init(const PointT& p1, const PointT& p2) {
        add_node(p1);
        add_node(p2);
    }

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

    void partial_fit(const PointT& sample) {
        one_train_update(sample);
    }

    int num_nodes() const {
        int count = 0;
        for (const auto& node : nodes) {
            if (node.id != -1) ++count;
        }
        return count;
    }

    int num_edges() const {
        int count = 0;
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id != -1) {
                count += static_cast<int>(neighbors.size());
            }
        }
        return count / 2;
    }

    void get_graph(std::vector<PointT>& out_nodes,
                   std::vector<std::pair<int, int>>& out_edges) const {
        out_nodes.clear();
        out_edges.clear();

        std::unordered_map<int, int> id_to_idx;

        for (const auto& node : nodes) {
            if (node.id != -1) {
                id_to_idx[node.id] = static_cast<int>(out_nodes.size());
                out_nodes.push_back(node.weight);
            }
        }

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

    std::vector<float> get_node_utilities() const {
        std::vector<float> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.utility);
            }
        }
        return result;
    }

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
    int add_node(const PointT& weight, float error = 1.0f, float utility = 0.0f) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = NeuronNodeU2<PointT>(node_id, weight, error, utility);
        edges_per_node[node_id] = std::unordered_set<int>();
        return node_id;
    }

    bool remove_node(int node_id, bool force = false) {
        auto it = edges_per_node.find(node_id);
        if (!force && it != edges_per_node.end() && !it->second.empty()) {
            return false;
        }

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

    void add_edge(int node1, int node2) {
        if (edges(node1, node2) > 0) {
            edges(node1, node2) = 1;
            edges(node2, node1) = 1;
        } else {
            edges_per_node[node1].insert(node2);
            edges_per_node[node2].insert(node1);
            edges(node1, node2) = 1;
            edges(node2, node1) = 1;
        }
    }

    void remove_edge(int node1, int node2) {
        edges_per_node[node1].erase(node2);
        edges_per_node[node2].erase(node1);
        edges(node1, node2) = 0;
        edges(node2, node1) = 0;
    }

    /**
     * @brief Find two nearest nodes with EUCLIDEAN distances.
     */
    std::tuple<int, int, Scalar, Scalar> find_two_nearest(const PointT& x) const {
        Scalar min_dist1 = std::numeric_limits<Scalar>::max();
        Scalar min_dist2 = std::numeric_limits<Scalar>::max();
        int s1_id = -1, s2_id = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;

            // Euclidean distance (not squared)
            Scalar dist = (x - node.weight).norm();

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
     */
    void check_utility_criterion() {
        // Don't remove if only 2 nodes remain
        if (num_nodes() <= 2) return;

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

        if (min_utility_id != -1 && min_utility > 0) {
            if (max_error / min_utility > params.utility_k) {
                if (remove_node(min_utility_id, true)) {
                    n_removals++;
                }
            }
        }
    }

    void one_train_update(const PointT& sample) {
        // Find two nearest nodes (with EUCLIDEAN distances)
        auto [s1_id, s2_id, dist1, dist2] = find_two_nearest(sample);
        if (s1_id == -1 || s2_id == -1) return;

        // Connect s1 and s2
        add_edge(s1_id, s2_id);

        // Update error using EUCLIDEAN distance
        nodes[s1_id].error += dist1;

        // Update utility using EUCLIDEAN distance difference
        nodes[s1_id].utility += dist2 - dist1;

        // Move winner toward sample
        nodes[s1_id].weight += params.eps_b * (sample - nodes[s1_id].weight);

        // Update neighbors and age edges
        std::vector<int> edges_to_remove;
        for (int neighbor_id : edges_per_node[s1_id]) {
            edges(s1_id, neighbor_id)++;
            edges(neighbor_id, s1_id)++;

            if (edges(s1_id, neighbor_id) > params.max_age) {
                edges_to_remove.push_back(neighbor_id);
            } else {
                nodes[neighbor_id].weight += params.eps_n * (sample - nodes[neighbor_id].weight);
            }
        }

        for (int neighbor_id : edges_to_remove) {
            remove_edge(s1_id, neighbor_id);
            if (edges_per_node[neighbor_id].empty()) {
                remove_node(neighbor_id);
            }
        }

        // GNG-U2: κ-interval utility check
        if (n_learning > 0 && n_learning % params.kappa == 0) {
            check_utility_criterion();
        }

        // Decay all errors and utilities
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error -= params.beta * node.error;
            node.utility -= params.chi * node.utility;
        }

        // Periodically add new node
        n_trial_++;
        if (n_trial_ >= params.lambda) {
            n_trial_ = 0;
            insert_node();
        }

        n_learning++;
    }

    int insert_node() {
        if (addable_indices_.empty()) return -1;

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

        if (q_id == -1) return -1;

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

        if (f_id == -1) return -1;

        // Add new node between q and f
        PointT new_weight = (nodes[q_id].weight + nodes[f_id].weight) * static_cast<Scalar>(0.5);

        // Update errors
        nodes[q_id].error *= (1.0f - params.alpha);
        nodes[f_id].error *= (1.0f - params.alpha);

        float new_error = (nodes[q_id].error + nodes[f_id].error) * 0.5f;
        float new_utility = (nodes[q_id].utility + nodes[f_id].utility) * 0.5f;

        int new_id = add_node(new_weight, new_error, new_utility);
        if (new_id == -1) return -1;

        // Update edges
        remove_edge(q_id, f_id);
        add_edge(q_id, new_id);
        add_edge(f_id, new_id);

        return new_id;
    }
};

// Common type aliases
using GNGU2_2f = GrowingNeuralGasU2<Eigen::Vector2f>;
using GNGU2_3f = GrowingNeuralGasU2<Eigen::Vector3f>;
using GNGU2_2d = GrowingNeuralGasU2<Eigen::Vector2d>;
using GNGU2_3d = GrowingNeuralGasU2<Eigen::Vector3d>;

}  // namespace gng_u2
