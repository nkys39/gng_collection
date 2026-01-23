/**
 * @file gcs.hpp
 * @brief Growing Cell Structures (GCS) implementation
 *
 * Based on:
 *   - Fritzke, B. (1994). "Growing cell structures - a self-organizing network
 *     for unsupervised and supervised learning"
 *
 * GCS maintains a k-dimensional simplicial complex (triangular mesh in 2D).
 * New nodes are inserted at the centroid of the cell with highest accumulated error.
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

namespace gcs {

/**
 * @brief GCS hyperparameters.
 */
struct GCSParams {
    int max_nodes = 100;    // Maximum number of nodes
    int lambda = 100;       // Node insertion interval
    float eps_b = 0.1f;     // Winner learning rate
    float eps_n = 0.01f;    // Neighbor learning rate
    float alpha = 0.5f;     // Error decay on split
    float beta = 0.005f;    // Global error decay
};

/**
 * @brief A node in the GCS network.
 *
 * @tparam PointT Point type (Eigen vector)
 */
template <typename PointT>
struct GCSNode {
    int id = -1;            // -1 means invalid/removed
    float error = 0.0f;     // Accumulated error (signal counter)
    PointT weight;          // Position vector

    GCSNode() = default;
    GCSNode(int id_, const PointT& weight_)
        : id(id_), error(0.0f), weight(weight_) {}
};

/**
 * @brief Growing Cell Structures algorithm.
 *
 * GCS is a self-organizing network that maintains a simplicial complex
 * (triangular mesh in 2D). Unlike GNG which learns arbitrary topologies,
 * GCS always maintains a connected mesh structure.
 *
 * @tparam PointT Point type (Eigen vector, e.g., Eigen::Vector2f)
 */
template <typename PointT>
class GrowingCellStructures {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GrowingCellStructures&, int)>;

    GCSParams params;
    std::vector<GCSNode<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges;
    int n_learning = 0;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    std::mt19937 rng_;

public:
    /**
     * @brief Construct GCS with given parameters.
     *
     * @param params GCS hyperparameters
     * @param seed Random seed (0 for random)
     */
    explicit GrowingCellStructures(const GCSParams& params = GCSParams(), unsigned int seed = 0)
        : params(params),
          nodes(params.max_nodes),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        // Initialize addable indices queue
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }

        // Initialize with a triangle (minimum 2D simplicial complex)
        init_triangle();
    }

    /**
     * @brief Initialize with a triangle of 3 nodes.
     */
    void init_triangle() {
        std::uniform_real_distribution<Scalar> noise(0.0, 0.1);

        // Create 3 nodes forming a triangle
        std::vector<PointT> positions(3);
        positions[0] = PointT(0.3, 0.3);
        positions[1] = PointT(0.7, 0.3);
        positions[2] = PointT(0.5, 0.7);

        std::vector<int> node_ids;
        for (int i = 0; i < 3; ++i) {
            PointT weight = positions[i];
            for (int j = 0; j < PointT::RowsAtCompileTime; ++j) {
                weight(j) += noise(rng_);
            }
            int node_id = add_node(weight);
            node_ids.push_back(node_id);
        }

        // Connect all pairs (triangle edges)
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                add_edge(node_ids[i], node_ids[j]);
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
        for (const auto& [node_id, neighbors] : edges) {
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
    std::vector<std::pair<int, PointT>> get_nodes_with_id() const {
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
    std::vector<std::pair<int, int>> get_edges_with_id() const {
        std::vector<std::pair<int, int>> result;
        std::unordered_set<int64_t> seen;

        for (const auto& [node_id, neighbors] : edges) {
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
        for (const auto& [node_id, neighbors] : edges) {
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
     * @return Node ID, or -1 if no space
     */
    int add_node(const PointT& weight) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = GCSNode<PointT>(node_id, weight);
        edges[node_id] = std::unordered_set<int>();
        return node_id;
    }

    /**
     * @brief Remove a node and its edges.
     *
     * @param node_id ID of node to remove
     */
    void remove_node(int node_id) {
        // Remove all edges
        auto it = edges.find(node_id);
        if (it != edges.end()) {
            for (int neighbor : std::vector<int>(it->second.begin(), it->second.end())) {
                edges[neighbor].erase(node_id);
            }
        }
        edges.erase(node_id);

        nodes[node_id].id = -1;
        addable_indices_.push_back(node_id);
    }

    /**
     * @brief Add edge between two nodes.
     */
    void add_edge(int n1, int n2) {
        edges[n1].insert(n2);
        edges[n2].insert(n1);
    }

    /**
     * @brief Remove edge between two nodes.
     */
    void remove_edge(int n1, int n2) {
        edges[n1].erase(n2);
        edges[n2].erase(n1);
    }

    /**
     * @brief Find the closest node to input x.
     *
     * @return ID of the winner node
     */
    int find_winner(const PointT& x) const {
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        int winner = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;
            Scalar dist = (x - node.weight).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                winner = node.id;
            }
        }

        return winner;
    }

    /**
     * @brief Find nodes that are neighbors of both n1 and n2.
     *
     * @return Set of common neighbor IDs
     */
    std::unordered_set<int> find_common_neighbors(int n1, int n2) const {
        std::unordered_set<int> common;
        auto it1 = edges.find(n1);
        auto it2 = edges.find(n2);
        if (it1 == edges.end() || it2 == edges.end()) return common;

        for (int neighbor : it1->second) {
            if (it2->second.count(neighbor) > 0) {
                common.insert(neighbor);
            }
        }
        return common;
    }

    /**
     * @brief Insert a new node in the highest-error region.
     *
     * @return ID of new node, or -1 if failed
     */
    int insert_node() {
        if (addable_indices_.empty()) return -1;

        // Find node q with maximum error
        float max_err = 0.0f;
        int q_id = -1;
        for (const auto& node : nodes) {
            if (node.id == -1) continue;
            if (node.error > max_err) {
                max_err = node.error;
                q_id = node.id;
            }
        }

        if (q_id == -1) return -1;

        // Find neighbor f of q with maximum error
        float max_err_f = -1.0f;
        int f_id = -1;
        auto it = edges.find(q_id);
        if (it != edges.end()) {
            for (int neighbor_id : it->second) {
                if (nodes[neighbor_id].error > max_err_f) {
                    max_err_f = nodes[neighbor_id].error;
                    f_id = neighbor_id;
                }
            }
        }

        if (f_id == -1) return -1;

        // Find common neighbors (they will also connect to the new node)
        auto common_neighbors = find_common_neighbors(q_id, f_id);

        // Create new node at midpoint
        PointT new_weight = (nodes[q_id].weight + nodes[f_id].weight) * static_cast<Scalar>(0.5);
        int new_id = add_node(new_weight);

        if (new_id == -1) return -1;

        // Update edges: remove (q, f), add (q, r) and (f, r)
        remove_edge(q_id, f_id);
        add_edge(q_id, new_id);
        add_edge(f_id, new_id);

        // Connect to common neighbors (maintaining simplicial structure)
        for (int cn : common_neighbors) {
            add_edge(new_id, cn);
        }

        // Update errors
        nodes[q_id].error *= (1.0f - params.alpha);
        nodes[f_id].error *= (1.0f - params.alpha);
        nodes[new_id].error = (nodes[q_id].error + nodes[f_id].error) * 0.5f;

        return new_id;
    }

    /**
     * @brief Single training update.
     */
    void one_train_update(const PointT& sample) {
        // Decay all errors
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error *= (1.0f - params.beta);
        }

        // Find winner
        int winner_id = find_winner(sample);
        if (winner_id == -1) return;

        // Update winner error (squared distance)
        Scalar dist_sq = (sample - nodes[winner_id].weight).squaredNorm();
        nodes[winner_id].error += static_cast<float>(dist_sq);

        // Move winner toward sample
        nodes[winner_id].weight += params.eps_b * (sample - nodes[winner_id].weight);

        // Move neighbors toward sample
        auto it = edges.find(winner_id);
        if (it != edges.end()) {
            for (int neighbor_id : it->second) {
                nodes[neighbor_id].weight += params.eps_n * (sample - nodes[neighbor_id].weight);
            }
        }

        // Periodically insert new node
        n_trial_++;
        if (n_trial_ >= params.lambda) {
            n_trial_ = 0;
            insert_node();
        }

        n_learning++;
    }
};

// Common type aliases
using GCS2f = GrowingCellStructures<Eigen::Vector2f>;
using GCS3f = GrowingCellStructures<Eigen::Vector3f>;
using GCS2d = GrowingCellStructures<Eigen::Vector2d>;
using GCS3d = GrowingCellStructures<Eigen::Vector3d>;

}  // namespace gcs
