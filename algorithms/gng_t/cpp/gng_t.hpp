/**
 * @file gng_t.hpp
 * @brief Growing Neural Gas with Triangulation (GNG-T) implementation
 *
 * Based on:
 *   - Kubota, N. & Satomi, M. (2008). "Growing Neural Gas with Triangulation
 *     for reconstructing a 3D surface model"
 *   - World Automation Congress (2008)
 *
 * GNG-T extends GNG with heuristic triangulation (quadrilateral search
 * and intersection search) to maintain proper triangle mesh structure.
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

namespace gng_t {

/**
 * @brief GNG-T hyperparameters.
 */
struct GNGTParams {
    int max_nodes = 100;    // Maximum number of nodes
    int lambda = 100;       // Node insertion interval
    float eps_b = 0.08f;    // Winner learning rate
    float eps_n = 0.008f;   // Neighbor learning rate
    float alpha = 0.5f;     // Error decay on split
    float beta = 0.005f;    // Global error decay
    int max_age = 100;      // Maximum edge age
};

/**
 * @brief A neuron node in the GNG-T network.
 */
template <typename PointT>
struct NeuronNode {
    int id = -1;            // -1 means invalid/removed
    float error = 1.0f;     // Accumulated error
    PointT weight;          // Position vector

    NeuronNode() = default;
    NeuronNode(int id_, const PointT& weight_)
        : id(id_), error(1.0f), weight(weight_) {}
};

/**
 * @brief Growing Neural Gas with Triangulation algorithm.
 *
 * @tparam PointT Point type (Eigen vector, e.g., Eigen::Vector2f)
 */
template <typename PointT>
class GrowingNeuralGasT {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GrowingNeuralGasT&, int)>;

    GNGTParams params;
    std::vector<NeuronNode<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edges;  // Edge age matrix (0 = no edge, >=1 = connected)
    int n_learning = 0;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    std::mt19937 rng_;

public:
    /**
     * @brief Construct GNG-T with given parameters.
     */
    explicit GrowingNeuralGasT(const GNGTParams& params = GNGTParams(), unsigned int seed = 0)
        : params(params),
          nodes(params.max_nodes),
          edges(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }
    }

    /**
     * @brief Initialize with 3 random nodes forming a triangle (2D simplex).
     */
    void init() {
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        std::vector<int> node_ids;

        for (int i = 0; i < 3; ++i) {
            PointT weight;
            for (int j = 0; j < weight.size(); ++j) {
                weight(j) = dist(rng_);
            }
            int id = add_node(weight);
            node_ids.push_back(id);
        }

        // Connect all three nodes to form a triangle
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                add_edge(node_ids[i], node_ids[j]);
            }
        }
    }

    /**
     * @brief Train on data for multiple iterations.
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
        return count / 2;
    }

    /**
     * @brief Get graph with sequential indices (for visualization).
     */
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

    /**
     * @brief Get triangles from the current graph structure.
     */
    std::vector<std::tuple<int, int, int>> get_triangles() const {
        std::vector<std::tuple<int, int, int>> triangles;

        // Create index mapping
        std::unordered_map<int, int> id_to_idx;
        std::vector<int> active_ids;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                id_to_idx[node.id] = static_cast<int>(active_ids.size());
                active_ids.push_back(node.id);
            }
        }

        // Find triangles (cliques of size 3)
        for (int a_id : active_ids) {
            auto it_a = edges_per_node.find(a_id);
            if (it_a == edges_per_node.end()) continue;

            std::vector<int> neighbors_a_list;
            for (int n : it_a->second) {
                if (n > a_id) neighbors_a_list.push_back(n);
            }

            for (int b_id : neighbors_a_list) {
                auto it_b = edges_per_node.find(b_id);
                if (it_b == edges_per_node.end()) continue;

                for (int c_id : neighbors_a_list) {
                    if (c_id <= b_id) continue;
                    if (it_b->second.count(c_id)) {
                        triangles.emplace_back(
                            id_to_idx[a_id],
                            id_to_idx[b_id],
                            id_to_idx[c_id]
                        );
                    }
                }
            }
        }

        return triangles;
    }

private:
    int add_node(const PointT& weight) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = NeuronNode<PointT>(node_id, weight);
        edges_per_node[node_id] = std::unordered_set<int>();
        return node_id;
    }

    void remove_node(int node_id) {
        auto it = edges_per_node.find(node_id);
        if (it != edges_per_node.end() && !it->second.empty()) return;

        edges_per_node.erase(node_id);
        nodes[node_id].id = -1;
        addable_indices_.push_back(node_id);
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

    bool has_edge(int node1, int node2) const {
        return edges(node1, node2) > 0;
    }

    std::vector<int> get_active_node_ids() const {
        std::vector<int> ids;
        for (const auto& node : nodes) {
            if (node.id != -1) ids.push_back(node.id);
        }
        return ids;
    }

    Scalar edge_length_sq(int node1, int node2) const {
        return (nodes[node1].weight - nodes[node2].weight).squaredNorm();
    }

    std::pair<int, int> find_two_nearest(const PointT& x) const {
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

        return {s1_id, s2_id};
    }

    /**
     * @brief Check if edges (a,b) and (c,d) intersect.
     */
    bool edges_intersect(int a_id, int b_id, int c_id, int d_id) const {
        const auto& pa = nodes[a_id].weight;
        const auto& pb = nodes[b_id].weight;
        const auto& pc = nodes[c_id].weight;
        const auto& pd = nodes[d_id].weight;

        // CCW (counter-clockwise) test
        auto ccw = [](const PointT& p1, const PointT& p2, const PointT& p3) {
            return (p3(1) - p1(1)) * (p2(0) - p1(0)) > (p2(1) - p1(1)) * (p3(0) - p1(0));
        };

        return (ccw(pa, pc, pd) != ccw(pb, pc, pd)) && (ccw(pa, pb, pc) != ccw(pa, pb, pd));
    }

    /**
     * @brief Quadrilateral search (Section 2.5.1 of Kubota & Satomi 2008).
     * Find quadrilaterals without diagonals, add shorter diagonal.
     */
    void quadrilateral_search() {
        auto active_ids = get_active_node_ids();

        for (int a_id : active_ids) {
            auto it_a = edges_per_node.find(a_id);
            if (it_a == edges_per_node.end()) continue;

            std::vector<int> neighbors_a(it_a->second.begin(), it_a->second.end());

            for (size_t i = 0; i < neighbors_a.size(); ++i) {
                int b_id = neighbors_a[i];
                for (size_t j = i + 1; j < neighbors_a.size(); ++j) {
                    int c_id = neighbors_a[j];

                    if (has_edge(b_id, c_id)) continue;  // B-C already connected

                    // Find common neighbor D of B and C (not A)
                    auto it_b = edges_per_node.find(b_id);
                    auto it_c = edges_per_node.find(c_id);
                    if (it_b == edges_per_node.end() || it_c == edges_per_node.end()) continue;

                    for (int d_id : it_b->second) {
                        if (d_id == a_id) continue;
                        if (it_c->second.count(d_id) == 0) continue;
                        if (has_edge(a_id, d_id)) continue;  // A-D already connected

                        // A-B-D-C forms a quadrilateral candidate
                        // Add shorter diagonal
                        Scalar dist_ad = edge_length_sq(a_id, d_id);
                        Scalar dist_bc = edge_length_sq(b_id, c_id);

                        if (dist_ad < dist_bc) {
                            add_edge(a_id, d_id);
                        } else {
                            add_edge(b_id, c_id);
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Intersection search (Section 2.5.2 of Kubota & Satomi 2008).
     * Detect edge crossings and remove longer edge.
     */
    void intersection_search() {
        // Collect all edges
        std::vector<std::pair<int, int>> all_edges;
        std::unordered_set<int64_t> seen;

        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id == -1) continue;
            for (int neighbor_id : neighbors) {
                int64_t key = std::min(node_id, neighbor_id) * 10000LL + std::max(node_id, neighbor_id);
                if (seen.find(key) == seen.end()) {
                    seen.insert(key);
                    all_edges.emplace_back(node_id, neighbor_id);
                }
            }
        }

        // Check all pairs for intersection
        std::vector<std::pair<int, int>> edges_to_remove;

        for (size_t i = 0; i < all_edges.size(); ++i) {
            auto [a_id, b_id] = all_edges[i];

            for (size_t j = i + 1; j < all_edges.size(); ++j) {
                auto [c_id, d_id] = all_edges[j];

                // Skip if edges share a vertex
                if (a_id == c_id || a_id == d_id || b_id == c_id || b_id == d_id) continue;

                if (edges_intersect(a_id, b_id, c_id, d_id)) {
                    // Remove longer edge
                    Scalar len_ab = edge_length_sq(a_id, b_id);
                    Scalar len_cd = edge_length_sq(c_id, d_id);

                    if (len_ab > len_cd) {
                        edges_to_remove.emplace_back(a_id, b_id);
                    } else {
                        edges_to_remove.emplace_back(c_id, d_id);
                    }
                }
            }
        }

        for (const auto& [n1, n2] : edges_to_remove) {
            remove_edge(n1, n2);
        }
    }

    /**
     * @brief Triangulation search (quadrilateral + intersection).
     */
    void triangulation_search() {
        quadrilateral_search();
        intersection_search();
    }

    /**
     * @brief Single training update.
     */
    void one_train_update(const PointT& sample) {
        // Decay all errors
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error -= params.beta * node.error;
        }

        // Find two nearest nodes
        auto [s1_id, s2_id] = find_two_nearest(sample);
        if (s1_id == -1 || s2_id == -1) return;

        // Update winner error
        Scalar dist_sq = (sample - nodes[s1_id].weight).squaredNorm();
        nodes[s1_id].error += dist_sq;

        // Move winner toward sample
        nodes[s1_id].weight += params.eps_b * (sample - nodes[s1_id].weight);

        // Connect s1 and s2
        add_edge(s1_id, s2_id);

        // Update neighbors and age edges
        std::vector<int> edges_to_remove_list;
        for (int neighbor_id : edges_per_node[s1_id]) {
            if (edges(s1_id, neighbor_id) > params.max_age) {
                edges_to_remove_list.push_back(neighbor_id);
            } else {
                nodes[neighbor_id].weight += params.eps_n * (sample - nodes[neighbor_id].weight);
                edges(s1_id, neighbor_id)++;
                edges(neighbor_id, s1_id)++;
            }
        }

        // Remove old edges and isolated nodes
        for (int neighbor_id : edges_to_remove_list) {
            remove_edge(s1_id, neighbor_id);
            if (edges_per_node[neighbor_id].empty()) {
                remove_node(neighbor_id);
            }
        }

        // Periodically add new node
        n_trial_++;
        bool topology_changed = false;

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

                if (q_id == -1) {
                    n_learning++;
                    return;
                }

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

                if (f_id != -1) {
                    // Add new node between q and f
                    PointT new_weight = (nodes[q_id].weight + nodes[f_id].weight) * static_cast<Scalar>(0.5);
                    int new_id = add_node(new_weight);

                    if (new_id != -1) {
                        // Update edges
                        remove_edge(q_id, f_id);
                        add_edge(q_id, new_id);
                        add_edge(f_id, new_id);

                        // Connect to common neighbors (GCS-style)
                        auto it_q = edges_per_node.find(q_id);
                        auto it_f = edges_per_node.find(f_id);
                        if (it_q != edges_per_node.end() && it_f != edges_per_node.end()) {
                            for (int common_id : it_q->second) {
                                if (it_f->second.count(common_id)) {
                                    add_edge(new_id, common_id);
                                }
                            }
                        }

                        // Update errors
                        nodes[q_id].error *= (1.0f - params.alpha);
                        nodes[f_id].error *= (1.0f - params.alpha);
                        nodes[new_id].error = (nodes[q_id].error + nodes[f_id].error) * 0.5f;

                        topology_changed = true;
                    }
                }
            }
        }

        // Perform triangulation search after topology changes
        if (topology_changed || !edges_to_remove_list.empty()) {
            triangulation_search();
        }

        n_learning++;
    }
};

// Common type aliases
using GNGT2f = GrowingNeuralGasT<Eigen::Vector2f>;
using GNGT3f = GrowingNeuralGasT<Eigen::Vector3f>;
using GNGT2d = GrowingNeuralGasT<Eigen::Vector2d>;
using GNGT3d = GrowingNeuralGasT<Eigen::Vector3d>;

}  // namespace gng_t
