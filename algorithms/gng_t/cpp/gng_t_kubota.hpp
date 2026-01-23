/**
 * @file gng_t_kubota.hpp
 * @brief Growing Neural Gas with Triangulation - Kubota paper-compliant implementation
 *
 * Based on:
 *   - Kubota, N. & Satomi, M. (2008). "自己増殖型ニューラルネットワークと教師無し分類学習"
 *
 * Key differences from standard GNG-T:
 *   - Node insertion selects neighbor by LONGEST EDGE (not max error)
 *   - Intersection detection uses γ formula (not CCW method)
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

namespace gng_t_kubota {

/**
 * @brief GNG-T Kubota hyperparameters.
 */
struct GNGTKubotaParams {
    int max_nodes = 100;
    int lambda = 100;
    float eps_b = 0.08f;
    float eps_n = 0.008f;
    float alpha = 0.5f;
    float beta = 0.005f;
    int max_age = 100;
};

/**
 * @brief A neuron node in the GNG-T network.
 */
template <typename PointT>
struct NeuronNode {
    int id = -1;
    float error = 1.0f;
    PointT weight;

    NeuronNode() = default;
    NeuronNode(int id_, const PointT& weight_)
        : id(id_), error(1.0f), weight(weight_) {}
};

/**
 * @brief Growing Neural Gas with Triangulation - Kubota paper-compliant.
 */
template <typename PointT>
class GNGTKubota {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GNGTKubota&, int)>;

    GNGTKubotaParams params;
    std::vector<NeuronNode<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edges;
    int n_learning = 0;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    std::mt19937 rng_;

public:
    explicit GNGTKubota(const GNGTKubotaParams& params = GNGTKubotaParams(), unsigned int seed = 0)
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
        std::vector<int> node_ids;

        for (int i = 0; i < 3; ++i) {
            PointT weight;
            for (int j = 0; j < weight.size(); ++j) {
                weight(j) = dist(rng_);
            }
            int id = add_node(weight);
            node_ids.push_back(id);
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                add_edge(node_ids[i], node_ids[j]);
            }
        }
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

    std::vector<std::tuple<int, int, int>> get_triangles() const {
        std::vector<std::tuple<int, int, int>> triangles;

        std::unordered_map<int, int> id_to_idx;
        std::vector<int> active_ids;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                id_to_idx[node.id] = static_cast<int>(active_ids.size());
                active_ids.push_back(node.id);
            }
        }

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
     * @brief KUBOTA: Check edge intersection using γ formula (Section 2.5.2).
     *
     * Edge B-D intersects edge C-E iff:
     *   γ1 * γ2 <= 0 AND γ3 * γ4 <= 0
     *
     * where:
     *   γ1 = (xC - xE)(yD - yC) + (yC - yE)(xC - xD)
     *   γ2 = (xC - xE)(yB - yC) + (yC - yE)(xC - xB)
     *   γ3 = (xB - xD)(yC - yB) + (yB - yD)(xB - xC)
     *   γ4 = (xB - xD)(yE - yB) + (yB - yD)(xB - xE)
     */
    bool edges_intersect_gamma(int b_id, int d_id, int c_id, int e_id) const {
        const auto& pb = nodes[b_id].weight;
        const auto& pd = nodes[d_id].weight;
        const auto& pc = nodes[c_id].weight;
        const auto& pe = nodes[e_id].weight;

        Scalar xB = pb(0), yB = pb(1);
        Scalar xD = pd(0), yD = pd(1);
        Scalar xC = pc(0), yC = pc(1);
        Scalar xE = pe(0), yE = pe(1);

        Scalar gamma1 = (xC - xE) * (yD - yC) + (yC - yE) * (xC - xD);
        Scalar gamma2 = (xC - xE) * (yB - yC) + (yC - yE) * (xC - xB);
        Scalar gamma3 = (xB - xD) * (yC - yB) + (yB - yD) * (xB - xC);
        Scalar gamma4 = (xB - xD) * (yE - yB) + (yB - yD) * (xB - xE);

        return (gamma1 * gamma2 <= 0) && (gamma3 * gamma4 <= 0);
    }

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

                    if (has_edge(b_id, c_id)) continue;

                    auto it_b = edges_per_node.find(b_id);
                    auto it_c = edges_per_node.find(c_id);
                    if (it_b == edges_per_node.end() || it_c == edges_per_node.end()) continue;

                    for (int d_id : it_b->second) {
                        if (d_id == a_id) continue;
                        if (it_c->second.count(d_id) == 0) continue;
                        if (has_edge(a_id, d_id)) continue;

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

    void intersection_search() {
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

        std::vector<std::pair<int, int>> edges_to_remove;

        for (size_t i = 0; i < all_edges.size(); ++i) {
            auto [a_id, b_id] = all_edges[i];

            for (size_t j = i + 1; j < all_edges.size(); ++j) {
                auto [c_id, d_id] = all_edges[j];

                if (a_id == c_id || a_id == d_id || b_id == c_id || b_id == d_id) continue;

                // KUBOTA: Use γ formula for intersection
                if (edges_intersect_gamma(a_id, b_id, c_id, d_id)) {
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

    void triangulation_search() {
        quadrilateral_search();
        intersection_search();
    }

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

                // KUBOTA: Find neighbor of q connected by LONGEST EDGE
                Scalar max_len = -1.0;
                int f_id = -1;
                auto it = edges_per_node.find(q_id);
                if (it != edges_per_node.end()) {
                    for (int neighbor_id : it->second) {
                        Scalar len = edge_length_sq(q_id, neighbor_id);
                        if (len > max_len) {
                            max_len = len;
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

                        // Connect to common neighbors
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
using GNGTKubota2f = GNGTKubota<Eigen::Vector2f>;
using GNGTKubota3f = GNGTKubota<Eigen::Vector3f>;
using GNGTKubota2d = GNGTKubota<Eigen::Vector2d>;
using GNGTKubota3d = GNGTKubota<Eigen::Vector3d>;

}  // namespace gng_t_kubota
