/**
 * @file gcs_kubota.hpp
 * @brief Growing Cell Structures - Kubota paper-compliant implementation
 *
 * Based on:
 *   - Kubota, N. & Satomi, M. (2008). "自己増殖型ニューラルネットワークと教師無し分類学習"
 *
 * Key difference from standard GCS:
 *   - Node insertion selects neighbor by LONGEST EDGE (not max error)
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

namespace gcs_kubota {

/**
 * @brief GCS Kubota hyperparameters.
 */
struct GCSKubotaParams {
    int max_nodes = 100;
    int lambda = 100;
    float eps_b = 0.1f;
    float eps_n = 0.01f;
    float alpha = 0.5f;
    float beta = 0.005f;
};

/**
 * @brief A node in the GCS network.
 */
template <typename PointT>
struct GCSNode {
    int id = -1;
    float error = 0.0f;
    PointT weight;

    GCSNode() = default;
    GCSNode(int id_, const PointT& weight_)
        : id(id_), error(0.0f), weight(weight_) {}
};

/**
 * @brief Growing Cell Structures - Kubota paper-compliant implementation.
 */
template <typename PointT>
class GCSKubota {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GCSKubota&, int)>;

    GCSKubotaParams params;
    std::vector<GCSNode<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges;
    int n_learning = 0;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    std::mt19937 rng_;

public:
    explicit GCSKubota(const GCSKubotaParams& params = GCSKubotaParams(), unsigned int seed = 0)
        : params(params),
          nodes(params.max_nodes),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }
        init_triangle();
    }

    void init_triangle() {
        std::uniform_real_distribution<Scalar> noise(0.0, 0.1);

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
        for (const auto& [node_id, neighbors] : edges) {
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

private:
    int add_node(const PointT& weight) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = GCSNode<PointT>(node_id, weight);
        edges[node_id] = std::unordered_set<int>();
        return node_id;
    }

    void remove_node(int node_id) {
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

    void add_edge(int n1, int n2) {
        edges[n1].insert(n2);
        edges[n2].insert(n1);
    }

    void remove_edge(int n1, int n2) {
        edges[n1].erase(n2);
        edges[n2].erase(n1);
    }

    Scalar edge_length_sq(int node1, int node2) const {
        return (nodes[node1].weight - nodes[node2].weight).squaredNorm();
    }

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

        // KUBOTA: Find neighbor f of q connected by LONGEST EDGE
        Scalar max_len = -1.0;
        int f_id = -1;
        auto it = edges.find(q_id);
        if (it != edges.end()) {
            for (int neighbor_id : it->second) {
                Scalar len = edge_length_sq(q_id, neighbor_id);
                if (len > max_len) {
                    max_len = len;
                    f_id = neighbor_id;
                }
            }
        }

        if (f_id == -1) return -1;

        // Find common neighbors
        auto common_neighbors = find_common_neighbors(q_id, f_id);

        // Create new node at midpoint
        PointT new_weight = (nodes[q_id].weight + nodes[f_id].weight) * static_cast<Scalar>(0.5);
        int new_id = add_node(new_weight);

        if (new_id == -1) return -1;

        // Update edges
        remove_edge(q_id, f_id);
        add_edge(q_id, new_id);
        add_edge(f_id, new_id);

        // Connect to common neighbors
        for (int cn : common_neighbors) {
            add_edge(new_id, cn);
        }

        // Update errors
        nodes[q_id].error *= (1.0f - params.alpha);
        nodes[f_id].error *= (1.0f - params.alpha);
        nodes[new_id].error = (nodes[q_id].error + nodes[f_id].error) * 0.5f;

        return new_id;
    }

    void one_train_update(const PointT& sample) {
        // Decay all errors
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error *= (1.0f - params.beta);
        }

        // Find winner
        int winner_id = find_winner(sample);
        if (winner_id == -1) return;

        // Update winner error
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
using GCSKubota2f = GCSKubota<Eigen::Vector2f>;
using GCSKubota3f = GCSKubota<Eigen::Vector3f>;
using GCSKubota2d = GCSKubota<Eigen::Vector2d>;
using GCSKubota3d = GCSKubota<Eigen::Vector3d>;

}  // namespace gcs_kubota
