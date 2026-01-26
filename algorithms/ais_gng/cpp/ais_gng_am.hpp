/**
 * @file ais_gng_am.hpp
 * @brief AiS-GNG-AM (SMC 2023) with Amount of Movement tracking
 *
 * Based on:
 *   Shoji, M., Obo, T., & Kubota, N. (2023).
 *   "Add-if-Silent Rule-Based Growing Neural Gas with Amount of Movement
 *    for High-Density Topological Structure Generation of Dynamic Object"
 *   IEEE SMC 2023, pp. 3040-3047.
 *
 * Extends AiS-GNG with:
 * 1. Range thresholds [θ_min, θ_max] for Add-if-Silent rule
 * 2. Amount of Movement (AM) tracking for dynamic object detection
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

namespace ais_gng_am {

struct AiSGNGAMParams {
    int max_nodes = 100;
    int lambda = 100;
    float eps_b = 0.08f;
    float eps_n = 0.008f;
    float alpha = 0.5f;
    float beta = 0.005f;
    float chi = 0.005f;
    int max_age = 88;
    float utility_k = 1000.0f;
    int kappa = 10;
    // SMC 2023: Range thresholds
    float theta_ais_min = 0.03f;
    float theta_ais_max = 0.15f;
    // Amount of Movement parameters
    float am_decay = 0.95f;      // Decay rate for AM (γ_AM)
    float am_threshold = 0.01f;  // Threshold for "moving" classification
};

template <typename PointT>
struct NeuronNodeAM {
    int id = -1;
    float error = 1.0f;
    float utility = 0.0f;
    float amount_of_movement = 0.0f;
    PointT weight;
    PointT prev_weight;

    NeuronNodeAM() = default;
    NeuronNodeAM(int id_, const PointT& weight_, float error_ = 1.0f,
                 float utility_ = 0.0f, float am_ = 0.0f)
        : id(id_), error(error_), utility(utility_), amount_of_movement(am_),
          weight(weight_), prev_weight(weight_) {}
};

template <typename PointT>
class AiSGNGAM {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const AiSGNGAM&, int)>;

    AiSGNGAMParams params;
    std::vector<NeuronNodeAM<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edges;
    int n_learning = 0;
    int n_ais_additions = 0;
    int n_utility_removals = 0;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    std::mt19937 rng_;

public:
    explicit AiSGNGAM(const AiSGNGAMParams& params = AiSGNGAMParams(), unsigned int seed = 0)
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
            if (callback) callback(*this, iter);
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
            if (nodes[node_id].id != -1) count += static_cast<int>(neighbors.size());
        }
        return count / 2;
    }

    int n_removals() const { return n_utility_removals; }

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

    // Get Amount of Movement for all active nodes
    std::vector<float> get_node_movements() const {
        std::vector<float> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.amount_of_movement);
            }
        }
        return result;
    }

    // Get mask of moving nodes (AM > threshold)
    std::vector<bool> get_moving_nodes_mask() const {
        std::vector<bool> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.amount_of_movement > params.am_threshold);
            }
        }
        return result;
    }

    // Count moving nodes
    int num_moving_nodes() const {
        int count = 0;
        for (const auto& node : nodes) {
            if (node.id != -1 && node.amount_of_movement > params.am_threshold) {
                ++count;
            }
        }
        return count;
    }

private:
    int add_node(const PointT& weight, float error = 1.0f, float utility = 0.0f, float am = 0.0f) {
        if (addable_indices_.empty()) return -1;
        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = NeuronNodeAM<PointT>(node_id, weight, error, utility, am);
        edges_per_node[node_id] = std::unordered_set<int>();
        return node_id;
    }

    bool remove_node(int node_id, bool force = false) {
        auto it = edges_per_node.find(node_id);
        if (!force && it != edges_per_node.end() && !it->second.empty()) return false;
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

    std::tuple<int, int, Scalar, Scalar> find_two_nearest(const PointT& x) const {
        Scalar min_dist1 = std::numeric_limits<Scalar>::max();
        Scalar min_dist2 = std::numeric_limits<Scalar>::max();
        int s1_id = -1, s2_id = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;
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

    void check_utility_criterion() {
        if (num_nodes() <= 2) return;
        float max_error = 0.0f;
        float min_utility = std::numeric_limits<float>::max();
        int min_utility_id = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;
            if (node.error > max_error) max_error = node.error;
            if (node.utility < min_utility) {
                min_utility = node.utility;
                min_utility_id = node.id;
            }
        }

        if (min_utility_id != -1 && min_utility > 0) {
            if (max_error / min_utility > params.utility_k) {
                if (remove_node(min_utility_id, true)) n_utility_removals++;
            }
        }
    }

    // SMC 2023: Range threshold AiS rule
    bool ais_growing_process(const PointT& sample, int s1_id, int s2_id,
                              Scalar dist1, Scalar dist2) {
        if (addable_indices_.empty()) return false;

        bool s1_in_range = dist1 > params.theta_ais_min && dist1 < params.theta_ais_max;
        bool s2_in_range = dist2 > params.theta_ais_min && dist2 < params.theta_ais_max;

        if (s1_in_range && s2_in_range) {
            float new_am = (nodes[s1_id].amount_of_movement + nodes[s2_id].amount_of_movement) * 0.5f;
            int new_id = add_node(sample, 1.0f, 0.0f, new_am);
            if (new_id == -1) return false;
            add_edge(new_id, s1_id);
            add_edge(new_id, s2_id);
            n_ais_additions++;
            return true;
        }
        return false;
    }

    // Update Amount of Movement for all nodes
    void update_amount_of_movement() {
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            Scalar movement = (node.weight - node.prev_weight).norm();
            node.amount_of_movement = params.am_decay * node.amount_of_movement + movement;
            node.prev_weight = node.weight;
        }
    }

    void one_train_update(const PointT& sample) {
        auto [s1_id, s2_id, dist1, dist2] = find_two_nearest(sample);
        if (s1_id == -1 || s2_id == -1) return;

        bool ais_added = ais_growing_process(sample, s1_id, s2_id, dist1, dist2);
        add_edge(s1_id, s2_id);

        nodes[s1_id].error += dist1;
        nodes[s1_id].utility += dist2 - dist1;
        nodes[s1_id].weight += params.eps_b * (sample - nodes[s1_id].weight);

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
            if (edges_per_node[neighbor_id].empty()) remove_node(neighbor_id);
        }

        if (n_learning > 0 && n_learning % params.kappa == 0) check_utility_criterion();

        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error -= params.beta * node.error;
            node.utility -= params.chi * node.utility;
        }

        // Update Amount of Movement
        update_amount_of_movement();

        n_trial_++;
        if (!ais_added && n_trial_ >= params.lambda) {
            n_trial_ = 0;
            insert_node();
        }
        n_learning++;
    }

    int insert_node() {
        if (addable_indices_.empty()) return -1;

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

        PointT new_weight = (nodes[q_id].weight + nodes[f_id].weight) * static_cast<Scalar>(0.5);
        nodes[q_id].error *= (1.0f - params.alpha);
        nodes[f_id].error *= (1.0f - params.alpha);
        float new_error = (nodes[q_id].error + nodes[f_id].error) * 0.5f;
        float new_utility = (nodes[q_id].utility + nodes[f_id].utility) * 0.5f;
        float new_am = (nodes[q_id].amount_of_movement + nodes[f_id].amount_of_movement) * 0.5f;

        int new_id = add_node(new_weight, new_error, new_utility, new_am);
        if (new_id == -1) return -1;

        remove_edge(q_id, f_id);
        add_edge(q_id, new_id);
        add_edge(f_id, new_id);
        return new_id;
    }
};

using AiSGNGAM2f = AiSGNGAM<Eigen::Vector2f>;
using AiSGNGAM3f = AiSGNGAM<Eigen::Vector3f>;
using AiSGNGAM2d = AiSGNGAM<Eigen::Vector2d>;
using AiSGNGAM3d = AiSGNGAM<Eigen::Vector3d>;

}  // namespace ais_gng_am
