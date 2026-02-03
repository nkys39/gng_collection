/**
 * @file dd_gng.hpp
 * @brief DD-GNG (Dynamic Density Growing Neural Gas) implementation
 *
 * Based on:
 *   - Saputra, A.A., et al. (2019). "Dynamic Density Topological Structure
 *     Generation for Real-Time Ladder Affordance Detection"
 *   - IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2019
 *
 * DD-GNG extends GNG-U with dynamic density control:
 * 1. Node strength: Nodes in attention regions have higher strength values
 * 2. Strength-weighted node insertion: error * (scale * strength)^power for priority
 * 3. Strength-weighted learning: eps_b / strength for stability
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

namespace dd_gng {

/**
 * @brief DD-GNG hyperparameters.
 */
struct DDGNGParams {
    int max_nodes = 100;          // Maximum number of nodes
    int lambda = 300;             // Node insertion interval
    float eps_b = 0.08f;          // Winner learning rate
    float eps_n = 0.008f;         // Neighbor learning rate
    float alpha = 0.5f;           // Error decay on split
    float beta = 0.005f;          // Global error decay
    float chi = 0.005f;           // Utility decay rate
    int max_age = 88;             // Maximum edge age
    float utility_k = 1000.0f;    // Utility threshold
    int kappa = 10;               // Utility check interval

    // DD-GNG specific parameters
    int strength_power = 4;       // Exponent for strength weighting
    float strength_scale = 4.0f;  // Scale factor for strength
    bool use_strength_learning = true;   // Apply strength to learning rate
    bool use_strength_insertion = true;  // Apply strength to node insertion
};

/**
 * @brief Defines an attention region for dynamic density control.
 */
template <typename PointT>
struct AttentionRegion {
    PointT center;            // Center position of the region
    PointT size;              // Size (half-extent) of the region in each dimension
    float strength_bonus;     // Additional strength for nodes in this region

    AttentionRegion() : strength_bonus(1.0f) {}
    AttentionRegion(const PointT& center_, const PointT& size_, float strength_ = 1.0f)
        : center(center_), size(size_), strength_bonus(strength_) {}

    bool contains(const PointT& point) const {
        for (int i = 0; i < point.size(); ++i) {
            if (std::abs(point(i) - center(i)) > size(i)) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief A neuron node in the DD-GNG network.
 */
template <typename PointT>
struct NeuronNodeDD {
    int id = -1;
    float error = 1.0f;
    float utility = 0.0f;
    float strength = 1.0f;  // DD-GNG: node strength
    PointT weight;

    NeuronNodeDD() = default;
    NeuronNodeDD(int id_, const PointT& weight_, float error_ = 1.0f,
                 float utility_ = 0.0f, float strength_ = 1.0f)
        : id(id_), error(error_), utility(utility_), strength(strength_), weight(weight_) {}
};

/**
 * @brief DD-GNG algorithm implementation.
 *
 * @tparam PointT Point type (Eigen vector)
 */
template <typename PointT>
class DynamicDensityGNG {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const DynamicDensityGNG&, int)>;

    DDGNGParams params;
    std::vector<NeuronNodeDD<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edges;
    std::vector<AttentionRegion<PointT>> attention_regions;
    int n_learning = 0;
    int n_removals = 0;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    std::mt19937 rng_;

public:
    explicit DynamicDensityGNG(const DDGNGParams& params = DDGNGParams(), unsigned int seed = 0)
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

    /**
     * @brief Add an attention region for dynamic density control.
     */
    void add_attention_region(const PointT& center, const PointT& size, float strength = 1.0f) {
        attention_regions.emplace_back(center, size, strength);
    }

    /**
     * @brief Clear all attention regions.
     */
    void clear_attention_regions() {
        attention_regions.clear();
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

    std::vector<float> get_node_strengths() const {
        std::vector<float> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.strength);
            }
        }
        return result;
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
    /**
     * @brief Calculate node strength based on position and attention regions.
     */
    float calculate_strength(const PointT& position) const {
        float strength = 1.0f;
        for (const auto& region : attention_regions) {
            if (region.contains(position)) {
                strength += region.strength_bonus;
            }
        }
        return strength;
    }

    /**
     * @brief Update strength for a node based on its current position.
     */
    void update_node_strength(int node_id) {
        if (nodes[node_id].id == -1) return;
        nodes[node_id].strength = calculate_strength(nodes[node_id].weight);
    }

    /**
     * @brief Update strength values for all active nodes.
     */
    void update_all_strengths() {
        for (auto& node : nodes) {
            if (node.id != -1) {
                node.strength = calculate_strength(node.weight);
            }
        }
    }

    int add_node(const PointT& weight, float error = 1.0f, float utility = 0.0f) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        float strength = calculate_strength(weight);
        nodes[node_id] = NeuronNodeDD<PointT>(node_id, weight, error, utility, strength);
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
     * @brief Find two nearest nodes with Euclidean distances.
     */
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

    /**
     * @brief Check and remove node with lowest utility if criterion met.
     */
    void check_utility_criterion() {
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
        auto [s1_id, s2_id, dist1, dist2] = find_two_nearest(sample);
        if (s1_id == -1 || s2_id == -1) return;

        // Connect s1 and s2
        add_edge(s1_id, s2_id);

        // Update error
        nodes[s1_id].error += dist1;

        // Update utility
        nodes[s1_id].utility += dist2 - dist1;

        // Update strength for winner
        update_node_strength(s1_id);
        float strength = nodes[s1_id].strength;

        // DD-GNG: Strength-weighted learning
        float effective_eps_b = params.eps_b;
        if (params.use_strength_learning) {
            effective_eps_b = params.eps_b / strength;
        }

        // Move winner toward sample
        nodes[s1_id].weight += effective_eps_b * (sample - nodes[s1_id].weight);

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

        // Utility check at κ-interval
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
            insert_node_with_density();
        }

        n_learning++;
    }

    /**
     * @brief Insert new node with DD-GNG density-weighted priority.
     */
    int insert_node_with_density() {
        if (addable_indices_.empty()) return -1;

        // Update all node strengths before selection
        update_all_strengths();

        // Find node with maximum weighted error
        float max_weighted_err = 0.0f;
        int u_id = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;

            float weighted_err = node.error;
            if (params.use_strength_insertion) {
                float scaled_strength = params.strength_scale * node.strength;
                weighted_err = node.error * std::pow(scaled_strength, params.strength_power);
            }

            if (weighted_err > max_weighted_err) {
                max_weighted_err = weighted_err;
                u_id = node.id;
            }
        }

        if (u_id == -1) return -1;

        // Find neighbor of u with maximum weighted error
        float max_weighted_err_f = 0.0f;
        int f_id = -1;

        auto it = edges_per_node.find(u_id);
        if (it != edges_per_node.end()) {
            for (int neighbor_id : it->second) {
                const auto& node = nodes[neighbor_id];
                float weighted_err = node.error;
                if (params.use_strength_insertion) {
                    float scaled_strength = params.strength_scale * node.strength;
                    weighted_err = node.error * std::pow(scaled_strength, params.strength_power);
                }

                if (weighted_err > max_weighted_err_f) {
                    max_weighted_err_f = weighted_err;
                    f_id = neighbor_id;
                }
            }
        }

        if (f_id == -1) return -1;

        // Add new node between u and f
        PointT new_weight = (nodes[u_id].weight + nodes[f_id].weight) * static_cast<Scalar>(0.5);

        // Update errors
        nodes[u_id].error *= (1.0f - params.alpha);
        nodes[f_id].error *= (1.0f - params.alpha);

        float new_error = (nodes[u_id].error + nodes[f_id].error) * 0.5f;
        float new_utility = (nodes[u_id].utility + nodes[f_id].utility) * 0.5f;

        int new_id = add_node(new_weight, new_error, new_utility);
        if (new_id == -1) return -1;

        // Update edges
        remove_edge(u_id, f_id);
        add_edge(u_id, new_id);
        add_edge(f_id, new_id);

        return new_id;
    }
};

// Common type aliases
using DDGNG_2f = DynamicDensityGNG<Eigen::Vector2f>;
using DDGNG_3f = DynamicDensityGNG<Eigen::Vector3f>;
using DDGNG_2d = DynamicDensityGNG<Eigen::Vector2d>;
using DDGNG_3d = DynamicDensityGNG<Eigen::Vector3d>;

}  // namespace dd_gng
