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
 * 4. Automatic attention detection: Stable corners detected via surface
 *    classification are automatically treated as attention regions
 *
 * Surface Classification (from reference implementation):
 *   - PLANE (0): Node on a flat surface (smallest eigenvalue very small)
 *   - EDGE (1): Node on an edge between surfaces
 *   - CORNER (2): Node at a corner (all eigenvalues similar)
 *   - STABLE_PLANE (4): Plane stable for > stability_threshold iterations
 *   - STABLE_EDGE (5): Edge stable for > stability_threshold iterations
 *   - STABLE_CORNER (6): Corner stable for > stability_threshold iterations
 *                        (automatically added to attention regions)
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
#include <Eigen/Eigenvalues>

namespace dd_gng {

/**
 * @brief Surface classification types from reference implementation.
 */
enum class SurfaceType : int {
    UNKNOWN = 3,        // Not yet classified
    PLANE = 0,          // Flat surface
    EDGE = 1,           // Edge between surfaces
    CORNER = 2,         // Corner point
    STABLE_PLANE = 4,   // Plane stable for > stability_threshold
    STABLE_EDGE = 5,    // Edge stable for > stability_threshold
    STABLE_CORNER = 6   // Corner stable for > stability_threshold (auto attention)
};

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

    // Auto-detection parameters (from reference implementation)
    bool auto_detect_attention = false;  // Enable automatic detection
    int stability_threshold = 16;        // Iterations for corner/edge stability
    int plane_stability_threshold = 8;   // Iterations for plane stability
    float corner_strength = 5.0f;        // Strength for auto-detected corners
    float plane_ev_ratio = 0.01f;        // Eigenvalue ratio for plane
    float edge_ev_ratio = 0.1f;          // Eigenvalue ratio for edge
    int surface_update_interval = 10;    // Surface classification update interval
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
    float strength = 1.0f;      // DD-GNG: node strength
    PointT weight;
    PointT normal;              // Normal vector (computed via PCA)
    SurfaceType surface_type = SurfaceType::UNKNOWN;  // Surface classification
    int stability_age = 0;      // How long node has maintained current surface type
    bool auto_attention = false;  // Whether auto-detected as attention region

    NeuronNodeDD() = default;
    NeuronNodeDD(int id_, const PointT& weight_, float error_ = 1.0f,
                 float utility_ = 0.0f, float strength_ = 1.0f)
        : id(id_), error(error_), utility(utility_), strength(strength_), weight(weight_) {
        normal.setZero();
        if (normal.size() >= 3) {
            normal(2) = 1.0f;  // Default Z-up normal
        } else if (normal.size() >= 1) {
            normal(normal.size() - 1) = 1.0f;
        }
    }
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
    static constexpr int Dim = PointT::RowsAtCompileTime;

    DDGNGParams params;
    std::vector<NeuronNodeDD<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edges;
    std::vector<AttentionRegion<PointT>> attention_regions;
    int n_learning = 0;
    int n_removals = 0;
    int n_auto_attention = 0;

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

    std::vector<PointT> get_node_normals() const {
        std::vector<PointT> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.normal);
            }
        }
        return result;
    }

    std::vector<int> get_node_surface_types() const {
        std::vector<int> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(static_cast<int>(node.surface_type));
            }
        }
        return result;
    }

    std::vector<bool> get_node_auto_attention() const {
        std::vector<bool> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.auto_attention);
            }
        }
        return result;
    }

    std::vector<PointT> get_auto_attention_nodes() const {
        std::vector<PointT> result;
        for (const auto& node : nodes) {
            if (node.id != -1 && node.auto_attention) {
                result.push_back(node.weight);
            }
        }
        return result;
    }

private:
    /**
     * @brief Calculate node strength based on position and attention regions.
     */
    float calculate_strength(const PointT& position, bool auto_attention = false) const {
        float strength = 1.0f;

        // Manual attention regions
        for (const auto& region : attention_regions) {
            if (region.contains(position)) {
                strength += region.strength_bonus;
            }
        }

        // Auto-detected attention (stable corners)
        if (auto_attention) {
            strength += params.corner_strength;
        }

        return strength;
    }

    /**
     * @brief Update strength for a node based on its current position.
     */
    void update_node_strength(int node_id) {
        if (nodes[node_id].id == -1) return;
        auto& node = nodes[node_id];
        node.strength = calculate_strength(node.weight, node.auto_attention);
    }

    /**
     * @brief Update strength values for all active nodes.
     */
    void update_all_strengths() {
        for (auto& node : nodes) {
            if (node.id != -1) {
                node.strength = calculate_strength(node.weight, node.auto_attention);
            }
        }
    }

    // =========================================================================
    // Surface Classification (Auto-detection feature)
    // =========================================================================

    /**
     * @brief Compute normal vector via PCA on node + neighbor positions.
     * @return Tuple of (normal_vector, eigenvalues_sorted).
     */
    std::pair<PointT, Eigen::Vector3f> compute_normal_pca(int node_id) {
        auto& node = nodes[node_id];

        PointT default_normal;
        default_normal.setZero();
        if constexpr (Dim >= 3) {
            default_normal(2) = 1.0f;  // Default Z-up normal
        } else {
            default_normal(Dim - 1) = 1.0f;
        }

        // Collect positions: this node + all neighbors
        std::vector<PointT> positions;
        positions.push_back(node.weight);

        auto it = edges_per_node.find(node_id);
        if (it != edges_per_node.end()) {
            for (int neighbor_id : it->second) {
                if (nodes[neighbor_id].id != -1) {
                    positions.push_back(nodes[neighbor_id].weight);
                }
            }
        }

        if (positions.size() < 3) {
            // Need at least 3 points for meaningful PCA
            return {default_normal, Eigen::Vector3f::Ones()};
        }

        // Compute centroid
        PointT centroid = PointT::Zero();
        for (const auto& pos : positions) {
            centroid += pos;
        }
        centroid /= static_cast<Scalar>(positions.size());

        // Compute covariance matrix (3x3 for 3D)
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        if constexpr (Dim >= 3) {
            for (const auto& pos : positions) {
                Eigen::Vector3f centered;
                for (int i = 0; i < 3; ++i) {
                    centered(i) = pos(i) - centroid(i);
                }
                cov += centered * centered.transpose();
            }
            cov /= static_cast<float>(positions.size());

            // Eigenvalue decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
            if (solver.info() != Eigen::Success) {
                return {default_normal, Eigen::Vector3f::Ones()};
            }

            // eigenvalues are sorted ascending
            Eigen::Vector3f eigenvalues = solver.eigenvalues();
            Eigen::Vector3f normal_vec = solver.eigenvectors().col(0);

            // Ensure consistent orientation (reference: if y < 0, flip)
            if (normal_vec(1) < 0) {
                normal_vec = -normal_vec;
            }

            // Normalize
            float norm = normal_vec.norm();
            if (norm > 1e-10f) {
                normal_vec /= norm;
            } else {
                return {default_normal, eigenvalues};
            }

            PointT normal;
            for (int i = 0; i < std::min(3, static_cast<int>(Dim)); ++i) {
                normal(i) = normal_vec(i);
            }
            for (int i = 3; i < Dim; ++i) {
                normal(i) = 0.0f;
            }

            return {normal, eigenvalues};
        } else {
            // 2D: all points are "corners"
            return {default_normal, Eigen::Vector3f::Ones()};
        }
    }

    /**
     * @brief Classify surface type based on eigenvalue ratios.
     */
    SurfaceType classify_surface_type(const Eigen::Vector3f& eigenvalues) {
        // Need 3D for proper classification
        if constexpr (Dim < 3) {
            return SurfaceType::CORNER;  // 2D points are all "corners"
        }

        // Avoid division by zero
        float max_ev = std::max(eigenvalues(2), 1e-10f);

        // Ratios: how small are the eigenvalues relative to the largest
        float ratio_small = eigenvalues(0) / max_ev;  // smallest / largest
        float ratio_mid = eigenvalues(1) / max_ev;    // middle / largest

        if (ratio_small < params.plane_ev_ratio) {
            // Smallest eigenvalue is very small -> PLANE
            return SurfaceType::PLANE;
        } else if (ratio_mid < params.edge_ev_ratio) {
            // Middle eigenvalue is small -> EDGE (linear feature)
            return SurfaceType::EDGE;
        } else {
            // All eigenvalues similar -> CORNER
            return SurfaceType::CORNER;
        }
    }

    /**
     * @brief Update surface classification for a single node.
     */
    void update_surface_classification(int node_id) {
        auto& node = nodes[node_id];
        if (node.id == -1) return;

        // Remember previous classification
        SurfaceType last_type = node.surface_type;

        // Compute normal and classify
        auto [normal, eigenvalues] = compute_normal_pca(node_id);
        node.normal = normal;
        SurfaceType new_type = classify_surface_type(eigenvalues);

        // Update stability tracking (from reference gng.cpp:523-548)
        if (last_type == SurfaceType::STABLE_CORNER && new_type != SurfaceType::CORNER) {
            // Was stable corner, now not a corner -> revert to corner, decrement age
            node.surface_type = SurfaceType::CORNER;
            node.stability_age = std::max(0, node.stability_age - 1);
            // Lost auto-attention status
            if (node.auto_attention) {
                node.auto_attention = false;
                n_auto_attention--;
            }
        } else if (new_type != last_type &&
                   (last_type == SurfaceType::PLANE ||
                    last_type == SurfaceType::EDGE ||
                    last_type == SurfaceType::CORNER ||
                    last_type == SurfaceType::UNKNOWN)) {
            // Classification changed -> reset age
            node.surface_type = new_type;
            node.stability_age = 0;
        } else if (new_type == SurfaceType::PLANE ||
                   new_type == SurfaceType::EDGE ||
                   new_type == SurfaceType::CORNER) {
            // Same classification -> increment age
            node.surface_type = new_type;
            node.stability_age = std::min(25, node.stability_age + 1);

            // Check for promotion to stable variant
            if (new_type == SurfaceType::PLANE &&
                node.stability_age > params.plane_stability_threshold) {
                node.surface_type = SurfaceType::STABLE_PLANE;
            } else if (new_type == SurfaceType::EDGE &&
                       node.stability_age > params.stability_threshold) {
                node.surface_type = SurfaceType::STABLE_EDGE;
            } else if (new_type == SurfaceType::CORNER &&
                       node.stability_age > params.stability_threshold) {
                // Promote to stable corner -> AUTO ATTENTION!
                node.surface_type = SurfaceType::STABLE_CORNER;
                if (!node.auto_attention) {
                    node.auto_attention = true;
                    n_auto_attention++;
                }
            }
        } else {
            // Already stable -> just decrement age slightly
            node.stability_age = std::max(0, node.stability_age - 1);
        }
    }

    /**
     * @brief Update surface classifications for all active nodes.
     */
    void update_all_surface_classifications() {
        if (!params.auto_detect_attention) return;
        if constexpr (Dim < 3) return;  // Surface classification requires 3D

        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (nodes[i].id != -1) {
                update_surface_classification(i);
            }
        }

        // Update strengths after classification changes
        update_all_strengths();
    }

    // =========================================================================

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

        // Track auto-attention removal
        if (nodes[node_id].auto_attention) {
            n_auto_attention--;
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

        // Auto-detection: Update surface classifications periodically
        if (params.auto_detect_attention &&
            n_learning > 0 &&
            n_learning % params.surface_update_interval == 0) {
            update_all_surface_classifications();
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
