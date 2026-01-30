/**
 * @file ais_gng_dt.hpp
 * @brief AiS-GNG-DT (Add-if-Silent GNG with Different Topologies) implementation
 *
 * Combines:
 *   - GNG-DT: Multiple independent edge topologies (position, color, normal)
 *   - AiS-GNG: Add-if-Silent rule for rapid high-density structure generation
 *
 * This experimental algorithm aims to:
 *   1. Learn multiple topologies for 3D point clouds (from GNG-DT)
 *   2. Generate high-density structures quickly (from AiS-GNG)
 *   3. Handle non-stationary distributions via utility-based node removal
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

namespace ais_gng_dt {

/**
 * @brief AiS-GNG-DT hyperparameters.
 *
 * Combines GNG-DT and AiS-GNG parameters.
 */
struct AiSGNGDTParams {
    // Basic GNG parameters
    int max_nodes = 150;
    int lambda = 100;          // Node insertion interval
    float eps_b = 0.05f;       // Winner learning rate
    float eps_n = 0.005f;      // Neighbor learning rate
    float alpha = 0.5f;        // Error decay rate when splitting
    float beta = 0.005f;       // Global error decay
    int max_age = 88;          // Maximum edge age

    // GNG-DT specific parameters
    float tau_color = 0.05f;   // Color similarity threshold
    float tau_normal = 0.998f; // Normal similarity threshold (dot product)
    float dis_thv = 0.5f;      // Distance threshold for new nodes

    // AiS-GNG specific parameters
    float theta_ais_min = 0.05f;  // Minimum distance for Add-if-Silent rule
    float theta_ais_max = 0.15f;  // Maximum distance for Add-if-Silent rule
    int kappa = 10;               // Utility check interval
    float utility_k = 1000.0f;    // Utility threshold for node removal
    float chi = 0.005f;           // Utility decay rate
};

/**
 * @brief A neuron node in the AiS-GNG-DT network.
 */
struct AiSGNGDTNode {
    int id = -1;               // -1 means invalid/removed
    Eigen::Vector3f position;  // 3D position
    Eigen::Vector3f color;     // RGB color
    Eigen::Vector3f normal;    // Unit normal vector
    float error = 0.0f;        // Accumulated error
    float utility = 0.0f;      // Utility value

    AiSGNGDTNode()
        : position(Eigen::Vector3f::Zero()),
          color(Eigen::Vector3f::Zero()),
          normal(0.0f, 0.0f, 1.0f) {}

    AiSGNGDTNode(int id_, const Eigen::Vector3f& pos,
                 const Eigen::Vector3f& col = Eigen::Vector3f::Zero(),
                 float err = 0.0f, float util = 0.0f)
        : id(id_), position(pos), color(col),
          normal(0.0f, 0.0f, 1.0f), error(err), utility(util) {}
};

/**
 * @brief AiS-GNG-DT algorithm implementation.
 *
 * Combines GNG-DT's multiple topology learning with AiS-GNG's
 * Add-if-Silent rule and utility-based node management.
 */
class AiSGNGDT {
public:
    using Callback = std::function<void(const AiSGNGDT&, int)>;

    AiSGNGDTParams params;
    std::vector<AiSGNGDTNode> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;

    // Edge matrices (three topologies)
    Eigen::MatrixXi edges_pos;    // Position-based edges
    Eigen::MatrixXi edges_color;  // Color-based edges
    Eigen::MatrixXi edges_normal; // Normal-based edges
    Eigen::MatrixXi edge_age;     // Edge ages

    int n_learning = 0;
    int n_ais_additions = 0;      // Number of nodes added by Add-if-Silent rule
    int n_utility_removals = 0;   // Number of nodes removed by utility criterion

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    float total_error_ = 0.0f;
    std::mt19937 rng_;

public:
    /**
     * @brief Construct AiS-GNG-DT with given parameters.
     */
    explicit AiSGNGDT(const AiSGNGDTParams& params = AiSGNGDTParams(),
                      unsigned int seed = 0)
        : params(params),
          nodes(params.max_nodes),
          edges_pos(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          edges_color(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          edges_normal(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          edge_age(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }
    }

    /**
     * @brief Initialize with two random nodes in [0,1] range.
     */
    void init() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < 2; ++i) {
            Eigen::Vector3f pos(dist(rng_), dist(rng_), dist(rng_));
            add_node(pos);
        }

        // Connect initial 2 nodes with all edge types
        connect_initial_nodes(0, 1);
    }

    /**
     * @brief Initialize with two specific points.
     */
    void init(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) {
        int id1 = add_node(p1);
        int id2 = add_node(p2);
        if (id1 != -1 && id2 != -1) {
            connect_initial_nodes(id1, id2);
        }
    }

    /**
     * @brief Train on data for multiple iterations.
     */
    void train(const std::vector<Eigen::Vector3f>& data,
               const std::vector<Eigen::Vector3f>& colors,
               int n_iterations,
               const Callback& callback = nullptr) {
        if (data.empty()) return;

        int n_cycles = n_iterations / params.lambda;
        if (n_cycles == 0) n_cycles = 1;

        for (int cycle = 0; cycle < n_cycles; ++cycle) {
            gng_main_cycle(data, colors);
            if (callback) {
                callback(*this, cycle * params.lambda);
            }
        }
    }

    /**
     * @brief Train without color data.
     */
    void train(const std::vector<Eigen::Vector3f>& data,
               int n_iterations,
               const Callback& callback = nullptr) {
        train(data, std::vector<Eigen::Vector3f>(), n_iterations, callback);
    }

    /**
     * @brief Single online learning step.
     */
    void partial_fit(const Eigen::Vector3f& position,
                     const Eigen::Vector3f& color = Eigen::Vector3f::Zero()) {
        float error = one_train_update(position, color);
        total_error_ += error;
        n_trial_++;

        if (n_trial_ >= params.lambda) {
            float avg_error = total_error_ / params.lambda;
            if (num_nodes() < params.max_nodes && avg_error > 1e-6f) {
                node_add();
            }
            n_trial_ = 0;
            total_error_ = 0.0f;
        }
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
     * @brief Get number of position edges.
     */
    int num_edges_pos() const {
        int count = 0;
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id != -1) {
                count += static_cast<int>(neighbors.size());
            }
        }
        return count / 2;
    }

    /**
     * @brief Get number of normal edges.
     */
    int num_edges_normal() const {
        int count = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (nodes[i].id == -1) continue;
            for (int j = i + 1; j < static_cast<int>(nodes.size()); ++j) {
                if (nodes[j].id == -1) continue;
                if (edges_normal(i, j) > 0) ++count;
            }
        }
        return count;
    }

    /**
     * @brief Get graph structure (position topology only).
     */
    void get_graph(std::vector<Eigen::Vector3f>& out_nodes,
                   std::vector<std::pair<int, int>>& out_edges) const {
        out_nodes.clear();
        out_edges.clear();

        std::unordered_map<int, int> id_to_idx;

        for (const auto& node : nodes) {
            if (node.id != -1) {
                id_to_idx[node.id] = static_cast<int>(out_nodes.size());
                out_nodes.push_back(node.position);
            }
        }

        std::unordered_set<int64_t> seen;
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id == -1) continue;
            for (int neighbor_id : neighbors) {
                if (nodes[neighbor_id].id == -1) continue;
                int64_t key = std::min(node_id, neighbor_id) * 10000LL +
                              std::max(node_id, neighbor_id);
                if (seen.find(key) == seen.end()) {
                    seen.insert(key);
                    out_edges.emplace_back(id_to_idx[node_id], id_to_idx[neighbor_id]);
                }
            }
        }
    }

    /**
     * @brief Get graph structure with all topologies.
     */
    void get_multi_graph(std::vector<Eigen::Vector3f>& out_nodes,
                         std::vector<std::pair<int, int>>& out_pos_edges,
                         std::vector<std::pair<int, int>>& out_color_edges,
                         std::vector<std::pair<int, int>>& out_normal_edges) const {
        out_nodes.clear();
        out_pos_edges.clear();
        out_color_edges.clear();
        out_normal_edges.clear();

        std::unordered_map<int, int> id_to_idx;

        for (const auto& node : nodes) {
            if (node.id != -1) {
                id_to_idx[node.id] = static_cast<int>(out_nodes.size());
                out_nodes.push_back(node.position);
            }
        }

        // Position edges
        std::unordered_set<int64_t> seen;
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id == -1) continue;
            for (int neighbor_id : neighbors) {
                if (nodes[neighbor_id].id == -1) continue;
                int64_t key = std::min(node_id, neighbor_id) * 10000LL +
                              std::max(node_id, neighbor_id);
                if (seen.find(key) == seen.end()) {
                    seen.insert(key);
                    out_pos_edges.emplace_back(id_to_idx[node_id], id_to_idx[neighbor_id]);
                }
            }
        }

        // Color and Normal edges
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (nodes[i].id == -1) continue;
            for (int j = i + 1; j < static_cast<int>(nodes.size()); ++j) {
                if (nodes[j].id == -1) continue;
                if (edges_color(i, j) > 0) {
                    out_color_edges.emplace_back(id_to_idx[i], id_to_idx[j]);
                }
                if (edges_normal(i, j) > 0) {
                    out_normal_edges.emplace_back(id_to_idx[i], id_to_idx[j]);
                }
            }
        }
    }

    /**
     * @brief Get normal vectors for active nodes.
     */
    std::vector<Eigen::Vector3f> get_node_normals() const {
        std::vector<Eigen::Vector3f> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.normal);
            }
        }
        return result;
    }

    /**
     * @brief Get color vectors for active nodes.
     */
    std::vector<Eigen::Vector3f> get_node_colors() const {
        std::vector<Eigen::Vector3f> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.color);
            }
        }
        return result;
    }

    /**
     * @brief Get error values for active nodes.
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

    /**
     * @brief Get utility values for active nodes.
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

private:
    /**
     * @brief Add a new node.
     */
    int add_node(const Eigen::Vector3f& position,
                 const Eigen::Vector3f& color = Eigen::Vector3f::Zero(),
                 float error = 0.0f, float utility = 0.0f) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = AiSGNGDTNode(node_id, position, color, error, utility);
        edges_per_node[node_id] = std::unordered_set<int>();
        return node_id;
    }

    /**
     * @brief Remove a node with cascading deletion.
     */
    void remove_node(int node_id) {
        std::vector<int> neighbors_to_check;
        auto it = edges_per_node.find(node_id);
        if (it != edges_per_node.end()) {
            neighbors_to_check.assign(it->second.begin(), it->second.end());
        }

        for (int other_id : neighbors_to_check) {
            remove_all_edges(node_id, other_id);
        }

        edges_per_node.erase(node_id);
        nodes[node_id].id = -1;
        addable_indices_.push_back(node_id);

        // Cascading deletion
        for (int neighbor_id : neighbors_to_check) {
            if (nodes[neighbor_id].id != -1) {
                auto neighbor_it = edges_per_node.find(neighbor_id);
                if (neighbor_it == edges_per_node.end() || neighbor_it->second.empty()) {
                    remove_node(neighbor_id);
                }
            }
        }
    }

    /**
     * @brief Connect initial two nodes with all edge types.
     */
    void connect_initial_nodes(int n1, int n2) {
        edges_pos(n1, n2) = 1;
        edges_pos(n2, n1) = 1;
        edges_color(n1, n2) = 1;
        edges_color(n2, n1) = 1;
        edges_normal(n1, n2) = 1;
        edges_normal(n2, n1) = 1;
        edges_per_node[n1].insert(n2);
        edges_per_node[n2].insert(n1);
    }

    /**
     * @brief Add position edge between two nodes.
     */
    void add_position_edge(int n1, int n2) {
        if (edges_pos(n1, n2) == 0) {
            edges_pos(n1, n2) = 1;
            edges_pos(n2, n1) = 1;
            edges_per_node[n1].insert(n2);
            edges_per_node[n2].insert(n1);
        }
    }

    /**
     * @brief Remove all edges between two nodes.
     */
    void remove_all_edges(int n1, int n2) {
        edges_pos(n1, n2) = 0;
        edges_pos(n2, n1) = 0;
        edges_color(n1, n2) = 0;
        edges_color(n2, n1) = 0;
        edges_normal(n1, n2) = 0;
        edges_normal(n2, n1) = 0;
        edge_age(n1, n2) = 0;
        edge_age(n2, n1) = 0;
        edges_per_node[n1].erase(n2);
        edges_per_node[n2].erase(n1);
    }

    /**
     * @brief Find the two nearest nodes to input position.
     */
    std::tuple<int, int, float, float> find_two_nearest(const Eigen::Vector3f& position) const {
        float min_dist1 = std::numeric_limits<float>::max();
        float min_dist2 = std::numeric_limits<float>::max();
        int s1_id = -1, s2_id = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;

            float dist_sq = (position - node.position).squaredNorm();

            if (dist_sq < min_dist1) {
                min_dist2 = min_dist1;
                s2_id = s1_id;
                min_dist1 = dist_sq;
                s1_id = node.id;
            } else if (dist_sq < min_dist2) {
                min_dist2 = dist_sq;
                s2_id = node.id;
            }
        }

        return {s1_id, s2_id, min_dist1, min_dist2};
    }

    /**
     * @brief Compute normal vector using PCA on positions.
     */
    Eigen::Vector3f compute_normal_from_positions(
            const std::vector<Eigen::Vector3f>& positions,
            const Eigen::Vector3f& cog_sum) const {
        int ect = static_cast<int>(positions.size());
        if (ect < 2) {
            return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        }

        Eigen::Vector3f cog = cog_sum / static_cast<float>(ect);

        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (const auto& pos : positions) {
            Eigen::Vector3f centered = pos - cog;
            cov += centered * centered.transpose();
        }
        cov /= static_cast<float>(ect);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        if (solver.info() != Eigen::Success) {
            return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        }

        Eigen::Vector3f normal = solver.eigenvectors().col(0);

        if (normal.y() < 0) {
            normal = -normal;
        }

        float norm = normal.norm();
        if (norm > 1e-10f) {
            normal /= norm;
        } else {
            return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        }

        return normal;
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
                remove_node(min_utility_id);
                n_utility_removals++;
            }
        }
    }

    /**
     * @brief Add-if-Silent rule-based growing process.
     *
     * If both winner nodes are within tolerance area [theta_min, theta_max],
     * add the input directly as a new node.
     */
    bool ais_growing_process(const Eigen::Vector3f& position,
                             const Eigen::Vector3f& color,
                             int s1_id, int s2_id,
                             float dist1_sq, float dist2_sq) {
        if (addable_indices_.empty()) return false;

        // Convert to Euclidean distances for AiS threshold comparison
        float dist1 = std::sqrt(dist1_sq);
        float dist2 = std::sqrt(dist2_sq);

        // Check AiS conditions
        bool s1_in_range = params.theta_ais_min < dist1 && dist1 < params.theta_ais_max;
        bool s2_in_range = params.theta_ais_min < dist2 && dist2 < params.theta_ais_max;

        if (s1_in_range && s2_in_range) {
            // Add input directly as new node
            float new_error = 0.5f * (nodes[s1_id].error + nodes[s2_id].error);
            float new_utility = 0.5f * (nodes[s1_id].utility + nodes[s2_id].utility);

            int new_id = add_node(position, color, new_error, new_utility);

            if (new_id == -1) return false;

            // Connect new node to s1 and s2
            add_position_edge(new_id, s1_id);
            add_position_edge(new_id, s2_id);

            // Initialize color edges based on similarity
            float color_dist_s1 = (nodes[new_id].color - nodes[s1_id].color).squaredNorm();
            if (color_dist_s1 < params.tau_color * params.tau_color) {
                edges_color(new_id, s1_id) = 1;
                edges_color(s1_id, new_id) = 1;
            }

            float color_dist_s2 = (nodes[new_id].color - nodes[s2_id].color).squaredNorm();
            if (color_dist_s2 < params.tau_color * params.tau_color) {
                edges_color(new_id, s2_id) = 1;
                edges_color(s2_id, new_id) = 1;
            }

            n_ais_additions++;
            return true;
        }

        return false;
    }

    /**
     * @brief Single learning step (GNG-DT style with multiple topologies).
     */
    void gng_learn(int s1, int s2,
                   const Eigen::Vector3f& v_pos,
                   const Eigen::Vector3f& v_color,
                   float e1, float e2) {
        auto& n1 = nodes[s1];
        auto& n2 = nodes[s2];

        // Add position edge
        add_position_edge(s1, s2);

        // Update color edge
        float color_dist_sq = (n1.color - n2.color).squaredNorm();
        if (color_dist_sq < params.tau_color * params.tau_color) {
            edges_color(s1, s2) = 1;
            edges_color(s2, s1) = 1;
        } else {
            edges_color(s1, s2) = 0;
            edges_color(s2, s1) = 0;
        }

        // Calculate normal dot product BEFORE updating
        float normal_dot = n1.normal.dot(n2.normal);

        // Store s1's original position for PCA
        Eigen::Vector3f s1_original_pos = n1.position;

        // Reset edge age
        edge_age(s1, s2) = 0;
        edge_age(s2, s1) = 0;

        // Update winner position
        n1.position += e1 * (v_pos - n1.position);

        // Update winner color
        n1.color += e1 * (v_color - n1.color);

        // Update neighbors and collect positions for PCA
        std::vector<int> neighbors_to_remove;
        std::vector<Eigen::Vector3f> pca_positions;
        pca_positions.push_back(s1_original_pos);
        Eigen::Vector3f pca_cog = s1_original_pos;

        for (int neighbor_id : edges_per_node[s1]) {
            if (neighbor_id == s1) continue;

            auto& neighbor = nodes[neighbor_id];

            // Move neighbor toward input
            neighbor.position += e2 * (v_pos - neighbor.position);

            // Increment edge age
            edge_age(s1, neighbor_id)++;
            edge_age(neighbor_id, s1)++;

            // Collect for PCA
            pca_positions.push_back(neighbor.position);
            pca_cog += neighbor.position;

            // Check age threshold
            if (edge_age(s1, neighbor_id) > params.max_age) {
                neighbors_to_remove.push_back(neighbor_id);
            }
        }

        // Remove old edges
        for (int neighbor_id : neighbors_to_remove) {
            remove_all_edges(s1, neighbor_id);
            auto it = edges_per_node.find(neighbor_id);
            if (it == edges_per_node.end() || it->second.empty()) {
                remove_node(neighbor_id);
            }
        }

        // Update color neighbors
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (nodes[i].id != -1 && edges_color(s1, i) == 1 && i != s1) {
                nodes[i].color += e2 * (v_color - nodes[i].color);
            }
        }

        // Compute normal via PCA
        n1.normal = compute_normal_from_positions(pca_positions, pca_cog);

        // Update normal edge
        if (std::abs(normal_dot) > params.tau_normal) {
            edges_normal(s1, s2) = 1;
            edges_normal(s2, s1) = 1;
        } else {
            edges_normal(s1, s2) = 0;
            edges_normal(s2, s1) = 0;
        }
    }

    /**
     * @brief Decay all node errors and utilities.
     */
    void discount_errors() {
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error -= params.beta * node.error;
            node.utility -= params.chi * node.utility;
            if (node.error < 0) node.error = 0.0f;
            if (node.utility < 0) node.utility = 0.0f;
        }
    }

    /**
     * @brief Add 2 new connected nodes at input position (for distant inputs).
     */
    void add_new_node_distance(const Eigen::Vector3f& position,
                               const Eigen::Vector3f& color) {
        int r = add_node(position, color);
        if (r == -1) return;

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        Eigen::Vector3f offset(
            dist(rng_) * params.dis_thv / 10.0f,
            dist(rng_) * params.dis_thv / 10.0f,
            dist(rng_) * params.dis_thv / 10.0f
        );
        Eigen::Vector3f q_pos = position + offset;
        int q = add_node(q_pos, color);
        if (q == -1) {
            remove_node(r);
            return;
        }

        add_position_edge(r, q);
    }

    /**
     * @brief Add a new node between highest-error node and its neighbor.
     */
    void node_add() {
        if (addable_indices_.empty()) return;

        float max_err = -1.0f;
        int q = -1;
        int first_node_id = -1;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;

            if (first_node_id == -1) {
                first_node_id = node.id;
            }

            if (node.error > max_err) {
                max_err = node.error;
                q = node.id;
            }
        }

        if (q == -1) return;

        // Find neighbor with maximum error
        float max_err_f = -1.0f;
        int f = -1;
        auto it = edges_per_node.find(q);
        if (it != edges_per_node.end()) {
            for (int neighbor_id : it->second) {
                if (nodes[neighbor_id].error > max_err_f) {
                    max_err_f = nodes[neighbor_id].error;
                    f = neighbor_id;
                }
            }
        }

        if (f == -1) return;

        // Add new node between q and f
        Eigen::Vector3f new_pos = 0.5f * (nodes[q].position + nodes[f].position);
        Eigen::Vector3f new_color = 0.5f * (nodes[q].color + nodes[f].color);
        int r = add_node(new_pos, new_color);

        if (r == -1) return;

        // Initialize normal
        Eigen::Vector3f new_normal = 0.5f * (nodes[q].normal + nodes[f].normal);
        float norm = new_normal.norm();
        if (norm > 1e-10f) {
            nodes[r].normal = new_normal / norm;
        } else {
            nodes[r].normal = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        }

        // Update edges
        edges_pos(q, f) = 0;
        edges_pos(f, q) = 0;
        edges_per_node[q].erase(f);
        edges_per_node[f].erase(q);

        // Inherit color/normal edges
        edges_color(q, r) = edges_color(q, f);
        edges_color(r, q) = edges_color(q, f);
        edges_color(f, r) = edges_color(q, f);
        edges_color(r, f) = edges_color(q, f);
        edges_color(q, f) = 0;
        edges_color(f, q) = 0;

        edges_normal(q, r) = edges_normal(q, f);
        edges_normal(r, q) = edges_normal(q, f);
        edges_normal(f, r) = edges_normal(q, f);
        edges_normal(r, f) = edges_normal(q, f);
        edges_normal(q, f) = 0;
        edges_normal(f, q) = 0;

        // Add position edges
        add_position_edge(q, r);
        add_position_edge(r, f);

        // Update errors and utilities
        nodes[q].error *= 0.5f;
        nodes[f].error *= 0.5f;
        nodes[q].utility *= 0.5f;
        nodes[f].utility *= 0.5f;
        nodes[r].error = nodes[q].error;
        nodes[r].utility = nodes[q].utility;
    }

    /**
     * @brief Single training iteration with AiS rule.
     */
    float one_train_update(const Eigen::Vector3f& position,
                           const Eigen::Vector3f& color) {
        auto [s1, s2, dist1_sq, dist2_sq] = find_two_nearest(position);

        if (s1 == -1 || s2 == -1) return 0.0f;

        // Distance threshold check (GNG-DT style)
        if (dist1_sq > params.dis_thv * params.dis_thv &&
            num_nodes() < params.max_nodes - 2) {
            add_new_node_distance(position, color);
            discount_errors();
            return 0.0f;
        }

        // Add-if-Silent rule (AiS-GNG style)
        ais_growing_process(position, color, s1, s2, dist1_sq, dist2_sq);

        // Update error and utility
        nodes[s1].error += dist1_sq;
        nodes[s1].utility += dist2_sq - dist1_sq;

        // Learning step
        gng_learn(s1, s2, position, color, params.eps_b, params.eps_n);

        // Decay errors and utilities
        discount_errors();

        // Check utility criterion every kappa iterations
        if (n_learning > 0 && n_learning % params.kappa == 0) {
            check_utility_criterion();
        }

        n_learning++;
        return dist1_sq;
    }

    /**
     * @brief Run one main cycle (lambda_ iterations + node addition).
     */
    void gng_main_cycle(const std::vector<Eigen::Vector3f>& data,
                        const std::vector<Eigen::Vector3f>& colors) {
        std::uniform_int_distribution<int> dist(0, static_cast<int>(data.size()) - 1);
        bool has_colors = !colors.empty();
        float total_error = 0.0f;

        for (int i = 0; i < params.lambda; ++i) {
            int idx = dist(rng_);
            Eigen::Vector3f color = has_colors ? colors[idx] : Eigen::Vector3f::Zero();
            float error = one_train_update(data[idx], color);
            total_error += error;
        }

        // Node addition
        total_error /= params.lambda;
        if (num_nodes() < params.max_nodes && total_error > 1e-6f) {
            node_add();
        }
    }
};

// Alias
using AddIfSilentGNGDT = AiSGNGDT;

}  // namespace ais_gng_dt
