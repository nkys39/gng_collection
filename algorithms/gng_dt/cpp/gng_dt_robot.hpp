/**
 * @file gng_dt_robot.hpp
 * @brief GNG-DT Robot Version with traversability analysis features
 *
 * Extends the base GNG-DT with robot-specific features from the original
 * toda_gngdt implementation:
 *   - pedge: Traversability edge (connects nodes with same traversability)
 *   - traversability_property: Whether node is on traversable surface
 *   - through_property: Based on surface inclination angle
 *   - dimension_property: Based on PCA eigenvalues (surface planarity)
 *   - contour: Edge detection based on angular gaps between neighbors
 *   - degree: Inclination cost for path planning
 *   - curvature: PCA residual based curvature
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

namespace gng_dt {

/**
 * @brief GNG-DT Robot hyperparameters.
 */
struct GNGDTRobotParams {
    int max_nodes = 100;       // GNGN in original
    int lambda = 200;          // ramda in original
    float eps_b = 0.05f;       // e1 in original
    float eps_n = 0.0005f;     // e2 in original
    float alpha = 0.5f;        // Error decay rate
    float beta = 0.0005f;      // dise in original
    int max_age = 88;          // MAX_AGE in original

    // GNG-DT specific
    float tau_color = 0.05f;   // cthv
    float tau_normal = 0.998f; // nthv
    float dis_thv = 0.5f;      // DIS_THV
    float thv = 0.000001f;     // THV

    // Robot-specific
    float max_angle = 20.0f;   // MAXANGLE in degrees
    float s1thv = 1.0f;        // Eigenvalue threshold for dimension property
    float contour_gap_threshold = 135.0f;  // Angular gap for contour detection
};

/**
 * @brief A neuron node in the GNG-DT Robot network.
 */
struct GNGDTRobotNode {
    int id = -1;
    Eigen::Vector3f position;
    Eigen::Vector3f color;
    Eigen::Vector3f normal;
    float error = 0.0f;
    float utility = 0.0f;

    // PCA results
    float pca_residual = -10.0f;
    Eigen::Vector3f eigenvalues;

    // Robot-specific properties
    int through_property = 0;        // 1 if surface is roughly horizontal
    int dimension_property = 0;      // 1 if surface is roughly planar
    int traversability_property = 0; // 1 if traversable
    int contour = 0;                 // 1 if on contour
    float degree = 0.0f;             // Inclination cost
    float curvature = 0.0f;          // Curvature cost

    GNGDTRobotNode()
        : position(Eigen::Vector3f::Zero()),
          color(Eigen::Vector3f::Zero()),
          normal(0.0f, 0.0f, 1.0f),
          eigenvalues(-10.0f, -10.0f, -10.0f) {}

    GNGDTRobotNode(int id_, const Eigen::Vector3f& pos,
                   const Eigen::Vector3f& col = Eigen::Vector3f::Zero())
        : id(id_), position(pos), color(col),
          normal(0.0f, 0.0f, 1.0f), error(0.0f), utility(0.0f),
          pca_residual(-10.0f), eigenvalues(-10.0f, -10.0f, -10.0f),
          through_property(0), dimension_property(0),
          traversability_property(0), contour(0),
          degree(0.0f), curvature(0.0f) {}
};

/**
 * @brief GNG-DT Robot algorithm implementation.
 */
class GrowingNeuralGasDTRobot {
public:
    using Callback = std::function<void(const GrowingNeuralGasDTRobot&, int)>;

    GNGDTRobotParams params;
    std::vector<GNGDTRobotNode> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;

    Eigen::MatrixXi edges_pos;
    Eigen::MatrixXi edges_color;
    Eigen::MatrixXi edges_normal;
    Eigen::MatrixXi edges_traversability;  // pedge
    Eigen::MatrixXi edge_age;

    int n_learning = 0;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    float total_error_ = 0.0f;
    std::mt19937 rng_;
    float cos_max_angle_;

public:
    explicit GrowingNeuralGasDTRobot(const GNGDTRobotParams& params = GNGDTRobotParams(),
                                      unsigned int seed = 0)
        : params(params),
          nodes(params.max_nodes),
          edges_pos(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          edges_color(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          edges_normal(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          edges_traversability(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          edge_age(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          rng_(seed == 0 ? std::random_device{}() : seed),
          cos_max_angle_(std::cos(params.max_angle * M_PI / 180.0f))
    {
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }
    }

    void init() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < 2; ++i) {
            Eigen::Vector3f pos(dist(rng_), dist(rng_), dist(rng_));
            add_node(pos);
        }
        connect_initial_nodes(0, 1);
    }

    void init(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) {
        int id1 = add_node(p1);
        int id2 = add_node(p2);
        if (id1 != -1 && id2 != -1) {
            connect_initial_nodes(id1, id2);
        }
    }

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

    void train(const std::vector<Eigen::Vector3f>& data,
               int n_iterations,
               const Callback& callback = nullptr) {
        train(data, std::vector<Eigen::Vector3f>(), n_iterations, callback);
    }

    int num_nodes() const {
        int count = 0;
        for (const auto& node : nodes) {
            if (node.id != -1) ++count;
        }
        return count;
    }

    int num_edges_pos() const {
        int count = 0;
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id != -1) {
                count += static_cast<int>(neighbors.size());
            }
        }
        return count / 2;
    }

    int num_edges_traversability() const {
        int count = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (nodes[i].id == -1) continue;
            for (int j = i + 1; j < static_cast<int>(nodes.size()); ++j) {
                if (nodes[j].id == -1) continue;
                if (edges_traversability(i, j) > 0) ++count;
            }
        }
        return count;
    }

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

    void get_multi_graph(std::vector<Eigen::Vector3f>& out_nodes,
                         std::vector<std::pair<int, int>>& out_pos_edges,
                         std::vector<std::pair<int, int>>& out_color_edges,
                         std::vector<std::pair<int, int>>& out_normal_edges,
                         std::vector<std::pair<int, int>>& out_traversability_edges) const {
        out_nodes.clear();
        out_pos_edges.clear();
        out_color_edges.clear();
        out_normal_edges.clear();
        out_traversability_edges.clear();

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

        // Other edge types
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
                if (edges_traversability(i, j) > 0) {
                    out_traversability_edges.emplace_back(id_to_idx[i], id_to_idx[j]);
                }
            }
        }
    }

    std::vector<Eigen::Vector3f> get_node_normals() const {
        std::vector<Eigen::Vector3f> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.normal);
            }
        }
        return result;
    }

    std::vector<int> get_traversability() const {
        std::vector<int> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.traversability_property);
            }
        }
        return result;
    }

    std::vector<int> get_contour() const {
        std::vector<int> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.contour);
            }
        }
        return result;
    }

    std::vector<float> get_degree() const {
        std::vector<float> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.degree);
            }
        }
        return result;
    }

    std::vector<float> get_curvature() const {
        std::vector<float> result;
        for (const auto& node : nodes) {
            if (node.id != -1) {
                result.push_back(node.curvature);
            }
        }
        return result;
    }

    std::vector<Eigen::Vector3f> get_traversable_nodes() const {
        std::vector<Eigen::Vector3f> result;
        for (const auto& node : nodes) {
            if (node.id != -1 && node.traversability_property == 1) {
                result.push_back(node.position);
            }
        }
        return result;
    }

    std::vector<Eigen::Vector3f> get_contour_nodes() const {
        std::vector<Eigen::Vector3f> result;
        for (const auto& node : nodes) {
            if (node.id != -1 && node.contour == 1) {
                result.push_back(node.position);
            }
        }
        return result;
    }

private:
    int add_node(const Eigen::Vector3f& position,
                 const Eigen::Vector3f& color = Eigen::Vector3f::Zero()) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = GNGDTRobotNode(node_id, position, color);
        edges_per_node[node_id] = std::unordered_set<int>();
        return node_id;
    }

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

        for (int neighbor_id : neighbors_to_check) {
            if (nodes[neighbor_id].id != -1) {
                auto neighbor_it = edges_per_node.find(neighbor_id);
                if (neighbor_it == edges_per_node.end() || neighbor_it->second.empty()) {
                    remove_node(neighbor_id);
                }
            }
        }
    }

    void connect_initial_nodes(int n1, int n2) {
        edges_pos(n1, n2) = 1;
        edges_pos(n2, n1) = 1;
        edges_color(n1, n2) = 1;
        edges_color(n2, n1) = 1;
        edges_normal(n1, n2) = 1;
        edges_normal(n2, n1) = 1;
        edges_traversability(n1, n2) = 1;
        edges_traversability(n2, n1) = 1;
        edges_per_node[n1].insert(n2);
        edges_per_node[n2].insert(n1);
    }

    void add_position_edge(int n1, int n2) {
        if (edges_pos(n1, n2) == 0) {
            edges_pos(n1, n2) = 1;
            edges_pos(n2, n1) = 1;
            edges_per_node[n1].insert(n2);
            edges_per_node[n2].insert(n1);
        }
    }

    void remove_all_edges(int n1, int n2) {
        edges_pos(n1, n2) = 0;
        edges_pos(n2, n1) = 0;
        edges_color(n1, n2) = 0;
        edges_color(n2, n1) = 0;
        edges_normal(n1, n2) = 0;
        edges_normal(n2, n1) = 0;
        edges_traversability(n1, n2) = 0;
        edges_traversability(n2, n1) = 0;
        edge_age(n1, n2) = 0;
        edge_age(n2, n1) = 0;
        edges_per_node[n1].erase(n2);
        edges_per_node[n2].erase(n1);
    }

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

    std::tuple<Eigen::Vector3f, float, Eigen::Vector3f>
    compute_normal_from_positions(const std::vector<Eigen::Vector3f>& positions,
                                  const Eigen::Vector3f& cog_sum) const {
        int ect = static_cast<int>(positions.size());
        if (ect < 2) {
            return {Eigen::Vector3f(0.0f, 0.0f, 1.0f), -10.0f,
                    Eigen::Vector3f(-10.0f, -10.0f, -10.0f)};
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
            return {Eigen::Vector3f(0.0f, 0.0f, 1.0f), -10.0f,
                    Eigen::Vector3f(-10.0f, -10.0f, -10.0f)};
        }

        Eigen::Vector3f normal = solver.eigenvectors().col(0);
        Eigen::Vector3f eigenvalues = solver.eigenvalues();

        if (normal.y() < 0) {
            normal = -normal;
        }

        float norm = normal.norm();
        if (norm > 1e-10f) {
            normal /= norm;
        } else {
            return {Eigen::Vector3f(0.0f, 0.0f, 1.0f), -10.0f,
                    Eigen::Vector3f(-10.0f, -10.0f, -10.0f)};
        }

        float pca_residual = eigenvalues(0);
        return {normal, pca_residual, eigenvalues};
    }

    int judge_contour(int node_id) const {
        const auto& node = nodes[node_id];

        // Collect pedge neighbors
        std::vector<int> neighbors;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (edges_traversability(node_id, i) == 1 && i != node_id) {
                if (nodes[i].id != -1) {
                    neighbors.push_back(i);
                }
            }
        }

        if (neighbors.size() < 2) {
            return 0;
        }

        // Calculate angles to each neighbor
        std::vector<float> angles;
        for (int neighbor_id : neighbors) {
            const auto& neighbor = nodes[neighbor_id];
            float dx = neighbor.position.x() - node.position.x();
            float dy = neighbor.position.y() - node.position.y();
            float angle = std::atan2(dy, dx) * 180.0f / M_PI;
            if (angle < 0) {
                angle = 360.0f - std::abs(angle);
            }
            angles.push_back(angle);
        }

        std::sort(angles.begin(), angles.end());

        // Check for gaps >= threshold
        float wrap_gap = std::abs(360.0f - angles.back() + angles.front());
        if (wrap_gap >= params.contour_gap_threshold) {
            return 1;
        }

        for (size_t i = 0; i < angles.size() - 1; ++i) {
            float gap = std::abs(angles[i + 1] - angles[i]);
            if (gap >= params.contour_gap_threshold) {
                return 1;
            }
        }

        return 0;
    }

    void update_robot_properties(int s1, int s2,
                                 int traversable_neighbor_count,
                                 int total_neighbors) {
        auto& n1 = nodes[s1];

        // Dimension property
        if (n1.eigenvalues(0) >= 0 && n1.eigenvalues(0) < params.s1thv) {
            n1.dimension_property = 1;
        } else {
            n1.dimension_property = 0;
        }

        // Through property
        if (std::abs(n1.normal.z()) > cos_max_angle_) {
            n1.through_property = 1;
        } else {
            n1.through_property = 0;
        }

        // Degree cost
        if (n1.through_property == 1) {
            n1.degree = (1.0f - std::abs(n1.normal.z())) / (1.0f - cos_max_angle_);
            if (n1.degree > 1.0f) {
                n1.degree = 99.0f;
            }
        } else {
            n1.degree = 99.0f;
        }

        // Curvature cost
        if (n1.pca_residual >= 0 && n1.pca_residual < 0.001f) {
            n1.curvature = n1.pca_residual / 0.001f;
        } else {
            n1.curvature = 99.0f;
        }

        // Traversability property
        if (n1.dimension_property == 1 && n1.through_property == 1) {
            n1.traversability_property = 1;
        } else {
            n1.traversability_property = 0;
            if (total_neighbors > 0 && total_neighbors < 3) {
                if (total_neighbors == traversable_neighbor_count) {
                    n1.traversability_property = 1;
                }
            }
        }

        // Update pedge
        for (int neighbor_id : edges_per_node[s1]) {
            if (neighbor_id == s1) continue;
            const auto& neighbor = nodes[neighbor_id];
            if (n1.traversability_property == neighbor.traversability_property) {
                edges_traversability(s1, neighbor_id) = 1;
                edges_traversability(neighbor_id, s1) = 1;
            } else {
                edges_traversability(s1, neighbor_id) = 0;
                edges_traversability(neighbor_id, s1) = 0;
            }
        }

        // Update contour for s1
        int cct_s1 = 0, ct_s1 = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (edges_traversability(s1, i) == 1 && nodes[i].id != -1) {
                ct_s1++;
                if (nodes[i].contour == 0) {
                    cct_s1++;
                }
            }
        }

        if (n1.traversability_property == 1) {
            n1.contour = judge_contour(s1);
            if (cct_s1 == ct_s1 && n1.through_property == 0) {
                n1.contour = 0;
            }
        } else {
            n1.contour = 0;
        }

        // Update contour for s2
        auto& n2 = nodes[s2];
        int cct_s2 = 0, ct_s2 = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (edges_traversability(s2, i) == 1 && nodes[i].id != -1) {
                ct_s2++;
                if (nodes[i].contour == 0) {
                    cct_s2++;
                }
            }
        }

        if (n2.traversability_property == 1) {
            n2.contour = judge_contour(s2);
            if (cct_s2 == ct_s2 && n2.through_property == 0) {
                n2.contour = 0;
            }
        } else {
            n2.contour = 0;
        }
    }

    void gng_learn(int s1, int s2,
                   const Eigen::Vector3f& v_pos,
                   const Eigen::Vector3f& v_color,
                   float e1, float e2) {
        auto& n1 = nodes[s1];
        auto& n2 = nodes[s2];

        add_position_edge(s1, s2);

        // cedge update
        float color_dist_sq = (n1.color - n2.color).squaredNorm();
        if (color_dist_sq < params.tau_color * params.tau_color) {
            edges_color(s1, s2) = 1;
            edges_color(s2, s1) = 1;
        } else {
            edges_color(s1, s2) = 0;
            edges_color(s2, s1) = 0;
        }

        float normal_dot = n1.normal.dot(n2.normal);
        Eigen::Vector3f s1_original_pos = n1.position;

        edge_age(s1, s2) = 0;
        edge_age(s2, s1) = 0;

        n1.position += e1 * (v_pos - n1.position);
        n1.color += e1 * (v_color - n1.color);

        std::vector<int> neighbors_to_remove;
        std::vector<Eigen::Vector3f> pca_positions;
        pca_positions.push_back(s1_original_pos);
        Eigen::Vector3f pca_cog = s1_original_pos;
        int traversable_neighbor_count = 0;

        for (int neighbor_id : edges_per_node[s1]) {
            if (neighbor_id == s1) continue;

            auto& neighbor = nodes[neighbor_id];
            neighbor.position += e2 * (v_pos - neighbor.position);

            edge_age(s1, neighbor_id)++;
            edge_age(neighbor_id, s1)++;

            pca_positions.push_back(neighbor.position);
            pca_cog += neighbor.position;

            if (neighbor.traversability_property == 1) {
                traversable_neighbor_count++;
            }

            if (edge_age(s1, neighbor_id) > params.max_age) {
                neighbors_to_remove.push_back(neighbor_id);
            }
        }

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

        // PCA
        auto [normal, pca_residual, eigenvalues] =
            compute_normal_from_positions(pca_positions, pca_cog);
        n1.normal = normal;
        n1.pca_residual = pca_residual;
        n1.eigenvalues = eigenvalues;

        // nedge update
        if (std::abs(normal_dot) > params.tau_normal) {
            edges_normal(s1, s2) = 1;
            edges_normal(s2, s1) = 1;
        } else {
            edges_normal(s1, s2) = 0;
            edges_normal(s2, s1) = 0;
        }

        // Robot properties
        int total_neighbors = static_cast<int>(pca_positions.size()) - 1;
        update_robot_properties(s1, s2, traversable_neighbor_count, total_neighbors);
    }

    void discount_errors() {
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error -= params.beta * node.error;
            node.utility -= params.beta * node.utility;
            if (node.error < 0) node.error = 0.0f;
            if (node.utility < 0) node.utility = 0.0f;
        }
    }

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

    bool delete_node_gngu() {
        if (num_nodes() <= 10) return false;

        float min_u = std::numeric_limits<float>::max();
        int min_u_id = -1;
        float min_err = std::numeric_limits<float>::max();

        for (const auto& node : nodes) {
            if (node.id == -1) continue;
            if (node.utility < min_u) {
                min_u = node.utility;
                min_u_id = node.id;
            }
            if (node.error < min_err) {
                min_err = node.error;
            }
        }

        if (min_err < params.thv && min_u_id != -1) {
            remove_node(min_u_id);
            return true;
        }

        return false;
    }

    void node_add() {
        if (addable_indices_.empty()) return;

        float max_err = -1.0f;
        int q = -1;
        float min_u = std::numeric_limits<float>::max();
        int min_u_id = -1;
        float min_err = std::numeric_limits<float>::max();
        std::vector<int> delete_list;
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
            if (node.utility < min_u) {
                min_u = node.utility;
                min_u_id = node.id;
            }
            if (node.error < min_err) {
                min_err = node.error;
            }

            if (node.utility < 0.0001f && node.id != first_node_id) {
                delete_list.push_back(node.id);
            }
        }

        if (q == -1) return;

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

        // Inherit robot properties
        nodes[r].through_property = nodes[q].through_property;
        nodes[r].dimension_property = nodes[q].dimension_property;
        nodes[r].traversability_property = nodes[q].traversability_property;

        // Update edges
        edges_pos(q, f) = 0;
        edges_pos(f, q) = 0;
        edges_per_node[q].erase(f);
        edges_per_node[f].erase(q);

        // Inherit cedge
        edges_color(q, r) = edges_color(q, f);
        edges_color(r, q) = edges_color(q, f);
        edges_color(f, r) = edges_color(q, f);
        edges_color(r, f) = edges_color(q, f);
        edges_color(q, f) = 0;
        edges_color(f, q) = 0;

        // Inherit nedge
        edges_normal(q, r) = edges_normal(q, f);
        edges_normal(r, q) = edges_normal(q, f);
        edges_normal(f, r) = edges_normal(q, f);
        edges_normal(r, f) = edges_normal(q, f);
        edges_normal(q, f) = 0;
        edges_normal(f, q) = 0;

        // Inherit pedge
        edges_traversability(q, r) = edges_traversability(q, f);
        edges_traversability(r, q) = edges_traversability(q, f);
        edges_traversability(f, r) = edges_traversability(q, f);
        edges_traversability(r, f) = edges_traversability(q, f);
        edges_traversability(q, f) = 0;
        edges_traversability(f, q) = 0;

        add_position_edge(q, r);
        add_position_edge(r, f);

        nodes[q].error *= 0.5f;
        nodes[f].error *= 0.5f;
        nodes[q].utility *= 0.5f;
        nodes[f].utility *= 0.5f;
        nodes[r].error = nodes[q].error;
        nodes[r].utility = nodes[q].utility;

        if (num_nodes() > 10 && min_err < params.thv) {
            for (int del_id : delete_list) {
                if (nodes[del_id].id != -1 && del_id != r) {
                    remove_node(del_id);
                }
            }
        }
    }

    float one_train_update(const Eigen::Vector3f& position,
                           const Eigen::Vector3f& color) {
        auto [s1, s2, dist1_sq, dist2_sq] = find_two_nearest(position);

        if (s1 == -1 || s2 == -1) return 0.0f;

        if (dist1_sq > params.dis_thv * params.dis_thv &&
            num_nodes() < params.max_nodes - 2) {
            add_new_node_distance(position, color);
            discount_errors();
            return 0.0f;
        }

        nodes[s1].error += dist1_sq;
        nodes[s1].utility += dist2_sq - dist1_sq;

        gng_learn(s1, s2, position, color, params.eps_b, params.eps_n);
        discount_errors();

        n_learning++;
        return dist1_sq;
    }

    void gng_main_cycle(const std::vector<Eigen::Vector3f>& data,
                        const std::vector<Eigen::Vector3f>& colors) {
        std::uniform_int_distribution<int> dist(0, static_cast<int>(data.size()) - 1);
        bool has_colors = !colors.empty();
        float total_error = 0.0f;

        for (int i = 0; i < params.lambda; ++i) {
            int idx = dist(rng_);
            Eigen::Vector3f color = has_colors ? colors[idx] : Eigen::Vector3f::Zero();

            if (i == params.lambda / 2) {
                float error = one_train_update(data[idx], color);
                total_error += error;
                if (num_nodes() > 2) {
                    delete_node_gngu();
                }
            } else {
                float error = one_train_update(data[idx], color);
                total_error += error;
            }
        }

        total_error /= params.lambda;
        if (num_nodes() < params.max_nodes && total_error > params.thv) {
            node_add();
        }
    }
};

// Alias
using GNGDTRobot = GrowingNeuralGasDTRobot;

}  // namespace gng_dt
