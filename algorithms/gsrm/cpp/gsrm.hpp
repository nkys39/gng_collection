/**
 * @file gsrm.hpp
 * @brief Growing Self-Reconstruction Meshes (GSRM) implementation
 *
 * Based on:
 *   - Rêgo, R. L. M. E., Araújo, A. F. R., & Lima Neto, F. B. (2007).
 *     "Growing Self-Organizing Maps for Surface Reconstruction from
 *     Unstructured Point Clouds" (IJCNN 2007)
 *
 * GSRM extends GNG to produce triangular meshes instead of wireframes.
 * Key differences from standard GNG:
 *   1. Extended Competitive Hebbian Learning (ECHL) - creates faces
 *   2. Edge and face removal - removes faces incident to old edges
 *   3. GCS-style vertex insertion - splits faces when inserting nodes
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace gsrm {

/**
 * @brief GSRM hyperparameters.
 */
struct GSRMParams {
    int max_nodes = 500;    // Maximum number of nodes
    int lambda = 50;        // Node insertion interval
    float eps_b = 0.1f;     // Winner learning rate
    float eps_n = 0.01f;    // Neighbor learning rate
    float alpha = 0.5f;     // Error decay on split
    float beta = 0.005f;    // Global error decay
    int max_age = 50;       // Maximum edge age
};

/**
 * @brief A node in the GSRM network.
 *
 * @tparam PointT Point type (Eigen vector)
 */
template <typename PointT>
struct GSRMNode {
    int id = -1;            // -1 means invalid/removed
    float error = 0.0f;     // Accumulated error
    PointT weight;          // Position vector

    GSRMNode() = default;
    GSRMNode(int id_, const PointT& weight_)
        : id(id_), error(0.0f), weight(weight_) {}
};

/**
 * @brief Helper for edge key (canonical pair with smaller ID first).
 */
inline std::pair<int, int> edge_key(int n1, int n2) {
    return (n1 < n2) ? std::make_pair(n1, n2) : std::make_pair(n2, n1);
}

/**
 * @brief Hash function for edge pairs.
 */
struct EdgeHash {
    std::size_t operator()(const std::pair<int, int>& edge) const {
        return std::hash<int64_t>()(
            static_cast<int64_t>(edge.first) * 100000LL + edge.second
        );
    }
};

/**
 * @brief Face represented as sorted tuple of 3 node IDs.
 */
struct Face {
    int v0, v1, v2;

    Face() : v0(-1), v1(-1), v2(-1) {}

    Face(int a, int b, int c) {
        // Sort vertices for canonical representation
        std::array<int, 3> arr = {a, b, c};
        std::sort(arr.begin(), arr.end());
        v0 = arr[0];
        v1 = arr[1];
        v2 = arr[2];
    }

    bool operator==(const Face& other) const {
        return v0 == other.v0 && v1 == other.v1 && v2 == other.v2;
    }

    std::array<std::pair<int, int>, 3> edges() const {
        return {
            edge_key(v0, v1),
            edge_key(v1, v2),
            edge_key(v0, v2)
        };
    }

    std::array<int, 3> vertices() const {
        return {v0, v1, v2};
    }
};

/**
 * @brief Hash function for faces.
 */
struct FaceHash {
    std::size_t operator()(const Face& f) const {
        return std::hash<int64_t>()(
            static_cast<int64_t>(f.v0) * 1000000000LL +
            static_cast<int64_t>(f.v1) * 10000LL +
            f.v2
        );
    }
};

/**
 * @brief Growing Self-Reconstruction Meshes algorithm.
 *
 * GSRM is a surface reconstruction method based on GNG that produces
 * triangular meshes from 3D point clouds. It uses Extended Competitive
 * Hebbian Learning to create triangular faces.
 *
 * @tparam PointT Point type (Eigen vector, typically Eigen::Vector3f)
 */
template <typename PointT>
class GSRM {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GSRM&, int)>;

    GSRMParams params;
    std::vector<GSRMNode<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edge_ages;  // Edge age matrix
    int n_learning = 0;

    // Face management
    std::unordered_map<int, Face> faces;  // face_id -> Face
    std::unordered_map<std::pair<int, int>, std::unordered_set<int>, EdgeHash> faces_per_edge;
    std::unordered_map<int, std::unordered_set<int>> faces_per_node;

private:
    std::deque<int> addable_indices_;
    int n_trial_ = 0;
    int next_face_id_ = 0;
    std::mt19937 rng_;

public:
    /**
     * @brief Construct GSRM with given parameters.
     *
     * @param params GSRM hyperparameters
     * @param seed Random seed (0 for random)
     */
    explicit GSRM(const GSRMParams& params = GSRMParams(), unsigned int seed = 0)
        : params(params),
          nodes(params.max_nodes),
          edge_ages(Eigen::MatrixXi::Zero(params.max_nodes, params.max_nodes)),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        // Initialize addable indices queue
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }

        // Initialize with a triangle (minimum simplicial complex)
        init_triangle();
    }

    /**
     * @brief Initialize with a triangle of 3 random nodes.
     */
    void init_triangle() {
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);

        // Create 3 random nodes
        std::vector<int> node_ids;
        for (int i = 0; i < 3; ++i) {
            PointT weight;
            for (int j = 0; j < static_cast<int>(weight.size()); ++j) {
                weight(j) = dist(rng_);
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

        // Create the initial face
        add_face(node_ids[0], node_ids[1], node_ids[2]);
    }

    /**
     * @brief Train on data for multiple iterations.
     *
     * @param data Vector of training samples (3D points)
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
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id != -1) {
                count += static_cast<int>(neighbors.size());
            }
        }
        return count / 2;  // Each edge counted twice
    }

    /**
     * @brief Get number of faces.
     */
    int num_faces() const {
        return static_cast<int>(faces.size());
    }

    /**
     * @brief Get mesh with sequential indices (for visualization).
     *
     * @param out_nodes Output: node positions array
     * @param out_edges Output: edges with sequential indices
     * @param out_faces Output: faces with sequential indices
     */
    void get_mesh(std::vector<PointT>& out_nodes,
                  std::vector<std::pair<int, int>>& out_edges,
                  std::vector<std::array<int, 3>>& out_faces) const {
        out_nodes.clear();
        out_edges.clear();
        out_faces.clear();

        std::unordered_map<int, int> id_to_idx;

        // Collect active nodes
        for (const auto& node : nodes) {
            if (node.id != -1) {
                id_to_idx[node.id] = static_cast<int>(out_nodes.size());
                out_nodes.push_back(node.weight);
            }
        }

        // Convert edges to new indices
        std::unordered_set<int64_t> seen_edges;
        for (const auto& [node_id, neighbors] : edges_per_node) {
            if (nodes[node_id].id == -1) continue;
            for (int neighbor_id : neighbors) {
                if (nodes[neighbor_id].id == -1) continue;
                int64_t key = std::min(node_id, neighbor_id) * 100000LL +
                              std::max(node_id, neighbor_id);
                if (seen_edges.find(key) == seen_edges.end()) {
                    seen_edges.insert(key);
                    out_edges.emplace_back(id_to_idx[node_id], id_to_idx[neighbor_id]);
                }
            }
        }

        // Convert faces to new indices
        for (const auto& [face_id, face] : faces) {
            auto verts = face.vertices();
            // Check all vertices are valid
            bool valid = true;
            for (int v : verts) {
                if (nodes[v].id == -1) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                out_faces.push_back({
                    id_to_idx[verts[0]],
                    id_to_idx[verts[1]],
                    id_to_idx[verts[2]]
                });
            }
        }
    }

    /**
     * @brief Get graph (nodes and edges only, for GNG compatibility).
     */
    void get_graph(std::vector<PointT>& out_nodes,
                   std::vector<std::pair<int, int>>& out_edges) const {
        std::vector<std::array<int, 3>> dummy_faces;
        get_mesh(out_nodes, out_edges, dummy_faces);
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
        nodes[node_id] = GSRMNode<PointT>(node_id, weight);
        edges_per_node[node_id] = std::unordered_set<int>();
        faces_per_node[node_id] = std::unordered_set<int>();
        return node_id;
    }

    /**
     * @brief Remove a node (only if isolated - no edges).
     */
    void remove_node(int node_id) {
        auto it = edges_per_node.find(node_id);
        if (it != edges_per_node.end() && !it->second.empty()) return;

        edges_per_node.erase(node_id);
        faces_per_node.erase(node_id);
        nodes[node_id].id = -1;
        addable_indices_.push_back(node_id);
    }

    /**
     * @brief Add or reset edge between two nodes.
     */
    void add_edge(int n1, int n2) {
        if (n1 == n2) return;

        auto key = edge_key(n1, n2);

        if (edge_ages(n1, n2) > 0) {
            // Edge exists, reset age
            edge_ages(n1, n2) = 1;
            edge_ages(n2, n1) = 1;
        } else {
            // New edge
            edges_per_node[n1].insert(n2);
            edges_per_node[n2].insert(n1);
            edge_ages(n1, n2) = 1;
            edge_ages(n2, n1) = 1;
            // Initialize faces_per_edge
            if (faces_per_edge.find(key) == faces_per_edge.end()) {
                faces_per_edge[key] = std::unordered_set<int>();
            }
        }
    }

    /**
     * @brief Remove edge between two nodes.
     */
    void remove_edge(int n1, int n2) {
        auto key = edge_key(n1, n2);

        edges_per_node[n1].erase(n2);
        edges_per_node[n2].erase(n1);
        edge_ages(n1, n2) = 0;
        edge_ages(n2, n1) = 0;

        // Remove from faces_per_edge tracking
        faces_per_edge.erase(key);
    }

    /**
     * @brief Add a triangular face.
     *
     * @return Face ID, or -1 if face already exists
     */
    int add_face(int n1, int n2, int n3) {
        Face new_face(n1, n2, n3);

        // Check if face already exists
        for (const auto& [face_id, face] : faces) {
            if (face == new_face) {
                return face_id;  // Already exists
            }
        }

        // Create new face
        int face_id = next_face_id_++;
        faces[face_id] = new_face;

        // Update faces_per_edge
        for (const auto& edge : new_face.edges()) {
            if (faces_per_edge.find(edge) == faces_per_edge.end()) {
                faces_per_edge[edge] = std::unordered_set<int>();
            }
            faces_per_edge[edge].insert(face_id);
        }

        // Update faces_per_node
        for (int v : new_face.vertices()) {
            faces_per_node[v].insert(face_id);
        }

        return face_id;
    }

    /**
     * @brief Remove a face.
     */
    void remove_face(int face_id) {
        auto it = faces.find(face_id);
        if (it == faces.end()) return;

        const Face& face = it->second;

        // Remove from faces_per_edge
        for (const auto& edge : face.edges()) {
            auto edge_it = faces_per_edge.find(edge);
            if (edge_it != faces_per_edge.end()) {
                edge_it->second.erase(face_id);
            }
        }

        // Remove from faces_per_node
        for (int v : face.vertices()) {
            auto node_it = faces_per_node.find(v);
            if (node_it != faces_per_node.end()) {
                node_it->second.erase(face_id);
            }
        }

        // Remove face
        faces.erase(face_id);
    }

    /**
     * @brief Find the three nearest nodes to input x.
     *
     * @return Tuple of (s1_id, s2_id, s3_id) - first, second, third nearest
     */
    std::tuple<int, int, int> find_three_nearest(const PointT& x) const {
        std::vector<std::pair<Scalar, int>> distances;

        for (const auto& node : nodes) {
            if (node.id == -1) continue;
            Scalar dist = (x - node.weight).squaredNorm();
            distances.emplace_back(dist, node.id);
        }

        // Sort by distance
        std::sort(distances.begin(), distances.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        if (distances.size() < 3) {
            return {-1, -1, -1};
        }

        return {distances[0].second, distances[1].second, distances[2].second};
    }

    /**
     * @brief Extended Competitive Hebbian Learning.
     *
     * Creates or reinforces edges between the three winners and
     * creates a triangular face if it doesn't exist.
     */
    void extended_chl(int s1, int s2, int s3) {
        // Create/reinforce edges between all pairs
        add_edge(s1, s2);
        add_edge(s2, s3);
        add_edge(s1, s3);

        // Create face
        add_face(s1, s2, s3);
    }

    /**
     * @brief Find nodes that are neighbors of both n1 and n2.
     */
    std::unordered_set<int> find_common_neighbors(int n1, int n2) const {
        std::unordered_set<int> common;
        auto it1 = edges_per_node.find(n1);
        auto it2 = edges_per_node.find(n2);
        if (it1 == edges_per_node.end() || it2 == edges_per_node.end()) {
            return common;
        }

        for (int neighbor : it1->second) {
            if (it2->second.count(neighbor) > 0) {
                common.insert(neighbor);
            }
        }
        return common;
    }

    /**
     * @brief Remove invalid edges and their incident faces.
     */
    void remove_invalid_edges_and_faces(int s1) {
        auto it = edges_per_node.find(s1);
        if (it == edges_per_node.end()) return;

        std::vector<int> edges_to_check(it->second.begin(), it->second.end());

        // Step 1: Find invalid edges and remove their incident faces
        std::vector<int> invalid_edges;
        for (int neighbor_id : edges_to_check) {
            if (edge_ages(s1, neighbor_id) > params.max_age) {
                invalid_edges.push_back(neighbor_id);

                // Remove faces incident to this edge
                auto key = edge_key(s1, neighbor_id);
                auto face_it = faces_per_edge.find(key);
                if (face_it != faces_per_edge.end()) {
                    std::vector<int> face_ids(face_it->second.begin(), face_it->second.end());
                    for (int face_id : face_ids) {
                        remove_face(face_id);
                    }
                }
            }
        }

        // Step 2: Remove invalid edges
        for (int neighbor_id : invalid_edges) {
            remove_edge(s1, neighbor_id);
        }

        // Step 3: Remove isolated nodes
        for (int neighbor_id : invalid_edges) {
            auto neighbor_it = edges_per_node.find(neighbor_id);
            if (neighbor_it == edges_per_node.end() || neighbor_it->second.empty()) {
                remove_node(neighbor_id);
            }
        }
    }

    /**
     * @brief Insert a new node using GCS-style insertion.
     *
     * @return ID of new node, or -1 if failed
     */
    int insert_node_gcs() {
        if (addable_indices_.empty()) return -1;

        // Find node q with maximum error
        float max_err = -1.0f;
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

        // Get faces incident to (q, f) edge before modification
        auto key = edge_key(q_id, f_id);
        std::vector<int> incident_face_ids;
        auto face_it = faces_per_edge.find(key);
        if (face_it != faces_per_edge.end()) {
            incident_face_ids = std::vector<int>(face_it->second.begin(), face_it->second.end());
        }

        // Find common neighbors
        auto common_neighbors = find_common_neighbors(q_id, f_id);

        // Create new node at midpoint
        PointT new_weight = (nodes[q_id].weight + nodes[f_id].weight) * static_cast<Scalar>(0.5);
        int r_id = add_node(new_weight);

        if (r_id == -1) return -1;

        // Update errors
        nodes[q_id].error *= (1.0f - params.alpha);
        nodes[f_id].error *= (1.0f - params.alpha);
        nodes[r_id].error = (nodes[q_id].error + nodes[f_id].error) * 0.5f;

        // Process incident faces: split each into two
        for (int face_id : incident_face_ids) {
            auto it = faces.find(face_id);
            if (it == faces.end()) continue;

            Face face = it->second;
            remove_face(face_id);

            // Find the third vertex (not q or f)
            int third_vertex = -1;
            for (int v : face.vertices()) {
                if (v != q_id && v != f_id) {
                    third_vertex = v;
                    break;
                }
            }

            if (third_vertex != -1) {
                // Create two new faces: (q, r, third) and (r, f, third)
                add_face(q_id, r_id, third_vertex);
                add_face(r_id, f_id, third_vertex);
            }
        }

        // Remove old edge (q, f)
        remove_edge(q_id, f_id);

        // Add new edges
        add_edge(q_id, r_id);
        add_edge(f_id, r_id);

        // Connect to common neighbors
        for (int cn : common_neighbors) {
            add_edge(r_id, cn);
        }

        return r_id;
    }

    /**
     * @brief Single training update following GSRM algorithm.
     */
    void one_train_update(const PointT& sample) {
        // Find three nearest nodes (Step 3)
        auto [s1_id, s2_id, s3_id] = find_three_nearest(sample);

        if (s1_id == -1 || s2_id == -1 || s3_id == -1) return;

        // Extended CHL: create/reinforce edges and face (Step 4)
        extended_chl(s1_id, s2_id, s3_id);

        // Update winner error (Step 5): ΔE_s1 = ||w_s1 - ξ||²
        Scalar dist_sq = (sample - nodes[s1_id].weight).squaredNorm();
        nodes[s1_id].error += static_cast<float>(dist_sq);

        // Move winner toward sample (Step 6)
        nodes[s1_id].weight += params.eps_b * (sample - nodes[s1_id].weight);

        // Move neighbors toward sample (Step 6)
        auto it = edges_per_node.find(s1_id);
        if (it != edges_per_node.end()) {
            for (int neighbor_id : it->second) {
                nodes[neighbor_id].weight += params.eps_n * (sample - nodes[neighbor_id].weight);
            }
        }

        // Update edge ages (Step 7): age = age + 1
        if (it != edges_per_node.end()) {
            for (int neighbor_id : it->second) {
                edge_ages(s1_id, neighbor_id)++;
                edge_ages(neighbor_id, s1_id)++;
            }
        }

        // Remove invalid edges and faces (Step 8)
        remove_invalid_edges_and_faces(s1_id);

        // Periodically insert new node (Step 9)
        n_trial_++;
        if (n_trial_ >= params.lambda) {
            n_trial_ = 0;
            if (!addable_indices_.empty()) {
                insert_node_gcs();
            }
        }

        // Decay all errors (Step 10): ΔE_s = -βE_s
        for (auto& node : nodes) {
            if (node.id == -1) continue;
            node.error *= (1.0f - params.beta);
        }

        n_learning++;
    }
};

// Common type aliases (GSRM is specifically for 3D surface reconstruction)
using GSRM3f = GSRM<Eigen::Vector3f>;
using GSRM3d = GSRM<Eigen::Vector3d>;

}  // namespace gsrm
