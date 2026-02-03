/**
 * @file gng_efficient.hpp
 * @brief Optimized Growing Neural Gas (GNG) implementation
 *
 * Based on:
 *   Fišer, D., Faigl, J., & Kulich, M. (2013).
 *   "Growing Neural Gas Efficiently"
 *   Neurocomputing.
 *
 * Two key optimizations:
 *   1. Uniform Grid for O(1) nearest neighbor search
 *   2. Lazy error evaluation with cycle counters
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace gng_efficient {

/**
 * @brief GNG Efficient hyperparameters.
 */
struct GNGEfficientParams {
    // Standard GNG parameters (paper Table 2 defaults)
    int max_nodes = 100;
    int lambda = 200;
    float eps_b = 0.05f;
    float eps_n = 0.0006f;
    float alpha = 0.95f;
    float beta = 0.9995f;
    int max_age = 200;

    // Optimization parameters
    float h_t = 0.1f;       // Grid density threshold
    float h_rho = 1.5f;     // Grid expansion factor
    bool use_uniform_grid = true;
    bool use_lazy_error = true;
};

// Forward declaration
template <typename PointT>
struct NeuronNode;

/**
 * @brief Uniform Grid for nearest neighbor search.
 *
 * @tparam PointT Point type (Eigen vector)
 */
template <typename PointT>
class UniformGrid {
public:
    using Scalar = typename PointT::Scalar;
    static constexpr int Dim = PointT::RowsAtCompileTime;

    float h_t;
    float h_rho;

private:
    PointT origin_;
    Scalar cell_size_;
    Eigen::Matrix<int, Dim, 1> grid_dims_;
    std::unordered_map<int64_t, std::vector<NeuronNode<PointT>*>> cells_;
    std::unordered_map<int, int64_t> node_cell_map_;
    int n_nodes_ = 0;

public:
    UniformGrid(float h_t_ = 0.1f, float h_rho_ = 1.5f)
        : h_t(h_t_), h_rho(h_rho_),
          origin_(PointT::Zero()),
          cell_size_(1.0),
          grid_dims_(Eigen::Matrix<int, Dim, 1>::Ones())
    {}

    void insert(NeuronNode<PointT>* node) {
        if (n_nodes_ == 0) {
            origin_ = node->weight - PointT::Constant(0.5);
            cell_size_ = 1.0;
            grid_dims_ = Eigen::Matrix<int, Dim, 1>::Ones();
        }

        expand_if_needed(node->weight);
        int64_t cell_idx = get_cell_index(node->weight);
        cells_[cell_idx].push_back(node);
        node_cell_map_[node->id] = cell_idx;
        n_nodes_++;

        if (get_density() > h_t) {
            rebuild_grid();
        }
    }

    void remove(NeuronNode<PointT>* node) {
        auto it = node_cell_map_.find(node->id);
        if (it == node_cell_map_.end()) return;

        int64_t cell_idx = it->second;
        auto& cell = cells_[cell_idx];
        cell.erase(std::remove(cell.begin(), cell.end(), node), cell.end());
        if (cell.empty()) cells_.erase(cell_idx);

        node_cell_map_.erase(it);
        n_nodes_--;
    }

    void update(NeuronNode<PointT>* node) {
        auto it = node_cell_map_.find(node->id);
        if (it == node_cell_map_.end()) {
            insert(node);
            return;
        }

        int64_t old_cell = it->second;
        int64_t new_cell = get_cell_index(node->weight);

        if (old_cell == new_cell) return;

        // Remove from old cell
        auto& old_cell_vec = cells_[old_cell];
        old_cell_vec.erase(std::remove(old_cell_vec.begin(), old_cell_vec.end(), node),
                          old_cell_vec.end());
        if (old_cell_vec.empty()) cells_.erase(old_cell);

        // Expand if needed
        expand_if_needed(node->weight);

        // Add to new cell
        new_cell = get_cell_index(node->weight);
        cells_[new_cell].push_back(node);
        node_cell_map_[node->id] = new_cell;
    }

    std::pair<NeuronNode<PointT>*, NeuronNode<PointT>*>
    find_two_nearest(const PointT& point) {
        if (n_nodes_ < 2) {
            std::vector<NeuronNode<PointT>*> all_nodes;
            for (auto& [idx, cell] : cells_) {
                for (auto* node : cell) {
                    all_nodes.push_back(node);
                }
            }
            if (all_nodes.empty()) return {nullptr, nullptr};
            if (all_nodes.size() == 1) return {all_nodes[0], nullptr};

            std::sort(all_nodes.begin(), all_nodes.end(),
                [&point](auto* a, auto* b) {
                    return (a->weight - point).squaredNorm() <
                           (b->weight - point).squaredNorm();
                });
            return {all_nodes[0], all_nodes[1]};
        }

        auto cell_coords = get_cell_coords(point);
        Scalar b = compute_boundary_distance(point, cell_coords);

        NeuronNode<PointT>* best1 = nullptr;
        NeuronNode<PointT>* best2 = nullptr;
        Scalar dist1 = std::numeric_limits<Scalar>::max();
        Scalar dist2 = std::numeric_limits<Scalar>::max();

        int max_radius = grid_dims_.maxCoeff();

        for (int radius = 0; radius <= max_radius; ++radius) {
            auto cells_to_search = get_cells_at_radius(cell_coords, radius);

            for (int64_t cell_idx : cells_to_search) {
                auto it = cells_.find(cell_idx);
                if (it == cells_.end()) continue;

                for (auto* node : it->second) {
                    Scalar d = (node->weight - point).squaredNorm();
                    if (d < dist1) {
                        dist2 = dist1;
                        best2 = best1;
                        dist1 = d;
                        best1 = node;
                    } else if (d < dist2) {
                        dist2 = d;
                        best2 = node;
                    }
                }
            }

            Scalar threshold = (b + radius * cell_size_) * (b + radius * cell_size_);
            if (best1 && best2 && dist1 <= threshold && dist2 <= threshold) {
                break;
            }
        }

        return {best1, best2};
    }

private:
    Eigen::Matrix<int, Dim, 1> get_cell_coords(const PointT& pos) const {
        Eigen::Matrix<int, Dim, 1> coords;
        for (int d = 0; d < Dim; ++d) {
            coords(d) = std::clamp(
                static_cast<int>(std::floor((pos(d) - origin_(d)) / cell_size_)),
                0, grid_dims_(d) - 1
            );
        }
        return coords;
    }

    int64_t get_cell_index(const PointT& pos) const {
        auto coords = get_cell_coords(pos);
        int64_t idx = 0;
        int64_t mult = 1;
        for (int d = 0; d < Dim; ++d) {
            idx += coords(d) * mult;
            mult *= grid_dims_(d);
        }
        return idx;
    }

    int64_t coords_to_index(const Eigen::Matrix<int, Dim, 1>& coords) const {
        int64_t idx = 0;
        int64_t mult = 1;
        for (int d = 0; d < Dim; ++d) {
            idx += coords(d) * mult;
            mult *= grid_dims_(d);
        }
        return idx;
    }

    float get_density() const {
        int64_t n_cells = 1;
        for (int d = 0; d < Dim; ++d) n_cells *= grid_dims_(d);
        return n_cells > 0 ? static_cast<float>(n_nodes_) / n_cells : 0;
    }

    void expand_if_needed(const PointT& pos) {
        PointT max_coords = origin_;
        for (int d = 0; d < Dim; ++d) {
            max_coords(d) += grid_dims_(d) * cell_size_;
        }

        bool needs_expand = false;
        PointT new_origin = origin_;
        Eigen::Matrix<int, Dim, 1> new_dims = grid_dims_;

        for (int d = 0; d < Dim; ++d) {
            if (pos(d) < origin_(d)) {
                int extra = static_cast<int>(
                    std::ceil((origin_(d) - pos(d)) / cell_size_)
                );
                new_origin(d) -= extra * cell_size_;
                new_dims(d) += extra;
                needs_expand = true;
            } else if (pos(d) >= max_coords(d)) {
                int extra = static_cast<int>(
                    std::ceil((pos(d) - max_coords(d) + 1e-6) / cell_size_)
                );
                new_dims(d) += extra;
                needs_expand = true;
            }
        }

        if (needs_expand) {
            std::vector<NeuronNode<PointT>*> all_nodes;
            for (auto& [idx, cell] : cells_) {
                for (auto* node : cell) {
                    all_nodes.push_back(node);
                }
            }

            origin_ = new_origin;
            grid_dims_ = new_dims;
            cells_.clear();
            node_cell_map_.clear();

            for (auto* node : all_nodes) {
                int64_t cell_idx = get_cell_index(node->weight);
                cells_[cell_idx].push_back(node);
                node_cell_map_[node->id] = cell_idx;
            }
        }
    }

    void rebuild_grid() {
        if (n_nodes_ == 0) return;

        std::vector<NeuronNode<PointT>*> all_nodes;
        for (auto& [idx, cell] : cells_) {
            for (auto* node : cell) {
                all_nodes.push_back(node);
            }
        }

        float scale = std::pow(h_rho, 1.0f / Dim);
        Eigen::Matrix<int, Dim, 1> new_dims;
        for (int d = 0; d < Dim; ++d) {
            new_dims(d) = std::max(static_cast<int>(std::ceil(grid_dims_(d) * scale)), 1);
        }

        Scalar min_new_size = std::numeric_limits<Scalar>::max();
        for (int d = 0; d < Dim; ++d) {
            Scalar old_extent = grid_dims_(d) * cell_size_;
            min_new_size = std::min(min_new_size, old_extent / new_dims(d));
        }

        grid_dims_ = new_dims;
        cell_size_ = min_new_size;
        cells_.clear();
        node_cell_map_.clear();

        for (auto* node : all_nodes) {
            int64_t cell_idx = get_cell_index(node->weight);
            cells_[cell_idx].push_back(node);
            node_cell_map_[node->id] = cell_idx;
        }
    }

    Scalar compute_boundary_distance(const PointT& point,
                                      const Eigen::Matrix<int, Dim, 1>& coords) const {
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        for (int d = 0; d < Dim; ++d) {
            Scalar lower = std::abs(point(d) - origin_(d) - coords(d) * cell_size_);
            Scalar upper = std::abs(point(d) - origin_(d) - (coords(d) + 1) * cell_size_);
            min_dist = std::min(min_dist, std::min(lower, upper));
        }
        return min_dist;
    }

    std::vector<int64_t> get_cells_at_radius(
            const Eigen::Matrix<int, Dim, 1>& center, int radius) {
        std::vector<int64_t> result;

        if (radius == 0) {
            bool in_bounds = true;
            for (int d = 0; d < Dim; ++d) {
                if (center(d) < 0 || center(d) >= grid_dims_(d)) {
                    in_bounds = false;
                    break;
                }
            }
            if (in_bounds) {
                result.push_back(coords_to_index(center));
            }
            return result;
        }

        // Generate all cells at Chebyshev distance == radius
        std::vector<Eigen::Matrix<int, Dim, 1>> cells;
        generate_cells_recursive(center, radius, 0, Eigen::Matrix<int, Dim, 1>(), cells);

        for (const auto& coords : cells) {
            // Check if exactly at radius
            int max_diff = 0;
            for (int d = 0; d < Dim; ++d) {
                max_diff = std::max(max_diff, std::abs(coords(d) - center(d)));
            }
            if (max_diff == radius) {
                result.push_back(coords_to_index(coords));
            }
        }

        return result;
    }

    void generate_cells_recursive(
            const Eigen::Matrix<int, Dim, 1>& center,
            int radius, int dim,
            Eigen::Matrix<int, Dim, 1> current,
            std::vector<Eigen::Matrix<int, Dim, 1>>& result) {
        if (dim == Dim) {
            result.push_back(current);
            return;
        }

        for (int offset = -radius; offset <= radius; ++offset) {
            int coord = center(dim) + offset;
            if (coord >= 0 && coord < grid_dims_(dim)) {
                current(dim) = coord;
                generate_cells_recursive(center, radius, dim + 1, current, result);
            }
        }
    }
};

/**
 * @brief A neuron node in the GNG Efficient network.
 */
template <typename PointT>
struct NeuronNode {
    int id = -1;
    float error = 0.0f;
    int cycle = 0;
    PointT weight;

    NeuronNode() = default;
    NeuronNode(int id_, const PointT& weight_, float error_ = 0.0f, int cycle_ = 0)
        : id(id_), error(error_), cycle(cycle_), weight(weight_) {}
};

/**
 * @brief Lazy Heap for error handling.
 *
 * @tparam PointT Point type
 */
template <typename PointT>
class LazyHeap {
public:
    using FixErrorFn = std::function<void(int, NeuronNode<PointT>*)>;

private:
    struct HeapEntry {
        float neg_error;
        int cycle;
        int node_id;
        NeuronNode<PointT>* node;

        bool operator<(const HeapEntry& other) const {
            return neg_error > other.neg_error;  // Min-heap for max error
        }
    };

    std::priority_queue<HeapEntry> heap_;
    std::vector<NeuronNode<PointT>*> pending_;
    std::unordered_set<int> valid_ids_;
    FixErrorFn fix_error_fn_;

public:
    explicit LazyHeap(FixErrorFn fix_fn) : fix_error_fn_(std::move(fix_fn)) {}

    void insert(NeuronNode<PointT>* node) {
        pending_.push_back(node);
    }

    void update(NeuronNode<PointT>* node) {
        valid_ids_.erase(node->id);
        pending_.push_back(node);
    }

    void remove(NeuronNode<PointT>* node) {
        valid_ids_.erase(node->id);
        pending_.erase(
            std::remove(pending_.begin(), pending_.end(), node),
            pending_.end()
        );
    }

    NeuronNode<PointT>* top(int current_cycle) {
        // Flush pending
        for (auto* node : pending_) {
            if (node->id < 0) continue;
            heap_.push({-node->error, node->cycle, node->id, node});
            valid_ids_.insert(node->id);
        }
        pending_.clear();

        while (!heap_.empty()) {
            HeapEntry entry = heap_.top();
            heap_.pop();

            if (valid_ids_.find(entry.node_id) == valid_ids_.end()) continue;
            if (entry.node->id < 0) {
                valid_ids_.erase(entry.node_id);
                continue;
            }

            if (entry.node->cycle == current_cycle) {
                heap_.push(entry);
                return entry.node;
            }

            fix_error_fn_(current_cycle, entry.node);
            heap_.push({-entry.node->error, current_cycle, entry.node_id, entry.node});
        }

        return nullptr;
    }

    void clear() {
        while (!heap_.empty()) heap_.pop();
        pending_.clear();
        valid_ids_.clear();
    }
};

/**
 * @brief Optimized Growing Neural Gas algorithm.
 *
 * @tparam PointT Point type (Eigen vector)
 */
template <typename PointT>
class GNGEfficient {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GNGEfficient&, int)>;
    static constexpr int Dim = PointT::RowsAtCompileTime;

    GNGEfficientParams params;
    std::vector<NeuronNode<PointT>> nodes;
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;
    Eigen::MatrixXi edges;
    int n_learning = 0;
    int cycle = 0;
    int step = 0;

private:
    std::deque<int> addable_indices_;
    std::mt19937 rng_;
    std::vector<float> beta_powers_;
    UniformGrid<PointT>* grid_ = nullptr;
    LazyHeap<PointT>* error_heap_ = nullptr;

public:
    explicit GNGEfficient(const GNGEfficientParams& params_ = GNGEfficientParams(),
                          unsigned int seed = 0)
        : params(params_),
          nodes(params_.max_nodes),
          edges(Eigen::MatrixXi::Zero(params_.max_nodes, params_.max_nodes)),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        for (int i = 0; i < params.max_nodes; ++i) {
            addable_indices_.push_back(i);
        }

        // Pre-compute beta powers
        beta_powers_.resize(params.lambda + 1);
        for (int i = 0; i <= params.lambda; ++i) {
            beta_powers_[i] = std::pow(params.beta, static_cast<float>(i));
        }

        // Initialize uniform grid
        if (params.use_uniform_grid) {
            grid_ = new UniformGrid<PointT>(params.h_t, params.h_rho);
        }

        // Initialize lazy heap
        if (params.use_lazy_error) {
            error_heap_ = new LazyHeap<PointT>(
                [this](int c, NeuronNode<PointT>* node) {
                    fix_error(c, node);
                }
            );
        }
    }

    ~GNGEfficient() {
        delete grid_;
        delete error_heap_;
    }

    void init() {
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (int i = 0; i < 2; ++i) {
            PointT weight;
            for (int j = 0; j < Dim; ++j) {
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
            adapt(data[idx]);
            n_learning++;

            if (n_learning % params.lambda == 0) {
                insert_node();
                cycle++;
            }

            if (callback) {
                callback(*this, iter);
            }
        }
    }

    void partial_fit(const PointT& sample) {
        adapt(sample);
        n_learning++;

        if (n_learning % params.lambda == 0) {
            insert_node();
            cycle++;
        }
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
                int64_t key = std::min(node_id, neighbor_id) * 10000LL +
                             std::max(node_id, neighbor_id);
                if (seen.find(key) == seen.end()) {
                    seen.insert(key);
                    out_edges.emplace_back(id_to_idx[node_id], id_to_idx[neighbor_id]);
                }
            }
        }
    }

private:
    int add_node(const PointT& weight, float error = 0.0f) {
        if (addable_indices_.empty()) return -1;

        int node_id = addable_indices_.front();
        addable_indices_.pop_front();
        nodes[node_id] = NeuronNode<PointT>(node_id, weight, error, cycle);
        edges_per_node[node_id] = std::unordered_set<int>();

        if (grid_) {
            grid_->insert(&nodes[node_id]);
        }
        if (error_heap_) {
            error_heap_->insert(&nodes[node_id]);
        }

        return node_id;
    }

    void remove_node(int node_id) {
        auto it = edges_per_node.find(node_id);
        if (it != edges_per_node.end() && !it->second.empty()) return;

        if (grid_) {
            grid_->remove(&nodes[node_id]);
        }
        if (error_heap_) {
            error_heap_->remove(&nodes[node_id]);
        }

        edges_per_node.erase(node_id);
        nodes[node_id].id = -1;
        addable_indices_.push_back(node_id);
    }

    void add_edge(int node1, int node2) {
        // Per Algorithm 3, step 6: A_ν,μ ← 0
        // Edge age is set to 0 here, then incremented in the neighbor loop.
        if (edges(node1, node2) == 0) {
            // New edge
            edges_per_node[node1].insert(node2);
            edges_per_node[node2].insert(node1);
        }
        // Reset age to 0 (will be incremented to 1 in the neighbor loop)
        edges(node1, node2) = 0;
        edges(node2, node1) = 0;
    }

    void remove_edge(int node1, int node2) {
        edges_per_node[node1].erase(node2);
        edges_per_node[node2].erase(node1);
        edges(node1, node2) = 0;
        edges(node2, node1) = 0;
    }

    std::pair<int, int> find_two_nearest(const PointT& x) {
        if (grid_) {
            auto [n1, n2] = grid_->find_two_nearest(x);
            int id1 = n1 ? n1->id : -1;
            int id2 = n2 ? n2->id : -1;
            return {id1, id2};
        }

        // Linear search fallback
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

    void fix_error(int c, NeuronNode<PointT>* node) {
        if (node->cycle == c) return;
        int cycles_diff = c - node->cycle;
        float decay = std::pow(params.beta, static_cast<float>(params.lambda * cycles_diff));
        node->error *= decay;
        node->cycle = c;
    }

    void inc_error(NeuronNode<PointT>* node, float value) {
        fix_error(cycle, node);
        float decay_factor = beta_powers_[params.lambda - step];
        node->error = decay_factor * node->error + value;
        if (error_heap_) {
            error_heap_->update(node);
        }
    }

    void dec_error(NeuronNode<PointT>* node, float alpha) {
        fix_error(cycle, node);
        node->error *= alpha;
        if (error_heap_) {
            error_heap_->update(node);
        }
    }

    std::pair<int, int> largest_error() {
        NeuronNode<PointT>* q = nullptr;

        if (error_heap_) {
            q = error_heap_->top(cycle);
        } else {
            // Linear search fallback
            float max_err = -std::numeric_limits<float>::max();
            for (auto& node : nodes) {
                if (node.id == -1) continue;
                fix_error(cycle, &node);
                if (node.error > max_err) {
                    max_err = node.error;
                    q = &node;
                }
            }
        }

        if (!q || q->id == -1) return {-1, -1};

        // Find neighbor with largest error
        float max_err = -std::numeric_limits<float>::max();
        int f_id = -1;
        for (int neighbor_id : edges_per_node[q->id]) {
            auto& neighbor = nodes[neighbor_id];
            if (neighbor.id == -1) continue;
            fix_error(cycle, &neighbor);
            if (neighbor.error > max_err) {
                max_err = neighbor.error;
                f_id = neighbor_id;
            }
        }

        return {q->id, f_id};
    }

    void adapt(const PointT& sample) {
        step = n_learning % params.lambda;

        auto [s1_id, s2_id] = find_two_nearest(sample);
        if (s1_id == -1 || s2_id == -1) return;

        auto& winner = nodes[s1_id];

        // Increment winner error
        Scalar dist_sq = (sample - winner.weight).squaredNorm();
        inc_error(&winner, static_cast<float>(dist_sq));

        // Move winner toward sample
        winner.weight += params.eps_b * (sample - winner.weight);
        if (grid_) {
            grid_->update(&winner);
        }

        // Connect s1 and s2
        add_edge(s1_id, s2_id);

        // Update neighbors and age edges
        std::vector<int> edges_to_remove;
        for (int neighbor_id : edges_per_node[s1_id]) {
            edges(s1_id, neighbor_id)++;
            edges(neighbor_id, s1_id)++;

            if (edges(s1_id, neighbor_id) > params.max_age) {
                edges_to_remove.push_back(neighbor_id);
            } else {
                auto& neighbor = nodes[neighbor_id];
                neighbor.weight += params.eps_n * (sample - neighbor.weight);
                if (grid_) {
                    grid_->update(&neighbor);
                }
            }
        }

        for (int neighbor_id : edges_to_remove) {
            remove_edge(s1_id, neighbor_id);
            if (edges_per_node[neighbor_id].empty()) {
                remove_node(neighbor_id);
            }
        }
    }

    void insert_node() {
        if (addable_indices_.empty()) return;

        auto [q_id, f_id] = largest_error();
        if (q_id == -1 || f_id == -1) return;

        auto& q = nodes[q_id];
        auto& f = nodes[f_id];

        PointT new_weight = (q.weight + f.weight) * static_cast<Scalar>(0.5);

        fix_error(cycle, &q);
        fix_error(cycle, &f);

        dec_error(&q, params.alpha);
        dec_error(&f, params.alpha);

        float new_error = (q.error + f.error) * 0.5f;
        int new_id = add_node(new_weight, new_error);

        if (new_id == -1) return;

        remove_edge(q_id, f_id);
        add_edge(q_id, new_id);
        add_edge(f_id, new_id);
    }
};

// Common type aliases
using GNGEfficient2f = GNGEfficient<Eigen::Vector2f>;
using GNGEfficient3f = GNGEfficient<Eigen::Vector3f>;
using GNGEfficient2d = GNGEfficient<Eigen::Vector2d>;
using GNGEfficient3d = GNGEfficient<Eigen::Vector3d>;

}  // namespace gng_efficient
