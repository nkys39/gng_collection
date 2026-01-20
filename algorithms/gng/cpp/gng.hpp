/**
 * @file gng.hpp
 * @brief Growing Neural Gas (GNG) implementation
 *
 * Reference:
 *   Fritzke, B. (1995). A Growing Neural Gas Network Learns Topologies.
 *   Advances in Neural Information Processing Systems 7 (NIPS 1994).
 */

#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <gng/core/base.hpp>
#include <gng/core/utils.hpp>

namespace gng {

/**
 * @brief Growing Neural Gas algorithm.
 *
 * @tparam Dim Dimensionality of input data (-1 for dynamic)
 */
template <int Dim = Eigen::Dynamic>
class GNG : public BaseGNG<Dim> {
public:
    using typename BaseGNG<Dim>::Scalar;
    using typename BaseGNG<Dim>::Vector;
    using typename BaseGNG<Dim>::Matrix;
    using typename BaseGNG<Dim>::Edge;
    using typename BaseGNG<Dim>::EdgeMap;

    /**
     * @brief Construct a GNG model.
     *
     * @param lambda Node insertion interval
     * @param eps_b Learning rate for winner node
     * @param eps_n Learning rate for neighbor nodes
     * @param alpha Error reduction factor when inserting a node
     * @param beta Global error decay factor
     * @param max_age Maximum edge age
     * @param max_nodes Maximum number of nodes (0 for unlimited)
     */
    GNG(int lambda = 100, double eps_b = 0.2, double eps_n = 0.006,
        double alpha = 0.5, double beta = 0.0005, int max_age = 50,
        int max_nodes = 0)
        : lambda_(lambda),
          eps_b_(eps_b),
          eps_n_(eps_n),
          alpha_(alpha),
          beta_(beta),
          max_age_(max_age),
          max_nodes_(max_nodes),
          step_(0) {}

    /**
     * @brief Fit the model to data.
     */
    void fit(const Matrix& data, int epochs = 1) override {
        if (this->nodes_.rows() == 0) {
            initialize(data);
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Shuffle indices
            std::vector<int> indices(data.rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);

            for (int idx : indices) {
                partial_fit(data.row(idx).transpose());
            }
        }

        this->is_fitted_ = true;
    }

    /**
     * @brief Incrementally fit with a single sample.
     */
    void partial_fit(const Vector& x) override {
        // Find two nearest nodes
        auto [indices, distances] = find_nearest_nodes(x, this->nodes_, 2);
        int s1 = indices[0];
        int s2 = indices[1];

        // Update error of winner
        errors_[s1] += distances[0] * distances[0];

        // Move winner toward input
        this->nodes_.row(s1) += eps_b_ * (x - this->nodes_.row(s1).transpose()).transpose();

        // Move neighbors toward input
        for (int neighbor : get_neighbors(s1)) {
            this->nodes_.row(neighbor) +=
                eps_n_ * (x - this->nodes_.row(neighbor).transpose()).transpose();
        }

        // Update edge between s1 and s2
        Edge edge = make_edge(s1, s2);
        this->edges_[edge] = 0;

        // Increment age of edges from s1 and remove old ones
        std::vector<Edge> to_remove;
        for (auto& [e, age] : this->edges_) {
            if (e.first == s1 || e.second == s1) {
                age++;
                if (age > max_age_) {
                    to_remove.push_back(e);
                }
            }
        }
        for (const auto& e : to_remove) {
            this->edges_.erase(e);
        }

        // Remove isolated nodes
        remove_isolated_nodes();

        // Insert new node periodically
        step_++;
        if (step_ % lambda_ == 0) {
            if (max_nodes_ == 0 || this->n_nodes() < max_nodes_) {
                insert_node();
            }
        }

        // Decay all errors
        for (auto& err : errors_) {
            err *= (1.0 - beta_);
        }
    }

private:
    int lambda_;
    double eps_b_;
    double eps_n_;
    double alpha_;
    double beta_;
    int max_age_;
    int max_nodes_;
    int step_;
    std::vector<double> errors_;

    void initialize(const Matrix& data) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, static_cast<int>(data.rows()) - 1);

        int idx1 = dis(gen);
        int idx2 = dis(gen);
        while (idx2 == idx1) {
            idx2 = dis(gen);
        }

        this->nodes_.resize(2, data.cols());
        this->nodes_.row(0) = data.row(idx1);
        this->nodes_.row(1) = data.row(idx2);
        errors_ = {0.0, 0.0};
        this->edges_[make_edge(0, 1)] = 0;
    }

    Edge make_edge(int i, int j) const {
        return {std::min(i, j), std::max(i, j)};
    }

    std::vector<int> get_neighbors(int node) const {
        std::vector<int> neighbors;
        for (const auto& [e, age] : this->edges_) {
            if (e.first == node) {
                neighbors.push_back(e.second);
            } else if (e.second == node) {
                neighbors.push_back(e.first);
            }
        }
        return neighbors;
    }

    void remove_isolated_nodes() {
        std::set<int> connected;
        for (const auto& [e, age] : this->edges_) {
            connected.insert(e.first);
            connected.insert(e.second);
        }

        std::vector<int> isolated;
        for (int i = 0; i < this->n_nodes(); ++i) {
            if (connected.find(i) == connected.end()) {
                isolated.push_back(i);
            }
        }

        // Remove from end to preserve indices
        for (auto it = isolated.rbegin(); it != isolated.rend(); ++it) {
            remove_node(*it);
        }
    }

    void remove_node(int idx) {
        // Remove row from nodes matrix
        Matrix new_nodes(this->nodes_.rows() - 1, this->nodes_.cols());
        new_nodes.topRows(idx) = this->nodes_.topRows(idx);
        new_nodes.bottomRows(this->nodes_.rows() - idx - 1) =
            this->nodes_.bottomRows(this->nodes_.rows() - idx - 1);
        this->nodes_ = new_nodes;

        // Remove error
        errors_.erase(errors_.begin() + idx);

        // Update edge indices
        EdgeMap new_edges;
        for (const auto& [e, age] : this->edges_) {
            int new_i = e.first < idx ? e.first : e.first - 1;
            int new_j = e.second < idx ? e.second : e.second - 1;
            if (new_i >= 0 && new_j >= 0) {
                new_edges[make_edge(new_i, new_j)] = age;
            }
        }
        this->edges_ = new_edges;
    }

    void insert_node() {
        if (this->n_nodes() < 2) return;

        // Find node with maximum error
        int q = static_cast<int>(
            std::max_element(errors_.begin(), errors_.end()) - errors_.begin());

        // Find neighbor with maximum error
        auto neighbors = get_neighbors(q);
        if (neighbors.empty()) return;

        int f = *std::max_element(neighbors.begin(), neighbors.end(),
                                   [this](int a, int b) { return errors_[a] < errors_[b]; });

        // Create new node between q and f
        Vector new_node = 0.5 * (this->nodes_.row(q) + this->nodes_.row(f)).transpose();
        this->nodes_.conservativeResize(this->nodes_.rows() + 1, Eigen::NoChange);
        this->nodes_.row(this->nodes_.rows() - 1) = new_node.transpose();
        errors_.push_back(0.0);
        int r = this->n_nodes() - 1;

        // Update edges
        Edge edge_qf = make_edge(q, f);
        this->edges_.erase(edge_qf);
        this->edges_[make_edge(q, r)] = 0;
        this->edges_[make_edge(f, r)] = 0;

        // Update errors
        errors_[q] *= alpha_;
        errors_[f] *= alpha_;
        errors_[r] = errors_[q];
    }
};

}  // namespace gng
