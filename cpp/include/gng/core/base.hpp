#pragma once

#include <Eigen/Dense>
#include <map>
#include <utility>
#include <vector>

namespace gng {

/**
 * @brief Abstract base class for all GNG algorithm variants.
 *
 * This class defines the common interface that all GNG implementations
 * should follow to ensure consistency across different algorithms.
 *
 * @tparam Dim Dimensionality of input data (-1 for dynamic)
 */
template <int Dim = Eigen::Dynamic>
class BaseGNG {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Dim, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Dim>;
    using Edge = std::pair<int, int>;
    using EdgeMap = std::map<Edge, int>;

    BaseGNG() = default;
    virtual ~BaseGNG() = default;

    /**
     * @brief Fit the model to the data.
     *
     * @param data Input data matrix (n_samples x n_features)
     * @param epochs Number of passes through the data
     */
    virtual void fit(const Matrix& data, int epochs = 1) = 0;

    /**
     * @brief Incrementally fit the model with a single sample.
     *
     * @param x Single input sample
     */
    virtual void partial_fit(const Vector& x) = 0;

    /**
     * @brief Get current node positions.
     *
     * @return Matrix of node positions (n_nodes x n_features)
     */
    Matrix get_nodes() const { return nodes_; }

    /**
     * @brief Get current edges.
     *
     * @return Vector of (node_i, node_j) pairs
     */
    std::vector<Edge> get_edges() const {
        std::vector<Edge> edges;
        edges.reserve(edges_.size());
        for (const auto& [edge, age] : edges_) {
            edges.push_back(edge);
        }
        return edges;
    }

    /**
     * @brief Get number of nodes.
     */
    int n_nodes() const { return static_cast<int>(nodes_.rows()); }

    /**
     * @brief Get number of edges.
     */
    int n_edges() const { return static_cast<int>(edges_.size()); }

protected:
    Matrix nodes_;
    EdgeMap edges_;
    bool is_fitted_ = false;
};

}  // namespace gng
