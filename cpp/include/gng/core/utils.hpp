#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace gng {

/**
 * @brief Calculate Euclidean distance between two points.
 */
template <typename Derived1, typename Derived2>
double euclidean_distance(const Eigen::MatrixBase<Derived1>& a,
                          const Eigen::MatrixBase<Derived2>& b) {
    return (a - b).norm();
}

/**
 * @brief Calculate squared Euclidean distance between two points.
 *
 * More efficient when only comparing distances (avoids sqrt).
 */
template <typename Derived1, typename Derived2>
double euclidean_distance_squared(const Eigen::MatrixBase<Derived1>& a,
                                  const Eigen::MatrixBase<Derived2>& b) {
    return (a - b).squaredNorm();
}

/**
 * @brief Find k nearest nodes to a given point.
 *
 * @param x Query point
 * @param nodes Matrix of node positions (n_nodes x n_features)
 * @param k Number of nearest nodes to find
 * @return Pair of (indices, distances) for k nearest nodes
 */
template <typename VectorType, typename MatrixType>
std::pair<std::vector<int>, std::vector<double>> find_nearest_nodes(
    const VectorType& x, const MatrixType& nodes, int k = 2) {
    const int n_nodes = static_cast<int>(nodes.rows());
    k = std::min(k, n_nodes);

    // Calculate all distances
    std::vector<std::pair<double, int>> dist_idx(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        dist_idx[i] = {(nodes.row(i).transpose() - x).norm(), i};
    }

    // Partial sort to find k smallest
    std::partial_sort(dist_idx.begin(), dist_idx.begin() + k, dist_idx.end());

    // Extract results
    std::vector<int> indices(k);
    std::vector<double> distances(k);
    for (int i = 0; i < k; ++i) {
        indices[i] = dist_idx[i].second;
        distances[i] = dist_idx[i].first;
    }

    return {indices, distances};
}

/**
 * @brief Calculate mean quantization error for a dataset.
 *
 * @param data Input data matrix (n_samples x n_features)
 * @param nodes Node positions matrix (n_nodes x n_features)
 * @return Mean distance from each point to its nearest node
 */
template <typename MatrixType>
double calculate_quantization_error(const MatrixType& data, const MatrixType& nodes) {
    double total_error = 0.0;
    const int n_samples = static_cast<int>(data.rows());

    for (int i = 0; i < n_samples; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        for (int j = 0; j < nodes.rows(); ++j) {
            double dist = (data.row(i) - nodes.row(j)).norm();
            min_dist = std::min(min_dist, dist);
        }
        total_error += min_dist;
    }

    return total_error / n_samples;
}

}  // namespace gng
