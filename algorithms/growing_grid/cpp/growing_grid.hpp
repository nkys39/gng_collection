/**
 * @file growing_grid.hpp
 * @brief Growing Grid (GG) algorithm implementation
 *
 * Based on:
 *   - Fritzke, B. (1995). "Growing Grid - a self-organizing network with constant
 *     neighborhood range and adaptation strength"
 *   - demogng.de reference implementation
 *
 * Growing Grid combines the structured topology of SOM with the ability to grow.
 * It starts with a small grid and adds rows/columns where error is highest.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace growing_grid {

/**
 * @brief Growing Grid hyperparameters.
 */
struct GGParams {
    int initial_height = 2;     // Initial grid height
    int initial_width = 2;      // Initial grid width
    int max_nodes = 100;        // Maximum number of nodes
    int lambda = 100;           // Growth interval
    float eps_b = 0.1f;         // Winner learning rate
    float eps_n = 0.01f;        // Neighbor learning rate
    float sigma = 1.5f;         // Neighborhood radius (constant)
    float tau = 0.005f;         // Error decay rate
};

/**
 * @brief Growing Grid algorithm.
 *
 * Growing Grid is a self-organizing network that starts with a small
 * rectangular grid and grows by inserting rows or columns in regions
 * with high accumulated error.
 *
 * Unlike SOM, the neighborhood range and learning rates are constant,
 * not decaying over time.
 *
 * @tparam PointT Point type (Eigen vector, e.g., Eigen::Vector2f)
 */
template <typename PointT>
class GrowingGrid {
public:
    using Scalar = typename PointT::Scalar;
    using Callback = std::function<void(const GrowingGrid&, int)>;

    GGParams params;
    std::vector<std::vector<PointT>> weights;  // Grid of weight vectors [height][width]
    std::vector<std::vector<float>> errors;    // Error accumulator
    int height;
    int width;
    int n_learning = 0;

private:
    std::mt19937 rng_;
    int n_trial_ = 0;

public:
    /**
     * @brief Construct Growing Grid with given parameters.
     *
     * @param params Growing Grid hyperparameters
     * @param seed Random seed (0 for random)
     */
    explicit GrowingGrid(const GGParams& params = GGParams(), unsigned int seed = 0)
        : params(params),
          height(params.initial_height),
          width(params.initial_width),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        // Initialize grid
        weights.resize(height);
        errors.resize(height);

        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (int i = 0; i < height; ++i) {
            weights[i].resize(width);
            errors[i].resize(width, 0.0f);
            for (int j = 0; j < width; ++j) {
                for (int k = 0; k < PointT::RowsAtCompileTime; ++k) {
                    weights[i][j](k) = dist(rng_);
                }
            }
        }
    }

    /**
     * @brief Initialize weights from data range.
     *
     * @param data Training data to determine initialization range
     */
    void init_from_data(const std::vector<PointT>& data) {
        if (data.empty()) return;

        // Find data bounds
        PointT min_val = data[0];
        PointT max_val = data[0];
        for (const auto& p : data) {
            for (int k = 0; k < PointT::RowsAtCompileTime; ++k) {
                min_val(k) = std::min(min_val(k), p(k));
                max_val(k) = std::max(max_val(k), p(k));
            }
        }

        // Initialize randomly within data bounds
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int k = 0; k < PointT::RowsAtCompileTime; ++k) {
                    std::uniform_real_distribution<Scalar> dist(min_val(k), max_val(k));
                    weights[i][j](k) = dist(rng_);
                }
            }
        }
    }

    /**
     * @brief Train on data for multiple iterations.
     *
     * @param data Vector of training samples
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
     * @brief Get number of nodes.
     */
    int num_nodes() const {
        return height * width;
    }

    /**
     * @brief Get number of grid edges.
     */
    int num_edges() const {
        int h_edges = height * (width - 1);
        int v_edges = (height - 1) * width;
        return h_edges + v_edges;
    }

    /**
     * @brief Get node positions.
     *
     * @return Vector of node positions (flattened grid, row-major)
     */
    std::vector<PointT> get_nodes() const {
        std::vector<PointT> result;
        result.reserve(num_nodes());
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                result.push_back(weights[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Get edges based on grid topology.
     *
     * @return Vector of (i, j) pairs
     */
    std::vector<std::pair<int, int>> get_edges() const {
        std::vector<std::pair<int, int>> edges;
        edges.reserve(num_edges());

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int idx = i * width + j;

                // Right neighbor
                if (j < width - 1) {
                    edges.emplace_back(idx, idx + 1);
                }

                // Bottom neighbor
                if (i < height - 1) {
                    edges.emplace_back(idx, idx + width);
                }
            }
        }

        return edges;
    }

    /**
     * @brief Get graph for visualization.
     *
     * @param out_nodes Output: node positions array
     * @param out_edges Output: edges as (i, j) pairs
     */
    void get_graph(std::vector<PointT>& out_nodes,
                   std::vector<std::pair<int, int>>& out_edges) const {
        out_nodes = get_nodes();
        out_edges = get_edges();
    }

    /**
     * @brief Get error values for all nodes.
     */
    std::vector<float> get_node_errors() const {
        std::vector<float> result;
        result.reserve(num_nodes());
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                result.push_back(errors[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Get current grid dimensions.
     */
    std::pair<int, int> get_grid_size() const {
        return {height, width};
    }

private:
    /**
     * @brief Find the Best Matching Unit (BMU) for input x.
     *
     * @return Pair of (row, col) indices of the BMU
     */
    std::pair<int, int> get_bmu(const PointT& x) const {
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        int bmu_i = 0, bmu_j = 0;

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                Scalar dist = (x - weights[i][j]).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    bmu_i = i;
                    bmu_j = j;
                }
            }
        }

        return {bmu_i, bmu_j};
    }

    /**
     * @brief Compute neighborhood function value.
     *
     * Uses Manhattan distance and constant sigma (unlike SOM).
     */
    Scalar get_neighborhood_value(int bmu_i, int bmu_j, int i, int j) const {
        Scalar grid_dist = static_cast<Scalar>(std::abs(i - bmu_i) + std::abs(j - bmu_j));
        return std::exp(-(grid_dist * grid_dist) / (2 * params.sigma * params.sigma));
    }

    /**
     * @brief Grow the grid by adding a row or column.
     *
     * @return true if grid was grown, false if max nodes reached
     */
    bool grow_grid() {
        // Check if we can grow
        int new_size_row = (height + 1) * width;
        int new_size_col = height * (width + 1);
        if (std::min(new_size_row, new_size_col) > params.max_nodes) {
            return false;
        }

        // Find boundary node with maximum error
        float max_error = -1.0f;
        bool grow_row = true;
        int grow_pos = 0;

        // Check top row
        for (int j = 0; j < width; ++j) {
            if (errors[0][j] > max_error && new_size_row <= params.max_nodes) {
                max_error = errors[0][j];
                grow_row = true;
                grow_pos = 0;  // Insert at top
            }
        }

        // Check bottom row
        for (int j = 0; j < width; ++j) {
            if (errors[height - 1][j] > max_error && new_size_row <= params.max_nodes) {
                max_error = errors[height - 1][j];
                grow_row = true;
                grow_pos = height;  // Insert at bottom
            }
        }

        // Check left column
        for (int i = 0; i < height; ++i) {
            if (errors[i][0] > max_error && new_size_col <= params.max_nodes) {
                max_error = errors[i][0];
                grow_row = false;
                grow_pos = 0;  // Insert at left
            }
        }

        // Check right column
        for (int i = 0; i < height; ++i) {
            if (errors[i][width - 1] > max_error && new_size_col <= params.max_nodes) {
                max_error = errors[i][width - 1];
                grow_row = false;
                grow_pos = width;  // Insert at right
            }
        }

        std::normal_distribution<Scalar> noise(0, 0.01);

        if (grow_row) {
            // Add a row
            std::vector<PointT> new_row(width);
            std::vector<float> new_errors(width, 0.0f);

            int ref_row = (grow_pos == 0) ? 0 : height - 1;
            for (int j = 0; j < width; ++j) {
                new_row[j] = weights[ref_row][j];
                for (int k = 0; k < PointT::RowsAtCompileTime; ++k) {
                    new_row[j](k) += noise(rng_);
                }
            }

            if (grow_pos == 0) {
                weights.insert(weights.begin(), new_row);
                errors.insert(errors.begin(), new_errors);
            } else {
                weights.push_back(new_row);
                errors.push_back(new_errors);
            }
            height++;
        } else {
            // Add a column
            int ref_col = (grow_pos == 0) ? 0 : width - 1;
            for (int i = 0; i < height; ++i) {
                PointT new_weight = weights[i][ref_col];
                for (int k = 0; k < PointT::RowsAtCompileTime; ++k) {
                    new_weight(k) += noise(rng_);
                }

                if (grow_pos == 0) {
                    weights[i].insert(weights[i].begin(), new_weight);
                    errors[i].insert(errors[i].begin(), 0.0f);
                } else {
                    weights[i].push_back(new_weight);
                    errors[i].push_back(0.0f);
                }
            }
            width++;
        }

        return true;
    }

    /**
     * @brief Single training update.
     */
    void one_train_update(const PointT& sample) {
        // Decay all errors
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                errors[i][j] *= (1.0f - params.tau);
            }
        }

        // Find BMU
        auto [bmu_i, bmu_j] = get_bmu(sample);

        // Accumulate error at BMU
        Scalar dist_sq = (sample - weights[bmu_i][bmu_j]).squaredNorm();
        errors[bmu_i][bmu_j] += static_cast<float>(dist_sq);

        // Update weights
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                Scalar lr;
                if (i == bmu_i && j == bmu_j) {
                    lr = params.eps_b;
                } else {
                    Scalar h = get_neighborhood_value(bmu_i, bmu_j, i, j);
                    lr = params.eps_n * h;
                }
                weights[i][j] += lr * (sample - weights[i][j]);
            }
        }

        // Periodically grow grid
        n_trial_++;
        if (n_trial_ >= params.lambda) {
            n_trial_ = 0;
            grow_grid();
        }

        n_learning++;
    }
};

// Common type aliases
using GG2f = GrowingGrid<Eigen::Vector2f>;
using GG3f = GrowingGrid<Eigen::Vector3f>;
using GG2d = GrowingGrid<Eigen::Vector2d>;
using GG3d = GrowingGrid<Eigen::Vector3d>;

}  // namespace growing_grid
