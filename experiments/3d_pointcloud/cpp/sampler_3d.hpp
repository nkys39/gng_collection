/**
 * @file sampler_3d.hpp
 * @brief 3D point sampling functions for testing GNG algorithms
 */

#pragma once

#include <random>
#include <vector>

#include <Eigen/Core>

namespace sampler_3d {

/**
 * @brief Parameters for floor and wall sampling
 */
struct FloorWallParams {
    float floor_size = 0.8f;
    float wall_height = 0.6f;
    float wall_depth = 0.02f;
    float floor_depth = 0.02f;
};

/**
 * @brief Sample points from a floor and wall configuration (L-shape in 3D)
 *
 * The floor is on the XZ plane at y=0, and the wall is on the XY plane at z=offset.
 * Both surfaces meet at the edge where y=0 and z=offset.
 *
 * @param n_samples Total number of points to sample
 * @param params Floor and wall parameters
 * @param seed Random seed
 * @return Vector of 3D points in [0, 1] range
 */
inline std::vector<Eigen::Vector3f> sample_floor_and_wall(
    int n_samples,
    const FloorWallParams& params = FloorWallParams(),
    unsigned int seed = 42
) {
    std::mt19937 rng(seed);

    float offset_x = (1.0f - params.floor_size) / 2.0f;
    float offset_z = (1.0f - params.floor_size) / 2.0f;

    // Split samples based on area ratio
    float floor_area = params.floor_size * params.floor_size;
    float wall_area = params.floor_size * params.wall_height;
    float total_area = floor_area + wall_area;
    int n_floor = static_cast<int>(n_samples * floor_area / total_area);
    int n_wall = n_samples - n_floor;

    std::vector<Eigen::Vector3f> points;
    points.reserve(n_samples);

    // Floor points (XZ plane, y near 0)
    std::uniform_real_distribution<float> floor_x_dist(offset_x, offset_x + params.floor_size);
    std::uniform_real_distribution<float> floor_y_dist(0.0f, params.floor_depth);
    std::uniform_real_distribution<float> floor_z_dist(offset_z, offset_z + params.floor_size);

    for (int i = 0; i < n_floor; ++i) {
        points.emplace_back(floor_x_dist(rng), floor_y_dist(rng), floor_z_dist(rng));
    }

    // Wall points (XY plane, z near offset_z)
    std::uniform_real_distribution<float> wall_x_dist(offset_x, offset_x + params.floor_size);
    std::uniform_real_distribution<float> wall_y_dist(0.0f, params.wall_height);
    std::uniform_real_distribution<float> wall_z_dist(offset_z, offset_z + params.wall_depth);

    for (int i = 0; i < n_wall; ++i) {
        points.emplace_back(wall_x_dist(rng), wall_y_dist(rng), wall_z_dist(rng));
    }

    return points;
}

/**
 * @brief Sample points from three spherical shells arranged in 3D space
 *
 * Analogous to triple_ring in 2D. Coordinates are in [0, 1] range.
 *
 * @param n_samples Total number of points to sample
 * @param seed Random seed
 * @return Vector of 3D points in [0, 1] range
 */
inline std::vector<Eigen::Vector3f> sample_triple_sphere(
    int n_samples,
    unsigned int seed = 42
) {
    std::mt19937 rng(seed);

    // Three spheres in a triangular arrangement
    // (center_x, center_y, center_z, inner_radius, outer_radius)
    struct Sphere {
        float cx, cy, cz, r_inner, r_outer;
    };
    std::vector<Sphere> spheres = {
        {0.50f, 0.25f, 0.50f, 0.08f, 0.12f},  // top center
        {0.30f, 0.65f, 0.30f, 0.08f, 0.12f},  // bottom left front
        {0.70f, 0.65f, 0.70f, 0.08f, 0.12f},  // bottom right back
    };

    int samples_per_sphere = n_samples / 3;
    std::vector<Eigen::Vector3f> points;
    points.reserve(n_samples);

    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    for (const auto& sphere : spheres) {
        int count = 0;
        while (count < samples_per_sphere) {
            // Generate random point in unit cube
            Eigen::Vector3f p(uniform(rng), uniform(rng), uniform(rng));
            float dist = p.norm();

            if (dist < 0.01f) continue;  // Avoid division by zero

            // Normalize to unit sphere and scale to random radius in shell
            std::uniform_real_distribution<float> radius_dist(sphere.r_inner, sphere.r_outer);
            float radius = radius_dist(rng);
            p = (p / dist) * radius;

            // Translate to center
            p += Eigen::Vector3f(sphere.cx, sphere.cy, sphere.cz);

            points.push_back(p);
            ++count;
        }
    }

    return points;
}

/**
 * @brief Sample points uniformly on a sphere surface
 *
 * @param n_samples Number of points to sample
 * @param radius Sphere radius
 * @param center_x X coordinate of center
 * @param center_y Y coordinate of center
 * @param center_z Z coordinate of center
 * @param seed Random seed
 * @return Vector of 3D points
 */
inline std::vector<Eigen::Vector3f> sample_sphere(
    int n_samples,
    float radius = 0.35f,
    float center_x = 0.5f,
    float center_y = 0.5f,
    float center_z = 0.5f,
    unsigned int seed = 42
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    std::vector<Eigen::Vector3f> points;
    points.reserve(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        // Use spherical coordinates with uniform distribution on sphere
        float theta = 2.0f * M_PI * uniform(rng);  // azimuthal angle
        float phi = std::acos(2.0f * uniform(rng) - 1.0f);  // polar angle

        float x = center_x + radius * std::sin(phi) * std::cos(theta);
        float y = center_y + radius * std::sin(phi) * std::sin(theta);
        float z = center_z + radius * std::cos(phi);

        points.emplace_back(x, y, z);
    }

    return points;
}

/**
 * @brief Sample points uniformly on a torus surface
 *
 * @param n_samples Number of points to sample
 * @param major_radius Distance from center of tube to center of torus
 * @param minor_radius Radius of the tube
 * @param center_x X coordinate of center
 * @param center_y Y coordinate of center
 * @param center_z Z coordinate of center
 * @param seed Random seed
 * @return Vector of 3D points
 */
inline std::vector<Eigen::Vector3f> sample_torus(
    int n_samples,
    float major_radius = 0.3f,
    float minor_radius = 0.12f,
    float center_x = 0.5f,
    float center_y = 0.5f,
    float center_z = 0.5f,
    unsigned int seed = 42
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    std::vector<Eigen::Vector3f> points;
    points.reserve(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        // Parametric torus: two angles
        float u = 2.0f * M_PI * uniform(rng);  // angle around torus
        float v = 2.0f * M_PI * uniform(rng);  // angle around tube

        // Torus parameterization
        float x = center_x + (major_radius + minor_radius * std::cos(v)) * std::cos(u);
        float y = center_y + (major_radius + minor_radius * std::cos(v)) * std::sin(u);
        float z = center_z + minor_radius * std::sin(v);

        points.emplace_back(x, y, z);
    }

    return points;
}

}  // namespace sampler_3d
