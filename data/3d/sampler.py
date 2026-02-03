"""Sample points from 3D shapes and surfaces."""

import numpy as np


def sample_floor_and_wall(
    n_samples: int = 2000,
    floor_size: float = 0.8,
    wall_height: float = 0.6,
    wall_depth: float = 0.02,
    floor_depth: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Sample points from a floor and wall configuration (L-shape in 3D).

    The floor is on the XZ plane at y=0, and the wall is on the XY plane at z=0.
    Both surfaces meet at the edge where y=0 and z=0.

    Coordinates are in [0, 1] range.

    Args:
        n_samples: Total number of points to sample.
        floor_size: Size of the floor in x and z directions.
        wall_height: Height of the wall in y direction.
        wall_depth: Thickness of the wall (noise in z direction).
        floor_depth: Thickness of the floor (noise in y direction).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, 3) with (x, y, z) coordinates in [0, 1] range.
    """
    if seed is not None:
        np.random.seed(seed)

    # Offset to center the shape
    offset_x = (1.0 - floor_size) / 2
    offset_z = (1.0 - floor_size) / 2

    # Split samples between floor and wall based on area ratio
    floor_area = floor_size * floor_size
    wall_area = floor_size * wall_height
    total_area = floor_area + wall_area
    n_floor = int(n_samples * floor_area / total_area)
    n_wall = n_samples - n_floor

    all_points = []

    # Floor points (XZ plane, y near 0)
    floor_x = np.random.uniform(offset_x, offset_x + floor_size, n_floor)
    floor_y = np.random.uniform(0, floor_depth, n_floor)
    floor_z = np.random.uniform(offset_z, offset_z + floor_size, n_floor)
    floor_points = np.column_stack([floor_x, floor_y, floor_z])
    all_points.append(floor_points)

    # Wall points (XY plane, z near 0)
    wall_x = np.random.uniform(offset_x, offset_x + floor_size, n_wall)
    wall_y = np.random.uniform(0, wall_height, n_wall)
    wall_z = np.random.uniform(offset_z, offset_z + wall_depth, n_wall)
    wall_points = np.column_stack([wall_x, wall_y, wall_z])
    all_points.append(wall_points)

    return np.vstack(all_points)


def sample_triple_sphere(
    n_samples: int = 2000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample points from three spherical shells arranged in 3D space.

    Analogous to triple_ring in 2D. Coordinates are in [0, 1] range.

    Args:
        n_samples: Total number of points to sample.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, 3) with (x, y, z) coordinates in [0, 1] range.
    """
    if seed is not None:
        np.random.seed(seed)

    # Three spheres in a triangular arrangement
    # (center_x, center_y, center_z, inner_radius, outer_radius)
    spheres = [
        (0.50, 0.25, 0.50, 0.08, 0.12),  # top center
        (0.30, 0.65, 0.30, 0.08, 0.12),  # bottom left front
        (0.70, 0.65, 0.70, 0.08, 0.12),  # bottom right back
    ]

    samples_per_sphere = n_samples // 3
    all_points = []

    for cx, cy, cz, r_inner, r_outer in spheres:
        # Sample using rejection method for uniform distribution on spherical shell
        points = []
        while len(points) < samples_per_sphere:
            # Generate random points in unit cube centered at origin
            batch_size = samples_per_sphere * 2
            candidates = np.random.uniform(-1, 1, (batch_size, 3))

            # Calculate distances from origin
            distances = np.linalg.norm(candidates, axis=1)

            # Keep points within shell (normalized to [r_inner, r_outer])
            # First normalize to unit sphere, then scale to shell
            mask = distances > 0.01  # Avoid division by zero
            candidates = candidates[mask]
            distances = distances[mask]

            # Normalize to unit sphere then scale to random radius in shell
            radii = np.random.uniform(r_inner, r_outer, len(candidates))
            shell_points = (candidates / distances[:, np.newaxis]) * radii[:, np.newaxis]

            # Translate to center
            shell_points += np.array([cx, cy, cz])

            points.extend(shell_points.tolist())

        all_points.append(np.array(points[:samples_per_sphere]))

    return np.vstack(all_points)


def sample_sphere(
    n_samples: int = 2000,
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    radius: float = 0.4,
    seed: int | None = None,
) -> np.ndarray:
    """Sample points uniformly from a sphere surface.

    Args:
        n_samples: Number of points to sample.
        center: Center of the sphere (x, y, z).
        radius: Radius of the sphere.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, 3) with (x, y, z) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    # Use Marsaglia's method for uniform sampling on sphere
    points = []
    while len(points) < n_samples:
        # Generate random points in [-1, 1]^2
        batch_size = n_samples * 2
        x1 = np.random.uniform(-1, 1, batch_size)
        x2 = np.random.uniform(-1, 1, batch_size)

        # Keep points where x1^2 + x2^2 < 1
        s = x1**2 + x2**2
        mask = s < 1

        x1 = x1[mask]
        x2 = x2[mask]
        s = s[mask]

        # Convert to sphere coordinates
        sqrt_term = np.sqrt(1 - s)
        x = 2 * x1 * sqrt_term
        y = 2 * x2 * sqrt_term
        z = 1 - 2 * s

        # Scale and translate
        sphere_points = np.column_stack([
            center[0] + radius * x,
            center[1] + radius * y,
            center[2] + radius * z,
        ])

        points.extend(sphere_points.tolist())

    return np.array(points[:n_samples])


def sample_torus(
    n_samples: int = 2000,
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    major_radius: float = 0.3,
    minor_radius: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Sample points uniformly from a torus surface.

    Args:
        n_samples: Number of points to sample.
        center: Center of the torus (x, y, z).
        major_radius: Distance from center of torus to center of tube.
        minor_radius: Radius of the tube.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, 3) with (x, y, z) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    # For uniform sampling, we need to account for varying circumference
    # Use rejection sampling based on circumference

    points = []
    while len(points) < n_samples:
        batch_size = n_samples * 2

        # Angle around the torus (main circle)
        u = np.random.uniform(0, 2 * np.pi, batch_size)
        # Angle around the tube
        v = np.random.uniform(0, 2 * np.pi, batch_size)

        # Rejection sampling: probability proportional to (R + r*cos(v))
        # where R is major radius, r is minor radius
        acceptance_prob = (major_radius + minor_radius * np.cos(v)) / (
            major_radius + minor_radius
        )
        accept = np.random.random(batch_size) < acceptance_prob

        u = u[accept]
        v = v[accept]

        # Parametric equations for torus
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
        z = minor_radius * np.sin(v)

        # Translate to center
        torus_points = np.column_stack([
            center[0] + x,
            center[1] + y,
            center[2] + z,
        ])

        points.extend(torus_points.tolist())

    return np.array(points[:n_samples])


def sample_cylinder(
    n_samples: int = 2000,
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    radius: float = 0.2,
    height: float = 0.6,
    include_caps: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Sample points from a cylinder surface.

    Args:
        n_samples: Number of points to sample.
        center: Center of the cylinder (x, y, z).
        radius: Radius of the cylinder.
        height: Height of the cylinder (along z-axis).
        include_caps: Whether to include top and bottom caps.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, 3) with (x, y, z) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate areas
    side_area = 2 * np.pi * radius * height
    cap_area = np.pi * radius**2 if include_caps else 0
    total_area = side_area + 2 * cap_area

    # Number of samples for each part
    n_side = int(n_samples * side_area / total_area)
    n_cap = (n_samples - n_side) // 2 if include_caps else 0
    n_side = n_samples - 2 * n_cap  # Adjust for rounding

    all_points = []

    # Side surface
    theta = np.random.uniform(0, 2 * np.pi, n_side)
    z = np.random.uniform(-height / 2, height / 2, n_side)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    side_points = np.column_stack([
        center[0] + x,
        center[1] + y,
        center[2] + z,
    ])
    all_points.append(side_points)

    # Caps (if included)
    if include_caps and n_cap > 0:
        for z_offset in [-height / 2, height / 2]:
            # Sample uniformly on disk
            r = radius * np.sqrt(np.random.random(n_cap))
            theta = np.random.uniform(0, 2 * np.pi, n_cap)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            cap_points = np.column_stack([
                center[0] + x,
                center[1] + y,
                np.full(n_cap, center[2] + z_offset),
            ])
            all_points.append(cap_points)

    return np.vstack(all_points)


def save_points(points: np.ndarray, output_path: str) -> None:
    """Save points to file (.npy or .csv).

    Args:
        points: Array of shape (n, 3).
        output_path: Output file path.
    """
    from pathlib import Path

    path = Path(output_path)
    if path.suffix == ".npy":
        np.save(output_path, points)
    elif path.suffix == ".csv":
        np.savetxt(output_path, points, delimiter=",", header="x,y,z", comments="")
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    print(f"Saved {len(points)} points to {output_path}")
