"""Sample points from shape images or mathematical definitions."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def sample_triple_ring(
    n_samples: int = 1500,
    seed: int | None = None,
) -> np.ndarray:
    """Sample points from triple ring pattern (3 separate rings in triangle arrangement).

    This matches the layout of triple_ring.png used in visualization tests.
    Coordinates are in [0, 1] x [0, 1] range.

    Args:
        n_samples: Total number of points to sample (divided equally among 3 rings).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, 2) with (x, y) coordinates in [0, 1] range.
    """
    if seed is not None:
        np.random.seed(seed)

    # Three separate rings arranged in triangle pattern (matching triple_ring.png)
    # Each ring: (center_x, center_y, inner_radius, outer_radius)
    rings = [
        (0.50, 0.23, 0.06, 0.14),  # top center
        (0.27, 0.68, 0.06, 0.14),  # bottom left
        (0.73, 0.68, 0.06, 0.14),  # bottom right
    ]

    samples_per_ring = n_samples // 3
    all_points = []

    for cx, cy, r_inner, r_outer in rings:
        # Sample angles uniformly
        angles = np.random.uniform(0, 2 * np.pi, samples_per_ring)
        # Sample radii uniformly in the annulus
        radii = np.random.uniform(r_inner, r_outer, samples_per_ring)
        # Convert to Cartesian coordinates
        x = cx + radii * np.cos(angles)
        y = cy + radii * np.sin(angles)
        all_points.append(np.column_stack([x, y]))

    return np.vstack(all_points)


def sample_from_image(
    image_path: str,
    n_samples: int = 1000,
    target_color: tuple = (135, 206, 235),  # Sky blue #87CEEB
    color_tolerance: int = 10,
    seed: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Sample random points from colored regions in an image.

    Args:
        image_path: Path to the input image.
        n_samples: Number of points to sample.
        target_color: RGB color to sample from.
        color_tolerance: Tolerance for color matching.
        seed: Random seed for reproducibility.
        normalize: If True, normalize coordinates to [0, 1].

    Returns:
        Array of shape (n_samples, 2) with (x, y) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Find pixels matching target color
    color_diff = np.abs(img_array.astype(int) - np.array(target_color))
    mask = np.all(color_diff <= color_tolerance, axis=2)

    # Get coordinates of matching pixels
    y_coords, x_coords = np.where(mask)

    if len(x_coords) == 0:
        raise ValueError(f"No pixels found matching color {target_color}")

    # Random sampling
    n_available = len(x_coords)
    if n_samples > n_available:
        print(f"Warning: Requested {n_samples} samples but only {n_available} available")
        n_samples = n_available

    indices = np.random.choice(n_available, size=n_samples, replace=False)
    sampled_x = x_coords[indices]
    sampled_y = y_coords[indices]

    # Stack as (x, y) pairs
    points = np.column_stack([sampled_x, sampled_y]).astype(float)

    if normalize:
        height, width = img_array.shape[:2]
        points[:, 0] /= width
        points[:, 1] /= height

    return points


def save_points(points: np.ndarray, output_path: str) -> None:
    """Save points to file (.npy or .csv).

    Args:
        points: Array of shape (n, 2).
        output_path: Output file path.
    """
    path = Path(output_path)
    if path.suffix == ".npy":
        np.save(output_path, points)
    elif path.suffix == ".csv":
        np.savetxt(output_path, points, delimiter=",", header="x,y", comments="")
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    print(f"Saved {len(points)} points to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample points from shape images")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("-n", "--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("-o", "--output", type=str, default="points.npy", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-normalize", action="store_true", help="Keep pixel coordinates (don't normalize)"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="87CEEB",
        help="Target color in hex (default: 87CEEB sky blue)",
    )

    args = parser.parse_args()

    # Parse hex color
    color_hex = args.color.lstrip("#")
    target_color = tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4))

    points = sample_from_image(
        args.image,
        n_samples=args.n_samples,
        target_color=target_color,
        seed=args.seed,
        normalize=not args.no_normalize,
    )

    save_points(points, args.output)


if __name__ == "__main__":
    main()
