"""Animate GNG nodes rotating along circular orbit.

After training GNG on a ring shape, this script rotates the learned
nodes along a circular path to create a smooth animation.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "gng" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import GrowingNeuralGas, GNGParams
from sampler import sample_from_image


def rotate_points(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    """Rotate points around a center by given angle.

    Args:
        points: Array of shape (n, 2) with (x, y) coordinates.
        angle: Rotation angle in radians.
        center: Center point (x, y) for rotation.

    Returns:
        Rotated points array.
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Translate to origin, rotate, translate back
    centered = points - center
    rotated = centered @ rotation_matrix.T
    return rotated + center


def create_rotation_frame(
    ax,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    center: np.ndarray,
    frame_num: int,
    total_frames: int,
    bg_image: np.ndarray | None = None,
) -> None:
    """Create a single frame for rotation animation."""
    ax.clear()

    if bg_image is not None:
        ax.imshow(bg_image, extent=[0, 1, 1, 0], alpha=0.3)

    # Plot edges
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "r-",
            linewidth=2,
            alpha=0.8,
        )

    # Plot nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=60, zorder=5, edgecolors="darkred", linewidths=1)

    # Plot center
    ax.scatter([center[0]], [center[1]], c="blue", s=30, marker="+", zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(f"GNG Rotation - Frame {frame_num}/{total_frames}")
    ax.axis("off")


def run_rotation_animation(
    image_path: str = "single_ring.png",
    n_samples: int = 1500,
    train_iterations: int = 5000,
    rotation_frames: int = 60,
    rotations: float = 1.0,
    output_gif: str = "gng_single_ring_rotation.gif",
    seed: int = 42,
) -> None:
    """Train GNG and create rotation animation.

    Args:
        image_path: Path to shape image.
        n_samples: Number of points to sample for training.
        train_iterations: Number of GNG training iterations.
        rotation_frames: Number of frames in rotation animation.
        rotations: Number of full rotations (1.0 = 360 degrees).
        output_gif: Output GIF path.
        seed: Random seed.
    """
    np.random.seed(seed)

    # Generate shape image if not exists
    if not Path(image_path).exists():
        print(f"Generating shape image: {image_path}")
        shapes_dir = Path(__file__).parents[2] / "data" / "2d" / "shapes"
        sys.path.insert(0, str(shapes_dir))
        from generate_shape import generate_single_ring

        generate_single_ring(image_path)

    # Load background image
    bg_image = np.array(Image.open(image_path).convert("RGB"))

    # Sample points from image
    print(f"Sampling {n_samples} points from {image_path}...")
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Setup and train GNG
    params = GNGParams(
        max_nodes=100,
        lambda_=100,
        eps_b=0.08,
        eps_n=0.008,
        alpha=0.5,
        beta=0.005,
        max_age=100,
    )
    gng = GrowingNeuralGas(n_dim=2, params=params, seed=seed)

    print(f"Training GNG for {train_iterations} iterations...")
    gng.train(points, n_iterations=train_iterations)

    # Get learned graph
    nodes, edges = gng.get_graph()
    print(f"Trained: {len(nodes)} nodes, {len(edges)} edges")

    # Calculate center (centroid of nodes)
    center = np.array([0.5, 0.5])  # Center of normalized image

    # Create rotation animation
    print(f"Creating rotation animation ({rotation_frames} frames, {rotations} rotations)...")
    frames = []
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")

    for frame in range(rotation_frames):
        angle = (frame / rotation_frames) * 2 * np.pi * rotations
        rotated_nodes = rotate_points(nodes, angle, center)

        create_rotation_frame(ax, rotated_nodes, edges, center, frame + 1, rotation_frames, bg_image)
        fig.canvas.draw()

        # Convert to PIL Image
        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        )
        frames.append(img.convert("RGB"))

        if (frame + 1) % 10 == 0:
            print(f"  Frame {frame + 1}/{rotation_frames}")

    plt.close(fig)

    # Save GIF
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # ms per frame (20 fps)
            loop=0,
        )
        print(f"Saved: {output_gif}")

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Animate GNG rotation on single ring")
    parser.add_argument("--image", type=str, default="single_ring.png", help="Shape image path")
    parser.add_argument("-n", "--n-samples", type=int, default=1500, help="Training samples")
    parser.add_argument("--train-iterations", type=int, default=5000, help="Training iterations")
    parser.add_argument("--frames", type=int, default=60, help="Animation frames")
    parser.add_argument("--rotations", type=float, default=1.0, help="Number of full rotations")
    parser.add_argument("--output", type=str, default="gng_single_ring_rotation.gif", help="Output GIF")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_rotation_animation(
        image_path=args.image,
        n_samples=args.n_samples,
        train_iterations=args.train_iterations,
        rotation_frames=args.frames,
        rotations=args.rotations,
        output_gif=args.output,
        seed=args.seed,
    )
