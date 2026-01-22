"""Test GNG algorithm on double ring data with GIF visualization.

Usage:
    # Generate shape image
    python ../../data/2d/shapes/generate_shape.py --shape double_ring --output double_ring.png

    # Run this script
    python test_gng_double_ring.py
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


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
    bg_image: np.ndarray | None = None,
) -> None:
    """Create a single frame for visualization."""
    ax.clear()

    if bg_image is not None:
        ax.imshow(bg_image, extent=[0, 1, 1, 0], alpha=0.5)

    # Plot sample points
    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3, label="Data")

    # Plot edges
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "r-",
            linewidth=1.5,
            alpha=0.7,
        )

    # Plot nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5, label="Nodes")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect("equal")
    ax.set_title(f"GNG Training - Iteration {iteration} ({len(nodes)} nodes)")
    ax.legend(loc="upper right")


def run_experiment(
    image_path: str = "double_ring.png",
    n_samples: int = 2000,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "gng_growth.gif",
    output_final: str = "gng_final.png",
    seed: int = 42,
) -> None:
    """Run GNG experiment with visualization.

    Args:
        image_path: Path to shape image.
        n_samples: Number of points to sample.
        n_iterations: Number of GNG training iterations.
        gif_frames: Number of frames in output GIF.
        output_gif: Output GIF path.
        output_final: Output final image path.
        seed: Random seed.
    """
    np.random.seed(seed)

    # Generate shape image if not exists
    if not Path(image_path).exists():
        print(f"Generating shape image: {image_path}")
        shapes_dir = Path(__file__).parents[2] / "data" / "2d" / "shapes"
        sys.path.insert(0, str(shapes_dir))
        from generate_shape import generate_double_ring

        generate_double_ring(image_path)

    # Load background image
    bg_image = np.array(Image.open(image_path).convert("RGB"))

    # Sample points from image
    print(f"Sampling {n_samples} points from {image_path}...")
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Setup GNG with parameters based on watanabe_gng reference
    params = GNGParams(
        max_nodes=100,
        lambda_=100,      # Node insertion interval
        eps_b=0.08,       # Winner learning rate (LEARNRATE_S1)
        eps_n=0.008,      # Neighbor learning rate (LEARNRATE_S2)
        alpha=0.5,        # Error decay on split (ALFA)
        beta=0.005,       # Global error decay (BETA)
        max_age=100,      # Maximum edge age (MAX_EDGE_AGE)
    )
    gng = GrowingNeuralGas(n_dim=2, params=params, seed=seed)

    # Collect frames for GIF
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()
            create_frame(ax, points, nodes, edges, iteration, bg_image)
            fig.canvas.draw()

            # Convert to PIL Image
            fig.canvas.draw()
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges")

    # Train
    print(f"Training GNG for {n_iterations} iterations...")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}, eps_n={params.eps_n}, max_age={params.max_age}")
    gng.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame
    nodes, edges = gng.get_graph()
    create_frame(ax, points, nodes, edges, n_iterations, bg_image)
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    # Save GIF
    if frames:
        # Add extra copies of final frame
        frames.extend([frames[-1]] * 10)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # ms per frame
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GNG on double ring data")
    parser.add_argument("--image", type=str, default="double_ring.png", help="Shape image path")
    parser.add_argument("-n", "--n-samples", type=int, default=2000, help="Number of samples")
    parser.add_argument(
        "--iterations", type=int, default=5000, help="Number of training iterations"
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of GIF frames")
    parser.add_argument("--output-gif", type=str, default="gng_growth.gif", help="Output GIF path")
    parser.add_argument(
        "--output-final", type=str, default="gng_final.png", help="Output final image path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_experiment(
        image_path=args.image,
        n_samples=args.n_samples,
        n_iterations=args.iterations,
        gif_frames=args.frames,
        output_gif=args.output_gif,
        output_final=args.output_final,
        seed=args.seed,
    )
