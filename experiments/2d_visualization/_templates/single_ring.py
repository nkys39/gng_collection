"""Test [ALGORITHM_NAME] on single ring data with GIF visualization.

Template for single ring static distribution test.

Usage:
    1. Copy this file to test_[algorithm]_single_ring.py
    2. Update imports and class names
    3. Adjust parameters if needed
    4. Run: python test_[algorithm]_single_ring.py

Output:
    - [algorithm]_single_ring_final.png: Final network state
    - [algorithm]_single_ring_growth.gif: Training animation
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add paths for imports - UPDATE THESE PATHS
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "[algorithm]" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

# UPDATE: Import your algorithm class and params
from model import AlgorithmName, AlgorithmParams
from sampler import sample_from_image


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
    bg_image: np.ndarray | None = None,
    extra_info: str = "",
) -> None:
    """Create a single frame for visualization.

    Args:
        ax: Matplotlib axes.
        points: Sample points array.
        nodes: Node positions array.
        edges: List of edge tuples.
        iteration: Current iteration number.
        bg_image: Optional background image.
        extra_info: Extra info to show in title (e.g., "5 removed").
    """
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

    # UPDATE: Change algorithm name in title
    title = f"[ALGORITHM] - Iteration {iteration} ({len(nodes)} nodes)"
    if extra_info:
        title += f" [{extra_info}]"
    ax.set_title(title)
    ax.legend(loc="upper right")


def run_experiment(
    image_path: str = "single_ring.png",
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "[algorithm]_single_ring_growth.gif",
    output_final: str = "[algorithm]_single_ring_final.png",
    seed: int = 42,
) -> None:
    """Run algorithm experiment with visualization.

    Args:
        image_path: Path to shape image.
        n_samples: Number of points to sample.
        n_iterations: Number of training iterations.
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
        from generate_shape import generate_single_ring

        generate_single_ring(image_path)

    # Load background image
    bg_image = np.array(Image.open(image_path).convert("RGB"))

    # Sample points from image
    print(f"Sampling {n_samples} points from {image_path}...")
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # UPDATE: Setup algorithm with parameters
    params = AlgorithmParams(
        max_nodes=100,
        lambda_=100,      # Node insertion interval
        eps_b=0.08,       # Winner learning rate
        eps_n=0.008,      # Neighbor learning rate
        alpha=0.5,        # Error decay on split
        beta=0.005,       # Global error decay
        max_age=100,      # Maximum edge age
        # Algorithm-specific parameters:
        # example_param=1.3,
    )
    model = AlgorithmName(n_dim=2, params=params, seed=seed)

    # Collect frames for GIF
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            # UPDATE: Add algorithm-specific info if needed
            extra_info = ""  # e.g., f"{m.n_removals} removed"
            create_frame(ax, points, nodes, edges, iteration, bg_image, extra_info)
            fig.canvas.draw()

            # Convert to PIL Image
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges")

    # Train
    print(f"Training [ALGORITHM] for {n_iterations} iterations...")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}")
    model.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame
    nodes, edges = model.get_graph()
    extra_info = ""  # UPDATE: Add algorithm-specific info
    create_frame(ax, points, nodes, edges, n_iterations, bg_image, extra_info)
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

    parser = argparse.ArgumentParser(description="Test [ALGORITHM] on single ring data")
    parser.add_argument("--image", type=str, default="single_ring.png", help="Shape image path")
    parser.add_argument("-n", "--n-samples", type=int, default=1500, help="Number of samples")
    parser.add_argument(
        "--iterations", type=int, default=5000, help="Number of training iterations"
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of GIF frames")
    parser.add_argument("--output-gif", type=str, default="[algorithm]_single_ring_growth.gif", help="Output GIF path")
    parser.add_argument(
        "--output-final", type=str, default="[algorithm]_single_ring_final.png", help="Output final image path"
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
