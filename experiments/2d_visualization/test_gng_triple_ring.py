"""Test GNG algorithm on triple ring data with GIF visualization.

Usage:
    python test_gng_triple_ring.py
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
from sampler import sample_triple_ring


# Triple ring geometry (matching C++ implementation)
TRIPLE_RING_PARAMS = [
    (0.50, 0.23, 0.06, 0.14),  # top center
    (0.27, 0.68, 0.06, 0.14),  # bottom left
    (0.73, 0.68, 0.06, 0.14),  # bottom right
]


def draw_triple_ring_background(ax) -> None:
    """Draw triple ring background using geometric shapes."""
    theta = np.linspace(0, 2 * np.pi, 100)
    for cx, cy, r_inner, r_outer in TRIPLE_RING_PARAMS:
        outer_x = cx + r_outer * np.cos(theta)
        outer_y = cy + r_outer * np.sin(theta)
        inner_x = cx + r_inner * np.cos(theta)
        inner_y = cy + r_inner * np.sin(theta)
        ax.fill(outer_x, outer_y, color="lightblue", alpha=0.3)
        ax.fill(inner_x, inner_y, color="white")


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
) -> None:
    """Create a single frame for visualization."""
    ax.clear()

    # Draw geometric background
    draw_triple_ring_background(ax)

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
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "gng_triple_ring_growth.gif",
    output_final: str = "gng_triple_ring_final.png",
    seed: int = 42,
) -> None:
    """Run GNG experiment with visualization.

    Args:
        n_samples: Number of points to sample.
        n_iterations: Number of GNG training iterations.
        gif_frames: Number of frames in output GIF.
        output_gif: Output GIF path.
        output_final: Output final image path.
        seed: Random seed.
    """
    np.random.seed(seed)

    # Sample points using mathematical formula
    print(f"Sampling {n_samples} points from triple ring (mathematical)...")
    points = sample_triple_ring(n_samples=n_samples, seed=seed)
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
            create_frame(ax, points, nodes, edges, iteration)
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
    create_frame(ax, points, nodes, edges, n_iterations)
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

    parser = argparse.ArgumentParser(description="Test GNG on triple ring data")
    parser.add_argument("-n", "--n-samples", type=int, default=1500, help="Number of samples")
    parser.add_argument(
        "--iterations", type=int, default=5000, help="Number of training iterations"
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of GIF frames")
    parser.add_argument("--output-gif", type=str, default="gng_triple_ring_growth.gif", help="Output GIF path")
    parser.add_argument(
        "--output-final", type=str, default="gng_triple_ring_final.png", help="Output final image path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_experiment(
        n_samples=args.n_samples,
        n_iterations=args.iterations,
        gif_frames=args.frames,
        output_gif=args.output_gif,
        output_final=args.output_final,
        seed=args.seed,
    )
