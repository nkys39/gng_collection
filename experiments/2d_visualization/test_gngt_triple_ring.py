"""Test GNG-T (Triangulation) learning on triple ring pattern.

GNG-T uses heuristic triangulation (quadrilateral search + intersection search)
based on Kubota & Satomi (2008) to maintain proper triangle mesh structure.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "gng_t" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import GrowingNeuralGasT, GNGTParams
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


def create_visualization_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    triangles: list[tuple[int, int, int]] | None,
    iteration: int,
    show_triangles: bool = True,
) -> None:
    """Create a single visualization frame."""
    ax.clear()

    # Draw geometric background
    draw_triple_ring_background(ax)

    # Plot sample points
    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3)

    # Draw triangles (filled)
    if show_triangles and triangles:
        for tri in triangles:
            triangle = plt.Polygon(
                nodes[list(tri)],
                fill=True,
                facecolor="lightgreen",
                edgecolor="green",
                alpha=0.2,
                linewidth=0.5,
            )
            ax.add_patch(triangle)

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
    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(f"GNG-T (Triangulation) - Iter {iteration} ({len(nodes)} nodes, {len(edges)} edges)")


def run_experiment(
    n_iterations: int = 5000,
    n_frames: int = 50,
    n_samples: int = 1500,
    output_gif: str = "gngt_triple_ring_growth.gif",
    output_png: str = "gngt_triple_ring_final.png",
    seed: int = 42,
    show_triangles: bool = True,
) -> None:
    """Run GNG-T triple ring experiment."""
    # Sample points using mathematical formula
    print(f"Sampling {n_samples} points from triple ring (mathematical)...")
    points = sample_triple_ring(n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Setup GNG-T
    params = GNGTParams(
        max_nodes=100,
        lambda_=100,
        eps_b=0.08,
        eps_n=0.008,
        alpha=0.5,
        beta=0.005,
        max_age=100,
    )
    gng_t = GrowingNeuralGasT(n_dim=2, params=params, seed=seed)

    print(f"Training GNG-T for {n_iterations} iterations...")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}, max_age={params.max_age}")

    # Create figure for animation
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    frames = []
    frame_interval = max(1, n_iterations // n_frames)

    def callback(model, iteration):
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: {model.n_nodes} nodes, {model.n_edges} edges")

        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()
            triangles = model.get_triangles() if show_triangles else None

            create_visualization_frame(
                ax, points, nodes, edges, triangles, iteration, show_triangles
            )
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))

    # Train
    gng_t.train(points, n_iterations=n_iterations, callback=callback)

    # Final frame
    nodes, edges = gng_t.get_graph()
    triangles = gng_t.get_triangles() if show_triangles else None
    print(f"Iteration {n_iterations - 1}: {gng_t.n_nodes} nodes, {gng_t.n_edges} edges")

    create_visualization_frame(
        ax, points, nodes, edges, triangles, n_iterations, show_triangles
    )
    fig.savefig(output_png, dpi=100, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_png}")

    plt.close(fig)

    # Save GIF
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GNG-T on triple ring")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-triangles", action="store_true", help="Hide triangle fill")

    args = parser.parse_args()

    run_experiment(
        n_iterations=args.iterations,
        n_frames=args.frames,
        n_samples=args.samples,
        seed=args.seed,
        show_triangles=not args.no_triangles,
    )
