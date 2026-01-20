"""Test GNG-T (Delaunay Triangulation) learning on triple ring pattern.

GNG-T uses explicit Delaunay triangulation instead of Competitive Hebbian
Learning for topology management.
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
from sampler import sample_from_image


def create_visualization_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    triangles: list[tuple[int, int, int]] | None,
    iteration: int,
    bg_image: np.ndarray | None = None,
    show_triangles: bool = True,
) -> None:
    """Create a single visualization frame."""
    ax.clear()

    if bg_image is not None:
        ax.imshow(bg_image, extent=[0, 1, 1, 0], alpha=0.3)

    # Plot sample points
    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3)

    # Draw Delaunay triangles (filled)
    if show_triangles and triangles:
        for tri in triangles:
            triangle = plt.Polygon(
                nodes[list(tri)],
                fill=True,
                facecolor="yellow",
                edgecolor="orange",
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
    ax.set_title(f"GNG-T (Delaunay) - Iter {iteration} ({len(nodes)} nodes, {len(edges)} edges)")


def run_experiment(
    n_iterations: int = 5000,
    n_frames: int = 50,
    n_samples: int = 1500,
    image_path: str = "triple_ring.png",
    output_gif: str = "gngt_triple_ring_growth.gif",
    output_png: str = "gngt_triple_ring_final.png",
    seed: int = 42,
    show_triangles: bool = True,
) -> None:
    """Run GNG-T triple ring experiment."""
    # Load and sample from image
    print(f"Sampling {n_samples} points from {image_path}...")

    bg_image = plt.imread(image_path)
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Setup GNG-T
    params = GNGTParams(
        max_nodes=100,
        lambda_=100,
        eps_b=0.05,
        eps_n=0.006,
        alpha=0.5,
        beta=0.0005,
        update_topology_every=10,  # Update Delaunay every 10 iterations
    )
    gng_t = GrowingNeuralGasT(n_dim=2, params=params, seed=seed)

    print(f"Training GNG-T for {n_iterations} iterations...")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}, update_every={params.update_topology_every}")

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
                ax, points, nodes, edges, triangles, iteration, bg_image, show_triangles
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
        ax, points, nodes, edges, triangles, n_iterations, bg_image, show_triangles
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
    parser.add_argument("--image", type=str, default="triple_ring.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-triangles", action="store_true", help="Hide triangle fill")

    args = parser.parse_args()

    run_experiment(
        n_iterations=args.iterations,
        n_frames=args.frames,
        n_samples=args.samples,
        image_path=args.image,
        seed=args.seed,
        show_triangles=not args.no_triangles,
    )
