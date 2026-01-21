"""Test Growing Grid on triple ring data with GIF visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "growing_grid" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import GrowingGrid, GrowingGridParams
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
    extra_info: str = "",
) -> None:
    """Create a single frame for visualization."""
    ax.clear()

    # Draw geometric background
    draw_triple_ring_background(ax)

    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3, label="Data")

    # Plot edges (grid structure)
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "r-",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5, label="Nodes")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")

    title = f"Growing Grid - Iteration {iteration} ({len(nodes)} nodes)"
    if extra_info:
        title += f" [{extra_info}]"
    ax.set_title(title)
    ax.legend(loc="upper right")


def run_experiment(
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "gg_triple_ring_growth.gif",
    output_final: str = "gg_triple_ring_final.png",
    seed: int = 42,
) -> None:
    """Run Growing Grid experiment with visualization."""
    np.random.seed(seed)

    # Sample points using mathematical formula
    print(f"Sampling {n_samples} points from triple ring (mathematical)...")
    points = sample_triple_ring(n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Growing Grid parameters
    params = GrowingGridParams(
        initial_height=2,
        initial_width=2,
        max_nodes=100,
        lambda_=100,      # Growth interval
        eps_b=0.1,        # Winner learning rate
        eps_n=0.01,       # Neighbor learning rate
        sigma=1.5,        # Constant neighborhood radius
        tau=0.005,        # Error decay rate
    )
    model = GrowingGrid(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            extra_info = f"{m.height}x{m.width} grid"
            create_frame(ax, points, nodes, edges, iteration, extra_info)
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes ({m.height}x{m.width}), {len(edges)} edges")

    print(f"Training Growing Grid for {n_iterations} iterations...")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}, sigma={params.sigma}")
    model.train(points, n_iterations=n_iterations, callback=callback)

    nodes, edges = model.get_graph()
    extra_info = f"{model.height}x{model.width} grid"
    create_frame(ax, points, nodes, edges, n_iterations, extra_info)
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    if frames:
        frames.extend([frames[-1]] * 10)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Growing Grid on triple ring data")
    parser.add_argument("-n", "--n-samples", type=int, default=1500)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output-gif", type=str, default="gg_triple_ring_growth.gif")
    parser.add_argument("--output-final", type=str, default="gg_triple_ring_final.png")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_experiment(
        n_samples=args.n_samples,
        n_iterations=args.iterations,
        gif_frames=args.frames,
        output_gif=args.output_gif,
        output_final=args.output_final,
        seed=args.seed,
    )
