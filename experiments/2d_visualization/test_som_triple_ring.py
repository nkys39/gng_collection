"""Test SOM (Self-Organizing Map) on triple ring data with GIF visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "som" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import SelfOrganizingMap, SOMParams
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
    grid_shape: tuple[int, int] = (10, 10),
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
            linewidth=1.0,
            alpha=0.5,
        )

    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=30, zorder=5, label="Nodes")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(f"SOM {grid_shape[0]}x{grid_shape[1]} - Iteration {iteration}")
    ax.legend(loc="upper right")


def run_experiment(
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "som_triple_ring_growth.gif",
    output_final: str = "som_triple_ring_final.png",
    seed: int = 42,
    grid_size: int = 8,
) -> None:
    """Run SOM experiment."""
    np.random.seed(seed)

    # Sample points using mathematical formula
    print(f"Sampling {n_samples} points from triple ring (mathematical)...")
    points = sample_triple_ring(n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    params = SOMParams(
        grid_height=grid_size,
        grid_width=grid_size,
        sigma_initial=grid_size / 2,
        sigma_final=0.5,
        learning_rate_initial=0.5,
        learning_rate_final=0.01,
    )
    som = SelfOrganizingMap(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()
            create_frame(ax, points, nodes, edges, iteration,
                        (params.grid_height, params.grid_width))
            fig.canvas.draw()
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes")

    print(f"Training SOM for {n_iterations} iterations...")
    print(f"Grid: {grid_size}x{grid_size}, sigma: {params.sigma_initial}->{params.sigma_final}")
    som.train(points, n_iterations=n_iterations, callback=callback)

    nodes, edges = som.get_graph()
    create_frame(ax, points, nodes, edges, n_iterations,
                (params.grid_height, params.grid_width))
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_final}")

    if frames:
        frames.extend([frames[-1]] * 10)
        frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                      duration=100, loop=0)
        print(f"Saved: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    run_experiment()
