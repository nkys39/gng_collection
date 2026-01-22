"""Test LBG (Linde-Buzo-Gray) on triple ring data with GIF visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "lbg" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import LindeBuzoGray, LBGParams
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

    # LBG has no edges, just plot nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5, label="Nodes")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")

    title = f"LBG - Epoch {iteration} ({len(nodes)} nodes)"
    if extra_info:
        title += f" [{extra_info}]"
    ax.set_title(title)
    ax.legend(loc="upper right")


def run_experiment(
    n_samples: int = 1500,
    n_iterations: int = 100,  # LBG uses epochs, not individual iterations
    gif_frames: int = 50,
    output_gif: str = "lbg_triple_ring_growth.gif",
    output_final: str = "lbg_triple_ring_final.png",
    seed: int = 42,
) -> None:
    """Run LBG experiment with visualization."""
    np.random.seed(seed)

    # Sample points using mathematical formula
    print(f"Sampling {n_samples} points from triple ring (mathematical)...")
    points = sample_triple_ring(n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # LBG parameters
    params = LBGParams(
        n_nodes=50,
        max_epochs=n_iterations,
        convergence_threshold=1e-6,
        use_utility=False,
    )
    model = LindeBuzoGray(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(m, epoch):
        if epoch % frame_interval == 0 or epoch == n_iterations - 1:
            nodes, edges = m.get_graph()
            create_frame(ax, points, nodes, edges, epoch)
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Epoch {epoch}: {len(nodes)} nodes")

    print(f"Training LBG for {n_iterations} epochs...")
    print(f"Parameters: n_nodes={params.n_nodes}")
    model.train(points, n_iterations=n_iterations, callback=callback)

    nodes, edges = model.get_graph()
    create_frame(ax, points, nodes, edges, model.n_learning)
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    if frames:
        frames.extend([frames[-1]] * 10)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # Slower for batch learning
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LBG on triple ring data")
    parser.add_argument("-n", "--n-samples", type=int, default=1500)
    parser.add_argument("--iterations", type=int, default=100)  # Epochs for LBG
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output-gif", type=str, default="triple_ring_growth.gif")
    parser.add_argument("--output-final", type=str, default="triple_ring_final.png")
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
