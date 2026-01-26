"""Test AiS-GNG-AM (SMC 2023) on triple ring data.

SMC 2023 version includes Amount of Movement (AM) tracking for
dynamic object detection. Nodes are colored by their movement level.

Usage:
    python test_aisgng_am_triple_ring.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "ais_gng" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model_am import AiSGNGAM, AiSGNGAMParams
from sampler import sample_triple_ring

TRIPLE_RING_PARAMS = [
    (0.50, 0.23, 0.06, 0.14),
    (0.27, 0.68, 0.06, 0.14),
    (0.73, 0.68, 0.06, 0.14),
]


def draw_triple_ring_background(ax) -> None:
    """Draw triple ring background."""
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
    movements: np.ndarray,
    iteration: int,
    n_ais_additions: int = 0,
    n_utility_removals: int = 0,
    n_moving: int = 0,
) -> None:
    """Create a single frame with AM-colored nodes."""
    ax.clear()
    draw_triple_ring_background(ax)

    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3)

    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "gray",
            linewidth=1.5,
            alpha=0.5,
        )

    # Color nodes by movement amount (blue=static, red=moving)
    if len(nodes) > 0 and len(movements) > 0:
        # Normalize movements for coloring
        max_movement = max(movements.max(), 0.001)
        norm_movements = movements / max_movement

        # Create colormap: blue (static) -> red (moving)
        colors = plt.cm.coolwarm(norm_movements)

        scatter = ax.scatter(
            nodes[:, 0], nodes[:, 1],
            c=norm_movements,
            cmap="coolwarm",
            s=60,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            vmin=0,
            vmax=1,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(
        f"AiS-GNG-AM - Iter {iteration} ({len(nodes)} nodes, "
        f"+{n_ais_additions} AiS, {n_moving} moving)"
    )


def run_experiment(
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "aisgng_am_triple_ring_growth.gif",
    output_final: str = "aisgng_am_triple_ring_final.png",
    seed: int = 42,
) -> None:
    """Run AiS-GNG-AM experiment."""
    np.random.seed(seed)

    print(f"Sampling {n_samples} points from triple ring...")
    points = sample_triple_ring(n_samples=n_samples, seed=seed)

    params = AiSGNGAMParams(
        max_nodes=100,
        lambda_=100,
        eps_b=0.08,
        eps_n=0.008,
        alpha=0.5,
        beta=0.005,
        chi=0.005,
        max_age=100,
        utility_k=1000.0,
        kappa=10,
        theta_ais_min=0.02,
        theta_ais_max=0.10,
        am_decay=0.95,
        am_threshold=0.005,
    )
    gng = AiSGNGAM(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()
            movements = model.get_node_movements()
            n_moving = sum(model.get_moving_nodes_mask())

            create_frame(
                ax, points, nodes, edges, movements, iteration,
                model.n_ais_additions, model.n_utility_removals, n_moving
            )
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(
                f"Iteration {iteration}: {len(nodes)} nodes, "
                f"+{model.n_ais_additions} AiS, {n_moving} moving"
            )

    print(f"Training AiS-GNG-AM for {n_iterations} iterations...")
    print(f"AM params: decay={params.am_decay}, threshold={params.am_threshold}")
    gng.train(points, n_iterations=n_iterations, callback=callback)

    nodes, edges = gng.get_graph()
    movements = gng.get_node_movements()
    n_moving = sum(gng.get_moving_nodes_mask())

    create_frame(
        ax, points, nodes, edges, movements, n_iterations,
        gng.n_ais_additions, gng.n_utility_removals, n_moving
    )
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
    print(f"Done! AiS: {gng.n_ais_additions}, Moving: {n_moving}/{len(nodes)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AiS-GNG-AM on triple ring")
    parser.add_argument("-n", "--n-samples", type=int, default=1500)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output-gif", type=str, default="aisgng_am_triple_ring_growth.gif")
    parser.add_argument("--output-final", type=str, default="aisgng_am_triple_ring_final.png")
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
