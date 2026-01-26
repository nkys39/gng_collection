"""Test AiS-GNG RO-MAN 2023 version on triple ring data.

RO-MAN 2023 version uses a single threshold θ_AiS instead of range [θ_min, θ_max].
Condition: ||v_t - h_s1|| < θ_AiS AND ||v_t - h_s2|| < θ_AiS

Usage:
    python test_aisgng_roman_triple_ring.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "ais_gng" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model_roman import AiSGNGRoman, AiSGNGRomanParams
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
    iteration: int,
    n_ais_additions: int = 0,
    n_utility_removals: int = 0,
) -> None:
    """Create a single frame for visualization."""
    ax.clear()
    draw_triple_ring_background(ax)

    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3)

    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "r-",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(
        f"AiS-GNG RO-MAN - Iter {iteration} ({len(nodes)} nodes, "
        f"+{n_ais_additions} AiS, -{n_utility_removals} utility)"
    )


def run_experiment(
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "aisgng_roman_triple_ring_growth.gif",
    output_final: str = "aisgng_roman_triple_ring_final.png",
    seed: int = 42,
    theta_ais: float = 0.10,
) -> None:
    """Run AiS-GNG RO-MAN experiment."""
    np.random.seed(seed)

    print(f"Sampling {n_samples} points from triple ring...")
    points = sample_triple_ring(n_samples=n_samples, seed=seed)

    params = AiSGNGRomanParams(
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
        theta_ais=theta_ais,  # Single threshold (RO-MAN 2023)
    )
    gng = AiSGNGRoman(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()
            create_frame(
                ax, points, nodes, edges, iteration,
                model.n_ais_additions, model.n_utility_removals
            )
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(
                f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges, "
                f"+{model.n_ais_additions} AiS"
            )

    print(f"Training AiS-GNG RO-MAN for {n_iterations} iterations...")
    print(f"Single threshold: theta_ais={params.theta_ais}")
    gng.train(points, n_iterations=n_iterations, callback=callback)

    nodes, edges = gng.get_graph()
    create_frame(
        ax, points, nodes, edges, n_iterations,
        gng.n_ais_additions, gng.n_utility_removals
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
    print(f"Done! AiS additions: {gng.n_ais_additions}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AiS-GNG RO-MAN on triple ring")
    parser.add_argument("-n", "--n-samples", type=int, default=1500)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output-gif", type=str, default="aisgng_roman_triple_ring_growth.gif")
    parser.add_argument("--output-final", type=str, default="aisgng_roman_triple_ring_final.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--theta-ais", type=float, default=0.10, help="Single AiS threshold")

    args = parser.parse_args()

    run_experiment(
        n_samples=args.n_samples,
        n_iterations=args.iterations,
        gif_frames=args.frames,
        output_gif=args.output_gif,
        output_final=args.output_final,
        seed=args.seed,
        theta_ais=args.theta_ais,
    )
