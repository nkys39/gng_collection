"""Test AiS-GNG-AM (SMC 2023) tracking on moving ring.

SMC 2023 version includes Amount of Movement (AM) tracking.
Nodes are colored by their movement level (blue=static, red=moving).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "ais_gng" / "python"))

from model_am import AiSGNGAM, AiSGNGAMParams


def generate_ring_samples(
    center: np.ndarray,
    r_inner: float,
    r_outer: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random samples from a ring shape."""
    samples = []
    while len(samples) < n_samples:
        theta = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(rng.uniform(r_inner**2, r_outer**2))
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append([x, y])
    return np.array(samples)


def create_tracking_frame(
    ax,
    ring_center: np.ndarray,
    ring_r_inner: float,
    ring_r_outer: float,
    current_samples: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    movements: np.ndarray,
    orbit_center: np.ndarray,
    orbit_radius: float,
    frame: int,
    total_frames: int,
    n_ais_additions: int = 0,
    n_moving: int = 0,
) -> None:
    """Create a frame with AM-colored nodes."""
    ax.clear()

    theta = np.linspace(0, 2 * np.pi, 100)
    orbit_x = orbit_center[0] + orbit_radius * np.cos(theta)
    orbit_y = orbit_center[1] + orbit_radius * np.sin(theta)
    ax.plot(orbit_x, orbit_y, "g--", alpha=0.3, linewidth=1)

    ring_outer_x = ring_center[0] + ring_r_outer * np.cos(theta)
    ring_outer_y = ring_center[1] + ring_r_outer * np.sin(theta)
    ring_inner_x = ring_center[0] + ring_r_inner * np.cos(theta)
    ring_inner_y = ring_center[1] + ring_r_inner * np.sin(theta)
    ax.fill(ring_outer_x, ring_outer_y, color="skyblue", alpha=0.3)
    ax.fill(ring_inner_x, ring_inner_y, color="white")

    ax.scatter(current_samples[:, 0], current_samples[:, 1], c="skyblue", s=5, alpha=0.5)

    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "gray",
            linewidth=1.5,
            alpha=0.5,
        )

    # Color nodes by movement amount
    if len(nodes) > 0 and len(movements) > 0:
        max_movement = max(movements.max(), 0.001)
        norm_movements = movements / max_movement
        ax.scatter(
            nodes[:, 0], nodes[:, 1],
            c=norm_movements, cmap="coolwarm",
            s=50, zorder=5, edgecolors="black", linewidths=0.5,
            vmin=0, vmax=1,
        )

    ax.scatter([ring_center[0]], [ring_center[1]], c="blue", s=30, marker="+", zorder=6)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(
        f"AiS-GNG-AM Tracking - Frame {frame}/{total_frames} "
        f"({len(nodes)} nodes, {n_moving} moving)"
    )


def run_tracking_experiment(
    orbit_center: tuple[float, float] = (0.5, 0.5),
    orbit_radius: float = 0.25,
    ring_r_inner: float = 0.08,
    ring_r_outer: float = 0.12,
    total_frames: int = 120,
    samples_per_frame: int = 50,
    output_gif: str = "aisgng_am_tracking.gif",
    seed: int = 42,
) -> None:
    """Run AiS-GNG-AM tracking experiment."""
    rng = np.random.default_rng(seed)
    orbit_center = np.array(orbit_center)

    params = AiSGNGAMParams(
        max_nodes=50,
        lambda_=20,
        eps_b=0.15,
        eps_n=0.01,
        alpha=0.5,
        beta=0.01,
        chi=0.01,
        max_age=30,
        utility_k=50.0,
        kappa=5,
        theta_ais_min=0.01,
        theta_ais_max=0.05,
        am_decay=0.9,      # Faster decay for tracking
        am_threshold=0.005,
    )
    gng = AiSGNGAM(n_dim=2, params=params, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    frames = []

    print(f"Running AiS-GNG-AM tracking ({total_frames} frames)...")
    print(f"AM params: decay={params.am_decay}, threshold={params.am_threshold}")

    for frame in range(total_frames):
        angle = (frame / total_frames) * 2 * np.pi
        ring_center = orbit_center + orbit_radius * np.array([np.cos(angle), np.sin(angle)])

        samples = generate_ring_samples(
            ring_center, ring_r_inner, ring_r_outer, samples_per_frame, rng
        )

        for sample in samples:
            gng.partial_fit(sample)

        nodes, edges = gng.get_graph()
        movements = gng.get_node_movements()
        n_moving = sum(gng.get_moving_nodes_mask())

        create_tracking_frame(
            ax, ring_center, ring_r_inner, ring_r_outer, samples, nodes, edges,
            movements, orbit_center, orbit_radius, frame + 1, total_frames,
            gng.n_ais_additions, n_moving,
        )
        fig.canvas.draw()

        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        )
        frames.append(img.convert("RGB"))

        if (frame + 1) % 30 == 0:
            print(f"  Frame {frame + 1}: {len(nodes)} nodes, {n_moving} moving")

    plt.close(fig)

    if frames:
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=80, loop=0)
        print(f"Saved: {output_gif}")

    print(f"Done! AiS: {gng.n_ais_additions}, Final moving: {n_moving}/{len(nodes)}")


if __name__ == "__main__":
    run_tracking_experiment()
