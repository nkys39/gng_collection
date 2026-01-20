"""Test SOM dynamic tracking on moving ring."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "som" / "python"))

from model import SelfOrganizingMap, SOMParams


def generate_ring_samples(
    center: np.ndarray,
    r_inner: float,
    r_outer: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
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
    orbit_center: np.ndarray,
    orbit_radius: float,
    frame: int,
    total_frames: int,
    grid_shape: tuple[int, int],
) -> None:
    ax.clear()

    theta = np.linspace(0, 2 * np.pi, 100)
    orbit_x = orbit_center[0] + orbit_radius * np.cos(theta)
    orbit_y = orbit_center[1] + orbit_radius * np.sin(theta)
    ax.plot(orbit_x, orbit_y, "g--", alpha=0.3, linewidth=1, label="Orbit")

    ring_outer_x = ring_center[0] + ring_r_outer * np.cos(theta)
    ring_outer_y = ring_center[1] + ring_r_outer * np.sin(theta)
    ring_inner_x = ring_center[0] + ring_r_inner * np.cos(theta)
    ring_inner_y = ring_center[1] + ring_r_inner * np.sin(theta)
    ax.fill(ring_outer_x, ring_outer_y, color="skyblue", alpha=0.3)
    ax.fill(ring_inner_x, ring_inner_y, color="white")

    ax.scatter(current_samples[:, 0], current_samples[:, 1],
               c="skyblue", s=5, alpha=0.5)

    for i, j in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
                "r-", linewidth=1.0, alpha=0.5)

    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=25, zorder=5,
               label=f"SOM {grid_shape[0]}x{grid_shape[1]}")

    ax.scatter([ring_center[0]], [ring_center[1]], c="blue", s=30, marker="+", zorder=6)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"SOM Tracking - Frame {frame}/{total_frames}")
    ax.legend(loc="upper right", fontsize=8)


def run_tracking_experiment(
    orbit_center: tuple[float, float] = (0.5, 0.5),
    orbit_radius: float = 0.25,
    ring_r_inner: float = 0.08,
    ring_r_outer: float = 0.12,
    total_frames: int = 120,
    samples_per_frame: int = 50,
    output_gif: str = "som_tracking.gif",
    seed: int = 42,
    grid_size: int = 6,
) -> None:
    rng = np.random.default_rng(seed)
    orbit_center = np.array(orbit_center)

    params = SOMParams(
        grid_height=grid_size,
        grid_width=grid_size,
        sigma_initial=grid_size / 2,
        sigma_final=1.0,
        learning_rate_initial=0.3,
        learning_rate_final=0.05,
    )
    som = SelfOrganizingMap(n_dim=2, params=params, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    frames = []

    print(f"Running SOM tracking experiment ({total_frames} frames)...")
    print(f"Grid: {grid_size}x{grid_size}")

    for frame in range(total_frames):
        angle = (frame / total_frames) * 2 * np.pi
        ring_center = orbit_center + orbit_radius * np.array([np.cos(angle), np.sin(angle)])

        samples = generate_ring_samples(
            ring_center, ring_r_inner, ring_r_outer, samples_per_frame, rng
        )

        for sample in samples:
            som.partial_fit(sample)

        nodes, edges = som.get_graph()

        create_tracking_frame(
            ax, ring_center, ring_r_inner, ring_r_outer, samples,
            nodes, edges, orbit_center, orbit_radius,
            frame + 1, total_frames, (params.grid_height, params.grid_width)
        )
        fig.canvas.draw()

        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        )
        frames.append(img.convert("RGB"))

        if (frame + 1) % 20 == 0:
            print(f"  Frame {frame + 1}/{total_frames}")

    plt.close(fig)

    if frames:
        frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                      duration=80, loop=0)
        print(f"Saved: {output_gif}")

    print("Done!")


if __name__ == "__main__":
    run_tracking_experiment()
