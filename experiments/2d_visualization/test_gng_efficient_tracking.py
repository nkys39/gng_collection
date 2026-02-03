"""Test GNG Efficient dynamic tracking on moving ring.

Tests the optimized GNG implementation's ability to track
a non-stationary distribution.

Output:
    - gng_efficient_tracking.gif: Tracking animation
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add paths for imports
sys.path.insert(
    0, str(Path(__file__).parents[2] / "algorithms" / "gng_efficient" / "python")
)

from model import GNGEfficient, GNGEfficientParams


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
    orbit_center: np.ndarray,
    orbit_radius: float,
    frame: int,
    total_frames: int,
    extra_info: str = "",
) -> None:
    """Create a single frame for tracking visualization."""
    ax.clear()

    # Draw orbit path
    theta = np.linspace(0, 2 * np.pi, 100)
    orbit_x = orbit_center[0] + orbit_radius * np.cos(theta)
    orbit_y = orbit_center[1] + orbit_radius * np.sin(theta)
    ax.plot(orbit_x, orbit_y, "g--", alpha=0.3, linewidth=1, label="Orbit")

    # Draw ring outline
    ring_outer_x = ring_center[0] + ring_r_outer * np.cos(theta)
    ring_outer_y = ring_center[1] + ring_r_outer * np.sin(theta)
    ring_inner_x = ring_center[0] + ring_r_inner * np.cos(theta)
    ring_inner_y = ring_center[1] + ring_r_inner * np.sin(theta)
    ax.fill(ring_outer_x, ring_outer_y, color="skyblue", alpha=0.3)
    ax.fill(ring_inner_x, ring_inner_y, color="white")
    ax.plot(ring_outer_x, ring_outer_y, "c-", alpha=0.5, linewidth=1)
    ax.plot(ring_inner_x, ring_inner_y, "c-", alpha=0.5, linewidth=1)

    # Draw current samples
    ax.scatter(
        current_samples[:, 0],
        current_samples[:, 1],
        c="skyblue",
        s=5,
        alpha=0.5,
        label="Current samples",
    )

    # Draw edges
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "r-",
            linewidth=1.5,
            alpha=0.7,
        )

    # Draw nodes
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c="red",
        s=40,
        zorder=5,
        label=f"Nodes ({len(nodes)})",
    )

    # Draw ring center
    ax.scatter([ring_center[0]], [ring_center[1]], c="blue", s=30, marker="+", zorder=6)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")

    title = f"GNG Efficient Tracking - Frame {frame}/{total_frames}"
    if extra_info:
        title += f" [{extra_info}]"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def run_tracking_experiment(
    orbit_center: tuple[float, float] = (0.5, 0.5),
    orbit_radius: float = 0.25,
    ring_r_inner: float = 0.08,
    ring_r_outer: float = 0.12,
    total_frames: int = 120,
    samples_per_frame: int = 50,
    output_gif: str = "gng_efficient_tracking.gif",
    seed: int = 42,
) -> None:
    """Run tracking experiment."""
    rng = np.random.default_rng(seed)
    orbit_center = np.array(orbit_center)

    # Setup GNG Efficient with tracking-optimized parameters
    params = GNGEfficientParams(
        max_nodes=50,
        lambda_=20,
        eps_b=0.15,
        eps_n=0.01,
        alpha=0.5,
        beta=0.01,
        max_age=30,
        # Optimization parameters
        use_uniform_grid=True,
        use_lazy_error=True,
        h_t=0.1,
        h_rho=1.5,
    )
    model = GNGEfficient(n_dim=2, params=params, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    frames = []

    print(f"Running GNG Efficient tracking experiment ({total_frames} frames)...")
    print(f"Ring moves along orbit: center={orbit_center}, radius={orbit_radius}")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}, max_age={params.max_age}")
    print(f"Optimizations: uniform_grid={params.use_uniform_grid}, lazy_error={params.use_lazy_error}")

    for frame in range(total_frames):
        angle = (frame / total_frames) * 2 * np.pi
        ring_center = orbit_center + orbit_radius * np.array([np.cos(angle), np.sin(angle)])

        samples = generate_ring_samples(
            ring_center, ring_r_inner, ring_r_outer, samples_per_frame, rng
        )

        for sample in samples:
            model.partial_fit(sample)

        nodes, edges = model.get_graph()

        extra_info = f"cycle={model.cycle}"

        create_tracking_frame(
            ax,
            ring_center,
            ring_r_inner,
            ring_r_outer,
            samples,
            nodes,
            edges,
            orbit_center,
            orbit_radius,
            frame + 1,
            total_frames,
            extra_info,
        )
        fig.canvas.draw()

        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        )
        frames.append(img.convert("RGB"))

        if (frame + 1) % 20 == 0:
            print(f"  Frame {frame + 1}/{total_frames}: {len(nodes)} nodes, {len(edges)} edges")

    plt.close(fig)

    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
        )
        print(f"Saved: {output_gif}")

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GNG Efficient tracking on moving ring")
    parser.add_argument("--orbit-radius", type=float, default=0.25, help="Orbit radius")
    parser.add_argument("--ring-inner", type=float, default=0.08, help="Ring inner radius")
    parser.add_argument("--ring-outer", type=float, default=0.12, help="Ring outer radius")
    parser.add_argument("--frames", type=int, default=120, help="Total frames")
    parser.add_argument("--samples", type=int, default=50, help="Samples per frame")
    parser.add_argument("--output", type=str, default="gng_efficient_tracking.gif")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_tracking_experiment(
        orbit_radius=args.orbit_radius,
        ring_r_inner=args.ring_inner,
        ring_r_outer=args.ring_outer,
        total_frames=args.frames,
        samples_per_frame=args.samples,
        output_gif=args.output,
        seed=args.seed,
    )
