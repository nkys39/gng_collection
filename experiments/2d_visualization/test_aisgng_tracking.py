"""Test AiS-GNG dynamic tracking on moving ring.

The ring (input data distribution) moves along a circular orbit,
and AiS-GNG learns online to track the moving distribution.

AiS-GNG is expected to generate high-density topological structures
faster than standard GNG-U due to the Add-if-Silent rule.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "ais_gng" / "python"))

from model import AiSGNG, AiSGNGParams


def generate_ring_samples(
    center: np.ndarray,
    r_inner: float,
    r_outer: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random samples from a ring shape.

    Args:
        center: Center position (x, y) of the ring.
        r_inner: Inner radius of the ring.
        r_outer: Outer radius of the ring.
        n_samples: Number of samples to generate.
        rng: Random number generator.

    Returns:
        Array of shape (n_samples, 2) with (x, y) coordinates.
    """
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
    n_ais_additions: int = 0,
    n_utility_removals: int = 0,
) -> None:
    """Create a single frame for tracking visualization."""
    ax.clear()

    # Draw orbit path (dashed circle)
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

    # Draw AiS-GNG edges
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "r-",
            linewidth=1.5,
            alpha=0.7,
        )

    # Draw AiS-GNG nodes
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c="red",
        s=40,
        zorder=5,
        label=f"AiS-GNG ({len(nodes)} nodes)",
    )

    # Draw ring center
    ax.scatter([ring_center[0]], [ring_center[1]], c="blue", s=30, marker="+", zorder=6)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(
        f"AiS-GNG Tracking - Frame {frame}/{total_frames} "
        f"(+{n_ais_additions} AiS, -{n_utility_removals} util)"
    )
    ax.legend(loc="upper right", fontsize=8)


def run_tracking_experiment(
    orbit_center: tuple[float, float] = (0.5, 0.5),
    orbit_radius: float = 0.25,
    ring_r_inner: float = 0.08,
    ring_r_outer: float = 0.12,
    total_frames: int = 120,
    samples_per_frame: int = 50,
    output_gif: str = "aisgng_tracking.gif",
    seed: int = 42,
) -> None:
    """Run AiS-GNG tracking experiment.

    Args:
        orbit_center: Center of the circular orbit.
        orbit_radius: Radius of the orbit path.
        ring_r_inner: Inner radius of the ring.
        ring_r_outer: Outer radius of the ring.
        total_frames: Number of animation frames.
        samples_per_frame: Number of samples to train per frame.
        output_gif: Output GIF path.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    orbit_center = np.array(orbit_center)

    # Setup AiS-GNG with parameters tuned for tracking
    # Ring width is 0.04 (0.12-0.08), so theta_max should be around that
    params = AiSGNGParams(
        max_nodes=50,
        lambda_=20,       # Insert nodes more frequently for tracking
        eps_b=0.15,       # Higher learning rate for faster adaptation
        eps_n=0.01,
        alpha=0.5,
        beta=0.01,        # Higher decay to forget old positions
        chi=0.01,         # Utility decay rate
        max_age=30,       # Shorter edge age for faster adaptation
        utility_k=500.0,  # Lower threshold for faster cleanup
        kappa=5,          # More frequent utility checks
        # Add-if-Silent tolerances for tracking (smaller for finer structure)
        theta_ais_min=0.01,
        theta_ais_max=0.05,
    )
    gng = AiSGNG(n_dim=2, params=params, seed=seed)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    frames = []

    print(f"Running AiS-GNG tracking experiment ({total_frames} frames)...")
    print(f"Ring moves along orbit: center={orbit_center}, radius={orbit_radius}")
    print(f"AiS-GNG params: lambda={params.lambda_}, eps_b={params.eps_b}, max_age={params.max_age}")
    print(f"Add-if-Silent: theta_min={params.theta_ais_min}, theta_max={params.theta_ais_max}")

    for frame in range(total_frames):
        # Calculate ring center position on orbit
        angle = (frame / total_frames) * 2 * np.pi
        ring_center = orbit_center + orbit_radius * np.array([np.cos(angle), np.sin(angle)])

        # Generate samples from current ring position
        samples = generate_ring_samples(
            ring_center, ring_r_inner, ring_r_outer, samples_per_frame, rng
        )

        # Train AiS-GNG on current samples
        for sample in samples:
            gng.partial_fit(sample)

        # Get current AiS-GNG state
        nodes, edges = gng.get_graph()

        # Create frame
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
            gng.n_ais_additions,
            gng.n_utility_removals,
        )
        fig.canvas.draw()

        # Convert to PIL Image
        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        )
        frames.append(img.convert("RGB"))

        if (frame + 1) % 20 == 0:
            print(
                f"  Frame {frame + 1}/{total_frames}: {len(nodes)} nodes, {len(edges)} edges, "
                f"+{gng.n_ais_additions} AiS, -{gng.n_utility_removals} util"
            )

    plt.close(fig)

    # Save GIF
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
        )
        print(f"Saved: {output_gif}")

    print(f"Done! AiS additions: {gng.n_ais_additions}, Utility removals: {gng.n_utility_removals}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AiS-GNG tracking on moving ring")
    parser.add_argument("--orbit-radius", type=float, default=0.25, help="Orbit radius")
    parser.add_argument("--ring-inner", type=float, default=0.08, help="Ring inner radius")
    parser.add_argument("--ring-outer", type=float, default=0.12, help="Ring outer radius")
    parser.add_argument("--frames", type=int, default=120, help="Total frames")
    parser.add_argument("--samples", type=int, default=50, help="Samples per frame")
    parser.add_argument("--output", type=str, default="aisgng_tracking.gif", help="Output GIF")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--theta-min", type=float, default=0.01, help="AiS minimum tolerance")
    parser.add_argument("--theta-max", type=float, default=0.05, help="AiS maximum tolerance")

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
