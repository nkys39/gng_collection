"""Test DD-GNG with auto-detection only (no manual attention regions).

Demonstrates automatic attention detection via surface classification:
- PCA-based normal computation for each node
- Surface type classification (plane/edge/corner) based on eigenvalue ratios
- Stable corners automatically become attention regions

Usage:
    python test_ddgng_auto_detect.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "dd_gng" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "3d"))

from model import DynamicDensityGNG, DDGNGParams, SurfaceType
from sampler import sample_floor_and_wall


def draw_floor_wall_surfaces(ax, floor_size=0.8, wall_height=0.6) -> None:
    """Draw semi-transparent floor and wall surfaces."""
    offset = (1.0 - floor_size) / 2

    # Floor surface (XZ plane at y=0)
    floor_x = np.array([[offset, offset + floor_size], [offset, offset + floor_size]])
    floor_z = np.array([[offset, offset], [offset + floor_size, offset + floor_size]])
    floor_y = np.zeros_like(floor_x)
    ax.plot_surface(floor_x, floor_y, floor_z, alpha=0.15, color="lightblue")

    # Wall surface (XY plane at z=offset)
    wall_x = np.array([[offset, offset + floor_size], [offset, offset + floor_size]])
    wall_y = np.array([[0, 0], [wall_height, wall_height]])
    wall_z = np.full_like(wall_x, offset)
    ax.plot_surface(wall_x, wall_y, wall_z, alpha=0.15, color="lightgreen")


def get_surface_colors(surface_types):
    """Get colors based on surface type classification."""
    colors = []
    for st in surface_types:
        if st in (SurfaceType.STABLE_CORNER, SurfaceType.CORNER):
            colors.append("red")  # Corner
        elif st in (SurfaceType.STABLE_EDGE, SurfaceType.EDGE):
            colors.append("yellow")  # Edge
        elif st in (SurfaceType.STABLE_PLANE, SurfaceType.PLANE):
            colors.append("limegreen")  # Plane
        else:
            colors.append("gray")  # Unknown
    return colors


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list,
    iteration: int,
    surface_types: np.ndarray,
    auto_attention: np.ndarray,
    n_auto_attention: int,
    elev: float = 25,
    azim: float = 120,
) -> None:
    """Create a single frame showing surface classification and auto-attention."""
    ax.clear()

    # Draw surface guides
    draw_floor_wall_surfaces(ax)

    # Plot sample points
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c="skyblue", s=2, alpha=0.15,
    )

    # Plot edges
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "gray", linewidth=0.5, alpha=0.5,
        )

    # Plot nodes with surface type coloring
    colors = get_surface_colors(surface_types)

    for idx in range(len(nodes)):
        # Auto-attention nodes get triangle marker and larger size
        marker = "^" if auto_attention[idx] else "o"
        size = 80 if auto_attention[idx] else 30
        ax.scatter(
            [nodes[idx, 0]], [nodes[idx, 1]], [nodes[idx, 2]],
            c=[colors[idx]], s=size, marker=marker,
            edgecolors="black", linewidths=0.3, zorder=5,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (depth)")
    ax.set_zlabel("Z (height)")
    ax.set_title(
        f"DD-GNG Auto-Detection - Iter {iteration}\n"
        f"{len(nodes)} nodes, {n_auto_attention} auto-detected corners"
    )
    ax.view_init(elev=elev, azim=azim)


def run_experiment(
    n_samples: int = 2000,
    n_iterations: int = 8000,
    gif_frames: int = 80,
    output_gif: str = "ddgng_auto_detect_growth.gif",
    output_final: str = "ddgng_auto_detect_final.png",
    seed: int = 42,
) -> None:
    """Run DD-GNG with auto-detection only.

    Args:
        n_samples: Number of points to sample.
        n_iterations: Number of training iterations.
        gif_frames: Number of frames in output GIF.
        output_gif: Output GIF path.
        output_final: Output final image path.
        seed: Random seed.
    """
    np.random.seed(seed)

    # Sample points
    print(f"Sampling {n_samples} points from floor and wall...")
    points = sample_floor_and_wall(n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # DD-GNG with AUTO-DETECTION ONLY (no manual attention regions)
    params = DDGNGParams(
        max_nodes=150,
        lambda_=100,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
        chi=0.005,
        max_age=100,
        utility_k=1000.0,
        kappa=10,
        # DD-GNG specific
        strength_power=4,
        strength_scale=4.0,
        use_strength_learning=True,
        use_strength_insertion=True,
        # AUTO-DETECTION ENABLED
        auto_detect_attention=True,
        stability_threshold=16,
        plane_stability_threshold=8,
        corner_strength=5.0,
        plane_ev_ratio=0.01,
        edge_ev_ratio=0.1,
        surface_update_interval=10,
    )
    model = DynamicDensityGNG(n_dim=3, params=params, seed=seed)

    # NO manual attention regions - relying entirely on auto-detection
    print("Auto-detection only mode:")
    print("  - NO manual attention regions")
    print("  - Auto-detection of stable corners enabled")
    print("  - Surface classification (plane/edge/corner)")

    # Collect frames
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)
    frame_count = [0]

    azim_start = 120
    azim_end = 180

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    def callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            surface_types = m.get_node_surface_types()
            auto_attention = m.get_node_auto_attention()

            progress = frame_count[0] / gif_frames
            azim = azim_start + (azim_end - azim_start) * progress

            create_frame(
                ax, points, nodes, edges, iteration,
                surface_types, auto_attention,
                m.n_auto_attention,
                azim=azim,
            )
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))

            # Count surface types
            n_planes = sum(1 for st in surface_types if st in (SurfaceType.PLANE, SurfaceType.STABLE_PLANE))
            n_edges = sum(1 for st in surface_types if st in (SurfaceType.EDGE, SurfaceType.STABLE_EDGE))
            n_corners = sum(1 for st in surface_types if st in (SurfaceType.CORNER, SurfaceType.STABLE_CORNER))

            print(
                f"Iter {iteration}: {len(nodes)} nodes | "
                f"Surface: P={n_planes} E={n_edges} C={n_corners} | "
                f"Auto-attention: {m.n_auto_attention}"
            )
            frame_count[0] += 1

    # Train
    print(f"\nTraining DD-GNG for {n_iterations} iterations...")
    model.train(points, n_iterations=n_iterations, callback=callback)

    # Final statistics
    nodes, edges = model.get_graph()
    surface_types = model.get_node_surface_types()
    auto_attention = model.get_node_auto_attention()
    strengths = model.get_node_strengths()

    print(f"\nFinal results:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Auto-detected corners: {model.n_auto_attention}")
    print(f"  Max strength: {strengths.max():.2f}")
    print(f"  Mean strength: {strengths.mean():.2f}")

    plt.close(fig)

    # Create final figure with multiple views
    fig_final = plt.figure(figsize=(18, 10), facecolor="white")

    # Left: 3D perspective with surface type coloring
    ax1 = fig_final.add_subplot(1, 2, 1, projection="3d")
    draw_floor_wall_surfaces(ax1)

    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.15)

    for i, j in edges:
        ax1.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "gray", linewidth=0.5, alpha=0.5,
        )

    colors = get_surface_colors(surface_types)
    for idx in range(len(nodes)):
        marker = "^" if auto_attention[idx] else "o"
        size = 100 if auto_attention[idx] else 40
        ax1.scatter(
            [nodes[idx, 0]], [nodes[idx, 1]], [nodes[idx, 2]],
            c=[colors[idx]], s=size, marker=marker,
            edgecolors="black", linewidths=0.3, zorder=5,
        )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y (depth)")
    ax1.set_zlabel("Z (height)")
    ax1.set_title(
        f"DD-GNG Auto-Detection Only - 3D View\n"
        f"{len(nodes)} nodes, {model.n_auto_attention} auto-detected\n"
        f"Green=Plane, Yellow=Edge, Red=Corner, Triangle=Auto-attention"
    )
    ax1.view_init(elev=25, azim=120)

    # Right: 2D side view (YZ plane) with strength coloring
    ax2 = fig_final.add_subplot(1, 2, 2)

    ax2.scatter(points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.15)

    for i, j in edges:
        ax2.plot(
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "gray", linewidth=0.5, alpha=0.5,
        )

    scatter = ax2.scatter(
        nodes[:, 1], nodes[:, 2],
        c=strengths, cmap="YlOrRd",
        s=50, edgecolors="black", linewidths=0.3,
        vmin=1.0, vmax=max(strengths.max(), 6.0),
        zorder=5,
    )
    plt.colorbar(scatter, ax=ax2, label="Strength")

    ax2.set_xlim(1, 0)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Y (depth)")
    ax2.set_ylabel("Z (height)")
    ax2.set_aspect("equal")
    ax2.set_title(
        f"DD-GNG Auto-Detection Only - Side View (YZ)\n"
        f"Strength colored, max={strengths.max():.1f}"
    )

    plt.tight_layout()
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    # Save GIF
    if frames:
        frames.extend([frames[-1]] * 15)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close()
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test DD-GNG with auto-detection only (no manual attention regions)"
    )
    parser.add_argument("-n", "--n-samples", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--output-gif", type=str, default="ddgng_auto_detect_growth.gif")
    parser.add_argument("--output-final", type=str, default="ddgng_auto_detect_final.png")
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
