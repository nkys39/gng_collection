"""Test DD-GNG algorithm on 3D floor and wall data with GIF visualization.

Demonstrates DD-GNG dynamic density control in 3D:
- Sets attention region at the edge where floor meets wall
- Higher node density along the edge boundary

Usage:
    python test_ddgng_floor_wall.py
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

from model import DynamicDensityGNG, DDGNGParams
from sampler import sample_floor_and_wall


def draw_floor_wall_surfaces(ax, floor_size=0.8, wall_height=0.6) -> None:
    """Draw semi-transparent floor and wall surfaces."""
    offset = (1.0 - floor_size) / 2

    # Floor surface (XZ plane at y=0)
    floor_x = np.array([[offset, offset + floor_size], [offset, offset + floor_size]])
    floor_z = np.array([[offset, offset], [offset + floor_size, offset + floor_size]])
    floor_y = np.zeros_like(floor_x)
    ax.plot_surface(floor_x, floor_y, floor_z, alpha=0.2, color="lightblue")

    # Wall surface (XY plane at z=offset)
    wall_x = np.array([[offset, offset + floor_size], [offset, offset + floor_size]])
    wall_y = np.array([[0, 0], [wall_height, wall_height]])
    wall_z = np.full_like(wall_x, offset)
    ax.plot_surface(wall_x, wall_y, wall_z, alpha=0.2, color="lightgreen")


def draw_attention_region_3d(ax, center, size, color="orange"):
    """Draw attention region as a box outline in 3D."""
    cx, cy, cz = center
    sx, sy, sz = size

    # Box corners
    x_min, x_max = cx - sx, cx + sx
    y_min, y_max = cy - sy, cy + sy
    z_min, z_max = cz - sz, cz + sz

    # Draw box edges
    # Bottom face
    ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_min, x_max], [y_max, y_max], [z_min, z_min], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_min, x_min], [y_min, y_max], [z_min, z_min], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_max, x_max], [y_min, y_max], [z_min, z_min], color=color, linestyle="--", alpha=0.7)

    # Top face
    ax.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_min, x_max], [y_max, y_max], [z_max, z_max], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_min, x_min], [y_min, y_max], [z_max, z_max], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_max, x_max], [y_min, y_max], [z_max, z_max], color=color, linestyle="--", alpha=0.7)

    # Vertical edges
    ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], color=color, linestyle="--", alpha=0.7)
    ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], color=color, linestyle="--", alpha=0.7)


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
    strengths: np.ndarray | None = None,
    attention_regions: list | None = None,
    elev: float = 25,
    azim: float = 120,
) -> None:
    """Create a single frame for 3D visualization."""
    ax.clear()

    # Draw surface guides
    draw_floor_wall_surfaces(ax)

    # Draw attention regions
    if attention_regions:
        for region in attention_regions:
            draw_attention_region_3d(ax, region.center, region.size)

    # Plot sample points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="skyblue",
        s=2,
        alpha=0.2,
        label="Data",
    )

    # Plot edges
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "r-",
            linewidth=1.0,
            alpha=0.7,
        )

    # Plot nodes with strength-based coloring
    if strengths is not None:
        scatter = ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            c=strengths,
            cmap="YlOrRd",
            s=30,
            zorder=5,
            edgecolors="black",
            linewidths=0.3,
        )
    else:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            c="red",
            s=30,
            zorder=5,
            label="Nodes",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (depth)")
    ax.set_zlabel("Z (height)")
    ax.set_title(f"DD-GNG 3D Floor+Wall - Iter {iteration} ({len(nodes)} nodes)")
    ax.legend(loc="upper right")
    ax.view_init(elev=elev, azim=azim)


def run_experiment(
    n_samples: int = 2000,
    n_iterations: int = 8000,
    gif_frames: int = 100,
    output_gif: str = "ddgng_floor_wall_growth.gif",
    output_final: str = "ddgng_floor_wall_final.png",
    seed: int = 42,
) -> None:
    """Run DD-GNG 3D experiment with visualization.

    Args:
        n_samples: Number of points to sample.
        n_iterations: Number of DD-GNG training iterations.
        gif_frames: Number of frames in output GIF.
        output_gif: Output GIF path.
        output_final: Output final image path.
        seed: Random seed.
    """
    np.random.seed(seed)

    # Sample points from floor and wall
    print(f"Sampling {n_samples} points from floor and wall...")
    points = sample_floor_and_wall(n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Setup DD-GNG with parameters tuned for 3D
    params = DDGNGParams(
        max_nodes=150,  # More nodes for 3D surfaces
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
    )
    model = DynamicDensityGNG(n_dim=3, params=params, seed=seed)

    # Add attention region at the edge where floor meets wall
    # The edge is at z=0.1 (floor level), along y=0 (wall base), x spans the surface
    # We create a strip along this edge
    model.add_attention_region(
        center=[0.5, 0.0, 0.1],  # Center of the edge
        size=[0.4, 0.08, 0.08],  # Narrow strip along the edge
        strength=5.0,
    )

    print("Added attention region at floor-wall edge (center=[0.5, 0.0, 0.1])")

    # Collect frames for GIF
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)
    frame_count = [0]

    # Rotation animation settings
    azim_start = 120
    azim_end = 180

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    def callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            strengths = m.get_node_strengths()

            # Calculate azimuth for rotation animation
            progress = frame_count[0] / gif_frames
            azim = azim_start + (azim_end - azim_start) * progress

            create_frame(
                ax,
                points,
                nodes,
                edges,
                iteration,
                strengths=strengths,
                attention_regions=m.attention_regions,
                azim=azim,
            )
            fig.canvas.draw()

            # Convert to PIL Image
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(
                f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges, "
                f"max_strength={strengths.max():.1f}"
            )
            frame_count[0] += 1

    # Train
    print(f"Training DD-GNG for {n_iterations} iterations...")
    print(
        f"Parameters: max_nodes={params.max_nodes}, lambda={params.lambda_}, "
        f"eps_b={params.eps_b}, max_age={params.max_age}"
    )
    model.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame with dual view
    nodes, edges = model.get_graph()
    strengths = model.get_node_strengths()
    plt.close(fig)

    fig_final = plt.figure(figsize=(16, 8), facecolor="white")

    # Left: 3D perspective view
    ax_3d = fig_final.add_subplot(1, 2, 1, projection="3d")
    create_frame(
        ax_3d,
        points,
        nodes,
        edges,
        n_iterations,
        strengths=strengths,
        attention_regions=model.attention_regions,
        elev=25,
        azim=azim_start,
    )
    ax_3d.set_title(f"DD-GNG 3D - Perspective View ({len(nodes)} nodes)")

    # Right: Pure 2D side view (YZ plane)
    ax_2d = fig_final.add_subplot(1, 2, 2)

    # Plot data points (Y horizontal, Z vertical)
    ax_2d.scatter(
        points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2, label="Data"
    )

    # Plot edges (Y horizontal, Z vertical)
    for i, j in edges:
        ax_2d.plot(
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "r-",
            linewidth=1.0,
            alpha=0.7,
        )

    # Plot nodes with strength-based coloring (Y horizontal, Z vertical)
    scatter = ax_2d.scatter(
        nodes[:, 1],
        nodes[:, 2],
        c=strengths,
        cmap="YlOrRd",
        s=30,
        zorder=5,
        edgecolors="black",
        linewidths=0.3,
        label="Nodes",
    )
    plt.colorbar(scatter, ax=ax_2d, label="Strength")

    # Draw attention region in 2D
    for region in model.attention_regions:
        rect = plt.Rectangle(
            (region.center[1] - region.size[1], region.center[2] - region.size[2]),
            region.size[1] * 2,
            region.size[2] * 2,
            fill=False,
            edgecolor="orange",
            linewidth=2,
            linestyle="--",
        )
        ax_2d.add_patch(rect)

    ax_2d.set_xlim(1, 0)  # Y axis: left = positive
    ax_2d.set_ylim(0, 1)  # Z axis: up = positive
    ax_2d.set_xlabel("Y (depth)")
    ax_2d.set_ylabel("Z (height)")
    ax_2d.set_aspect("equal")
    ax_2d.set_title("DD-GNG 3D - Side View (YZ plane)")
    ax_2d.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    # Save GIF
    if frames:
        # Add extra copies of final frame
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

    parser = argparse.ArgumentParser(
        description="Test DD-GNG on 3D floor and wall data"
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=2000, help="Number of samples"
    )
    parser.add_argument(
        "--iterations", type=int, default=8000, help="Number of training iterations"
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of GIF frames")
    parser.add_argument(
        "--output-gif",
        type=str,
        default="ddgng_floor_wall_growth.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--output-final",
        type=str,
        default="ddgng_floor_wall_final.png",
        help="Output final image path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_experiment(
        n_samples=args.n_samples,
        n_iterations=args.iterations,
        gif_frames=args.frames,
        output_gif=args.output_gif,
        output_final=args.output_final,
        seed=args.seed,
    )
