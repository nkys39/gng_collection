"""Test GNG-DT algorithm on 3D floor and wall data with multi-topology visualization.

Creates a visualization of GNG-DT learning on a floor-wall L-shaped surface,
showing both position-based and normal-based edge topologies.

GNG-DT Key Features:
    - Multiple independent edge topologies
    - Position edges (red): Standard GNG edges
    - Normal edges (blue): Edges between nodes with similar normals
    - Floor and wall should have separate normal-edge clusters

Usage:
    python test_gngdt_floor_wall.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "gng_dt" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "3d"))

from model import GrowingNeuralGasDT, GNGDTParams
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


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    pos_edges: list[tuple[int, int]],
    normal_edges: list[tuple[int, int]],
    normals: np.ndarray,
    iteration: int,
    show_normals: bool = True,
    elev: float = 25,
    azim: float = 120,
) -> None:
    """Create a single frame for 3D visualization with multi-topology edges."""
    ax.clear()

    # Draw surface guides
    draw_floor_wall_surfaces(ax)

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

    # Plot position edges (red)
    for i, j in pos_edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "r-",
            linewidth=0.8,
            alpha=0.5,
        )

    # Plot normal edges (blue) - edges between nodes with similar normals
    for i, j in normal_edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "b-",
            linewidth=1.2,
            alpha=0.7,
        )

    # Plot nodes
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        c="red",
        s=30,
        zorder=5,
        label="Nodes",
    )

    # Optionally show normal vectors
    if show_normals and len(normals) > 0:
        scale = 0.05
        ax.quiver(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            normals[:, 0] * scale,
            normals[:, 1] * scale,
            normals[:, 2] * scale,
            color="green",
            alpha=0.6,
            arrow_length_ratio=0.3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (depth)")
    ax.set_zlabel("Z (height)")
    ax.set_title(
        f"GNG-DT Floor+Wall - Iter {iteration}\n"
        f"({len(nodes)} nodes, {len(pos_edges)} pos-edges, {len(normal_edges)} normal-edges)"
    )
    ax.legend(loc="upper right")
    ax.view_init(elev=elev, azim=azim)


def run_experiment(
    n_samples: int = 2000,
    n_iterations: int = 8000,
    gif_frames: int = 100,
    output_gif: str = "gngdt_floor_wall_growth.gif",
    output_final: str = "gngdt_floor_wall_final.png",
    seed: int = 42,
) -> None:
    """Run GNG-DT 3D experiment with multi-topology visualization.

    Args:
        n_samples: Number of points to sample.
        n_iterations: Number of GNG training iterations.
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

    # Setup GNG-DT with parameters tuned for 3D and normal detection
    params = GNGDTParams(
        max_nodes=150,  # More nodes for 3D surfaces
        lambda_=100,  # Node insertion interval
        eps_b=0.1,  # Winner learning rate
        eps_n=0.01,  # Neighbor learning rate
        alpha=0.5,  # Error decay on split
        beta=0.005,  # Global error decay
        max_age=100,  # Maximum edge age
        tau_normal=0.90,  # Normal similarity threshold (cos(25°) ≈ 0.90)
        pca_min_neighbors=3,  # Minimum neighbors for PCA
    )
    gng = GrowingNeuralGasDT(params=params, seed=seed)

    # Collect frames for GIF
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)
    frame_count = [0]

    # Rotation animation settings
    azim_start = 120
    azim_end = 180

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, pos_edges, _, normal_edges = model.get_multi_graph()
            normals = model.get_node_normals()

            # Calculate azimuth for rotation animation
            progress = frame_count[0] / gif_frames
            azim = azim_start + (azim_end - azim_start) * progress

            create_frame(
                ax,
                points,
                nodes,
                pos_edges,
                normal_edges,
                normals,
                iteration,
                show_normals=False,  # Hide normals in animation for clarity
                azim=azim,
            )
            fig.canvas.draw()

            # Convert to PIL Image
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(
                f"Iteration {iteration}: {len(nodes)} nodes, "
                f"{len(pos_edges)} pos-edges, {len(normal_edges)} normal-edges"
            )
            frame_count[0] += 1

    # Train
    print(f"Training GNG-DT for {n_iterations} iterations...")
    print(
        f"Parameters: max_nodes={params.max_nodes}, lambda={params.lambda_}, "
        f"tau_normal={params.tau_normal}"
    )
    gng.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame with multi-view
    nodes, pos_edges, _, normal_edges = gng.get_multi_graph()
    normals = gng.get_node_normals()
    plt.close(fig)

    fig_final = plt.figure(figsize=(18, 9), facecolor="white")

    # Left: 3D perspective with position edges
    ax_3d = fig_final.add_subplot(1, 3, 1, projection="3d")
    draw_floor_wall_surfaces(ax_3d)
    ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2)
    for i, j in pos_edges:
        ax_3d.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "r-",
            linewidth=0.8,
            alpha=0.6,
        )
    ax_3d.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="red", s=30, zorder=5)
    ax_3d.set_xlim(0, 1)
    ax_3d.set_ylim(0, 1)
    ax_3d.set_zlim(0, 1)
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title(f"Position Topology\n({len(pos_edges)} edges)")
    ax_3d.view_init(elev=25, azim=azim_start)

    # Middle: 3D perspective with normal edges
    ax_normal = fig_final.add_subplot(1, 3, 2, projection="3d")
    draw_floor_wall_surfaces(ax_normal)
    ax_normal.scatter(
        points[:, 0], points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2
    )
    for i, j in normal_edges:
        ax_normal.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "b-",
            linewidth=1.0,
            alpha=0.7,
        )
    ax_normal.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="blue", s=30, zorder=5)
    # Show normal vectors
    scale = 0.06
    ax_normal.quiver(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        normals[:, 0] * scale,
        normals[:, 1] * scale,
        normals[:, 2] * scale,
        color="green",
        alpha=0.8,
        arrow_length_ratio=0.3,
    )
    ax_normal.set_xlim(0, 1)
    ax_normal.set_ylim(0, 1)
    ax_normal.set_zlim(0, 1)
    ax_normal.set_xlabel("X")
    ax_normal.set_ylabel("Y")
    ax_normal.set_zlabel("Z")
    ax_normal.set_title(f"Normal Topology\n({len(normal_edges)} edges, green=normals)")
    ax_normal.view_init(elev=25, azim=azim_start)

    # Right: 2D side view (YZ plane) with both topologies
    ax_2d = fig_final.add_subplot(1, 3, 3)
    ax_2d.scatter(points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2, label="Data")

    # Position edges
    for i, j in pos_edges:
        ax_2d.plot(
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "r-",
            linewidth=0.5,
            alpha=0.4,
        )

    # Normal edges
    for i, j in normal_edges:
        ax_2d.plot(
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "b-",
            linewidth=1.0,
            alpha=0.7,
        )

    ax_2d.scatter(nodes[:, 1], nodes[:, 2], c="red", s=30, zorder=5, label="Nodes")
    ax_2d.set_xlim(1, 0)
    ax_2d.set_ylim(0, 1)
    ax_2d.set_xlabel("Y (depth)")
    ax_2d.set_ylabel("Z (height)")
    ax_2d.set_aspect("equal")
    ax_2d.set_title("Side View (YZ)\nRed=Position, Blue=Normal")
    ax_2d.legend(loc="upper left")

    plt.suptitle(
        f"GNG-DT: {len(nodes)} nodes | "
        f"Position: {len(pos_edges)} edges | Normal: {len(normal_edges)} edges",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    # Save GIF
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

    plt.close(fig_final)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test GNG-DT on 3D floor and wall data"
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
        default="gngdt_floor_wall_growth.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--output-final",
        type=str,
        default="gngdt_floor_wall_final.png",
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
