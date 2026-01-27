"""Test GNG-DT Robot algorithm on 3D floor and wall data with traversability analysis.

Creates a visualization of GNG-DT Robot learning on a floor-wall L-shaped surface,
showing traversability classification (floor=traversable, wall=non-traversable).

GNG-DT Robot Key Features:
    - All GNG-DT features (multiple topologies)
    - Traversability analysis based on surface inclination
    - Contour detection for traversable region boundaries
    - Traversability edges (connects same traversability nodes)

Usage:
    python test_gngdt_robot_floor_wall.py
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

from model_robot import GrowingNeuralGasDTRobot, GNGDTRobotParams
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
    trav_edges: list[tuple[int, int]],
    traversability: np.ndarray,
    iteration: int,
    elev: float = 25,
    azim: float = 120,
) -> None:
    """Create a single frame for 3D visualization with traversability."""
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

    # Plot position edges (gray, thin)
    for i, j in pos_edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            color="gray",
            linewidth=0.5,
            alpha=0.3,
        )

    # Plot traversability edges (green for traversable connections)
    for i, j in trav_edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "g-",
            linewidth=1.5,
            alpha=0.7,
        )

    # Plot nodes colored by traversability
    # Green = traversable, Red = non-traversable
    colors = ["green" if t == 1 else "red" for t in traversability]
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        c=colors,
        s=40,
        zorder=5,
        edgecolors="black",
        linewidths=0.5,
    )

    n_traversable = int(np.sum(traversability))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (depth)")
    ax.set_zlabel("Z (height)")
    ax.set_title(
        f"GNG-DT Robot - Iter {iteration}\n"
        f"({len(nodes)} nodes, {n_traversable} traversable, {len(trav_edges)} trav-edges)"
    )
    ax.view_init(elev=elev, azim=azim)


def run_experiment(
    n_samples: int = 2000,
    n_iterations: int = 8000,
    gif_frames: int = 100,
    output_gif: str = "gngdt_robot_floor_wall_growth.gif",
    output_final: str = "gngdt_robot_floor_wall_final.png",
    seed: int = 42,
) -> None:
    """Run GNG-DT Robot 3D experiment with traversability visualization.

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

    # Setup GNG-DT Robot
    params = GNGDTRobotParams(
        max_nodes=150,
        lambda_=100,
        eps_b=0.05,
        eps_n=0.0005,
        alpha=0.5,
        beta=0.0005,
        max_age=88,
        tau_normal=0.95,
        max_angle=20.0,  # 20 degrees from horizontal is traversable
        s1thv=1.0,
    )
    gng = GrowingNeuralGasDTRobot(params=params, seed=seed)

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
            nodes, pos_edges, _, _, trav_edges = model.get_multi_graph()
            traversability = model.get_traversability()

            # Calculate azimuth for rotation animation
            progress = frame_count[0] / gif_frames
            azim = azim_start + (azim_end - azim_start) * progress

            create_frame(
                ax,
                points,
                nodes,
                pos_edges,
                trav_edges,
                traversability,
                iteration,
                azim=azim,
            )
            fig.canvas.draw()

            # Convert to PIL Image
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))

            n_trav = int(np.sum(traversability))
            print(
                f"Iteration {iteration}: {len(nodes)} nodes, "
                f"{n_trav} traversable, {len(trav_edges)} trav-edges"
            )
            frame_count[0] += 1

    # Train
    print(f"Training GNG-DT Robot for {n_iterations} iterations...")
    print(
        f"Parameters: max_nodes={params.max_nodes}, max_angle={params.max_angle}"
    )
    gng.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame with multi-view
    nodes, pos_edges, _, normal_edges, trav_edges = gng.get_multi_graph()
    traversability = gng.get_traversability()
    contour = gng.get_contour()
    degree = gng.get_degree()
    normals = gng.get_node_normals()
    plt.close(fig)

    fig_final = plt.figure(figsize=(18, 9), facecolor="white")

    # Left: 3D view with traversability coloring
    ax_3d = fig_final.add_subplot(1, 3, 1, projection="3d")
    draw_floor_wall_surfaces(ax_3d)
    ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2)

    # Position edges (gray)
    for i, j in pos_edges:
        ax_3d.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            color="gray",
            linewidth=0.5,
            alpha=0.3,
        )

    # Traversability edges (green)
    for i, j in trav_edges:
        ax_3d.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "g-",
            linewidth=1.2,
            alpha=0.7,
        )

    # Nodes colored by traversability
    colors = ["green" if t == 1 else "red" for t in traversability]
    ax_3d.scatter(
        nodes[:, 0], nodes[:, 1], nodes[:, 2],
        c=colors, s=40, zorder=5,
        edgecolors="black", linewidths=0.5
    )

    ax_3d.set_xlim(0, 1)
    ax_3d.set_ylim(0, 1)
    ax_3d.set_zlim(0, 1)
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    n_trav = int(np.sum(traversability))
    ax_3d.set_title(
        f"Traversability\nGreen={n_trav} traversable, Red={len(nodes)-n_trav} non-trav"
    )
    ax_3d.view_init(elev=25, azim=azim_start)

    # Middle: Contour visualization
    ax_contour = fig_final.add_subplot(1, 3, 2, projection="3d")
    draw_floor_wall_surfaces(ax_contour)
    ax_contour.scatter(
        points[:, 0], points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2
    )

    # Traversability edges
    for i, j in trav_edges:
        ax_contour.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "g-",
            linewidth=0.8,
            alpha=0.5,
        )

    # Nodes: contour nodes in orange, others by traversability
    node_colors = []
    for i, (t, c) in enumerate(zip(traversability, contour)):
        if c == 1:
            node_colors.append("orange")
        elif t == 1:
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightcoral")

    ax_contour.scatter(
        nodes[:, 0], nodes[:, 1], nodes[:, 2],
        c=node_colors, s=50, zorder=5,
        edgecolors="black", linewidths=0.5
    )

    ax_contour.set_xlim(0, 1)
    ax_contour.set_ylim(0, 1)
    ax_contour.set_zlim(0, 1)
    ax_contour.set_xlabel("X")
    ax_contour.set_ylabel("Y")
    ax_contour.set_zlabel("Z")
    n_contour = int(np.sum(contour))
    ax_contour.set_title(
        f"Contour Detection\nOrange={n_contour} contour nodes"
    )
    ax_contour.view_init(elev=25, azim=azim_start)

    # Right: 2D side view (YZ plane)
    ax_2d = fig_final.add_subplot(1, 3, 3)
    ax_2d.scatter(points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2, label="Data")

    # Traversability edges
    for i, j in trav_edges:
        ax_2d.plot(
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            "g-",
            linewidth=1.0,
            alpha=0.6,
        )

    # Nodes colored by traversability
    for i, (y, z, t, c) in enumerate(zip(nodes[:, 1], nodes[:, 2], traversability, contour)):
        if c == 1:
            color = "orange"
            size = 60
        elif t == 1:
            color = "green"
            size = 40
        else:
            color = "red"
            size = 40
        ax_2d.scatter(y, z, c=color, s=size, zorder=5, edgecolors="black", linewidths=0.5)

    ax_2d.set_xlim(1, 0)
    ax_2d.set_ylim(0, 1)
    ax_2d.set_xlabel("Y (depth)")
    ax_2d.set_ylabel("Z (height)")
    ax_2d.set_aspect("equal")
    ax_2d.set_title("Side View (YZ)\nGreen=Traversable, Red=Non-trav, Orange=Contour")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="Traversable"),
        Patch(facecolor="red", edgecolor="black", label="Non-traversable"),
        Patch(facecolor="orange", edgecolor="black", label="Contour"),
    ]
    ax_2d.legend(handles=legend_elements, loc="upper left")

    plt.suptitle(
        f"GNG-DT Robot: {len(nodes)} nodes | "
        f"Traversable: {n_trav} | Contour: {n_contour} | "
        f"Trav-edges: {len(trav_edges)}",
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
        description="Test GNG-DT Robot on 3D floor and wall data"
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
        default="gngdt_robot_floor_wall_growth.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--output-final",
        type=str,
        default="gngdt_robot_floor_wall_final.png",
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
