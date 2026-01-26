"""Test GNG algorithm on 3D floor and wall data with GIF visualization.

Creates a visualization of GNG learning on a floor-wall L-shaped surface.

Usage:
    python test_gng_floor_wall.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "gng" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "3d"))

from model import GrowingNeuralGas, GNGParams
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
    edges: list[tuple[int, int]],
    iteration: int,
    elev: float = 25,
    azim: float = 120,
) -> None:
    """Create a single frame for 3D visualization."""
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

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (height)")
    ax.set_zlabel("Z")
    ax.set_title(f"GNG 3D Floor+Wall - Iter {iteration} ({len(nodes)} nodes)")
    ax.legend(loc="upper right")
    ax.view_init(elev=elev, azim=azim)


def run_experiment(
    n_samples: int = 2000,
    n_iterations: int = 8000,
    gif_frames: int = 100,
    output_gif: str = "gng_floor_wall_growth.gif",
    output_final: str = "gng_floor_wall_final.png",
    seed: int = 42,
) -> None:
    """Run GNG 3D experiment with visualization.

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

    # Setup GNG with parameters tuned for 3D
    params = GNGParams(
        max_nodes=150,  # More nodes for 3D surfaces
        lambda_=100,  # Node insertion interval
        eps_b=0.1,  # Winner learning rate
        eps_n=0.01,  # Neighbor learning rate
        alpha=0.5,  # Error decay on split
        beta=0.005,  # Global error decay
        max_age=100,  # Maximum edge age
    )
    gng = GrowingNeuralGas(n_dim=3, params=params, seed=seed)

    # Collect frames for GIF
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)
    frame_count = [0]  # Use list to allow modification in nested function

    # Rotation animation settings
    azim_start = 120  # Wall at back
    azim_end = 180  # Rotate 60 degrees during animation

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()

            # Calculate azimuth for rotation animation
            progress = frame_count[0] / gif_frames
            azim = azim_start + (azim_end - azim_start) * progress

            create_frame(ax, points, nodes, edges, iteration, azim=azim)
            fig.canvas.draw()

            # Convert to PIL Image
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges")
            frame_count[0] += 1

    # Train
    print(f"Training GNG for {n_iterations} iterations...")
    print(
        f"Parameters: max_nodes={params.max_nodes}, lambda={params.lambda_}, "
        f"eps_b={params.eps_b}, max_age={params.max_age}"
    )
    gng.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame with dual view (initial view + side view from YZ plane)
    nodes, edges = gng.get_graph()
    plt.close(fig)

    fig_final, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="white",
                                    subplot_kw={"projection": "3d"})

    # Left: Initial view (azim=120)
    create_frame(axes[0], points, nodes, edges, n_iterations, elev=25, azim=azim_start)
    axes[0].set_title(f"GNG 3D - Perspective View ({len(nodes)} nodes)")

    # Right: Side view from YZ plane (looking along X axis)
    create_frame(axes[1], points, nodes, edges, n_iterations, elev=0, azim=0)
    axes[1].set_title(f"GNG 3D - Side View (YZ plane)")

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
            duration=100,  # ms per frame
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GNG on 3D floor and wall data")
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
        default="gng_floor_wall_growth.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--output-final",
        type=str,
        default="gng_floor_wall_final.png",
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
