"""Test GCS algorithm on 3D floor and wall data with GIF visualization.

Usage:
    python test_gcs_floor_wall.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "gcs" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "3d"))

from model import GrowingCellStructures, GCSParams
from sampler import sample_floor_and_wall


def draw_floor_wall_surfaces(ax, floor_size=0.8, wall_height=0.6) -> None:
    offset = (1.0 - floor_size) / 2
    floor_x = np.array([[offset, offset + floor_size], [offset, offset + floor_size]])
    floor_z = np.array([[offset, offset], [offset + floor_size, offset + floor_size]])
    floor_y = np.zeros_like(floor_x)
    ax.plot_surface(floor_x, floor_y, floor_z, alpha=0.2, color="lightblue")
    wall_x = np.array([[offset, offset + floor_size], [offset, offset + floor_size]])
    wall_y = np.array([[0, 0], [wall_height, wall_height]])
    wall_z = np.full_like(wall_x, offset)
    ax.plot_surface(wall_x, wall_y, wall_z, alpha=0.2, color="lightgreen")


def create_frame(ax, points, nodes, edges, iteration, elev=25, azim=120):
    ax.clear()
    draw_floor_wall_surfaces(ax)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2, label="Data")
    for i, j in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], [nodes[i, 2], nodes[j, 2]], "r-", linewidth=1.0, alpha=0.7)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="red", s=30, zorder=5, label="Nodes")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xlabel("X"); ax.set_ylabel("Y (depth)"); ax.set_zlabel("Z (height)")
    ax.set_title(f"GCS 3D Floor+Wall - Iter {iteration} ({len(nodes)} nodes)")
    ax.legend(loc="upper right")
    ax.view_init(elev=elev, azim=azim)


def run_experiment(n_samples=2000, n_iterations=8000, gif_frames=100,
                   output_gif="gcs_floor_wall_growth.gif",
                   output_final="gcs_floor_wall_final.png", seed=42):
    np.random.seed(seed)
    print(f"Sampling {n_samples} points from floor and wall...")
    points = sample_floor_and_wall(n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    params = GCSParams(
        max_nodes=150, lambda_=100, eps_b=0.1, eps_n=0.01,
        alpha=0.5, beta=0.005,
    )
    gcs = GrowingCellStructures(n_dim=3, params=params, seed=seed)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)
    frame_count = [0]
    azim_start, azim_end = 120, 180

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()
            progress = frame_count[0] / gif_frames
            azim = azim_start + (azim_end - azim_start) * progress
            create_frame(ax, points, nodes, edges, iteration, azim=azim)
            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges")
            frame_count[0] += 1

    print(f"Training GCS for {n_iterations} iterations...")
    gcs.train(points, n_iterations=n_iterations, callback=callback)

    nodes, edges = gcs.get_graph()
    plt.close(fig)

    fig_final = plt.figure(figsize=(16, 8), facecolor="white")
    ax_3d = fig_final.add_subplot(1, 2, 1, projection="3d")
    create_frame(ax_3d, points, nodes, edges, n_iterations, elev=25, azim=azim_start)
    ax_3d.set_title(f"GCS 3D - Perspective View ({len(nodes)} nodes)")

    ax_2d = fig_final.add_subplot(1, 2, 2)
    ax_2d.scatter(points[:, 1], points[:, 2], c="skyblue", s=2, alpha=0.2, label="Data")
    for i, j in edges:
        ax_2d.plot([nodes[i, 1], nodes[j, 1]], [nodes[i, 2], nodes[j, 2]], "r-", linewidth=1.0, alpha=0.7)
    ax_2d.scatter(nodes[:, 1], nodes[:, 2], c="red", s=30, zorder=5, label="Nodes")
    ax_2d.set_xlim(1, 0); ax_2d.set_ylim(0, 1)
    ax_2d.set_xlabel("Y (depth)"); ax_2d.set_ylabel("Z (height)")
    ax_2d.set_aspect("equal")
    ax_2d.set_title(f"GCS 3D - Side View (YZ plane)")
    ax_2d.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    if frames:
        frames.extend([frames[-1]] * 10)
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-samples", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output-gif", type=str, default="gcs_floor_wall_growth.gif")
    parser.add_argument("--output-final", type=str, default="gcs_floor_wall_final.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.n_samples, args.iterations, args.frames, args.output_gif, args.output_final, args.seed)
