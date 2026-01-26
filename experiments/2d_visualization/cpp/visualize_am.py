"""Visualize AiS-GNG-AM results with movement-colored nodes.

Nodes are colored by their Amount of Movement (AM):
- Blue = static nodes (low AM)
- Red = moving nodes (high AM)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_samples(path: Path) -> np.ndarray:
    """Load sample points from CSV."""
    df = pd.read_csv(path)
    return df[["x", "y"]].values


def load_nodes_with_movement(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load node positions and movement values from CSV."""
    if not path.exists():
        return np.array([]).reshape(0, 2), np.array([])
    df = pd.read_csv(path)
    if len(df) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    positions = df[["x", "y"]].values
    if "movement" in df.columns:
        movements = df["movement"].values
    else:
        movements = np.zeros(len(df))
    return positions, movements


def load_edges(path: Path) -> list[tuple[int, int]]:
    """Load edge list from CSV."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [(int(row["i"]), int(row["j"])) for _, row in df.iterrows()]


def create_frame_am(
    ax,
    samples: np.ndarray,
    nodes: np.ndarray,
    movements: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
    algorithm: str = "AiS-GNG-AM",
) -> None:
    """Create a frame with movement-colored nodes."""
    ax.clear()

    # Draw triple ring background
    theta = np.linspace(0, 2 * np.pi, 100)
    rings = [
        (0.50, 0.23, 0.06, 0.14),
        (0.27, 0.68, 0.06, 0.14),
        (0.73, 0.68, 0.06, 0.14),
    ]
    for cx, cy, r_inner, r_outer in rings:
        outer_x = cx + r_outer * np.cos(theta)
        outer_y = cy + r_outer * np.sin(theta)
        inner_x = cx + r_inner * np.cos(theta)
        inner_y = cy + r_inner * np.sin(theta)
        ax.fill(outer_x, outer_y, color="lightblue", alpha=0.3)
        ax.fill(inner_x, inner_y, color="white")

    # Plot sample points
    ax.scatter(samples[:, 0], samples[:, 1], c="skyblue", s=3, alpha=0.3)

    # Plot edges (gray for AM visualization)
    for i, j in edges:
        if i < len(nodes) and j < len(nodes):
            ax.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                color="gray",
                linewidth=1.5,
                alpha=0.5,
            )

    # Plot nodes colored by movement
    if len(nodes) > 0:
        if len(movements) > 0 and movements.max() > 0:
            norm_movements = movements / max(movements.max(), 0.001)
        else:
            norm_movements = np.zeros(len(nodes))

        scatter = ax.scatter(
            nodes[:, 0], nodes[:, 1],
            c=norm_movements,
            cmap="coolwarm",
            s=60,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            vmin=0,
            vmax=1,
        )

    n_moving = np.sum(movements > 0.005) if len(movements) > 0 else 0
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(f"{algorithm} (C++) - Iter {iteration} ({len(nodes)} nodes, {n_moving} moving)")


def visualize_am(
    input_dir: str,
    output_gif: str,
    output_png: str | None = None,
    algorithm: str = "AiS-GNG-AM",
) -> None:
    """Create visualization with movement colors."""
    input_path = Path(input_dir)

    samples = load_samples(input_path / "samples.csv")
    frames_df = pd.read_csv(input_path / "frames.csv")
    frame_iterations = frames_df["iteration"].tolist()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    frames = []

    for iteration in frame_iterations:
        suffix = f"{iteration:05d}"
        nodes, movements = load_nodes_with_movement(input_path / f"nodes_{suffix}.csv")
        edges = load_edges(input_path / f"edges_{suffix}.csv")

        create_frame_am(ax, samples, nodes, movements, edges, iteration, algorithm)
        fig.canvas.draw()

        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        )
        frames.append(img.convert("RGB"))

        if iteration % 1000 == 0 or iteration == frame_iterations[-1]:
            print(f"Rendered frame at iteration {iteration}")

    plt.close(fig)

    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    if output_png:
        final_nodes, final_movements = load_nodes_with_movement(input_path / "final_nodes.csv")
        final_edges = load_edges(input_path / "final_edges.csv")

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
        create_frame_am(ax, samples, final_nodes, final_movements, final_edges,
                       frame_iterations[-1], algorithm)
        fig.savefig(output_png, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved PNG: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AiS-GNG-AM results")
    parser.add_argument("input_dir", help="Directory with CSV output files")
    parser.add_argument("--output", "-o", default="output.gif", help="Output GIF filename")
    parser.add_argument("--png", default=None, help="Output PNG filename")
    parser.add_argument("--algorithm", "-a", default="AiS-GNG-AM", help="Algorithm name")

    args = parser.parse_args()
    visualize_am(args.input_dir, args.output, args.png, args.algorithm)


if __name__ == "__main__":
    main()
