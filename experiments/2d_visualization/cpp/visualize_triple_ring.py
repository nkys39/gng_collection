"""Visualize triple ring GNG/GNG-T results from C++ experiments.

Usage:
    # GNG
    python visualize_triple_ring.py gng_triple_ring_output --output gng_cpp_triple_ring.gif

    # GNG-T (with triangles)
    python visualize_triple_ring.py gngt_triple_ring_output --output gngt_cpp_triple_ring.gif --triangles
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


def load_nodes(path: Path) -> np.ndarray:
    """Load node positions from CSV."""
    if not path.exists():
        return np.array([]).reshape(0, 2)
    df = pd.read_csv(path)
    if len(df) == 0:
        return np.array([]).reshape(0, 2)
    return df[["x", "y"]].values


def load_edges(path: Path) -> list[tuple[int, int]]:
    """Load edge list from CSV."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [(int(row["i"]), int(row["j"])) for _, row in df.iterrows()]


def load_triangles(path: Path) -> list[tuple[int, int, int]]:
    """Load triangle list from CSV."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [(int(row["i"]), int(row["j"]), int(row["k"])) for _, row in df.iterrows()]


def create_frame(
    ax,
    samples: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    triangles: list[tuple[int, int, int]] | None,
    iteration: int,
    algorithm: str = "GNG",
    show_triangles: bool = False,
) -> None:
    """Create a single frame for triple ring visualization."""
    ax.clear()

    # Draw triple ring background
    theta = np.linspace(0, 2 * np.pi, 100)
    rings = [
        (0.15, 0.20),  # inner ring
        (0.25, 0.30),  # middle ring
        (0.35, 0.40),  # outer ring
    ]
    for r_inner, r_outer in rings:
        outer_x = 0.5 + r_outer * np.cos(theta)
        outer_y = 0.5 + r_outer * np.sin(theta)
        inner_x = 0.5 + r_inner * np.cos(theta)
        inner_y = 0.5 + r_inner * np.sin(theta)
        ax.fill(outer_x, outer_y, color="lightblue", alpha=0.3)
        ax.fill(inner_x, inner_y, color="white")

    # Plot sample points
    ax.scatter(samples[:, 0], samples[:, 1], c="skyblue", s=3, alpha=0.3)

    # Draw triangles (filled)
    if show_triangles and triangles and len(nodes) > 0:
        for tri in triangles:
            if all(idx < len(nodes) for idx in tri):
                triangle = plt.Polygon(
                    nodes[list(tri)],
                    fill=True,
                    facecolor="lightgreen",
                    edgecolor="green",
                    alpha=0.2,
                    linewidth=0.5,
                )
                ax.add_patch(triangle)

    # Plot edges
    for i, j in edges:
        if i < len(nodes) and j < len(nodes):
            ax.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                "r-",
                linewidth=1.5,
                alpha=0.7,
            )

    # Plot nodes
    if len(nodes) > 0:
        ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Flip y-axis
    ax.set_aspect("equal")
    ax.set_title(f"{algorithm} (C++) - Iter {iteration} ({len(nodes)} nodes, {len(edges)} edges)")


def visualize(
    input_dir: str,
    output_gif: str,
    output_png: str | None = None,
    show_triangles: bool = False,
    algorithm: str = "GNG",
) -> None:
    """Create visualization from C++ output."""
    input_path = Path(input_dir)

    # Load samples
    samples = load_samples(input_path / "samples.csv")

    # Load frame list
    frames_df = pd.read_csv(input_path / "frames.csv")
    frame_iterations = frames_df["iteration"].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    frames = []

    for iteration in frame_iterations:
        suffix = f"{iteration:05d}"
        nodes = load_nodes(input_path / f"nodes_{suffix}.csv")
        edges = load_edges(input_path / f"edges_{suffix}.csv")

        triangles = None
        if show_triangles:
            tri_path = input_path / f"triangles_{suffix}.csv"
            if tri_path.exists():
                triangles = load_triangles(tri_path)

        create_frame(ax, samples, nodes, edges, triangles, iteration, algorithm, show_triangles)
        fig.canvas.draw()

        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        )
        frames.append(img.convert("RGB"))

        if iteration % 5000 == 0 or iteration == frame_iterations[-1]:
            print(f"Rendered frame at iteration {iteration}")

    plt.close(fig)

    # Save GIF
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    # Save final PNG
    if output_png:
        final_nodes = load_nodes(input_path / "final_nodes.csv")
        final_edges = load_edges(input_path / "final_edges.csv")
        final_triangles = None
        if show_triangles:
            final_tri_path = input_path / "final_triangles.csv"
            if final_tri_path.exists():
                final_triangles = load_triangles(final_tri_path)

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
        create_frame(ax, samples, final_nodes, final_edges, final_triangles,
                     frame_iterations[-1], algorithm, show_triangles)
        fig.savefig(output_png, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved PNG: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Visualize triple ring results")
    parser.add_argument("input_dir", help="Directory with CSV output files")
    parser.add_argument("--output", "-o", default="output.gif", help="Output GIF filename")
    parser.add_argument("--png", default=None, help="Output PNG filename (final frame)")
    parser.add_argument("--triangles", "-t", action="store_true", help="Show triangles (for GNG-T)")
    parser.add_argument("--algorithm", "-a", default="GNG", help="Algorithm name for title")

    args = parser.parse_args()

    visualize(
        args.input_dir,
        args.output,
        args.png,
        show_triangles=args.triangles,
        algorithm=args.algorithm,
    )


if __name__ == "__main__":
    main()
