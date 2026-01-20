"""Visualize GNG results from C++ experiments.

Usage:
    # Visualize single ring test
    python visualize_results.py single_ring_output --output gng_single_ring.gif

    # Visualize tracking test
    python visualize_results.py tracking_output --mode tracking --output gng_tracking.gif
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


def create_frame(
    ax,
    samples: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    center: np.ndarray | None = None,
) -> None:
    """Create a single visualization frame."""
    ax.clear()

    # Plot samples
    ax.scatter(samples[:, 0], samples[:, 1], c="skyblue", s=5, alpha=0.4, label="Samples")

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
        ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5, label="Nodes")

    # Plot center (for tracking)
    if center is not None:
        ax.scatter(center[0], center[1], c="green", s=100, marker="x", zorder=6, label="Center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right")


def visualize_single_ring(input_dir: Path, output_path: str, fps: int = 10):
    """Visualize single ring test results."""
    print(f"Loading results from {input_dir}")

    # Load samples
    samples = load_samples(input_dir / "samples.csv")
    print(f"Loaded {len(samples)} samples")

    # Load frame list
    frames_df = pd.read_csv(input_dir / "frames.csv")
    iterations = frames_df["iteration"].tolist()
    print(f"Found {len(iterations)} frames")

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    images = []

    for i, iteration in enumerate(iterations):
        suffix = f"{iteration:05d}"
        nodes = load_nodes(input_dir / f"nodes_{suffix}.csv")
        edges = load_edges(input_dir / f"edges_{suffix}.csv")

        create_frame(
            ax, samples, nodes, edges,
            f"GNG Training (C++) - Iteration {iteration} ({len(nodes)} nodes)"
        )

        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        images.append(img.convert("RGB"))

        if i % 10 == 0:
            print(f"Frame {i+1}/{len(iterations)}: iteration {iteration}, {len(nodes)} nodes")

    # Add final frame copies
    images.extend([images[-1]] * 10)

    # Save GIF
    duration = 1000 // fps  # ms per frame
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved GIF: {output_path}")

    # Save final frame as PNG
    final_png = output_path.replace(".gif", "_final.png")
    plt.savefig(final_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final image: {final_png}")

    plt.close(fig)


def visualize_tracking(input_dir: Path, output_path: str, fps: int = 20):
    """Visualize tracking test results."""
    print(f"Loading results from {input_dir}")

    # Load metadata
    meta_df = pd.read_csv(input_dir / "metadata.csv")
    metadata = dict(zip(meta_df["key"], meta_df["value"]))
    n_frames = int(metadata["n_frames"])
    print(f"Tracking: {n_frames} frames")

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    images = []

    for frame in range(n_frames):
        suffix = f"{frame:04d}"
        samples = load_samples(input_dir / f"frame_{suffix}_samples.csv")
        nodes = load_nodes(input_dir / f"frame_{suffix}_nodes.csv")
        edges = load_edges(input_dir / f"frame_{suffix}_edges.csv")

        center_df = pd.read_csv(input_dir / f"frame_{suffix}_center.csv")
        center = center_df[["x", "y"]].values[0]

        create_frame(
            ax, samples, nodes, edges,
            f"GNG Tracking (C++) - Frame {frame} ({len(nodes)} nodes)",
            center=center,
        )

        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        images.append(img.convert("RGB"))

        if frame % 20 == 0:
            print(f"Frame {frame}/{n_frames}: {len(nodes)} nodes, center=({center[0]:.2f}, {center[1]:.2f})")

    # Add final frame copies
    images.extend([images[-1]] * 10)

    # Save GIF
    duration = 1000 // fps  # ms per frame
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved GIF: {output_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize GNG C++ experiment results")
    parser.add_argument("input_dir", type=str, help="Input directory with CSV files")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "tracking"],
                        help="Visualization mode")
    parser.add_argument("--output", type=str, default=None, help="Output GIF path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    if args.output is None:
        args.output = f"gng_{args.mode}.gif"

    if args.mode == "single":
        visualize_single_ring(input_dir, args.output, args.fps)
    elif args.mode == "tracking":
        visualize_tracking(input_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
