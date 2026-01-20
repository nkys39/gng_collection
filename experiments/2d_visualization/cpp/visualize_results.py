"""Visualize GNG results from C++ experiments.

This script matches the visualization style of the Python test scripts.

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


def create_single_ring_frame(
    ax,
    samples: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
    ring_center: np.ndarray | None = None,
    ring_r_inner: float | None = None,
    ring_r_outer: float | None = None,
) -> None:
    """Create a single frame for single ring visualization.

    Matches the Python version: test_gng_single_ring.py
    """
    ax.clear()

    # Draw ring background (like Python version with bg_image)
    if ring_center is not None and ring_r_inner is not None and ring_r_outer is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        # Outer ring
        ring_outer_x = ring_center[0] + ring_r_outer * np.cos(theta)
        ring_outer_y = ring_center[1] + ring_r_outer * np.sin(theta)
        # Inner ring (hole)
        ring_inner_x = ring_center[0] + ring_r_inner * np.cos(theta)
        ring_inner_y = ring_center[1] + ring_r_inner * np.sin(theta)
        # Fill ring
        ax.fill(ring_outer_x, ring_outer_y, color="skyblue", alpha=0.5)
        ax.fill(ring_inner_x, ring_inner_y, color="white")

    # Plot sample points (matching Python version style)
    ax.scatter(samples[:, 0], samples[:, 1], c="skyblue", s=3, alpha=0.3, label="Data")

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

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Flip y-axis to match image coordinates (like Python version)
    ax.set_aspect("equal")
    ax.set_title(f"GNG Training - Iteration {iteration} ({len(nodes)} nodes)")
    ax.legend(loc="upper right")


def create_tracking_frame(
    ax,
    samples: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    ring_center: np.ndarray,
    ring_r_inner: float,
    ring_r_outer: float,
    orbit_center: np.ndarray,
    orbit_radius: float,
    frame: int,
    total_frames: int,
) -> None:
    """Create a single frame for tracking visualization.

    Matches the Python version: test_gng_tracking.py
    """
    ax.clear()

    # Draw orbit path (dashed circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    orbit_x = orbit_center[0] + orbit_radius * np.cos(theta)
    orbit_y = orbit_center[1] + orbit_radius * np.sin(theta)
    ax.plot(orbit_x, orbit_y, "g--", alpha=0.3, linewidth=1, label="Orbit")

    # Draw ring outline with fill
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
        samples[:, 0],
        samples[:, 1],
        c="skyblue",
        s=5,
        alpha=0.5,
        label="Current samples",
    )

    # Draw GNG edges
    for i, j in edges:
        if i < len(nodes) and j < len(nodes):
            ax.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                "r-",
                linewidth=1.5,
                alpha=0.7,
            )

    # Draw GNG nodes
    if len(nodes) > 0:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            c="red",
            s=40,
            zorder=5,
            label=f"GNG ({len(nodes)} nodes)",
        )

    # Draw ring center (blue + marker like Python version)
    ax.scatter([ring_center[0]], [ring_center[1]], c="blue", s=30, marker="+", zorder=6)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"GNG Tracking - Frame {frame}/{total_frames}")
    ax.legend(loc="upper right", fontsize=8)


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

    # Load ring metadata if available
    ring_center = None
    ring_r_inner = None
    ring_r_outer = None
    metadata_path = input_dir / "metadata.csv"
    if metadata_path.exists():
        meta_df = pd.read_csv(metadata_path)
        metadata = dict(zip(meta_df["key"], meta_df["value"]))
        ring_center = np.array([float(metadata["ring_center_x"]), float(metadata["ring_center_y"])])
        ring_r_inner = float(metadata["ring_r_inner"])
        ring_r_outer = float(metadata["ring_r_outer"])
        print(f"Ring: center=({ring_center[0]}, {ring_center[1]}), r_inner={ring_r_inner}, r_outer={ring_r_outer}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    images = []

    for i, iteration in enumerate(iterations):
        suffix = f"{iteration:05d}"
        nodes = load_nodes(input_dir / f"nodes_{suffix}.csv")
        edges = load_edges(input_dir / f"edges_{suffix}.csv")

        create_single_ring_frame(ax, samples, nodes, edges, iteration,
                                  ring_center, ring_r_inner, ring_r_outer)

        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        images.append(img.convert("RGB"))

        if i % 10 == 0:
            print(f"Frame {i+1}/{len(iterations)}: iteration {iteration}, {len(nodes)} nodes")

    # Add final frame copies
    images.extend([images[-1]] * 10)

    # Save GIF (duration=100ms like Python version)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )
    print(f"Saved GIF: {output_path}")

    # Save final frame as PNG
    final_png = output_path.replace(".gif", "_final.png")
    plt.savefig(final_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final image: {final_png}")

    plt.close(fig)


def visualize_tracking(input_dir: Path, output_path: str, fps: int = 12):
    """Visualize tracking test results."""
    print(f"Loading results from {input_dir}")

    # Load metadata
    meta_df = pd.read_csv(input_dir / "metadata.csv")
    metadata = dict(zip(meta_df["key"], meta_df["value"]))
    n_frames = int(metadata["n_frames"])
    orbit_center_x = float(metadata["orbit_center_x"])
    orbit_center_y = float(metadata["orbit_center_y"])
    orbit_radius = float(metadata["orbit_radius"])
    ring_r_inner = float(metadata["ring_r_inner"])
    ring_r_outer = float(metadata["ring_r_outer"])

    orbit_center = np.array([orbit_center_x, orbit_center_y])

    print(f"Tracking: {n_frames} frames")
    print(f"Orbit: center=({orbit_center_x}, {orbit_center_y}), radius={orbit_radius}")
    print(f"Ring: r_inner={ring_r_inner}, r_outer={ring_r_outer}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    images = []

    for frame in range(n_frames):
        suffix = f"{frame:04d}"
        samples = load_samples(input_dir / f"frame_{suffix}_samples.csv")
        nodes = load_nodes(input_dir / f"frame_{suffix}_nodes.csv")
        edges = load_edges(input_dir / f"frame_{suffix}_edges.csv")

        center_df = pd.read_csv(input_dir / f"frame_{suffix}_center.csv")
        ring_center = center_df[["x", "y"]].values[0]

        create_tracking_frame(
            ax, samples, nodes, edges,
            ring_center, ring_r_inner, ring_r_outer,
            orbit_center, orbit_radius,
            frame + 1, n_frames,
        )

        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        images.append(img.convert("RGB"))

        if frame % 20 == 0:
            print(f"Frame {frame}/{n_frames}: {len(nodes)} nodes, center=({ring_center[0]:.2f}, {ring_center[1]:.2f})")

    # Add final frame copies
    images.extend([images[-1]] * 10)

    # Save GIF (duration=80ms like Python version)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=80,
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
