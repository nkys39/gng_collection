#!/usr/bin/env python3
"""Compare Toda and Furuta contour detection methods on GNG-DT Robot.

This experiment compares two contour detection algorithms:
- Toda et al. (2021): Angle gap threshold method (conventional)
- Furuta et al. (FSS2022): CCW traversal method (proposed)

Usage:
    python test_contour_comparison.py [--output-dir DIR]
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "gng_dt" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "3d"))
sys.path.insert(0, str(Path(__file__).parent))

from model_robot import GrowingNeuralGasDTRobot, GNGDTRobotParams
from sampler import sample_floor_and_wall
from contour_detection import (
    ContourResult,
    FurutaContourDetector,
    TodaContourDetector,
)


def train_gng_dt_robot(points: np.ndarray, params: GNGDTRobotParams, n_iterations: int):
    """Train GNG-DT Robot model."""
    print(f"Training GNG-DT Robot for {n_iterations} iterations...")
    print(f"Parameters: max_nodes={params.max_nodes}, max_angle={params.max_angle}")

    gng = GrowingNeuralGasDTRobot(params=params)
    gng.train(points, n_iterations=n_iterations)

    return gng


def extract_gng_data(gng: GrowingNeuralGasDTRobot):
    """Extract node data from trained GNG-DT Robot.

    Returns data with remapped indices (contiguous from 0).
    """
    # Get active nodes and build index mapping
    active_nodes = []
    index_map = {}  # old_id -> new_id

    for node in gng.nodes:
        if node.id != -1:
            index_map[node.id] = len(active_nodes)
            active_nodes.append(node)

    n_active = len(active_nodes)

    # Extract node positions
    nodes = np.array([node.position for node in active_nodes])

    # Extract traversability with remapped indices
    traversability = np.array([node.traversability_property for node in active_nodes])

    # Build remapped pedge matrix
    pedge = np.zeros((n_active, n_active), dtype=int)
    for i, node in enumerate(active_nodes):
        old_id = node.id
        for j, other_node in enumerate(active_nodes):
            other_old_id = other_node.id
            if gng.edges_traversability[old_id, other_old_id] == 1:
                pedge[i, j] = 1

    return nodes, pedge, traversability


def compare_methods(nodes, pedge, traversability):
    """Compare Toda and Furuta contour detection methods."""
    print("\n=== Comparing Contour Detection Methods ===\n")

    # Toda method
    print("Running Toda method (angle gap threshold)...")
    toda = TodaContourDetector()
    t0 = time.perf_counter()
    toda_contour = toda.detect(nodes, pedge, traversability)
    toda_time = (time.perf_counter() - t0) * 1000
    toda_count = int(np.sum(toda_contour))
    print(f"  Contour nodes: {toda_count}")
    print(f"  Time: {toda_time:.2f} ms")

    # Furuta method
    print("\nRunning Furuta method (CCW traversal)...")
    furuta = FurutaContourDetector()
    t0 = time.perf_counter()
    furuta_result = furuta.detect(nodes, pedge, traversability)
    furuta_time = (time.perf_counter() - t0) * 1000
    furuta_count = len(furuta_result.all_contour_nodes)
    print(f"  Outer contour nodes: {len(furuta_result.outer_contour)}")
    print(f"  Inner contours: {len(furuta_result.inner_contours)}")
    print(f"  Total contour nodes: {furuta_count}")
    print(f"  Time: {furuta_time:.2f} ms")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Method':<20} {'Nodes':<10} {'Time (ms)':<10}")
    print("-" * 40)
    print(f"{'Toda (conventional)':<20} {toda_count:<10} {toda_time:<10.2f}")
    print(f"{'Furuta (CCW)':<20} {furuta_count:<10} {furuta_time:<10.2f}")
    print(f"\nTime ratio (Furuta/Toda): {furuta_time/toda_time:.2f}x")

    return toda_contour, furuta_result, toda_time, furuta_time


def visualize_comparison(
    nodes, traversability, toda_contour, furuta_result, output_dir
):
    """Visualize comparison of both methods."""
    fig = plt.figure(figsize=(16, 6))

    n_trav = int(np.sum(traversability))

    # Common settings
    def setup_ax(ax, title):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.view_init(elev=25, azim=120)

    # 1. Toda method result
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")

    # Non-traversable nodes (red)
    non_trav_mask = traversability == 0
    if np.any(non_trav_mask):
        ax1.scatter(
            nodes[non_trav_mask, 0],
            nodes[non_trav_mask, 1],
            nodes[non_trav_mask, 2],
            c="red",
            s=30,
            alpha=0.3,
            label="Non-traversable",
        )

    # Traversable non-contour (green)
    trav_non_contour = (traversability == 1) & (toda_contour == 0)
    if np.any(trav_non_contour):
        ax1.scatter(
            nodes[trav_non_contour, 0],
            nodes[trav_non_contour, 1],
            nodes[trav_non_contour, 2],
            c="green",
            s=50,
            alpha=0.6,
            label="Traversable",
        )

    # Contour nodes (orange)
    contour_mask = toda_contour == 1
    if np.any(contour_mask):
        ax1.scatter(
            nodes[contour_mask, 0],
            nodes[contour_mask, 1],
            nodes[contour_mask, 2],
            c="orange",
            s=80,
            marker="o",
            label=f"Contour ({int(np.sum(toda_contour))})",
        )

    ax1.legend(loc="upper left", fontsize=8)
    setup_ax(ax1, f"Toda Method (Angle Gap)\nContour: {int(np.sum(toda_contour))} nodes")

    # 2. Furuta method result
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")

    # Non-traversable nodes (red)
    if np.any(non_trav_mask):
        ax2.scatter(
            nodes[non_trav_mask, 0],
            nodes[non_trav_mask, 1],
            nodes[non_trav_mask, 2],
            c="red",
            s=30,
            alpha=0.3,
            label="Non-traversable",
        )

    # Traversable non-contour (green)
    furuta_contour = furuta_result.contour_flags
    trav_non_contour_f = (traversability == 1) & (furuta_contour == 0)
    if np.any(trav_non_contour_f):
        ax2.scatter(
            nodes[trav_non_contour_f, 0],
            nodes[trav_non_contour_f, 1],
            nodes[trav_non_contour_f, 2],
            c="green",
            s=50,
            alpha=0.6,
            label="Traversable",
        )

    # Outer contour (orange) - draw as connected line
    if furuta_result.outer_contour:
        outer_nodes = nodes[furuta_result.outer_contour]
        # Close the loop
        outer_closed = np.vstack([outer_nodes, outer_nodes[0:1]])
        ax2.plot(
            outer_closed[:, 0],
            outer_closed[:, 1],
            outer_closed[:, 2],
            c="orange",
            linewidth=2,
            label=f"Outer ({len(furuta_result.outer_contour)})",
        )
        ax2.scatter(
            outer_nodes[:, 0],
            outer_nodes[:, 1],
            outer_nodes[:, 2],
            c="orange",
            s=60,
        )

    # Inner contours (purple)
    for i, inner in enumerate(furuta_result.inner_contours):
        if len(inner) > 0:
            inner_nodes = nodes[inner]
            inner_closed = np.vstack([inner_nodes, inner_nodes[0:1]])
            label = f"Inner ({len(inner)})" if i == 0 else None
            ax2.plot(
                inner_closed[:, 0],
                inner_closed[:, 1],
                inner_closed[:, 2],
                c="purple",
                linewidth=2,
                label=label,
            )
            ax2.scatter(
                inner_nodes[:, 0], inner_nodes[:, 1], inner_nodes[:, 2], c="purple", s=60
            )

    ax2.legend(loc="upper left", fontsize=8)
    n_furuta = len(furuta_result.all_contour_nodes)
    setup_ax(ax2, f"Furuta Method (CCW Traversal)\nContour: {n_furuta} nodes")

    # 3. Comparison overlay (top-down view)
    ax3 = fig.add_subplot(1, 3, 3)

    # Plot traversable nodes
    trav_mask = traversability == 1
    ax3.scatter(
        nodes[trav_mask, 0],
        nodes[trav_mask, 1],
        c="lightgreen",
        s=30,
        alpha=0.5,
        label="Traversable",
    )

    # Toda contour (blue circles)
    toda_mask = toda_contour == 1
    if np.any(toda_mask):
        ax3.scatter(
            nodes[toda_mask, 0],
            nodes[toda_mask, 1],
            facecolors="none",
            edgecolors="blue",
            s=100,
            marker="o",
            linewidths=2,
            label=f"Toda ({int(np.sum(toda_contour))})",
        )

    # Furuta outer contour (orange line)
    if furuta_result.outer_contour:
        outer_nodes = nodes[furuta_result.outer_contour]
        outer_closed = np.vstack([outer_nodes, outer_nodes[0:1]])
        ax3.plot(
            outer_closed[:, 0],
            outer_closed[:, 1],
            c="orange",
            linewidth=2,
            label=f"Furuta Outer ({len(furuta_result.outer_contour)})",
        )

    # Furuta inner contours
    for inner in furuta_result.inner_contours:
        if len(inner) > 0:
            inner_nodes = nodes[inner]
            inner_closed = np.vstack([inner_nodes, inner_nodes[0:1]])
            ax3.plot(
                inner_closed[:, 0], inner_closed[:, 1], c="purple", linewidth=2
            )

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title("Top-Down Comparison\n(Blue circles=Toda, Orange line=Furuta)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = output_dir / "contour_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison figure: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare contour detection methods")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=8000,
        help="Number of training iterations",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample data
    print("Sampling 2000 points from floor and wall...")
    points = sample_floor_and_wall(n_samples=2000)
    print(f"Sampled {len(points)} points")

    # Train GNG-DT Robot
    params = GNGDTRobotParams(
        max_nodes=150,
        lambda_=200,
        eps_b=0.05,
        eps_n=0.0005,
        alpha=0.5,
        beta=0.0005,
        max_age=88,
        max_angle=20.0,
        tau_normal=0.998,
    )

    gng = train_gng_dt_robot(points, params, args.n_iterations)

    # Extract data
    nodes, pedge, traversability = extract_gng_data(gng)
    print(f"\nTrained network: {len(nodes)} nodes, {int(np.sum(traversability))} traversable")

    # Compare methods
    toda_contour, furuta_result, toda_time, furuta_time = compare_methods(
        nodes, pedge, traversability
    )

    # Visualize
    visualize_comparison(nodes, traversability, toda_contour, furuta_result, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
