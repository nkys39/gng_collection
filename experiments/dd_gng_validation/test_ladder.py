"""Test 3: Ladder Detection Scenario (Paper Reproduction).

Simulates the ladder detection use case from Saputra et al. (2019).
The paper uses DD-GNG for detecting ladder rungs for a quadruped robot.

Expected results:
- Higher node density on ladder rungs (attention regions)
- Accurate detection of all rung positions
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.dd_gng.python.model import DynamicDensityGNG, DDGNGParams

OUTPUT_DIR = Path(__file__).parent / "outputs"


def generate_ladder_pointcloud(n_background: int = 1000, n_rung_points: int = 200, seed: int = 42) -> tuple:
    """Generate 3D point cloud with ladder-like structure.

    Returns:
        (all_points, ladder_points, rung_heights)
    """
    np.random.seed(seed)

    # Background: random points in a box
    background = np.random.rand(n_background, 3)
    background[:, 2] *= 2.0  # Extend in Z

    # Ladder rungs: horizontal bars at specific heights
    rung_heights = [0.3, 0.6, 0.9, 1.2, 1.5]
    ladder_points = []

    for h in rung_heights:
        rung = np.zeros((n_rung_points, 3))
        rung[:, 0] = np.random.uniform(0.4, 0.6, n_rung_points)  # X: narrow range
        rung[:, 1] = np.random.uniform(0.3, 0.7, n_rung_points)  # Y: depth
        rung[:, 2] = h + np.random.normal(0, 0.02, n_rung_points)  # Z: at rung height
        ladder_points.append(rung)

    ladder_points = np.vstack(ladder_points)
    all_points = np.vstack([background, ladder_points])

    return all_points, ladder_points, rung_heights


def run_ladder_test(seed: int = 42) -> dict:
    """Run ladder detection test.

    Returns:
        dict with test results and data for visualization.
    """
    print("=" * 60)
    print("Test 3: Ladder Detection Scenario (Paper Reproduction)")
    print("=" * 60)

    # Generate ladder point cloud
    all_points, ladder_points, rung_heights = generate_ladder_pointcloud(seed=seed)

    print(f"\nGenerated {len(all_points)} points:")
    print(f"  - Background: {len(all_points) - len(ladder_points)}")
    print(f"  - Ladder rungs: {len(ladder_points)} ({len(rung_heights)} rungs)")

    # DD-GNG with attention regions on ladder rungs
    params = DDGNGParams(
        max_nodes=150,
        lambda_=100,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
        max_age=100,
        utility_k=1000.0,
        kappa=10,
        strength_power=4,
        strength_scale=4.0,
        use_strength_learning=True,
        use_strength_insertion=True,
    )

    model = DynamicDensityGNG(n_dim=3, params=params, seed=seed)

    # Add attention regions for each ladder rung
    attention_regions = []
    for h in rung_heights:
        region = {
            "center": [0.5, 0.5, h],
            "size": [0.15, 0.25, 0.05],
        }
        model.add_attention_region(
            center=region["center"],
            size=region["size"],
            strength=5.0,
        )
        attention_regions.append(region)

    print(f"Added {len(attention_regions)} attention regions for ladder rungs")

    # Collect frames for GIF
    frames = []
    n_iterations = 8000
    frame_interval = 200

    fig = plt.figure(figsize=(14, 6))

    def callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            strengths = m.get_node_strengths()

            fig.clear()

            # Left: 3D view
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")

            # Plot data points
            ax1.scatter(
                all_points[:, 0],
                all_points[:, 1],
                all_points[:, 2],
                c="lightgray",
                s=1,
                alpha=0.2,
            )

            # Plot edges
            for i, j in edges:
                ax1.plot(
                    [nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    [nodes[i, 2], nodes[j, 2]],
                    "gray",
                    linewidth=0.3,
                    alpha=0.5,
                )

            # Plot nodes colored by strength
            scatter = ax1.scatter(
                nodes[:, 0],
                nodes[:, 1],
                nodes[:, 2],
                c=strengths,
                cmap="YlOrRd",
                s=30,
                edgecolors="black",
                linewidths=0.2,
                vmin=1.0,
                vmax=6.0,
                zorder=5,
            )

            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_zlim(0, 2)
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            ax1.set_title(f"Iter {iteration} ({len(nodes)} nodes)")
            ax1.view_init(elev=20, azim=45)

            # Right: Side view (YZ plane)
            ax2 = fig.add_subplot(1, 2, 2)

            # Plot data points
            ax2.scatter(all_points[:, 1], all_points[:, 2], c="lightgray", s=1, alpha=0.2)

            # Draw rung regions
            for h in rung_heights:
                ax2.axhline(y=h, color="orange", linestyle="--", alpha=0.5, linewidth=1)
                ax2.fill_between([0.25, 0.75], h - 0.05, h + 0.05, color="orange", alpha=0.1)

            # Plot edges
            for i, j in edges:
                ax2.plot(
                    [nodes[i, 1], nodes[j, 1]],
                    [nodes[i, 2], nodes[j, 2]],
                    "gray",
                    linewidth=0.3,
                    alpha=0.5,
                )

            # Plot nodes colored by strength
            ax2.scatter(
                nodes[:, 1],
                nodes[:, 2],
                c=strengths,
                cmap="YlOrRd",
                s=30,
                edgecolors="black",
                linewidths=0.2,
                vmin=1.0,
                vmax=6.0,
                zorder=5,
            )

            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 2)
            ax2.set_xlabel("Y (depth)")
            ax2.set_ylabel("Z (height)")
            ax2.set_title(f"Side View - Iter {iteration}")

            fig.suptitle("DD-GNG Ladder Detection", fontsize=14)
            plt.tight_layout()

            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
            frames.append(img.convert("RGB"))

    # Train
    print(f"\nTraining DD-GNG for {n_iterations} iterations...")
    model.train(all_points, n_iterations=n_iterations, callback=callback)

    plt.close(fig)

    # Get final results
    nodes, edges = model.get_graph()
    strengths = model.get_node_strengths()

    # Analyze results
    high_strength_nodes = strengths > 3.0
    rung_node_count = high_strength_nodes.sum()

    # Count nodes near each rung
    rung_counts = {}
    rung_node_z = nodes[high_strength_nodes, 2]
    for h in rung_heights:
        near_rung = np.sum(np.abs(rung_node_z - h) < 0.1)
        rung_counts[h] = near_rung

    results = {
        "all_points": all_points,
        "ladder_points": ladder_points,
        "rung_heights": rung_heights,
        "attention_regions": attention_regions,
        "nodes": nodes,
        "edges": edges,
        "strengths": strengths,
        "high_strength_count": rung_node_count,
        "high_strength_ratio": rung_node_count / len(nodes),
        "rung_counts": rung_counts,
        "frames": frames,
    }

    # Print results
    print(f"\nResults:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  High-strength nodes (on rungs): {rung_node_count}")
    print(f"  Percentage on rungs: {100 * results['high_strength_ratio']:.1f}%")
    print(f"  Max strength: {strengths.max():.1f}")

    print(f"\nNodes per rung height:")
    for h, count in rung_counts.items():
        print(f"  Height {h:.1f}: {count} nodes")

    return results


def create_visualizations(results: dict) -> None:
    """Create and save visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save GIF
    if results["frames"]:
        frames = results["frames"]
        frames.extend([frames[-1]] * 15)  # Hold final frame
        frames[0].save(
            OUTPUT_DIR / "ladder_detection.gif",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        print(f"\nSaved: {OUTPUT_DIR / 'ladder_detection.gif'}")

    # Create final figure
    fig = plt.figure(figsize=(18, 6))

    # Left: 3D view
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")

    ax1.scatter(
        results["all_points"][:, 0],
        results["all_points"][:, 1],
        results["all_points"][:, 2],
        c="lightgray",
        s=1,
        alpha=0.2,
    )

    for i, j in results["edges"]:
        ax1.plot(
            [results["nodes"][i, 0], results["nodes"][j, 0]],
            [results["nodes"][i, 1], results["nodes"][j, 1]],
            [results["nodes"][i, 2], results["nodes"][j, 2]],
            "gray",
            linewidth=0.3,
            alpha=0.5,
        )

    scatter = ax1.scatter(
        results["nodes"][:, 0],
        results["nodes"][:, 1],
        results["nodes"][:, 2],
        c=results["strengths"],
        cmap="YlOrRd",
        s=40,
        edgecolors="black",
        linewidths=0.3,
        vmin=1.0,
        vmax=6.0,
        zorder=5,
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 2)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"3D View ({len(results['nodes'])} nodes)")
    ax1.view_init(elev=20, azim=45)

    # Middle: Side view with rung regions
    ax2 = fig.add_subplot(1, 3, 2)

    ax2.scatter(results["all_points"][:, 1], results["all_points"][:, 2], c="lightgray", s=1, alpha=0.2)

    for h in results["rung_heights"]:
        ax2.axhline(y=h, color="orange", linestyle="--", alpha=0.7, linewidth=1)
        ax2.fill_between([0.25, 0.75], h - 0.05, h + 0.05, color="orange", alpha=0.15,
                         label=f"Rung {h:.1f}" if h == results["rung_heights"][0] else "")

    for i, j in results["edges"]:
        ax2.plot(
            [results["nodes"][i, 1], results["nodes"][j, 1]],
            [results["nodes"][i, 2], results["nodes"][j, 2]],
            "gray",
            linewidth=0.3,
            alpha=0.5,
        )

    scatter2 = ax2.scatter(
        results["nodes"][:, 1],
        results["nodes"][:, 2],
        c=results["strengths"],
        cmap="YlOrRd",
        s=40,
        edgecolors="black",
        linewidths=0.3,
        vmin=1.0,
        vmax=6.0,
        zorder=5,
    )
    plt.colorbar(scatter2, ax=ax2, label="Strength")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 2)
    ax2.set_xlabel("Y (depth)")
    ax2.set_ylabel("Z (height)")
    ax2.set_title("Side View (YZ plane)")

    # Right: Bar chart of nodes per rung
    ax3 = fig.add_subplot(1, 3, 3)

    heights = list(results["rung_counts"].keys())
    counts = list(results["rung_counts"].values())

    bars = ax3.barh(
        [f"h={h:.1f}" for h in heights],
        counts,
        color="coral",
        edgecolor="black",
    )

    ax3.set_xlabel("Number of High-Strength Nodes")
    ax3.set_ylabel("Rung Height")
    ax3.set_title("Node Distribution per Rung")

    # Add value labels
    for bar, count in zip(bars, counts):
        ax3.annotate(
            f"{count}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
        )

    # Add summary text
    summary = (
        f"Total nodes: {len(results['nodes'])}\n"
        f"High-strength: {results['high_strength_count']} ({100*results['high_strength_ratio']:.1f}%)\n"
        f"Detected rungs: {len([c for c in counts if c > 0])}/{len(heights)}"
    )
    ax3.text(
        0.95, 0.05, summary,
        transform=ax3.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle("DD-GNG Ladder Detection - Final Results", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ladder_detection.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'ladder_detection.png'}")
    plt.close()


def main():
    """Run test and create visualizations."""
    results = run_ladder_test()
    create_visualizations(results)

    # Return pass/fail (>25% of nodes should be on rungs, all rungs detected)
    ratio_pass = results["high_strength_ratio"] > 0.25
    all_rungs_detected = all(count > 0 for count in results["rung_counts"].values())
    return ratio_pass and all_rungs_detected


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    print(f"Test 3 Result: {'PASS' if success else 'FAIL'}")
    print(f"{'='*60}")
