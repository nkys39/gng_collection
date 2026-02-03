"""Test 4: Automatic Attention Region Detection.

Tests the automatic detection of attention regions based on surface classification.
Stable corners (nodes that maintain corner classification for > stability_threshold
iterations) are automatically treated as attention regions.

Expected results:
- Corners in the 3D L-shape should be auto-detected as attention regions
- Auto-detected nodes should have higher strength
- Node density should increase around auto-detected corners
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

from algorithms.dd_gng.python.model import DynamicDensityGNG, DDGNGParams, SurfaceType

OUTPUT_DIR = Path(__file__).parent / "outputs"


def sample_floor_and_wall(n_samples: int = 2000, seed: int = None) -> np.ndarray:
    """Sample points from a floor and wall configuration (L-shape in 3D)."""
    if seed is not None:
        np.random.seed(seed)

    floor_size = 0.8
    wall_height = 0.6
    floor_depth = 0.02
    wall_depth = 0.02

    offset_x = (1.0 - floor_size) / 2
    offset_z = (1.0 - floor_size) / 2

    floor_area = floor_size * floor_size
    wall_area = floor_size * wall_height
    total_area = floor_area + wall_area
    n_floor = int(n_samples * floor_area / total_area)
    n_wall = n_samples - n_floor

    all_points = []

    # Floor points (XZ plane, y near 0)
    floor_x = np.random.uniform(offset_x, offset_x + floor_size, n_floor)
    floor_y = np.random.uniform(0, floor_depth, n_floor)
    floor_z = np.random.uniform(offset_z, offset_z + floor_size, n_floor)
    floor_points = np.column_stack([floor_x, floor_y, floor_z])
    all_points.append(floor_points)

    # Wall points (XY plane, z near 0)
    wall_x = np.random.uniform(offset_x, offset_x + floor_size, n_wall)
    wall_y = np.random.uniform(0, wall_height, n_wall)
    wall_z = np.random.uniform(offset_z, offset_z + wall_depth, n_wall)
    wall_points = np.column_stack([wall_x, wall_y, wall_z])
    all_points.append(wall_points)

    return np.vstack(all_points)


def run_auto_detection_test(seed: int = 42) -> dict:
    """Run auto-detection test on L-shaped 3D data.

    Returns:
        dict with test results and data for visualization.
    """
    print("=" * 60)
    print("Test 4: Automatic Attention Region Detection")
    print("=" * 60)

    # Generate L-shaped point cloud (floor + wall with corner)
    np.random.seed(seed)
    data = sample_floor_and_wall(n_samples=2000)

    print(f"\nGenerated {len(data)} points (floor + wall L-shape)")
    print("The corner edge should be auto-detected as attention region")

    # DD-GNG with auto-detection ENABLED
    params_auto = DDGNGParams(
        max_nodes=100,
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
        # Enable auto-detection
        auto_detect_attention=True,
        stability_threshold=16,
        plane_stability_threshold=8,
        corner_strength=5.0,
        plane_ev_ratio=0.01,
        edge_ev_ratio=0.1,
        surface_update_interval=10,
    )

    # DD-GNG without auto-detection (baseline)
    params_manual = DDGNGParams(
        max_nodes=100,
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
        auto_detect_attention=False,
    )

    model_auto = DynamicDensityGNG(n_dim=3, params=params_auto, seed=seed)
    model_manual = DynamicDensityGNG(n_dim=3, params=params_manual, seed=seed)

    # Collect frames
    frames = []
    n_iterations = 8000
    frame_interval = 200

    fig = plt.figure(figsize=(16, 6))

    def callback(m_auto, m_manual, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes_auto, edges_auto = m_auto.get_graph()
            nodes_manual, edges_manual = m_manual.get_graph()
            strengths_auto = m_auto.get_node_strengths()
            surface_types = m_auto.get_node_surface_types()
            auto_attention = m_auto.get_node_auto_attention()

            fig.clear()

            # Left: Manual (no auto-detection)
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c="lightgray", s=1, alpha=0.2)

            for i, j in edges_manual:
                ax1.plot(
                    [nodes_manual[i, 0], nodes_manual[j, 0]],
                    [nodes_manual[i, 1], nodes_manual[j, 1]],
                    [nodes_manual[i, 2], nodes_manual[j, 2]],
                    "gray",
                    linewidth=0.3,
                    alpha=0.5,
                )

            ax1.scatter(
                nodes_manual[:, 0],
                nodes_manual[:, 1],
                nodes_manual[:, 2],
                c="blue",
                s=30,
                edgecolors="black",
                linewidths=0.2,
                zorder=5,
            )

            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_zlim(0, 1)
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            ax1.set_title(f"Manual (no auto) - {len(nodes_manual)} nodes")
            ax1.view_init(elev=20, azim=45)

            # Middle: Auto-detection with strength coloring
            ax2 = fig.add_subplot(1, 3, 2, projection="3d")
            ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c="lightgray", s=1, alpha=0.2)

            for i, j in edges_auto:
                ax2.plot(
                    [nodes_auto[i, 0], nodes_auto[j, 0]],
                    [nodes_auto[i, 1], nodes_auto[j, 1]],
                    [nodes_auto[i, 2], nodes_auto[j, 2]],
                    "gray",
                    linewidth=0.3,
                    alpha=0.5,
                )

            scatter = ax2.scatter(
                nodes_auto[:, 0],
                nodes_auto[:, 1],
                nodes_auto[:, 2],
                c=strengths_auto,
                cmap="YlOrRd",
                s=30,
                edgecolors="black",
                linewidths=0.2,
                vmin=1.0,
                vmax=6.0,
                zorder=5,
            )

            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_zlim(0, 1)
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
            ax2.set_title(f"Auto-detect - {len(nodes_auto)} nodes\nAuto-attention: {m_auto.n_auto_attention}")
            ax2.view_init(elev=20, azim=45)

            # Right: Surface classification coloring
            ax3 = fig.add_subplot(1, 3, 3, projection="3d")
            ax3.scatter(data[:, 0], data[:, 1], data[:, 2], c="lightgray", s=1, alpha=0.2)

            # Color by surface type
            colors = []
            for st in surface_types:
                if st == SurfaceType.STABLE_CORNER or st == SurfaceType.CORNER:
                    colors.append("red")  # Corner
                elif st == SurfaceType.STABLE_EDGE or st == SurfaceType.EDGE:
                    colors.append("yellow")  # Edge
                elif st == SurfaceType.STABLE_PLANE or st == SurfaceType.PLANE:
                    colors.append("green")  # Plane
                else:
                    colors.append("gray")  # Unknown

            for i, j in edges_auto:
                ax3.plot(
                    [nodes_auto[i, 0], nodes_auto[j, 0]],
                    [nodes_auto[i, 1], nodes_auto[j, 1]],
                    [nodes_auto[i, 2], nodes_auto[j, 2]],
                    "gray",
                    linewidth=0.3,
                    alpha=0.5,
                )

            # Highlight auto-attention nodes
            for idx in range(len(nodes_auto)):
                marker = "^" if auto_attention[idx] else "o"
                size = 80 if auto_attention[idx] else 30
                ax3.scatter(
                    [nodes_auto[idx, 0]],
                    [nodes_auto[idx, 1]],
                    [nodes_auto[idx, 2]],
                    c=[colors[idx]],
                    s=size,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=5,
                )

            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_zlim(0, 1)
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_zlabel("Z")
            ax3.set_title("Surface Type\nGreen=Plane, Yellow=Edge, Red=Corner\n(Triangle=Auto-attention)")
            ax3.view_init(elev=20, azim=45)

            fig.suptitle(f"DD-GNG Auto-Detection - Iter {iteration}", fontsize=14)
            plt.tight_layout()

            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
            frames.append(img.convert("RGB"))

    # Train both models
    print(f"\nTraining for {n_iterations} iterations...")

    for i in range(n_iterations):
        idx = np.random.randint(0, len(data))
        model_auto._one_train_update(data[idx])
        model_manual._one_train_update(data[idx])

        if i % frame_interval == 0 or i == n_iterations - 1:
            callback(model_auto, model_manual, i)

    plt.close(fig)

    # Get final results
    nodes_auto, edges_auto = model_auto.get_graph()
    nodes_manual, edges_manual = model_manual.get_graph()
    strengths_auto = model_auto.get_node_strengths()
    surface_types = model_auto.get_node_surface_types()
    auto_attention = model_auto.get_node_auto_attention()

    # Count surface types
    type_counts = {
        "plane": sum(1 for st in surface_types if st in (SurfaceType.PLANE, SurfaceType.STABLE_PLANE)),
        "edge": sum(1 for st in surface_types if st in (SurfaceType.EDGE, SurfaceType.STABLE_EDGE)),
        "corner": sum(1 for st in surface_types if st in (SurfaceType.CORNER, SurfaceType.STABLE_CORNER)),
        "unknown": sum(1 for st in surface_types if st == SurfaceType.UNKNOWN),
    }

    # Analyze corner region density
    corner_region_center = np.array([0.5, 0.5, 0.0])  # L-shape corner
    corner_region_size = np.array([0.15, 0.15, 0.15])

    def count_in_region(nodes, center, size):
        inside = np.all(np.abs(nodes - center) <= size, axis=1)
        return inside.sum()

    auto_in_corner = count_in_region(nodes_auto, corner_region_center, corner_region_size)
    manual_in_corner = count_in_region(nodes_manual, corner_region_center, corner_region_size)

    results = {
        "data": data,
        "nodes_auto": nodes_auto,
        "edges_auto": edges_auto,
        "nodes_manual": nodes_manual,
        "edges_manual": edges_manual,
        "strengths_auto": strengths_auto,
        "surface_types": surface_types,
        "auto_attention": auto_attention,
        "type_counts": type_counts,
        "n_auto_attention": model_auto.n_auto_attention,
        "auto_in_corner": auto_in_corner,
        "manual_in_corner": manual_in_corner,
        "frames": frames,
    }

    # Print results
    print(f"\nResults:")
    print(f"  Auto-detection model:")
    print(f"    Total nodes: {len(nodes_auto)}")
    print(f"    Auto-attention nodes: {model_auto.n_auto_attention}")
    print(f"    Nodes in corner region: {auto_in_corner}")
    print(f"    Surface types: {type_counts}")
    print(f"    Mean strength: {strengths_auto.mean():.2f}")
    print(f"    Max strength: {strengths_auto.max():.2f}")
    print(f"\n  Manual model (no auto-detection):")
    print(f"    Total nodes: {len(nodes_manual)}")
    print(f"    Nodes in corner region: {manual_in_corner}")

    if auto_in_corner > manual_in_corner:
        print(f"\n  Corner density improvement: {auto_in_corner / max(manual_in_corner, 1):.2f}x")

    return results


def create_visualizations(results: dict) -> None:
    """Create and save visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save GIF
    if results["frames"]:
        frames = results["frames"]
        frames.extend([frames[-1]] * 15)  # Hold final frame
        frames[0].save(
            OUTPUT_DIR / "auto_detection.gif",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        print(f"\nSaved: {OUTPUT_DIR / 'auto_detection.gif'}")

    # Create final figure
    fig = plt.figure(figsize=(18, 12))

    # Top row: 3D views
    # Left: Manual
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.scatter(results["data"][:, 0], results["data"][:, 1], results["data"][:, 2],
                c="lightgray", s=1, alpha=0.2)

    for i, j in results["edges_manual"]:
        ax1.plot(
            [results["nodes_manual"][i, 0], results["nodes_manual"][j, 0]],
            [results["nodes_manual"][i, 1], results["nodes_manual"][j, 1]],
            [results["nodes_manual"][i, 2], results["nodes_manual"][j, 2]],
            "gray", linewidth=0.3, alpha=0.5,
        )

    ax1.scatter(
        results["nodes_manual"][:, 0],
        results["nodes_manual"][:, 1],
        results["nodes_manual"][:, 2],
        c="blue", s=40, edgecolors="black", linewidths=0.3, zorder=5,
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"Manual (no auto-detection)\n{len(results['nodes_manual'])} nodes")
    ax1.view_init(elev=20, azim=45)

    # Middle: Auto with strength coloring
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.scatter(results["data"][:, 0], results["data"][:, 1], results["data"][:, 2],
                c="lightgray", s=1, alpha=0.2)

    for i, j in results["edges_auto"]:
        ax2.plot(
            [results["nodes_auto"][i, 0], results["nodes_auto"][j, 0]],
            [results["nodes_auto"][i, 1], results["nodes_auto"][j, 1]],
            [results["nodes_auto"][i, 2], results["nodes_auto"][j, 2]],
            "gray", linewidth=0.3, alpha=0.5,
        )

    scatter = ax2.scatter(
        results["nodes_auto"][:, 0],
        results["nodes_auto"][:, 1],
        results["nodes_auto"][:, 2],
        c=results["strengths_auto"],
        cmap="YlOrRd",
        s=40,
        edgecolors="black",
        linewidths=0.3,
        vmin=1.0,
        vmax=6.0,
        zorder=5,
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(0, 1)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title(f"Auto-detection (strength colored)\n{len(results['nodes_auto'])} nodes, {results['n_auto_attention']} auto-attention")
    ax2.view_init(elev=20, azim=45)

    # Right: Surface type coloring
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax3.scatter(results["data"][:, 0], results["data"][:, 1], results["data"][:, 2],
                c="lightgray", s=1, alpha=0.2)

    colors = []
    for st in results["surface_types"]:
        if st in (SurfaceType.STABLE_CORNER, SurfaceType.CORNER):
            colors.append("red")
        elif st in (SurfaceType.STABLE_EDGE, SurfaceType.EDGE):
            colors.append("yellow")
        elif st in (SurfaceType.STABLE_PLANE, SurfaceType.PLANE):
            colors.append("green")
        else:
            colors.append("gray")

    for i, j in results["edges_auto"]:
        ax3.plot(
            [results["nodes_auto"][i, 0], results["nodes_auto"][j, 0]],
            [results["nodes_auto"][i, 1], results["nodes_auto"][j, 1]],
            [results["nodes_auto"][i, 2], results["nodes_auto"][j, 2]],
            "gray", linewidth=0.3, alpha=0.5,
        )

    for idx in range(len(results["nodes_auto"])):
        marker = "^" if results["auto_attention"][idx] else "o"
        size = 80 if results["auto_attention"][idx] else 40
        ax3.scatter(
            [results["nodes_auto"][idx, 0]],
            [results["nodes_auto"][idx, 1]],
            [results["nodes_auto"][idx, 2]],
            c=[colors[idx]],
            s=size,
            marker=marker,
            edgecolors="black",
            linewidths=0.3,
            zorder=5,
        )
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_zlim(0, 1)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("Surface Classification\nGreen=Plane, Yellow=Edge, Red=Corner")
    ax3.view_init(elev=20, azim=45)

    # Bottom row: Statistics
    # Left: Surface type distribution
    ax4 = fig.add_subplot(2, 3, 4)
    types = list(results["type_counts"].keys())
    counts = list(results["type_counts"].values())
    colors_bar = ["green", "yellow", "red", "gray"]
    bars = ax4.bar(types, counts, color=colors_bar)
    ax4.set_xlabel("Surface Type")
    ax4.set_ylabel("Count")
    ax4.set_title("Surface Type Distribution")
    for bar, count in zip(bars, counts):
        ax4.annotate(f"{count}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    # Middle: Strength histogram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(results["strengths_auto"], bins=20, color="coral", edgecolor="black", alpha=0.7)
    ax5.axvline(x=1.0, color="blue", linestyle="--", label="Base strength")
    ax5.axvline(x=6.0, color="red", linestyle="--", label="Auto-attention strength")
    ax5.set_xlabel("Strength")
    ax5.set_ylabel("Count")
    ax5.set_title("Node Strength Distribution")
    ax5.legend()

    # Right: Comparison bar chart
    ax6 = fig.add_subplot(2, 3, 6)
    comparison = {
        "Corner\nRegion": [results["manual_in_corner"], results["auto_in_corner"]],
        "Total\nNodes": [len(results["nodes_manual"]), len(results["nodes_auto"])],
    }
    x = np.arange(len(comparison))
    width = 0.35
    manual_vals = [v[0] for v in comparison.values()]
    auto_vals = [v[1] for v in comparison.values()]

    bars1 = ax6.bar(x - width/2, manual_vals, width, label="Manual", color="blue", alpha=0.7)
    bars2 = ax6.bar(x + width/2, auto_vals, width, label="Auto-detect", color="red", alpha=0.7)

    ax6.set_ylabel("Count")
    ax6.set_title("Corner Region Density Comparison")
    ax6.set_xticks(x)
    ax6.set_xticklabels(comparison.keys())
    ax6.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax6.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle("DD-GNG Automatic Attention Detection - Final Results", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "auto_detection.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'auto_detection.png'}")
    plt.close()


def main():
    """Run test and create visualizations."""
    results = run_auto_detection_test()
    create_visualizations(results)

    # Return pass/fail
    # Pass if:
    # 1. Auto-detection found some stable corners (n_auto_attention > 0)
    # 2. Surface classification is working (has both planes and corners)
    # 3. Auto-detected nodes have elevated strength (max strength > base)
    has_auto_attention = results["n_auto_attention"] > 0
    has_surface_classification = (
        results["type_counts"]["plane"] > 0 and
        results["type_counts"]["corner"] > 0
    )
    has_elevated_strength = results["strengths_auto"].max() > 1.5

    success = has_auto_attention and has_surface_classification and has_elevated_strength

    print(f"\nPass Criteria:")
    print(f"  Auto-attention nodes found: {has_auto_attention} ({results['n_auto_attention']} nodes)")
    print(f"  Surface classification working: {has_surface_classification}")
    print(f"  Elevated strength for corners: {has_elevated_strength} (max={results['strengths_auto'].max():.2f})")

    return success


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    print(f"Test 4 Result: {'PASS' if success else 'FAIL'}")
    print(f"{'='*60}")
