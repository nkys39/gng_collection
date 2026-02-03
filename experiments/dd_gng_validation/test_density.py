"""Test 2: DD-GNG vs GNG-U2 Density Comparison.

Compares node density distribution between DD-GNG and standard GNG-U2.

Expected results:
- DD-GNG should have significantly higher node density in the attention region
- The density ratio (inside/outside) should be much higher for DD-GNG
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.dd_gng.python.model import DynamicDensityGNG, DDGNGParams
from algorithms.gng_u2.python.model import GrowingNeuralGasU2, GNGU2Params

OUTPUT_DIR = Path(__file__).parent / "outputs"


def run_density_comparison(seed: int = 42) -> dict:
    """Run density comparison test.

    Returns:
        dict with test results and data for visualization.
    """
    print("=" * 60)
    print("Test 2: Density Comparison (DD-GNG vs GNG-U2)")
    print("=" * 60)

    # Generate uniform 2D data
    np.random.seed(seed)
    data = np.random.rand(1000, 2)

    # Define attention region (center square)
    attention_center = np.array([0.5, 0.5])
    attention_size = np.array([0.15, 0.15])

    # Common parameters
    common_params = {
        "max_nodes": 80,
        "lambda_": 50,
        "eps_b": 0.1,
        "eps_n": 0.01,
        "alpha": 0.5,
        "beta": 0.005,
        "max_age": 80,
        "utility_k": 1000.0,
        "kappa": 10,
    }

    n_iterations = 5000
    frame_interval = 100

    # Collect frames for both algorithms
    gngu2_frames = []
    ddgng_frames = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def draw_frame(ax, nodes, edges, title, attention_center, attention_size, show_region=True):
        ax.clear()
        ax.scatter(data[:, 0], data[:, 1], c="lightgray", s=3, alpha=0.3)

        if show_region:
            rect = plt.Rectangle(
                (attention_center[0] - attention_size[0], attention_center[1] - attention_size[1]),
                attention_size[0] * 2,
                attention_size[1] * 2,
                fill=True,
                facecolor="orange",
                edgecolor="darkorange",
                alpha=0.2,
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(rect)

        for i, j in edges:
            ax.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                "gray",
                linewidth=0.5,
                alpha=0.5,
            )

        ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=40, edgecolors="black", linewidths=0.3, zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title)

    # Train GNG-U2
    print("\nTraining GNG-U2...")
    gngu2_params = GNGU2Params(**common_params)
    gngu2 = GrowingNeuralGasU2(n_dim=2, params=gngu2_params, seed=seed)

    def gngu2_callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            draw_frame(axes[0], nodes, edges, f"GNG-U2 - Iter {iteration} ({len(nodes)} nodes)",
                       attention_center, attention_size)
            axes[1].clear()
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
            axes[1].set_title("DD-GNG (waiting...)")

            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
            gngu2_frames.append(img.convert("RGB"))

    gngu2.train(data, n_iterations=n_iterations, callback=gngu2_callback)
    gngu2_nodes, gngu2_edges = gngu2.get_graph()

    # Train DD-GNG
    print("Training DD-GNG...")
    ddgng_params = DDGNGParams(
        **common_params,
        strength_power=4,
        strength_scale=4.0,
        use_strength_learning=True,
        use_strength_insertion=True,
    )
    ddgng = DynamicDensityGNG(n_dim=2, params=ddgng_params, seed=seed)
    ddgng.add_attention_region(
        center=attention_center.tolist(),
        size=attention_size.tolist(),
        strength=5.0,
    )

    def ddgng_callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            draw_frame(axes[0], gngu2_nodes, gngu2_edges, f"GNG-U2 ({len(gngu2_nodes)} nodes)",
                       attention_center, attention_size)
            draw_frame(axes[1], nodes, edges, f"DD-GNG - Iter {iteration} ({len(nodes)} nodes)",
                       attention_center, attention_size)

            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
            ddgng_frames.append(img.convert("RGB"))

    ddgng.train(data, n_iterations=n_iterations, callback=ddgng_callback)
    ddgng_nodes, ddgng_edges = ddgng.get_graph()

    plt.close(fig)

    # Calculate density metrics
    def count_in_region(nodes, center, size):
        inside = np.all(np.abs(nodes - center) <= size, axis=1)
        return inside.sum()

    attention_area = (2 * attention_size[0]) * (2 * attention_size[1])
    total_area = 1.0
    outside_area = total_area - attention_area

    gngu2_inside = count_in_region(gngu2_nodes, attention_center, attention_size)
    gngu2_outside = len(gngu2_nodes) - gngu2_inside
    gngu2_density_inside = gngu2_inside / attention_area
    gngu2_density_outside = gngu2_outside / outside_area

    ddgng_inside = count_in_region(ddgng_nodes, attention_center, attention_size)
    ddgng_outside = len(ddgng_nodes) - ddgng_inside
    ddgng_density_inside = ddgng_inside / attention_area
    ddgng_density_outside = ddgng_outside / outside_area

    gngu2_ratio = gngu2_density_inside / gngu2_density_outside if gngu2_density_outside > 0 else 0
    ddgng_ratio = ddgng_density_inside / ddgng_density_outside if ddgng_density_outside > 0 else 0

    results = {
        "data": data,
        "attention_center": attention_center,
        "attention_size": attention_size,
        "gngu2": {
            "nodes": gngu2_nodes,
            "edges": gngu2_edges,
            "inside": gngu2_inside,
            "outside": gngu2_outside,
            "density_inside": gngu2_density_inside,
            "density_outside": gngu2_density_outside,
            "ratio": gngu2_ratio,
        },
        "ddgng": {
            "nodes": ddgng_nodes,
            "edges": ddgng_edges,
            "inside": ddgng_inside,
            "outside": ddgng_outside,
            "density_inside": ddgng_density_inside,
            "density_outside": ddgng_density_outside,
            "ratio": ddgng_ratio,
        },
        "frames": ddgng_frames,  # Combined frames showing both
        "improvement": ddgng_ratio / gngu2_ratio if gngu2_ratio > 0 else 0,
    }

    # Print results
    print(f"\nAttention region: center={attention_center}, size={attention_size}")
    print(f"Attention area: {attention_area:.4f}, Outside area: {outside_area:.4f}")

    print(f"\n{'Algorithm':<12} {'Total':<8} {'Inside':<8} {'Outside':<8} {'Density In':<12} {'Density Out':<12} {'Ratio':<8}")
    print("-" * 80)
    print(f"{'GNG-U2':<12} {len(gngu2_nodes):<8} {gngu2_inside:<8} {gngu2_outside:<8} {gngu2_density_inside:<12.2f} {gngu2_density_outside:<12.2f} {gngu2_ratio:<8.2f}")
    print(f"{'DD-GNG':<12} {len(ddgng_nodes):<8} {ddgng_inside:<8} {ddgng_outside:<8} {ddgng_density_inside:<12.2f} {ddgng_density_outside:<12.2f} {ddgng_ratio:<8.2f}")

    print(f"\nDensity ratio improvement: {results['improvement']:.2f}x")

    return results


def create_visualizations(results: dict) -> None:
    """Create and save visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save GIF
    if results["frames"]:
        frames = results["frames"]
        frames.extend([frames[-1]] * 15)  # Hold final frame
        frames[0].save(
            OUTPUT_DIR / "density_comparison.gif",
            save_all=True,
            append_images=frames[1:],
            duration=150,
            loop=0,
        )
        print(f"\nSaved: {OUTPUT_DIR / 'density_comparison.gif'}")

    # Create final comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    attention_center = results["attention_center"]
    attention_size = results["attention_size"]

    # Left: GNG-U2
    ax1 = axes[0]
    ax1.scatter(results["data"][:, 0], results["data"][:, 1], c="lightgray", s=3, alpha=0.3)

    rect1 = plt.Rectangle(
        (attention_center[0] - attention_size[0], attention_center[1] - attention_size[1]),
        attention_size[0] * 2,
        attention_size[1] * 2,
        fill=True,
        facecolor="orange",
        edgecolor="darkorange",
        alpha=0.2,
        linewidth=2,
        linestyle="--",
    )
    ax1.add_patch(rect1)

    for i, j in results["gngu2"]["edges"]:
        ax1.plot(
            [results["gngu2"]["nodes"][i, 0], results["gngu2"]["nodes"][j, 0]],
            [results["gngu2"]["nodes"][i, 1], results["gngu2"]["nodes"][j, 1]],
            "gray",
            linewidth=0.5,
            alpha=0.5,
        )

    ax1.scatter(
        results["gngu2"]["nodes"][:, 0],
        results["gngu2"]["nodes"][:, 1],
        c="blue",
        s=50,
        edgecolors="black",
        linewidths=0.3,
        zorder=5,
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(f"GNG-U2 ({len(results['gngu2']['nodes'])} nodes)\n"
                  f"Inside: {results['gngu2']['inside']}, Outside: {results['gngu2']['outside']}")

    # Middle: DD-GNG
    ax2 = axes[1]
    ax2.scatter(results["data"][:, 0], results["data"][:, 1], c="lightgray", s=3, alpha=0.3)

    rect2 = plt.Rectangle(
        (attention_center[0] - attention_size[0], attention_center[1] - attention_size[1]),
        attention_size[0] * 2,
        attention_size[1] * 2,
        fill=True,
        facecolor="orange",
        edgecolor="darkorange",
        alpha=0.2,
        linewidth=2,
        linestyle="--",
    )
    ax2.add_patch(rect2)

    for i, j in results["ddgng"]["edges"]:
        ax2.plot(
            [results["ddgng"]["nodes"][i, 0], results["ddgng"]["nodes"][j, 0]],
            [results["ddgng"]["nodes"][i, 1], results["ddgng"]["nodes"][j, 1]],
            "gray",
            linewidth=0.5,
            alpha=0.5,
        )

    ax2.scatter(
        results["ddgng"]["nodes"][:, 0],
        results["ddgng"]["nodes"][:, 1],
        c="red",
        s=50,
        edgecolors="black",
        linewidths=0.3,
        zorder=5,
    )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title(f"DD-GNG ({len(results['ddgng']['nodes'])} nodes)\n"
                  f"Inside: {results['ddgng']['inside']}, Outside: {results['ddgng']['outside']}")

    # Right: Bar chart comparison
    ax3 = axes[2]
    x = np.arange(2)
    width = 0.35

    gngu2_vals = [results["gngu2"]["density_inside"], results["gngu2"]["density_outside"]]
    ddgng_vals = [results["ddgng"]["density_inside"], results["ddgng"]["density_outside"]]

    bars1 = ax3.bar(x - width / 2, gngu2_vals, width, label="GNG-U2", color="blue", alpha=0.7)
    bars2 = ax3.bar(x + width / 2, ddgng_vals, width, label="DD-GNG", color="red", alpha=0.7)

    ax3.set_ylabel("Node Density (nodes/area)")
    ax3.set_title("Density Comparison")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Inside Region", "Outside Region"])
    ax3.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    # Add summary text
    summary = (
        f"Density Ratio (In/Out):\n"
        f"  GNG-U2: {results['gngu2']['ratio']:.2f}\n"
        f"  DD-GNG: {results['ddgng']['ratio']:.2f}\n"
        f"  Improvement: {results['improvement']:.2f}x"
    )
    ax3.text(0.95, 0.95, summary, transform=ax3.transAxes, fontsize=10,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "density_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'density_comparison.png'}")
    plt.close()


def main():
    """Run test and create visualizations."""
    results = run_density_comparison()
    create_visualizations(results)

    # Return pass/fail (DD-GNG should have higher density ratio)
    return results["ddgng"]["ratio"] > results["gngu2"]["ratio"]


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    print(f"Test 2 Result: {'PASS' if success else 'FAIL'}")
    print(f"{'='*60}")
