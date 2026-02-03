"""Test 1: DD-GNG Strength Calculation Verification.

Verifies that nodes in attention regions have correctly computed strength values.

Expected results:
- Nodes inside attention region: strength = base (1.0) + bonus (5.0) = 6.0
- Nodes outside attention region: strength = base (1.0)
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

OUTPUT_DIR = Path(__file__).parent / "outputs"


def run_strength_test(seed: int = 42) -> dict:
    """Run strength calculation test.

    Returns:
        dict with test results and data for visualization.
    """
    print("=" * 60)
    print("Test 1: Strength Calculation Verification")
    print("=" * 60)

    # Create DD-GNG with attention region
    params = DDGNGParams(
        max_nodes=60,
        lambda_=50,
        eps_b=0.1,
        eps_n=0.01,
        max_age=50,
        strength_power=4,
        strength_scale=4.0,
        use_strength_learning=True,
        use_strength_insertion=True,
    )
    model = DynamicDensityGNG(n_dim=2, params=params, seed=seed)

    # Add attention region at center with strength bonus = 5.0
    attention_center = np.array([0.5, 0.5])
    attention_size = np.array([0.2, 0.2])
    strength_bonus = 5.0

    model.add_attention_region(
        center=attention_center.tolist(),
        size=attention_size.tolist(),
        strength=strength_bonus,
    )

    # Generate uniform data
    np.random.seed(seed)
    data = np.random.rand(800, 2)

    # Collect frames for GIF
    frames = []
    frame_interval = 100

    fig, ax = plt.subplots(figsize=(8, 8))

    def callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == 2999:
            nodes, edges = m.get_graph()
            strengths = m.get_node_strengths()

            ax.clear()

            # Plot data points
            ax.scatter(data[:, 0], data[:, 1], c="lightgray", s=5, alpha=0.3, label="Data")

            # Plot attention region
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
                label="Attention Region",
            )
            ax.add_patch(rect)

            # Plot edges
            for i, j in edges:
                ax.plot(
                    [nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    "gray",
                    linewidth=0.5,
                    alpha=0.5,
                )

            # Plot nodes colored by strength
            scatter = ax.scatter(
                nodes[:, 0],
                nodes[:, 1],
                c=strengths,
                cmap="YlOrRd",
                s=80,
                edgecolors="black",
                linewidths=0.5,
                vmin=1.0,
                vmax=6.0,
                zorder=5,
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"DD-GNG Strength Test - Iter {iteration} ({len(nodes)} nodes)")
            ax.legend(loc="upper right")

            if iteration == 0:
                plt.colorbar(scatter, ax=ax, label="Strength")

            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
            frames.append(img.convert("RGB"))

    # Train
    model.train(data, n_iterations=3000, callback=callback)

    plt.close(fig)

    # Get final results
    nodes, edges = model.get_graph()
    strengths = model.get_node_strengths()

    # Analyze results
    region = model.attention_regions[0]
    inside_mask = np.array([region.contains(node) for node in nodes])

    inside_strengths = strengths[inside_mask]
    outside_strengths = strengths[~inside_mask]

    results = {
        "nodes": nodes,
        "edges": edges,
        "strengths": strengths,
        "inside_mask": inside_mask,
        "inside_count": inside_mask.sum(),
        "outside_count": (~inside_mask).sum(),
        "inside_mean": inside_strengths.mean() if len(inside_strengths) > 0 else 0,
        "outside_mean": outside_strengths.mean() if len(outside_strengths) > 0 else 0,
        "attention_center": attention_center,
        "attention_size": attention_size,
        "data": data,
        "frames": frames,
    }

    # Print results
    print(f"\nNodes inside attention region: {results['inside_count']}")
    print(f"Nodes outside attention region: {results['outside_count']}")

    if results["inside_count"] > 0:
        print(f"\nInside region strengths:")
        print(f"  Mean: {results['inside_mean']:.2f}")
        print(f"  Expected: 6.0 (base 1.0 + bonus 5.0)")
        print(f"  Match: {'PASS' if np.allclose(results['inside_mean'], 6.0, atol=0.1) else 'FAIL'}")

    if results["outside_count"] > 0:
        print(f"\nOutside region strengths:")
        print(f"  Mean: {results['outside_mean']:.2f}")
        print(f"  Expected: 1.0 (base only)")
        print(f"  Match: {'PASS' if np.allclose(results['outside_mean'], 1.0, atol=0.1) else 'FAIL'}")

    return results


def create_visualizations(results: dict) -> None:
    """Create and save visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save GIF
    if results["frames"]:
        frames = results["frames"]
        frames.extend([frames[-1]] * 10)  # Hold final frame
        frames[0].save(
            OUTPUT_DIR / "strength_test.gif",
            save_all=True,
            append_images=frames[1:],
            duration=150,
            loop=0,
        )
        print(f"\nSaved: {OUTPUT_DIR / 'strength_test.gif'}")

    # Create final figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Node distribution with strength coloring
    ax1 = axes[0]
    ax1.scatter(results["data"][:, 0], results["data"][:, 1], c="lightgray", s=5, alpha=0.3)

    rect = plt.Rectangle(
        (results["attention_center"][0] - results["attention_size"][0],
         results["attention_center"][1] - results["attention_size"][1]),
        results["attention_size"][0] * 2,
        results["attention_size"][1] * 2,
        fill=True,
        facecolor="orange",
        edgecolor="darkorange",
        alpha=0.2,
        linewidth=2,
        linestyle="--",
    )
    ax1.add_patch(rect)

    for i, j in results["edges"]:
        ax1.plot(
            [results["nodes"][i, 0], results["nodes"][j, 0]],
            [results["nodes"][i, 1], results["nodes"][j, 1]],
            "gray",
            linewidth=0.5,
            alpha=0.5,
        )

    scatter = ax1.scatter(
        results["nodes"][:, 0],
        results["nodes"][:, 1],
        c=results["strengths"],
        cmap="YlOrRd",
        s=100,
        edgecolors="black",
        linewidths=0.5,
        vmin=1.0,
        vmax=6.0,
        zorder=5,
    )
    plt.colorbar(scatter, ax=ax1, label="Strength")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(f"Node Distribution ({len(results['nodes'])} nodes)")

    # Right: Strength histogram
    ax2 = axes[1]
    inside_strengths = results["strengths"][results["inside_mask"]]
    outside_strengths = results["strengths"][~results["inside_mask"]]

    bins = np.linspace(0.5, 7, 14)
    ax2.hist(outside_strengths, bins=bins, alpha=0.7, label=f"Outside ({len(outside_strengths)})", color="blue")
    ax2.hist(inside_strengths, bins=bins, alpha=0.7, label=f"Inside ({len(inside_strengths)})", color="red")

    ax2.axvline(x=1.0, color="blue", linestyle="--", linewidth=2, label="Expected Outside (1.0)")
    ax2.axvline(x=6.0, color="red", linestyle="--", linewidth=2, label="Expected Inside (6.0)")

    ax2.set_xlabel("Strength")
    ax2.set_ylabel("Count")
    ax2.set_title("Strength Distribution")
    ax2.legend()

    # Add summary text
    summary = (
        f"Inside:  mean={results['inside_mean']:.2f} (expected: 6.0)\n"
        f"Outside: mean={results['outside_mean']:.2f} (expected: 1.0)"
    )
    ax2.text(0.95, 0.95, summary, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strength_test.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'strength_test.png'}")
    plt.close()


def main():
    """Run test and create visualizations."""
    results = run_strength_test()
    create_visualizations(results)

    # Return pass/fail
    inside_pass = np.allclose(results["inside_mean"], 6.0, atol=0.1)
    outside_pass = np.allclose(results["outside_mean"], 1.0, atol=0.1)
    return inside_pass and outside_pass


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    print(f"Test 1 Result: {'PASS' if success else 'FAIL'}")
    print(f"{'='*60}")
