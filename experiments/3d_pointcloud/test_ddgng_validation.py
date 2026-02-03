"""DD-GNG implementation validation tests.

Verifies that the DD-GNG implementation correctly reproduces the behavior
described in Saputra et al. (2019) paper:

1. Strength calculation: Nodes in attention regions have higher strength
2. Density control: More nodes are placed in attention regions
3. Ladder detection scenario: Simulate the paper's use case

Usage:
    python test_ddgng_validation.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Add paths for imports
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data" / "3d"))

from algorithms.dd_gng.python.model import DynamicDensityGNG, DDGNGParams, AttentionRegion
from sampler import sample_floor_and_wall


def test_strength_calculation():
    """Test 1: Verify strength calculation for nodes in attention regions."""
    print("=" * 60)
    print("Test 1: Strength Calculation Verification")
    print("=" * 60)

    # Create DD-GNG with attention region
    params = DDGNGParams(
        max_nodes=50,
        lambda_=50,
        max_age=50,
        strength_power=4,
        strength_scale=4.0,
        use_strength_learning=True,
        use_strength_insertion=True,
    )
    model = DynamicDensityGNG(n_dim=2, params=params, seed=42)

    # Add attention region at center with strength bonus = 5.0
    model.add_attention_region(
        center=[0.5, 0.5],
        size=[0.2, 0.2],
        strength=5.0,
    )

    # Generate uniform data
    np.random.seed(42)
    data = np.random.rand(500, 2)

    # Train
    model.train(data, n_iterations=2000)

    # Get node positions and strengths
    nodes, _ = model.get_graph()
    strengths = model.get_node_strengths()

    # Check nodes inside vs outside attention region
    region = model.attention_regions[0]
    inside_mask = np.array([region.contains(node) for node in nodes])

    inside_strengths = strengths[inside_mask]
    outside_strengths = strengths[~inside_mask]

    print(f"\nNodes inside attention region: {inside_mask.sum()}")
    print(f"Nodes outside attention region: {(~inside_mask).sum()}")

    if len(inside_strengths) > 0:
        print(f"\nInside region strengths:")
        print(f"  Mean: {inside_strengths.mean():.2f}")
        print(f"  Expected: 6.0 (base 1.0 + bonus 5.0)")
        print(f"  Match: {'PASS' if np.allclose(inside_strengths.mean(), 6.0, atol=0.1) else 'FAIL'}")

    if len(outside_strengths) > 0:
        print(f"\nOutside region strengths:")
        print(f"  Mean: {outside_strengths.mean():.2f}")
        print(f"  Expected: 1.0 (base only)")
        print(f"  Match: {'PASS' if np.allclose(outside_strengths.mean(), 1.0, atol=0.1) else 'FAIL'}")

    return inside_mask.sum(), (~inside_mask).sum(), inside_strengths, outside_strengths


def test_density_comparison():
    """Test 2: Compare node density between DD-GNG and GNG-U2."""
    print("\n" + "=" * 60)
    print("Test 2: Density Comparison (DD-GNG vs GNG-U2)")
    print("=" * 60)

    # Import GNG-U2 for comparison
    from algorithms.gng_u2.python.model import GrowingNeuralGasU2, GNGU2Params

    # Generate uniform 2D data
    np.random.seed(42)
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

    # Train GNG-U2 (baseline)
    gngu2_params = GNGU2Params(**common_params)
    gngu2 = GrowingNeuralGasU2(n_dim=2, params=gngu2_params, seed=42)
    gngu2.train(data, n_iterations=5000)
    gngu2_nodes, _ = gngu2.get_graph()

    # Train DD-GNG with attention region
    ddgng_params = DDGNGParams(
        **common_params,
        strength_power=4,
        strength_scale=4.0,
        use_strength_learning=True,
        use_strength_insertion=True,
    )
    ddgng = DynamicDensityGNG(n_dim=2, params=ddgng_params, seed=42)
    ddgng.add_attention_region(
        center=attention_center.tolist(),
        size=attention_size.tolist(),
        strength=5.0,
    )
    ddgng.train(data, n_iterations=5000)
    ddgng_nodes, _ = ddgng.get_graph()

    # Count nodes in attention region
    def count_in_region(nodes, center, size):
        inside = np.all(np.abs(nodes - center) <= size, axis=1)
        return inside.sum()

    # Calculate region areas
    attention_area = (2 * attention_size[0]) * (2 * attention_size[1])
    total_area = 1.0  # Unit square
    outside_area = total_area - attention_area

    # GNG-U2 counts
    gngu2_inside = count_in_region(gngu2_nodes, attention_center, attention_size)
    gngu2_outside = len(gngu2_nodes) - gngu2_inside
    gngu2_density_inside = gngu2_inside / attention_area
    gngu2_density_outside = gngu2_outside / outside_area

    # DD-GNG counts
    ddgng_inside = count_in_region(ddgng_nodes, attention_center, attention_size)
    ddgng_outside = len(ddgng_nodes) - ddgng_inside
    ddgng_density_inside = ddgng_inside / attention_area
    ddgng_density_outside = ddgng_outside / outside_area

    print(f"\nAttention region: center={attention_center}, size={attention_size}")
    print(f"Attention area: {attention_area:.4f}, Outside area: {outside_area:.4f}")

    print(f"\n{'Algorithm':<12} {'Total':<8} {'Inside':<8} {'Outside':<8} {'Density In':<12} {'Density Out':<12}")
    print("-" * 70)
    print(f"{'GNG-U2':<12} {len(gngu2_nodes):<8} {gngu2_inside:<8} {gngu2_outside:<8} {gngu2_density_inside:<12.2f} {gngu2_density_outside:<12.2f}")
    print(f"{'DD-GNG':<12} {len(ddgng_nodes):<8} {ddgng_inside:<8} {ddgng_outside:<8} {ddgng_density_inside:<12.2f} {ddgng_density_outside:<12.2f}")

    # Calculate density ratio improvement
    gngu2_ratio = gngu2_density_inside / gngu2_density_outside if gngu2_density_outside > 0 else 0
    ddgng_ratio = ddgng_density_inside / ddgng_density_outside if ddgng_density_outside > 0 else 0

    print(f"\nDensity ratio (inside/outside):")
    print(f"  GNG-U2: {gngu2_ratio:.2f}")
    print(f"  DD-GNG: {ddgng_ratio:.2f}")
    print(f"  Improvement: {ddgng_ratio / gngu2_ratio:.2f}x" if gngu2_ratio > 0 else "  N/A")

    # Verification
    density_improved = ddgng_ratio > gngu2_ratio
    print(f"\nDensity improvement in attention region: {'PASS' if density_improved else 'FAIL'}")

    return {
        "gngu2": (gngu2_nodes, gngu2_inside, gngu2_outside),
        "ddgng": (ddgng_nodes, ddgng_inside, ddgng_outside),
        "attention": (attention_center, attention_size),
    }


def test_ladder_scenario():
    """Test 3: Ladder detection scenario from the paper.

    The paper uses DD-GNG for detecting ladder rungs for a quadruped robot.
    We simulate this with horizontal bars (ladder rungs) where higher density
    is needed for precise foot placement.
    """
    print("\n" + "=" * 60)
    print("Test 3: Ladder Detection Scenario (Paper Reproduction)")
    print("=" * 60)

    np.random.seed(42)

    # Generate 3D point cloud with ladder-like structure
    # Background: random points in a box
    n_background = 1000
    background = np.random.rand(n_background, 3)
    background[:, 2] *= 2.0  # Extend in Z

    # Ladder rungs: horizontal bars at specific heights
    n_rung_points = 200
    rung_heights = [0.3, 0.6, 0.9, 1.2, 1.5]
    ladder_points = []

    for h in rung_heights:
        # Each rung is a horizontal line at height h
        rung = np.zeros((n_rung_points, 3))
        rung[:, 0] = np.random.uniform(0.4, 0.6, n_rung_points)  # X: narrow range
        rung[:, 1] = np.random.uniform(0.3, 0.7, n_rung_points)  # Y: depth
        rung[:, 2] = h + np.random.normal(0, 0.02, n_rung_points)  # Z: at rung height
        ladder_points.append(rung)

    ladder_points = np.vstack(ladder_points)
    all_points = np.vstack([background, ladder_points])

    print(f"Generated {len(all_points)} points:")
    print(f"  - Background: {n_background}")
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

    model = DynamicDensityGNG(n_dim=3, params=params, seed=42)

    # Add attention regions for each ladder rung
    for h in rung_heights:
        model.add_attention_region(
            center=[0.5, 0.5, h],
            size=[0.15, 0.25, 0.05],  # Narrow in X and Z, wider in Y
            strength=5.0,
        )

    print(f"\nAdded {len(model.attention_regions)} attention regions for ladder rungs")

    # Train
    model.train(all_points, n_iterations=8000)
    nodes, edges = model.get_graph()
    strengths = model.get_node_strengths()

    # Analyze results
    high_strength_nodes = strengths > 3.0  # Nodes with significant strength bonus
    rung_node_count = high_strength_nodes.sum()

    print(f"\nResults:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  High-strength nodes (in rung regions): {rung_node_count}")
    print(f"  Percentage on rungs: {100 * rung_node_count / len(nodes):.1f}%")
    print(f"  Max strength: {strengths.max():.1f}")

    # Verify: rungs should have higher node density
    # Check if nodes are concentrated near rung heights
    rung_node_z = nodes[high_strength_nodes, 2]
    print(f"\nHigh-strength node Z distribution:")
    for h in rung_heights:
        near_rung = np.sum(np.abs(rung_node_z - h) < 0.1)
        print(f"  Near height {h:.1f}: {near_rung} nodes")

    # Success criteria: at least 30% of nodes should be high-strength (on rungs)
    success = rung_node_count / len(nodes) > 0.25
    print(f"\nLadder rung detection: {'PASS' if success else 'FAIL'}")
    print(f"  (Expected: >25% of nodes on ladder rungs)")

    return nodes, edges, strengths, all_points, rung_heights


def visualize_results(density_results, ladder_results, output_path="ddgng_validation.png"):
    """Visualize all test results."""
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Density comparison (2D)
    ax1 = fig.add_subplot(1, 3, 1)
    gngu2_nodes, gngu2_in, gngu2_out = density_results["gngu2"]
    ddgng_nodes, ddgng_in, ddgng_out = density_results["ddgng"]
    center, size = density_results["attention"]

    ax1.scatter(gngu2_nodes[:, 0], gngu2_nodes[:, 1], c="blue", s=30, alpha=0.6, label=f"GNG-U2 ({len(gngu2_nodes)})")
    rect = plt.Rectangle(
        (center[0] - size[0], center[1] - size[1]),
        size[0] * 2,
        size[1] * 2,
        fill=False,
        edgecolor="orange",
        linewidth=2,
        linestyle="--",
        label="Attention region",
    )
    ax1.add_patch(rect)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_title(f"GNG-U2 (inside: {gngu2_in}, outside: {gngu2_out})")
    ax1.legend()

    # Plot 2: DD-GNG density (2D)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(ddgng_nodes[:, 0], ddgng_nodes[:, 1], c="red", s=30, alpha=0.6, label=f"DD-GNG ({len(ddgng_nodes)})")
    rect2 = plt.Rectangle(
        (center[0] - size[0], center[1] - size[1]),
        size[0] * 2,
        size[1] * 2,
        fill=False,
        edgecolor="orange",
        linewidth=2,
        linestyle="--",
    )
    ax2.add_patch(rect2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.set_title(f"DD-GNG (inside: {ddgng_in}, outside: {ddgng_out})")
    ax2.legend()

    # Plot 3: Ladder scenario (3D side view)
    ax3 = fig.add_subplot(1, 3, 3)
    nodes, edges, strengths, points, rung_heights = ladder_results

    # Plot points (light)
    ax3.scatter(points[:, 1], points[:, 2], c="lightgray", s=1, alpha=0.3, label="Data")

    # Plot nodes colored by strength
    scatter = ax3.scatter(
        nodes[:, 1],
        nodes[:, 2],
        c=strengths,
        cmap="YlOrRd",
        s=40,
        edgecolors="black",
        linewidths=0.3,
        zorder=5,
    )
    plt.colorbar(scatter, ax=ax3, label="Strength")

    # Draw rung regions
    for h in rung_heights:
        ax3.axhline(y=h, color="orange", linestyle="--", alpha=0.5)

    ax3.set_xlabel("Y (depth)")
    ax3.set_ylabel("Z (height)")
    ax3.set_title(f"Ladder Scenario ({len(nodes)} nodes)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization: {output_path}")
    plt.close()


def main():
    """Run all validation tests."""
    print("DD-GNG Implementation Validation Tests")
    print("Based on: Saputra et al. (2019)")
    print("=" * 60)

    # Test 1: Strength calculation
    test_strength_calculation()

    # Test 2: Density comparison
    density_results = test_density_comparison()

    # Test 3: Ladder scenario
    ladder_results = test_ladder_scenario()

    # Visualize
    visualize_results(density_results, ladder_results)

    print("\n" + "=" * 60)
    print("All validation tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
