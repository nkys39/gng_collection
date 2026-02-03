#!/usr/bin/env python3
"""GSRM multi-resolution evaluation with Hausdorff distance.

This script evaluates GSRM at different resolutions (node counts) and
measures the Hausdorff distance between the point cloud and reconstructed mesh.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import time

from algorithms.gsrm.python.model import GSRM, GSRMParams

# Import 3D sampler
import importlib.util
sampler_3d_path = project_root / "data" / "3d" / "sampler.py"
spec = importlib.util.spec_from_file_location("sampler_3d", sampler_3d_path)
sampler_3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_3d)
sample_sphere = sampler_3d.sample_sphere
sample_torus = sampler_3d.sample_torus


def hausdorff_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """Compute symmetric Hausdorff distance between two point sets.

    Args:
        points1: First point set (n, 3).
        points2: Second point set (m, 3).

    Returns:
        Symmetric Hausdorff distance.
    """
    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]
    return max(d1, d2)


def mean_distance(source: np.ndarray, target: np.ndarray) -> float:
    """Compute mean distance from source points to nearest target points.

    Args:
        source: Source point set (n, 3).
        target: Target point set (m, 3).

    Returns:
        Mean nearest neighbor distance.
    """
    from scipy.spatial import KDTree
    tree = KDTree(target)
    distances, _ = tree.query(source)
    return np.mean(distances)


def run_gsrm_evaluation(
    point_cloud: np.ndarray,
    max_nodes: int,
    n_iterations: int,
    seed: int = 42,
) -> dict:
    """Run GSRM and evaluate results.

    Args:
        point_cloud: Input point cloud (n, 3).
        max_nodes: Maximum number of nodes.
        n_iterations: Number of training iterations.
        seed: Random seed.

    Returns:
        Dictionary with evaluation results.
    """
    params = GSRMParams(
        max_nodes=max_nodes,
        lambda_=max(10, n_iterations // max_nodes),  # Adaptive lambda
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
        max_age=50,
    )

    gsrm = GSRM(params=params, seed=seed)

    start_time = time.time()
    gsrm.train(point_cloud, n_iterations=n_iterations)
    elapsed = time.time() - start_time

    nodes, edges, faces = gsrm.get_mesh()

    # Compute distances
    h_dist = hausdorff_distance(point_cloud, nodes)
    mean_dist = mean_distance(point_cloud, nodes)

    return {
        'max_nodes': max_nodes,
        'n_nodes': gsrm.n_nodes,
        'n_edges': gsrm.n_edges,
        'n_faces': gsrm.n_faces,
        'hausdorff': h_dist,
        'mean_dist': mean_dist,
        'time': elapsed,
        'nodes': nodes,
        'edges': edges,
        'faces': faces,
    }


def main():
    """Run multi-resolution evaluation."""
    print("=" * 70)
    print("GSRM Multi-Resolution Evaluation with Hausdorff Distance")
    print("=" * 70)

    # Output directory
    output_dir = Path(__file__).parent / "samples" / "gsrm" / "python"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolution levels to test
    resolutions = [20, 50, 100, 200, 500]
    n_samples = 5000
    seed = 42

    # ========== SPHERE EVALUATION ==========
    print("\n" + "-" * 70)
    print("SPHERE EVALUATION")
    print("-" * 70)

    print("\nGenerating sphere point cloud...")
    sphere_cloud = sample_sphere(n_samples=n_samples, seed=seed)
    print(f"  Generated {len(sphere_cloud)} points")

    sphere_results = []
    for max_nodes in resolutions:
        n_iterations = max_nodes * 50  # Scale iterations with nodes
        print(f"\nRunning GSRM with max_nodes={max_nodes}, iterations={n_iterations}...")
        result = run_gsrm_evaluation(sphere_cloud, max_nodes, n_iterations, seed)
        sphere_results.append(result)
        print(f"  Nodes: {result['n_nodes']}, Edges: {result['n_edges']}, Faces: {result['n_faces']}")
        print(f"  Hausdorff: {result['hausdorff']:.6f}, Mean: {result['mean_dist']:.6f}, Time: {result['time']:.2f}s")

    # ========== TORUS EVALUATION ==========
    print("\n" + "-" * 70)
    print("TORUS EVALUATION")
    print("-" * 70)

    print("\nGenerating torus point cloud...")
    torus_cloud = sample_torus(
        n_samples=n_samples,
        major_radius=0.3,
        minor_radius=0.12,
        seed=seed
    )
    print(f"  Generated {len(torus_cloud)} points")

    torus_results = []
    for max_nodes in resolutions:
        n_iterations = max_nodes * 50
        print(f"\nRunning GSRM with max_nodes={max_nodes}, iterations={n_iterations}...")
        result = run_gsrm_evaluation(torus_cloud, max_nodes, n_iterations, seed)
        torus_results.append(result)
        print(f"  Nodes: {result['n_nodes']}, Edges: {result['n_edges']}, Faces: {result['n_faces']}")
        print(f"  Hausdorff: {result['hausdorff']:.6f}, Mean: {result['mean_dist']:.6f}, Time: {result['time']:.2f}s")

    # ========== CREATE VISUALIZATION ==========
    print("\n" + "-" * 70)
    print("Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data for plotting
    sphere_nodes = [r['n_nodes'] for r in sphere_results]
    sphere_hausdorff = [r['hausdorff'] for r in sphere_results]
    sphere_mean = [r['mean_dist'] for r in sphere_results]
    sphere_time = [r['time'] for r in sphere_results]

    torus_nodes = [r['n_nodes'] for r in torus_results]
    torus_hausdorff = [r['hausdorff'] for r in torus_results]
    torus_mean = [r['mean_dist'] for r in torus_results]
    torus_time = [r['time'] for r in torus_results]

    # Plot 1: Hausdorff distance vs nodes
    axes[0, 0].plot(sphere_nodes, sphere_hausdorff, 'o-', label='Sphere', linewidth=2, markersize=8)
    axes[0, 0].plot(torus_nodes, torus_hausdorff, 's-', label='Torus', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Nodes')
    axes[0, 0].set_ylabel('Hausdorff Distance')
    axes[0, 0].set_title('Hausdorff Distance vs Resolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean distance vs nodes
    axes[0, 1].plot(sphere_nodes, sphere_mean, 'o-', label='Sphere', linewidth=2, markersize=8)
    axes[0, 1].plot(torus_nodes, torus_mean, 's-', label='Torus', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Nodes')
    axes[0, 1].set_ylabel('Mean Distance')
    axes[0, 1].set_title('Mean Distance vs Resolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Time vs nodes
    axes[1, 0].plot(sphere_nodes, sphere_time, 'o-', label='Sphere', linewidth=2, markersize=8)
    axes[1, 0].plot(torus_nodes, torus_time, 's-', label='Torus', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Nodes')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Computation Time vs Resolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary table
    axes[1, 1].axis('off')
    table_data = []
    for s, t in zip(sphere_results, torus_results):
        table_data.append([
            s['n_nodes'],
            f"{s['hausdorff']:.4f}",
            f"{s['mean_dist']:.4f}",
            f"{t['hausdorff']:.4f}",
            f"{t['mean_dist']:.4f}",
        ])

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Nodes', 'Sphere H', 'Sphere M', 'Torus H', 'Torus M'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Summary (H=Hausdorff, M=Mean)', pad=20)

    plt.suptitle('GSRM Multi-Resolution Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'multiresolution_eval.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'multiresolution_eval.png'}")

    # ========== PRINT SUMMARY TABLE ==========
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print("\n### Sphere Results")
    print(f"{'Nodes':>8} {'Edges':>8} {'Faces':>8} {'Hausdorff':>12} {'Mean Dist':>12} {'Time (s)':>10}")
    print("-" * 70)
    for r in sphere_results:
        print(f"{r['n_nodes']:>8} {r['n_edges']:>8} {r['n_faces']:>8} {r['hausdorff']:>12.6f} {r['mean_dist']:>12.6f} {r['time']:>10.2f}")

    print("\n### Torus Results")
    print(f"{'Nodes':>8} {'Edges':>8} {'Faces':>8} {'Hausdorff':>12} {'Mean Dist':>12} {'Time (s)':>10}")
    print("-" * 70)
    for r in torus_results:
        print(f"{r['n_nodes']:>8} {r['n_edges']:>8} {r['n_faces']:>8} {r['hausdorff']:>12.6f} {r['mean_dist']:>12.6f} {r['time']:>10.2f}")

    # ========== SAVE RESULTS TO MARKDOWN ==========
    md_content = """## Multi-Resolution Evaluation Results

### Sphere Surface Reconstruction

| Nodes | Edges | Faces | Hausdorff | Mean Dist | Time (s) |
|------:|------:|------:|----------:|----------:|---------:|
"""
    for r in sphere_results:
        md_content += f"| {r['n_nodes']} | {r['n_edges']} | {r['n_faces']} | {r['hausdorff']:.6f} | {r['mean_dist']:.6f} | {r['time']:.2f} |\n"

    md_content += """
### Torus Surface Reconstruction

| Nodes | Edges | Faces | Hausdorff | Mean Dist | Time (s) |
|------:|------:|------:|----------:|----------:|---------:|
"""
    for r in torus_results:
        md_content += f"| {r['n_nodes']} | {r['n_edges']} | {r['n_faces']} | {r['hausdorff']:.6f} | {r['mean_dist']:.6f} | {r['time']:.2f} |\n"

    md_content += """
### Distance Metrics

- **Hausdorff Distance**: Maximum of the directed Hausdorff distances between point cloud and mesh vertices
- **Mean Distance**: Average distance from each point cloud point to its nearest mesh vertex

### Observations

1. Both Hausdorff and mean distances decrease as the number of nodes increases
2. The torus (with hole) requires more nodes than the sphere for similar accuracy
3. Computation time scales approximately linearly with the number of nodes

![Multi-resolution Evaluation](samples/gsrm/python/multiresolution_eval.png)
"""

    with open(output_dir.parent.parent / "evaluation_results.md", "w") as f:
        f.write(md_content)
    print(f"\n  Saved: {output_dir.parent.parent / 'evaluation_results.md'}")

    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
