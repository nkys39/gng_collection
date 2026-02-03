#!/usr/bin/env python3
"""Compare GSRM with GNG and GCS on 3D surface reconstruction.

This script compares the three methods on sphere and torus shapes,
highlighting the differences in their capabilities:
- GNG: Wireframe only (no faces)
- GCS: Mesh with topological constraints (can't learn holes)
- GSRM: Mesh with topology learning (can learn holes)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

from algorithms.gng.python.model import GrowingNeuralGas, GNGParams
from algorithms.gcs.python.model import GrowingCellStructures, GCSParams
from algorithms.gsrm.python.model import GSRM, GSRMParams

# Import 3D sampler
import importlib.util
sampler_3d_path = project_root / "data" / "3d" / "sampler.py"
spec = importlib.util.spec_from_file_location("sampler_3d", sampler_3d_path)
sampler_3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_3d)
sample_sphere = sampler_3d.sample_sphere
sample_torus = sampler_3d.sample_torus


def visualize_comparison(
    point_cloud: np.ndarray,
    results: dict,
    shape_name: str,
    output_path: Path,
) -> None:
    """Create comparison visualization.

    Args:
        point_cloud: Original point cloud.
        results: Dict with 'gng', 'gcs', 'gsrm' keys, each containing
                 'nodes', 'edges', 'faces', 'time', 'n_nodes', 'n_edges', 'n_faces'.
        shape_name: Name of the shape for title.
        output_path: Path to save the image.
    """
    fig = plt.figure(figsize=(18, 12))

    methods = ['gng', 'gcs', 'gsrm']
    titles = ['GNG (Wireframe)', 'GCS (Constrained Mesh)', 'GSRM (Learned Mesh)']
    colors = ['royalblue', 'forestgreen', 'coral']

    for col, (method, title, color) in enumerate(zip(methods, titles, colors)):
        data = results[method]
        nodes = data['nodes']
        edges = data['edges']
        faces = data.get('faces', [])

        # Row 1: Wireframe view
        ax1 = fig.add_subplot(2, 3, col + 1, projection='3d')
        ax1.scatter(
            point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
            c='lightgray', s=1, alpha=0.2
        )
        if len(nodes) > 0:
            ax1.scatter(
                nodes[:, 0], nodes[:, 1], nodes[:, 2],
                c=color, s=15, alpha=0.8
            )
            for i, j in edges:
                ax1.plot3D(
                    [nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    [nodes[i, 2], nodes[j, 2]],
                    c=color, alpha=0.4, linewidth=0.5
                )
        ax1.set_title(f"{title}\nNodes: {data['n_nodes']}, Edges: {data['n_edges']}")
        ax1.view_init(elev=30, azim=45)
        _set_equal_aspect(ax1, nodes if len(nodes) > 0 else point_cloud)

        # Row 2: Mesh view (faces)
        ax2 = fig.add_subplot(2, 3, col + 4, projection='3d')
        if faces:
            face_vertices = [[nodes[i], nodes[j], nodes[k]] for i, j, k in faces]
            poly = Poly3DCollection(
                face_vertices,
                alpha=0.7,
                facecolor=color,
                edgecolor='darkgray',
                linewidth=0.2
            )
            ax2.add_collection3d(poly)
            ax2.scatter(
                nodes[:, 0], nodes[:, 1], nodes[:, 2],
                c='black', s=5, alpha=0.5
            )
            ax2.set_title(f"Faces: {data['n_faces']}\nTime: {data['time']:.2f}s")
        else:
            ax2.text(
                0.5, 0.5, 0.5,
                "No faces\n(Wireframe only)",
                ha='center', va='center',
                fontsize=14, color='gray'
            )
            ax2.set_title(f"Faces: 0\nTime: {data['time']:.2f}s")

        ax2.view_init(elev=30, azim=45)
        _set_equal_aspect(ax2, nodes if len(nodes) > 0 else point_cloud)

    plt.suptitle(f"{shape_name} Surface Reconstruction Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _set_equal_aspect(ax, points):
    """Set equal aspect ratio for 3D axes."""
    if len(points) == 0:
        return
    max_range = np.max([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min(),
    ]) / 2
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def run_gng(point_cloud: np.ndarray, max_nodes: int, n_iterations: int, seed: int) -> dict:
    """Run GNG on point cloud."""
    params = GNGParams(
        max_nodes=max_nodes,
        lambda_=50,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
        max_age=50,
    )
    gng = GrowingNeuralGas(n_dim=3, params=params, seed=seed)

    start_time = time.time()
    gng.train(point_cloud, n_iterations=n_iterations)
    elapsed = time.time() - start_time

    nodes, edges = gng.get_graph()
    return {
        'nodes': nodes,
        'edges': edges,
        'faces': [],
        'n_nodes': gng.n_nodes,
        'n_edges': gng.n_edges,
        'n_faces': 0,
        'time': elapsed,
    }


def run_gcs(point_cloud: np.ndarray, max_nodes: int, n_iterations: int, seed: int) -> dict:
    """Run GCS on point cloud."""
    params = GCSParams(
        max_nodes=max_nodes,
        lambda_=50,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
    )
    gcs = GrowingCellStructures(n_dim=3, params=params, seed=seed)

    start_time = time.time()
    gcs.train(point_cloud, n_iterations=n_iterations)
    elapsed = time.time() - start_time

    nodes, edges = gcs.get_graph()

    # GCS doesn't explicitly track faces, but we can infer triangles
    # from the mesh structure (find all triangles in the graph)
    faces = _find_triangles(nodes, edges)

    return {
        'nodes': nodes,
        'edges': edges,
        'faces': faces,
        'n_nodes': gcs.n_nodes,
        'n_edges': gcs.n_edges,
        'n_faces': len(faces),
        'time': elapsed,
    }


def _find_triangles(nodes: np.ndarray, edges: list) -> list:
    """Find all triangles in a graph."""
    if len(nodes) == 0:
        return []

    # Build adjacency set
    adj = {i: set() for i in range(len(nodes))}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    # Find triangles
    triangles = set()
    for i in range(len(nodes)):
        for j in adj[i]:
            if j > i:
                # Find common neighbors
                common = adj[i] & adj[j]
                for k in common:
                    if k > j:
                        triangles.add(tuple(sorted([i, j, k])))

    return list(triangles)


def run_gsrm(point_cloud: np.ndarray, max_nodes: int, n_iterations: int, seed: int) -> dict:
    """Run GSRM on point cloud."""
    params = GSRMParams(
        max_nodes=max_nodes,
        lambda_=50,
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
    return {
        'nodes': nodes,
        'edges': edges,
        'faces': faces,
        'n_nodes': gsrm.n_nodes,
        'n_edges': gsrm.n_edges,
        'n_faces': gsrm.n_faces,
        'time': elapsed,
    }


def main():
    """Run comparison experiments."""
    print("=" * 60)
    print("GSRM vs GNG vs GCS Comparison")
    print("=" * 60)

    # Create output directory
    output_dir = Path(__file__).parent / "samples" / "gsrm" / "python"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    max_nodes = 200
    n_iterations = 10000
    n_samples = 5000
    seed = 42

    print(f"\nParameters:")
    print(f"  max_nodes: {max_nodes}")
    print(f"  n_iterations: {n_iterations}")
    print(f"  n_samples: {n_samples}")

    # ========== SPHERE TEST ==========
    print("\n" + "-" * 60)
    print("SPHERE TEST")
    print("-" * 60)

    print("\nGenerating sphere point cloud...")
    sphere_cloud = sample_sphere(n_samples=n_samples, seed=seed)
    print(f"  Generated {len(sphere_cloud)} points")

    print("\nRunning GNG...")
    gng_result = run_gng(sphere_cloud, max_nodes, n_iterations, seed)
    print(f"  GNG: {gng_result['n_nodes']} nodes, {gng_result['n_edges']} edges, "
          f"{gng_result['n_faces']} faces ({gng_result['time']:.2f}s)")

    print("\nRunning GCS...")
    gcs_result = run_gcs(sphere_cloud, max_nodes, n_iterations, seed)
    print(f"  GCS: {gcs_result['n_nodes']} nodes, {gcs_result['n_edges']} edges, "
          f"{gcs_result['n_faces']} faces ({gcs_result['time']:.2f}s)")

    print("\nRunning GSRM...")
    gsrm_result = run_gsrm(sphere_cloud, max_nodes, n_iterations, seed)
    print(f"  GSRM: {gsrm_result['n_nodes']} nodes, {gsrm_result['n_edges']} edges, "
          f"{gsrm_result['n_faces']} faces ({gsrm_result['time']:.2f}s)")

    print("\nCreating comparison visualization...")
    visualize_comparison(
        sphere_cloud,
        {'gng': gng_result, 'gcs': gcs_result, 'gsrm': gsrm_result},
        "Sphere",
        output_dir / "compare_sphere.png"
    )

    # ========== TORUS TEST ==========
    print("\n" + "-" * 60)
    print("TORUS TEST")
    print("-" * 60)

    print("\nGenerating torus point cloud...")
    torus_cloud = sample_torus(
        n_samples=n_samples,
        major_radius=0.3,
        minor_radius=0.12,
        seed=seed
    )
    print(f"  Generated {len(torus_cloud)} points")

    print("\nRunning GNG...")
    gng_result = run_gng(torus_cloud, max_nodes, n_iterations, seed)
    print(f"  GNG: {gng_result['n_nodes']} nodes, {gng_result['n_edges']} edges, "
          f"{gng_result['n_faces']} faces ({gng_result['time']:.2f}s)")

    print("\nRunning GCS...")
    gcs_result = run_gcs(torus_cloud, max_nodes, n_iterations, seed)
    print(f"  GCS: {gcs_result['n_nodes']} nodes, {gcs_result['n_edges']} edges, "
          f"{gcs_result['n_faces']} faces ({gcs_result['time']:.2f}s)")

    print("\nRunning GSRM...")
    gsrm_result = run_gsrm(torus_cloud, max_nodes, n_iterations, seed)
    print(f"  GSRM: {gsrm_result['n_nodes']} nodes, {gsrm_result['n_edges']} edges, "
          f"{gsrm_result['n_faces']} faces ({gsrm_result['time']:.2f}s)")

    print("\nCreating comparison visualization...")
    visualize_comparison(
        torus_cloud,
        {'gng': gng_result, 'gcs': gcs_result, 'gsrm': gsrm_result},
        "Torus",
        output_dir / "compare_torus.png"
    )

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("""
| Feature              | GNG       | GCS       | GSRM      |
|----------------------|-----------|-----------|-----------|
| Wireframe            | Yes       | Yes       | Yes       |
| Triangle Faces       | No        | Yes       | Yes       |
| Topology Learning    | No        | No        | Yes       |
| Hole Reconstruction  | No        | No        | Yes       |

Key Observations:
1. GNG: Only produces wireframes (edges), no triangle faces
2. GCS: Produces mesh but can't learn topology with holes
3. GSRM: Produces mesh AND can learn topology with holes
""")
    print("=" * 60)
    print("Comparison completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
