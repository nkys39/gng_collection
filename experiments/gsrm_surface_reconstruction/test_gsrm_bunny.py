#!/usr/bin/env python3
"""GSRM Stanford Bunny surface reconstruction test.

This script tests GSRM on the Stanford Bunny point cloud to verify
the ability to reconstruct complex surfaces with concave regions.

Requirements:
    - Place bunny.ply in experiments/gsrm_surface_reconstruction/data/
    - Download from: https://graphics.stanford.edu/data/3Dscanrep/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.distance import directed_hausdorff

from algorithms.gsrm.python.model import GSRM, GSRMParams


def load_ply(filepath: str) -> np.ndarray:
    """Load vertices from a PLY file.

    Args:
        filepath: Path to PLY file.

    Returns:
        Array of shape (n_vertices, 3) with (x, y, z) coordinates.
    """
    vertices = []
    with open(filepath, 'r') as f:
        # Read header
        in_header = True
        vertex_count = 0

        for line in f:
            line = line.strip()

            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line == 'end_header':
                    in_header = False
                continue

            # Read vertex data
            if len(vertices) < vertex_count:
                parts = line.split()
                if len(parts) >= 3:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.array(vertices)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalize points to [0, 1] range.

    Args:
        points: Array of shape (n, 3).

    Returns:
        Normalized points in [0, 1] range.
    """
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals

    # Center and scale to [0.1, 0.9] to leave margin
    normalized = (points - min_vals) / range_vals.max() * 0.8 + 0.1
    return normalized


def hausdorff_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """Compute symmetric Hausdorff distance."""
    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]
    return max(d1, d2)


def visualize_mesh_3d(
    ax: Axes3D,
    nodes: np.ndarray,
    edges: list,
    faces: list,
    point_cloud: np.ndarray | None = None,
    show_points: bool = True,
    show_edges: bool = True,
    show_faces: bool = True,
    elev: float = 20,
    azim: float = 45,
    title: str = "",
) -> None:
    """Visualize 3D mesh on given axes."""
    ax.clear()

    # Plot point cloud
    if point_cloud is not None and show_points:
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            c="lightgray",
            s=0.5,
            alpha=0.2,
        )

    # Plot nodes
    if len(nodes) > 0 and not show_faces:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            c="blue",
            s=10,
            alpha=0.8,
        )

    # Plot edges
    if show_edges and edges and not show_faces:
        for i, j in edges:
            ax.plot3D(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                [nodes[i, 2], nodes[j, 2]],
                "b-",
                alpha=0.3,
                linewidth=0.3,
            )

    # Plot faces
    if show_faces and faces:
        face_vertices = []
        for i, j, k in faces:
            face_vertices.append([nodes[i], nodes[j], nodes[k]])

        poly = Poly3DCollection(
            face_vertices,
            alpha=0.7,
            facecolor="sandybrown",
            edgecolor="saddlebrown",
            linewidth=0.2,
        )
        ax.add_collection3d(poly)

    # Set view and labels
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set equal aspect ratio
    if len(nodes) > 0:
        max_range = np.max([
            nodes[:, 0].max() - nodes[:, 0].min(),
            nodes[:, 1].max() - nodes[:, 1].min(),
            nodes[:, 2].max() - nodes[:, 2].min(),
        ]) / 2
        mid = nodes.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def main():
    """Run Stanford Bunny reconstruction test."""
    print("=" * 60)
    print("GSRM Stanford Bunny Surface Reconstruction Test")
    print("=" * 60)

    # Check for data file
    data_dir = Path(__file__).parent / "data"
    bunny_path = data_dir / "bunny.ply"

    if not bunny_path.exists():
        print(f"\nError: Bunny PLY file not found at {bunny_path}")
        print("\nPlease download the Stanford Bunny:")
        print("  1. Visit: https://graphics.stanford.edu/data/3Dscanrep/")
        print("  2. Download bunny.tar.gz")
        print("  3. Extract and copy bunny/reconstruction/bun_zipper.ply")
        print(f"  4. Place at: {bunny_path}")
        return

    # Load and normalize point cloud
    print("\nLoading Stanford Bunny...")
    point_cloud = load_ply(str(bunny_path))
    print(f"  Loaded {len(point_cloud)} vertices")

    point_cloud = normalize_points(point_cloud)
    print(f"  Normalized to [0.1, 0.9] range")

    # Parameters - larger for complex shape
    params = GSRMParams(
        max_nodes=1000,
        lambda_=100,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
        max_age=80,
    )

    n_iterations = 50000

    print(f"\nParameters:")
    print(f"  max_nodes: {params.max_nodes}")
    print(f"  lambda_: {params.lambda_}")
    print(f"  n_iterations: {n_iterations}")

    # Initialize GSRM
    print("\nInitializing GSRM...")
    gsrm = GSRM(params=params, seed=42)

    # Store snapshots for animation
    snapshots = []
    snapshot_interval = n_iterations // 40  # 40 frames

    def callback(model: GSRM, iteration: int):
        if iteration % snapshot_interval == 0 or iteration == n_iterations - 1:
            nodes, edges, faces = model.get_mesh()
            snapshots.append({
                "iteration": iteration,
                "nodes": nodes.copy(),
                "edges": edges.copy(),
                "faces": faces.copy(),
                "n_nodes": model.n_nodes,
                "n_edges": model.n_edges,
                "n_faces": model.n_faces,
            })

        if iteration % 5000 == 0:
            print(f"  Iteration {iteration}: {model.n_nodes} nodes, "
                  f"{model.n_edges} edges, {model.n_faces} faces")

    # Train
    print("\nTraining...")
    gsrm.train(point_cloud, n_iterations=n_iterations, callback=callback)

    # Final state
    nodes, edges, faces = gsrm.get_mesh()
    h_dist = hausdorff_distance(point_cloud, nodes)

    print(f"\nFinal: {gsrm.n_nodes} nodes, {gsrm.n_edges} edges, {gsrm.n_faces} faces")
    print(f"Hausdorff distance: {h_dist:.6f}")

    # Create output directory
    output_dir = Path(__file__).parent / "samples" / "gsrm" / "python"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create growth animation
    print("\nCreating growth animation...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def animate(frame_idx):
        snapshot = snapshots[frame_idx]
        azim = 45 + frame_idx * 4
        visualize_mesh_3d(
            ax,
            snapshot["nodes"],
            snapshot["edges"],
            snapshot["faces"],
            point_cloud=point_cloud,
            show_points=True,
            show_edges=False,
            show_faces=True,
            elev=15,
            azim=azim,
            title=f"Stanford Bunny - Iteration {snapshot['iteration']}\n"
                  f"Nodes: {snapshot['n_nodes']}, Faces: {snapshot['n_faces']}",
        )
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=len(snapshots), interval=200, blit=False
    )
    anim.save(output_dir / "bunny_growth.gif", writer="pillow", fps=5)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'bunny_growth.gif'}")

    # Create final state images
    print("\nCreating final state images...")
    fig = plt.figure(figsize=(16, 10))

    # Multiple views
    views = [
        (15, 45, "Front-Left"),
        (15, 135, "Front-Right"),
        (15, 225, "Back-Right"),
        (15, 315, "Back-Left"),
        (60, 45, "Top View"),
        (0, 90, "Side View"),
    ]

    for idx, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        visualize_mesh_3d(
            ax,
            nodes,
            edges,
            faces,
            point_cloud=None,
            show_points=False,
            show_edges=False,
            show_faces=True,
            elev=elev,
            azim=azim,
            title=title,
        )

    plt.suptitle(
        f"Stanford Bunny Reconstruction\n"
        f"{gsrm.n_nodes} nodes, {gsrm.n_faces} faces, Hausdorff: {h_dist:.4f}",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_dir / "bunny_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'bunny_final.png'}")

    # Comparison: Point cloud vs Mesh
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
        c="gray", s=0.5, alpha=0.5
    )
    ax1.view_init(elev=15, azim=45)
    ax1.set_title(f"Original Point Cloud\n({len(point_cloud)} points)")

    ax2 = fig.add_subplot(122, projection="3d")
    visualize_mesh_3d(
        ax2,
        nodes, edges, faces,
        point_cloud=None,
        show_points=False,
        show_edges=False,
        show_faces=True,
        elev=15,
        azim=45,
        title=f"GSRM Reconstruction\n({gsrm.n_nodes} nodes, {gsrm.n_faces} faces)",
    )

    plt.suptitle(f"Hausdorff Distance: {h_dist:.6f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "bunny_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'bunny_comparison.png'}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
