#!/usr/bin/env python3
"""GSRM sphere surface reconstruction test.

This script tests GSRM on a sphere point cloud to verify basic
mesh generation and surface reconstruction capability.
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

from algorithms.gsrm.python.model import GSRM, GSRMParams

# Import 3D sampler (directory name starts with number, so use importlib)
import importlib.util
sampler_3d_path = project_root / "data" / "3d" / "sampler.py"
spec = importlib.util.spec_from_file_location("sampler_3d", sampler_3d_path)
sampler_3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_3d)
sample_sphere = sampler_3d.sample_sphere


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
    """Visualize 3D mesh on given axes.

    Args:
        ax: Matplotlib 3D axes.
        nodes: Node positions (n, 3).
        edges: List of (i, j) edge tuples.
        faces: List of (i, j, k) face tuples.
        point_cloud: Optional point cloud to show.
        show_points: Whether to show point cloud.
        show_edges: Whether to show edges.
        show_faces: Whether to show faces.
        elev: Elevation angle.
        azim: Azimuth angle.
        title: Plot title.
    """
    ax.clear()

    # Plot point cloud
    if point_cloud is not None and show_points:
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            c="lightgray",
            s=1,
            alpha=0.3,
            label="Point Cloud",
        )

    # Plot nodes
    if len(nodes) > 0:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            c="blue",
            s=20,
            alpha=0.8,
            label=f"Nodes ({len(nodes)})",
        )

    # Plot edges
    if show_edges and edges:
        for i, j in edges:
            ax.plot3D(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                [nodes[i, 2], nodes[j, 2]],
                "b-",
                alpha=0.3,
                linewidth=0.5,
            )

    # Plot faces
    if show_faces and faces:
        face_vertices = []
        for i, j, k in faces:
            face_vertices.append([nodes[i], nodes[j], nodes[k]])

        poly = Poly3DCollection(
            face_vertices,
            alpha=0.6,
            facecolor="gold",
            edgecolor="darkgoldenrod",
            linewidth=0.3,
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
    """Run sphere reconstruction test."""
    print("=" * 60)
    print("GSRM Sphere Surface Reconstruction Test")
    print("=" * 60)

    # Parameters
    params = GSRMParams(
        max_nodes=300,
        lambda_=50,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
        max_age=50,
    )

    n_iterations = 15000
    n_samples = 5000

    print(f"\nParameters:")
    print(f"  max_nodes: {params.max_nodes}")
    print(f"  lambda_: {params.lambda_}")
    print(f"  eps_b: {params.eps_b}")
    print(f"  eps_n: {params.eps_n}")
    print(f"  alpha: {params.alpha}")
    print(f"  beta: {params.beta}")
    print(f"  max_age: {params.max_age}")
    print(f"  n_iterations: {n_iterations}")
    print(f"  n_samples: {n_samples}")

    # Generate sphere point cloud
    print("\nGenerating sphere point cloud...")
    point_cloud = sample_sphere(n_samples=n_samples, seed=42)
    print(f"  Generated {len(point_cloud)} points")

    # Initialize GSRM
    print("\nInitializing GSRM...")
    gsrm = GSRM(params=params, seed=42)
    print(f"  Initial: {gsrm.n_nodes} nodes, {gsrm.n_edges} edges, {gsrm.n_faces} faces")

    # Store snapshots for animation
    snapshots = []
    snapshot_interval = n_iterations // 50  # 50 frames

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

        if iteration % 1000 == 0:
            print(f"  Iteration {iteration}: {model.n_nodes} nodes, {model.n_edges} edges, {model.n_faces} faces")

    # Train
    print("\nTraining...")
    gsrm.train(point_cloud, n_iterations=n_iterations, callback=callback)

    # Final state
    nodes, edges, faces = gsrm.get_mesh()
    print(f"\nFinal: {gsrm.n_nodes} nodes, {gsrm.n_edges} edges, {gsrm.n_faces} faces")

    # Create output directory
    output_dir = Path(__file__).parent / "samples" / "gsrm" / "python"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create growth animation
    print("\nCreating growth animation...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def animate(frame_idx):
        snapshot = snapshots[frame_idx]
        # Rotate view
        azim = 45 + frame_idx * 3
        visualize_mesh_3d(
            ax,
            snapshot["nodes"],
            snapshot["edges"],
            snapshot["faces"],
            point_cloud=point_cloud,
            show_points=True,
            show_edges=True,
            show_faces=True,
            elev=20,
            azim=azim,
            title=f"GSRM Sphere - Iteration {snapshot['iteration']}\n"
                  f"Nodes: {snapshot['n_nodes']}, Edges: {snapshot['n_edges']}, Faces: {snapshot['n_faces']}",
        )
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=len(snapshots), interval=200, blit=False
    )
    anim.save(output_dir / "sphere_growth.gif", writer="pillow", fps=5)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'sphere_growth.gif'}")

    # Create final state image
    print("\nCreating final state image...")
    fig = plt.figure(figsize=(16, 6))

    # Wireframe view
    ax1 = fig.add_subplot(131, projection="3d")
    visualize_mesh_3d(
        ax1,
        nodes,
        edges,
        [],  # No faces
        point_cloud=point_cloud,
        show_points=True,
        show_edges=True,
        show_faces=False,
        elev=20,
        azim=45,
        title=f"Wireframe\n({len(nodes)} nodes, {len(edges)} edges)",
    )

    # Mesh view (faces)
    ax2 = fig.add_subplot(132, projection="3d")
    visualize_mesh_3d(
        ax2,
        nodes,
        edges,
        faces,
        point_cloud=None,
        show_points=False,
        show_edges=False,
        show_faces=True,
        elev=20,
        azim=45,
        title=f"Mesh Faces\n({len(faces)} faces)",
    )

    # Combined view
    ax3 = fig.add_subplot(133, projection="3d")
    visualize_mesh_3d(
        ax3,
        nodes,
        edges,
        faces,
        point_cloud=point_cloud,
        show_points=True,
        show_edges=True,
        show_faces=True,
        elev=20,
        azim=135,
        title="Combined View\n(different angle)",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "sphere_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'sphere_final.png'}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
