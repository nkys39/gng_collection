"""Test GSRM-F (Feature-Preserving GSRM) on floor and wall dataset.

This test verifies that GSRM-F can detect and preserve the sharp edge
where the floor and wall meet.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from algorithms.gsrm.python.model_feature import GSRMF, GSRMFParams


def sample_floor_and_wall(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate points on floor (XZ plane) and wall (XY plane).

    The floor and wall meet at a 90-degree angle along the X-axis.

    Args:
        n_samples: Total number of samples.
        rng: Random number generator.

    Returns:
        Array of shape (n_samples, 3).
    """
    n_floor = n_samples // 2
    n_wall = n_samples - n_floor

    # Floor: Y = 0, X and Z in [0, 1]
    floor_x = rng.uniform(0, 1, n_floor)
    floor_y = np.zeros(n_floor) + rng.uniform(-0.01, 0.01, n_floor)  # Small noise
    floor_z = rng.uniform(0, 1, n_floor)
    floor_pts = np.column_stack([floor_x, floor_y, floor_z])

    # Wall: Z = 0, X and Y in [0, 1]
    wall_x = rng.uniform(0, 1, n_wall)
    wall_y = rng.uniform(0, 1, n_wall)
    wall_z = np.zeros(n_wall) + rng.uniform(-0.01, 0.01, n_wall)  # Small noise
    wall_pts = np.column_stack([wall_x, wall_y, wall_z])

    return np.vstack([floor_pts, wall_pts])


def main():
    # Parameters
    seed = 42
    rng = np.random.default_rng(seed)
    n_samples = 3000
    n_iterations = 8000

    # GSRM-F parameters
    params = GSRMFParams(
        max_nodes=200,
        lambda_=50,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
        max_age=100,
        # Feature-preserving parameters
        tau_normal=0.5,  # cos(60°) - detect edges where normals differ by > 60°
        edge_learning_factor=0.2,  # Strongly reduce learning on edges
        edge_insertion_bias=3.0,  # Prefer inserting on edges
        min_neighbors_for_normal=3,
    )

    # Generate data
    print("Generating floor and wall point cloud...")
    data = sample_floor_and_wall(n_samples, rng)
    print(f"  Samples: {len(data)}")

    # Create GSRM-F
    print("\nTraining GSRM-F...")
    gsrmf = GSRMF(params=params, seed=seed)

    # Store states for animation
    states = []

    def callback(model: GSRMF, iteration: int):
        if iteration % 200 == 0 or iteration == n_iterations - 1:
            nodes, edges, faces = model.get_mesh()
            edge_nodes = model.get_edge_nodes()
            is_edge = model.get_is_edge()
            states.append({
                "iteration": iteration,
                "nodes": nodes.copy(),
                "edges": edges.copy(),
                "faces": faces.copy(),
                "edge_nodes": edge_nodes.copy(),
                "is_edge": is_edge.copy(),
                "n_nodes": model.n_nodes,
                "n_edges": model.n_edges,
                "n_faces": model.n_faces,
                "n_edge_nodes": model.n_edge_nodes,
            })
            print(f"  Iteration {iteration}: {model.n_nodes} nodes, "
                  f"{model.n_edges} edges, {model.n_faces} faces, "
                  f"{model.n_edge_nodes} edge nodes")

    gsrmf.train(data, n_iterations=n_iterations, callback=callback)

    # Final state
    print(f"\nFinal: {gsrmf.n_nodes} nodes, {gsrmf.n_edges} edges, "
          f"{gsrmf.n_faces} faces, {gsrmf.n_edge_nodes} edge nodes")

    # Create output directory
    output_dir = Path(__file__).parent / "samples" / "gsrm_f" / "python"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create animation
    print("\nCreating animation...")
    fig = plt.figure(figsize=(12, 5))

    # 3D view
    ax1 = fig.add_subplot(121, projection="3d")

    # 2D side view (XY plane looking from +Z)
    ax2 = fig.add_subplot(122)

    def update(frame_idx):
        state = states[min(frame_idx, len(states) - 1)]
        nodes = state["nodes"]
        edges = state["edges"]
        faces = state["faces"]
        is_edge = state["is_edge"]

        # Clear axes
        ax1.clear()
        ax2.clear()

        # 3D view
        if len(nodes) > 0:
            # Color nodes by edge status
            colors = np.array(["blue" if e else "gray" for e in is_edge])

            # Draw faces
            if len(faces) > 0:
                triangles = []
                for f in faces:
                    if f[0] < len(nodes) and f[1] < len(nodes) and f[2] < len(nodes):
                        triangles.append([nodes[f[0]], nodes[f[1]], nodes[f[2]]])
                if triangles:
                    mesh = Poly3DCollection(
                        triangles, alpha=0.3, facecolor="lightblue",
                        edgecolor="gray", linewidth=0.3
                    )
                    ax1.add_collection3d(mesh)

            # Draw edge nodes (larger, blue)
            edge_mask = np.array(is_edge)
            if np.any(edge_mask):
                ax1.scatter(
                    nodes[edge_mask, 0],
                    nodes[edge_mask, 1],
                    nodes[edge_mask, 2],
                    c="blue", s=30, alpha=1.0, label="Edge nodes"
                )

            # Draw non-edge nodes (smaller, gray)
            non_edge_mask = ~edge_mask
            if np.any(non_edge_mask):
                ax1.scatter(
                    nodes[non_edge_mask, 0],
                    nodes[non_edge_mask, 1],
                    nodes[non_edge_mask, 2],
                    c="gray", s=10, alpha=0.5, label="Surface nodes"
                )

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_xlim(-0.1, 1.1)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_zlim(-0.1, 1.1)
        ax1.set_title(
            f"GSRM-F: iter {state['iteration']}\n"
            f"Nodes: {state['n_nodes']}, Faces: {state['n_faces']}, "
            f"Edge nodes: {state['n_edge_nodes']}"
        )
        ax1.view_init(elev=25, azim=120 + frame_idx * 2)
        ax1.legend(loc="upper right", fontsize=8)

        # 2D side view (XY plane)
        if len(nodes) > 0:
            # Draw edges
            for e in edges:
                if e[0] < len(nodes) and e[1] < len(nodes):
                    ax2.plot(
                        [nodes[e[0], 0], nodes[e[1], 0]],
                        [nodes[e[0], 1], nodes[e[1], 1]],
                        c="lightgray", linewidth=0.5
                    )

            # Draw edge nodes
            edge_mask = np.array(is_edge)
            if np.any(edge_mask):
                ax2.scatter(
                    nodes[edge_mask, 0],
                    nodes[edge_mask, 1],
                    c="blue", s=30, alpha=1.0, label="Edge nodes"
                )
            if np.any(~edge_mask):
                ax2.scatter(
                    nodes[~edge_mask, 0],
                    nodes[~edge_mask, 1],
                    c="gray", s=10, alpha=0.5, label="Surface nodes"
                )

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_aspect("equal")
        ax2.set_title("Top view (XY plane)")
        ax2.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, update, frames=len(states), interval=200, blit=False)
    gif_path = output_dir / "floor_wall_growth.gif"
    anim.save(str(gif_path), writer=PillowWriter(fps=5))
    print(f"  Saved: {gif_path}")
    plt.close()

    # Create final state figure
    print("Creating final state figure...")
    fig = plt.figure(figsize=(14, 5))

    final_state = states[-1]
    nodes = final_state["nodes"]
    edges = final_state["edges"]
    faces = final_state["faces"]
    is_edge = final_state["is_edge"]

    # 3D view
    ax1 = fig.add_subplot(131, projection="3d")
    if len(nodes) > 0:
        # Draw faces
        if len(faces) > 0:
            triangles = []
            for f in faces:
                if f[0] < len(nodes) and f[1] < len(nodes) and f[2] < len(nodes):
                    triangles.append([nodes[f[0]], nodes[f[1]], nodes[f[2]]])
            if triangles:
                mesh = Poly3DCollection(
                    triangles, alpha=0.3, facecolor="lightblue",
                    edgecolor="gray", linewidth=0.3
                )
                ax1.add_collection3d(mesh)

        # Draw nodes
        edge_mask = np.array(is_edge)
        if np.any(edge_mask):
            ax1.scatter(
                nodes[edge_mask, 0], nodes[edge_mask, 1], nodes[edge_mask, 2],
                c="blue", s=30, alpha=1.0, label="Edge nodes"
            )
        if np.any(~edge_mask):
            ax1.scatter(
                nodes[~edge_mask, 0], nodes[~edge_mask, 1], nodes[~edge_mask, 2],
                c="gray", s=10, alpha=0.5, label="Surface nodes"
            )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_zlim(-0.1, 1.1)
    ax1.set_title("3D View")
    ax1.view_init(elev=25, azim=120)
    ax1.legend(loc="upper right", fontsize=8)

    # Top view (XY plane)
    ax2 = fig.add_subplot(132)
    if len(nodes) > 0:
        for e in edges:
            if e[0] < len(nodes) and e[1] < len(nodes):
                ax2.plot(
                    [nodes[e[0], 0], nodes[e[1], 0]],
                    [nodes[e[0], 1], nodes[e[1], 1]],
                    c="lightgray", linewidth=0.5
                )
        edge_mask = np.array(is_edge)
        if np.any(edge_mask):
            ax2.scatter(nodes[edge_mask, 0], nodes[edge_mask, 1],
                       c="blue", s=30, label="Edge")
        if np.any(~edge_mask):
            ax2.scatter(nodes[~edge_mask, 0], nodes[~edge_mask, 1],
                       c="gray", s=10, alpha=0.5, label="Surface")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_aspect("equal")
    ax2.set_title("Top View (XY)")
    ax2.legend(fontsize=8)

    # Side view (XZ plane)
    ax3 = fig.add_subplot(133)
    if len(nodes) > 0:
        for e in edges:
            if e[0] < len(nodes) and e[1] < len(nodes):
                ax3.plot(
                    [nodes[e[0], 0], nodes[e[1], 0]],
                    [nodes[e[0], 2], nodes[e[1], 2]],
                    c="lightgray", linewidth=0.5
                )
        edge_mask = np.array(is_edge)
        if np.any(edge_mask):
            ax3.scatter(nodes[edge_mask, 0], nodes[edge_mask, 2],
                       c="blue", s=30, label="Edge")
        if np.any(~edge_mask):
            ax3.scatter(nodes[~edge_mask, 0], nodes[~edge_mask, 2],
                       c="gray", s=10, alpha=0.5, label="Surface")

    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_aspect("equal")
    ax3.set_title("Side View (XZ)")
    ax3.legend(fontsize=8)

    plt.suptitle(
        f"GSRM-F Floor and Wall: {final_state['n_nodes']} nodes, "
        f"{final_state['n_faces']} faces, {final_state['n_edge_nodes']} edge nodes",
        fontsize=12
    )
    plt.tight_layout()

    png_path = output_dir / "floor_wall_final.png"
    plt.savefig(str(png_path), dpi=150)
    print(f"  Saved: {png_path}")
    plt.close()

    # Compare with standard GSRM
    print("\nComparing with standard GSRM...")
    from algorithms.gsrm.python.model import GSRM, GSRMParams

    std_params = GSRMParams(
        max_nodes=200,
        lambda_=50,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.005,
        max_age=100,
    )

    gsrm_std = GSRM(params=std_params, seed=seed)
    gsrm_std.train(data, n_iterations=n_iterations)

    print(f"  Standard GSRM: {gsrm_std.n_nodes} nodes, {gsrm_std.n_faces} faces")
    print(f"  GSRM-F: {gsrmf.n_nodes} nodes, {gsrmf.n_faces} faces, "
          f"{gsrmf.n_edge_nodes} edge nodes")

    # Create comparison figure
    fig = plt.figure(figsize=(12, 5))

    # Standard GSRM
    ax1 = fig.add_subplot(121, projection="3d")
    nodes_std, edges_std, faces_std = gsrm_std.get_mesh()
    if len(nodes_std) > 0 and len(faces_std) > 0:
        triangles = []
        for f in faces_std:
            if f[0] < len(nodes_std) and f[1] < len(nodes_std) and f[2] < len(nodes_std):
                triangles.append([nodes_std[f[0]], nodes_std[f[1]], nodes_std[f[2]]])
        if triangles:
            mesh = Poly3DCollection(
                triangles, alpha=0.4, facecolor="lightcoral",
                edgecolor="darkred", linewidth=0.3
            )
            ax1.add_collection3d(mesh)
        ax1.scatter(nodes_std[:, 0], nodes_std[:, 1], nodes_std[:, 2],
                   c="red", s=10, alpha=0.7)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_zlim(-0.1, 1.1)
    ax1.set_title(f"Standard GSRM\n{gsrm_std.n_nodes} nodes, {gsrm_std.n_faces} faces")
    ax1.view_init(elev=25, azim=120)

    # GSRM-F
    ax2 = fig.add_subplot(122, projection="3d")
    nodes_f = final_state["nodes"]
    faces_f = final_state["faces"]
    is_edge = final_state["is_edge"]

    if len(nodes_f) > 0 and len(faces_f) > 0:
        triangles = []
        for f in faces_f:
            if f[0] < len(nodes_f) and f[1] < len(nodes_f) and f[2] < len(nodes_f):
                triangles.append([nodes_f[f[0]], nodes_f[f[1]], nodes_f[f[2]]])
        if triangles:
            mesh = Poly3DCollection(
                triangles, alpha=0.4, facecolor="lightblue",
                edgecolor="darkblue", linewidth=0.3
            )
            ax2.add_collection3d(mesh)

        edge_mask = np.array(is_edge)
        if np.any(edge_mask):
            ax2.scatter(nodes_f[edge_mask, 0], nodes_f[edge_mask, 1], nodes_f[edge_mask, 2],
                       c="blue", s=30, alpha=1.0, label="Edge")
        if np.any(~edge_mask):
            ax2.scatter(nodes_f[~edge_mask, 0], nodes_f[~edge_mask, 1], nodes_f[~edge_mask, 2],
                       c="lightblue", s=10, alpha=0.5, label="Surface")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_zlim(-0.1, 1.1)
    ax2.set_title(f"GSRM-F (Feature-Preserving)\n{final_state['n_nodes']} nodes, "
                 f"{final_state['n_faces']} faces, {final_state['n_edge_nodes']} edge nodes")
    ax2.view_init(elev=25, azim=120)
    ax2.legend(loc="upper right", fontsize=8)

    plt.suptitle("GSRM vs GSRM-F on Floor and Wall", fontsize=14)
    plt.tight_layout()

    compare_path = output_dir / "compare_floor_wall.png"
    plt.savefig(str(compare_path), dpi=150)
    print(f"  Saved: {compare_path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
