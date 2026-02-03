"""GSRM (Growing Self-Reconstruction Meshes) implementation.

Based on:
    - Rêgo, R. L. M. E., Araújo, A. F. R., & Lima Neto, F. B. (2007).
      "Growing Self-Organizing Maps for Surface Reconstruction from
      Unstructured Point Clouds" (IJCNN 2007)

GSRM extends GNG to produce triangular meshes instead of wireframes.
Key differences from standard GNG:
    1. Extended Competitive Hebbian Learning (ECHL) - creates faces
    2. Edge and face removal - removes faces incident to old edges
    3. GCS-style vertex insertion - splits faces when inserting nodes
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GSRMParams:
    """GSRM hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for neighbor nodes.
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        max_age: Maximum edge age before removal.
    """

    max_nodes: int = 500
    lambda_: int = 50
    eps_b: float = 0.1
    eps_n: float = 0.01
    alpha: float = 0.5
    beta: float = 0.005
    max_age: int = 50


@dataclass
class GSRMNode:
    """A node in the GSRM network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector (3D coordinates).
        error: Accumulated error.
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 0.0


class GSRM:
    """Growing Self-Reconstruction Meshes algorithm.

    GSRM is a surface reconstruction method based on GNG that produces
    triangular meshes from 3D point clouds. It uses Extended Competitive
    Hebbian Learning to create triangular faces and maintains a valid
    mesh structure throughout the learning process.

    Attributes:
        params: GSRM hyperparameters.
        nodes: List of nodes.
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GSRM, GSRMParams
    >>> # Generate sphere point cloud
    >>> n_points = 5000
    >>> theta = np.random.uniform(0, 2*np.pi, n_points)
    >>> phi = np.random.uniform(0, np.pi, n_points)
    >>> r = 0.4
    >>> X = np.column_stack([
    ...     0.5 + r * np.sin(phi) * np.cos(theta),
    ...     0.5 + r * np.sin(phi) * np.sin(theta),
    ...     0.5 + r * np.cos(phi)
    ... ])
    >>> # Train GSRM
    >>> gsrm = GSRM(params=GSRMParams(max_nodes=200))
    >>> gsrm.train(X, n_iterations=5000)
    >>> nodes, edges, faces = gsrm.get_mesh()
    """

    def __init__(
        self,
        params: GSRMParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GSRM.

        Args:
            params: GSRM hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = 3  # GSRM is specifically for 3D surface reconstruction
        self.params = params or GSRMParams()
        self.rng = np.random.default_rng(seed)

        # Node management
        self.nodes: list[GSRMNode] = [
            GSRMNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        # Edge management (age matrix and adjacency sets)
        self.edges = np.zeros(
            (self.params.max_nodes, self.params.max_nodes), dtype=np.int32
        )
        self.edges_per_node: dict[int, set[int]] = {}

        # Face management
        # faces: dict mapping face_id -> tuple of 3 node IDs (sorted)
        self._next_face_id = 0
        self.faces: dict[int, tuple[int, int, int]] = {}
        # faces_per_edge: mapping from edge (min_id, max_id) -> set of face IDs
        self.faces_per_edge: dict[tuple[int, int], set[int]] = {}
        # faces_per_node: mapping from node_id -> set of face IDs
        self.faces_per_node: dict[int, set[int]] = {}

        # Counters
        self.n_learning = 0
        self._n_trial = 0

        # Initialize with 3 nodes (initial triangle)
        self._init_triangle()

    def _init_triangle(self) -> None:
        """Initialize with 3 random nodes forming a triangle."""
        # Sample 3 random points from unit cube
        node_ids = []
        for _ in range(3):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            node_id = self._add_node(weight)
            node_ids.append(node_id)

        # Connect all pairs (triangle edges)
        for i in range(3):
            for j in range(i + 1, 3):
                self._add_edge(node_ids[i], node_ids[j])

        # Create the initial face
        self._add_face(node_ids[0], node_ids[1], node_ids[2])

    def _add_node(self, weight: np.ndarray) -> int:
        """Add a new node with given weight.

        Args:
            weight: Position vector.

        Returns:
            ID of new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = GSRMNode(id=node_id, weight=weight.copy(), error=0.0)
        self.edges_per_node[node_id] = set()
        self.faces_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node (only if isolated - no edges).

        Args:
            node_id: ID of node to remove.
        """
        if self.edges_per_node.get(node_id):
            return  # Has edges, don't remove

        self.edges_per_node.pop(node_id, None)
        self.faces_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _edge_key(self, n1: int, n2: int) -> tuple[int, int]:
        """Get canonical edge key (smaller id first)."""
        return (min(n1, n2), max(n1, n2))

    def _add_edge(self, n1: int, n2: int) -> None:
        """Add or reset edge between two nodes.

        Args:
            n1: First node ID.
            n2: Second node ID.
        """
        if n1 == n2:
            return

        key = self._edge_key(n1, n2)

        if self.edges[n1, n2] > 0:
            # Edge exists, reset age
            self.edges[n1, n2] = 1
            self.edges[n2, n1] = 1
        else:
            # New edge
            self.edges_per_node[n1].add(n2)
            self.edges_per_node[n2].add(n1)
            self.edges[n1, n2] = 1
            self.edges[n2, n1] = 1
            # Initialize faces_per_edge for this edge
            if key not in self.faces_per_edge:
                self.faces_per_edge[key] = set()

    def _remove_edge(self, n1: int, n2: int) -> None:
        """Remove edge between two nodes.

        Args:
            n1: First node ID.
            n2: Second node ID.
        """
        key = self._edge_key(n1, n2)

        self.edges_per_node[n1].discard(n2)
        self.edges_per_node[n2].discard(n1)
        self.edges[n1, n2] = 0
        self.edges[n2, n1] = 0

        # Remove from faces_per_edge tracking
        self.faces_per_edge.pop(key, None)

    def _add_face(self, n1: int, n2: int, n3: int) -> int:
        """Add a triangular face.

        Args:
            n1, n2, n3: Node IDs forming the triangle.

        Returns:
            Face ID, or -1 if face already exists or invalid.
        """
        # Sort to get canonical representation
        vertices = tuple(sorted([n1, n2, n3]))

        # Check if face already exists
        for face_id, face_vertices in self.faces.items():
            if face_vertices == vertices:
                return face_id  # Already exists

        # Create new face
        face_id = self._next_face_id
        self._next_face_id += 1
        self.faces[face_id] = vertices

        # Update faces_per_edge
        edges = [
            self._edge_key(vertices[0], vertices[1]),
            self._edge_key(vertices[1], vertices[2]),
            self._edge_key(vertices[0], vertices[2]),
        ]
        for edge in edges:
            if edge not in self.faces_per_edge:
                self.faces_per_edge[edge] = set()
            self.faces_per_edge[edge].add(face_id)

        # Update faces_per_node
        for v in vertices:
            self.faces_per_node[v].add(face_id)

        return face_id

    def _remove_face(self, face_id: int) -> None:
        """Remove a face.

        Args:
            face_id: ID of face to remove.
        """
        if face_id not in self.faces:
            return

        vertices = self.faces[face_id]

        # Remove from faces_per_edge
        edges = [
            self._edge_key(vertices[0], vertices[1]),
            self._edge_key(vertices[1], vertices[2]),
            self._edge_key(vertices[0], vertices[2]),
        ]
        for edge in edges:
            if edge in self.faces_per_edge:
                self.faces_per_edge[edge].discard(face_id)

        # Remove from faces_per_node
        for v in vertices:
            if v in self.faces_per_node:
                self.faces_per_node[v].discard(face_id)

        # Remove face
        del self.faces[face_id]

    def _find_three_nearest(self, x: np.ndarray) -> tuple[int, int, int]:
        """Find the three nearest nodes to input x.

        Args:
            x: Input vector (3D point).

        Returns:
            Tuple of (s1_id, s2_id, s3_id) - first, second, third nearest.
        """
        distances = []
        for node in self.nodes:
            if node.id == -1:
                continue
            dist = np.sum((x - node.weight) ** 2)
            distances.append((dist, node.id))

        # Sort by distance and get top 3
        distances.sort(key=lambda x: x[0])

        if len(distances) < 3:
            # Not enough nodes
            return (-1, -1, -1)

        return (distances[0][1], distances[1][1], distances[2][1])

    def _extended_chl(self, s1: int, s2: int, s3: int) -> None:
        """Extended Competitive Hebbian Learning.

        Creates or reinforces edges between the three winners and
        creates a triangular face if it doesn't exist.

        Args:
            s1, s2, s3: The three winner node IDs.
        """
        # Create/reinforce edges between all pairs
        self._add_edge(s1, s2)
        self._add_edge(s2, s3)
        self._add_edge(s1, s3)

        # Create face if the three connections form a triangle
        # and the face doesn't already exist
        self._add_face(s1, s2, s3)

    def _remove_invalid_edges_and_faces(self, s1: int) -> None:
        """Remove invalid edges and their incident faces.

        Follows GSRM paper:
        1. Remove faces incident to invalid edges (age > max_age)
        2. Remove edges without incident faces
        3. Remove nodes without incident edges

        Args:
            s1: Winner node ID (edges from this node are checked).
        """
        p = self.params
        edges_to_check = list(self.edges_per_node.get(s1, set()))

        # Step 1: Find invalid edges and remove their incident faces
        invalid_edges = []
        for neighbor_id in edges_to_check:
            if self.edges[s1, neighbor_id] > p.max_age:
                invalid_edges.append(neighbor_id)

                # Remove faces incident to this edge
                edge_key = self._edge_key(s1, neighbor_id)
                face_ids = list(self.faces_per_edge.get(edge_key, set()))
                for face_id in face_ids:
                    self._remove_face(face_id)

        # Step 2: Remove invalid edges
        for neighbor_id in invalid_edges:
            self._remove_edge(s1, neighbor_id)

        # Step 3: Remove edges without incident faces (optional, be careful)
        # This is done conservatively - only remove if edge has no faces AND
        # removing it doesn't disconnect the graph
        # For now, we skip this to avoid disconnecting the mesh

        # Step 4: Remove isolated nodes
        for neighbor_id in invalid_edges:
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

    def _find_common_neighbors(self, n1: int, n2: int) -> set[int]:
        """Find nodes that are neighbors of both n1 and n2.

        Args:
            n1: First node ID.
            n2: Second node ID.

        Returns:
            Set of common neighbor IDs.
        """
        return self.edges_per_node.get(n1, set()) & self.edges_per_node.get(n2, set())

    def _insert_node_gcs(self) -> int:
        """Insert a new node using GCS-style insertion.

        1. Find node q with maximum error
        2. Find neighbor f of q with maximum error
        3. Insert new node r at midpoint of q and f
        4. Connect r to q, f, and their common neighbors
        5. Split faces incident to (q, f) edge

        Returns:
            ID of new node, or -1 if failed.
        """
        p = self.params

        if not self._addable_indices:
            return -1

        # Find node q with maximum error
        max_err = -1.0
        q_id = -1
        for node in self.nodes:
            if node.id == -1:
                continue
            if node.error > max_err:
                max_err = node.error
                q_id = node.id

        if q_id == -1:
            return -1

        # Find neighbor f of q with maximum error
        max_err_f = -1.0
        f_id = -1
        for neighbor_id in self.edges_per_node.get(q_id, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f_id = neighbor_id

        if f_id == -1:
            return -1

        # Get faces incident to (q, f) edge before modification
        edge_key = self._edge_key(q_id, f_id)
        incident_faces = list(self.faces_per_edge.get(edge_key, set()))

        # Find common neighbors (nodes connected to both q and f)
        common_neighbors = self._find_common_neighbors(q_id, f_id)

        # Create new node at midpoint
        new_weight = (self.nodes[q_id].weight + self.nodes[f_id].weight) * 0.5
        r_id = self._add_node(new_weight)

        if r_id == -1:
            return -1

        # Update errors
        self.nodes[q_id].error *= (1 - p.alpha)
        self.nodes[f_id].error *= (1 - p.alpha)
        self.nodes[r_id].error = (
            self.nodes[q_id].error + self.nodes[f_id].error
        ) * 0.5

        # Remove old faces incident to (q, f)
        for face_id in incident_faces:
            if face_id in self.faces:
                vertices = self.faces[face_id]
                self._remove_face(face_id)

                # Find the third vertex of this face (not q or f)
                third_vertex = None
                for v in vertices:
                    if v != q_id and v != f_id:
                        third_vertex = v
                        break

                if third_vertex is not None:
                    # Create two new faces: (q, r, third) and (r, f, third)
                    self._add_face(q_id, r_id, third_vertex)
                    self._add_face(r_id, f_id, third_vertex)

        # Remove old edge (q, f)
        self._remove_edge(q_id, f_id)

        # Add new edges
        self._add_edge(q_id, r_id)
        self._add_edge(f_id, r_id)

        # Connect to common neighbors
        for cn in common_neighbors:
            self._add_edge(r_id, cn)

        return r_id

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration following GSRM algorithm.

        Args:
            sample: Input sample vector (3D point).
        """
        p = self.params

        # Find three nearest nodes (Step 3)
        s1_id, s2_id, s3_id = self._find_three_nearest(sample)

        if s1_id == -1 or s2_id == -1 or s3_id == -1:
            return

        # Extended CHL: create/reinforce edges and face (Step 4)
        self._extended_chl(s1_id, s2_id, s3_id)

        # Update winner error (Step 5): ΔE_s1 = ||w_s1 - ξ||²
        dist_sq = np.sum((sample - self.nodes[s1_id].weight) ** 2)
        self.nodes[s1_id].error += dist_sq

        # Move winner toward sample (Step 6)
        self.nodes[s1_id].weight += p.eps_b * (sample - self.nodes[s1_id].weight)

        # Move neighbors toward sample (Step 6)
        for neighbor_id in list(self.edges_per_node.get(s1_id, set())):
            self.nodes[neighbor_id].weight += p.eps_n * (
                sample - self.nodes[neighbor_id].weight
            )

        # Update edge ages (Step 7): age = age + 1
        for neighbor_id in list(self.edges_per_node.get(s1_id, set())):
            self.edges[s1_id, neighbor_id] += 1
            self.edges[neighbor_id, s1_id] += 1

        # Remove invalid edges and faces (Step 8)
        self._remove_invalid_edges_and_faces(s1_id)

        # Periodically insert new node (Step 9)
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            if self._addable_indices:
                self._insert_node_gcs()

        # Decay all errors (Step 10): ΔE_s = -βE_s
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error *= (1 - p.beta)

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 10000,
        callback: Callable[[GSRM, int], None] | None = None,
    ) -> GSRM:
        """Train on 3D point cloud data.

        Each iteration randomly samples one point from data.

        Args:
            data: Point cloud of shape (n_samples, 3).
            n_iterations: Number of training iterations.
            callback: Optional callback(self, iteration) called each iteration.

        Returns:
            self for chaining.
        """
        if data.shape[1] != 3:
            raise ValueError(f"Expected 3D data, got shape {data.shape}")

        n_samples = len(data)

        for i in range(n_iterations):
            idx = self.rng.integers(0, n_samples)
            self._one_train_update(data[idx])

            if callback is not None:
                callback(self, i)

        return self

    def partial_fit(self, sample: np.ndarray) -> GSRM:
        """Single online learning step.

        Args:
            sample: Input vector of shape (3,).

        Returns:
            self for chaining.
        """
        self._one_train_update(sample)
        return self

    @property
    def n_nodes(self) -> int:
        """Number of active nodes."""
        return sum(1 for node in self.nodes if node.id != -1)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        count = 0
        for node_id, neighbors in self.edges_per_node.items():
            if self.nodes[node_id].id != -1:
                count += len(neighbors)
        return count // 2

    @property
    def n_faces(self) -> int:
        """Number of triangular faces."""
        return len(self.faces)

    def get_mesh(self) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int, int]]]:
        """Get current mesh structure.

        Returns:
            Tuple of:
                - nodes: Array of shape (n_active_nodes, 3) with positions.
                - edges: List of (i, j) tuples indexing into nodes array.
                - faces: List of (i, j, k) tuples indexing into nodes array.
        """
        # Get active nodes and create index mapping
        active_nodes = []
        index_map = {}

        for node in self.nodes:
            if node.id != -1:
                index_map[node.id] = len(active_nodes)
                active_nodes.append(node.weight.copy())

        nodes = (
            np.array(active_nodes)
            if active_nodes
            else np.array([]).reshape(0, self.n_dim)
        )

        # Get edges using new indices
        edges = []
        seen = set()
        for node_id, neighbors in self.edges_per_node.items():
            if self.nodes[node_id].id == -1:
                continue
            for neighbor_id in neighbors:
                if self.nodes[neighbor_id].id == -1:
                    continue
                edge = tuple(sorted([node_id, neighbor_id]))
                if edge not in seen:
                    seen.add(edge)
                    edges.append((index_map[node_id], index_map[neighbor_id]))

        # Get faces using new indices
        faces = []
        for face_vertices in self.faces.values():
            # Check all vertices are valid
            if all(self.nodes[v].id != -1 for v in face_vertices):
                faces.append(tuple(index_map[v] for v in face_vertices))

        return nodes, edges, faces

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure (nodes and edges only).

        For compatibility with GNG interface.

        Returns:
            Tuple of:
                - nodes: Array of shape (n_active_nodes, 3) with positions.
                - edges: List of (i, j) tuples indexing into nodes array.
        """
        nodes, edges, _ = self.get_mesh()
        return nodes, edges

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes.

        Returns:
            Array of error values in same order as get_mesh() nodes.
        """
        return np.array([node.error for node in self.nodes if node.id != -1])
