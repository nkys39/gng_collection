"""GSRM-F (Feature-Preserving GSRM) implementation.

Extends GSRM with sharp edge detection and preservation capabilities.

Key additions over standard GSRM:
    1. PCA-based normal computation for each node
    2. Edge detection using normal similarity (low dot product = sharp edge)
    3. Adaptive learning rate - lower near detected edges
    4. Edge-aware node insertion - prefer inserting nodes along edges

Based on concepts from:
    - GNG-DT Robot (normal computation via PCA)
    - Feature-preserving mesh reconstruction techniques
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .model import GSRM, GSRMParams, GSRMNode


@dataclass
class GSRMFParams(GSRMParams):
    """GSRM-F hyperparameters.

    Extends GSRMParams with feature-preserving parameters.

    Attributes:
        tau_normal: Normal similarity threshold for edge detection.
            If dot(n1, n2) < tau_normal, the connection is marked as a sharp edge.
        edge_learning_factor: Learning rate multiplier for edge nodes (< 1.0).
        edge_insertion_bias: Bias factor for inserting nodes on edges (> 1.0).
        min_neighbors_for_normal: Minimum neighbors required for normal computation.
    """

    tau_normal: float = 0.5  # cos(60°) - edges where normals differ by > 60°
    edge_learning_factor: float = 0.3  # Reduce learning rate on edges
    edge_insertion_bias: float = 2.0  # Prefer inserting on edges
    min_neighbors_for_normal: int = 3  # Need at least 3 neighbors for good PCA


@dataclass
class GSRMFNode(GSRMNode):
    """Extended node with feature-preserving properties.

    Attributes:
        normal: Surface normal vector (computed via PCA).
        pca_residual: PCA residual (smallest eigenvalue) - indicates surface curvature.
        is_edge: Whether this node is on a sharp edge.
        edge_strength: Strength of edge detection (0 = flat, 1 = sharp).
    """

    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    pca_residual: float = -1.0  # -1 indicates not computed
    is_edge: bool = False
    edge_strength: float = 0.0


class GSRMF(GSRM):
    """Feature-Preserving Growing Self-Reconstruction Meshes.

    Extends GSRM to detect and preserve sharp edges in the reconstructed mesh.
    Uses PCA to compute surface normals and detects edges where normals
    change abruptly. Learning rates are reduced near edges to preserve them.

    Attributes:
        params: GSRM-F hyperparameters.
        nodes: List of extended nodes with normal information.
        edge_nodes: Set of node IDs that are on sharp edges.

    Examples
    --------
    >>> import numpy as np
    >>> from model_feature import GSRMF, GSRMFParams
    >>> # Generate floor + wall point cloud (L-shape with sharp edge)
    >>> n_points = 5000
    >>> # Floor (XZ plane at y=0)
    >>> floor_pts = np.column_stack([
    ...     np.random.uniform(0, 1, n_points//2),
    ...     np.zeros(n_points//2),
    ...     np.random.uniform(0, 1, n_points//2)
    ... ])
    >>> # Wall (XY plane at z=0)
    >>> wall_pts = np.column_stack([
    ...     np.random.uniform(0, 1, n_points//2),
    ...     np.random.uniform(0, 1, n_points//2),
    ...     np.zeros(n_points//2)
    ... ])
    >>> X = np.vstack([floor_pts, wall_pts])
    >>> # Train GSRM-F
    >>> gsrmf = GSRMF(params=GSRMFParams(max_nodes=200, tau_normal=0.5))
    >>> gsrmf.train(X, n_iterations=10000)
    >>> nodes, edges, faces = gsrmf.get_mesh()
    >>> edge_nodes = gsrmf.get_edge_nodes()  # Nodes on sharp edges
    """

    def __init__(
        self,
        params: GSRMFParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GSRM-F.

        Args:
            params: GSRM-F hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        # Use GSRMFParams if not provided
        if params is None:
            params = GSRMFParams()
        elif not isinstance(params, GSRMFParams):
            # Convert GSRMParams to GSRMFParams
            params = GSRMFParams(
                max_nodes=params.max_nodes,
                lambda_=params.lambda_,
                eps_b=params.eps_b,
                eps_n=params.eps_n,
                alpha=params.alpha,
                beta=params.beta,
                max_age=params.max_age,
            )

        # Initialize parent
        self.n_dim = 3
        self.params = params
        self.rng = np.random.default_rng(seed)

        # Node management with extended nodes
        self.nodes: list[GSRMFNode] = [
            GSRMFNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        # Edge management (from parent)
        self.edges = np.zeros(
            (self.params.max_nodes, self.params.max_nodes), dtype=np.int32
        )
        self.edges_per_node: dict[int, set[int]] = {}

        # Face management (from parent)
        self._next_face_id = 0
        self.faces: dict[int, tuple[int, int, int]] = {}
        self.faces_per_edge: dict[tuple[int, int], set[int]] = {}
        self.faces_per_node: dict[int, set[int]] = {}

        # Feature-specific: track edge nodes
        self.edge_nodes: set[int] = set()

        # Counters
        self.n_learning = 0
        self._n_trial = 0

        # Initialize with 3 nodes (initial triangle)
        self._init_triangle()

    def _add_node(self, weight: np.ndarray) -> int:
        """Add a new node with extended properties.

        Args:
            weight: Position vector.

        Returns:
            ID of new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = GSRMFNode(
            id=node_id,
            weight=weight.copy(),
            error=0.0,
            normal=np.array([0.0, 0.0, 1.0]),
            pca_residual=-1.0,
            is_edge=False,
            edge_strength=0.0,
        )
        self.edges_per_node[node_id] = set()
        self.faces_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node.

        Args:
            node_id: ID of node to remove.
        """
        if self.edges_per_node.get(node_id):
            return  # Has edges, don't remove

        self.edges_per_node.pop(node_id, None)
        self.faces_per_node.pop(node_id, None)
        self.edge_nodes.discard(node_id)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _compute_node_normal(self, node_id: int) -> None:
        """Compute normal for a node using PCA of neighbors.

        Args:
            node_id: ID of node to compute normal for.
        """
        p = self.params
        node = self.nodes[node_id]
        neighbors = self.edges_per_node.get(node_id, set())

        if len(neighbors) < p.min_neighbors_for_normal:
            # Not enough neighbors for reliable normal
            node.pca_residual = -1.0
            return

        # Collect positions (node + neighbors)
        positions = [node.weight.copy()]
        for neighbor_id in neighbors:
            if self.nodes[neighbor_id].id != -1:
                positions.append(self.nodes[neighbor_id].weight.copy())

        if len(positions) < p.min_neighbors_for_normal + 1:
            return

        # PCA to find normal (smallest eigenvector)
        positions_arr = np.array(positions)
        cog = np.mean(positions_arr, axis=0)
        centered = positions_arr - cog

        try:
            cov = np.dot(centered.T, centered) / len(positions)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Normal is eigenvector with smallest eigenvalue
            normal = eigenvectors[:, 0].copy()

            # Consistent orientation (pointing "up" or "outward")
            if normal[2] < 0:
                normal = -normal

            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                node.normal = normal / norm

            # PCA residual is smallest eigenvalue (indicates curvature)
            node.pca_residual = eigenvalues[0]

        except np.linalg.LinAlgError:
            pass  # Keep previous normal

    def _detect_edge(self, node_id: int) -> None:
        """Detect if node is on a sharp edge based on normal differences.

        Args:
            node_id: ID of node to check.
        """
        p = self.params
        node = self.nodes[node_id]
        neighbors = self.edges_per_node.get(node_id, set())

        if node.pca_residual < 0:
            # Normal not computed
            node.is_edge = False
            node.edge_strength = 0.0
            return

        # Check normal similarity with neighbors
        min_dot = 1.0  # Track minimum dot product (most different normal)
        edge_count = 0

        for neighbor_id in neighbors:
            neighbor = self.nodes[neighbor_id]
            if neighbor.id == -1 or neighbor.pca_residual < 0:
                continue

            dot = np.abs(np.dot(node.normal, neighbor.normal))
            min_dot = min(min_dot, dot)

            if dot < p.tau_normal:
                edge_count += 1

        # Node is on edge if any neighbor has significantly different normal
        node.is_edge = edge_count > 0
        node.edge_strength = max(0.0, 1.0 - min_dot)  # 0 = same normals, 1 = perpendicular

        # Update edge_nodes set
        if node.is_edge:
            self.edge_nodes.add(node_id)
        else:
            self.edge_nodes.discard(node_id)

    def _get_adaptive_learning_rate(self, node_id: int, base_rate: float) -> float:
        """Get adaptive learning rate based on edge detection.

        Args:
            node_id: ID of node.
            base_rate: Base learning rate (eps_b or eps_n).

        Returns:
            Adjusted learning rate.
        """
        p = self.params
        node = self.nodes[node_id]

        if node.is_edge:
            # Reduce learning rate on edges to preserve them
            return base_rate * p.edge_learning_factor
        else:
            return base_rate

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration with feature preservation.

        Args:
            sample: Input sample vector (3D point).
        """
        p = self.params

        # Find three nearest nodes
        s1_id, s2_id, s3_id = self._find_three_nearest(sample)

        if s1_id == -1 or s2_id == -1 or s3_id == -1:
            return

        # Extended CHL: create/reinforce edges and face
        self._extended_chl(s1_id, s2_id, s3_id)

        # Update winner error
        dist_sq = np.sum((sample - self.nodes[s1_id].weight) ** 2)
        self.nodes[s1_id].error += dist_sq

        # Get adaptive learning rates
        eps_b = self._get_adaptive_learning_rate(s1_id, p.eps_b)

        # Move winner toward sample (with adaptive rate)
        self.nodes[s1_id].weight += eps_b * (sample - self.nodes[s1_id].weight)

        # Move neighbors toward sample (with adaptive rates)
        for neighbor_id in list(self.edges_per_node.get(s1_id, set())):
            eps_n = self._get_adaptive_learning_rate(neighbor_id, p.eps_n)
            self.nodes[neighbor_id].weight += eps_n * (
                sample - self.nodes[neighbor_id].weight
            )

        # Update edge ages
        for neighbor_id in list(self.edges_per_node.get(s1_id, set())):
            self.edges[s1_id, neighbor_id] += 1
            self.edges[neighbor_id, s1_id] += 1

        # Remove invalid edges and faces
        self._remove_invalid_edges_and_faces(s1_id)

        # Update normals and edge detection for winner and neighbors
        self._compute_node_normal(s1_id)
        self._detect_edge(s1_id)
        for neighbor_id in self.edges_per_node.get(s1_id, set()):
            self._compute_node_normal(neighbor_id)
            self._detect_edge(neighbor_id)

        # Periodically insert new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            if self._addable_indices:
                self._insert_node_gcs_feature()

        # Decay all errors
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error *= (1 - p.beta)

        self.n_learning += 1

    def _insert_node_gcs_feature(self) -> int:
        """Insert a new node with edge-aware bias.

        Extends GCS-style insertion to prefer inserting on edges.

        Returns:
            ID of new node, or -1 if failed.
        """
        p = self.params

        if not self._addable_indices:
            return -1

        # Find node q with maximum weighted error
        # Weight by edge_insertion_bias if on edge
        max_weighted_err = -1.0
        q_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            weighted_err = node.error
            if node.is_edge:
                weighted_err *= p.edge_insertion_bias

            if weighted_err > max_weighted_err:
                max_weighted_err = weighted_err
                q_id = node.id

        if q_id == -1:
            return -1

        # Find neighbor f of q with maximum weighted error
        max_err_f = -1.0
        f_id = -1

        for neighbor_id in self.edges_per_node.get(q_id, set()):
            neighbor = self.nodes[neighbor_id]
            weighted_err = neighbor.error
            if neighbor.is_edge:
                weighted_err *= p.edge_insertion_bias

            if weighted_err > max_err_f:
                max_err_f = weighted_err
                f_id = neighbor_id

        if f_id == -1:
            return -1

        # Get faces incident to (q, f) edge
        edge_key = self._edge_key(q_id, f_id)
        incident_faces = list(self.faces_per_edge.get(edge_key, set()))

        # Find common neighbors
        common_neighbors = self._find_common_neighbors(q_id, f_id)

        # Create new node at midpoint
        new_weight = (self.nodes[q_id].weight + self.nodes[f_id].weight) * 0.5
        r_id = self._add_node(new_weight)

        if r_id == -1:
            return -1

        # Inherit edge properties
        q_node = self.nodes[q_id]
        f_node = self.nodes[f_id]

        # New node inherits edge status if both parents are edges
        if q_node.is_edge and f_node.is_edge:
            self.nodes[r_id].is_edge = True
            self.nodes[r_id].edge_strength = (q_node.edge_strength + f_node.edge_strength) / 2
            self.edge_nodes.add(r_id)

        # Interpolate normal
        new_normal = (q_node.normal + f_node.normal) / 2
        norm = np.linalg.norm(new_normal)
        if norm > 1e-10:
            self.nodes[r_id].normal = new_normal / norm

        # Update errors
        self.nodes[q_id].error *= (1 - p.alpha)
        self.nodes[f_id].error *= (1 - p.alpha)
        self.nodes[r_id].error = (
            self.nodes[q_id].error + self.nodes[f_id].error
        ) * 0.5

        # Remove old faces and create new ones (split)
        for face_id in incident_faces:
            if face_id in self.faces:
                vertices = self.faces[face_id]
                self._remove_face(face_id)

                # Find third vertex
                third_vertex = None
                for v in vertices:
                    if v != q_id and v != f_id:
                        third_vertex = v
                        break

                if third_vertex is not None:
                    self._add_face(q_id, r_id, third_vertex)
                    self._add_face(r_id, f_id, third_vertex)

        # Remove old edge and add new edges
        self._remove_edge(q_id, f_id)
        self._add_edge(q_id, r_id)
        self._add_edge(f_id, r_id)

        # Connect to common neighbors
        for cn in common_neighbors:
            self._add_edge(r_id, cn)

        return r_id

    def get_edge_nodes(self) -> np.ndarray:
        """Get positions of nodes on sharp edges.

        Returns:
            Array of shape (n_edge_nodes, 3) with positions.
        """
        edge_positions = []
        for node_id in self.edge_nodes:
            if self.nodes[node_id].id != -1:
                edge_positions.append(self.nodes[node_id].weight.copy())

        return np.array(edge_positions) if edge_positions else np.zeros((0, 3))

    def get_node_normals(self) -> np.ndarray:
        """Get normal vectors for active nodes.

        Returns:
            Array of shape (n_active_nodes, 3) with normals.
        """
        return np.array([
            node.normal for node in self.nodes if node.id != -1
        ])

    def get_edge_strength(self) -> np.ndarray:
        """Get edge strength for active nodes.

        Returns:
            Array of edge strengths (0 = flat, 1 = sharp edge).
        """
        return np.array([
            node.edge_strength for node in self.nodes if node.id != -1
        ])

    def get_is_edge(self) -> np.ndarray:
        """Get edge flags for active nodes.

        Returns:
            Boolean array indicating which nodes are on edges.
        """
        return np.array([
            node.is_edge for node in self.nodes if node.id != -1
        ])

    @property
    def n_edge_nodes(self) -> int:
        """Number of nodes detected as being on sharp edges."""
        return len(self.edge_nodes)
