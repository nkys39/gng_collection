"""GNG-DT (Growing Neural Gas with Different Topologies) implementation.

Based on:
    - Toda, Y., et al. (2022). "GNG with Different Topologies for 3D Point Cloud"
    - Reference implementation: toda_gngdt (C)

GNG-DT learns multiple independent edge topologies based on different properties:
    - Position topology: Standard GNG edges based on spatial proximity
    - Color topology: Edges based on color similarity (threshold-based)
    - Normal topology: Edges based on normal vector similarity (dot product threshold)

Key difference from standard GNG:
    - Winner selection uses ONLY position information
    - Each property (color, normal) has its own independent edge topology
    - Normal vectors are computed via PCA on neighbor node positions

See REFERENCE.md for details.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GNGDTParams:
    """GNG-DT hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node (η1 in paper).
        eps_n: Learning rate for neighbor nodes (η2 in paper).
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        max_age: Maximum edge age before removal (g_max in paper).
        tau_color: Color similarity threshold (τ^col in paper).
        tau_normal: Normal similarity threshold as dot product (τ^nor in paper).
            Higher value means more similar (1.0 = identical, 0.0 = perpendicular).
        pca_min_neighbors: Minimum neighbors for PCA normal computation.
    """

    max_nodes: int = 100
    lambda_: int = 200  # Original: ramda = 200
    eps_b: float = 0.05  # Original: e1 = 0.05
    eps_n: float = 0.0005  # Original: e2 = 0.0005
    alpha: float = 0.5
    beta: float = 0.005
    max_age: int = 88  # Original: MAX_AGE = 88
    # GNG-DT specific parameters
    tau_color: float = 0.05  # Original: cthv = 0.05 (Euclidean distance)
    tau_normal: float = 0.998  # Original: nthv = 0.998 (|dot product| > 0.998)
    pca_min_neighbors: int = 3  # Minimum neighbors for PCA


@dataclass
class GNGDTNode:
    """A neuron node in the GNG-DT network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        position: 3D position vector.
        color: RGB color vector (normalized to [0, 1]).
        normal: Normal vector (unit vector).
        error: Accumulated error.
    """

    id: int = -1
    position: np.ndarray = field(default_factory=lambda: np.array([]))
    color: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    error: float = 1.0


class GrowingNeuralGasDT:
    """GNG-DT (Different Topologies) algorithm implementation.

    This implementation follows the GNG-DT paper by Toda et al. (2022).
    It maintains multiple independent edge topologies for different properties.

    Attributes:
        params: GNG-DT hyperparameters.
        nodes: List of neuron nodes.
        edges_pos: Position-based edge age matrix.
        edges_color: Color-based edge connectivity matrix.
        edges_normal: Normal-based edge connectivity matrix.
        edges_per_node: Adjacency list for position edges (quick neighbor lookup).
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GrowingNeuralGasDT, GNGDTParams
    >>> # Sample 3D points with position and color
    >>> points = np.random.rand(1000, 3)  # Just position
    >>> gng = GrowingNeuralGasDT(params=GNGDTParams())
    >>> gng.train(points, n_iterations=5000)
    >>> nodes, pos_edges, normal_edges = gng.get_multi_graph()
    """

    def __init__(
        self,
        params: GNGDTParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG-DT.

        Args:
            params: GNG-DT hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = 3  # GNG-DT is for 3D point clouds
        self.params = params or GNGDTParams()
        self.rng = np.random.default_rng(seed)

        # Node management (fixed-size array)
        self.nodes: list[GNGDTNode] = [
            GNGDTNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        # Multiple edge topologies
        max_n = self.params.max_nodes

        # Position edges: age matrix (0 = no edge, >=1 = connected with age)
        self.edges_pos = np.zeros((max_n, max_n), dtype=np.int32)
        self.edges_per_node: dict[int, set[int]] = {}

        # Color edges: binary connectivity (0 = no edge, 1 = connected)
        self.edges_color = np.zeros((max_n, max_n), dtype=np.int32)

        # Normal edges: binary connectivity (0 = no edge, 1 = connected)
        self.edges_normal = np.zeros((max_n, max_n), dtype=np.int32)

        # Counters
        self.n_learning = 0
        self._n_trial = 0

        # Initialize with 2 random nodes
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes."""
        for _ in range(2):
            position = self.rng.random(3).astype(np.float32)
            self._add_node(position)

    def _add_node(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
        normal: np.ndarray | None = None,
    ) -> int:
        """Add a new node with given properties.

        Args:
            position: 3D position vector.
            color: RGB color vector (optional).
            normal: Normal vector (optional).

        Returns:
            ID of the new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1  # No space

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = GNGDTNode(
            id=node_id,
            position=position.copy(),
            color=color.copy() if color is not None else np.array([0.0, 0.0, 0.0]),
            normal=normal.copy() if normal is not None else np.array([0.0, 0.0, 1.0]),
            error=1.0,
        )
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node (only if isolated in position topology).

        Args:
            node_id: ID of node to remove.
        """
        if self.edges_per_node.get(node_id):
            return  # Has edges, don't remove

        # Clear all edge topologies
        for other_id in range(self.params.max_nodes):
            self.edges_color[node_id, other_id] = 0
            self.edges_color[other_id, node_id] = 0
            self.edges_normal[node_id, other_id] = 0
            self.edges_normal[other_id, node_id] = 0

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _add_edge_pos(self, node1: int, node2: int) -> None:
        """Add or reset position edge between two nodes.

        Args:
            node1: First node ID.
            node2: Second node ID.
        """
        if self.edges_pos[node1, node2] > 0:
            # Edge exists, reset age
            self.edges_pos[node1, node2] = 1
            self.edges_pos[node2, node1] = 1
        else:
            # New edge
            self.edges_per_node[node1].add(node2)
            self.edges_per_node[node2].add(node1)
            self.edges_pos[node1, node2] = 1
            self.edges_pos[node2, node1] = 1

    def _remove_edge_pos(self, node1: int, node2: int) -> None:
        """Remove position edge between two nodes.

        Args:
            node1: First node ID.
            node2: Second node ID.
        """
        self.edges_per_node[node1].discard(node2)
        self.edges_per_node[node2].discard(node1)
        self.edges_pos[node1, node2] = 0
        self.edges_pos[node2, node1] = 0

    def _update_property_edges(self, node1: int, node2: int) -> None:
        """Update color and normal edges based on thresholds.

        This is called after position edge is created/updated.
        Property edges are independent of position edges.

        Args:
            node1: First node ID.
            node2: Second node ID.
        """
        p = self.params
        n1 = self.nodes[node1]
        n2 = self.nodes[node2]

        # Color edge: connected if color difference < tau_color
        color_diff = np.linalg.norm(n1.color - n2.color)
        if color_diff < p.tau_color:
            self.edges_color[node1, node2] = 1
            self.edges_color[node2, node1] = 1
        else:
            self.edges_color[node1, node2] = 0
            self.edges_color[node2, node1] = 0

        # Normal edge: connected if |dot product| > tau_normal
        # Original uses fabs() to handle normals pointing in opposite directions
        dot_product = np.abs(np.dot(n1.normal, n2.normal))
        if dot_product > p.tau_normal:
            self.edges_normal[node1, node2] = 1
            self.edges_normal[node2, node1] = 1
        else:
            self.edges_normal[node1, node2] = 0
            self.edges_normal[node2, node1] = 0

    def _compute_normal_pca(self, node_id: int) -> np.ndarray:
        """Compute normal vector using PCA on neighbor positions.

        The normal is the eigenvector corresponding to the smallest eigenvalue.

        Args:
            node_id: Node ID.

        Returns:
            Unit normal vector.
        """
        neighbors = list(self.edges_per_node.get(node_id, set()))
        if len(neighbors) < self.params.pca_min_neighbors:
            # Not enough neighbors, keep current normal
            return self.nodes[node_id].normal.copy()

        # Collect neighbor positions including self
        positions = [self.nodes[node_id].position]
        for nid in neighbors:
            if self.nodes[nid].id != -1:
                positions.append(self.nodes[nid].position)

        if len(positions) < self.params.pca_min_neighbors:
            return self.nodes[node_id].normal.copy()

        positions = np.array(positions)

        # Compute covariance matrix
        centroid = np.mean(positions, axis=0)
        centered = positions - centroid
        cov = np.dot(centered.T, centered) / len(positions)

        # Eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Smallest eigenvalue corresponds to normal direction
            normal = eigenvectors[:, 0]
            # Ensure consistent orientation (pointing up-ish if vertical surface)
            if normal[2] < 0:
                normal = -normal
            return normal
        except np.linalg.LinAlgError:
            return self.nodes[node_id].normal.copy()

    def _find_two_nearest(self, position: np.ndarray) -> tuple[int, int]:
        """Find the two nearest nodes to input position.

        Winner selection uses ONLY position (key GNG-DT feature).

        Args:
            position: Input 3D position vector.

        Returns:
            Tuple of (winner_id, second_winner_id).
        """
        min_dist1 = float("inf")
        min_dist2 = float("inf")
        s1_id = -1
        s2_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            # Distance based on POSITION ONLY
            dist = np.sum((position - node.position) ** 2)

            if dist < min_dist1:
                min_dist2 = min_dist1
                s2_id = s1_id
                min_dist1 = dist
                s1_id = node.id
            elif dist < min_dist2:
                min_dist2 = dist
                s2_id = node.id

        return s1_id, s2_id

    def _one_train_update(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
    ) -> None:
        """Single training iteration.

        Args:
            position: Input 3D position vector.
            color: Optional RGB color vector.
        """
        p = self.params

        # Decay all errors
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error

        # Find two nearest nodes (using position only)
        s1_id, s2_id = self._find_two_nearest(position)

        if s1_id == -1 or s2_id == -1:
            return

        # Update winner error
        dist_sq = np.sum((position - self.nodes[s1_id].position) ** 2)
        self.nodes[s1_id].error += dist_sq

        # Move winner toward sample
        self.nodes[s1_id].position += p.eps_b * (
            position - self.nodes[s1_id].position
        )

        # Update winner color if provided
        if color is not None:
            self.nodes[s1_id].color += p.eps_b * (color - self.nodes[s1_id].color)

        # Connect s1 and s2 (position topology)
        self._add_edge_pos(s1_id, s2_id)

        # Update property edges between s1 and s2
        self._update_property_edges(s1_id, s2_id)

        # Update neighbors and age edges
        edges_to_remove = []
        for neighbor_id in list(self.edges_per_node[s1_id]):
            if self.edges_pos[s1_id, neighbor_id] > p.max_age:
                edges_to_remove.append(neighbor_id)
            else:
                # Move neighbor toward sample
                self.nodes[neighbor_id].position += p.eps_n * (
                    position - self.nodes[neighbor_id].position
                )
                # Update neighbor color if provided
                if color is not None:
                    self.nodes[neighbor_id].color += p.eps_n * (
                        color - self.nodes[neighbor_id].color
                    )
                # Increment edge age
                self.edges_pos[s1_id, neighbor_id] += 1
                self.edges_pos[neighbor_id, s1_id] += 1

        # Remove old edges and isolated nodes
        for neighbor_id in edges_to_remove:
            self._remove_edge_pos(s1_id, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # Periodically update normals using PCA
        if self.n_learning % 50 == 0:
            self.nodes[s1_id].normal = self._compute_normal_pca(s1_id)
            # Update normal edges for all neighbors
            for neighbor_id in self.edges_per_node.get(s1_id, set()):
                self._update_property_edges(s1_id, neighbor_id)

        # Periodically add new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0

            if self._addable_indices:
                # Find node with maximum error
                max_err_q = 0.0
                q_id = -1
                for node in self.nodes:
                    if node.id == -1:
                        continue
                    if node.error > max_err_q:
                        max_err_q = node.error
                        q_id = node.id

                if q_id == -1:
                    self.n_learning += 1
                    return

                # Find neighbor of q with maximum error
                max_err_f = 0.0
                f_id = -1
                for neighbor_id in self.edges_per_node.get(q_id, set()):
                    if self.nodes[neighbor_id].error > max_err_f:
                        max_err_f = self.nodes[neighbor_id].error
                        f_id = neighbor_id

                if f_id == -1:
                    self.n_learning += 1
                    return

                # Add new node between q and f
                new_pos = (
                    self.nodes[q_id].position + self.nodes[f_id].position
                ) * 0.5
                new_color = (self.nodes[q_id].color + self.nodes[f_id].color) * 0.5
                new_id = self._add_node(new_pos, new_color)

                if new_id == -1:
                    self.n_learning += 1
                    return

                # Update edges
                self._remove_edge_pos(q_id, f_id)
                self._add_edge_pos(q_id, new_id)
                self._add_edge_pos(f_id, new_id)

                # Update property edges for new node
                self._update_property_edges(q_id, new_id)
                self._update_property_edges(f_id, new_id)

                # Update errors
                self.nodes[q_id].error *= 1 - p.alpha
                self.nodes[f_id].error *= 1 - p.alpha
                self.nodes[new_id].error = (
                    self.nodes[q_id].error + self.nodes[f_id].error
                ) * 0.5

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        colors: np.ndarray | None = None,
        n_iterations: int = 1000,
        callback: Callable[[GrowingNeuralGasDT, int], None] | None = None,
    ) -> GrowingNeuralGasDT:
        """Train on data for multiple iterations.

        Each iteration randomly samples one point from data.

        Args:
            data: Training data of shape (n_samples, 3) for positions.
            colors: Optional color data of shape (n_samples, 3).
            n_iterations: Number of training iterations.
            callback: Optional callback(self, iteration) called each iteration.

        Returns:
            self for chaining.
        """
        n_samples = len(data)

        for i in range(n_iterations):
            idx = self.rng.integers(0, n_samples)
            color = colors[idx] if colors is not None else None
            self._one_train_update(data[idx], color)

            if callback is not None:
                callback(self, i)

        # Final normal computation for all nodes
        self._update_all_normals()

        return self

    def _update_all_normals(self) -> None:
        """Update normal vectors for all nodes using PCA."""
        for node in self.nodes:
            if node.id == -1:
                continue
            node.normal = self._compute_normal_pca(node.id)

        # Update all normal edges
        for node in self.nodes:
            if node.id == -1:
                continue
            for neighbor_id in self.edges_per_node.get(node.id, set()):
                self._update_property_edges(node.id, neighbor_id)

    def partial_fit(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
    ) -> GrowingNeuralGasDT:
        """Single online learning step.

        Args:
            position: Input position vector of shape (3,).
            color: Optional color vector of shape (3,).

        Returns:
            self for chaining.
        """
        self._one_train_update(position, color)
        return self

    @property
    def n_nodes(self) -> int:
        """Number of active nodes."""
        return sum(1 for node in self.nodes if node.id != -1)

    @property
    def n_edges_pos(self) -> int:
        """Number of position edges."""
        count = 0
        for node_id, neighbors in self.edges_per_node.items():
            if self.nodes[node_id].id != -1:
                count += len(neighbors)
        return count // 2

    @property
    def n_edges_normal(self) -> int:
        """Number of normal edges."""
        count = 0
        for i, node in enumerate(self.nodes):
            if node.id == -1:
                continue
            for j in range(i + 1, self.params.max_nodes):
                if self.nodes[j].id == -1:
                    continue
                if self.edges_normal[i, j] > 0:
                    count += 1
        return count

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure (position topology only).

        For compatibility with standard GNG visualization.

        Returns:
            Tuple of:
                - nodes: Array of shape (n_active_nodes, 3) with positions.
                - edges: List of (i, j) tuples indexing into nodes array.
        """
        # Get active nodes and create index mapping
        active_nodes = []
        index_map = {}

        for node in self.nodes:
            if node.id != -1:
                index_map[node.id] = len(active_nodes)
                active_nodes.append(node.position.copy())

        nodes = (
            np.array(active_nodes)
            if active_nodes
            else np.array([]).reshape(0, 3)
        )

        # Get position edges using new indices
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

        return nodes, edges

    def get_multi_graph(
        self,
    ) -> tuple[
        np.ndarray,
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
    ]:
        """Get current graph structure with all topologies.

        Returns:
            Tuple of:
                - nodes: Array of shape (n_active_nodes, 3) with positions.
                - pos_edges: Position-based edges (standard GNG edges).
                - color_edges: Color-similarity edges.
                - normal_edges: Normal-similarity edges.
        """
        # Get active nodes and create index mapping
        active_nodes = []
        index_map = {}

        for node in self.nodes:
            if node.id != -1:
                index_map[node.id] = len(active_nodes)
                active_nodes.append(node.position.copy())

        nodes = (
            np.array(active_nodes)
            if active_nodes
            else np.array([]).reshape(0, 3)
        )

        # Get position edges
        pos_edges = []
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
                    pos_edges.append((index_map[node_id], index_map[neighbor_id]))

        # Get color edges
        color_edges = []
        for i, node_i in enumerate(self.nodes):
            if node_i.id == -1:
                continue
            for j in range(i + 1, self.params.max_nodes):
                node_j = self.nodes[j]
                if node_j.id == -1:
                    continue
                if self.edges_color[i, j] > 0:
                    color_edges.append((index_map[i], index_map[j]))

        # Get normal edges
        normal_edges = []
        for i, node_i in enumerate(self.nodes):
            if node_i.id == -1:
                continue
            for j in range(i + 1, self.params.max_nodes):
                node_j = self.nodes[j]
                if node_j.id == -1:
                    continue
                if self.edges_normal[i, j] > 0:
                    normal_edges.append((index_map[i], index_map[j]))

        return nodes, pos_edges, color_edges, normal_edges

    def get_node_normals(self) -> np.ndarray:
        """Get normal vectors for active nodes.

        Returns:
            Array of normal vectors in same order as get_graph() nodes.
        """
        return np.array([node.normal for node in self.nodes if node.id != -1])

    def get_node_colors(self) -> np.ndarray:
        """Get color vectors for active nodes.

        Returns:
            Array of color vectors in same order as get_graph() nodes.
        """
        return np.array([node.color for node in self.nodes if node.id != -1])

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes.

        Returns:
            Array of error values in same order as get_graph() nodes.
        """
        return np.array([node.error for node in self.nodes if node.id != -1])


# Alias
GNGDT = GrowingNeuralGasDT
