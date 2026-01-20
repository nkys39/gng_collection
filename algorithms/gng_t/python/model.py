"""Growing Neural Gas with explicit Delaunay Triangulation (GNG-T).

This variant replaces Competitive Hebbian Learning (CHL) with explicit
Delaunay triangulation for topology management.

Based on:
    - Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
    - Martinetz, T. & Schulten, K. (1994). "Topology representing networks"

The original papers mention two approaches for forming Delaunay triangulation:
1. CHL (online, incremental) - used in standard GNG
2. Explicit Delaunay computation - used in this GNG-T variant

Key differences from standard GNG:
- No edge aging (max_age parameter removed)
- Edges determined purely by geometric Delaunay triangulation
- Topology updated periodically via scipy.spatial.Delaunay
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.spatial import Delaunay


@dataclass
class GNGTParams:
    """GNG-T hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for Delaunay neighbor nodes.
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        update_topology_every: Recompute Delaunay triangulation interval.
    """

    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.05
    eps_n: float = 0.006
    alpha: float = 0.5
    beta: float = 0.0005
    update_topology_every: int = 1  # Update topology every N iterations


@dataclass
class NeuronNode:
    """A neuron node in the GNG-T network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector.
        error: Accumulated error.
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 0.0


class GrowingNeuralGasT:
    """Growing Neural Gas with explicit Delaunay Triangulation.

    This implementation uses scipy.spatial.Delaunay for topology management
    instead of Competitive Hebbian Learning (CHL).

    Attributes:
        params: GNG-T hyperparameters.
        nodes: List of neuron nodes.
        edges: Set of edges as (i, j) tuples (node IDs, i < j).
        neighbors: Adjacency list from Delaunay triangulation.
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GrowingNeuralGasT
    >>> X = np.random.rand(1000, 2)
    >>> gng_t = GrowingNeuralGasT(n_dim=2)
    >>> gng_t.train(X, n_iterations=5000)
    >>> nodes, edges = gng_t.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GNGTParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG-T.

        Args:
            n_dim: Dimension of input data (must be >= 2 for Delaunay).
            params: GNG-T hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        if n_dim < 2:
            raise ValueError("GNG-T requires n_dim >= 2 for Delaunay triangulation")

        self.n_dim = n_dim
        self.params = params or GNGTParams()
        self.rng = np.random.default_rng(seed)

        # Node management (fixed-size array)
        self.nodes: list[NeuronNode] = [
            NeuronNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        # Edge management (set of edges, adjacency list)
        self.edges: set[tuple[int, int]] = set()
        self.neighbors: dict[int, set[int]] = {}

        # Counters
        self.n_learning = 0
        self._n_trial = 0
        self._topology_update_counter = 0

        # Initialize with 2 random nodes
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes."""
        for _ in range(2):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            self._add_node(weight)

    def _add_node(self, weight: np.ndarray) -> int:
        """Add a new node with given weight.

        Args:
            weight: Position vector for the new node.

        Returns:
            ID of the new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1  # No space

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = NeuronNode(id=node_id, weight=weight.copy(), error=0.0)
        self.neighbors[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node.

        Args:
            node_id: ID of node to remove.
        """
        # Remove edges involving this node
        edges_to_remove = [e for e in self.edges if node_id in e]
        for edge in edges_to_remove:
            self.edges.discard(edge)
            other = edge[0] if edge[1] == node_id else edge[1]
            self.neighbors.get(other, set()).discard(node_id)

        self.neighbors.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _get_active_node_ids(self) -> list[int]:
        """Get list of active node IDs."""
        return [node.id for node in self.nodes if node.id != -1]

    def _update_topology(self) -> None:
        """Recompute Delaunay triangulation for current node positions.

        This replaces CHL-based edge management with explicit geometric
        computation of Delaunay triangulation.
        """
        active_ids = self._get_active_node_ids()
        n_active = len(active_ids)

        # Clear current edges
        self.edges.clear()
        for node_id in self.neighbors:
            self.neighbors[node_id].clear()

        # Need at least 3 non-collinear points for Delaunay
        if n_active < 3:
            # Just connect the nodes we have
            if n_active == 2:
                edge = tuple(sorted(active_ids))
                self.edges.add(edge)
                self.neighbors[active_ids[0]].add(active_ids[1])
                self.neighbors[active_ids[1]].add(active_ids[0])
            return

        # Get positions and create ID mapping
        positions = np.array([self.nodes[i].weight for i in active_ids])
        id_to_idx = {node_id: idx for idx, node_id in enumerate(active_ids)}
        idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}

        # Compute Delaunay triangulation
        try:
            tri = Delaunay(positions)
        except Exception:
            # Delaunay may fail for degenerate cases (collinear points, etc.)
            # Fall back to connecting all nodes to their nearest neighbor
            self._fallback_connect_nearest(active_ids)
            return

        # Extract edges from simplices
        for simplex in tri.simplices:
            # Each simplex is a triangle (in 2D) or tetrahedron (in 3D)
            n_vertices = len(simplex)
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    idx1, idx2 = simplex[i], simplex[j]
                    node_id1 = idx_to_id[idx1]
                    node_id2 = idx_to_id[idx2]
                    edge = tuple(sorted([node_id1, node_id2]))

                    if edge not in self.edges:
                        self.edges.add(edge)
                        self.neighbors[node_id1].add(node_id2)
                        self.neighbors[node_id2].add(node_id1)

    def _fallback_connect_nearest(self, active_ids: list[int]) -> None:
        """Fallback connection when Delaunay fails.

        Connects each node to its nearest neighbor.

        Args:
            active_ids: List of active node IDs.
        """
        for i, node_id1 in enumerate(active_ids):
            min_dist = float("inf")
            nearest_id = -1

            for j, node_id2 in enumerate(active_ids):
                if i == j:
                    continue
                dist = np.sum(
                    (self.nodes[node_id1].weight - self.nodes[node_id2].weight) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_id = node_id2

            if nearest_id != -1:
                edge = tuple(sorted([node_id1, nearest_id]))
                if edge not in self.edges:
                    self.edges.add(edge)
                    self.neighbors[node_id1].add(nearest_id)
                    self.neighbors[nearest_id].add(node_id1)

    def _find_winner(self, x: np.ndarray) -> tuple[int, float]:
        """Find the nearest node to input x.

        Args:
            x: Input vector.

        Returns:
            Tuple of (winner_id, squared_distance).
        """
        min_dist = float("inf")
        winner_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            dist = np.sum((x - node.weight) ** 2)
            if dist < min_dist:
                min_dist = dist
                winner_id = node.id

        return winner_id, min_dist

    def _find_max_error_node(self) -> int:
        """Find the node with maximum error.

        Returns:
            Node ID with maximum error, or -1 if no nodes.
        """
        max_err = -1.0
        max_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue
            if node.error > max_err:
                max_err = node.error
                max_id = node.id

        return max_id

    def _find_max_error_neighbor(self, node_id: int) -> int:
        """Find the neighbor with maximum error.

        Args:
            node_id: Node ID to search neighbors of.

        Returns:
            Neighbor ID with maximum error, or -1 if no neighbors.
        """
        max_err = -1.0
        max_id = -1

        for neighbor_id in self.neighbors.get(node_id, set()):
            if self.nodes[neighbor_id].error > max_err:
                max_err = self.nodes[neighbor_id].error
                max_id = neighbor_id

        return max_id

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration.

        Args:
            sample: Input sample vector.
        """
        p = self.params

        # Decay all errors
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error *= (1 - p.beta)

        # Find winner
        winner_id, dist_sq = self._find_winner(sample)

        if winner_id == -1:
            return

        # Update winner error
        self.nodes[winner_id].error += dist_sq

        # Move winner toward sample
        self.nodes[winner_id].weight += p.eps_b * (
            sample - self.nodes[winner_id].weight
        )

        # Move Delaunay neighbors toward sample
        for neighbor_id in self.neighbors.get(winner_id, set()):
            self.nodes[neighbor_id].weight += p.eps_n * (
                sample - self.nodes[neighbor_id].weight
            )

        # Periodically update topology
        self._topology_update_counter += 1
        if self._topology_update_counter >= p.update_topology_every:
            self._topology_update_counter = 0
            self._update_topology()

        # Periodically add new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0

            if self._addable_indices:
                # Find node with maximum error
                q_id = self._find_max_error_node()
                if q_id == -1:
                    return

                # Find neighbor of q with maximum error
                f_id = self._find_max_error_neighbor(q_id)
                if f_id == -1:
                    # No neighbors, pick second highest error node
                    f_id = -1
                    max_err = -1.0
                    for node in self.nodes:
                        if node.id == -1 or node.id == q_id:
                            continue
                        if node.error > max_err:
                            max_err = node.error
                            f_id = node.id

                if f_id == -1:
                    return

                # Add new node between q and f
                new_weight = (
                    self.nodes[q_id].weight + self.nodes[f_id].weight
                ) * 0.5
                new_id = self._add_node(new_weight)

                if new_id == -1:
                    return

                # Update errors
                self.nodes[q_id].error *= (1 - p.alpha)
                self.nodes[f_id].error *= (1 - p.alpha)
                self.nodes[new_id].error = (
                    self.nodes[q_id].error + self.nodes[f_id].error
                ) * 0.5

                # Force topology update after adding node
                self._update_topology()

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GrowingNeuralGasT, int], None] | None = None,
    ) -> GrowingNeuralGasT:
        """Train on data for multiple iterations.

        Each iteration randomly samples one point from data.

        Args:
            data: Training data of shape (n_samples, n_dim).
            n_iterations: Number of training iterations.
            callback: Optional callback(self, iteration) called each iteration.

        Returns:
            self for chaining.
        """
        n_samples = len(data)

        for i in range(n_iterations):
            idx = self.rng.integers(0, n_samples)
            self._one_train_update(data[idx])

            if callback is not None:
                callback(self, i)

        return self

    def partial_fit(self, sample: np.ndarray) -> GrowingNeuralGasT:
        """Single online learning step.

        Args:
            sample: Input vector of shape (n_dim,).

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
        return len(self.edges)

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure.

        Returns:
            Tuple of:
                - nodes: Array of shape (n_active_nodes, n_dim) with positions.
                - edges: List of (i, j) tuples indexing into nodes array.
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
        for edge in self.edges:
            if edge[0] in index_map and edge[1] in index_map:
                edges.append((index_map[edge[0]], index_map[edge[1]]))

        return nodes, edges

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes.

        Returns:
            Array of error values in same order as get_graph() nodes.
        """
        return np.array([node.error for node in self.nodes if node.id != -1])

    def get_triangles(self) -> list[tuple[int, int, int]]:
        """Get triangles from Delaunay triangulation.

        Returns:
            List of (i, j, k) tuples representing triangles,
            indexing into get_graph() nodes array.
        """
        active_ids = self._get_active_node_ids()
        n_active = len(active_ids)

        if n_active < 3:
            return []

        positions = np.array([self.nodes[i].weight for i in active_ids])
        id_to_idx = {node_id: idx for idx, node_id in enumerate(active_ids)}

        try:
            tri = Delaunay(positions)
        except Exception:
            return []

        # Create index mapping for get_graph output
        graph_index_map = {}
        idx = 0
        for node in self.nodes:
            if node.id != -1:
                graph_index_map[node.id] = idx
                idx += 1

        triangles = []
        for simplex in tri.simplices:
            if len(simplex) == 3:  # 2D case
                ids = [active_ids[i] for i in simplex]
                graph_indices = tuple(graph_index_map[id_] for id_ in ids)
                triangles.append(graph_indices)

        return triangles


# Aliases
GNGT = GrowingNeuralGasT
GNG_T = GrowingNeuralGasT
