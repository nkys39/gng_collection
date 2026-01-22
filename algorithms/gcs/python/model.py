"""Growing Cell Structures (GCS) implementation.

Based on:
    - Fritzke, B. (1994). "Growing cell structures - a self-organizing network
      for unsupervised and supervised learning"

GCS maintains a k-dimensional simplicial complex (triangular mesh in 2D).
New nodes are inserted at the centroid of the cell with highest accumulated error.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GCSParams:
    """GCS hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval.
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for neighbor nodes.
        alpha: Error decay rate when inserting.
        beta: Global error decay rate.
    """

    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.1
    eps_n: float = 0.01
    alpha: float = 0.5
    beta: float = 0.005


@dataclass
class GCSNode:
    """A node in the GCS network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector.
        error: Accumulated error (signal counter).
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 0.0


class GrowingCellStructures:
    """Growing Cell Structures (GCS) implementation.

    GCS is a self-organizing network that maintains a simplicial complex
    (triangular mesh in 2D). Unlike GNG which learns arbitrary topologies,
    GCS always maintains a connected mesh structure.

    Attributes:
        params: GCS hyperparameters.
        nodes: List of nodes.
        edges: Adjacency set for each node.
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GrowingCellStructures
    >>> X = np.random.rand(1000, 2)
    >>> gcs = GrowingCellStructures(n_dim=2)
    >>> gcs.train(X, n_iterations=5000)
    >>> nodes, edges = gcs.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GCSParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GCS.

        Args:
            n_dim: Dimension of input data.
            params: GCS hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or GCSParams()
        self.rng = np.random.default_rng(seed)

        # Node management
        self.nodes: list[GCSNode] = [
            GCSNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        # Edge management (adjacency sets)
        self.edges: dict[int, set[int]] = {}

        # Counters
        self.n_learning = 0
        self._n_trial = 0

        # Initialize with a triangle (minimum 2D simplicial complex)
        self._init_triangle()

    def _init_triangle(self) -> None:
        """Initialize with a triangle of 3 nodes."""
        # Create 3 nodes forming a triangle
        positions = [
            np.array([0.3, 0.3]),
            np.array([0.7, 0.3]),
            np.array([0.5, 0.7]),
        ]

        node_ids = []
        for pos in positions:
            weight = pos + self.rng.random(self.n_dim) * 0.1
            node_id = self._add_node(weight.astype(np.float32))
            node_ids.append(node_id)

        # Connect all pairs (triangle edges)
        for i in range(3):
            for j in range(i + 1, 3):
                self._add_edge(node_ids[i], node_ids[j])

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
        self.nodes[node_id] = GCSNode(id=node_id, weight=weight.copy(), error=0.0)
        self.edges[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node and its edges.

        Args:
            node_id: ID of node to remove.
        """
        # Remove all edges
        for neighbor in list(self.edges.get(node_id, set())):
            self.edges[neighbor].discard(node_id)
        self.edges.pop(node_id, None)

        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _add_edge(self, n1: int, n2: int) -> None:
        """Add edge between two nodes.

        Args:
            n1: First node ID.
            n2: Second node ID.
        """
        self.edges[n1].add(n2)
        self.edges[n2].add(n1)

    def _remove_edge(self, n1: int, n2: int) -> None:
        """Remove edge between two nodes.

        Args:
            n1: First node ID.
            n2: Second node ID.
        """
        self.edges[n1].discard(n2)
        self.edges[n2].discard(n1)

    def _find_winner(self, x: np.ndarray) -> int:
        """Find the closest node to input x.

        Args:
            x: Input vector.

        Returns:
            ID of the winner node.
        """
        min_dist = float("inf")
        winner = -1

        for node in self.nodes:
            if node.id == -1:
                continue
            dist = np.sum((x - node.weight) ** 2)
            if dist < min_dist:
                min_dist = dist
                winner = node.id

        return winner

    def _find_common_neighbors(self, n1: int, n2: int) -> set[int]:
        """Find nodes that are neighbors of both n1 and n2.

        Args:
            n1: First node ID.
            n2: Second node ID.

        Returns:
            Set of common neighbor IDs.
        """
        return self.edges[n1] & self.edges[n2]

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
        winner_id = self._find_winner(sample)
        if winner_id == -1:
            return

        # Update winner error (squared distance for consistency with GNG)
        dist_sq = np.sum((sample - self.nodes[winner_id].weight) ** 2)
        self.nodes[winner_id].error += dist_sq

        # Move winner toward sample
        self.nodes[winner_id].weight += p.eps_b * (
            sample - self.nodes[winner_id].weight
        )

        # Move neighbors toward sample
        for neighbor_id in self.edges.get(winner_id, set()):
            self.nodes[neighbor_id].weight += p.eps_n * (
                sample - self.nodes[neighbor_id].weight
            )

        # Periodically insert new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            self._insert_node()

        self.n_learning += 1

    def _insert_node(self) -> int:
        """Insert a new node in the highest-error region.

        In GCS, a new node is inserted by finding the node q with highest error,
        then finding its neighbor f with highest error. The new node r is placed
        at the midpoint, and the edge (q, f) is replaced with edges (q, r) and (r, f).

        Additionally, r is connected to any common neighbors of q and f.

        Returns:
            ID of new node, or -1 if failed.
        """
        p = self.params

        if not self._addable_indices:
            return -1

        # Find node q with maximum error
        max_err = 0.0
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
        for neighbor_id in self.edges.get(q_id, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f_id = neighbor_id

        if f_id == -1:
            return -1

        # Find common neighbors (they will also connect to the new node)
        common_neighbors = self._find_common_neighbors(q_id, f_id)

        # Create new node at midpoint
        new_weight = (self.nodes[q_id].weight + self.nodes[f_id].weight) * 0.5
        new_id = self._add_node(new_weight)

        if new_id == -1:
            return -1

        # Update edges: remove (q, f), add (q, r) and (f, r)
        self._remove_edge(q_id, f_id)
        self._add_edge(q_id, new_id)
        self._add_edge(f_id, new_id)

        # Connect to common neighbors (maintaining simplicial structure)
        for cn in common_neighbors:
            self._add_edge(new_id, cn)

        # Update errors
        self.nodes[q_id].error *= (1 - p.alpha)
        self.nodes[f_id].error *= (1 - p.alpha)
        self.nodes[new_id].error = (
            self.nodes[q_id].error + self.nodes[f_id].error
        ) * 0.5

        return new_id

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GrowingCellStructures, int], None] | None = None,
    ) -> GrowingCellStructures:
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

    def partial_fit(self, sample: np.ndarray) -> GrowingCellStructures:
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
        count = 0
        for node_id, neighbors in self.edges.items():
            if self.nodes[node_id].id != -1:
                count += len(neighbors)
        return count // 2

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
        seen = set()
        for node_id, neighbors in self.edges.items():
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

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes.

        Returns:
            Array of error values in same order as get_graph() nodes.
        """
        return np.array([node.error for node in self.nodes if node.id != -1])


# Alias
GCS = GrowingCellStructures
