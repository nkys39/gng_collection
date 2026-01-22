"""[ALGORITHM_NAME] implementation.

Based on:
    - [Author]. ([Year]). "[Paper Title]"
    - Reference implementation: [reference_name] (if any)

See REFERENCE.md for details.

Usage:
    1. Copy this directory to algorithms/[algorithm_name]/python/
    2. Rename class and update docstrings
    3. Implement algorithm-specific logic
    4. Create corresponding test files
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class AlgorithmParams:
    """Algorithm hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for neighbor nodes.
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        max_age: Maximum edge age before removal.
        # Add algorithm-specific parameters below:
        # example_param: float = 1.0  # Description
    """

    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.08
    eps_n: float = 0.008
    alpha: float = 0.5
    beta: float = 0.005
    max_age: int = 100
    # Algorithm-specific parameters:
    # example_param: float = 1.0


@dataclass
class NeuronNode:
    """A neuron node in the network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector.
        error: Accumulated error.
        # Add algorithm-specific fields below:
        # example_field: float = 0.0
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0
    # Algorithm-specific fields:
    # example_field: float = 0.0


class AlgorithmName:
    """[ALGORITHM_NAME] algorithm implementation.

    Brief description of the algorithm and its key features.

    Attributes:
        params: Algorithm hyperparameters.
        nodes: List of neuron nodes.
        edges: Edge age matrix (0 = no edge, >=1 = connected with age).
        edges_per_node: Adjacency list for quick neighbor lookup.
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import AlgorithmName
    >>> X = np.random.rand(1000, 2)
    >>> model = AlgorithmName(n_dim=2)
    >>> model.train(X, n_iterations=5000)
    >>> nodes, edges = model.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: AlgorithmParams | None = None,
        seed: int | None = None,
    ):
        """Initialize algorithm.

        Args:
            n_dim: Dimension of input data.
            params: Algorithm hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or AlgorithmParams()
        self.rng = np.random.default_rng(seed)

        # Node management (fixed-size array)
        self.nodes: list[NeuronNode] = [
            NeuronNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        # Edge management (adjacency matrix for age, dict for quick lookup)
        self.edges = np.zeros(
            (self.params.max_nodes, self.params.max_nodes), dtype=np.int32
        )
        self.edges_per_node: dict[int, set[int]] = {}

        # Counters
        self.n_learning = 0
        self._n_trial = 0
        # Algorithm-specific counters:
        # self.n_removals = 0  # Example: count of removed nodes

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
        self.nodes[node_id] = NeuronNode(id=node_id, weight=weight.copy(), error=1.0)
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int, force: bool = False) -> bool:
        """Remove a node.

        Args:
            node_id: ID of node to remove.
            force: If True, remove even if node has edges.

        Returns:
            True if node was removed.
        """
        if not force and self.edges_per_node.get(node_id):
            return False  # Has edges, don't remove

        # Remove all edges connected to this node
        for neighbor_id in list(self.edges_per_node.get(node_id, set())):
            self._remove_edge(node_id, neighbor_id)

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)
        return True

    def _add_edge(self, node1: int, node2: int) -> None:
        """Add or reset edge between two nodes.

        Args:
            node1: First node ID.
            node2: Second node ID.
        """
        if self.edges[node1, node2] > 0:
            # Edge exists, reset age
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1
        else:
            # New edge
            self.edges_per_node[node1].add(node2)
            self.edges_per_node[node2].add(node1)
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1

    def _remove_edge(self, node1: int, node2: int) -> None:
        """Remove edge between two nodes.

        Args:
            node1: First node ID.
            node2: Second node ID.
        """
        self.edges_per_node[node1].discard(node2)
        self.edges_per_node[node2].discard(node1)
        self.edges[node1, node2] = 0
        self.edges[node2, node1] = 0

    def _find_two_nearest(self, x: np.ndarray) -> tuple[int, int, float, float]:
        """Find the two nearest nodes to input x.

        Args:
            x: Input vector.

        Returns:
            Tuple of (winner_id, second_winner_id, dist1_sq, dist2_sq).
        """
        min_dist1 = float("inf")
        min_dist2 = float("inf")
        s1_id = -1
        s2_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            dist = np.sum((x - node.weight) ** 2)  # Squared distance

            if dist < min_dist1:
                min_dist2 = min_dist1
                s2_id = s1_id
                min_dist1 = dist
                s1_id = node.id
            elif dist < min_dist2:
                min_dist2 = dist
                s2_id = node.id

        return s1_id, s2_id, min_dist1, min_dist2

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration.

        Args:
            sample: Input sample vector.
        """
        p = self.params

        # Decay all errors (and algorithm-specific values)
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error
            # Algorithm-specific decay:
            # node.example_field *= some_decay

        # Find two nearest nodes
        s1_id, s2_id, dist1_sq, dist2_sq = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        # Update winner error
        dist = np.sqrt(dist1_sq)
        self.nodes[s1_id].error += dist

        # Algorithm-specific update (example):
        # self.nodes[s1_id].example_field += some_value

        # Move winner toward sample
        self.nodes[s1_id].weight += p.eps_b * (sample - self.nodes[s1_id].weight)

        # Connect s1 and s2
        self._add_edge(s1_id, s2_id)

        # Update neighbors and age edges
        edges_to_remove = []
        for neighbor_id in list(self.edges_per_node[s1_id]):
            if self.edges[s1_id, neighbor_id] > p.max_age:
                edges_to_remove.append(neighbor_id)
            else:
                # Move neighbor toward sample
                self.nodes[neighbor_id].weight += p.eps_n * (
                    sample - self.nodes[neighbor_id].weight
                )
                # Increment edge age
                self.edges[s1_id, neighbor_id] += 1
                self.edges[neighbor_id, s1_id] += 1

        # Remove old edges and isolated nodes
        for neighbor_id in edges_to_remove:
            self._remove_edge(s1_id, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # Periodically add new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            self._insert_node()
            # Algorithm-specific check after insertion:
            # self._check_algorithm_criterion()

        self.n_learning += 1

    def _insert_node(self) -> int:
        """Insert a new node between highest-error node and its highest-error neighbor.

        Returns:
            ID of new node, or -1 if insertion failed.
        """
        p = self.params

        if not self._addable_indices:
            return -1

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
            return -1

        # Find neighbor of q with maximum error
        max_err_f = 0.0
        f_id = -1
        for neighbor_id in self.edges_per_node.get(q_id, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f_id = neighbor_id

        if f_id == -1:
            return -1

        # Add new node between q and f
        new_weight = (self.nodes[q_id].weight + self.nodes[f_id].weight) * 0.5
        new_id = self._add_node(new_weight)

        if new_id == -1:
            return -1

        # Update edges
        self._remove_edge(q_id, f_id)
        self._add_edge(q_id, new_id)
        self._add_edge(f_id, new_id)

        # Update errors
        self.nodes[q_id].error *= 1 - p.alpha
        self.nodes[f_id].error *= 1 - p.alpha
        self.nodes[new_id].error = (
            self.nodes[q_id].error + self.nodes[f_id].error
        ) * 0.5

        # Algorithm-specific: update new node's fields
        # self.nodes[new_id].example_field = ...

        return new_id

    # Algorithm-specific methods (example):
    # def _check_algorithm_criterion(self) -> None:
    #     """Check and apply algorithm-specific criterion."""
    #     pass

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[AlgorithmName, int], None] | None = None,
    ) -> AlgorithmName:
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

    def partial_fit(self, sample: np.ndarray) -> AlgorithmName:
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
        for node_id, neighbors in self.edges_per_node.items():
            if self.nodes[node_id].id != -1:
                count += len(neighbors)
        return count // 2  # Each edge counted twice

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

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes.

        Returns:
            Array of error values in same order as get_graph() nodes.
        """
        return np.array([node.error for node in self.nodes if node.id != -1])
