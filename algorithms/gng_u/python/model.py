"""Growing Neural Gas with Utility (GNG-U) implementation.

Based on:
    - Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
    - Fritzke, B. (1997). "Some Competitive Learning Methods"
    - Fritzke, B. (1999). "Be Busy and Unique — or Be History—The Utility Criterion"

GNG-U extends GNG with a utility measure that allows tracking non-stationary
distributions by removing nodes with low utility.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GNGUParams:
    """GNG-U hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for neighbor nodes.
        alpha: Error decay rate when splitting.
        beta: Global error/utility decay rate.
        max_age: Maximum edge age before removal.
        utility_k: Utility threshold for node removal (recommended: 1.3).
    """

    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.08
    eps_n: float = 0.008
    alpha: float = 0.5
    beta: float = 0.005
    max_age: int = 100
    utility_k: float = 1.3  # GNG-U specific


@dataclass
class NeuronNode:
    """A neuron node in the GNG-U network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector.
        error: Accumulated error.
        utility: Utility measure (how useful this node is).
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0
    utility: float = 1.0  # GNG-U specific


class GrowingNeuralGasU:
    """Growing Neural Gas with Utility (GNG-U) implementation.

    GNG-U extends the standard GNG algorithm with a utility measure that
    tracks how useful each node is for reducing the network error.
    Nodes with low utility can be removed, allowing the network to
    track non-stationary distributions.

    The utility of a node represents how much the total network error
    would increase if that node were removed.

    Attributes:
        params: GNG-U hyperparameters.
        nodes: List of neuron nodes.
        edges: Edge age matrix.
        edges_per_node: Adjacency list for quick neighbor lookup.
        n_learning: Total number of learning iterations.
        n_removals: Number of nodes removed by utility criterion.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GrowingNeuralGasU, GNGUParams
    >>> X = np.random.rand(1000, 2)
    >>> params = GNGUParams(utility_k=1.3)
    >>> gng = GrowingNeuralGasU(n_dim=2, params=params)
    >>> gng.train(X, n_iterations=5000)
    >>> nodes, edges = gng.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GNGUParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG-U.

        Args:
            n_dim: Dimension of input data.
            params: GNG-U hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or GNGUParams()
        self.rng = np.random.default_rng(seed)

        # Node management (fixed-size array like reference impl)
        self.nodes: list[NeuronNode] = [
            NeuronNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        # Edge management
        self.edges = np.zeros(
            (self.params.max_nodes, self.params.max_nodes), dtype=np.int32
        )
        self.edges_per_node: dict[int, set[int]] = {}

        # Counters
        self.n_learning = 0
        self._n_trial = 0
        self.n_removals = 0  # GNG-U: track utility-based removals

        # Initialize with 2 random nodes
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes."""
        for _ in range(2):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            self._add_node(weight)

    def _add_node(self, weight: np.ndarray, utility: float = 1.0) -> int:
        """Add a new node with given weight.

        Args:
            weight: Position vector for the new node.
            utility: Initial utility value.

        Returns:
            ID of the new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = NeuronNode(
            id=node_id, weight=weight.copy(), error=1.0, utility=utility
        )
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int, force: bool = False) -> bool:
        """Remove a node.

        Args:
            node_id: ID of node to remove.
            force: If True, remove even if node has edges (for utility removal).

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
        """Add or reset edge between two nodes."""
        if self.edges[node1, node2] > 0:
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1
        else:
            self.edges_per_node[node1].add(node2)
            self.edges_per_node[node2].add(node1)
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1

    def _remove_edge(self, node1: int, node2: int) -> None:
        """Remove edge between two nodes."""
        self.edges_per_node[node1].discard(node2)
        self.edges_per_node[node2].discard(node1)
        self.edges[node1, node2] = 0
        self.edges[node2, node1] = 0

    def _find_two_nearest(self, x: np.ndarray) -> tuple[int, int, float, float]:
        """Find the two nearest nodes to input x.

        Args:
            x: Input vector.

        Returns:
            Tuple of (winner_id, second_winner_id, dist_winner, dist_second).
        """
        min_dist1 = float("inf")
        min_dist2 = float("inf")
        s1_id = -1
        s2_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            dist = np.sum((x - node.weight) ** 2)

            if dist < min_dist1:
                min_dist2 = min_dist1
                s2_id = s1_id
                min_dist1 = dist
                s1_id = node.id
            elif dist < min_dist2:
                min_dist2 = dist
                s2_id = node.id

        return s1_id, s2_id, min_dist1, min_dist2

    def _check_utility_criterion(self) -> None:
        """Check and remove node with lowest utility if criterion met.

        GNG-U specific: Remove node if max_error / min_utility > k
        """
        p = self.params

        # Find max error and min utility
        max_error = 0.0
        min_utility = float("inf")
        min_utility_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue
            if node.error > max_error:
                max_error = node.error
            if node.utility < min_utility:
                min_utility = node.utility
                min_utility_id = node.id

        # Check criterion: max_error / min_utility > k
        if min_utility_id != -1 and min_utility > 0:
            if max_error / min_utility > p.utility_k:
                # Remove node with minimum utility
                if self._remove_node(min_utility_id, force=True):
                    self.n_removals += 1

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration with utility update.

        Args:
            sample: Input sample vector.
        """
        p = self.params

        # Decay all errors and utilities
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error
            node.utility -= p.beta * node.utility  # GNG-U: decay utility

        # Find two nearest nodes (with distances)
        s1_id, s2_id, dist1_sq, dist2_sq = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        # Update winner error
        dist1 = np.sqrt(dist1_sq)
        self.nodes[s1_id].error += dist1

        # GNG-U: Update winner utility
        # Utility represents how much error would increase if this node were removed
        self.nodes[s1_id].utility += dist2_sq - dist1_sq

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
                    return

                # Find neighbor of q with maximum error
                max_err_f = 0.0
                f_id = -1
                for neighbor_id in self.edges_per_node.get(q_id, set()):
                    if self.nodes[neighbor_id].error > max_err_f:
                        max_err_f = self.nodes[neighbor_id].error
                        f_id = neighbor_id

                if f_id == -1:
                    return

                # Add new node between q and f
                new_weight = (self.nodes[q_id].weight + self.nodes[f_id].weight) * 0.5
                # GNG-U: New node inherits averaged utility
                new_utility = (
                    self.nodes[q_id].utility + self.nodes[f_id].utility
                ) * 0.5
                new_id = self._add_node(new_weight, utility=new_utility)

                if new_id == -1:
                    return

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

                # GNG-U: Check utility criterion after insertion
                self._check_utility_criterion()

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GrowingNeuralGasU, int], None] | None = None,
    ) -> GrowingNeuralGasU:
        """Train on data for multiple iterations.

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

    def partial_fit(self, sample: np.ndarray) -> GrowingNeuralGasU:
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
        return count // 2

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure.

        Returns:
            Tuple of (nodes_array, edges_list).
        """
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

    def get_node_utilities(self) -> np.ndarray:
        """Get utility values for active nodes.

        Returns:
            Array of utility values in same order as get_graph() nodes.
        """
        return np.array([node.utility for node in self.nodes if node.id != -1])

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes.

        Returns:
            Array of error values in same order as get_graph() nodes.
        """
        return np.array([node.error for node in self.nodes if node.id != -1])


# Aliases
GNGU = GrowingNeuralGasU
