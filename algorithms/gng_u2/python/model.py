"""Growing Neural Gas with Utility - Variant 2 (GNG-U2) implementation.

Based on:
    - Toda, Y., et al. (2016). "Real-time 3D point cloud segmentation
      using Growing Neural Gas with Utility"
    - IEEE International Conference on Robotics and Automation (ICRA) 2016

GNG-U2 is an enhanced version of GNG-U (Fritzke 1997) with the following
key differences:
    1. Utility criterion is checked at κ-interval (not λ-interval)
    2. Uses Euclidean distance (not squared) for error and utility
    3. Separate decay rate χ for utility (can differ from β)
    4. More frequent utility checks enable faster adaptation to
       non-stationary distributions

This implementation serves as the base for AiS-GNG (Shoji et al. 2023).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GNGU2Params:
    """GNG-U2 hyperparameters.

    Parameters follow the naming from Toda et al. (2016) paper.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node (η1 in paper).
        eps_n: Learning rate for neighbor nodes (η2 in paper).
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        chi: Utility decay rate (χ in paper, can differ from β).
        max_age: Maximum edge age before removal (g_max in paper).
        utility_k: Utility threshold for node removal (k in paper).
        kappa: Utility check interval (κ in paper, typically 10).
    """

    max_nodes: int = 100
    lambda_: int = 300  # Paper uses 300
    eps_b: float = 0.08  # η1 in paper
    eps_n: float = 0.008  # η2 in paper
    alpha: float = 0.5
    beta: float = 0.005
    chi: float = 0.005  # Utility decay rate
    max_age: int = 88  # g_max in paper
    utility_k: float = 1000.0  # k in paper
    kappa: int = 10  # Utility check interval


@dataclass
class NeuronNode:
    """A neuron node in the GNG-U2 network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector (reference vector h).
        error: Accumulated error (E).
        utility: Utility measure (U).
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0
    utility: float = 0.0


class GrowingNeuralGasU2:
    """Growing Neural Gas with Utility - Variant 2 (GNG-U2) implementation.

    GNG-U2 extends GNG-U (Fritzke 1997) with more frequent utility checks
    at κ-interval and uses Euclidean distance for error/utility calculations.

    Key differences from GNG-U:
    1. Utility criterion checked every κ iterations (not just at λ-interval)
    2. Uses Euclidean distance (||x - w||) instead of squared distance
    3. Separate decay rate χ for utility

    This enables faster adaptation to non-stationary distributions and
    serves as the base algorithm for AiS-GNG.

    Attributes:
        params: GNG-U2 hyperparameters.
        nodes: List of neuron nodes.
        edges: Edge age matrix.
        edges_per_node: Adjacency list for quick neighbor lookup.
        n_learning: Total number of learning iterations.
        n_removals: Number of nodes removed by utility criterion.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GrowingNeuralGasU2, GNGU2Params
    >>> X = np.random.rand(1000, 2)
    >>> params = GNGU2Params(kappa=10, utility_k=1000.0)
    >>> gng = GrowingNeuralGasU2(n_dim=2, params=params)
    >>> gng.train(X, n_iterations=5000)
    >>> nodes, edges = gng.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GNGU2Params | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG-U2.

        Args:
            n_dim: Dimension of input data.
            params: GNG-U2 hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or GNGU2Params()
        self.rng = np.random.default_rng(seed)

        # Node management (fixed-size array)
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
        self.n_removals = 0  # Track utility-based removals

        # Initialize with 2 random nodes
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes."""
        for _ in range(2):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            self._add_node(weight)

    def _add_node(
        self, weight: np.ndarray, error: float = 1.0, utility: float = 0.0
    ) -> int:
        """Add a new node with given weight.

        Args:
            weight: Position vector for the new node.
            error: Initial error value.
            utility: Initial utility value.

        Returns:
            ID of the new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = NeuronNode(
            id=node_id, weight=weight.copy(), error=error, utility=utility
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
            Distances are Euclidean (not squared) per GNG-U2 algorithm.
        """
        min_dist1_sq = float("inf")
        min_dist2_sq = float("inf")
        s1_id = -1
        s2_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            dist_sq = np.sum((x - node.weight) ** 2)

            if dist_sq < min_dist1_sq:
                min_dist2_sq = min_dist1_sq
                s2_id = s1_id
                min_dist1_sq = dist_sq
                s1_id = node.id
            elif dist_sq < min_dist2_sq:
                min_dist2_sq = dist_sq
                s2_id = node.id

        # Return Euclidean distances (sqrt)
        dist1 = np.sqrt(min_dist1_sq) if min_dist1_sq < float("inf") else float("inf")
        dist2 = np.sqrt(min_dist2_sq) if min_dist2_sq < float("inf") else float("inf")

        return s1_id, s2_id, dist1, dist2

    def _check_utility_criterion(self) -> None:
        """Check and remove node with lowest utility if criterion met.

        GNG-U2 Algorithm (Toda et al. 2016):
        Every κ iterations, check if E_u / U_l > k, where:
        - u = argmax(E_i) - node with max error
        - l = argmin(U_i) - node with min utility

        Remove node l if criterion is met.
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
        """Single training iteration with GNG-U2 algorithm.

        Key differences from GNG-U:
        1. Uses Euclidean distance for error and utility
        2. Utility check at κ-interval (not λ-interval)
        3. Separate decay rate χ for utility

        Args:
            sample: Input sample vector (v_t).
        """
        p = self.params

        # Find two nearest nodes (with Euclidean distances)
        s1_id, s2_id, dist1, dist2 = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        # Connect s1 and s2
        self._add_edge(s1_id, s2_id)

        # Update error using Euclidean distance (not squared)
        self.nodes[s1_id].error += dist1

        # Update utility using Euclidean distance difference
        self.nodes[s1_id].utility += dist2 - dist1

        # Move winner toward sample
        self.nodes[s1_id].weight += p.eps_b * (sample - self.nodes[s1_id].weight)

        # Update neighbors and age edges
        edges_to_remove = []
        for neighbor_id in list(self.edges_per_node[s1_id]):
            # Increment edge age
            self.edges[s1_id, neighbor_id] += 1
            self.edges[neighbor_id, s1_id] += 1

            if self.edges[s1_id, neighbor_id] > p.max_age:
                # Edge too old
                edges_to_remove.append(neighbor_id)
            else:
                # Move neighbor toward sample
                self.nodes[neighbor_id].weight += p.eps_n * (
                    sample - self.nodes[neighbor_id].weight
                )

        # Remove old edges and isolated nodes
        for neighbor_id in edges_to_remove:
            self._remove_edge(s1_id, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # GNG-U2: Check utility criterion every κ iterations
        if self.n_learning > 0 and self.n_learning % p.kappa == 0:
            self._check_utility_criterion()

        # Decay all errors and utilities
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error
            node.utility -= p.chi * node.utility  # Separate χ for utility

        # Periodically add new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            self._insert_node()

        self.n_learning += 1

    def _insert_node(self) -> int:
        """Insert new node via standard GNG method (based on accumulated error).

        Returns:
            ID of new node, or -1 if insertion failed.
        """
        p = self.params

        if not self._addable_indices:
            return -1

        # Find node with maximum error
        max_err_u = 0.0
        u_id = -1
        for node in self.nodes:
            if node.id == -1:
                continue
            if node.error > max_err_u:
                max_err_u = node.error
                u_id = node.id

        if u_id == -1:
            return -1

        # Find neighbor of u with maximum error
        max_err_f = 0.0
        f_id = -1
        for neighbor_id in self.edges_per_node.get(u_id, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f_id = neighbor_id

        if f_id == -1:
            return -1

        # Add new node between u and f
        new_weight = (self.nodes[u_id].weight + self.nodes[f_id].weight) * 0.5

        # Decay errors
        self.nodes[u_id].error *= 1 - p.alpha
        self.nodes[f_id].error *= 1 - p.alpha

        # New node error and utility
        new_error = 0.5 * (self.nodes[u_id].error + self.nodes[f_id].error)
        new_utility = 0.5 * (self.nodes[u_id].utility + self.nodes[f_id].utility)

        new_id = self._add_node(new_weight, error=new_error, utility=new_utility)

        if new_id == -1:
            return -1

        # Update edges
        self._remove_edge(u_id, f_id)
        self._add_edge(u_id, new_id)
        self._add_edge(f_id, new_id)

        return new_id

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GrowingNeuralGasU2, int], None] | None = None,
    ) -> GrowingNeuralGasU2:
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

    def partial_fit(self, sample: np.ndarray) -> GrowingNeuralGasU2:
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
GNGU2 = GrowingNeuralGasU2
