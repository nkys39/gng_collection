"""Growing Neural Gas (GNG) - Kubota paper compliant version.

Based on:
    - Kubota, N. & Satomi, M. (2008). "自己増殖型ニューラルネットワークと教師無し分類学習"
    - Section 2.4: GNG algorithm description

Key difference from demogng version:
    - Node insertion selects neighbor f by LONGEST EDGE (not max error)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GNGKubotaParams:
    """GNG hyperparameters (Kubota version).

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node (η1).
        eps_n: Learning rate for neighbor nodes (η2).
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        max_age: Maximum edge age before removal (αmax).
    """

    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.08
    eps_n: float = 0.008
    alpha: float = 0.5
    beta: float = 0.005
    max_age: int = 100


@dataclass
class NeuronNode:
    """A neuron node in the GNG network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector.
        error: Accumulated error.
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0


class GrowingNeuralGasKubota:
    """Growing Neural Gas (GNG) - Kubota paper compliant.

    This implementation follows Kubota & Satomi (2008) Section 2.4 exactly:
    - Step 8.ii: Select neighbor f connected by the LONGEST EDGE from q

    Attributes:
        params: GNG hyperparameters.
        nodes: List of neuron nodes.
        edges: Edge age matrix (0 = no edge, >=1 = connected with age).
        edges_per_node: Adjacency list for quick neighbor lookup.
        n_learning: Total number of learning iterations.
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GNGKubotaParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG.

        Args:
            n_dim: Dimension of input data.
            params: GNG hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or GNGKubotaParams()
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

        # Step 0: Initialize with 2 random nodes
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes (Step 0)."""
        for _ in range(2):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            self._add_node(weight)

    def _add_node(self, weight: np.ndarray) -> int:
        """Add a new node with given weight."""
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = NeuronNode(id=node_id, weight=weight.copy(), error=1.0)
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node (only if isolated)."""
        if self.edges_per_node.get(node_id):
            return  # Has edges, don't remove

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _add_edge(self, node1: int, node2: int) -> None:
        """Add or reset edge between two nodes (Step 5)."""
        if self.edges[node1, node2] > 0:
            # Edge exists, reset age to 0 (paper says reset to 0)
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1
        else:
            # New edge
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

    def _find_two_nearest(self, x: np.ndarray) -> tuple[int, int]:
        """Find the two nearest nodes to input x (Step 2)."""
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

        return s1_id, s2_id

    def _edge_length_sq(self, n1: int, n2: int) -> float:
        """Calculate squared distance between two nodes."""
        return float(np.sum(
            (self.nodes[n1].weight - self.nodes[n2].weight) ** 2
        ))

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration following Kubota paper exactly."""
        p = self.params

        # Step 9: Decay all errors
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error

        # Step 2: Find two nearest nodes
        s1_id, s2_id = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        # Step 3: Update winner error (squared distance)
        dist_sq = np.sum((sample - self.nodes[s1_id].weight) ** 2)
        self.nodes[s1_id].error += dist_sq

        # Step 4: Move winner toward sample
        self.nodes[s1_id].weight += p.eps_b * (sample - self.nodes[s1_id].weight)

        # Step 4: Move neighbors toward sample
        for neighbor_id in list(self.edges_per_node[s1_id]):
            self.nodes[neighbor_id].weight += p.eps_n * (
                sample - self.nodes[neighbor_id].weight
            )

        # Step 5: Connect s1 and s2 (reset age)
        self._add_edge(s1_id, s2_id)

        # Step 6: Increment age of all edges connected to s1
        for neighbor_id in list(self.edges_per_node[s1_id]):
            self.edges[s1_id, neighbor_id] += 1
            self.edges[neighbor_id, s1_id] += 1

        # Step 7: Remove old edges and isolated nodes
        edges_to_remove = []
        for neighbor_id in list(self.edges_per_node[s1_id]):
            if self.edges[s1_id, neighbor_id] > p.max_age:
                edges_to_remove.append(neighbor_id)

        for neighbor_id in edges_to_remove:
            self._remove_edge(s1_id, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # Step 8: Periodically add new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0

            if self._addable_indices:
                # Step 8.i: Find node q with maximum error
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

                # Step 8.ii: Find neighbor f connected by LONGEST EDGE (Kubota paper)
                max_len = -1.0
                f_id = -1
                for neighbor_id in self.edges_per_node.get(q_id, set()):
                    edge_len = self._edge_length_sq(q_id, neighbor_id)
                    if edge_len > max_len:
                        max_len = edge_len
                        f_id = neighbor_id

                if f_id == -1:
                    self.n_learning += 1
                    return

                # Step 8.iii: Add new node r between q and f
                new_weight = (self.nodes[q_id].weight + self.nodes[f_id].weight) * 0.5
                new_id = self._add_node(new_weight)

                if new_id == -1:
                    self.n_learning += 1
                    return

                # Update edges: remove (q, f), add (q, r) and (r, f)
                self._remove_edge(q_id, f_id)
                self._add_edge(q_id, new_id)
                self._add_edge(f_id, new_id)

                # Step 8.iv-v: Update errors
                self.nodes[q_id].error *= (1 - p.alpha)
                self.nodes[f_id].error *= (1 - p.alpha)
                self.nodes[new_id].error = (
                    self.nodes[q_id].error + self.nodes[f_id].error
                ) * 0.5

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GrowingNeuralGasKubota, int], None] | None = None,
    ) -> GrowingNeuralGasKubota:
        """Train on data for multiple iterations."""
        n_samples = len(data)

        for i in range(n_iterations):
            idx = self.rng.integers(0, n_samples)
            self._one_train_update(data[idx])

            if callback is not None:
                callback(self, i)

        return self

    def partial_fit(self, sample: np.ndarray) -> GrowingNeuralGasKubota:
        """Single online learning step."""
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
        """Get current graph structure."""
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

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes."""
        return np.array([node.error for node in self.nodes if node.id != -1])


# Aliases
GNGKubota = GrowingNeuralGasKubota
