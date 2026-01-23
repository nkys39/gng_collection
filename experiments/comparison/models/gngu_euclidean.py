"""GNG-U with Euclidean distance for comparison experiments.

This variant uses Euclidean distance (not squared) for error and utility
calculations, matching GNG-U2's distance metric.

Changes from original GNG-U:
- Error: E += ||x - w|| (was ||x - w||²)
- Utility: U += ||x - w2|| - ||x - w1|| (was squared)
- Added kappa parameter for utility check interval
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GNGUEuclideanParams:
    """GNG-U (Euclidean) hyperparameters."""

    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.08
    eps_n: float = 0.008
    alpha: float = 0.5
    beta: float = 0.005
    chi: float = 0.005  # Utility decay rate (separate from beta)
    max_age: int = 100
    utility_k: float = 1000.0  # Adjusted for Euclidean distance scale
    kappa: int = 10  # Utility check interval (like GNG-U2)


@dataclass
class NeuronNode:
    """A neuron node."""

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0
    utility: float = 0.0


class GNGUEuclidean:
    """GNG-U with Euclidean distance metric.

    Uses Euclidean distance for error/utility calculations to match GNG-U2.
    Also uses kappa-interval utility checks.
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GNGUEuclideanParams | None = None,
        seed: int | None = None,
    ):
        self.n_dim = n_dim
        self.params = params or GNGUEuclideanParams()
        self.rng = np.random.default_rng(seed)

        self.nodes: list[NeuronNode] = [
            NeuronNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        self.edges = np.zeros(
            (self.params.max_nodes, self.params.max_nodes), dtype=np.int32
        )
        self.edges_per_node: dict[int, set[int]] = {}

        self.n_learning = 0
        self._n_trial = 0
        self.n_removals = 0

        self._init_nodes()

    def _init_nodes(self) -> None:
        for _ in range(2):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            self._add_node(weight)

    def _add_node(self, weight: np.ndarray, error: float = 1.0, utility: float = 0.0) -> int:
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = NeuronNode(
            id=node_id, weight=weight.copy(), error=error, utility=utility
        )
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int, force: bool = False) -> bool:
        if not force and self.edges_per_node.get(node_id):
            return False

        for neighbor_id in list(self.edges_per_node.get(node_id, set())):
            self._remove_edge(node_id, neighbor_id)

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)
        return True

    def _add_edge(self, node1: int, node2: int) -> None:
        if self.edges[node1, node2] > 0:
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1
        else:
            self.edges_per_node[node1].add(node2)
            self.edges_per_node[node2].add(node1)
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1

    def _remove_edge(self, node1: int, node2: int) -> None:
        self.edges_per_node[node1].discard(node2)
        self.edges_per_node[node2].discard(node1)
        self.edges[node1, node2] = 0
        self.edges[node2, node1] = 0

    def _find_two_nearest(self, x: np.ndarray) -> tuple[int, int, float, float]:
        """Find two nearest nodes, returning EUCLIDEAN distances."""
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
        """Check and remove node with lowest utility if criterion met."""
        p = self.params

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

        if min_utility_id != -1 and min_utility > 0:
            if max_error / min_utility > p.utility_k:
                if self._remove_node(min_utility_id, force=True):
                    self.n_removals += 1

    def _one_train_update(self, sample: np.ndarray) -> None:
        p = self.params

        # Find two nearest nodes (with EUCLIDEAN distances)
        s1_id, s2_id, dist1, dist2 = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        # Connect s1 and s2
        self._add_edge(s1_id, s2_id)

        # Update error using EUCLIDEAN distance
        self.nodes[s1_id].error += dist1

        # Update utility using EUCLIDEAN distance difference
        self.nodes[s1_id].utility += dist2 - dist1

        # Move winner toward sample
        self.nodes[s1_id].weight += p.eps_b * (sample - self.nodes[s1_id].weight)

        # Update neighbors and age edges
        edges_to_remove = []
        for neighbor_id in list(self.edges_per_node[s1_id]):
            self.edges[s1_id, neighbor_id] += 1
            self.edges[neighbor_id, s1_id] += 1

            if self.edges[s1_id, neighbor_id] > p.max_age:
                edges_to_remove.append(neighbor_id)
            else:
                self.nodes[neighbor_id].weight += p.eps_n * (
                    sample - self.nodes[neighbor_id].weight
                )

        for neighbor_id in edges_to_remove:
            self._remove_edge(s1_id, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # kappa-interval utility check (like GNG-U2)
        if self.n_learning > 0 and self.n_learning % p.kappa == 0:
            self._check_utility_criterion()

        # Decay all errors and utilities
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error
            node.utility -= p.chi * node.utility

        # Periodically add new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            self._insert_node()

        self.n_learning += 1

    def _insert_node(self) -> int:
        p = self.params

        if not self._addable_indices:
            return -1

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

        max_err_f = 0.0
        f_id = -1
        for neighbor_id in self.edges_per_node.get(q_id, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f_id = neighbor_id

        if f_id == -1:
            return -1

        new_weight = (self.nodes[q_id].weight + self.nodes[f_id].weight) * 0.5

        self.nodes[q_id].error *= 1 - p.alpha
        self.nodes[f_id].error *= 1 - p.alpha

        new_error = 0.5 * (self.nodes[q_id].error + self.nodes[f_id].error)
        new_utility = 0.5 * (self.nodes[q_id].utility + self.nodes[f_id].utility)

        new_id = self._add_node(new_weight, error=new_error, utility=new_utility)

        if new_id == -1:
            return -1

        self._remove_edge(q_id, f_id)
        self._add_edge(q_id, new_id)
        self._add_edge(f_id, new_id)

        return new_id

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GNGUEuclidean, int], None] | None = None,
    ) -> GNGUEuclidean:
        n_samples = len(data)

        for i in range(n_iterations):
            idx = self.rng.integers(0, n_samples)
            self._one_train_update(data[idx])

            if callback is not None:
                callback(self, i)

        return self

    def partial_fit(self, sample: np.ndarray) -> GNGUEuclidean:
        self._one_train_update(sample)
        return self

    @property
    def n_nodes(self) -> int:
        return sum(1 for node in self.nodes if node.id != -1)

    @property
    def n_edges(self) -> int:
        count = 0
        for node_id, neighbors in self.edges_per_node.items():
            if self.nodes[node_id].id != -1:
                count += len(neighbors)
        return count // 2

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
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
        return np.array([node.utility for node in self.nodes if node.id != -1])

    def get_node_errors(self) -> np.ndarray:
        return np.array([node.error for node in self.nodes if node.id != -1])
