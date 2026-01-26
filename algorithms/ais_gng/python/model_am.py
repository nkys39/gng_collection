"""AiS-GNG-AM (SMC 2023) implementation with Amount of Movement.

Based on:
    Shoji, M., Obo, T., & Kubota, N. (2023).
    "Add-if-Silent Rule-Based Growing Neural Gas with Amount of Movement
     for High-Density Topological Structure Generation of Dynamic Object"
    IEEE SMC 2023, pp. 3040-3047.

This extends the RO-MAN 2023 version with:
1. Range thresholds [θ_min, θ_max] instead of single θ_AiS
2. Amount of Movement (AM) tracking for dynamic object detection

The AM feature tracks how much each node moves over time, enabling
detection and segmentation of dynamic objects in the environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class AiSGNGAMParams:
    """AiS-GNG-AM (SMC 2023) hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node (η1 in paper).
        eps_n: Learning rate for neighbor nodes (η2 in paper).
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        chi: Utility decay rate (χ in paper).
        max_age: Maximum edge age before removal (AgeMax in paper).
        utility_k: Utility threshold for node removal (k in paper).
        kappa: Utility check interval (κ in paper).
        theta_ais_min: Minimum tolerance for Add-if-Silent rule (θ_AiS^Min).
        theta_ais_max: Maximum tolerance for Add-if-Silent rule (θ_AiS^Max).
        am_decay: Decay rate for Amount of Movement (γ_AM in paper).
        am_threshold: Threshold to classify node as "moving" (θ_AM).
    """

    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.08
    eps_n: float = 0.008
    alpha: float = 0.5
    beta: float = 0.005
    chi: float = 0.005
    max_age: int = 88
    utility_k: float = 1000.0
    kappa: int = 10
    # SMC 2023: Range thresholds
    theta_ais_min: float = 0.03
    theta_ais_max: float = 0.15
    # Amount of Movement parameters
    am_decay: float = 0.95  # Decay rate for AM (γ_AM)
    am_threshold: float = 0.01  # Threshold for "moving" classification (θ_AM)


@dataclass
class NeuronNodeAM:
    """A neuron node with Amount of Movement tracking.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector (reference vector h).
        prev_weight: Previous position (for AM calculation).
        error: Accumulated error (E).
        utility: Utility measure (U).
        amount_of_movement: Accumulated movement amount (AM).
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    prev_weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0
    utility: float = 0.0
    amount_of_movement: float = 0.0


class AiSGNGAM:
    """AiS-GNG-AM (SMC 2023) with Amount of Movement tracking.

    This is the full SMC 2023 implementation that extends AiS-GNG with:
    1. Range thresholds [θ_min, θ_max] for Add-if-Silent rule
    2. Amount of Movement (AM) tracking for each node

    The AM feature enables detection of dynamic objects by tracking
    how much each node moves over time. Nodes with high AM values
    are likely part of moving objects.

    Attributes:
        params: AiS-GNG-AM hyperparameters.
        nodes: List of neuron nodes with AM tracking.
        n_ais_additions: Number of nodes added by Add-if-Silent rule.
        n_utility_removals: Number of nodes removed by utility criterion.

    Examples
    --------
    >>> import numpy as np
    >>> from model_am import AiSGNGAM, AiSGNGAMParams
    >>> X = np.random.rand(1000, 2)
    >>> params = AiSGNGAMParams(am_threshold=0.02)
    >>> gng = AiSGNGAM(n_dim=2, params=params)
    >>> gng.train(X, n_iterations=5000)
    >>> nodes, edges = gng.get_graph()
    >>> moving_mask = gng.get_moving_nodes_mask()  # Dynamic object detection
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: AiSGNGAMParams | None = None,
        seed: int | None = None,
    ):
        """Initialize AiS-GNG-AM.

        Args:
            n_dim: Dimension of input data.
            params: Hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or AiSGNGAMParams()
        self.rng = np.random.default_rng(seed)

        # Node management
        self.nodes: list[NeuronNodeAM] = [
            NeuronNodeAM() for _ in range(self.params.max_nodes)
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
        self.n_ais_additions = 0
        self.n_utility_removals = 0

        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes."""
        for _ in range(2):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            self._add_node(weight)

    def _add_node(
        self,
        weight: np.ndarray,
        error: float = 1.0,
        utility: float = 0.0,
        amount_of_movement: float = 0.0,
    ) -> int:
        """Add a new node with given weight."""
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = NeuronNodeAM(
            id=node_id,
            weight=weight.copy(),
            prev_weight=weight.copy(),  # Initialize prev_weight same as weight
            error=error,
            utility=utility,
            amount_of_movement=amount_of_movement,
        )
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int, force: bool = False) -> bool:
        """Remove a node."""
        if not force and self.edges_per_node.get(node_id):
            return False

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

        Returns:
            Tuple of (winner_id, second_winner_id, dist_winner, dist_second).
            Distances are Euclidean (not squared).
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

        dist1 = np.sqrt(min_dist1_sq) if min_dist1_sq < float("inf") else float("inf")
        dist2 = np.sqrt(min_dist2_sq) if min_dist2_sq < float("inf") else float("inf")

        return s1_id, s2_id, dist1, dist2

    def _check_utility_criterion(self) -> None:
        """Check and remove node with lowest utility if criterion met."""
        p = self.params

        if self.n_nodes <= 2:
            return

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
                    self.n_utility_removals += 1

    def _ais_growing_process(
        self, sample: np.ndarray, s1_id: int, s2_id: int, dist1: float, dist2: float
    ) -> bool:
        """Add-if-Silent rule (SMC 2023 version with range thresholds).

        Condition:
            θ_min < ||v_t - h_s1|| < θ_max AND θ_min < ||v_t - h_s2|| < θ_max

        Args:
            sample: Input vector (v_t).
            s1_id: First winner node ID.
            s2_id: Second winner node ID.
            dist1: Distance from sample to s1.
            dist2: Distance from sample to s2.

        Returns:
            True if a new node was added.
        """
        p = self.params

        if not self._addable_indices:
            return False

        # SMC 2023: Range threshold condition
        s1_in_range = p.theta_ais_min < dist1 < p.theta_ais_max
        s2_in_range = p.theta_ais_min < dist2 < p.theta_ais_max

        if s1_in_range and s2_in_range:
            new_error = 0.5 * (self.nodes[s1_id].error + self.nodes[s2_id].error)
            new_utility = 0.5 * (self.nodes[s1_id].utility + self.nodes[s2_id].utility)
            # New node inherits average AM from parents
            new_am = 0.5 * (
                self.nodes[s1_id].amount_of_movement
                + self.nodes[s2_id].amount_of_movement
            )

            new_id = self._add_node(
                sample, error=new_error, utility=new_utility, amount_of_movement=new_am
            )

            if new_id == -1:
                return False

            self._add_edge(new_id, s1_id)
            self._add_edge(new_id, s2_id)

            self.n_ais_additions += 1
            return True

        return False

    def _update_amount_of_movement(self) -> None:
        """Update Amount of Movement for all nodes.

        SMC 2023 Algorithm:
            AM_i(t) = γ_AM * AM_i(t-1) + ||h_i(t) - h_i(t-1)||

        This accumulates movement with exponential decay, allowing
        detection of nodes that have been moving consistently.
        """
        p = self.params

        for node in self.nodes:
            if node.id == -1:
                continue

            # Calculate movement since last update
            movement = np.sqrt(np.sum((node.weight - node.prev_weight) ** 2))

            # Update AM with decay: AM = γ * AM_prev + movement
            node.amount_of_movement = p.am_decay * node.amount_of_movement + movement

            # Store current position for next update
            node.prev_weight = node.weight.copy()

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration with AM tracking."""
        p = self.params

        s1_id, s2_id, dist1, dist2 = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        self._add_edge(s1_id, s2_id)

        self._ais_growing_process(sample, s1_id, s2_id, dist1, dist2)

        self.nodes[s1_id].error += dist1
        self.nodes[s1_id].utility += dist2 - dist1

        self.nodes[s1_id].weight += p.eps_b * (sample - self.nodes[s1_id].weight)

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

        if self.n_learning > 0 and self.n_learning % p.kappa == 0:
            self._check_utility_criterion()

        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error
            node.utility -= p.chi * node.utility

        # Update Amount of Movement for all nodes
        self._update_amount_of_movement()

        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            self._insert_node_standard()

        self.n_learning += 1

    def _insert_node_standard(self) -> int:
        """Insert new node via standard GNG method."""
        p = self.params

        if not self._addable_indices:
            return -1

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

        max_err_f = 0.0
        f_id = -1
        for neighbor_id in self.edges_per_node.get(u_id, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f_id = neighbor_id

        if f_id == -1:
            return -1

        new_weight = (self.nodes[u_id].weight + self.nodes[f_id].weight) * 0.5

        self.nodes[u_id].error *= 1 - p.alpha
        self.nodes[f_id].error *= 1 - p.alpha

        new_error = 0.5 * (self.nodes[u_id].error + self.nodes[f_id].error)
        new_utility = 0.5 * (self.nodes[u_id].utility + self.nodes[f_id].utility)
        new_am = 0.5 * (
            self.nodes[u_id].amount_of_movement
            + self.nodes[f_id].amount_of_movement
        )

        new_id = self._add_node(
            new_weight, error=new_error, utility=new_utility, amount_of_movement=new_am
        )

        if new_id == -1:
            return -1

        self._remove_edge(u_id, f_id)
        self._add_edge(u_id, new_id)
        self._add_edge(f_id, new_id)

        return new_id

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[AiSGNGAM, int], None] | None = None,
    ) -> AiSGNGAM:
        """Train on data for multiple iterations."""
        n_samples = len(data)

        for i in range(n_iterations):
            idx = self.rng.integers(0, n_samples)
            self._one_train_update(data[idx])

            if callback is not None:
                callback(self, i)

        return self

    def partial_fit(self, sample: np.ndarray) -> AiSGNGAM:
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

    @property
    def n_removals(self) -> int:
        """Total number of removed nodes."""
        return self.n_utility_removals

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

    def get_node_utilities(self) -> np.ndarray:
        """Get utility values for active nodes."""
        return np.array([node.utility for node in self.nodes if node.id != -1])

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes."""
        return np.array([node.error for node in self.nodes if node.id != -1])

    def get_node_movements(self) -> np.ndarray:
        """Get Amount of Movement values for active nodes.

        Returns:
            Array of AM values in same order as get_graph() nodes.
        """
        return np.array(
            [node.amount_of_movement for node in self.nodes if node.id != -1]
        )

    def get_moving_nodes_mask(self) -> np.ndarray:
        """Get boolean mask indicating which nodes are "moving".

        A node is classified as moving if its AM exceeds am_threshold.

        Returns:
            Boolean array where True indicates a moving node.
        """
        movements = self.get_node_movements()
        return movements > self.params.am_threshold

    def get_moving_nodes_indices(self) -> list[int]:
        """Get indices of moving nodes.

        Returns:
            List of indices (in get_graph() order) of moving nodes.
        """
        mask = self.get_moving_nodes_mask()
        return list(np.where(mask)[0])

    def get_static_nodes_indices(self) -> list[int]:
        """Get indices of static (non-moving) nodes.

        Returns:
            List of indices (in get_graph() order) of static nodes.
        """
        mask = self.get_moving_nodes_mask()
        return list(np.where(~mask)[0])

    def segment_by_movement(
        self,
    ) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, list[tuple[int, int]]]:
        """Segment graph into moving and static parts.

        Returns:
            Tuple of (moving_nodes, moving_edges, static_nodes, static_edges).
        """
        nodes, edges = self.get_graph()
        mask = self.get_moving_nodes_mask()

        # Create index mappings
        moving_indices = np.where(mask)[0]
        static_indices = np.where(~mask)[0]

        moving_map = {old: new for new, old in enumerate(moving_indices)}
        static_map = {old: new for new, old in enumerate(static_indices)}

        # Extract nodes
        moving_nodes = nodes[moving_indices] if len(moving_indices) > 0 else np.array(
            []
        ).reshape(0, self.n_dim)
        static_nodes = nodes[static_indices] if len(static_indices) > 0 else np.array(
            []
        ).reshape(0, self.n_dim)

        # Extract edges (only edges where both nodes are in same segment)
        moving_edges = []
        static_edges = []

        for i, j in edges:
            if mask[i] and mask[j]:
                moving_edges.append((moving_map[i], moving_map[j]))
            elif not mask[i] and not mask[j]:
                static_edges.append((static_map[i], static_map[j]))
            # Edges between moving and static nodes are discarded

        return moving_nodes, moving_edges, static_nodes, static_edges


# Alias
AiSGNGWithMovement = AiSGNGAM
