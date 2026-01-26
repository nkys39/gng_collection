"""Add-if-Silent Rule-Based Growing Neural Gas (AiS-GNG) implementation.

Based on:
    - Shoji, M., Obo, T., & Kubota, N. (2023).
      "Add-if-Silent Rule-Based Growing Neural Gas for High-Density
       Topological Structure of Unknown Objects"
      IEEE RO-MAN 2023.
    - Shoji, M., Obo, T., & Kubota, N. (2023).
      "Add-if-Silent Rule-Based Growing Neural Gas with Amount of Movement
       for High-Density Topological Structure Generation of Dynamic Object"
      IEEE SMC 2023.

AiS-GNG extends GNG-U with the Add-if-Silent rule, which directly adds
input data as new nodes when certain conditions are met, enabling faster
generation of high-density topological structures.

The key idea is based on the Add-if-Silent rule from neocognitron:
if no neuron responds to useful input, add a new neuron at that position.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class AiSGNGParams:
    """AiS-GNG hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node (η1 in paper).
        eps_n: Learning rate for neighbor nodes (η2 in paper).
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        chi: Utility decay rate (χ in paper, often same as beta).
        max_age: Maximum edge age before removal (AgeMax in paper).
        utility_k: Utility threshold for node removal (k in paper).
        kappa: Utility check interval (κ in paper).
        theta_ais_min: Minimum tolerance for Add-if-Silent rule (θ_AiS^Min).
        theta_ais_max: Maximum tolerance for Add-if-Silent rule (θ_AiS^Max).
    """

    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.08
    eps_n: float = 0.008
    alpha: float = 0.5
    beta: float = 0.005
    chi: float = 0.005  # Utility decay rate
    max_age: int = 88  # Paper uses 88
    utility_k: float = 1000.0  # Paper uses k=1000
    kappa: int = 10  # Utility check interval
    # Add-if-Silent rule tolerances (scaled for 2D [0,1] range)
    # Original paper: theta_ais=0.50 for 3D, theta_min=0.25, theta_max=0.50
    # For 2D [0,1] with ring radius ~0.1: scale down by factor of ~5
    theta_ais_min: float = 0.03  # Minimum distance threshold
    theta_ais_max: float = 0.15  # Maximum distance threshold


@dataclass
class NeuronNode:
    """A neuron node in the AiS-GNG network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector (reference vector h).
        error: Accumulated error (E).
        utility: Utility measure (U) - how useful this node is.
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0
    utility: float = 0.0  # Initialized to 0 per demogng.de


class AiSGNG:
    """Add-if-Silent Rule-Based Growing Neural Gas implementation.

    AiS-GNG extends GNG-U with the Add-if-Silent rule, which enables
    rapid generation of high-density topological structures by directly
    adding input data as new nodes when conditions are met.

    The Add-if-Silent rule (from neocognitron) states:
    If no neuron responds adequately to useful input, add a new neuron
    at that position.

    In AiS-GNG, if the input is within a tolerance area (between theta_min
    and theta_max) from both winner nodes, the input is directly added
    as a new reference vector.

    Attributes:
        params: AiS-GNG hyperparameters.
        nodes: List of neuron nodes.
        edges: Edge age matrix.
        edges_per_node: Adjacency list for quick neighbor lookup.
        n_learning: Total number of learning iterations.
        n_ais_additions: Number of nodes added by Add-if-Silent rule.
        n_utility_removals: Number of nodes removed by utility criterion.

    Examples
    --------
    >>> import numpy as np
    >>> from model import AiSGNG, AiSGNGParams
    >>> X = np.random.rand(1000, 2)
    >>> params = AiSGNGParams(theta_ais_max=0.1)
    >>> gng = AiSGNG(n_dim=2, params=params)
    >>> gng.train(X, n_iterations=5000)
    >>> nodes, edges = gng.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: AiSGNGParams | None = None,
        seed: int | None = None,
    ):
        """Initialize AiS-GNG.

        Args:
            n_dim: Dimension of input data.
            params: AiS-GNG hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or AiSGNGParams()
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
        self.n_ais_additions = 0  # AiS-GNG: count Add-if-Silent additions
        self.n_utility_removals = 0  # Count utility-based removals

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
            Distances are Euclidean (not squared) per paper's algorithm.
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

        Paper Algorithm 1, lines 24-30:
        Every κ iterations, check if E_u / U_l > k, where:
        - u = argmax(E_i) - node with max error
        - l = argmax(U_i) - node with max utility (min utility in some variants)

        Note: The paper's notation is confusing. Looking at the intent:
        Remove the node with lowest utility if the ratio of max_error
        to lowest_utility exceeds threshold k.
        """
        p = self.params

        # Don't remove if only 2 nodes remain
        if self.n_nodes <= 2:
            return

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
                    self.n_utility_removals += 1

    def _ais_growing_process(
        self, sample: np.ndarray, s1_id: int, s2_id: int, dist1: float, dist2: float
    ) -> bool:
        """Add-if-Silent rule-based growing process.

        Paper Algorithm 1, lines 9-15:
        If both winner nodes would benefit from the input (within tolerance area),
        add the input directly as a new reference vector.

        In the original paper, this checks object category labels.
        For 2D visualization, we treat all nodes as "unknown objects".

        The second paper (SMC 2023) uses a range [θ_min, θ_max] for both
        winner nodes, which we follow here.

        Args:
            sample: Input vector (v_t).
            s1_id: First winner node ID.
            s2_id: Second winner node ID.
            dist1: Distance from sample to s1 (||v_t - h_s1||).
            dist2: Distance from sample to s2 (||v_t - h_s2||).

        Returns:
            True if a new node was added via Add-if-Silent rule.
        """
        p = self.params

        # Check if there's space for new node
        if not self._addable_indices:
            return False

        # Add-if-Silent rule conditions (from SMC 2023 paper):
        # θ_AiS_Min < ||v_t - h_s1|| < θ_AiS_Max AND
        # θ_AiS_Min < ||v_t - h_s2|| < θ_AiS_Max
        s1_in_range = p.theta_ais_min < dist1 < p.theta_ais_max
        s2_in_range = p.theta_ais_min < dist2 < p.theta_ais_max

        if s1_in_range and s2_in_range:
            # Add input directly as new node
            # h_r = v_t
            # E_r = 0.5 * (E_s1 + E_s2)
            # U_r = 0.5 * (U_s1 + U_s2)
            new_error = 0.5 * (self.nodes[s1_id].error + self.nodes[s2_id].error)
            new_utility = 0.5 * (self.nodes[s1_id].utility + self.nodes[s2_id].utility)

            new_id = self._add_node(sample, error=new_error, utility=new_utility)

            if new_id == -1:
                return False

            # Connect new node to s1 and s2
            # c_r,s1 = 1, c_r,s2 = 1
            self._add_edge(new_id, s1_id)
            self._add_edge(new_id, s2_id)

            self.n_ais_additions += 1
            return True

        return False

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration with Add-if-Silent rule.

        Follows Algorithm 1 from the RO-MAN 2023 paper.

        Args:
            sample: Input sample vector (v_t).
        """
        p = self.params

        # Find two nearest nodes (with Euclidean distances)
        s1_id, s2_id, dist1, dist2 = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        # Connect s1 and s2 if not already connected (lines 6-8)
        self._add_edge(s1_id, s2_id)

        # Add-if-Silent rule-based growing process (lines 9-15)
        self._ais_growing_process(sample, s1_id, s2_id, dist1, dist2)

        # Update error (line 16): E_s1 += ||v_t - h_s1||
        # Note: Paper uses non-squared distance
        self.nodes[s1_id].error += dist1

        # Update utility (line 16): U_s1 += ||v_t - h_s2|| - ||v_t - h_s1||
        self.nodes[s1_id].utility += dist2 - dist1

        # Update winner reference vector (line 17): h_s1 += η1 * (v_t - h_s1)
        self.nodes[s1_id].weight += p.eps_b * (sample - self.nodes[s1_id].weight)

        # Update neighbors and age edges (lines 17-23)
        edges_to_remove = []
        for neighbor_id in list(self.edges_per_node[s1_id]):
            # Increment edge age (line 18)
            self.edges[s1_id, neighbor_id] += 1
            self.edges[neighbor_id, s1_id] += 1

            if self.edges[s1_id, neighbor_id] > p.max_age:
                # Edge too old (lines 19-23)
                edges_to_remove.append(neighbor_id)
            else:
                # Move neighbor toward sample (line 17): h_j += η2 * (v_t - h_j)
                self.nodes[neighbor_id].weight += p.eps_n * (
                    sample - self.nodes[neighbor_id].weight
                )

        # Remove old edges and isolated nodes
        for neighbor_id in edges_to_remove:
            self._remove_edge(s1_id, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # Check utility criterion every κ iterations (lines 24-30)
        if self.n_learning > 0 and self.n_learning % p.kappa == 0:
            self._check_utility_criterion()

        # Decay all errors and utilities (lines 31-32)
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error  # E_i -= β * E_i
            node.utility -= p.chi * node.utility  # U_i -= χ * U_i

        # Periodically add new node (lines 35-41)
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            self._insert_node_standard()

        self.n_learning += 1

    def _insert_node_standard(self) -> int:
        """Insert new node via standard GNG method (based on accumulated error).

        Paper Algorithm 1, lines 35-41:
        Add a new node between the highest-error node and its highest-error neighbor.

        Returns:
            ID of new node, or -1 if insertion failed.
        """
        p = self.params

        if not self._addable_indices:
            return -1

        # Find node with maximum error (line 37): u = argmax(E_i)
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

        # Find neighbor of u with maximum error (line 37): f = argmax_{c_i,u=1}(E_i)
        max_err_f = 0.0
        f_id = -1
        for neighbor_id in self.edges_per_node.get(u_id, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f_id = neighbor_id

        if f_id == -1:
            return -1

        # Add new node between u and f (line 38): h_r = 0.5 * (h_u + h_f)
        new_weight = (self.nodes[u_id].weight + self.nodes[f_id].weight) * 0.5

        # Decay errors (line 39): E_u -= α*E_u, E_f -= α*E_f
        self.nodes[u_id].error *= 1 - p.alpha
        self.nodes[f_id].error *= 1 - p.alpha

        # New node error (line 40): E_r = 0.5 * (E_u + E_f)
        new_error = 0.5 * (self.nodes[u_id].error + self.nodes[f_id].error)
        # New node utility: average of u and f
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
        callback: Callable[[AiSGNG, int], None] | None = None,
    ) -> AiSGNG:
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

    def partial_fit(self, sample: np.ndarray) -> AiSGNG:
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

    @property
    def n_removals(self) -> int:
        """Total number of removed nodes (for compatibility)."""
        return self.n_utility_removals

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
AddIfSilentGNG = AiSGNG
