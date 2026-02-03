"""AiS-GNG-DT (Add-if-Silent GNG with Different Topologies) implementation.

Combines:
    - GNG-DT: Multiple independent edge topologies (position, color, normal)
    - AiS-GNG: Add-if-Silent rule for rapid high-density structure generation

This experimental algorithm aims to:
    1. Learn multiple topologies for 3D point clouds (from GNG-DT)
    2. Generate high-density structures quickly (from AiS-GNG)
    3. Handle non-stationary distributions via utility-based node removal

Key features:
    - Add-if-Silent rule: Directly add input as node when conditions met
    - Utility tracking: Remove low-utility nodes to adapt to changes
    - Multiple topologies: Position/color/normal edges learned independently
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class AiSGNGDTParams:
    """AiS-GNG-DT hyperparameters.

    Combines GNG-DT and AiS-GNG parameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval.
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for neighbor nodes.
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        max_age: Maximum edge age before removal.

        # GNG-DT specific
        tau_color: Color similarity threshold.
        tau_normal: Normal similarity threshold (dot product).
        dis_thv: Distance threshold for adding new nodes far from network.

        # AiS-GNG specific
        theta_ais_min: Minimum distance for Add-if-Silent rule.
        theta_ais_max: Maximum distance for Add-if-Silent rule.
        kappa: Utility check interval.
        utility_k: Utility threshold for node removal.
        chi: Utility decay rate.
    """

    max_nodes: int = 150
    lambda_: int = 100
    eps_b: float = 0.05
    eps_n: float = 0.005
    alpha: float = 0.5
    beta: float = 0.005
    max_age: int = 88

    # GNG-DT specific
    tau_color: float = 0.05
    tau_normal: float = 0.998
    dis_thv: float = 0.5

    # AiS-GNG specific
    theta_ais_min: float = 0.05  # Minimum distance for AiS (3D scale)
    theta_ais_max: float = 0.15  # Maximum distance for AiS (3D scale)
    kappa: int = 10  # Utility check interval
    utility_k: float = 1000.0  # Utility removal threshold
    chi: float = 0.005  # Utility decay rate


@dataclass
class AiSGNGDTNode:
    """A neuron node in the AiS-GNG-DT network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        position: 3D position vector.
        color: RGB color vector.
        normal: Normal vector (unit vector, computed via PCA).
        error: Accumulated error.
        utility: Utility value for adaptive node removal.
    """

    id: int = -1
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    error: float = 0.0
    utility: float = 0.0


class AiSGNGDT:
    """AiS-GNG-DT algorithm implementation.

    Combines GNG-DT's multiple topology learning with AiS-GNG's
    Add-if-Silent rule and utility-based node management.

    Attributes:
        params: Hyperparameters.
        nodes: List of neuron nodes.
        edges_pos: Position-based edge age matrix.
        edges_color: Color-based edge connectivity matrix.
        edges_normal: Normal-based edge connectivity matrix.
        n_learning: Total number of learning iterations.
        n_ais_additions: Number of nodes added by Add-if-Silent rule.
        n_utility_removals: Number of nodes removed by utility criterion.
    """

    def __init__(
        self,
        params: AiSGNGDTParams | None = None,
        seed: int | None = None,
    ):
        """Initialize AiS-GNG-DT.

        Args:
            params: Hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.params = params or AiSGNGDTParams()
        self.rng = np.random.default_rng(seed)

        max_n = self.params.max_nodes

        # Node management
        self.nodes: list[AiSGNGDTNode] = [AiSGNGDTNode() for _ in range(max_n)]
        self._addable_indices: deque[int] = deque(range(max_n))

        # Edge matrices (three topologies)
        self.edges_pos = np.zeros((max_n, max_n), dtype=np.int32)
        self.edges_color = np.zeros((max_n, max_n), dtype=np.int32)
        self.edges_normal = np.zeros((max_n, max_n), dtype=np.int32)
        self.edge_age = np.zeros((max_n, max_n), dtype=np.int32)

        # Adjacency list for position edges
        self.edges_per_node: dict[int, set[int]] = {}

        # Counters
        self.n_learning = 0
        self._n_trial = 0
        self._total_error = 0.0

        # AiS-GNG statistics
        self.n_ais_additions = 0
        self.n_utility_removals = 0

        # Initialize with 2 random nodes
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes connected by edges."""
        for i in range(2):
            pos = self.rng.random(3).astype(np.float64)
            self._add_node(pos)

        # Connect initial 2 nodes
        self.edges_pos[0, 1] = 1
        self.edges_pos[1, 0] = 1
        self.edges_color[0, 1] = 1
        self.edges_color[1, 0] = 1
        self.edges_normal[0, 1] = 1
        self.edges_normal[1, 0] = 1
        self.edges_per_node[0] = {1}
        self.edges_per_node[1] = {0}

    def _add_node(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
        error: float = 0.0,
        utility: float = 0.0,
    ) -> int:
        """Add a new node.

        Args:
            position: 3D position vector.
            color: RGB color vector (optional).
            error: Initial error value.
            utility: Initial utility value.

        Returns:
            ID of the new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = AiSGNGDTNode(
            id=node_id,
            position=position.copy(),
            color=color.copy() if color is not None else np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            error=error,
            utility=utility,
        )
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node with cascading deletion of isolated neighbors."""
        neighbors_to_check = list(self.edges_per_node.get(node_id, set()))

        for other_id in neighbors_to_check:
            self._remove_all_edges(node_id, other_id)

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

        # Cascading deletion: delete neighbors that became isolated
        for neighbor_id in neighbors_to_check:
            if (
                self.nodes[neighbor_id].id != -1
                and not self.edges_per_node.get(neighbor_id)
            ):
                self._remove_node(neighbor_id)

    def _add_position_edge(self, n1: int, n2: int) -> None:
        """Add position edge between two nodes."""
        if self.edges_pos[n1, n2] == 0:
            self.edges_pos[n1, n2] = 1
            self.edges_pos[n2, n1] = 1
            self.edges_per_node[n1].add(n2)
            self.edges_per_node[n2].add(n1)

    def _remove_all_edges(self, n1: int, n2: int) -> None:
        """Remove all edges between two nodes."""
        self.edges_pos[n1, n2] = 0
        self.edges_pos[n2, n1] = 0
        self.edges_color[n1, n2] = 0
        self.edges_color[n2, n1] = 0
        self.edges_normal[n1, n2] = 0
        self.edges_normal[n2, n1] = 0
        self.edge_age[n1, n2] = 0
        self.edge_age[n2, n1] = 0
        self.edges_per_node[n1].discard(n2)
        self.edges_per_node[n2].discard(n1)

    def _compute_normal_from_positions(
        self, positions: list[np.ndarray], cog_sum: np.ndarray
    ) -> np.ndarray:
        """Compute normal vector from positions using PCA."""
        ect = len(positions)
        if ect < 2:
            return np.array([0.0, 0.0, 1.0])

        positions_arr = np.array(positions)
        cog = cog_sum / ect
        centered = positions_arr - cog
        cov = np.dot(centered.T, centered) / ect

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0].copy()
            if normal[1] < 0:
                normal = -normal
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            return normal
        except np.linalg.LinAlgError:
            return np.array([0.0, 0.0, 1.0])

    def _find_two_nearest(self, position: np.ndarray) -> tuple[int, int, float, float]:
        """Find the two nearest nodes to input position.

        Returns:
            Tuple of (s1_id, s2_id, dist1_sq, dist2_sq).
        """
        min_dist1 = float("inf")
        min_dist2 = float("inf")
        s1_id = -1
        s2_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            dist_sq = np.sum((position - node.position) ** 2)

            if dist_sq < min_dist1:
                min_dist2 = min_dist1
                s2_id = s1_id
                min_dist1 = dist_sq
                s1_id = node.id
            elif dist_sq < min_dist2:
                min_dist2 = dist_sq
                s2_id = node.id

        return s1_id, s2_id, min_dist1, min_dist2

    def _check_utility_criterion(self) -> None:
        """Check and remove node with lowest utility if criterion met.

        Every kappa iterations, check if max_error / min_utility > utility_k.
        """
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
                self._remove_node(min_utility_id)
                self.n_utility_removals += 1

    def _ais_growing_process(
        self,
        position: np.ndarray,
        color: np.ndarray | None,
        s1_id: int,
        s2_id: int,
        dist1_sq: float,
        dist2_sq: float,
    ) -> bool:
        """Add-if-Silent rule-based growing process.

        If both winner nodes are within tolerance area [theta_min, theta_max],
        add the input directly as a new node.

        Uses Euclidean distances (not squared) for threshold comparison.

        Returns:
            True if a new node was added.
        """
        p = self.params

        if not self._addable_indices:
            return False

        # Convert to Euclidean distances for AiS threshold comparison
        dist1 = np.sqrt(dist1_sq)
        dist2 = np.sqrt(dist2_sq)

        # Check AiS conditions
        s1_in_range = p.theta_ais_min < dist1 < p.theta_ais_max
        s2_in_range = p.theta_ais_min < dist2 < p.theta_ais_max

        if s1_in_range and s2_in_range:
            # Add input directly as new node
            new_error = 0.5 * (self.nodes[s1_id].error + self.nodes[s2_id].error)
            new_utility = 0.5 * (self.nodes[s1_id].utility + self.nodes[s2_id].utility)

            new_id = self._add_node(position, color, error=new_error, utility=new_utility)

            if new_id == -1:
                return False

            # Connect new node to s1 and s2
            self._add_position_edge(new_id, s1_id)
            self._add_position_edge(new_id, s2_id)

            # Initialize color and normal edges based on similarity
            # Color edge with s1
            color_dist_s1 = np.sum((self.nodes[new_id].color - self.nodes[s1_id].color) ** 2)
            if color_dist_s1 < p.tau_color * p.tau_color:
                self.edges_color[new_id, s1_id] = 1
                self.edges_color[s1_id, new_id] = 1

            # Color edge with s2
            color_dist_s2 = np.sum((self.nodes[new_id].color - self.nodes[s2_id].color) ** 2)
            if color_dist_s2 < p.tau_color * p.tau_color:
                self.edges_color[new_id, s2_id] = 1
                self.edges_color[s2_id, new_id] = 1

            self.n_ais_additions += 1
            return True

        return False

    def _add_new_node_distance(
        self, position: np.ndarray, color: np.ndarray | None = None
    ) -> None:
        """Add 2 new connected nodes at input position (for distant inputs)."""
        p = self.params

        r = self._add_node(position, color)
        if r == -1:
            return

        offset = self.rng.random(3) * p.dis_thv / 10.0
        q_pos = position + offset
        q = self._add_node(q_pos, color)
        if q == -1:
            self._remove_node(r)
            return

        self._add_position_edge(r, q)

    def _gng_learn(
        self,
        s1: int,
        s2: int,
        v_pos: np.ndarray,
        v_color: np.ndarray | None,
        e1: float,
        e2: float,
    ) -> None:
        """Single learning step (GNG-DT style with multiple topologies)."""
        p = self.params
        n1 = self.nodes[s1]
        n2 = self.nodes[s2]

        # Add position edge
        self._add_position_edge(s1, s2)

        # Update color edge
        color_dist_sq = np.sum((n1.color - n2.color) ** 2)
        if color_dist_sq < p.tau_color * p.tau_color:
            self.edges_color[s1, s2] = 1
            self.edges_color[s2, s1] = 1
        else:
            self.edges_color[s1, s2] = 0
            self.edges_color[s2, s1] = 0

        # Calculate normal dot product BEFORE updating
        normal_dot = np.dot(n1.normal, n2.normal)

        # Store s1's original position for PCA
        s1_original_pos = n1.position.copy()

        # Reset edge age
        self.edge_age[s1, s2] = 0
        self.edge_age[s2, s1] = 0

        # Update winner position
        n1.position += e1 * (v_pos - n1.position)

        # Update winner color
        if v_color is not None:
            n1.color += e1 * (v_color - n1.color)

        # Update neighbors and collect positions for PCA
        neighbors_to_remove = []
        pca_positions = [s1_original_pos]
        pca_cog = s1_original_pos.copy()

        for neighbor_id in list(self.edges_per_node[s1]):
            if neighbor_id == s1:
                continue

            neighbor = self.nodes[neighbor_id]

            # Move neighbor toward input
            neighbor.position += e2 * (v_pos - neighbor.position)

            # Increment edge age
            self.edge_age[s1, neighbor_id] += 1
            self.edge_age[neighbor_id, s1] += 1

            # Collect for PCA
            pca_positions.append(neighbor.position.copy())
            pca_cog += neighbor.position

            # Check age threshold
            if self.edge_age[s1, neighbor_id] > p.max_age:
                neighbors_to_remove.append(neighbor_id)

        # Remove old edges
        for neighbor_id in neighbors_to_remove:
            self._remove_all_edges(s1, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # Update color neighbors
        if v_color is not None:
            for i in range(len(self.nodes)):
                if self.nodes[i].id != -1 and self.edges_color[s1, i] == 1 and i != s1:
                    self.nodes[i].color += e2 * (v_color - self.nodes[i].color)

        # Compute normal via PCA
        n1.normal = self._compute_normal_from_positions(pca_positions, pca_cog)

        # Update normal edge
        if np.abs(normal_dot) > p.tau_normal:
            self.edges_normal[s1, s2] = 1
            self.edges_normal[s2, s1] = 1
        else:
            self.edges_normal[s1, s2] = 0
            self.edges_normal[s2, s1] = 0

    def _discount_errors(self) -> None:
        """Decay all node errors and utilities."""
        p = self.params
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error
            node.utility -= p.chi * node.utility
            if node.error < 0:
                node.error = 0.0
            if node.utility < 0:
                node.utility = 0.0

    def _node_add(self) -> None:
        """Add a new node between highest-error node and its neighbor."""
        p = self.params

        if not self._addable_indices:
            return

        # Find node with maximum error
        max_err = -1.0
        q = -1
        first_node_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue
            if first_node_id == -1:
                first_node_id = node.id
            if node.error > max_err:
                max_err = node.error
                q = node.id

        if q == -1:
            return

        # Find neighbor with maximum error
        max_err_f = -1.0
        f = -1
        for neighbor_id in self.edges_per_node.get(q, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f = neighbor_id

        if f == -1:
            return

        # Add new node between q and f
        new_pos = 0.5 * (self.nodes[q].position + self.nodes[f].position)
        new_color = 0.5 * (self.nodes[q].color + self.nodes[f].color)
        r = self._add_node(new_pos, new_color)

        if r == -1:
            return

        # Initialize normal
        new_normal = 0.5 * (self.nodes[q].normal + self.nodes[f].normal)
        norm = np.linalg.norm(new_normal)
        if norm > 1e-10:
            self.nodes[r].normal = new_normal / norm
        else:
            self.nodes[r].normal = np.array([0.0, 0.0, 1.0])

        # Update edges
        self.edges_pos[q, f] = 0
        self.edges_pos[f, q] = 0
        self.edges_per_node[q].discard(f)
        self.edges_per_node[f].discard(q)

        # Inherit color/normal edges
        self.edges_color[q, r] = self.edges_color[q, f]
        self.edges_color[r, q] = self.edges_color[q, f]
        self.edges_color[f, r] = self.edges_color[q, f]
        self.edges_color[r, f] = self.edges_color[q, f]
        self.edges_color[q, f] = 0
        self.edges_color[f, q] = 0

        self.edges_normal[q, r] = self.edges_normal[q, f]
        self.edges_normal[r, q] = self.edges_normal[q, f]
        self.edges_normal[f, r] = self.edges_normal[q, f]
        self.edges_normal[r, f] = self.edges_normal[q, f]
        self.edges_normal[q, f] = 0
        self.edges_normal[f, q] = 0

        # Add position edges
        self._add_position_edge(q, r)
        self._add_position_edge(r, f)

        # Update errors and utilities
        self.nodes[q].error *= 0.5
        self.nodes[f].error *= 0.5
        self.nodes[q].utility *= 0.5
        self.nodes[f].utility *= 0.5
        self.nodes[r].error = self.nodes[q].error
        self.nodes[r].utility = self.nodes[q].utility

    def _one_train_update(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
    ) -> float:
        """Single training iteration with AiS rule.

        Returns the squared distance to the winner.
        """
        p = self.params

        # Find two nearest nodes
        s1, s2, dist1_sq, dist2_sq = self._find_two_nearest(position)

        if s1 == -1 or s2 == -1:
            return 0.0

        # Distance threshold check (GNG-DT style)
        if dist1_sq > p.dis_thv * p.dis_thv and self.n_nodes < p.max_nodes - 2:
            self._add_new_node_distance(position, color)
            self._discount_errors()
            return 0.0

        # Add-if-Silent rule (AiS-GNG style)
        self._ais_growing_process(position, color, s1, s2, dist1_sq, dist2_sq)

        # Update error and utility
        self.nodes[s1].error += dist1_sq
        self.nodes[s1].utility += dist2_sq - dist1_sq

        # Learning step
        self._gng_learn(s1, s2, position, color, p.eps_b, p.eps_n)

        # Decay errors and utilities
        self._discount_errors()

        # Check utility criterion every kappa iterations
        if self.n_learning > 0 and self.n_learning % p.kappa == 0:
            self._check_utility_criterion()

        self.n_learning += 1
        return dist1_sq

    def _gng_main_cycle(
        self,
        data: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        """Run one main cycle (lambda_ iterations + node addition)."""
        p = self.params
        n_samples = len(data)
        total_error = 0.0

        for i in range(p.lambda_):
            idx = self.rng.integers(0, n_samples)
            color = colors[idx] if colors is not None else None
            error = self._one_train_update(data[idx], color)
            total_error += error

        # Node addition
        total_error /= p.lambda_
        if self.n_nodes < p.max_nodes and total_error > 1e-6:
            self._node_add()

    def train(
        self,
        data: np.ndarray,
        colors: np.ndarray | None = None,
        n_iterations: int = 1000,
        callback: Callable[[AiSGNGDT, int], None] | None = None,
    ) -> AiSGNGDT:
        """Train on data for multiple iterations.

        Args:
            data: Training data of shape (n_samples, 3) for positions.
            colors: Optional color data of shape (n_samples, 3).
            n_iterations: Number of training iterations.
            callback: Optional callback(self, iteration) called per cycle.

        Returns:
            self for chaining.
        """
        p = self.params

        n_cycles = n_iterations // p.lambda_
        if n_cycles == 0:
            n_cycles = 1

        for cycle in range(n_cycles):
            self._gng_main_cycle(data, colors)

            if callback is not None:
                callback(self, cycle * p.lambda_)

        return self

    def partial_fit(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
    ) -> AiSGNGDT:
        """Single online learning step."""
        p = self.params

        error = self._one_train_update(position, color)
        self._total_error += error
        self._n_trial += 1

        if self._n_trial >= p.lambda_:
            avg_error = self._total_error / p.lambda_
            if self.n_nodes < p.max_nodes and avg_error > 1e-6:
                self._node_add()
            self._n_trial = 0
            self._total_error = 0.0

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
        for i in range(len(self.nodes)):
            if self.nodes[i].id == -1:
                continue
            for j in range(i + 1, len(self.nodes)):
                if self.nodes[j].id == -1:
                    continue
                if self.edges_normal[i, j] > 0:
                    count += 1
        return count

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure (position topology only)."""
        active_nodes = []
        index_map = {}

        for node in self.nodes:
            if node.id != -1:
                index_map[node.id] = len(active_nodes)
                active_nodes.append(node.position.copy())

        nodes = np.array(active_nodes) if active_nodes else np.zeros((0, 3))

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
            Tuple of (nodes, pos_edges, color_edges, normal_edges).
        """
        active_nodes = []
        index_map = {}

        for node in self.nodes:
            if node.id != -1:
                index_map[node.id] = len(active_nodes)
                active_nodes.append(node.position.copy())

        nodes = np.array(active_nodes) if active_nodes else np.zeros((0, 3))

        # Position edges
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

        # Color edges
        color_edges = []
        for i in range(len(self.nodes)):
            if self.nodes[i].id == -1:
                continue
            for j in range(i + 1, len(self.nodes)):
                if self.nodes[j].id == -1:
                    continue
                if self.edges_color[i, j] > 0:
                    color_edges.append((index_map[i], index_map[j]))

        # Normal edges
        normal_edges = []
        for i in range(len(self.nodes)):
            if self.nodes[i].id == -1:
                continue
            for j in range(i + 1, len(self.nodes)):
                if self.nodes[j].id == -1:
                    continue
                if self.edges_normal[i, j] > 0:
                    normal_edges.append((index_map[i], index_map[j]))

        return nodes, pos_edges, color_edges, normal_edges

    def get_node_normals(self) -> np.ndarray:
        """Get normal vectors for active nodes."""
        return np.array([node.normal for node in self.nodes if node.id != -1])

    def get_node_colors(self) -> np.ndarray:
        """Get color vectors for active nodes."""
        return np.array([node.color for node in self.nodes if node.id != -1])

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes."""
        return np.array([node.error for node in self.nodes if node.id != -1])

    def get_node_utilities(self) -> np.ndarray:
        """Get utility values for active nodes."""
        return np.array([node.utility for node in self.nodes if node.id != -1])


# Aliases
AddIfSilentGNGDT = AiSGNGDT
