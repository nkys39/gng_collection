"""GNG-DT (Growing Neural Gas with Different Topologies) implementation.

Based on:
    - Toda, Y., et al. (2022). "GNG with Different Topologies for 3D Point Cloud"
    - Reference implementation: toda_gngdt (C) - gng_livox/src/gng.c

GNG-DT learns multiple independent edge topologies based on different properties:
    - Position topology (edge): Standard GNG edges based on spatial proximity
    - Color topology (cedge): Edges based on color similarity (threshold-based)
    - Normal topology (nedge): Edges based on normal vector similarity (dot product)

Key features from original code:
    - Winner selection uses ONLY position information (first 3 dimensions)
    - Normal vectors are computed via PCA on winner + neighbor positions EVERY iteration
    - Normal edge (nedge) is updated ONLY between s1 and s2 (the two winners)
    - Normal is stored in node dimensions [4-6]

See REFERENCE.md for details.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GNGDTParams:
    """GNG-DT hyperparameters.

    Based on original toda_gngdt/gng.c parameters.

    Attributes:
        max_nodes: Maximum number of nodes (GNGN in original).
        lambda_: Node insertion interval (ramda in original = 200).
        eps_b: Learning rate for the winner node (e1 = 0.05 in original).
        eps_n: Learning rate for neighbor nodes (e2 = 0.0005 in original).
        alpha: Error decay rate when splitting.
        beta: Global error decay rate (dise in original = 0.0005).
        max_age: Maximum edge age before removal (MAX_AGE = 88 in original).
        tau_color: Color similarity threshold (cthv = 0.05 in original).
        tau_normal: Normal similarity threshold (nthv = 0.998 in original).
            Edge created if |dot product| > tau_normal.
        dis_thv: Distance threshold for adding new nodes (DIS_THV = 0.5 in original).
            If winner is further than this, add 2 new nodes at input.
        thv: Error threshold for node addition (THV = 0.000001 in original).
            Only add node if average error > thv.
    """

    max_nodes: int = 100
    lambda_: int = 200  # Original: ramda = 200
    eps_b: float = 0.05  # Original: e1 = 0.05
    eps_n: float = 0.0005  # Original: e2 = 0.0005
    alpha: float = 0.5
    beta: float = 0.0005  # Original: dise = 0.0005
    max_age: int = 88  # Original: MAX_AGE = 88
    # GNG-DT specific parameters
    tau_color: float = 0.05  # Original: cthv = 0.05
    tau_normal: float = 0.998  # Original: nthv = 0.998
    dis_thv: float = 0.5  # Original: DIS_THV = 0.5
    thv: float = 0.000001  # Original: THV = 0.001*0.001


@dataclass
class GNGDTNode:
    """A neuron node in the GNG-DT network.

    Mirrors the original node structure:
        node[0-2]: position (x, y, z)
        node[3]: color (simplified, original has LDIM dimensions)
        node[4-6]: normal vector (nx, ny, nz)

    Attributes:
        id: Node ID (-1 means invalid/removed).
        position: 3D position vector.
        color: RGB color vector.
        normal: Normal vector (unit vector, computed via PCA).
        error: Accumulated error (gng_err).
        utility: Utility value (gng_u).
    """

    id: int = -1
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    error: float = 0.0
    utility: float = 0.0


class GrowingNeuralGasDT:
    """GNG-DT (Different Topologies) algorithm implementation.

    Faithfully implements the original toda_gngdt/gng.c algorithm.

    Key behaviors from original:
        1. Winner selection uses ONLY position (node[0-2])
        2. Normal computed via PCA on s1 + neighbors EVERY iteration
        3. nedge updated ONLY between s1 and s2
        4. cedge updated between s1 and s2 based on color difference
        5. Edge age incremented for all s1's neighbors

    Attributes:
        params: GNG-DT hyperparameters.
        nodes: List of neuron nodes.
        edges_pos: Position-based edge age matrix (0 = no edge).
        edges_color: Color-based edge connectivity matrix.
        edges_normal: Normal-based edge connectivity matrix.
        edges_per_node: Adjacency list for position edges.
        n_learning: Total number of learning iterations.
    """

    def __init__(
        self,
        params: GNGDTParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG-DT.

        Args:
            params: GNG-DT hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.params = params or GNGDTParams()
        self.rng = np.random.default_rng(seed)

        max_n = self.params.max_nodes

        # Node management
        self.nodes: list[GNGDTNode] = [GNGDTNode() for _ in range(max_n)]
        self._addable_indices: deque[int] = deque(range(max_n))

        # Edge matrices (like original: edge, cedge, nedge, age)
        self.edges_pos = np.zeros((max_n, max_n), dtype=np.int32)  # edge
        self.edges_color = np.zeros((max_n, max_n), dtype=np.int32)  # cedge
        self.edges_normal = np.zeros((max_n, max_n), dtype=np.int32)  # nedge
        self.edge_age = np.zeros((max_n, max_n), dtype=np.int32)  # age

        # Adjacency list for quick neighbor lookup
        self.edges_per_node: dict[int, set[int]] = {}

        # Counters
        self.n_learning = 0
        self._n_trial = 0
        self._total_error = 0.0  # Accumulated error for node addition decision

        # Initialize with 2 random nodes (like original init_gng)
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes connected by edges."""
        for i in range(2):
            pos = self.rng.random(3).astype(np.float64)
            self._add_node(pos)

        # Connect initial 2 nodes (like original)
        self.edges_pos[0, 1] = 1
        self.edges_pos[1, 0] = 1
        self.edges_color[0, 1] = 1
        self.edges_color[1, 0] = 1
        self.edges_normal[0, 1] = 1
        self.edges_normal[1, 0] = 1
        self.edges_per_node[0] = {1}
        self.edges_per_node[1] = {0}

    def _add_node(self, position: np.ndarray, color: np.ndarray | None = None) -> int:
        """Add a new node.

        Args:
            position: 3D position vector.
            color: RGB color vector (optional).

        Returns:
            ID of the new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = GNGDTNode(
            id=node_id,
            position=position.copy(),
            color=color.copy() if color is not None else np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            error=0.0,
            utility=0.0,
        )
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node (like original node_delete)."""
        # Clear all edges involving this node
        for other_id in list(self.edges_per_node.get(node_id, set())):
            self._remove_all_edges(node_id, other_id)

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _add_position_edge(self, n1: int, n2: int) -> None:
        """Add position edge between two nodes."""
        if self.edges_pos[n1, n2] == 0:
            self.edges_pos[n1, n2] = 1
            self.edges_pos[n2, n1] = 1
            self.edges_per_node[n1].add(n2)
            self.edges_per_node[n2].add(n1)

    def _remove_all_edges(self, n1: int, n2: int) -> None:
        """Remove all edges between two nodes (like original when age > MAX_AGE)."""
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

    def _add_new_node_distance(
        self, position: np.ndarray, color: np.ndarray | None = None
    ) -> None:
        """Add 2 new connected nodes at the input position.

        Original: add_new_node_distance (gng.c:845-899)
        Called when the winner node is too far from the input (> DIS_THV).

        Args:
            position: Input 3D position vector.
            color: Input color vector (optional).
        """
        p = self.params

        # Add first node at input position
        r = self._add_node(position, color)
        if r == -1:
            return

        # Add second node slightly offset (original: rnd() * DIS_THV / 10)
        offset = self.rng.random(3) * p.dis_thv / 10.0
        q_pos = position + offset
        q = self._add_node(q_pos, color)
        if q == -1:
            self._remove_node(r)
            return

        # Connect the two new nodes with position edge only
        # (original: cedge, nedge, pedge all 0 for new distance-based nodes)
        self._add_position_edge(r, q)

    def _delete_node_gngu(self) -> bool:
        """Delete node with minimum utility.

        Original: delete_node_gngu (gng.c:552-581)
        Called at ramda/2 during gng_main loop.

        Returns:
            True if a node was deleted, False otherwise.
        """
        p = self.params

        if self.n_nodes <= 10:
            return False

        # Find node with minimum utility and minimum error
        min_u = float("inf")
        min_u_id = -1
        min_err = float("inf")

        for node in self.nodes:
            if node.id == -1:
                continue
            if node.utility < min_u:
                min_u = node.utility
                min_u_id = node.id
            if node.error < min_err:
                min_err = node.error

        # Delete if min_err < THV (original gng.c:574-578)
        if min_err < p.thv and min_u_id != -1:
            self._remove_node(min_u_id)
            return True

        return False

    def _compute_normal_pca(self, node_id: int) -> np.ndarray:
        """Compute normal vector using PCA on node + neighbor positions.

        Follows original gng.c:712-728:
            - Collect s1's position and all position-edge neighbors
            - Compute PCA
            - Normal = eigenvector of smallest eigenvalue
            - Normalize to unit length

        Args:
            node_id: Node ID (s1 in original).

        Returns:
            Unit normal vector.
        """
        # Collect positions: s1 + all neighbors with position edge
        positions = [self.nodes[node_id].position.copy()]
        for neighbor_id in self.edges_per_node.get(node_id, set()):
            if self.nodes[neighbor_id].id != -1:
                positions.append(self.nodes[neighbor_id].position.copy())

        ect = len(positions)
        if ect < 2:
            # Not enough points for PCA, return default
            return np.array([0.0, 0.0, 1.0])

        positions = np.array(positions)

        # Compute centroid (cog in original)
        cog = np.mean(positions, axis=0)

        # Compute covariance matrix
        centered = positions - cog
        cov = np.dot(centered.T, centered) / ect

        # Eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Smallest eigenvalue corresponds to normal direction
            normal = eigenvectors[:, 0].copy()

            # Original: if (ev1[1] < 0) multiply by -1
            # This ensures consistent orientation (y-component positive)
            if normal[1] < 0:
                normal = -normal

            # Normalize (original uses invSqrt)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm

            return normal
        except np.linalg.LinAlgError:
            return np.array([0.0, 0.0, 1.0])

    def _find_two_nearest(self, position: np.ndarray) -> tuple[int, int, float, float]:
        """Find the two nearest nodes to input position.

        Winner selection uses ONLY position (original gng.c:914-948).

        Args:
            position: Input 3D position vector.

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

            # Distance based on POSITION ONLY (first 3 dimensions)
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

    def _gng_learn(
        self,
        s1: int,
        s2: int,
        v_pos: np.ndarray,
        v_color: np.ndarray | None,
        e1: float,
        e2: float,
    ) -> None:
        """Single learning step (mirrors original gng_learn function).

        Args:
            s1: First winner node ID.
            s2: Second winner node ID.
            v_pos: Input position vector.
            v_color: Input color vector (optional).
            e1: Winner learning rate.
            e2: Neighbor learning rate.
        """
        p = self.params
        n1 = self.nodes[s1]
        n2 = self.nodes[s2]

        # Add position edge if not exists (original gng.c:611-616)
        self._add_position_edge(s1, s2)

        # Calculate color distance and update cedge (original gng.c:618-630)
        if v_color is not None:
            color_dist_sq = np.sum((n1.color - n2.color) ** 2)
            if color_dist_sq < p.tau_color * p.tau_color:
                self.edges_color[s1, s2] = 1
                self.edges_color[s2, s1] = 1
            else:
                self.edges_color[s1, s2] = 0
                self.edges_color[s2, s1] = 0

        # Calculate normal dot product BEFORE updating normal (original gng.c:632-635)
        normal_dot = np.dot(n1.normal, n2.normal)

        # Reset edge age between s1 and s2 (original gng.c:645-646)
        self.edge_age[s1, s2] = 0
        self.edge_age[s2, s1] = 0

        # Update winner position (original gng.c:650-652)
        n1.position += e1 * (v_pos - n1.position)

        # Update winner color if provided
        if v_color is not None:
            n1.color += e1 * (v_color - n1.color)

        # Update neighbors and handle edge aging (original gng.c:655-695)
        neighbors_to_remove = []
        for neighbor_id in list(self.edges_per_node[s1]):
            if neighbor_id == s1:
                continue

            neighbor = self.nodes[neighbor_id]

            # Move neighbor toward input (original gng.c:657-658)
            neighbor.position += e2 * (v_pos - neighbor.position)

            # Increment edge age (original gng.c:660-661)
            self.edge_age[s1, neighbor_id] += 1
            self.edge_age[neighbor_id, s1] += 1

            # Check age threshold (original gng.c:673-694)
            if self.edge_age[s1, neighbor_id] > p.max_age:
                neighbors_to_remove.append(neighbor_id)

        # Remove old edges and isolated nodes
        for neighbor_id in neighbors_to_remove:
            self._remove_all_edges(s1, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # Update color neighbors (original gng.c:697-699)
        if v_color is not None:
            for i in range(len(self.nodes)):
                if self.nodes[i].id != -1 and self.edges_color[s1, i] == 1 and i != s1:
                    self.nodes[i].color += e2 * (v_color - self.nodes[i].color)

        # Compute normal via PCA (original gng.c:712-728)
        n1.normal = self._compute_normal_pca(s1)

        # Update nedge between s1 and s2 based on normal dot product (original gng.c:741-748)
        # NOTE: Uses the dot product calculated BEFORE the PCA update
        if np.abs(normal_dot) > p.tau_normal:
            self.edges_normal[s1, s2] = 1
            self.edges_normal[s2, s1] = 1
        else:
            self.edges_normal[s1, s2] = 0
            self.edges_normal[s2, s1] = 0

    def _discount_errors(self) -> None:
        """Decay all node errors (original discount_err_gng)."""
        p = self.params
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error
            node.utility -= p.beta * node.utility
            if node.error < 0:
                node.error = 0.0
            if node.utility < 0:
                node.utility = 0.0

    def _node_add(self) -> None:
        """Add a new node (original node_add_gng).

        Original gng.c:432-550.
        Also includes utility-based deletion of low-utility nodes.
        """
        p = self.params

        if not self._addable_indices:
            return

        # Find node q with maximum error, minimum utility, and minimum error
        # Also build delete list (original gng.c:445-468)
        max_err = -1.0
        q = -1
        min_u = float("inf")
        min_u_id = -1
        min_err = float("inf")
        delete_list = []

        for node in self.nodes:
            if node.id == -1:
                continue

            if node.error > max_err:
                max_err = node.error
                q = node.id

            if node.utility < min_u:
                min_u = node.utility
                min_u_id = node.id

            if node.error < min_err:
                min_err = node.error

            # Original: if(net->gng_u[i]*1000000.0 < 100.0) -> u < 0.0001
            if node.utility < 0.0001:
                delete_list.append(node.id)

        if q == -1:
            return

        # Find neighbor f of q with maximum error
        max_err_f = -1.0
        f = -1
        for neighbor_id in self.edges_per_node.get(q, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f = neighbor_id

        if f == -1:
            return

        # Add new node r between q and f (original gng.c:483-485)
        new_pos = 0.5 * (self.nodes[q].position + self.nodes[f].position)
        new_color = 0.5 * (self.nodes[q].color + self.nodes[f].color)
        r = self._add_node(new_pos, new_color)

        if r == -1:
            return

        # Update edges (original gng.c:487-528)
        # Remove edge between q and f
        self.edges_pos[q, f] = 0
        self.edges_pos[f, q] = 0
        self.edges_per_node[q].discard(f)
        self.edges_per_node[f].discard(q)

        # Inherit cedge and nedge for new node
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

        # Add position edges q-r and r-f
        self._add_position_edge(q, r)
        self._add_position_edge(r, f)

        # Update errors (original gng.c:530-537)
        self.nodes[q].error *= 0.5
        self.nodes[f].error *= 0.5
        self.nodes[q].utility *= 0.5
        self.nodes[f].utility *= 0.5
        self.nodes[r].error = self.nodes[q].error
        self.nodes[r].utility = self.nodes[q].utility

        # Utility-based deletion (original gng.c:544-549)
        # Delete low-utility nodes if network is large and error is small
        if self.n_nodes > 10 and min_err < p.thv:
            for del_id in delete_list:
                if self.nodes[del_id].id != -1 and del_id != r:
                    self._remove_node(del_id)

    def _one_train_update(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
    ) -> float:
        """Single training iteration (mirrors original learn_epoch).

        Returns the squared distance to the winner for error accumulation.
        """
        p = self.params

        # Find two nearest nodes
        s1, s2, dist1_sq, dist2_sq = self._find_two_nearest(position)

        if s1 == -1 or s2 == -1:
            return 0.0

        # DIS_THV check (original gng.c:953-957)
        # If winner is too far, add 2 new nodes at input and return
        if dist1_sq > p.dis_thv * p.dis_thv and self.n_nodes < p.max_nodes - 2:
            self._add_new_node_distance(position, color)
            self._discount_errors()
            return 0.0

        # Update accumulated error and utility (original gng.c:960-962)
        self.nodes[s1].error += dist1_sq
        self.nodes[s1].utility += dist2_sq - dist1_sq

        # Learning step
        self._gng_learn(s1, s2, position, color, p.eps_b, p.eps_n)

        # Decay errors (original: inside gng_learn for all nodes)
        self._discount_errors()

        self.n_learning += 1
        return dist1_sq

    def _gng_main_cycle(
        self,
        data: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        """Run one gng_main cycle (lambda_ iterations + node addition).

        Original gng_main (gng.c:973-993):
            - Run ramda iterations
            - At ramda/2, call delete_node_gngu (flag=2)
            - After ramda, add node if total_error > THV
        """
        p = self.params
        n_samples = len(data)
        total_error = 0.0

        for i in range(p.lambda_):
            # Random sample selection
            idx = self.rng.integers(0, n_samples)
            color = colors[idx] if colors is not None else None

            # At ramda/2, use flag=2 (call delete_node_gngu)
            if i == p.lambda_ // 2:
                error = self._one_train_update(data[idx], color)
                total_error += error
                # flag=2: call delete_node_gngu
                if self.n_nodes > 2:
                    self._delete_node_gngu()
            else:
                # flag=1: normal learning
                error = self._one_train_update(data[idx], color)
                total_error += error

        # Node addition (original gng.c:986-990)
        total_error /= p.lambda_
        if self.n_nodes < p.max_nodes and total_error > p.thv:
            self._node_add()

    def train(
        self,
        data: np.ndarray,
        colors: np.ndarray | None = None,
        n_iterations: int = 1000,
        callback: Callable[[GrowingNeuralGasDT, int], None] | None = None,
    ) -> GrowingNeuralGasDT:
        """Train on data for multiple iterations.

        Uses the original gng_main cycle approach:
            - Every lambda_ iterations = 1 gng_main cycle
            - n_iterations specifies total learning steps

        Args:
            data: Training data of shape (n_samples, 3) for positions.
            colors: Optional color data of shape (n_samples, 3).
            n_iterations: Number of training iterations.
            callback: Optional callback(self, iteration) called per gng_main cycle.

        Returns:
            self for chaining.
        """
        p = self.params

        # Calculate number of gng_main cycles
        n_cycles = n_iterations // p.lambda_
        if n_cycles == 0:
            n_cycles = 1

        for cycle in range(n_cycles):
            self._gng_main_cycle(data, colors)

            if callback is not None:
                # Call callback with iteration count (cycle * lambda_)
                callback(self, cycle * p.lambda_)

        return self

    def partial_fit(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
    ) -> GrowingNeuralGasDT:
        """Single online learning step.

        Follows original gng_main logic:
            - Accumulates error over lambda_ iterations
            - At lambda_/2, calls delete_node_gngu
            - After lambda_ iterations, adds node if error > THV
        """
        p = self.params

        # Run single learning iteration
        error = self._one_train_update(position, color)
        self._total_error += error
        self._n_trial += 1

        # At lambda_/2, call delete_node_gngu (flag=2)
        if self._n_trial == p.lambda_ // 2:
            if self.n_nodes > 2:
                self._delete_node_gngu()

        # After lambda_ iterations, check for node addition
        if self._n_trial >= p.lambda_:
            avg_error = self._total_error / p.lambda_
            if self.n_nodes < p.max_nodes and avg_error > p.thv:
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


# Alias
GNGDT = GrowingNeuralGasDT
