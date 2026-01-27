"""GNG-DT Robot Version with traversability analysis features.

This extends the base GNG-DT with robot-specific features from the original
toda_gngdt implementation:
    - pedge: Traversability edge (connects nodes with same traversability)
    - traversability_property: Whether node is on traversable surface
    - through_property: Based on surface inclination angle
    - dimension_property: Based on PCA eigenvalues (surface planarity)
    - contour: Edge detection based on angular gaps between neighbors
    - degree: Inclination cost for path planning
    - curvature: PCA residual based curvature

See REFERENCE.md for details on the original implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GNGDTRobotParams:
    """GNG-DT Robot hyperparameters.

    Includes all base GNG-DT parameters plus robot-specific thresholds.

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
        dis_thv: Distance threshold for adding new nodes (DIS_THV = 0.5).
        thv: Error threshold for node addition (THV = 0.000001).
        max_angle: Maximum traversable angle in degrees (MAXANGLE = 20).
        s1thv: Eigenvalue threshold for dimension property (s1thv = 1.0).
        contour_gap_threshold: Angular gap threshold for contour detection (135 degrees).
    """

    max_nodes: int = 100
    lambda_: int = 200
    eps_b: float = 0.05
    eps_n: float = 0.0005
    alpha: float = 0.5
    beta: float = 0.0005
    max_age: int = 88
    # GNG-DT specific parameters
    tau_color: float = 0.05
    tau_normal: float = 0.998
    dis_thv: float = 0.5
    thv: float = 0.000001
    # Robot-specific parameters
    max_angle: float = 20.0  # MAXANGLE in degrees
    s1thv: float = 1.0  # Eigenvalue threshold for dimension property
    contour_gap_threshold: float = 135.0  # Angular gap threshold for contour


@dataclass
class GNGDTRobotNode:
    """A neuron node in the GNG-DT Robot network.

    Extends base node with robot-specific properties.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        position: 3D position vector.
        color: RGB color vector.
        normal: Normal vector (unit vector, computed via PCA).
        error: Accumulated error (gng_err).
        utility: Utility value (gng_u).
        pca_residual: PCA residual r (node[7] in original).
        eigenvalues: PCA eigenvalues (node[8-10] in original).
        through_property: 1 if surface is roughly horizontal (within max_angle).
        dimension_property: 1 if surface is roughly planar (eigenvalue < s1thv).
        traversability_property: 1 if node is on traversable surface.
        contour: 1 if node is on the contour/edge of traversable region.
        degree: Inclination cost for path planning.
        curvature: Curvature cost based on PCA residual.
    """

    id: int = -1
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    error: float = 0.0
    utility: float = 0.0
    # PCA results
    pca_residual: float = -10.0  # -10 indicates not computed
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([-10.0, -10.0, -10.0]))
    # Robot-specific properties
    through_property: int = 0
    dimension_property: int = 0
    traversability_property: int = 0
    contour: int = 0
    degree: float = 0.0
    curvature: float = 0.0


class GrowingNeuralGasDTRobot:
    """GNG-DT Robot version with traversability analysis.

    Extends base GNG-DT with robot-specific features for terrain analysis
    and path planning applications.

    Key features:
        - Traversability edge (pedge): Connects nodes with same traversability
        - Surface inclination analysis (through_property)
        - Surface planarity analysis (dimension_property)
        - Traversability classification (traversability_property)
        - Contour/edge detection for traversable regions
        - Cost metrics for path planning (degree, curvature)

    Attributes:
        params: GNG-DT Robot hyperparameters.
        nodes: List of neuron nodes.
        edges_pos: Position-based edge age matrix.
        edges_color: Color-based edge connectivity matrix.
        edges_normal: Normal-based edge connectivity matrix.
        edges_traversability: Traversability-based edge connectivity matrix.
        edges_per_node: Adjacency list for position edges.
        n_learning: Total number of learning iterations.
    """

    def __init__(
        self,
        params: GNGDTRobotParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG-DT Robot.

        Args:
            params: GNG-DT Robot hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.params = params or GNGDTRobotParams()
        self.rng = np.random.default_rng(seed)

        max_n = self.params.max_nodes

        # Node management
        self.nodes: list[GNGDTRobotNode] = [GNGDTRobotNode() for _ in range(max_n)]
        self._addable_indices: deque[int] = deque(range(max_n))

        # Edge matrices
        self.edges_pos = np.zeros((max_n, max_n), dtype=np.int32)
        self.edges_color = np.zeros((max_n, max_n), dtype=np.int32)
        self.edges_normal = np.zeros((max_n, max_n), dtype=np.int32)
        self.edges_traversability = np.zeros((max_n, max_n), dtype=np.int32)  # pedge
        self.edge_age = np.zeros((max_n, max_n), dtype=np.int32)

        # Adjacency list for quick neighbor lookup
        self.edges_per_node: dict[int, set[int]] = {}

        # Counters
        self.n_learning = 0
        self._n_trial = 0
        self._total_error = 0.0

        # Precompute cos(max_angle)
        self._cos_max_angle = np.cos(np.radians(self.params.max_angle))

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
        self.edges_traversability[0, 1] = 1
        self.edges_traversability[1, 0] = 1
        self.edges_per_node[0] = {1}
        self.edges_per_node[1] = {0}

    def _add_node(self, position: np.ndarray, color: np.ndarray | None = None) -> int:
        """Add a new node."""
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = GNGDTRobotNode(
            id=node_id,
            position=position.copy(),
            color=color.copy() if color is not None else np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            error=0.0,
            utility=0.0,
            pca_residual=-10.0,
            eigenvalues=np.array([-10.0, -10.0, -10.0]),
            through_property=0,
            dimension_property=0,
            traversability_property=0,
            contour=0,
            degree=0.0,
            curvature=0.0,
        )
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node with cascading deletion."""
        neighbors_to_check = list(self.edges_per_node.get(node_id, set()))

        for other_id in neighbors_to_check:
            self._remove_all_edges(node_id, other_id)

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

        # Cascading deletion
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
        self.edges_traversability[n1, n2] = 0
        self.edges_traversability[n2, n1] = 0
        self.edge_age[n1, n2] = 0
        self.edge_age[n2, n1] = 0
        self.edges_per_node[n1].discard(n2)
        self.edges_per_node[n2].discard(n1)

    def _add_new_node_distance(
        self, position: np.ndarray, color: np.ndarray | None = None
    ) -> None:
        """Add 2 new connected nodes at the input position."""
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
        # Note: pedge is 0 for distance-based nodes (like original)

    def _delete_node_gngu(self) -> bool:
        """Delete node with minimum utility."""
        p = self.params

        if self.n_nodes <= 10:
            return False

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

        if min_err < p.thv and min_u_id != -1:
            self._remove_node(min_u_id)
            return True

        return False

    def _compute_normal_from_positions(
        self, positions: list[np.ndarray], cog_sum: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Compute normal vector from pre-collected positions.

        Returns:
            Tuple of (normal, pca_residual, eigenvalues).
        """
        ect = len(positions)
        if ect < 2:
            return (
                np.array([0.0, 0.0, 1.0]),
                -10.0,
                np.array([-10.0, -10.0, -10.0]),
            )

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

            # PCA residual is the smallest eigenvalue
            pca_residual = eigenvalues[0]

            return normal, pca_residual, eigenvalues.copy()
        except np.linalg.LinAlgError:
            return (
                np.array([0.0, 0.0, 1.0]),
                -10.0,
                np.array([-10.0, -10.0, -10.0]),
            )

    def _judge_contour(self, node_id: int) -> int:
        """Judge if a node is on the contour based on angular gaps.

        Original: judge_contour function (gng.c:39-67)
        Returns 1 if there's an angular gap >= 135 degrees between neighbors.
        """
        p = self.params
        node = self.nodes[node_id]

        # Collect pedge neighbors
        neighbors = []
        for i in range(len(self.nodes)):
            if self.edges_traversability[node_id, i] == 1 and i != node_id:
                if self.nodes[i].id != -1:
                    neighbors.append(i)

        if len(neighbors) < 2:
            return 0

        # Calculate angles to each neighbor
        angles = []
        for neighbor_id in neighbors:
            neighbor = self.nodes[neighbor_id]
            dx = neighbor.position[0] - node.position[0]
            dy = neighbor.position[1] - node.position[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle = 360.0 - abs(angle)
            angles.append(angle)

        # Sort angles
        angles.sort()

        # Check for gaps >= threshold
        # First check wrap-around gap
        wrap_gap = abs(360.0 - angles[-1] + angles[0])
        if wrap_gap >= p.contour_gap_threshold:
            return 1

        # Check consecutive gaps
        for i in range(len(angles) - 1):
            gap = abs(angles[i + 1] - angles[i])
            if gap >= p.contour_gap_threshold:
                return 1

        return 0

    def _update_robot_properties(
        self, s1: int, s2: int, traversable_neighbor_count: int, total_neighbors: int
    ) -> None:
        """Update robot-specific properties for winner node.

        Original: gng.c:750-841
        """
        p = self.params
        n1 = self.nodes[s1]

        # Dimension property (gng.c:752-757)
        # Based on eigenvalue - 1 if surface is roughly planar
        if n1.eigenvalues[0] >= 0 and n1.eigenvalues[0] < p.s1thv:
            n1.dimension_property = 1
        else:
            n1.dimension_property = 0

        # Through property (gng.c:759-766)
        # Based on surface inclination - 1 if roughly horizontal
        if abs(n1.normal[2]) > self._cos_max_angle:
            n1.through_property = 1
        else:
            n1.through_property = 0

        # Degree cost (gng.c:801-809)
        # Inclination cost for path planning
        if n1.through_property == 1:
            n1.degree = (1.0 - abs(n1.normal[2])) / (1.0 - self._cos_max_angle)
            if n1.degree > 1.0:
                n1.degree = 99.0
        else:
            n1.degree = 99.0

        # Curvature cost (gng.c:811-816)
        if n1.pca_residual >= 0 and n1.pca_residual < 0.001:
            n1.curvature = n1.pca_residual / 0.001
        else:
            n1.curvature = 99.0

        # Traversability property (gng.c:818-828)
        if n1.dimension_property == 1 and n1.through_property == 1:
            n1.traversability_property = 1
        else:
            n1.traversability_property = 0
            # Special case: few neighbors and all are traversable
            if total_neighbors > 0 and total_neighbors < 3:
                if total_neighbors == traversable_neighbor_count:
                    n1.traversability_property = 1

        # Update pedge based on traversability (gng.c:830-841)
        for neighbor_id in self.edges_per_node.get(s1, set()):
            if neighbor_id == s1:
                continue
            neighbor = self.nodes[neighbor_id]
            if n1.traversability_property == neighbor.traversability_property:
                self.edges_traversability[s1, neighbor_id] = 1
                self.edges_traversability[neighbor_id, s1] = 1
            else:
                self.edges_traversability[s1, neighbor_id] = 0
                self.edges_traversability[neighbor_id, s1] = 0

        # Update contour for s1 and s2
        # Count non-contour pedge neighbors before updating
        cct_s1 = 0
        ct_s1 = 0
        for i in range(len(self.nodes)):
            if self.edges_traversability[s1, i] == 1 and self.nodes[i].id != -1:
                ct_s1 += 1
                if self.nodes[i].contour == 0:
                    cct_s1 += 1

        n1.contour = self._judge_contour(s1)
        if cct_s1 == ct_s1 and n1.through_property == 0:
            n1.contour = 0

        # Update s2 contour as well
        n2 = self.nodes[s2]
        cct_s2 = 0
        ct_s2 = 0
        for i in range(len(self.nodes)):
            if self.edges_traversability[s2, i] == 1 and self.nodes[i].id != -1:
                ct_s2 += 1
                if self.nodes[i].contour == 0:
                    cct_s2 += 1

        n2.contour = self._judge_contour(s2)
        if cct_s2 == ct_s2 and n2.through_property == 0:
            n2.contour = 0

    def _find_two_nearest(self, position: np.ndarray) -> tuple[int, int, float, float]:
        """Find the two nearest nodes to input position."""
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

    def _gng_learn(
        self,
        s1: int,
        s2: int,
        v_pos: np.ndarray,
        v_color: np.ndarray | None,
        e1: float,
        e2: float,
    ) -> None:
        """Single learning step with robot feature updates."""
        p = self.params
        n1 = self.nodes[s1]
        n2 = self.nodes[s2]

        # Add position edge
        self._add_position_edge(s1, s2)

        # cedge update
        color_dist_sq = np.sum((n1.color - n2.color) ** 2)
        if color_dist_sq < p.tau_color * p.tau_color:
            self.edges_color[s1, s2] = 1
            self.edges_color[s2, s1] = 1
        else:
            self.edges_color[s1, s2] = 0
            self.edges_color[s2, s1] = 0

        # Normal dot product BEFORE update
        normal_dot = np.dot(n1.normal, n2.normal)

        # Collect s1's ORIGINAL position for PCA BEFORE update
        s1_original_pos = n1.position.copy()

        # Reset edge age
        self.edge_age[s1, s2] = 0
        self.edge_age[s2, s1] = 0

        # Update winner position
        n1.position += e1 * (v_pos - n1.position)

        if v_color is not None:
            n1.color += e1 * (v_color - n1.color)

        # Update neighbors and collect PCA positions
        neighbors_to_remove = []
        pca_positions = [s1_original_pos]
        pca_cog = s1_original_pos.copy()
        traversable_neighbor_count = 0

        for neighbor_id in list(self.edges_per_node[s1]):
            if neighbor_id == s1:
                continue

            neighbor = self.nodes[neighbor_id]

            neighbor.position += e2 * (v_pos - neighbor.position)

            self.edge_age[s1, neighbor_id] += 1
            self.edge_age[neighbor_id, s1] += 1

            pca_positions.append(neighbor.position.copy())
            pca_cog += neighbor.position

            # Count traversable neighbors
            if neighbor.traversability_property == 1:
                traversable_neighbor_count += 1

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

        # Compute normal via PCA (returns normal, residual, eigenvalues)
        normal, pca_residual, eigenvalues = self._compute_normal_from_positions(
            pca_positions, pca_cog
        )
        n1.normal = normal
        n1.pca_residual = pca_residual
        n1.eigenvalues = eigenvalues

        # nedge update using pre-PCA dot product
        if np.abs(normal_dot) > p.tau_normal:
            self.edges_normal[s1, s2] = 1
            self.edges_normal[s2, s1] = 1
        else:
            self.edges_normal[s1, s2] = 0
            self.edges_normal[s2, s1] = 0

        # Update robot-specific properties
        total_neighbors = len(pca_positions) - 1  # Exclude s1 itself
        self._update_robot_properties(
            s1, s2, traversable_neighbor_count, total_neighbors
        )

    def _discount_errors(self) -> None:
        """Decay all node errors."""
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
        """Add a new node."""
        p = self.params

        if not self._addable_indices:
            return

        max_err = -1.0
        q = -1
        min_u = float("inf")
        min_u_id = -1
        min_err = float("inf")
        delete_list = []
        first_node_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            if first_node_id == -1:
                first_node_id = node.id

            if node.error > max_err:
                max_err = node.error
                q = node.id

            if node.utility < min_u:
                min_u = node.utility
                min_u_id = node.id

            if node.error < min_err:
                min_err = node.error

            if node.utility < 0.0001 and node.id != first_node_id:
                delete_list.append(node.id)

        if q == -1:
            return

        max_err_f = -1.0
        f = -1
        for neighbor_id in self.edges_per_node.get(q, set()):
            if self.nodes[neighbor_id].error > max_err_f:
                max_err_f = self.nodes[neighbor_id].error
                f = neighbor_id

        if f == -1:
            return

        # Add new node r between q and f
        new_pos = 0.5 * (self.nodes[q].position + self.nodes[f].position)
        new_color = 0.5 * (self.nodes[q].color + self.nodes[f].color)
        r = self._add_node(new_pos, new_color)

        if r == -1:
            return

        # Initialize normal (normalized average)
        new_normal = 0.5 * (self.nodes[q].normal + self.nodes[f].normal)
        norm = np.linalg.norm(new_normal)
        if norm > 1e-10:
            self.nodes[r].normal = new_normal / norm
        else:
            self.nodes[r].normal = np.array([0.0, 0.0, 1.0])

        # Inherit robot properties
        self.nodes[r].through_property = self.nodes[q].through_property
        self.nodes[r].dimension_property = self.nodes[q].dimension_property
        self.nodes[r].traversability_property = self.nodes[q].traversability_property

        # Update edges
        self.edges_pos[q, f] = 0
        self.edges_pos[f, q] = 0
        self.edges_per_node[q].discard(f)
        self.edges_per_node[f].discard(q)

        # Inherit cedge
        self.edges_color[q, r] = self.edges_color[q, f]
        self.edges_color[r, q] = self.edges_color[q, f]
        self.edges_color[f, r] = self.edges_color[q, f]
        self.edges_color[r, f] = self.edges_color[q, f]
        self.edges_color[q, f] = 0
        self.edges_color[f, q] = 0

        # Inherit nedge
        self.edges_normal[q, r] = self.edges_normal[q, f]
        self.edges_normal[r, q] = self.edges_normal[q, f]
        self.edges_normal[f, r] = self.edges_normal[q, f]
        self.edges_normal[r, f] = self.edges_normal[q, f]
        self.edges_normal[q, f] = 0
        self.edges_normal[f, q] = 0

        # Inherit pedge
        self.edges_traversability[q, r] = self.edges_traversability[q, f]
        self.edges_traversability[r, q] = self.edges_traversability[q, f]
        self.edges_traversability[f, r] = self.edges_traversability[q, f]
        self.edges_traversability[r, f] = self.edges_traversability[q, f]
        self.edges_traversability[q, f] = 0
        self.edges_traversability[f, q] = 0

        # Add position edges
        self._add_position_edge(q, r)
        self._add_position_edge(r, f)

        # Update errors
        self.nodes[q].error *= 0.5
        self.nodes[f].error *= 0.5
        self.nodes[q].utility *= 0.5
        self.nodes[f].utility *= 0.5
        self.nodes[r].error = self.nodes[q].error
        self.nodes[r].utility = self.nodes[q].utility

        # Utility-based deletion
        if self.n_nodes > 10 and min_err < p.thv:
            for del_id in delete_list:
                if self.nodes[del_id].id != -1 and del_id != r:
                    self._remove_node(del_id)

    def _one_train_update(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
    ) -> float:
        """Single training iteration."""
        p = self.params

        s1, s2, dist1_sq, dist2_sq = self._find_two_nearest(position)

        if s1 == -1 or s2 == -1:
            return 0.0

        if dist1_sq > p.dis_thv * p.dis_thv and self.n_nodes < p.max_nodes - 2:
            self._add_new_node_distance(position, color)
            self._discount_errors()
            return 0.0

        self.nodes[s1].error += dist1_sq
        self.nodes[s1].utility += dist2_sq - dist1_sq

        self._gng_learn(s1, s2, position, color, p.eps_b, p.eps_n)
        self._discount_errors()

        self.n_learning += 1
        return dist1_sq

    def _gng_main_cycle(
        self,
        data: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        """Run one gng_main cycle."""
        p = self.params
        n_samples = len(data)
        total_error = 0.0

        for i in range(p.lambda_):
            idx = self.rng.integers(0, n_samples)
            color = colors[idx] if colors is not None else None

            if i == p.lambda_ // 2:
                error = self._one_train_update(data[idx], color)
                total_error += error
                if self.n_nodes > 2:
                    self._delete_node_gngu()
            else:
                error = self._one_train_update(data[idx], color)
                total_error += error

        total_error /= p.lambda_
        if self.n_nodes < p.max_nodes and total_error > p.thv:
            self._node_add()

    def train(
        self,
        data: np.ndarray,
        colors: np.ndarray | None = None,
        n_iterations: int = 1000,
        callback: Callable[[GrowingNeuralGasDTRobot, int], None] | None = None,
    ) -> GrowingNeuralGasDTRobot:
        """Train on data for multiple iterations."""
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
    ) -> GrowingNeuralGasDTRobot:
        """Single online learning step."""
        p = self.params

        error = self._one_train_update(position, color)
        self._total_error += error
        self._n_trial += 1

        if self._n_trial == p.lambda_ // 2:
            if self.n_nodes > 2:
                self._delete_node_gngu()

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
    def n_edges_traversability(self) -> int:
        """Number of traversability edges."""
        count = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].id == -1:
                continue
            for j in range(i + 1, len(self.nodes)):
                if self.nodes[j].id == -1:
                    continue
                if self.edges_traversability[i, j] > 0:
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
        list[tuple[int, int]],
    ]:
        """Get current graph structure with all topologies.

        Returns:
            Tuple of (nodes, pos_edges, color_edges, normal_edges, traversability_edges).
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

        # Traversability edges
        traversability_edges = []
        for i in range(len(self.nodes)):
            if self.nodes[i].id == -1:
                continue
            for j in range(i + 1, len(self.nodes)):
                if self.nodes[j].id == -1:
                    continue
                if self.edges_traversability[i, j] > 0:
                    traversability_edges.append((index_map[i], index_map[j]))

        return nodes, pos_edges, color_edges, normal_edges, traversability_edges

    def get_node_normals(self) -> np.ndarray:
        """Get normal vectors for active nodes."""
        return np.array([node.normal for node in self.nodes if node.id != -1])

    def get_node_colors(self) -> np.ndarray:
        """Get color vectors for active nodes."""
        return np.array([node.color for node in self.nodes if node.id != -1])

    def get_traversability(self) -> np.ndarray:
        """Get traversability property for active nodes."""
        return np.array(
            [node.traversability_property for node in self.nodes if node.id != -1]
        )

    def get_contour(self) -> np.ndarray:
        """Get contour property for active nodes."""
        return np.array([node.contour for node in self.nodes if node.id != -1])

    def get_degree(self) -> np.ndarray:
        """Get degree (inclination cost) for active nodes."""
        return np.array([node.degree for node in self.nodes if node.id != -1])

    def get_curvature(self) -> np.ndarray:
        """Get curvature cost for active nodes."""
        return np.array([node.curvature for node in self.nodes if node.id != -1])

    def get_traversable_nodes(self) -> np.ndarray:
        """Get positions of traversable nodes only."""
        return np.array(
            [
                node.position
                for node in self.nodes
                if node.id != -1 and node.traversability_property == 1
            ]
        )

    def get_contour_nodes(self) -> np.ndarray:
        """Get positions of contour nodes only."""
        return np.array(
            [node.position for node in self.nodes if node.id != -1 and node.contour == 1]
        )


# Aliases
GNGDTRobot = GrowingNeuralGasDTRobot
