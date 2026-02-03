"""DD-GNG (Dynamic Density Growing Neural Gas) implementation.

Based on:
    - Saputra, A.A., et al. (2019). "Dynamic Density Topological Structure
      Generation for Real-Time Ladder Affordance Detection"
    - IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2019
    - Reference implementation: azhar_ddgng

DD-GNG extends GNG-U with dynamic density control:
    1. Node strength: Nodes in attention regions have higher strength values
    2. Strength-weighted node insertion: error * strength^4 for priority
    3. Strength-weighted learning: (1/strength) * eps_b for stability
    4. Dynamic sampling: Optional priority sampling from attention regions
    5. Automatic attention detection: Stable corners detected via surface
       classification are automatically treated as attention regions

This enables higher node density in regions of interest (e.g., obstacles,
object details) while maintaining lower density in safe/uninteresting areas.

Surface Classification (from reference implementation):
    - PLANE (0): Node on a flat surface (smallest eigenvalue very small)
    - EDGE (1): Node on an edge between surfaces
    - CORNER (2): Node at a corner (all eigenvalues similar)
    - STABLE_PLANE (4): Plane stable for > stability_threshold iterations
    - STABLE_EDGE (5): Edge stable for > stability_threshold iterations
    - STABLE_CORNER (6): Corner stable for > stability_threshold iterations
                         (automatically added to attention regions)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import numpy as np


class SurfaceType(IntEnum):
    """Surface classification types from reference implementation.

    Based on calc_node_normal_vector() in gng.cpp.
    """

    UNKNOWN = 3  # Not yet classified
    PLANE = 0  # Flat surface
    EDGE = 1  # Edge between surfaces
    CORNER = 2  # Corner point
    STABLE_PLANE = 4  # Plane stable for > stability_threshold
    STABLE_EDGE = 5  # Edge stable for > stability_threshold
    STABLE_CORNER = 6  # Corner stable for > stability_threshold (auto attention)


@dataclass
class DDGNGParams:
    """DD-GNG hyperparameters.

    Parameters follow the naming from Saputra et al. (2019) paper.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node (η1 in paper).
        eps_n: Learning rate for neighbor nodes (η2 in paper).
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        chi: Utility decay rate (χ in paper).
        max_age: Maximum edge age before removal (g_max in paper).
        utility_k: Utility threshold for node removal (k in paper).
        kappa: Utility check interval (κ in paper).
        strength_power: Exponent for strength weighting (default: 4).
        strength_scale: Scale factor for strength in weighted error (default: 4.0).
        use_strength_learning: Apply strength to learning rate (default: True).
        use_strength_insertion: Apply strength to node insertion (default: True).
        auto_detect_attention: Enable automatic attention region detection based
            on surface classification (default: False for backward compatibility).
        stability_threshold: Number of iterations a node must maintain its
            surface type to be considered stable (default: 16, from reference).
        plane_stability_threshold: Stability threshold for planes (default: 8).
        corner_strength: Strength bonus for auto-detected stable corners
            (default: 5.0, reference uses 20).
        plane_ev_ratio: Eigenvalue ratio threshold for plane classification.
            If smallest_ev / largest_ev < ratio, classify as plane (default: 0.01).
        edge_ev_ratio: Eigenvalue ratio threshold for edge classification.
            If middle_ev / largest_ev < ratio, classify as edge (default: 0.1).
        surface_update_interval: How often to update surface classification
            (default: 10, every 10 iterations).
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
    # DD-GNG specific parameters
    strength_power: int = 4  # Exponent for strength weighting
    strength_scale: float = 4.0  # Scale factor for strength
    use_strength_learning: bool = True  # Apply strength to learning rate
    use_strength_insertion: bool = True  # Apply strength to node insertion
    # Auto-detection parameters (from reference implementation)
    auto_detect_attention: bool = False  # Enable automatic detection
    stability_threshold: int = 16  # Iterations for corner/edge stability
    plane_stability_threshold: int = 8  # Iterations for plane stability
    corner_strength: float = 5.0  # Strength for auto-detected corners
    plane_ev_ratio: float = 0.01  # Eigenvalue ratio for plane
    edge_ev_ratio: float = 0.1  # Eigenvalue ratio for edge
    surface_update_interval: int = 10  # Surface classification update interval


@dataclass
class AttentionRegion:
    """Defines an attention region for dynamic density control.

    Attributes:
        center: Center position of the region.
        size: Size (half-extent) of the region in each dimension.
        strength_bonus: Additional strength for nodes in this region.
    """

    center: np.ndarray
    size: np.ndarray
    strength_bonus: float = 1.0

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is within this region."""
        return np.all(np.abs(point - self.center) <= self.size)


@dataclass
class NeuronNode:
    """A neuron node in the DD-GNG network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector (reference vector h).
        error: Accumulated error (E).
        utility: Utility measure (U).
        strength: Node strength for dynamic density control (δ).
        normal: Normal vector at this node (computed via PCA).
        surface_type: Surface classification (plane/edge/corner).
        stability_age: How long node has maintained current surface type.
        auto_attention: Whether this node is auto-detected as attention region.
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0
    utility: float = 0.0
    strength: float = 1.0
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    surface_type: int = SurfaceType.UNKNOWN
    stability_age: int = 0
    auto_attention: bool = False


class DynamicDensityGNG:
    """DD-GNG (Dynamic Density Growing Neural Gas) implementation.

    DD-GNG extends GNG-U with dynamic density control. Nodes in attention
    regions (e.g., obstacles, objects of interest) have higher strength
    values, which causes:
    - Higher node density through strength-weighted insertion
    - More stable positioning through strength-weighted learning

    Key features:
    1. Manual attention regions: Define regions where higher node density is desired
    2. Automatic attention detection: Stable corners detected via surface
       classification are automatically treated as attention regions
    3. Node strength: δ = 1 + sum(strength_bonus for containing regions)
       + corner_strength (if auto_attention)
    4. Weighted insertion: priority = error * (scale * strength)^power
    5. Weighted learning: effective_eps_b = eps_b / strength

    Surface Classification (3D only, requires n_dim >= 3):
    - Computes normal vectors via PCA on node + neighbors
    - Classifies nodes as plane/edge/corner based on eigenvalue ratios
    - Tracks stability over time (how long node maintains same classification)
    - Stable corners (stability_age > threshold) automatically become attention

    Attributes:
        params: DD-GNG hyperparameters.
        nodes: List of neuron nodes.
        edges: Edge age matrix.
        edges_per_node: Adjacency list for quick neighbor lookup.
        attention_regions: List of manual attention regions.
        n_learning: Total number of learning iterations.
        n_removals: Number of nodes removed by utility criterion.
        n_auto_attention: Number of auto-detected attention nodes.

    Examples
    --------
    >>> import numpy as np
    >>> from model import DynamicDensityGNG, DDGNGParams
    >>> # Manual attention region (2D/3D)
    >>> X = np.random.rand(1000, 2)
    >>> params = DDGNGParams(max_nodes=100)
    >>> gng = DynamicDensityGNG(n_dim=2, params=params)
    >>> gng.add_attention_region(center=[0.75, 0.75], size=[0.25, 0.25], strength=5.0)
    >>> gng.train(X, n_iterations=5000)
    >>>
    >>> # Automatic attention detection (3D)
    >>> X3d = np.random.rand(1000, 3)
    >>> params = DDGNGParams(max_nodes=100, auto_detect_attention=True)
    >>> gng3d = DynamicDensityGNG(n_dim=3, params=params)
    >>> gng3d.train(X3d, n_iterations=5000)
    >>> # Stable corners are automatically detected as attention regions
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: DDGNGParams | None = None,
        seed: int | None = None,
    ):
        """Initialize DD-GNG.

        Args:
            n_dim: Dimension of input data.
            params: DD-GNG hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or DDGNGParams()
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

        # Attention regions for dynamic density control
        self.attention_regions: list[AttentionRegion] = []

        # Counters
        self.n_learning = 0
        self._n_trial = 0
        self.n_removals = 0
        self.n_auto_attention = 0

        # Initialize with 2 random nodes
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes."""
        for _ in range(2):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            self._add_node(weight)

    def add_attention_region(
        self,
        center: np.ndarray | list,
        size: np.ndarray | list,
        strength: float = 1.0,
    ) -> None:
        """Add an attention region for dynamic density control.

        Nodes within this region will have higher strength values,
        resulting in higher node density.

        Args:
            center: Center position of the region.
            size: Size (half-extent) of the region in each dimension.
            strength: Additional strength bonus for nodes in this region.
        """
        region = AttentionRegion(
            center=np.asarray(center, dtype=np.float32),
            size=np.asarray(size, dtype=np.float32),
            strength_bonus=strength,
        )
        self.attention_regions.append(region)

    def clear_attention_regions(self) -> None:
        """Remove all attention regions."""
        self.attention_regions.clear()

    def _calculate_strength(
        self, position: np.ndarray, auto_attention: bool = False
    ) -> float:
        """Calculate node strength based on position and attention regions.

        Strength formula (from paper Algorithm 2):
            δ = 1 + sum(strength_bonus for each containing region)
            + corner_strength (if auto_attention is True)

        Args:
            position: Node position.
            auto_attention: Whether this node is auto-detected as attention.

        Returns:
            Strength value (δ >= 1.0).
        """
        p = self.params
        strength = 1.0

        # Manual attention regions
        for region in self.attention_regions:
            if region.contains(position):
                strength += region.strength_bonus

        # Auto-detected attention (stable corners)
        if auto_attention:
            strength += p.corner_strength

        return strength

    def _update_node_strength(self, node_id: int) -> None:
        """Update strength for a node based on its current position."""
        if self.nodes[node_id].id == -1:
            return
        node = self.nodes[node_id]
        node.strength = self._calculate_strength(node.weight, node.auto_attention)

    def _update_all_strengths(self) -> None:
        """Update strength values for all active nodes."""
        for node in self.nodes:
            if node.id != -1:
                node.strength = self._calculate_strength(
                    node.weight, node.auto_attention
                )

    # =========================================================================
    # Surface Classification (Auto-detection feature)
    # =========================================================================

    def _compute_normal_pca(self, node_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute normal vector via PCA on node + neighbor positions.

        Based on calc_node_normal_vector() in reference gng.cpp.
        Uses eigenvalue decomposition of covariance matrix:
        - Smallest eigenvalue direction = normal vector
        - Eigenvalue ratios used for surface classification

        Args:
            node_id: Node ID to compute normal for.

        Returns:
            Tuple of (normal_vector, eigenvalues_sorted).
            normal_vector: Unit normal vector (or [0,0,1] default).
            eigenvalues_sorted: Eigenvalues sorted ascending [λ1, λ2, λ3].
        """
        node = self.nodes[node_id]
        default_normal = np.zeros(self.n_dim)
        if self.n_dim >= 3:
            default_normal[2] = 1.0  # Default Z-up normal
        else:
            default_normal[-1] = 1.0

        # Collect positions: this node + all neighbors
        positions = [node.weight.copy()]
        for neighbor_id in self.edges_per_node.get(node_id, set()):
            if self.nodes[neighbor_id].id != -1:
                positions.append(self.nodes[neighbor_id].weight.copy())

        if len(positions) < 3:
            # Need at least 3 points for meaningful PCA
            return default_normal, np.ones(self.n_dim)

        positions_arr = np.array(positions)

        # Compute centroid
        centroid = np.mean(positions_arr, axis=0)

        # Compute covariance matrix
        centered = positions_arr - centroid
        cov = np.dot(centered.T, centered) / len(positions)

        try:
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # eigenvalues are sorted ascending by eigh
            # Smallest eigenvalue = normal direction
            normal = eigenvectors[:, 0].copy()

            # Ensure consistent orientation (reference: if y < 0, flip)
            if self.n_dim >= 2 and normal[1] < 0:
                normal = -normal

            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            else:
                normal = default_normal

            return normal, eigenvalues

        except np.linalg.LinAlgError:
            return default_normal, np.ones(self.n_dim)

    def _classify_surface_type(self, eigenvalues: np.ndarray) -> int:
        """Classify surface type based on eigenvalue ratios.

        Based on reference implementation calc_node_normal_vector():
        - Plane: Smallest eigenvalue << others (flat surface)
        - Edge: Two large eigenvalues, one small (linear feature)
        - Corner: All eigenvalues similar (3D point)

        Args:
            eigenvalues: Eigenvalues sorted ascending [λ1, λ2, λ3].

        Returns:
            SurfaceType value (PLANE, EDGE, or CORNER).
        """
        p = self.params

        # Need at least 3D for proper classification
        if len(eigenvalues) < 3:
            return SurfaceType.CORNER  # 2D points are all "corners"

        # Avoid division by zero
        max_ev = max(eigenvalues[-1], 1e-10)

        # Ratios: how small are the eigenvalues relative to the largest
        ratio_small = eigenvalues[0] / max_ev  # smallest / largest
        ratio_mid = eigenvalues[1] / max_ev  # middle / largest

        if ratio_small < p.plane_ev_ratio:
            # Smallest eigenvalue is very small -> PLANE
            return SurfaceType.PLANE
        elif ratio_mid < p.edge_ev_ratio:
            # Middle eigenvalue is small -> EDGE (linear feature)
            return SurfaceType.EDGE
        else:
            # All eigenvalues similar -> CORNER
            return SurfaceType.CORNER

    def _update_surface_classification(self, node_id: int) -> None:
        """Update surface classification for a single node.

        Based on reference calc_node_normal_vector() logic:
        1. Compute normal via PCA
        2. Classify as plane/edge/corner
        3. Track stability (how long same classification maintained)
        4. Promote to stable variant after threshold iterations
        5. Auto-detect attention for stable corners

        Args:
            node_id: Node ID to update.
        """
        p = self.params
        node = self.nodes[node_id]

        if node.id == -1:
            return

        # Remember previous classification
        last_type = node.surface_type

        # Compute normal and classify
        normal, eigenvalues = self._compute_normal_pca(node_id)
        node.normal = normal
        new_type = self._classify_surface_type(eigenvalues)

        # Update stability tracking (from reference gng.cpp:523-548)
        if last_type == SurfaceType.STABLE_CORNER and new_type != SurfaceType.CORNER:
            # Was stable corner, now not a corner -> revert to corner, decrement age
            node.surface_type = SurfaceType.CORNER
            node.stability_age = max(0, node.stability_age - 1)
            # Lost auto-attention status
            if node.auto_attention:
                node.auto_attention = False
                self.n_auto_attention -= 1

        elif new_type != last_type and last_type in (
            SurfaceType.PLANE,
            SurfaceType.EDGE,
            SurfaceType.CORNER,
            SurfaceType.UNKNOWN,
        ):
            # Classification changed -> reset age
            node.surface_type = new_type
            node.stability_age = 0

        elif new_type in (SurfaceType.PLANE, SurfaceType.EDGE, SurfaceType.CORNER):
            # Same classification -> increment age
            node.surface_type = new_type
            node.stability_age = min(25, node.stability_age + 1)

            # Check for promotion to stable variant
            if (
                new_type == SurfaceType.PLANE
                and node.stability_age > p.plane_stability_threshold
            ):
                node.surface_type = SurfaceType.STABLE_PLANE

            elif (
                new_type == SurfaceType.EDGE
                and node.stability_age > p.stability_threshold
            ):
                node.surface_type = SurfaceType.STABLE_EDGE

            elif (
                new_type == SurfaceType.CORNER
                and node.stability_age > p.stability_threshold
            ):
                # Promote to stable corner -> AUTO ATTENTION!
                node.surface_type = SurfaceType.STABLE_CORNER
                if not node.auto_attention:
                    node.auto_attention = True
                    self.n_auto_attention += 1

        else:
            # Already stable -> just decrement age slightly
            node.stability_age = max(0, node.stability_age - 1)

    def _update_all_surface_classifications(self) -> None:
        """Update surface classifications for all active nodes.

        Should be called periodically (every surface_update_interval iterations).
        """
        if not self.params.auto_detect_attention:
            return

        if self.n_dim < 3:
            # Surface classification requires 3D
            return

        for node in self.nodes:
            if node.id != -1:
                self._update_surface_classification(node.id)

        # Update strengths after classification changes
        self._update_all_strengths()

    def _add_node(
        self,
        weight: np.ndarray,
        error: float = 1.0,
        utility: float = 0.0,
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
        strength = self._calculate_strength(weight)
        self.nodes[node_id] = NeuronNode(
            id=node_id,
            weight=weight.copy(),
            error=error,
            utility=utility,
            strength=strength,
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

        # Track auto-attention removal
        if self.nodes[node_id].auto_attention:
            self.n_auto_attention -= 1

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

        # Return Euclidean distances
        dist1 = np.sqrt(min_dist1_sq) if min_dist1_sq < float("inf") else float("inf")
        dist2 = np.sqrt(min_dist2_sq) if min_dist2_sq < float("inf") else float("inf")

        return s1_id, s2_id, dist1, dist2

    def _check_utility_criterion(self) -> None:
        """Check and remove node with lowest utility if criterion met.

        GNG-U2 Algorithm: Every κ iterations, check if E_u / U_l > k.
        DD-GNG modification: Strength is considered in the criterion.
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
                    self.n_removals += 1

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration with DD-GNG algorithm.

        Key DD-GNG features:
        1. Strength-weighted learning: effective_eps_b = eps_b / strength
        2. Utility check at κ-interval

        Args:
            sample: Input sample vector (v_t).
        """
        p = self.params

        # Find two nearest nodes
        s1_id, s2_id, dist1, dist2 = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        # Connect s1 and s2
        self._add_edge(s1_id, s2_id)

        # Update error using Euclidean distance
        self.nodes[s1_id].error += dist1

        # Update utility using Euclidean distance difference
        self.nodes[s1_id].utility += dist2 - dist1

        # Update strength for winner
        self._update_node_strength(s1_id)
        strength = self.nodes[s1_id].strength

        # DD-GNG: Strength-weighted learning
        # Higher strength = slower learning = more stable position
        if p.use_strength_learning:
            effective_eps_b = p.eps_b / strength
        else:
            effective_eps_b = p.eps_b

        # Move winner toward sample
        self.nodes[s1_id].weight += effective_eps_b * (
            sample - self.nodes[s1_id].weight
        )

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
            node.utility -= p.chi * node.utility

        # Periodically add new node
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            self._insert_node_with_density()

        # Auto-detection: Update surface classifications periodically
        if (
            p.auto_detect_attention
            and self.n_learning > 0
            and self.n_learning % p.surface_update_interval == 0
        ):
            self._update_all_surface_classifications()

        self.n_learning += 1

    def _insert_node_with_density(self) -> int:
        """Insert new node with DD-GNG density-weighted priority.

        DD-GNG Algorithm (from paper node_add_gng2):
        - Priority = error * (scale * strength)^power
        - Higher strength nodes have higher insertion priority
        - This creates higher node density in attention regions

        Returns:
            ID of new node, or -1 if insertion failed.
        """
        p = self.params

        if not self._addable_indices:
            return -1

        # Update all node strengths before selection
        self._update_all_strengths()

        # Find node with maximum weighted error
        max_weighted_err = 0.0
        u_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            # DD-GNG: weight error by strength^power
            if p.use_strength_insertion:
                weighted_err = node.error * (
                    (p.strength_scale * node.strength) ** p.strength_power
                )
            else:
                weighted_err = node.error

            if weighted_err > max_weighted_err:
                max_weighted_err = weighted_err
                u_id = node.id

        if u_id == -1:
            return -1

        # Find neighbor of u with maximum weighted error
        max_weighted_err_f = 0.0
        f_id = -1

        for neighbor_id in self.edges_per_node.get(u_id, set()):
            node = self.nodes[neighbor_id]
            if p.use_strength_insertion:
                weighted_err = node.error * (
                    (p.strength_scale * node.strength) ** p.strength_power
                )
            else:
                weighted_err = node.error

            if weighted_err > max_weighted_err_f:
                max_weighted_err_f = weighted_err
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
        callback: Callable[[DynamicDensityGNG, int], None] | None = None,
    ) -> DynamicDensityGNG:
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

    def train_with_density_sampling(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        attention_sampling_ratio: float = 0.5,
        callback: Callable[[DynamicDensityGNG, int], None] | None = None,
    ) -> DynamicDensityGNG:
        """Train with priority sampling from attention regions.

        DD-GNG feature: Samples are drawn more frequently from attention
        regions, which accelerates learning in those areas.

        Args:
            data: Training data of shape (n_samples, n_dim).
            n_iterations: Number of training iterations.
            attention_sampling_ratio: Fraction of samples to draw from
                attention regions (0.0 to 1.0).
            callback: Optional callback(self, iteration) called each iteration.

        Returns:
            self for chaining.
        """
        if not self.attention_regions or attention_sampling_ratio <= 0:
            return self.train(data, n_iterations, callback)

        n_samples = len(data)

        # Pre-compute which samples are in attention regions
        attention_indices = []
        other_indices = []

        for i in range(n_samples):
            in_attention = any(
                region.contains(data[i]) for region in self.attention_regions
            )
            if in_attention:
                attention_indices.append(i)
            else:
                other_indices.append(i)

        attention_indices = np.array(attention_indices)
        other_indices = np.array(other_indices)

        for i in range(n_iterations):
            # Decide whether to sample from attention region
            if (
                len(attention_indices) > 0
                and self.rng.random() < attention_sampling_ratio
            ):
                idx = self.rng.choice(attention_indices)
            elif len(other_indices) > 0:
                idx = self.rng.choice(other_indices)
            else:
                idx = self.rng.integers(0, n_samples)

            self._one_train_update(data[idx])

            if callback is not None:
                callback(self, i)

        return self

    def partial_fit(self, sample: np.ndarray) -> DynamicDensityGNG:
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

    def get_node_strengths(self) -> np.ndarray:
        """Get strength values for active nodes.

        Returns:
            Array of strength values in same order as get_graph() nodes.
        """
        return np.array([node.strength for node in self.nodes if node.id != -1])

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

    def get_node_normals(self) -> np.ndarray:
        """Get normal vectors for active nodes.

        Returns:
            Array of normal vectors in same order as get_graph() nodes.
            Shape: (n_nodes, n_dim).
        """
        normals = [node.normal for node in self.nodes if node.id != -1]
        if not normals:
            return np.array([]).reshape(0, self.n_dim)
        return np.array(normals)

    def get_node_surface_types(self) -> np.ndarray:
        """Get surface type classifications for active nodes.

        Returns:
            Array of SurfaceType values in same order as get_graph() nodes.
            Values: 0=PLANE, 1=EDGE, 2=CORNER, 3=UNKNOWN,
                   4=STABLE_PLANE, 5=STABLE_EDGE, 6=STABLE_CORNER
        """
        return np.array(
            [node.surface_type for node in self.nodes if node.id != -1],
            dtype=np.int32,
        )

    def get_node_auto_attention(self) -> np.ndarray:
        """Get auto-attention status for active nodes.

        Returns:
            Boolean array indicating which nodes are auto-detected attention.
        """
        return np.array(
            [node.auto_attention for node in self.nodes if node.id != -1],
            dtype=bool,
        )

    def get_auto_attention_nodes(self) -> np.ndarray:
        """Get positions of auto-detected attention nodes (stable corners).

        Returns:
            Array of positions for auto-attention nodes.
            Shape: (n_auto_attention, n_dim).
        """
        positions = [
            node.weight for node in self.nodes
            if node.id != -1 and node.auto_attention
        ]
        if not positions:
            return np.array([]).reshape(0, self.n_dim)
        return np.array(positions)


# Aliases
DDGNG = DynamicDensityGNG
