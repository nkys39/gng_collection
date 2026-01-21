"""Growing Neural Gas with Triangulation (GNG-T).

Based on:
    - Kubota, N. & Satomi, M. (2008). "Growing Neural Gas with Triangulation
      for reconstructing a 3D surface model"
    - World Automation Congress (2008)

GNG-T extends GNG with heuristic triangulation to maintain proper triangle
mesh structure. Unlike GNG-D (which uses scipy.spatial.Delaunay), GNG-T
uses quadrilateral search and intersection search heuristics.

Key differences from standard GNG:
- Initialization with 3 nodes (2D simplex) like GCS
- After node/edge changes, performs triangulation search:
  1. Quadrilateral search: Find quadrilaterals without diagonals, add shorter diagonal
  2. Intersection search: Detect edge crossings, remove longer edge
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GNGTParams:
    """GNG-T hyperparameters.

    Attributes:
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for neighbor nodes.
        alpha: Error decay rate when splitting.
        beta: Global error decay rate.
        max_age: Maximum edge age before removal.
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
    """A neuron node in the GNG-T network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector.
        error: Accumulated error.
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 1.0


class GrowingNeuralGasT:
    """Growing Neural Gas with Triangulation (GNG-T).

    This implementation follows Kubota & Satomi (2008) which adds heuristic
    triangulation (quadrilateral and intersection search) to standard GNG.

    Attributes:
        params: GNG-T hyperparameters.
        nodes: List of neuron nodes.
        edges: Edge age matrix (0 = no edge, >=1 = connected with age).
        edges_per_node: Adjacency list for quick neighbor lookup.
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GrowingNeuralGasT
    >>> X = np.random.rand(1000, 2)
    >>> gng_t = GrowingNeuralGasT(n_dim=2)
    >>> gng_t.train(X, n_iterations=5000)
    >>> nodes, edges = gng_t.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GNGTParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG-T.

        Args:
            n_dim: Dimension of input data (must be >= 2).
            params: GNG-T hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        if n_dim < 2:
            raise ValueError("GNG-T requires n_dim >= 2")

        self.n_dim = n_dim
        self.params = params or GNGTParams()
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

        # Initialize with 3 random nodes (2D simplex like GCS)
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 3 random nodes forming a 2D simplex."""
        node_ids = []
        for _ in range(3):
            weight = self.rng.random(self.n_dim).astype(np.float32)
            node_id = self._add_node(weight)
            node_ids.append(node_id)

        # Connect all three nodes to form a triangle
        for i in range(3):
            for j in range(i + 1, 3):
                self._add_edge(node_ids[i], node_ids[j])

    def _add_node(self, weight: np.ndarray) -> int:
        """Add a new node with given weight.

        Args:
            weight: Position vector for the new node.

        Returns:
            ID of the new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1  # No space

        node_id = self._addable_indices.popleft()
        self.nodes[node_id] = NeuronNode(id=node_id, weight=weight.copy(), error=1.0)
        self.edges_per_node[node_id] = set()
        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node (only if isolated).

        Args:
            node_id: ID of node to remove.
        """
        if self.edges_per_node.get(node_id):
            return  # Has edges, don't remove

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _add_edge(self, node1: int, node2: int) -> None:
        """Add or reset edge between two nodes.

        Args:
            node1: First node ID.
            node2: Second node ID.
        """
        if self.edges[node1, node2] > 0:
            # Edge exists, reset age
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1
        else:
            # New edge
            self.edges_per_node[node1].add(node2)
            self.edges_per_node[node2].add(node1)
            self.edges[node1, node2] = 1
            self.edges[node2, node1] = 1

    def _remove_edge(self, node1: int, node2: int) -> None:
        """Remove edge between two nodes.

        Args:
            node1: First node ID.
            node2: Second node ID.
        """
        self.edges_per_node[node1].discard(node2)
        self.edges_per_node[node2].discard(node1)
        self.edges[node1, node2] = 0
        self.edges[node2, node1] = 0

    def _has_edge(self, node1: int, node2: int) -> bool:
        """Check if edge exists between two nodes."""
        return self.edges[node1, node2] > 0

    def _get_active_node_ids(self) -> list[int]:
        """Get list of active node IDs."""
        return [node.id for node in self.nodes if node.id != -1]

    def _find_two_nearest(self, x: np.ndarray) -> tuple[int, int]:
        """Find the two nearest nodes to input x.

        Args:
            x: Input vector.

        Returns:
            Tuple of (winner_id, second_winner_id).
        """
        min_dist1 = float("inf")
        min_dist2 = float("inf")
        s1_id = -1
        s2_id = -1

        for node in self.nodes:
            if node.id == -1:
                continue

            dist = np.sum((x - node.weight) ** 2)  # Squared distance

            if dist < min_dist1:
                min_dist2 = min_dist1
                s2_id = s1_id
                min_dist1 = dist
                s1_id = node.id
            elif dist < min_dist2:
                min_dist2 = dist
                s2_id = node.id

        return s1_id, s2_id

    def _edge_length_sq(self, node1: int, node2: int) -> float:
        """Calculate squared distance between two nodes."""
        return float(np.sum(
            (self.nodes[node1].weight - self.nodes[node2].weight) ** 2
        ))

    def _quadrilateral_search(self) -> None:
        """Perform quadrilateral search to add missing diagonals.

        Find quadrilaterals A-B-D-C without diagonals and add the shorter one.
        Based on Section 2.5.1 of Kubota & Satomi (2008).
        """
        active_ids = self._get_active_node_ids()

        for a_id in active_ids:
            neighbors_a = self.edges_per_node.get(a_id, set())
            neighbors_a_list = list(neighbors_a)

            # Find pairs of neighbors (B, C) not connected to each other
            for i, b_id in enumerate(neighbors_a_list):
                for c_id in neighbors_a_list[i + 1:]:
                    if self._has_edge(b_id, c_id):
                        continue  # B-C already connected

                    # Find common neighbor D of B and C (not A)
                    neighbors_b = self.edges_per_node.get(b_id, set())
                    neighbors_c = self.edges_per_node.get(c_id, set())
                    common = neighbors_b & neighbors_c - {a_id}

                    for d_id in common:
                        if self._has_edge(a_id, d_id):
                            continue  # A-D already connected

                        # A-B-D-C forms a quadrilateral candidate
                        # Verify no other node inside (check all active nodes)
                        is_valid = True
                        for e_id in active_ids:
                            if e_id in (a_id, b_id, c_id, d_id):
                                continue
                            # Check if E is connected to all of A, B, C, D
                            neighbors_e = self.edges_per_node.get(e_id, set())
                            if {a_id, b_id, c_id, d_id} <= neighbors_e:
                                is_valid = False
                                break

                        if not is_valid:
                            continue

                        # Add shorter diagonal
                        dist_ad = self._edge_length_sq(a_id, d_id)
                        dist_bc = self._edge_length_sq(b_id, c_id)

                        if dist_ad < dist_bc:
                            self._add_edge(a_id, d_id)
                        else:
                            self._add_edge(b_id, c_id)

    def _edges_intersect(
        self, a_id: int, b_id: int, c_id: int, d_id: int
    ) -> bool:
        """Check if edge A-B intersects with edge C-D.

        Uses the formula from Section 2.5.2 of Kubota & Satomi (2008):
        γ1 = (xC - xE)(yD - yC) + (yC - yE)(xC - xD)
        γ2 = (xC - xE)(yB - yC) + (yC - yE)(xC - xB)
        If γ1 * γ2 <= 0, edges intersect.

        Note: In the paper, the formula checks if line BD and CE intersect.
        We adapt it to check A-B and C-D intersection.
        """
        # Get positions
        pa = self.nodes[a_id].weight
        pb = self.nodes[b_id].weight
        pc = self.nodes[c_id].weight
        pd = self.nodes[d_id].weight

        # Check if segments AB and CD intersect using cross product method
        def ccw(p1, p2, p3):
            """Check if three points are counterclockwise."""
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

        # Segments intersect if A and B are on opposite sides of CD,
        # and C and D are on opposite sides of AB
        return (
            ccw(pa, pc, pd) != ccw(pb, pc, pd) and
            ccw(pa, pb, pc) != ccw(pa, pb, pd)
        )

    def _intersection_search(self) -> None:
        """Perform intersection search to remove crossing edges.

        Check for edge intersections and remove the longer edge.
        Based on Section 2.5.2 of Kubota & Satomi (2008).
        """
        # Collect all edges
        all_edges = []
        seen = set()
        for node_id, neighbors in self.edges_per_node.items():
            if self.nodes[node_id].id == -1:
                continue
            for neighbor_id in neighbors:
                edge = tuple(sorted([node_id, neighbor_id]))
                if edge not in seen:
                    seen.add(edge)
                    all_edges.append(edge)

        # Check all pairs of edges for intersection
        edges_to_remove = set()
        for i, (a_id, b_id) in enumerate(all_edges):
            if (a_id, b_id) in edges_to_remove or (b_id, a_id) in edges_to_remove:
                continue

            for c_id, d_id in all_edges[i + 1:]:
                if (c_id, d_id) in edges_to_remove or (d_id, c_id) in edges_to_remove:
                    continue

                # Skip if edges share a vertex
                if a_id in (c_id, d_id) or b_id in (c_id, d_id):
                    continue

                # Check intersection
                if self._edges_intersect(a_id, b_id, c_id, d_id):
                    # Remove the longer edge
                    len_ab = self._edge_length_sq(a_id, b_id)
                    len_cd = self._edge_length_sq(c_id, d_id)

                    if len_ab > len_cd:
                        edges_to_remove.add((a_id, b_id))
                    else:
                        edges_to_remove.add((c_id, d_id))

        # Actually remove the edges
        for edge in edges_to_remove:
            self._remove_edge(edge[0], edge[1])

    def _triangulation_search(self) -> None:
        """Perform triangulation search (quadrilateral + intersection)."""
        self._quadrilateral_search()
        self._intersection_search()

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration.

        Follows GNG algorithm with triangulation search added after Step 8.

        Args:
            sample: Input sample vector.
        """
        p = self.params

        # Decay all errors
        for node in self.nodes:
            if node.id == -1:
                continue
            node.error -= p.beta * node.error

        # Find two nearest nodes
        s1_id, s2_id = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        # Update winner error (using squared distance)
        dist_sq = np.sum((sample - self.nodes[s1_id].weight) ** 2)
        self.nodes[s1_id].error += dist_sq

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
        topology_changed = False

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
                    self.n_learning += 1
                    return

                # Find neighbor of q with maximum error
                max_err_f = 0.0
                f_id = -1
                for neighbor_id in self.edges_per_node.get(q_id, set()):
                    if self.nodes[neighbor_id].error > max_err_f:
                        max_err_f = self.nodes[neighbor_id].error
                        f_id = neighbor_id

                if f_id != -1:
                    # Add new node between q and f
                    new_weight = (self.nodes[q_id].weight + self.nodes[f_id].weight) * 0.5
                    new_id = self._add_node(new_weight)

                    if new_id != -1:
                        # Update edges
                        self._remove_edge(q_id, f_id)
                        self._add_edge(q_id, new_id)
                        self._add_edge(f_id, new_id)

                        # Connect to common neighbors of q and f (GCS-style)
                        neighbors_q = self.edges_per_node.get(q_id, set())
                        neighbors_f = self.edges_per_node.get(f_id, set())
                        common_neighbors = neighbors_q & neighbors_f
                        for common_id in common_neighbors:
                            self._add_edge(new_id, common_id)

                        # Update errors
                        self.nodes[q_id].error *= 1 - p.alpha
                        self.nodes[f_id].error *= 1 - p.alpha
                        self.nodes[new_id].error = (
                            self.nodes[q_id].error + self.nodes[f_id].error
                        ) * 0.5

                        topology_changed = True

        # Perform triangulation search after topology changes
        if topology_changed or edges_to_remove:
            self._triangulation_search()

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GrowingNeuralGasT, int], None] | None = None,
    ) -> GrowingNeuralGasT:
        """Train on data for multiple iterations.

        Each iteration randomly samples one point from data.

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

    def partial_fit(self, sample: np.ndarray) -> GrowingNeuralGasT:
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
        return count // 2  # Each edge counted twice

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure.

        Returns:
            Tuple of:
                - nodes: Array of shape (n_active_nodes, n_dim) with positions.
                - edges: List of (i, j) tuples indexing into nodes array.
        """
        # Get active nodes and create index mapping
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

        # Get edges using new indices
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
        """Get error values for active nodes.

        Returns:
            Array of error values in same order as get_graph() nodes.
        """
        return np.array([node.error for node in self.nodes if node.id != -1])

    def get_triangles(self) -> list[tuple[int, int, int]]:
        """Get triangles from the current graph structure.

        Extracts triangles by finding all cliques of size 3.

        Returns:
            List of (i, j, k) tuples representing triangles,
            indexing into get_graph() nodes array.
        """
        active_ids = self._get_active_node_ids()
        if len(active_ids) < 3:
            return []

        # Create index mapping for get_graph output
        graph_index_map = {}
        idx = 0
        for node in self.nodes:
            if node.id != -1:
                graph_index_map[node.id] = idx
                idx += 1

        # Find triangles (cliques of size 3)
        triangles = []
        for a_id in active_ids:
            neighbors_a = self.edges_per_node.get(a_id, set())
            neighbors_a_list = [n for n in neighbors_a if n > a_id]

            for b_id in neighbors_a_list:
                neighbors_b = self.edges_per_node.get(b_id, set())

                for c_id in neighbors_a_list:
                    if c_id <= b_id:
                        continue
                    if c_id in neighbors_b:
                        # a, b, c form a triangle
                        triangles.append((
                            graph_index_map[a_id],
                            graph_index_map[b_id],
                            graph_index_map[c_id],
                        ))

        return triangles


# Aliases
GNGT = GrowingNeuralGasT
GNG_T = GrowingNeuralGasT
