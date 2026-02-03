"""GNG Efficient - Optimized Growing Neural Gas implementation.

Based on:
    Fišer, D., Faigl, J., & Kulich, M. (2013).
    "Growing Neural Gas Efficiently"
    Neurocomputing.

Two key optimizations:
    1. Uniform Grid for O(1) nearest neighbor search
    2. Lazy error evaluation with cycle counters
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from .lazy_heap import LazyHeap, SimpleLazyHeap
    from .uniform_grid import GrowingUniformGrid, UniformGridParams
except ImportError:
    from lazy_heap import LazyHeap, SimpleLazyHeap
    from uniform_grid import GrowingUniformGrid, UniformGridParams


@dataclass
class GNGEfficientParams:
    """GNG Efficient hyperparameters.

    Combines standard GNG parameters with optimization parameters.

    Standard GNG Parameters (from Fišer 2013 Table 2):
        max_nodes: Maximum number of nodes.
        lambda_: Node insertion interval (every lambda_ iterations).
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for neighbor nodes.
        alpha: Error decay rate when splitting.
        beta: Global error decay rate (per step).
        max_age: Maximum edge age before removal.

    Optimization Parameters:
        h_t: Grid density threshold for rebuild.
        h_rho: Grid expansion factor.
        use_uniform_grid: Enable Uniform Grid optimization.
        use_lazy_error: Enable lazy error evaluation.
    """

    # Standard GNG parameters (paper Table 2 defaults)
    max_nodes: int = 100
    lambda_: int = 200
    eps_b: float = 0.05
    eps_n: float = 0.0006
    alpha: float = 0.95
    beta: float = 0.9995
    max_age: int = 200

    # Optimization parameters
    h_t: float = 0.1
    h_rho: float = 1.5
    use_uniform_grid: bool = True
    use_lazy_error: bool = True


@dataclass
class NeuronNode:
    """A neuron node in the GNG network.

    Attributes:
        id: Node ID (-1 means invalid/removed).
        weight: Position vector.
        error: Accumulated error (may need fix_error correction).
        cycle: Cycle counter when error was last updated.
    """

    id: int = -1
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    error: float = 0.0
    cycle: int = 0


class GNGEfficient:
    """Optimized Growing Neural Gas algorithm.

    Implements the GNG algorithm with two major optimizations:

    1. **Uniform Grid** (Section 4): Replaces linear nearest neighbor search
       with a grid-based spatial index. Provides O(1) average case for
       nearest neighbor queries in 2D/3D spaces.

    2. **Lazy Error Evaluation** (Section 5): Instead of decaying all errors
       every step (O(n)), each node tracks when its error was last updated.
       The actual error is computed on-demand using the formula:
           E_v,c = beta^((c-C_v)*lambda) * E_v,C_v

    These optimizations preserve the exact behavior of standard GNG while
    reducing complexity from O(n) to near O(1) per step.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GNGEfficient, GNGEfficientParams
    >>> params = GNGEfficientParams(max_nodes=100, lambda_=100)
    >>> gng = GNGEfficient(n_dim=2, params=params)
    >>> X = np.random.rand(1000, 2)
    >>> gng.train(X, n_iterations=5000)
    >>> nodes, edges = gng.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GNGEfficientParams | None = None,
        seed: int | None = None,
    ):
        """Initialize GNG Efficient.

        Args:
            n_dim: Dimension of input data.
            params: Algorithm parameters.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or GNGEfficientParams()
        self.rng = np.random.default_rng(seed)

        # Pre-compute beta powers for lazy error evaluation
        # beta_powers[i] = beta^i for i = 0 to lambda
        self._beta_powers = np.array(
            [self.params.beta**i for i in range(self.params.lambda_ + 1)]
        )

        # Node management
        self.nodes: list[NeuronNode] = [
            NeuronNode() for _ in range(self.params.max_nodes)
        ]
        self._addable_indices: deque[int] = deque(range(self.params.max_nodes))

        # Edge management
        self.edges = np.zeros(
            (self.params.max_nodes, self.params.max_nodes), dtype=np.int32
        )
        self.edges_per_node: dict[int, set[int]] = {}

        # Uniform Grid for nearest neighbor search
        if self.params.use_uniform_grid:
            grid_params = UniformGridParams(
                h_t=self.params.h_t, h_rho=self.params.h_rho
            )
            self._grid = GrowingUniformGrid(n_dim, grid_params)
        else:
            self._grid = None

        # Lazy Heap for error tracking
        if self.params.use_lazy_error:
            self._error_heap: LazyHeap | SimpleLazyHeap = LazyHeap(self._fix_error)
        else:
            self._error_heap = SimpleLazyHeap(self._fix_error)

        # Counters
        self.cycle = 0  # c in paper
        self.step = 0  # s in paper (0 to lambda-1)
        self.n_learning = 0

        # Initialize with 2 random nodes
        self._init_nodes()

    def _init_nodes(self) -> None:
        """Initialize with 2 random nodes."""
        for _ in range(2):
            weight = self.rng.random(self.n_dim).astype(np.float64)
            self._add_node(weight)

    def _add_node(self, weight: np.ndarray, error: float = 0.0) -> int:
        """Add a new node.

        Args:
            weight: Position vector.
            error: Initial error value.

        Returns:
            ID of new node, or -1 if no space.
        """
        if not self._addable_indices:
            return -1

        node_id = self._addable_indices.popleft()
        node = NeuronNode(
            id=node_id,
            weight=weight.copy().astype(np.float64),
            error=error,
            cycle=self.cycle,
        )
        self.nodes[node_id] = node
        self.edges_per_node[node_id] = set()

        # Add to spatial index
        if self._grid is not None:
            self._grid.insert(node)

        # Add to error heap
        self._error_heap.insert(node)

        return node_id

    def _remove_node(self, node_id: int) -> None:
        """Remove a node (only if isolated)."""
        if self.edges_per_node.get(node_id):
            return

        node = self.nodes[node_id]

        # Remove from spatial index
        if self._grid is not None:
            self._grid.remove(node)

        # Remove from error heap
        self._error_heap.remove(node)

        self.edges_per_node.pop(node_id, None)
        self.nodes[node_id].id = -1
        self._addable_indices.append(node_id)

    def _add_edge(self, node1: int, node2: int) -> None:
        """Add or reset edge between two nodes.

        Per Algorithm 3, step 6: A_ν,μ ← 0
        The edge age is set to 0 here, then incremented in the neighbor loop.
        """
        if self.edges[node1, node2] == 0:
            # New edge
            self.edges_per_node[node1].add(node2)
            self.edges_per_node[node2].add(node1)
        # Reset age to 0 (will be incremented to 1 in the neighbor loop)
        self.edges[node1, node2] = 0
        self.edges[node2, node1] = 0

    def _remove_edge(self, node1: int, node2: int) -> None:
        """Remove edge between two nodes."""
        self.edges_per_node[node1].discard(node2)
        self.edges_per_node[node2].discard(node1)
        self.edges[node1, node2] = 0
        self.edges[node2, node1] = 0

    def _find_two_nearest(self, x: np.ndarray) -> tuple[int, int]:
        """Find the two nearest nodes to input x.

        Uses Uniform Grid if enabled, otherwise linear search.
        """
        if self._grid is not None:
            n1, n2 = self._grid.find_two_nearest(x)
            id1 = n1.id if n1 is not None else -1
            id2 = n2.id if n2 is not None else -1
            return id1, id2

        # Linear search fallback
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

    def _fix_error(self, cycle: int, node: NeuronNode) -> None:
        """Fix node error to account for decay since last update.

        Implements fix_error from Algorithm 4:
            E_v <- beta^(lambda * (c - C_v)) * E_v
            C_v <- c

        Args:
            cycle: Current cycle counter.
            node: Node to fix.
        """
        if node.cycle == cycle:
            return

        cycles_diff = cycle - node.cycle
        # beta^(lambda * cycles_diff)
        decay = self.params.beta ** (self.params.lambda_ * cycles_diff)
        node.error *= decay
        node.cycle = cycle

    def _inc_error(self, node: NeuronNode, value: float) -> None:
        """Increment node error with lazy evaluation.

        Implements inc_error* from Algorithm 4:
            1. fix_error(c, v)
            2. E_v <- beta^(lambda-s) * E_v + value
            3. update node in heap

        Args:
            node: Node to update.
            value: Error value to add (typically squared distance).
        """
        # Fix error to current cycle
        self._fix_error(self.cycle, node)

        # Apply the formula: E_v <- beta^(lambda-s) * E_v + value
        # This accounts for the remaining decay within this cycle
        decay_factor = self._beta_powers[self.params.lambda_ - self.step]
        node.error = decay_factor * node.error + value

        # Update in heap
        self._error_heap.update(node)

    def _dec_error(self, node: NeuronNode, alpha: float) -> None:
        """Decrease node error.

        Implements dec_error* from Algorithm 4.

        Args:
            node: Node to update.
            alpha: Decay factor.
        """
        self._fix_error(self.cycle, node)
        node.error *= alpha
        self._error_heap.update(node)

    def _set_error(self, node: NeuronNode, value: float) -> None:
        """Set node error to specific value.

        Implements set_error* from Algorithm 4.

        Args:
            node: Node to update.
            value: New error value.
        """
        node.error = value
        node.cycle = self.cycle
        self._error_heap.insert(node)

    def _largest_error(self) -> tuple[int, int]:
        """Find node with largest error and its neighbor with largest error.

        Implements largest_error* from Algorithm 4.

        Returns:
            Tuple of (q_id, f_id) where q has largest error
            and f is q's neighbor with largest error.
        """
        q = self._error_heap.top(self.cycle)
        if q is None or q.id == -1:
            return -1, -1

        # Find neighbor with largest error
        max_err = -float("inf")
        f_id = -1
        for neighbor_id in self.edges_per_node.get(q.id, set()):
            neighbor = self.nodes[neighbor_id]
            if neighbor.id == -1:
                continue
            # Fix neighbor's error before comparison
            self._fix_error(self.cycle, neighbor)
            if neighbor.error > max_err:
                max_err = neighbor.error
                f_id = neighbor_id

        return q.id, f_id

    def _adapt(self, sample: np.ndarray) -> None:
        """Single adaptation step.

        Implements gng_adapt from Algorithm 3.

        Args:
            sample: Input signal.
        """
        p = self.params

        # Find two nearest nodes
        s1_id, s2_id = self._find_two_nearest(sample)

        if s1_id == -1 or s2_id == -1:
            return

        winner = self.nodes[s1_id]

        # Increment winner error
        dist_sq = np.sum((sample - winner.weight) ** 2)
        self._inc_error(winner, dist_sq)

        # Move winner toward sample
        winner.weight += p.eps_b * (sample - winner.weight)
        if self._grid is not None:
            self._grid.update(winner)

        # Connect s1 and s2
        self._add_edge(s1_id, s2_id)

        # Update neighbors and age edges
        edges_to_remove = []
        for neighbor_id in list(self.edges_per_node[s1_id]):
            # Increment edge age
            self.edges[s1_id, neighbor_id] += 1
            self.edges[neighbor_id, s1_id] += 1

            if self.edges[s1_id, neighbor_id] > p.max_age:
                edges_to_remove.append(neighbor_id)
            else:
                # Move neighbor toward sample
                neighbor = self.nodes[neighbor_id]
                neighbor.weight += p.eps_n * (sample - neighbor.weight)
                if self._grid is not None:
                    self._grid.update(neighbor)

        # Remove old edges and isolated nodes
        for neighbor_id in edges_to_remove:
            self._remove_edge(s1_id, neighbor_id)
            if not self.edges_per_node.get(neighbor_id):
                self._remove_node(neighbor_id)

        # Note: dec_all_error is a no-op in lazy evaluation
        # The decay is handled in _fix_error and _inc_error

    def _insert_node(self) -> None:
        """Insert a new node.

        Implements gng_new_node from Algorithm 3.
        """
        if not self._addable_indices:
            return

        # Find node with largest error and its neighbor
        q_id, f_id = self._largest_error()

        if q_id == -1 or f_id == -1:
            return

        q = self.nodes[q_id]
        f = self.nodes[f_id]

        # Create new node between q and f
        new_weight = (q.weight + f.weight) * 0.5

        # Fix errors before computing new node's error
        self._fix_error(self.cycle, q)
        self._fix_error(self.cycle, f)

        # Decrease errors of q and f
        self._dec_error(q, self.params.alpha)
        self._dec_error(f, self.params.alpha)

        # Add new node with average error
        new_error = (q.error + f.error) * 0.5
        new_id = self._add_node(new_weight, error=new_error)

        if new_id == -1:
            return

        # Update edges
        self._remove_edge(q_id, f_id)
        self._add_edge(q_id, new_id)
        self._add_edge(f_id, new_id)

    def _one_cycle(self, data: np.ndarray, callback: Callable | None = None) -> None:
        """Execute one cycle (lambda adaptation steps + node insertion).

        Args:
            data: Training data.
            callback: Optional callback function.
        """
        n_samples = len(data)

        # Adaptation phase: lambda steps
        for self.step in range(self.params.lambda_):
            idx = self.rng.integers(0, n_samples)
            self._adapt(data[idx])

            if callback is not None:
                callback(self, self.n_learning)

            self.n_learning += 1

        # Growing phase: insert new node
        self._insert_node()
        self.cycle += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GNGEfficient, int], None] | None = None,
    ) -> GNGEfficient:
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

        for _ in range(n_iterations):
            # Update step counter (s in Algorithm 3)
            self.step = self.n_learning % self.params.lambda_

            idx = self.rng.integers(0, n_samples)
            self._adapt(data[idx])

            if callback is not None:
                callback(self, self.n_learning)

            self.n_learning += 1

            # Check for node insertion
            if self.n_learning % self.params.lambda_ == 0:
                self._insert_node()
                self.cycle += 1

        return self

    def partial_fit(self, sample: np.ndarray) -> GNGEfficient:
        """Single online learning step.

        Args:
            sample: Input vector of shape (n_dim,).

        Returns:
            self for chaining.
        """
        # Update step counter (s in Algorithm 3)
        self.step = self.n_learning % self.params.lambda_

        self._adapt(sample)
        self.n_learning += 1

        if self.n_learning % self.params.lambda_ == 0:
            self._insert_node()
            self.cycle += 1

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
            Tuple of:
                - nodes: Array of shape (n_active_nodes, n_dim) with positions.
                - edges: List of (i, j) tuples indexing into nodes array.
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

    def get_node_errors(self) -> np.ndarray:
        """Get error values for active nodes.

        Returns:
            Array of error values (fixed to current cycle).
        """
        errors = []
        for node in self.nodes:
            if node.id != -1:
                self._fix_error(self.cycle, node)
                errors.append(node.error)
        return np.array(errors)
