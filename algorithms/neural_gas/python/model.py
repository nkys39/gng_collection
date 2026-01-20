"""Neural Gas algorithm implementation.

Based on:
    - Martinetz, T. and Schulten, K. (1991). "A Neural-Gas Network Learns Topologies"
    - Martinetz, T. and Schulten, K. (1994). "Topology Representing Networks"

Neural Gas uses a rank-based neighborhood function where all nodes are
updated with strength decreasing exponentially with their distance rank.
Combined with Competitive Hebbian Learning (CHL) to learn topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class NeuralGasParams:
    """Neural Gas hyperparameters.

    Attributes:
        n_nodes: Number of reference vectors (fixed).
        lambda_initial: Initial neighborhood range.
        lambda_final: Final neighborhood range.
        eps_initial: Initial learning rate.
        eps_final: Final learning rate.
        max_age: Maximum edge age (for CHL).
        use_chl: Whether to use Competitive Hebbian Learning for edges.
    """

    n_nodes: int = 50
    lambda_initial: float = 10.0
    lambda_final: float = 0.1
    eps_initial: float = 0.5
    eps_final: float = 0.005
    max_age: int = 50
    use_chl: bool = True  # Competitive Hebbian Learning


class NeuralGas:
    """Neural Gas algorithm implementation.

    Neural Gas performs vector quantization with soft competitive learning.
    All reference vectors are updated for each input, with adaptation strength
    decreasing exponentially with the rank (distance order).

    When use_chl=True, edges are created between the two closest nodes
    (Competitive Hebbian Learning) to learn the data topology.

    Attributes:
        params: Neural Gas hyperparameters.
        weights: Reference vectors, shape (n_nodes, n_dim).
        edges: Edge age matrix (for CHL).
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import NeuralGas
    >>> X = np.random.rand(1000, 2)
    >>> ng = NeuralGas(n_dim=2)
    >>> ng.train(X, n_iterations=5000)
    >>> nodes, edges = ng.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: NeuralGasParams | None = None,
        seed: int | None = None,
    ):
        """Initialize Neural Gas.

        Args:
            n_dim: Dimension of input data.
            params: Neural Gas hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or NeuralGasParams()
        self.rng = np.random.default_rng(seed)

        p = self.params

        # Initialize reference vectors randomly in [0, 1]
        self.weights = self.rng.random((p.n_nodes, n_dim)).astype(np.float32)

        # Edge management for Competitive Hebbian Learning
        self.edges = np.zeros((p.n_nodes, p.n_nodes), dtype=np.int32)

        # Counters
        self.n_learning = 0
        self._total_iterations = 1

    def _get_ranks(self, x: np.ndarray) -> np.ndarray:
        """Get distance ranks for all nodes given input x.

        Args:
            x: Input vector.

        Returns:
            Array of ranks (0 = closest, 1 = second closest, etc.)
        """
        distances = np.sum((self.weights - x) ** 2, axis=1)
        return np.argsort(np.argsort(distances))

    def _one_train_update(
        self, sample: np.ndarray, iteration: int, total_iterations: int
    ) -> None:
        """Single training iteration.

        Args:
            sample: Input sample vector.
            iteration: Current iteration number.
            total_iterations: Total number of iterations.
        """
        p = self.params

        # Compute decay factor
        t = iteration / max(1, total_iterations - 1)

        # Exponential decay of lambda and epsilon
        lambda_t = p.lambda_initial * (p.lambda_final / p.lambda_initial) ** t
        eps_t = p.eps_initial * (p.eps_final / p.eps_initial) ** t

        # Get ranks
        ranks = self._get_ranks(sample)

        # Compute neighborhood function: h_k = exp(-k / lambda)
        h = np.exp(-ranks / lambda_t)

        # Update all weights
        diff = sample - self.weights
        self.weights += eps_t * h[:, np.newaxis] * diff

        # Competitive Hebbian Learning: connect two closest nodes
        if p.use_chl:
            # Find winner and second winner
            distances = np.sum((self.weights - sample) ** 2, axis=1)
            sorted_idx = np.argsort(distances)
            s1, s2 = sorted_idx[0], sorted_idx[1]

            # Age existing edges from winner (only where edges exist)
            existing_s1 = self.edges[s1, :] > 0
            self.edges[s1, existing_s1] += 1
            self.edges[existing_s1, s1] += 1

            # Create/reset edge between s1 and s2
            self.edges[s1, s2] = 1
            self.edges[s2, s1] = 1

            # Remove old edges
            old_edges = self.edges > p.max_age
            self.edges[old_edges] = 0

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[NeuralGas, int], None] | None = None,
    ) -> NeuralGas:
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
        self._total_iterations = n_iterations

        for i in range(n_iterations):
            idx = self.rng.integers(0, n_samples)
            self._one_train_update(data[idx], i, n_iterations)

            if callback is not None:
                callback(self, i)

        return self

    def partial_fit(self, sample: np.ndarray) -> NeuralGas:
        """Single online learning step.

        Uses final (small) values for lambda and epsilon.

        Args:
            sample: Input vector of shape (n_dim,).

        Returns:
            self for chaining.
        """
        p = self.params

        # Use final values for online learning
        lambda_t = p.lambda_final
        eps_t = p.eps_final

        # Get ranks
        ranks = self._get_ranks(sample)

        # Compute neighborhood function
        h = np.exp(-ranks / lambda_t)

        # Update weights
        diff = sample - self.weights
        self.weights += eps_t * h[:, np.newaxis] * diff

        # CHL
        if p.use_chl:
            distances = np.sum((self.weights - sample) ** 2, axis=1)
            sorted_idx = np.argsort(distances)
            s1, s2 = sorted_idx[0], sorted_idx[1]

            # Age existing edges from winner
            existing_s1 = self.edges[s1, :] > 0
            self.edges[s1, existing_s1] += 1
            self.edges[existing_s1, s1] += 1

            self.edges[s1, s2] = 1
            self.edges[s2, s1] = 1

            old_edges = self.edges > p.max_age
            self.edges[old_edges] = 0

        self.n_learning += 1
        return self

    @property
    def n_nodes(self) -> int:
        """Number of reference vectors."""
        return self.params.n_nodes

    @property
    def n_edges(self) -> int:
        """Number of active edges."""
        return np.sum(self.edges > 0) // 2

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure.

        Returns:
            Tuple of:
                - nodes: Array of shape (n_nodes, n_dim) with positions.
                - edges: List of (i, j) tuples for connected nodes.
        """
        nodes = self.weights.copy()

        # Extract edges from adjacency matrix
        edges = []
        for i in range(self.params.n_nodes):
            for j in range(i + 1, self.params.n_nodes):
                if self.edges[i, j] > 0:
                    edges.append((i, j))

        return nodes, edges

    def get_quantization_error(self, data: np.ndarray) -> float:
        """Compute mean quantization error on data.

        Args:
            data: Data points of shape (n_samples, n_dim).

        Returns:
            Mean distance from each point to its nearest reference vector.
        """
        total_error = 0.0
        for x in data:
            distances = np.sum((self.weights - x) ** 2, axis=1)
            total_error += np.sqrt(np.min(distances))
        return total_error / len(data)


# Alias
NG = NeuralGas
