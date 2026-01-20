"""Hard Competitive Learning (HCL) implementation.

Based on:
    - Rumelhart, D. E., & Zipser, D. (1985). "Feature discovery by competitive learning"
    - demogng.de reference implementation

HCL is the simplest competitive learning algorithm where only the winner
(Best Matching Unit) is updated for each input signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class HCLParams:
    """HCL hyperparameters.

    Attributes:
        n_nodes: Number of reference vectors (fixed).
        learning_rate_initial: Initial learning rate.
        learning_rate_final: Final learning rate.
    """

    n_nodes: int = 50
    learning_rate_initial: float = 0.5
    learning_rate_final: float = 0.01


class HardCompetitiveLearning:
    """Hard Competitive Learning (HCL) implementation.

    HCL is a winner-take-all competitive learning algorithm where only
    the closest node (Best Matching Unit) is updated for each input.
    No neighborhood function is used - only the winner moves.

    This is the simplest form of competitive learning and forms the
    basis for more sophisticated algorithms like SOM and Neural Gas.

    Attributes:
        params: HCL hyperparameters.
        weights: Reference vectors, shape (n_nodes, n_dim).
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import HardCompetitiveLearning
    >>> X = np.random.rand(1000, 2)
    >>> hcl = HardCompetitiveLearning(n_dim=2)
    >>> hcl.train(X, n_iterations=5000)
    >>> nodes, edges = hcl.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: HCLParams | None = None,
        seed: int | None = None,
    ):
        """Initialize HCL.

        Args:
            n_dim: Dimension of input data.
            params: HCL hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or HCLParams()
        self.rng = np.random.default_rng(seed)

        p = self.params

        # Initialize reference vectors randomly in [0, 1]
        self.weights = self.rng.random((p.n_nodes, n_dim)).astype(np.float32)

        # Counters
        self.n_learning = 0
        self._total_iterations = 1

    def _find_bmu(self, x: np.ndarray) -> int:
        """Find the Best Matching Unit (BMU) for input x.

        Args:
            x: Input vector.

        Returns:
            Index of the BMU.
        """
        distances = np.sum((self.weights - x) ** 2, axis=1)
        return int(np.argmin(distances))

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

        # Exponential decay of learning rate
        lr = p.learning_rate_initial * (
            p.learning_rate_final / p.learning_rate_initial
        ) ** t

        # Find BMU
        bmu_idx = self._find_bmu(sample)

        # Update only the winner (hard competitive learning)
        self.weights[bmu_idx] += lr * (sample - self.weights[bmu_idx])

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[HardCompetitiveLearning, int], None] | None = None,
    ) -> HardCompetitiveLearning:
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

    def partial_fit(self, sample: np.ndarray) -> HardCompetitiveLearning:
        """Single online learning step.

        Uses final (small) learning rate.

        Args:
            sample: Input vector of shape (n_dim,).

        Returns:
            self for chaining.
        """
        p = self.params

        bmu_idx = self._find_bmu(sample)
        self.weights[bmu_idx] += p.learning_rate_final * (
            sample - self.weights[bmu_idx]
        )

        self.n_learning += 1
        return self

    @property
    def n_nodes(self) -> int:
        """Number of reference vectors."""
        return self.params.n_nodes

    @property
    def n_edges(self) -> int:
        """Number of edges (always 0 for HCL - no topology)."""
        return 0

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure.

        Note: HCL has no edges (no topology learning).

        Returns:
            Tuple of:
                - nodes: Array of shape (n_nodes, n_dim) with positions.
                - edges: Empty list (HCL has no topology).
        """
        return self.weights.copy(), []

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
HCL = HardCompetitiveLearning
