"""Linde-Buzo-Gray (LBG) algorithm implementation.

Based on:
    - Linde, Y., Buzo, A., & Gray, R. (1980). "An Algorithm for Vector Quantizer Design"
    - demogng.de reference implementation

LBG is a batch learning algorithm for vector quantization that iteratively
assigns data points to clusters and moves centroids to cluster centers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class LBGParams:
    """LBG hyperparameters.

    Attributes:
        n_nodes: Number of codebook vectors (fixed).
        max_epochs: Maximum number of epochs (full passes through data).
        convergence_threshold: Stop if distortion change is below this.
        use_utility: Whether to use utility-based node management.
        utility_threshold: Threshold for removing low-utility nodes.
    """

    n_nodes: int = 50
    max_epochs: int = 100
    convergence_threshold: float = 1e-6
    use_utility: bool = False
    utility_threshold: float = 0.01


class LindeBuzoGray:
    """Linde-Buzo-Gray (LBG) vector quantization algorithm.

    LBG is a batch algorithm that works by:
    1. Assigning each data point to its nearest codebook vector
    2. Moving each codebook vector to the centroid of its assigned points
    3. Repeating until convergence

    When use_utility=True (LBG-U), nodes with low utility (few assignments)
    can be removed and reinitialized in high-error regions.

    Attributes:
        params: LBG hyperparameters.
        weights: Codebook vectors, shape (n_nodes, n_dim).
        utility: Assignment counts for each node (for LBG-U).
        n_learning: Total number of epochs.

    Examples
    --------
    >>> import numpy as np
    >>> from model import LindeBuzoGray
    >>> X = np.random.rand(1000, 2)
    >>> lbg = LindeBuzoGray(n_dim=2)
    >>> lbg.train(X)
    >>> nodes, edges = lbg.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: LBGParams | None = None,
        seed: int | None = None,
    ):
        """Initialize LBG.

        Args:
            n_dim: Dimension of input data.
            params: LBG hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or LBGParams()
        self.rng = np.random.default_rng(seed)

        p = self.params

        # Initialize codebook vectors randomly in [0, 1]
        self.weights = self.rng.random((p.n_nodes, n_dim)).astype(np.float32)

        # Utility tracking (assignment counts)
        self.utility = np.zeros(p.n_nodes, dtype=np.float32)

        # Error tracking
        self.errors = np.zeros(p.n_nodes, dtype=np.float32)

        # Counters
        self.n_learning = 0

    def _assign_to_nearest(self, data: np.ndarray) -> np.ndarray:
        """Assign each data point to its nearest codebook vector.

        Args:
            data: Data points of shape (n_samples, n_dim).

        Returns:
            Array of shape (n_samples,) with cluster assignments.
        """
        n_samples = len(data)
        assignments = np.zeros(n_samples, dtype=np.int32)

        for i, x in enumerate(data):
            distances = np.sum((self.weights - x) ** 2, axis=1)
            assignments[i] = np.argmin(distances)

        return assignments

    def _compute_distortion(self, data: np.ndarray, assignments: np.ndarray) -> float:
        """Compute total distortion (quantization error).

        Args:
            data: Data points.
            assignments: Cluster assignments.

        Returns:
            Total squared distortion.
        """
        distortion = 0.0
        for i, x in enumerate(data):
            distortion += np.sum((x - self.weights[assignments[i]]) ** 2)
        return distortion

    def _update_centroids(
        self, data: np.ndarray, assignments: np.ndarray
    ) -> None:
        """Update codebook vectors to cluster centroids.

        Args:
            data: Data points.
            assignments: Cluster assignments.
        """
        p = self.params

        # Reset utility and errors
        self.utility.fill(0)
        self.errors.fill(0)

        # Accumulate sums and counts
        sums = np.zeros_like(self.weights)
        counts = np.zeros(p.n_nodes, dtype=np.int32)

        for i, x in enumerate(data):
            cluster = assignments[i]
            sums[cluster] += x
            counts[cluster] += 1
            # Accumulate squared error
            self.errors[cluster] += np.sum((x - self.weights[cluster]) ** 2)

        # Update utility (normalized assignment counts)
        total_samples = len(data)
        self.utility = counts.astype(np.float32) / total_samples

        # Update centroids (only for non-empty clusters)
        for j in range(p.n_nodes):
            if counts[j] > 0:
                self.weights[j] = sums[j] / counts[j]

    def _handle_utility(self, data: np.ndarray) -> None:
        """Handle low-utility nodes (LBG-U).

        Remove nodes with utility below threshold and reinitialize
        them near high-error regions.

        Args:
            data: Data points for reinitialization.
        """
        p = self.params

        if not p.use_utility:
            return

        # Find low-utility nodes
        low_utility_mask = self.utility < p.utility_threshold

        if not np.any(low_utility_mask):
            return

        # Find node with highest error
        max_error_idx = np.argmax(self.errors)

        # Reinitialize low-utility nodes near high-error region
        for idx in np.where(low_utility_mask)[0]:
            # Add small random perturbation to high-error node position
            noise = self.rng.normal(0, 0.1, self.n_dim)
            self.weights[idx] = self.weights[max_error_idx] + noise

    def train(
        self,
        data: np.ndarray,
        n_iterations: int | None = None,
        callback: Callable[[LindeBuzoGray, int], None] | None = None,
    ) -> LindeBuzoGray:
        """Train on data using batch updates.

        Args:
            data: Training data of shape (n_samples, n_dim).
            n_iterations: Number of epochs (overrides max_epochs if provided).
            callback: Optional callback(self, epoch) called each epoch.

        Returns:
            self for chaining.
        """
        p = self.params
        max_epochs = n_iterations if n_iterations is not None else p.max_epochs

        prev_distortion = float("inf")

        for epoch in range(max_epochs):
            # Assign data points to nearest codebook vectors
            assignments = self._assign_to_nearest(data)

            # Compute distortion
            distortion = self._compute_distortion(data, assignments)

            # Update centroids
            self._update_centroids(data, assignments)

            # Handle utility-based management (LBG-U)
            self._handle_utility(data)

            self.n_learning += 1

            if callback is not None:
                callback(self, epoch)

            # Check convergence
            if abs(prev_distortion - distortion) < p.convergence_threshold:
                break

            prev_distortion = distortion

        return self

    def partial_fit(self, sample: np.ndarray) -> LindeBuzoGray:
        """Single online learning step (not typical for LBG).

        For true batch learning, use train() instead.

        Args:
            sample: Input vector of shape (n_dim,).

        Returns:
            self for chaining.
        """
        # Find nearest codebook vector
        distances = np.sum((self.weights - sample) ** 2, axis=1)
        bmu_idx = np.argmin(distances)

        # Simple online update (move slightly toward sample)
        lr = 0.01
        self.weights[bmu_idx] += lr * (sample - self.weights[bmu_idx])

        self.n_learning += 1
        return self

    @property
    def n_nodes(self) -> int:
        """Number of codebook vectors."""
        return self.params.n_nodes

    @property
    def n_edges(self) -> int:
        """Number of edges (always 0 for LBG - no topology)."""
        return 0

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure.

        Note: LBG has no edges (no topology learning).

        Returns:
            Tuple of:
                - nodes: Array of shape (n_nodes, n_dim) with positions.
                - edges: Empty list (LBG has no topology).
        """
        return self.weights.copy(), []

    def get_node_utilities(self) -> np.ndarray:
        """Get utility (assignment ratio) for each node.

        Returns:
            Array of utility values.
        """
        return self.utility.copy()

    def get_node_errors(self) -> np.ndarray:
        """Get accumulated error for each node.

        Returns:
            Array of error values.
        """
        return self.errors.copy()

    def get_quantization_error(self, data: np.ndarray) -> float:
        """Compute mean quantization error on data.

        Args:
            data: Data points of shape (n_samples, n_dim).

        Returns:
            Mean distance from each point to its nearest codebook vector.
        """
        total_error = 0.0
        for x in data:
            distances = np.sum((self.weights - x) ** 2, axis=1)
            total_error += np.sqrt(np.min(distances))
        return total_error / len(data)


# Aliases
LBG = LindeBuzoGray
LBGU = LindeBuzoGray  # Use with params.use_utility=True
