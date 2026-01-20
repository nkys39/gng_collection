"""Self-Organizing Map (SOM) implementation.

Based on:
    - Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps"
    - Kohonen, T. (2001). "Self-Organizing Maps" (3rd ed.)

SOM uses a fixed grid topology where neurons are arranged in a 2D lattice.
The neighborhood function is based on grid distance, not data space distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SOMParams:
    """SOM hyperparameters.

    Attributes:
        grid_height: Height of the neuron grid.
        grid_width: Width of the neuron grid.
        sigma_initial: Initial neighborhood radius.
        sigma_final: Final neighborhood radius.
        learning_rate_initial: Initial learning rate.
        learning_rate_final: Final learning rate.
    """

    grid_height: int = 10
    grid_width: int = 10
    sigma_initial: float = 5.0
    sigma_final: float = 0.5
    learning_rate_initial: float = 0.5
    learning_rate_final: float = 0.01


class SelfOrganizingMap:
    """Self-Organizing Map (Kohonen Map) implementation.

    A neural network with a fixed 2D grid topology that learns to represent
    the input data distribution while preserving topological properties.

    Attributes:
        params: SOM hyperparameters.
        weights: Weight vectors for each neuron, shape (height, width, n_dim).
        grid_coords: Grid coordinates for each neuron, shape (height, width, 2).

    Examples
    --------
    >>> import numpy as np
    >>> from model import SelfOrganizingMap
    >>> X = np.random.rand(1000, 2)
    >>> som = SelfOrganizingMap(n_dim=2)
    >>> som.train(X, n_iterations=5000)
    >>> nodes, edges = som.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: SOMParams | None = None,
        seed: int | None = None,
    ):
        """Initialize SOM.

        Args:
            n_dim: Dimension of input data.
            params: SOM hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or SOMParams()
        self.rng = np.random.default_rng(seed)

        p = self.params

        # Initialize weights randomly in [0, 1]
        self.weights = self.rng.random((p.grid_height, p.grid_width, n_dim)).astype(
            np.float32
        )

        # Pre-compute grid coordinates for neighborhood calculations
        self.grid_coords = np.zeros((p.grid_height, p.grid_width, 2))
        for i in range(p.grid_height):
            for j in range(p.grid_width):
                self.grid_coords[i, j] = [i, j]

        # Counters
        self.n_learning = 0
        self._total_iterations = 1  # Will be set in train()

    def _get_bmu(self, x: np.ndarray) -> tuple[int, int]:
        """Find the Best Matching Unit (BMU) for input x.

        Args:
            x: Input vector.

        Returns:
            Tuple of (row, col) indices of the BMU.
        """
        # Compute distances to all neurons
        diff = self.weights - x
        distances = np.sum(diff ** 2, axis=2)

        # Find minimum
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx[0], bmu_idx[1]

    def _get_neighborhood(
        self, bmu: tuple[int, int], sigma: float
    ) -> np.ndarray:
        """Compute neighborhood function values for all neurons.

        Uses Manhattan distance on the grid per demogng.de reference.

        Args:
            bmu: (row, col) of the Best Matching Unit.
            sigma: Current neighborhood radius.

        Returns:
            Array of shape (height, width) with neighborhood values.
        """
        bmu_coord = np.array([bmu[0], bmu[1]])

        # Compute Manhattan grid distances (per demogng.de)
        diff = self.grid_coords - bmu_coord
        grid_distances = np.sum(np.abs(diff), axis=2)

        # Gaussian neighborhood function: exp(-d^2 / (2 * sigma^2))
        return np.exp(-(grid_distances ** 2) / (2 * sigma ** 2))

    def _one_train_update(
        self, sample: np.ndarray, iteration: int, total_iterations: int
    ) -> None:
        """Single training iteration.

        Args:
            sample: Input sample vector.
            iteration: Current iteration number.
            total_iterations: Total number of iterations for decay calculation.
        """
        p = self.params

        # Compute decay factor (linear decay)
        t = iteration / max(1, total_iterations - 1)

        # Decay learning rate and sigma
        sigma = p.sigma_initial * (p.sigma_final / p.sigma_initial) ** t
        lr = p.learning_rate_initial * (p.learning_rate_final / p.learning_rate_initial) ** t

        # Find BMU
        bmu = self._get_bmu(sample)

        # Get neighborhood values
        h = self._get_neighborhood(bmu, sigma)

        # Update all weights
        # w_new = w_old + lr * h * (x - w_old)
        diff = sample - self.weights
        self.weights += lr * h[:, :, np.newaxis] * diff

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[SelfOrganizingMap, int], None] | None = None,
    ) -> SelfOrganizingMap:
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

    def partial_fit(self, sample: np.ndarray) -> SelfOrganizingMap:
        """Single online learning step.

        Note: For proper online learning, you should track iteration count
        externally and use train() with appropriate decay schedules.

        Args:
            sample: Input vector of shape (n_dim,).

        Returns:
            self for chaining.
        """
        # Use a fixed small learning rate and sigma for online learning
        p = self.params

        bmu = self._get_bmu(sample)
        h = self._get_neighborhood(bmu, p.sigma_final)

        diff = sample - self.weights
        self.weights += p.learning_rate_final * h[:, :, np.newaxis] * diff

        self.n_learning += 1
        return self

    @property
    def n_nodes(self) -> int:
        """Number of neurons (fixed)."""
        return self.params.grid_height * self.params.grid_width

    @property
    def n_edges(self) -> int:
        """Number of grid edges."""
        p = self.params
        # Horizontal edges + vertical edges
        h_edges = p.grid_height * (p.grid_width - 1)
        v_edges = (p.grid_height - 1) * p.grid_width
        return h_edges + v_edges

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure (grid topology).

        Returns:
            Tuple of:
                - nodes: Array of shape (n_nodes, n_dim) with positions.
                - edges: List of (i, j) tuples representing grid connections.
        """
        p = self.params

        # Flatten weights to node list
        nodes = self.weights.reshape(-1, self.n_dim)

        # Create edges based on grid topology
        edges = []
        for i in range(p.grid_height):
            for j in range(p.grid_width):
                idx = i * p.grid_width + j

                # Right neighbor
                if j < p.grid_width - 1:
                    edges.append((idx, idx + 1))

                # Bottom neighbor
                if i < p.grid_height - 1:
                    edges.append((idx, idx + p.grid_width))

        return nodes, edges

    def get_node_activations(self, x: np.ndarray) -> np.ndarray:
        """Get activation (inverse distance) for each neuron given input x.

        Args:
            x: Input vector.

        Returns:
            Array of shape (height, width) with activation values.
        """
        diff = self.weights - x
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        # Inverse distance as activation (avoid division by zero)
        return 1.0 / (distances + 1e-8)


# Alias
SOM = SelfOrganizingMap
