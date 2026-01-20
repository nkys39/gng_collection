"""Growing Grid (GG) algorithm implementation.

Based on:
    - Fritzke, B. (1995). "Growing Grid - a self-organizing network with constant
      neighborhood range and adaptation strength"
    - demogng.de reference implementation

Growing Grid combines the structured topology of SOM with the ability to grow.
It starts with a small grid and adds rows/columns where error is highest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class GrowingGridParams:
    """Growing Grid hyperparameters.

    Attributes:
        initial_height: Initial grid height.
        initial_width: Initial grid width.
        max_nodes: Maximum number of nodes.
        lambda_: Growth interval (add row/column every lambda_ iterations).
        eps_b: Learning rate for the winner node.
        eps_n: Learning rate for neighbor nodes.
        sigma: Neighborhood radius (constant, unlike SOM).
        tau: Error decay rate.
    """

    initial_height: int = 2
    initial_width: int = 2
    max_nodes: int = 100
    lambda_: int = 100
    eps_b: float = 0.1
    eps_n: float = 0.01
    sigma: float = 1.5
    tau: float = 0.005


class GrowingGrid:
    """Growing Grid algorithm implementation.

    Growing Grid is a self-organizing network that starts with a small
    rectangular grid and grows by inserting rows or columns in regions
    with high accumulated error.

    Unlike SOM, the neighborhood range and learning rates are constant,
    not decaying over time.

    Attributes:
        params: Growing Grid hyperparameters.
        weights: Weight vectors, shape (height, width, n_dim).
        errors: Error accumulator for each node.
        height: Current grid height.
        width: Current grid width.
        n_learning: Total number of learning iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GrowingGrid
    >>> X = np.random.rand(1000, 2)
    >>> gg = GrowingGrid(n_dim=2)
    >>> gg.train(X, n_iterations=5000)
    >>> nodes, edges = gg.get_graph()
    """

    def __init__(
        self,
        n_dim: int = 2,
        params: GrowingGridParams | None = None,
        seed: int | None = None,
    ):
        """Initialize Growing Grid.

        Args:
            n_dim: Dimension of input data.
            params: Growing Grid hyperparameters. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.n_dim = n_dim
        self.params = params or GrowingGridParams()
        self.rng = np.random.default_rng(seed)

        p = self.params

        # Initialize grid
        self.height = p.initial_height
        self.width = p.initial_width

        # Initialize weights randomly in [0, 1]
        self.weights = self.rng.random(
            (self.height, self.width, n_dim)
        ).astype(np.float32)

        # Error accumulator for each node
        self.errors = np.zeros((self.height, self.width), dtype=np.float32)

        # Pre-compute grid coordinates
        self._update_grid_coords()

        # Counters
        self.n_learning = 0
        self._n_trial = 0

    def _update_grid_coords(self) -> None:
        """Update grid coordinate array after resize."""
        self.grid_coords = np.zeros((self.height, self.width, 2))
        for i in range(self.height):
            for j in range(self.width):
                self.grid_coords[i, j] = [i, j]

    def _get_bmu(self, x: np.ndarray) -> tuple[int, int]:
        """Find the Best Matching Unit (BMU) for input x.

        Args:
            x: Input vector.

        Returns:
            Tuple of (row, col) indices of the BMU.
        """
        diff = self.weights - x
        distances = np.sum(diff ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return int(bmu_idx[0]), int(bmu_idx[1])

    def _get_neighborhood(self, bmu: tuple[int, int]) -> np.ndarray:
        """Compute neighborhood function values.

        Uses Manhattan distance and constant sigma (unlike SOM).

        Args:
            bmu: (row, col) of the Best Matching Unit.

        Returns:
            Array of shape (height, width) with neighborhood values.
        """
        p = self.params
        bmu_coord = np.array([bmu[0], bmu[1]])

        # Manhattan distance on grid
        diff = self.grid_coords - bmu_coord
        grid_distances = np.sum(np.abs(diff), axis=2)

        # Gaussian neighborhood with constant sigma
        return np.exp(-(grid_distances ** 2) / (2 * p.sigma ** 2))

    def _grow_grid(self) -> bool:
        """Grow the grid by adding a row or column.

        Adds row/column near the boundary node with highest error.

        Returns:
            True if grid was grown, False if max nodes reached.
        """
        p = self.params

        # Check if we can grow (need space for at least one row or column)
        min_new_size = min(self.height + 1, self.width + 1) * max(self.height, self.width)
        if min_new_size > p.max_nodes:
            return False

        # Find boundary node with maximum error
        max_error = -1.0
        grow_row = True
        grow_pos = 0

        # Check top row
        for j in range(self.width):
            if self.errors[0, j] > max_error:
                max_error = self.errors[0, j]
                grow_row = True
                grow_pos = 0  # Insert at top

        # Check bottom row
        for j in range(self.width):
            if self.errors[self.height - 1, j] > max_error:
                max_error = self.errors[self.height - 1, j]
                grow_row = True
                grow_pos = self.height  # Insert at bottom

        # Check left column
        for i in range(self.height):
            if self.errors[i, 0] > max_error:
                max_error = self.errors[i, 0]
                grow_row = False
                grow_pos = 0  # Insert at left

        # Check right column
        for i in range(self.height):
            if self.errors[i, self.width - 1] > max_error:
                max_error = self.errors[i, self.width - 1]
                grow_row = False
                grow_pos = self.width  # Insert at right

        # Check if the chosen growth would exceed max_nodes
        if grow_row:
            new_size = (self.height + 1) * self.width
        else:
            new_size = self.height * (self.width + 1)

        if new_size > p.max_nodes:
            return False

        # Grow the grid
        if grow_row:
            # Add a row
            if grow_pos == 0:
                # Insert at top: interpolate from first row
                new_row = self.weights[0, :, :] + self.rng.normal(0, 0.01, (self.width, self.n_dim))
                self.weights = np.vstack([new_row[np.newaxis, :, :], self.weights])
                new_errors = np.zeros((1, self.width))
                self.errors = np.vstack([new_errors, self.errors])
            else:
                # Insert at bottom: interpolate from last row
                new_row = self.weights[-1, :, :] + self.rng.normal(0, 0.01, (self.width, self.n_dim))
                self.weights = np.vstack([self.weights, new_row[np.newaxis, :, :]])
                new_errors = np.zeros((1, self.width))
                self.errors = np.vstack([self.errors, new_errors])
            self.height += 1
        else:
            # Add a column
            if grow_pos == 0:
                # Insert at left: interpolate from first column
                new_col = self.weights[:, 0, :] + self.rng.normal(0, 0.01, (self.height, self.n_dim))
                self.weights = np.concatenate([new_col[:, np.newaxis, :], self.weights], axis=1)
                new_errors = np.zeros((self.height, 1))
                self.errors = np.concatenate([new_errors, self.errors], axis=1)
            else:
                # Insert at right: interpolate from last column
                new_col = self.weights[:, -1, :] + self.rng.normal(0, 0.01, (self.height, self.n_dim))
                self.weights = np.concatenate([self.weights, new_col[:, np.newaxis, :]], axis=1)
                new_errors = np.zeros((self.height, 1))
                self.errors = np.concatenate([self.errors, new_errors], axis=1)
            self.width += 1

        # Update grid coordinates
        self._update_grid_coords()

        return True

    def _one_train_update(self, sample: np.ndarray) -> None:
        """Single training iteration.

        Args:
            sample: Input sample vector.
        """
        p = self.params

        # Decay all errors
        self.errors *= (1 - p.tau)

        # Find BMU
        bmu = self._get_bmu(sample)

        # Accumulate error at BMU
        dist_sq = np.sum((sample - self.weights[bmu[0], bmu[1]]) ** 2)
        self.errors[bmu[0], bmu[1]] += dist_sq

        # Get neighborhood values
        h = self._get_neighborhood(bmu)

        # Update weights
        diff = sample - self.weights
        # Winner gets eps_b, neighbors get eps_n scaled by neighborhood
        lr = np.where(
            (self.grid_coords[:, :, 0] == bmu[0]) &
            (self.grid_coords[:, :, 1] == bmu[1]),
            p.eps_b,
            p.eps_n * h
        )
        self.weights += lr[:, :, np.newaxis] * diff

        # Periodically grow grid
        self._n_trial += 1
        if self._n_trial >= p.lambda_:
            self._n_trial = 0
            self._grow_grid()

        self.n_learning += 1

    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        callback: Callable[[GrowingGrid, int], None] | None = None,
    ) -> GrowingGrid:
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

    def partial_fit(self, sample: np.ndarray) -> GrowingGrid:
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
        """Number of neurons."""
        return self.height * self.width

    @property
    def n_edges(self) -> int:
        """Number of grid edges."""
        h_edges = self.height * (self.width - 1)
        v_edges = (self.height - 1) * self.width
        return h_edges + v_edges

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get current graph structure (grid topology).

        Returns:
            Tuple of:
                - nodes: Array of shape (n_nodes, n_dim) with positions.
                - edges: List of (i, j) tuples representing grid connections.
        """
        # Flatten weights to node list
        nodes = self.weights.reshape(-1, self.n_dim)

        # Create edges based on grid topology
        edges = []
        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j

                # Right neighbor
                if j < self.width - 1:
                    edges.append((idx, idx + 1))

                # Bottom neighbor
                if i < self.height - 1:
                    edges.append((idx, idx + self.width))

        return nodes, edges

    def get_node_errors(self) -> np.ndarray:
        """Get error values for all nodes.

        Returns:
            Array of error values in same order as get_graph() nodes.
        """
        return self.errors.flatten()


# Alias
GG = GrowingGrid
