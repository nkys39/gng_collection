"""Base class for GNG algorithms."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseGNG(ABC):
    """Abstract base class for all GNG algorithm variants.

    This class defines the common interface that all GNG implementations
    should follow to ensure consistency across different algorithms.
    """

    def __init__(self, dim: Optional[int] = None):
        """Initialize the GNG base.

        Args:
            dim: Dimensionality of input data. If None, will be inferred from data.
        """
        self.dim = dim
        self.nodes: Optional[np.ndarray] = None
        self.edges: dict[tuple[int, int], int] = {}
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, epochs: int = 1) -> "BaseGNG":
        """Fit the model to the data.

        Args:
            X: Input data of shape (n_samples, n_features).
            epochs: Number of passes through the data.

        Returns:
            self
        """
        pass

    @abstractmethod
    def partial_fit(self, x: np.ndarray) -> "BaseGNG":
        """Incrementally fit the model with a single sample.

        Args:
            x: Single input sample of shape (n_features,).

        Returns:
            self
        """
        pass

    def get_nodes(self) -> np.ndarray:
        """Get current node positions.

        Returns:
            Array of node positions with shape (n_nodes, n_features).
        """
        if self.nodes is None:
            raise ValueError("Model has not been fitted yet.")
        return self.nodes.copy()

    def get_edges(self) -> list[tuple[int, int]]:
        """Get current edges.

        Returns:
            List of (node_i, node_j) tuples representing edges.
        """
        return list(self.edges.keys())

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get the learned graph structure.

        Returns:
            Tuple of (nodes, edges) where nodes is an array of positions
            and edges is a list of node index pairs.
        """
        return self.get_nodes(), self.get_edges()

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the network."""
        return 0 if self.nodes is None else len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges in the network."""
        return len(self.edges)
