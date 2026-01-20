"""Growing Neural Gas (GNG) implementation.

Reference:
    Fritzke, B. (1995). A Growing Neural Gas Network Learns Topologies.
    Advances in Neural Information Processing Systems 7 (NIPS 1994).
"""

import numpy as np

import sys
from pathlib import Path

# Add python path for imports
sys.path.insert(0, str(Path(__file__).parents[3] / "python"))

from gng.core.base import BaseGNG
from gng.core.utils import find_nearest_nodes


class GNG(BaseGNG):
    """Growing Neural Gas algorithm.

    Parameters
    ----------
    lambda_ : int
        Number of steps between node insertions.
    eps_b : float
        Learning rate for the winning node.
    eps_n : float
        Learning rate for neighbors of the winning node.
    alpha : float
        Error reduction factor when inserting a new node.
    beta : float
        Global error decay factor.
    max_age : int
        Maximum age for edges before removal.
    max_nodes : int or None
        Maximum number of nodes. None for unlimited.

    Examples
    --------
    >>> import numpy as np
    >>> from model import GNG
    >>> X = np.random.rand(1000, 2)
    >>> gng = GNG()
    >>> gng.fit(X, epochs=10)
    >>> nodes, edges = gng.get_graph()
    """

    def __init__(
        self,
        lambda_: int = 100,
        eps_b: float = 0.2,
        eps_n: float = 0.006,
        alpha: float = 0.5,
        beta: float = 0.0005,
        max_age: int = 50,
        max_nodes: int | None = None,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.alpha = alpha
        self.beta = beta
        self.max_age = max_age
        self.max_nodes = max_nodes

        self.errors: np.ndarray | None = None
        self._step = 0

    def _initialize(self, X: np.ndarray) -> None:
        """Initialize with two random nodes from the data."""
        self.dim = X.shape[1]
        indices = np.random.choice(len(X), 2, replace=False)
        self.nodes = X[indices].copy()
        self.errors = np.zeros(2)
        self.edges = {(0, 1): 0}

    def fit(self, X: np.ndarray, epochs: int = 1) -> "GNG":
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        epochs : int
            Number of passes through the data.

        Returns
        -------
        self
        """
        if self.nodes is None:
            self._initialize(X)

        for _ in range(epochs):
            indices = np.random.permutation(len(X))
            for idx in indices:
                self.partial_fit(X[idx])

        self._is_fitted = True
        return self

    def partial_fit(self, x: np.ndarray) -> "GNG":
        """Incrementally fit with a single sample.

        Parameters
        ----------
        x : np.ndarray
            Single input sample of shape (n_features,).

        Returns
        -------
        self
        """
        if self.nodes is None:
            raise ValueError("Model must be initialized first. Call fit() with data.")

        # Step 1: Find two nearest nodes
        indices, distances = find_nearest_nodes(x, self.nodes, k=2)
        s1, s2 = indices[0], indices[1]

        # Step 2: Update error of winner
        self.errors[s1] += distances[0] ** 2

        # Step 3: Move winner and neighbors toward input
        self.nodes[s1] += self.eps_b * (x - self.nodes[s1])

        for neighbor in self._get_neighbors(s1):
            self.nodes[neighbor] += self.eps_n * (x - self.nodes[neighbor])

        # Step 4: Update edge between s1 and s2
        edge = self._make_edge(s1, s2)
        self.edges[edge] = 0

        # Step 5: Increment age of all edges from s1
        edges_to_remove = []
        for e in list(self.edges.keys()):
            if s1 in e:
                self.edges[e] += 1
                if self.edges[e] > self.max_age:
                    edges_to_remove.append(e)

        # Step 6: Remove old edges
        for e in edges_to_remove:
            del self.edges[e]

        # Step 7: Remove isolated nodes
        self._remove_isolated_nodes()

        # Step 8: Insert new node
        self._step += 1
        if self._step % self.lambda_ == 0:
            if self.max_nodes is None or self.n_nodes < self.max_nodes:
                self._insert_node()

        # Step 9: Decay all errors
        self.errors *= 1 - self.beta

        return self

    def _get_neighbors(self, node: int) -> list[int]:
        """Get all neighbors of a node."""
        neighbors = []
        for e in self.edges.keys():
            if e[0] == node:
                neighbors.append(e[1])
            elif e[1] == node:
                neighbors.append(e[0])
        return neighbors

    def _make_edge(self, i: int, j: int) -> tuple[int, int]:
        """Create a canonical edge representation (smaller index first)."""
        return (min(i, j), max(i, j))

    def _remove_isolated_nodes(self) -> None:
        """Remove nodes with no edges."""
        connected = set()
        for e in self.edges.keys():
            connected.add(e[0])
            connected.add(e[1])

        isolated = [i for i in range(self.n_nodes) if i not in connected]

        if isolated:
            # Remove from end to preserve indices
            for i in sorted(isolated, reverse=True):
                self._remove_node(i)

    def _remove_node(self, idx: int) -> None:
        """Remove a node and update edge indices."""
        self.nodes = np.delete(self.nodes, idx, axis=0)
        self.errors = np.delete(self.errors, idx)

        # Update edge indices
        new_edges = {}
        for (i, j), age in self.edges.items():
            new_i = i if i < idx else i - 1
            new_j = j if j < idx else j - 1
            if new_i >= 0 and new_j >= 0:
                new_edges[self._make_edge(new_i, new_j)] = age
        self.edges = new_edges

    def _insert_node(self) -> None:
        """Insert a new node between the node with highest error and its worst neighbor."""
        if self.n_nodes < 2:
            return

        # Find node with maximum error
        q = int(np.argmax(self.errors))

        # Find neighbor of q with maximum error
        neighbors = self._get_neighbors(q)
        if not neighbors:
            return

        f = max(neighbors, key=lambda n: self.errors[n])

        # Create new node between q and f
        new_node = 0.5 * (self.nodes[q] + self.nodes[f])
        self.nodes = np.vstack([self.nodes, new_node])
        self.errors = np.append(self.errors, 0.0)
        r = self.n_nodes - 1

        # Update edges
        edge_qf = self._make_edge(q, f)
        if edge_qf in self.edges:
            del self.edges[edge_qf]
        self.edges[self._make_edge(q, r)] = 0
        self.edges[self._make_edge(f, r)] = 0

        # Update errors
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.errors[r] = self.errors[q]
