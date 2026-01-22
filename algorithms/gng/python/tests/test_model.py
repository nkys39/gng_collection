"""Tests for GNG implementation."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[4] / "python"))
sys.path.insert(0, str(Path(__file__).parents[1]))

from model import GNG


class TestGNG:
    """Test cases for GNG algorithm."""

    def test_initialization(self):
        """Test GNG initialization."""
        gng = GNG()
        assert gng.nodes is None
        assert gng.n_nodes == 0

    def test_fit_2d(self):
        """Test fitting with 2D data."""
        np.random.seed(42)
        X = np.random.rand(500, 2)

        gng = GNG(lambda_=50, max_nodes=20)
        gng.fit(X, epochs=5)

        assert gng.n_nodes > 2
        assert gng.n_nodes <= 20
        assert gng.n_edges > 0

    def test_fit_3d(self):
        """Test fitting with 3D data."""
        np.random.seed(42)
        X = np.random.rand(500, 3)

        gng = GNG(lambda_=50)
        gng.fit(X, epochs=3)

        assert gng.n_nodes > 2
        assert gng.nodes.shape[1] == 3

    def test_get_graph(self):
        """Test getting the graph structure."""
        np.random.seed(42)
        X = np.random.rand(200, 2)

        gng = GNG(lambda_=30)
        gng.fit(X, epochs=3)

        nodes, edges = gng.get_graph()
        assert nodes.shape[0] == gng.n_nodes
        assert len(edges) == gng.n_edges

    def test_partial_fit(self):
        """Test incremental learning."""
        np.random.seed(42)
        X = np.random.rand(100, 2)

        gng = GNG(lambda_=20)
        gng.fit(X[:10], epochs=1)  # Initialize
        initial_nodes = gng.n_nodes

        for x in X[10:50]:
            gng.partial_fit(x)

        assert gng.n_nodes >= initial_nodes

    def test_cluster_data(self):
        """Test with clustered data."""
        np.random.seed(42)
        # Create 3 clusters
        c1 = np.random.randn(100, 2) * 0.5 + [0, 0]
        c2 = np.random.randn(100, 2) * 0.5 + [5, 0]
        c3 = np.random.randn(100, 2) * 0.5 + [2.5, 4]
        X = np.vstack([c1, c2, c3])

        gng = GNG(lambda_=50, max_nodes=30)
        gng.fit(X, epochs=10)

        # Should have learned some structure
        assert gng.n_nodes > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
