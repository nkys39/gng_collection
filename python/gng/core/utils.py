"""Utility functions for GNG algorithms."""

import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        a: First point.
        b: Second point.

    Returns:
        Euclidean distance.
    """
    return float(np.linalg.norm(a - b))


def euclidean_distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate squared Euclidean distance between two points.

    More efficient when only comparing distances (avoids sqrt).

    Args:
        a: First point.
        b: Second point.

    Returns:
        Squared Euclidean distance.
    """
    diff = a - b
    return float(np.dot(diff, diff))


def find_nearest_nodes(
    x: np.ndarray, nodes: np.ndarray, k: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Find k nearest nodes to a given point.

    Args:
        x: Query point of shape (n_features,).
        nodes: Array of node positions with shape (n_nodes, n_features).
        k: Number of nearest nodes to find.

    Returns:
        Tuple of (indices, distances) for k nearest nodes.
    """
    distances = np.linalg.norm(nodes - x, axis=1)
    indices = np.argsort(distances)[:k]
    return indices, distances[indices]


def calculate_quantization_error(X: np.ndarray, nodes: np.ndarray) -> float:
    """Calculate mean quantization error for a dataset.

    Args:
        X: Input data of shape (n_samples, n_features).
        nodes: Node positions of shape (n_nodes, n_features).

    Returns:
        Mean distance from each point to its nearest node.
    """
    total_error = 0.0
    for x in X:
        distances = np.linalg.norm(nodes - x, axis=1)
        total_error += np.min(distances)
    return total_error / len(X)
