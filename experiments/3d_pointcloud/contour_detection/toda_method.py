"""Toda et al. (2021) contour detection method.

Reference:
    Toda, Y. et al. (2021). "Growing Neural Gas に基づく環境の
    トポロジカルマップの構築と未知環境における経路計画",
    知能と情報, Vol. 33, No. 4, pp. 872-884

Algorithm:
    1. Calculate angles to pedge neighbors
    2. Sort angles and check gaps between consecutive angles
    3. If gap >= 135 degrees, mark as contour node
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TodaContourParams:
    """Parameters for Toda contour detection."""

    gap_threshold: float = 135.0  # Angular gap threshold (degrees)


class TodaContourDetector:
    """Toda et al. (2021) angle gap-based contour detection.

    This is the conventional method that checks each node independently
    for angular gaps between its pedge neighbors.
    """

    def __init__(self, params: TodaContourParams | None = None):
        self.params = params or TodaContourParams()

    def detect(
        self,
        nodes: np.ndarray,
        pedge: np.ndarray,
        traversability: np.ndarray,
    ) -> np.ndarray:
        """Detect contour nodes using angle gap method.

        Args:
            nodes: Node positions (N, 3) - uses X, Y only
            pedge: Traversability edge adjacency matrix (N, N)
            traversability: Traversability flags (N,)

        Returns:
            contour: Contour flags (N,) - 1 if contour, 0 otherwise
        """
        n_nodes = len(nodes)
        contour = np.zeros(n_nodes, dtype=int)

        for i in range(n_nodes):
            # Only check traversable nodes
            if traversability[i] != 1:
                continue

            contour[i] = self._judge_contour(i, nodes, pedge)

        return contour

    def _judge_contour(
        self, node_id: int, nodes: np.ndarray, pedge: np.ndarray
    ) -> int:
        """Judge if a node is on the contour based on angular gaps.

        Args:
            node_id: Node ID to check
            nodes: Node positions (N, 3)
            pedge: Traversability edge adjacency matrix (N, N)

        Returns:
            1 if contour node, 0 otherwise
        """
        # Collect pedge neighbors
        neighbors = np.where(pedge[node_id] == 1)[0]
        neighbors = neighbors[neighbors != node_id]

        if len(neighbors) < 2:
            return 0

        # Calculate angles to each neighbor (using X, Y only)
        node_pos = nodes[node_id]
        angles = []
        for neighbor_id in neighbors:
            neighbor_pos = nodes[neighbor_id]
            dx = neighbor_pos[0] - node_pos[0]
            dy = neighbor_pos[1] - node_pos[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle = 360.0 + angle
            angles.append(angle)

        # Sort angles
        angles = sorted(angles)

        # Check for gaps >= threshold
        # First check wrap-around gap
        wrap_gap = abs(360.0 - angles[-1] + angles[0])
        if wrap_gap >= self.params.gap_threshold:
            return 1

        # Check consecutive gaps
        for j in range(len(angles) - 1):
            gap = abs(angles[j + 1] - angles[j])
            if gap >= self.params.gap_threshold:
                return 1

        return 0

    def get_contour_nodes(
        self, nodes: np.ndarray, contour: np.ndarray
    ) -> np.ndarray:
        """Get positions of contour nodes.

        Args:
            nodes: Node positions (N, 3)
            contour: Contour flags (N,)

        Returns:
            Contour node positions (M, 3)
        """
        return nodes[contour == 1]
