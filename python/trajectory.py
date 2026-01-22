"""Trajectory visualization utilities for node movement tracking.

Based on demogng.de Voronoi visualization concept.
Tracks and visualizes the movement history of nodes during training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


@dataclass
class TrajectoryTracker:
    """Tracks node position history for trajectory visualization.

    Attributes:
        max_history: Maximum number of positions to keep per node.
        node_histories: Dictionary mapping node index to position history.
        fade_alpha: Whether to fade older positions.
    """

    max_history: int = 50
    fade_alpha: bool = True
    _histories: dict[int, deque] = field(default_factory=dict)

    def update(self, nodes: np.ndarray) -> None:
        """Update trajectory with current node positions.

        Args:
            nodes: Current node positions, shape (n_nodes, n_dim).
        """
        n_nodes = len(nodes)

        # Initialize histories for new nodes
        for i in range(n_nodes):
            if i not in self._histories:
                self._histories[i] = deque(maxlen=self.max_history)

        # Update existing node positions
        for i in range(n_nodes):
            self._histories[i].append(nodes[i].copy())

        # Remove histories for nodes that no longer exist
        to_remove = [k for k in self._histories if k >= n_nodes]
        for k in to_remove:
            del self._histories[k]

    def clear(self) -> None:
        """Clear all trajectory history."""
        self._histories.clear()

    def get_trajectories(self) -> dict[int, np.ndarray]:
        """Get all trajectories as arrays.

        Returns:
            Dictionary mapping node index to trajectory array (n_points, n_dim).
        """
        return {
            idx: np.array(list(history))
            for idx, history in self._histories.items()
            if len(history) > 1
        }


def draw_trajectories(
    ax,
    tracker: TrajectoryTracker,
    color: str = "blue",
    linewidth: float = 1.0,
    alpha_start: float = 0.1,
    alpha_end: float = 0.8,
) -> None:
    """Draw node trajectories on matplotlib axes.

    Args:
        ax: Matplotlib axes.
        tracker: TrajectoryTracker with position history.
        color: Line color.
        linewidth: Line width.
        alpha_start: Alpha for oldest positions.
        alpha_end: Alpha for newest positions.
    """
    trajectories = tracker.get_trajectories()

    for idx, traj in trajectories.items():
        if len(traj) < 2:
            continue

        # Create line segments
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create alpha gradient
        n_segments = len(segments)
        if tracker.fade_alpha and n_segments > 1:
            alphas = np.linspace(alpha_start, alpha_end, n_segments)
        else:
            alphas = np.full(n_segments, alpha_end)

        # Draw each segment with its alpha
        for i, (seg, alpha) in enumerate(zip(segments, alphas)):
            ax.plot(
                seg[:, 0], seg[:, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle='round',
            )


def draw_trajectories_fast(
    ax,
    tracker: TrajectoryTracker,
    color: str = "blue",
    linewidth: float = 1.0,
    alpha: float = 0.5,
) -> LineCollection:
    """Draw trajectories using LineCollection for better performance.

    Args:
        ax: Matplotlib axes.
        tracker: TrajectoryTracker with position history.
        color: Line color.
        linewidth: Line width.
        alpha: Line alpha.

    Returns:
        LineCollection object.
    """
    trajectories = tracker.get_trajectories()

    all_segments = []
    for idx, traj in trajectories.items():
        if len(traj) < 2:
            continue
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        all_segments.extend(segments)

    if not all_segments:
        return None

    lc = LineCollection(
        all_segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection(lc)
    return lc


def create_frame_with_trajectory(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    tracker: TrajectoryTracker,
    iteration: int,
    bg_image: np.ndarray | None = None,
    title: str = "",
    show_trajectory: bool = True,
    trajectory_color: str = "blue",
) -> None:
    """Create a visualization frame with node trajectories.

    Args:
        ax: Matplotlib axes.
        points: Sample points array.
        nodes: Current node positions.
        edges: List of edge tuples.
        tracker: TrajectoryTracker for trajectory visualization.
        iteration: Current iteration number.
        bg_image: Optional background image.
        title: Plot title.
        show_trajectory: Whether to show trajectories.
        trajectory_color: Color for trajectory lines.
    """
    ax.clear()

    if bg_image is not None:
        ax.imshow(bg_image, extent=[0, 1, 1, 0], alpha=0.5)

    # Plot sample points
    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3)

    # Draw trajectories before edges and nodes
    if show_trajectory:
        draw_trajectories(
            ax, tracker,
            color=trajectory_color,
            linewidth=1.0,
            alpha_start=0.1,
            alpha_end=0.6,
        )

    # Plot edges
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "r-",
            linewidth=1.5,
            alpha=0.7,
        )

    # Plot nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(title or f"Iteration {iteration} ({len(nodes)} nodes)")
