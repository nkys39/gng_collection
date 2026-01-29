"""Furuta et al. (FSS2022) contour detection method.

Reference:
    Furuta, Y., Toda, Y., & Matsuno, T. (2022). "Growing Neural Gasを用いた
    3次元形状認識のための輪郭検出手法の検討", FSS2022, pp. 810-814

Algorithm:
    Combines binary image contour detection with GNG angle-based traversal.
    - Outer contour: Start from Y_min node, CCW traversal
    - Inner contour: Find nodes with angle gap >= threshold, then CCW traversal

Paper correspondence:
    - Direction update: V_new = (V_old - 225) mod 360 where V_old = angle to found neighbor
    - CCW nearest: Select neighbor with smallest (angle - direction) % 360
    - Outer start: Y_min node, direction 180°
    - Inner threshold: θ_thv = 135°

Implementation extensions (not in paper):
    The paper identifies stability issues but provides no countermeasures.
    This implementation adds:
    - Start node requires 2+ neighbors (CCW traversal needs choices)
    - Exclude previous node from search (prevent immediate backtrack)
    - Minimum contour length check (prevent premature completion)
    - Sub-loop detection (force termination on revisit)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContourResult:
    """Result of contour detection."""

    outer_contour: list[int] = field(default_factory=list)  # Outer contour node IDs
    inner_contours: list[list[int]] = field(
        default_factory=list
    )  # List of inner contours
    contour_flags: np.ndarray | None = None  # (N,) array of contour flags

    @property
    def all_contour_nodes(self) -> list[int]:
        """Get all contour node IDs."""
        nodes = set(self.outer_contour)
        for inner in self.inner_contours:
            nodes.update(inner)
        return list(nodes)


@dataclass
class FurutaContourParams:
    """Parameters for Furuta contour detection."""

    angle_threshold: float = 135.0  # Inner contour detection threshold (degrees)
    max_iterations: int = 1000  # Max iterations for CCW traversal (safety)
    debug: bool = False  # Enable debug output


class FurutaContourDetector:
    """Furuta et al. (FSS2022) CCW traversal-based contour detection.

    This method traces contours using counter-clockwise traversal,
    similar to binary image contour detection algorithms.
    """

    def __init__(self, params: FurutaContourParams | None = None):
        self.params = params or FurutaContourParams()

    def detect(
        self,
        nodes: np.ndarray,
        pedge: np.ndarray,
        traversability: np.ndarray,
    ) -> ContourResult:
        """Detect contour nodes using CCW traversal method.

        Args:
            nodes: Node positions (N, 3) - uses X, Y only
            pedge: Traversability edge adjacency matrix (N, N)
            traversability: Traversability flags (N,)

        Returns:
            ContourResult with outer and inner contours
        """
        # Get traversable node indices
        trav_indices = np.where(traversability == 1)[0]

        if len(trav_indices) < 3:
            return ContourResult()

        # Build adjacency for traversable nodes only
        trav_pedge = self._build_traversable_pedge(pedge, trav_indices)

        # Detect outer contour
        outer_contour = self._detect_outer_contour(nodes, trav_pedge, trav_indices)

        # Detect inner contours
        inner_contours = self._detect_inner_contours(
            nodes, trav_pedge, trav_indices, outer_contour
        )

        # Build contour flags array
        n_nodes = len(nodes)
        contour_flags = np.zeros(n_nodes, dtype=int)
        for node_id in outer_contour:
            contour_flags[node_id] = 1
        for inner in inner_contours:
            for node_id in inner:
                contour_flags[node_id] = 1

        return ContourResult(
            outer_contour=outer_contour,
            inner_contours=inner_contours,
            contour_flags=contour_flags,
        )

    def _build_traversable_pedge(
        self, pedge: np.ndarray, trav_indices: np.ndarray
    ) -> dict[int, list[int]]:
        """Build pedge adjacency dict for traversable nodes only.

        Args:
            pedge: Full pedge adjacency matrix (N, N)
            trav_indices: Indices of traversable nodes

        Returns:
            Dict mapping node_id -> list of pedge neighbor node_ids
        """
        trav_set = set(trav_indices)
        adjacency = {}

        for i in trav_indices:
            neighbors = []
            for j in trav_indices:
                if i != j and pedge[i, j] == 1:
                    neighbors.append(j)
            adjacency[int(i)] = neighbors

        return adjacency

    def _detect_outer_contour(
        self,
        nodes: np.ndarray,
        trav_pedge: dict[int, list[int]],
        trav_indices: np.ndarray,
    ) -> list[int]:
        """Detect outer contour using CCW traversal.

        Algorithm:
            1. Sort traversable nodes by Y coordinate
            2. Start from Y_min node with direction 180 degrees
            3. Find next node in CCW direction
            4. Update direction: V_new = (V_old - 225) mod 360
            5. Repeat until returning to start node

        Args:
            nodes: Node positions (N, 3)
            trav_pedge: Traversable pedge adjacency
            trav_indices: Indices of traversable nodes

        Returns:
            List of node IDs forming the outer contour
        """
        if len(trav_indices) < 3:
            return []

        # Step 1: Find a good starting node
        # Original paper uses Y_min, but we need a node with at least 2 neighbors
        # to enable CCW traversal
        y_coords = nodes[trav_indices, 1]
        sorted_order = np.argsort(y_coords)

        # Find the first node (by Y_min) that has at least 2 neighbors
        start_node = None
        for idx in sorted_order:
            candidate = int(trav_indices[idx])
            if candidate in trav_pedge and len(trav_pedge[candidate]) >= 2:
                start_node = candidate
                break

        # Fallback to Y_min if no good candidate found
        if start_node is None:
            start_node = int(trav_indices[sorted_order[0]])

        # Check if start node has neighbors
        if start_node not in trav_pedge or len(trav_pedge[start_node]) == 0:
            return [start_node]

        # Step 2: Initialize CCW traversal
        contour = [start_node]
        current = start_node
        direction = 180.0  # Start searching from 180 degrees (left)

        if self.params.debug:
            print(f"  Start node: {start_node} at Y={nodes[start_node, 1]:.3f}")
            print(f"  Neighbors: {trav_pedge.get(start_node, [])}")

        # Step 3-5: CCW traversal
        visited_in_traversal = {start_node}
        termination_reason = "max_iterations"
        prev_node = None  # Track previous node to avoid going back

        min_contour_length = 3  # Minimum nodes before allowing return to start

        for iteration in range(self.params.max_iterations):
            # Only allow returning to start after visiting minimum nodes
            allow_start = start_node if len(contour) >= min_contour_length else None

            next_node, new_direction = self._find_next_ccw(
                current, direction, nodes, trav_pedge,
                exclude_node=prev_node, allow_node=allow_start
            )

            if self.params.debug and iteration < 10:
                print(f"  Iter {iteration}: current={current}, dir={direction:.1f}° -> next={next_node}, new_dir={new_direction:.1f}° (allow_start={allow_start is not None})")

            if next_node is None:
                # No neighbor found - dead end
                termination_reason = "dead_end"
                break

            if next_node == start_node and len(contour) >= min_contour_length:
                # Completed the loop successfully
                termination_reason = "completed"
                break

            if next_node in visited_in_traversal:
                # Stuck in a sub-loop (not returning to start)
                # This is a stability issue - break to avoid infinite loop
                termination_reason = f"sub_loop_at_{next_node}"
                break

            visited_in_traversal.add(next_node)
            contour.append(next_node)
            prev_node = current
            current = next_node
            direction = new_direction

        if self.params.debug:
            print(f"  Termination: {termination_reason}, contour length: {len(contour)}")

        return contour

    def _find_next_ccw(
        self,
        current: int,
        direction: float,
        nodes: np.ndarray,
        trav_pedge: dict[int, list[int]],
        exclude_node: int | None = None,
        allow_node: int | None = None,
    ) -> tuple[int | None, float]:
        """Find next node in counter-clockwise direction.

        Args:
            current: Current node ID
            direction: Current search direction (degrees)
            nodes: Node positions
            trav_pedge: Traversable pedge adjacency
            exclude_node: Node to exclude from search (typically previous node)
            allow_node: Node that is always allowed (typically start node)

        Returns:
            (next_node_id, new_direction) or (None, 0) if no neighbor found
        """
        if current not in trav_pedge:
            return None, 0.0

        # Filter neighbors: exclude previous node but always allow start node
        neighbors = [
            n for n in trav_pedge[current]
            if n != exclude_node or n == allow_node
        ]
        if len(neighbors) == 0:
            return None, 0.0

        # Calculate angle to each neighbor
        current_pos = nodes[current]
        angle_list = []

        for neighbor in neighbors:
            neighbor_pos = nodes[neighbor]
            dx = neighbor_pos[0] - current_pos[0]
            dy = neighbor_pos[1] - current_pos[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle += 360.0
            angle_list.append((neighbor, angle))

        # Find the neighbor with smallest CCW angle from direction
        best_node = None
        best_ccw_angle = float("inf")
        best_angle = 0.0  # Angle to the best neighbor

        for neighbor, angle in angle_list:
            # Calculate CCW angle from direction
            ccw_angle = (angle - direction) % 360.0
            if ccw_angle < best_ccw_angle:
                best_ccw_angle = ccw_angle
                best_node = neighbor
                best_angle = angle

        if best_node is None:
            return None, 0.0

        # Update direction: V_new = (V_old - 225) mod 360
        # V_old is the angle TO the found neighbor (not the search direction)
        # This means: look back from where we came (angle + 180) then go 45° CCW (- 45°)
        # = angle + 180 - 45 = angle + 135 = angle - 225 (mod 360)
        new_direction = (best_angle - 225.0) % 360.0

        return best_node, new_direction

    def _detect_inner_contours(
        self,
        nodes: np.ndarray,
        trav_pedge: dict[int, list[int]],
        trav_indices: np.ndarray,
        outer_contour: list[int],
    ) -> list[list[int]]:
        """Detect inner contours (holes) using CCW traversal.

        Algorithm:
            1. Raster scan from Y_min to Y_max
            2. For each unvisited node, check if angle gap >= threshold
            3. If yes, start CCW traversal for inner contour

        Args:
            nodes: Node positions (N, 3)
            trav_pedge: Traversable pedge adjacency
            trav_indices: Indices of traversable nodes
            outer_contour: Already detected outer contour

        Returns:
            List of inner contours (each is a list of node IDs)
        """
        inner_contours = []
        visited = set(outer_contour)

        # Sort by Y for raster scan
        y_coords = nodes[trav_indices, 1]
        sorted_order = np.argsort(y_coords)

        for idx in sorted_order:
            node_id = int(trav_indices[idx])

            if node_id in visited:
                continue

            # Check if this node has angle gap >= threshold (potential inner contour start)
            if self._has_large_angle_gap(node_id, nodes, trav_pedge):
                # Start inner contour detection
                inner = self._trace_inner_contour(node_id, nodes, trav_pedge, visited)
                if len(inner) > 2:
                    inner_contours.append(inner)
                    visited.update(inner)

            visited.add(node_id)

        return inner_contours

    def _has_large_angle_gap(
        self, node_id: int, nodes: np.ndarray, trav_pedge: dict[int, list[int]]
    ) -> bool:
        """Check if node has angle gap >= threshold.

        Args:
            node_id: Node to check
            nodes: Node positions
            trav_pedge: Traversable pedge adjacency

        Returns:
            True if has large angle gap
        """
        if node_id not in trav_pedge:
            return False

        neighbors = trav_pedge[node_id]
        if len(neighbors) < 2:
            return False

        # Calculate angles to neighbors
        node_pos = nodes[node_id]
        angles = []

        for neighbor in neighbors:
            neighbor_pos = nodes[neighbor]
            dx = neighbor_pos[0] - node_pos[0]
            dy = neighbor_pos[1] - node_pos[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle += 360.0
            angles.append(angle)

        angles = sorted(angles)

        # Check wrap-around gap
        wrap_gap = 360.0 - angles[-1] + angles[0]
        if wrap_gap >= self.params.angle_threshold:
            return True

        # Check consecutive gaps
        for i in range(len(angles) - 1):
            gap = angles[i + 1] - angles[i]
            if gap >= self.params.angle_threshold:
                return True

        return False

    def _trace_inner_contour(
        self,
        start_node: int,
        nodes: np.ndarray,
        trav_pedge: dict[int, list[int]],
        visited: set[int],
    ) -> list[int]:
        """Trace inner contour starting from given node.

        Args:
            start_node: Starting node ID
            nodes: Node positions
            trav_pedge: Traversable pedge adjacency
            visited: Already visited nodes

        Returns:
            List of node IDs forming the inner contour
        """
        if start_node not in trav_pedge:
            return []

        # Find initial direction (towards the neighbor with largest gap)
        neighbors = trav_pedge[start_node]
        if len(neighbors) == 0:
            return [start_node]

        # Calculate angles and find the gap start direction
        node_pos = nodes[start_node]
        angle_list = []

        for neighbor in neighbors:
            neighbor_pos = nodes[neighbor]
            dx = neighbor_pos[0] - node_pos[0]
            dy = neighbor_pos[1] - node_pos[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle += 360.0
            angle_list.append((neighbor, angle))

        # Sort by angle
        angle_list.sort(key=lambda x: x[1])

        # Find largest gap and use its end as start direction
        max_gap = 0
        direction = 0.0

        # Check wrap-around
        wrap_gap = 360.0 - angle_list[-1][1] + angle_list[0][1]
        if wrap_gap > max_gap:
            max_gap = wrap_gap
            direction = angle_list[0][1]

        # Check consecutive
        for i in range(len(angle_list) - 1):
            gap = angle_list[i + 1][1] - angle_list[i][1]
            if gap > max_gap:
                max_gap = gap
                direction = angle_list[i + 1][1]

        # Trace contour
        contour = [start_node]
        current = start_node

        for _ in range(self.params.max_iterations):
            next_node, new_direction = self._find_next_ccw(
                current, direction, nodes, trav_pedge
            )

            if next_node is None:
                break

            if next_node == start_node:
                break

            contour.append(next_node)
            current = next_node
            direction = new_direction

        return contour

    def get_contour_nodes(
        self, nodes: np.ndarray, result: ContourResult
    ) -> np.ndarray:
        """Get positions of contour nodes.

        Args:
            nodes: Node positions (N, 3)
            result: ContourResult from detect()

        Returns:
            Contour node positions (M, 3)
        """
        if result.contour_flags is None:
            return np.array([])
        return nodes[result.contour_flags == 1]

    def get_ordered_contour_positions(
        self, nodes: np.ndarray, result: ContourResult
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Get ordered contour positions for line drawing.

        Args:
            nodes: Node positions (N, 3)
            result: ContourResult from detect()

        Returns:
            (outer_positions, list of inner_positions)
            Each is (M, 3) array in traversal order
        """
        outer_pos = nodes[result.outer_contour] if result.outer_contour else np.array([])
        inner_pos_list = [nodes[inner] for inner in result.inner_contours]
        return outer_pos, inner_pos_list
