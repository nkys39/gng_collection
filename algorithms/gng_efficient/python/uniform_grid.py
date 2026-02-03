"""Uniform Grid for efficient nearest neighbor search.

Based on Section 4 of:
    Fišer, D., Faigl, J., & Kulich, M. (2013).
    "Growing Neural Gas Efficiently"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import NeuronNode


@dataclass
class UniformGridParams:
    """Parameters for Growing Uniform Grid.

    Attributes:
        h_t: Allowed maximal density (average nodes per cell threshold).
        h_rho: Expansion factor (ratio of cells in new grid vs old).
    """

    h_t: float = 0.1
    h_rho: float = 1.5


class GrowingUniformGrid:
    """Growing Uniform Grid for nearest neighbor search.

    Implements the algorithm from Section 4.1 and 4.2 of Fišer et al. (2013).

    The grid partitions D-dimensional space into uniform cells.
    Each cell maintains a list of nodes whose weight vectors fall within it.

    Key operations:
        - insert/remove: O(1)
        - find_two_nearest: O(1) average case, O(n) worst case

    The grid automatically expands when density exceeds threshold.
    """

    def __init__(
        self,
        n_dim: int,
        params: UniformGridParams | None = None,
        initial_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """Initialize the Growing Uniform Grid.

        Args:
            n_dim: Dimensionality of the space.
            params: Grid parameters (h_t, h_rho).
            initial_bounds: Optional (min_coords, max_coords) for initial grid.
        """
        self.n_dim = n_dim
        self.params = params or UniformGridParams()

        # Grid state
        self.origin: np.ndarray = np.zeros(n_dim, dtype=np.float64)
        self.cell_size: float = 1.0
        self.grid_dims: np.ndarray = np.ones(n_dim, dtype=np.int32)

        # Cell storage: dict mapping cell index tuple to list of nodes
        self.cells: dict[tuple, list[NeuronNode]] = {}

        # Node to cell mapping for O(1) updates
        self.node_cell_map: dict[int, tuple] = {}

        # Statistics
        self.n_nodes: int = 0

        if initial_bounds is not None:
            self._initialize_grid(initial_bounds[0], initial_bounds[1])

    def _initialize_grid(self, min_coords: np.ndarray, max_coords: np.ndarray) -> None:
        """Initialize grid covering the given bounding box.

        Args:
            min_coords: Minimum coordinates of bounding box.
            max_coords: Maximum coordinates of bounding box.
        """
        self.origin = min_coords.copy().astype(np.float64)

        # Start with a single cell covering the bounding box
        extent = max_coords - min_coords
        self.cell_size = float(np.max(extent)) + 1e-6  # Avoid zero
        self.grid_dims = np.ones(self.n_dim, dtype=np.int32)
        self.cells = {}
        self.node_cell_map = {}

    def _get_cell_coords(self, position: np.ndarray) -> tuple:
        """Compute cell coordinates for a position.

        From Eq. (1): p = floor((w - o) / l)

        Args:
            position: Position vector.

        Returns:
            Tuple of cell coordinates.
        """
        coords = np.floor((position - self.origin) / self.cell_size).astype(np.int32)
        # Clamp to grid bounds
        coords = np.clip(coords, 0, self.grid_dims - 1)
        return tuple(coords)

    def _get_density(self) -> float:
        """Calculate current grid density (average nodes per cell)."""
        n_cells = int(np.prod(self.grid_dims))
        if n_cells == 0:
            return float("inf")
        return self.n_nodes / n_cells

    def _rebuild_grid(self) -> None:
        """Rebuild grid with more cells.

        Called when density exceeds threshold h_t.
        Creates new grid with h_rho times more cells.
        """
        if self.n_nodes == 0:
            return

        # Collect all nodes
        all_nodes = []
        for node_list in self.cells.values():
            all_nodes.extend(node_list)

        if not all_nodes:
            return

        # Compute new grid parameters
        # New number of cells = h_rho * old number of cells
        old_n_cells = int(np.prod(self.grid_dims))
        new_n_cells = int(old_n_cells * self.params.h_rho)

        # Distribute cells across dimensions
        # For simplicity, increase each dimension proportionally
        scale_factor = self.params.h_rho ** (1.0 / self.n_dim)
        new_grid_dims = np.maximum(
            np.ceil(self.grid_dims * scale_factor).astype(np.int32), 1
        )

        # Update cell size
        old_extent = self.grid_dims * self.cell_size
        new_cell_size = np.min(old_extent / new_grid_dims)

        # Update grid state
        self.grid_dims = new_grid_dims
        self.cell_size = float(new_cell_size)
        self.cells = {}
        self.node_cell_map = {}

        # Re-insert all nodes
        for node in all_nodes:
            cell_coords = self._get_cell_coords(node.weight)
            if cell_coords not in self.cells:
                self.cells[cell_coords] = []
            self.cells[cell_coords].append(node)
            self.node_cell_map[node.id] = cell_coords

    def insert(self, node: NeuronNode) -> None:
        """Insert a node into the grid.

        O(1) time complexity.

        Args:
            node: Node to insert.
        """
        if self.n_nodes == 0:
            # First node: initialize grid around it
            self.origin = node.weight.copy() - 0.5
            self.cell_size = 1.0
            self.grid_dims = np.ones(self.n_dim, dtype=np.int32)

        # Expand grid if node is outside current bounds
        self._expand_if_needed(node.weight)

        cell_coords = self._get_cell_coords(node.weight)

        if cell_coords not in self.cells:
            self.cells[cell_coords] = []
        self.cells[cell_coords].append(node)
        self.node_cell_map[node.id] = cell_coords
        self.n_nodes += 1

        # Check if rebuild is needed
        if self._get_density() > self.params.h_t:
            self._rebuild_grid()

    def _expand_if_needed(self, position: np.ndarray) -> None:
        """Expand grid if position is outside current bounds."""
        # Calculate extent
        max_coords = self.origin + self.grid_dims * self.cell_size

        needs_expand = False
        new_origin = self.origin.copy()
        new_dims = self.grid_dims.copy()

        for d in range(self.n_dim):
            if position[d] < self.origin[d]:
                # Expand in negative direction
                extra_cells = int(
                    np.ceil((self.origin[d] - position[d]) / self.cell_size)
                )
                new_origin[d] -= extra_cells * self.cell_size
                new_dims[d] += extra_cells
                needs_expand = True
            elif position[d] >= max_coords[d]:
                # Expand in positive direction
                extra_cells = int(
                    np.ceil((position[d] - max_coords[d] + 1e-6) / self.cell_size)
                )
                new_dims[d] += extra_cells
                needs_expand = True

        if needs_expand:
            # Collect all nodes
            all_nodes = []
            for node_list in self.cells.values():
                all_nodes.extend(node_list)

            # Update grid
            self.origin = new_origin
            self.grid_dims = new_dims
            self.cells = {}
            self.node_cell_map = {}

            # Re-insert nodes
            for node in all_nodes:
                cell_coords = self._get_cell_coords(node.weight)
                if cell_coords not in self.cells:
                    self.cells[cell_coords] = []
                self.cells[cell_coords].append(node)
                self.node_cell_map[node.id] = cell_coords

    def remove(self, node: NeuronNode) -> None:
        """Remove a node from the grid.

        O(1) average time complexity.

        Args:
            node: Node to remove.
        """
        if node.id not in self.node_cell_map:
            return

        cell_coords = self.node_cell_map[node.id]
        if cell_coords in self.cells:
            try:
                self.cells[cell_coords].remove(node)
                if not self.cells[cell_coords]:
                    del self.cells[cell_coords]
            except ValueError:
                pass

        del self.node_cell_map[node.id]
        self.n_nodes -= 1

    def update(self, node: NeuronNode) -> None:
        """Update a node's position in the grid.

        O(1) time complexity (remove + insert without density check).

        Args:
            node: Node with updated weight.
        """
        if node.id not in self.node_cell_map:
            self.insert(node)
            return

        old_cell = self.node_cell_map[node.id]
        new_cell = self._get_cell_coords(node.weight)

        if old_cell == new_cell:
            return  # No change needed

        # Remove from old cell
        if old_cell in self.cells:
            try:
                self.cells[old_cell].remove(node)
                if not self.cells[old_cell]:
                    del self.cells[old_cell]
            except ValueError:
                pass

        # Expand if needed
        self._expand_if_needed(node.weight)

        # Add to new cell
        new_cell = self._get_cell_coords(node.weight)
        if new_cell not in self.cells:
            self.cells[new_cell] = []
        self.cells[new_cell].append(node)
        self.node_cell_map[node.id] = new_cell

    def find_two_nearest(
        self, point: np.ndarray
    ) -> tuple[NeuronNode | None, NeuronNode | None]:
        """Find two nearest nodes to a point.

        Implements the search procedure from Section 4.1.

        Args:
            point: Query point.

        Returns:
            Tuple of (nearest, second_nearest) nodes, or None if not found.
        """
        if self.n_nodes < 2:
            # Linear search for small grids
            nodes = []
            for node_list in self.cells.values():
                nodes.extend(node_list)
            if len(nodes) == 0:
                return None, None
            if len(nodes) == 1:
                return nodes[0], None
            # Sort by distance
            nodes.sort(key=lambda n: np.sum((n.weight - point) ** 2))
            return nodes[0], nodes[1]

        # Get cell coordinates for query point
        cell_coords = self._get_cell_coords(point)

        # Calculate boundary distance b (Eq. 2)
        # b = min distance to any cell boundary
        b = self._compute_boundary_distance(point, cell_coords)

        # Search iteratively expanding radius
        best1: NeuronNode | None = None
        best2: NeuronNode | None = None
        dist1 = float("inf")
        dist2 = float("inf")

        radius = 0
        max_radius = int(np.max(self.grid_dims))

        while radius <= max_radius:
            # Search cells at this radius
            cells_to_search = self._get_cells_at_radius(cell_coords, radius)

            for cell in cells_to_search:
                if cell not in self.cells:
                    continue
                for node in self.cells[cell]:
                    d = np.sum((node.weight - point) ** 2)
                    if d < dist1:
                        dist2 = dist1
                        best2 = best1
                        dist1 = d
                        best1 = node
                    elif d < dist2:
                        dist2 = d
                        best2 = node

            # Check if we can stop
            # We need both best1 and best2 to be within boundary + radius * cell_size
            threshold = (b + radius * self.cell_size) ** 2
            if best1 is not None and best2 is not None:
                if dist1 <= threshold and dist2 <= threshold:
                    break

            radius += 1

        return best1, best2

    def _compute_boundary_distance(
        self, point: np.ndarray, cell_coords: tuple
    ) -> float:
        """Compute minimum distance from point to cell boundary.

        From Eq. (2): b = min over dimensions of min(|xi - oi - pi*l|, |xi - oi - (pi+1)*l|)

        Args:
            point: Query point.
            cell_coords: Cell coordinates.

        Returns:
            Minimum orthogonal distance to cell boundary.
        """
        min_dist = float("inf")
        for d in range(self.n_dim):
            # Distance to lower boundary of cell
            lower = abs(point[d] - self.origin[d] - cell_coords[d] * self.cell_size)
            # Distance to upper boundary of cell
            upper = abs(
                point[d] - self.origin[d] - (cell_coords[d] + 1) * self.cell_size
            )
            min_dist = min(min_dist, lower, upper)
        return min_dist

    def _get_cells_at_radius(self, center: tuple, radius: int) -> list[tuple]:
        """Get all cell coordinates at a given Chebyshev radius from center.

        Args:
            center: Center cell coordinates.
            radius: Search radius (0 = just center cell).

        Returns:
            List of cell coordinate tuples.
        """
        if radius == 0:
            # Only check if center is within bounds
            in_bounds = all(
                0 <= center[d] < self.grid_dims[d] for d in range(self.n_dim)
            )
            return [center] if in_bounds else []

        # Generate all cells with Chebyshev distance == radius
        cells = []
        self._generate_cells_at_radius(center, radius, 0, [], cells)
        return cells

    def _generate_cells_at_radius(
        self,
        center: tuple,
        radius: int,
        dim: int,
        current: list[int],
        result: list[tuple],
    ) -> None:
        """Recursively generate cell coordinates at given radius."""
        if dim == self.n_dim:
            # Check if this cell is exactly at the specified radius
            max_diff = max(abs(current[d] - center[d]) for d in range(self.n_dim))
            if max_diff == radius:
                result.append(tuple(current))
            return

        # For each dimension, try all coordinates within range
        for offset in range(-radius, radius + 1):
            coord = center[dim] + offset
            if 0 <= coord < self.grid_dims[dim]:
                current.append(coord)
                self._generate_cells_at_radius(center, radius, dim + 1, current, result)
                current.pop()
