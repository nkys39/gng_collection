"""Lazy Heap for efficient error handling in GNG.

Based on Section 5.2 of:
    Fišer, D., Faigl, J., & Kulich, M. (2013).
    "Growing Neural Gas Efficiently"
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .model import NeuronNode


@dataclass(order=True)
class HeapEntry:
    """Entry in the lazy heap.

    Ordered by negative error (max-heap behavior with heapq's min-heap).

    Attributes:
        neg_error: Negative of the error value (for max-heap).
        cycle: Cycle counter when this entry was created.
        node_id: ID of the node.
        node: Reference to the actual node (not used for comparison).
    """

    neg_error: float
    cycle: int
    node_id: int
    node: "NeuronNode" = field(compare=False)


class LazyHeap:
    """Lazy Heap for finding the node with largest error.

    Implements the lazy heap from Section 5.2 of Fišer et al. (2013).

    Key insight: Instead of updating node positions in the heap immediately,
    we defer updates to the 'top' operation. This is efficient because
    updates happen frequently but 'top' is called only every λ steps.

    Operations:
        - insert: O(1) - just adds to pending list
        - update: O(1) - removes from heap, adds to pending list
        - top: O(k log n) where k is number of outdated nodes checked

    Attributes:
        fix_error_fn: Function to fix error value given (cycle, node).
    """

    def __init__(self, fix_error_fn: Callable[[int, "NeuronNode"], None]):
        """Initialize the lazy heap.

        Args:
            fix_error_fn: Function fix_error(cycle, node) that updates
                node.error to account for decay since node.cycle.
        """
        self.fix_error_fn = fix_error_fn

        # Main heap (min-heap of HeapEntry with negative errors for max-heap)
        self._heap: list[HeapEntry] = []

        # Pending list of nodes waiting to be inserted
        self._pending: list["NeuronNode"] = []

        # Set of node IDs currently valid in the heap
        # (used to detect stale entries)
        self._valid_ids: set[int] = set()

        # Map from node_id to whether it's in heap or pending
        self._node_locations: dict[int, str] = {}  # 'heap', 'pending', or absent

    def insert(self, node: "NeuronNode") -> None:
        """Insert a node into the heap (lazily).

        O(1) - just adds to pending list.

        Args:
            node: Node to insert.
        """
        self._pending.append(node)
        self._node_locations[node.id] = "pending"

    def update(self, node: "NeuronNode") -> None:
        """Update a node's position in the heap (lazily).

        O(1) - invalidates heap entry, adds to pending.

        Args:
            node: Node with updated error.
        """
        # Invalidate any existing entry
        self._valid_ids.discard(node.id)

        # Add to pending
        self._pending.append(node)
        self._node_locations[node.id] = "pending"

    def remove(self, node: "NeuronNode") -> None:
        """Remove a node from the heap.

        O(1) - just invalidates the entry.

        Args:
            node: Node to remove.
        """
        self._valid_ids.discard(node.id)
        self._node_locations.pop(node.id, None)

        # Also remove from pending if present
        self._pending = [n for n in self._pending if n.id != node.id]

    def top(self, current_cycle: int) -> "NeuronNode | None":
        """Get the node with the largest error.

        This is where the lazy evaluation happens:
        1. Flush pending list into heap
        2. Pop nodes until we find one with matching cycle
        3. Fix errors for outdated nodes and re-insert

        Args:
            current_cycle: Current cycle counter.

        Returns:
            Node with largest error, or None if heap is empty.
        """
        # Step 1: Flush pending list into heap
        for node in self._pending:
            if node.id < 0:  # Invalid node
                continue
            entry = HeapEntry(
                neg_error=-node.error,
                cycle=node.cycle,
                node_id=node.id,
                node=node,
            )
            heapq.heappush(self._heap, entry)
            self._valid_ids.add(node.id)
            self._node_locations[node.id] = "heap"
        self._pending.clear()

        # Step 2: Find valid node with largest error
        while self._heap:
            entry = heapq.heappop(self._heap)

            # Check if this entry is still valid
            if entry.node_id not in self._valid_ids:
                continue  # Stale entry, skip

            if entry.node.id < 0:  # Node was removed
                self._valid_ids.discard(entry.node_id)
                continue

            # Check if cycle matches
            if entry.node.cycle == current_cycle:
                # This is the correct result
                # Re-insert for next query
                heapq.heappush(self._heap, entry)
                return entry.node

            # Cycle doesn't match - fix the error and re-insert
            self.fix_error_fn(current_cycle, entry.node)

            # Create new entry with updated error
            new_entry = HeapEntry(
                neg_error=-entry.node.error,
                cycle=current_cycle,
                node_id=entry.node_id,
                node=entry.node,
            )
            heapq.heappush(self._heap, new_entry)

        return None

    def clear(self) -> None:
        """Clear the heap."""
        self._heap.clear()
        self._pending.clear()
        self._valid_ids.clear()
        self._node_locations.clear()


class SimpleLazyHeap:
    """Simpler lazy heap implementation using just a list.

    For smaller networks (< 1000 nodes), a simple list with lazy evaluation
    can be faster than a heap due to lower overhead.

    This implementation stores all nodes in a list and finds the maximum
    during the 'top' operation, applying fix_error as needed.
    """

    def __init__(self, fix_error_fn: Callable[[int, "NeuronNode"], None]):
        """Initialize.

        Args:
            fix_error_fn: Function to fix error values.
        """
        self.fix_error_fn = fix_error_fn
        self._nodes: dict[int, "NeuronNode"] = {}

    def insert(self, node: "NeuronNode") -> None:
        """Insert a node."""
        self._nodes[node.id] = node

    def update(self, node: "NeuronNode") -> None:
        """Update a node (no-op, just keep reference)."""
        self._nodes[node.id] = node

    def remove(self, node: "NeuronNode") -> None:
        """Remove a node."""
        self._nodes.pop(node.id, None)

    def top(self, current_cycle: int) -> "NeuronNode | None":
        """Find node with largest error.

        Fixes all outdated errors and returns the maximum.
        """
        if not self._nodes:
            return None

        # Fix all errors and find maximum
        max_error = -float("inf")
        max_node = None

        for node in self._nodes.values():
            if node.id < 0:
                continue
            if node.cycle != current_cycle:
                self.fix_error_fn(current_cycle, node)

            if node.error > max_error:
                max_error = node.error
                max_node = node

        return max_node

    def clear(self) -> None:
        """Clear all nodes."""
        self._nodes.clear()
