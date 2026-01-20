"""Test GNG algorithm on double ring data with GIF visualization.

Usage:
    # Generate shape image
    python ../../data/2d/shapes/generate_shape.py --shape double_ring --output double_ring.png

    # Run this script
    python test_gng_double_ring.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from sampler import sample_from_image


class SimpleGNG:
    """Simple GNG implementation for testing visualization."""

    def __init__(
        self,
        lambda_: int = 100,
        eps_b: float = 0.2,
        eps_n: float = 0.006,
        alpha: float = 0.5,
        beta: float = 0.0005,
        max_age: int = 50,
        max_nodes: int = 100,
    ):
        self.lambda_ = lambda_
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.alpha = alpha
        self.beta = beta
        self.max_age = max_age
        self.max_nodes = max_nodes

        self.nodes = None
        self.edges = None
        self.errors = None
        self.n_iter = 0

    def _init_nodes(self, X: np.ndarray) -> None:
        """Initialize with 2 random nodes from data."""
        indices = np.random.choice(len(X), size=2, replace=False)
        self.nodes = X[indices].copy()
        self.edges = np.zeros((self.max_nodes, self.max_nodes), dtype=int)
        self.errors = np.ones(self.max_nodes) * 0.0
        self.errors[:2] = 1.0

    def _find_nearest_two(self, x: np.ndarray) -> tuple[int, int]:
        """Find indices of two nearest nodes."""
        dists = np.linalg.norm(self.nodes - x, axis=1)
        indices = np.argsort(dists)[:2]
        return indices[0], indices[1]

    def _get_neighbors(self, node_idx: int) -> list[int]:
        """Get indices of neighbors connected by edges."""
        return list(np.where(self.edges[node_idx] > 0)[0])

    def partial_fit(self, x: np.ndarray) -> None:
        """Single training step."""
        if self.nodes is None:
            raise ValueError("Model not initialized. Call fit() first.")

        n_nodes = len(self.nodes)

        # Find two nearest nodes
        s1, s2 = self._find_nearest_two(x)

        # Update error of winner
        dist = np.linalg.norm(x - self.nodes[s1])
        self.errors[s1] += dist

        # Move winner and neighbors toward input
        self.nodes[s1] += self.eps_b * (x - self.nodes[s1])
        for neighbor in self._get_neighbors(s1):
            self.nodes[neighbor] += self.eps_n * (x - self.nodes[neighbor])

        # Update edge between s1 and s2
        self.edges[s1, s2] = 1
        self.edges[s2, s1] = 1

        # Increment age of edges from s1
        for neighbor in self._get_neighbors(s1):
            if neighbor != s2:
                self.edges[s1, neighbor] += 1
                self.edges[neighbor, s1] += 1

        # Remove old edges
        old_edges = np.where(self.edges[s1] > self.max_age)[0]
        for neighbor in old_edges:
            self.edges[s1, neighbor] = 0
            self.edges[neighbor, s1] = 0

        # Remove isolated nodes (no edges)
        for i in range(n_nodes):
            if i < 2:  # Keep at least 2 nodes
                continue
            if np.sum(self.edges[i]) == 0 and self.errors[i] > 0:
                # Mark as removed
                self.errors[i] = 0

        # Add new node periodically
        self.n_iter += 1
        if self.n_iter % self.lambda_ == 0 and n_nodes < self.max_nodes:
            # Find node with maximum error
            active_mask = self.errors > 0
            if np.sum(active_mask) >= 2:
                q = np.argmax(np.where(active_mask, self.errors, -np.inf))
                # Find neighbor of q with maximum error
                neighbors = self._get_neighbors(q)
                if neighbors:
                    neighbor_errors = [(n, self.errors[n]) for n in neighbors]
                    f = max(neighbor_errors, key=lambda x: x[1])[0]

                    # Add new node between q and f
                    new_pos = (self.nodes[q] + self.nodes[f]) / 2
                    self.nodes = np.vstack([self.nodes, new_pos])
                    new_idx = len(self.nodes) - 1

                    # Update edges
                    new_edges = np.zeros((self.max_nodes, self.max_nodes), dtype=int)
                    new_edges[: self.edges.shape[0], : self.edges.shape[1]] = self.edges
                    self.edges = new_edges

                    self.edges[q, f] = 0
                    self.edges[f, q] = 0
                    self.edges[q, new_idx] = 1
                    self.edges[new_idx, q] = 1
                    self.edges[f, new_idx] = 1
                    self.edges[new_idx, f] = 1

                    # Update errors
                    new_errors = np.zeros(self.max_nodes)
                    new_errors[: len(self.errors)] = self.errors
                    self.errors = new_errors

                    self.errors[q] *= 1 - self.alpha
                    self.errors[f] *= 1 - self.alpha
                    self.errors[new_idx] = (self.errors[q] + self.errors[f]) / 2

        # Decay all errors
        self.errors *= 1 - self.beta

    def fit(self, X: np.ndarray, n_iterations: int = 1000, callback=None) -> "SimpleGNG":
        """Train on data.

        Args:
            X: Training data of shape (n_samples, n_features).
            n_iterations: Number of training iterations.
            callback: Optional callback function called each iteration with (model, iter).
        """
        self._init_nodes(X)

        for i in range(n_iterations):
            # Random sample
            idx = np.random.randint(len(X))
            self.partial_fit(X[idx])

            if callback is not None:
                callback(self, i)

        return self

    def get_graph(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Get nodes and edges.

        Returns:
            Tuple of (nodes array, list of edge tuples).
        """
        active_mask = self.errors > 0
        active_indices = np.where(active_mask)[0]

        # Map old indices to new
        index_map = {old: new for new, old in enumerate(active_indices)}

        nodes = self.nodes[active_indices]
        edges = []
        for i, old_i in enumerate(active_indices):
            for old_j in active_indices[i + 1 :]:
                if self.edges[old_i, old_j] > 0:
                    edges.append((index_map[old_i], index_map[old_j]))

        return nodes, edges


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
    bg_image: np.ndarray | None = None,
) -> None:
    """Create a single frame for visualization."""
    ax.clear()

    if bg_image is not None:
        ax.imshow(bg_image, extent=[0, 1, 1, 0], alpha=0.5)

    # Plot sample points
    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3, label="Data")

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
    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5, label="Nodes")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect("equal")
    ax.set_title(f"GNG Training - Iteration {iteration} ({len(nodes)} nodes)")
    ax.legend(loc="upper right")


def run_experiment(
    image_path: str = "double_ring.png",
    n_samples: int = 2000,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "gng_growth.gif",
    output_final: str = "gng_final.png",
    seed: int = 42,
) -> None:
    """Run GNG experiment with visualization.

    Args:
        image_path: Path to shape image.
        n_samples: Number of points to sample.
        n_iterations: Number of GNG training iterations.
        gif_frames: Number of frames in output GIF.
        output_gif: Output GIF path.
        output_final: Output final image path.
        seed: Random seed.
    """
    np.random.seed(seed)

    # Generate shape image if not exists
    if not Path(image_path).exists():
        print(f"Generating shape image: {image_path}")
        shapes_dir = Path(__file__).parents[2] / "data" / "2d" / "shapes"
        sys.path.insert(0, str(shapes_dir))
        from generate_shape import generate_double_ring

        generate_double_ring(image_path)

    # Load background image
    bg_image = np.array(Image.open(image_path).convert("RGB"))

    # Sample points from image
    print(f"Sampling {n_samples} points from {image_path}...")
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Setup GNG
    gng = SimpleGNG(
        lambda_=50,
        eps_b=0.1,
        eps_n=0.01,
        alpha=0.5,
        beta=0.0005,
        max_age=80,
        max_nodes=100,
    )

    # Collect frames for GIF
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()
            create_frame(ax, points, nodes, edges, iteration, bg_image)
            fig.canvas.draw()

            # Convert to PIL Image
            img = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            frames.append(img)
            print(f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges")

    # Train
    print(f"Training GNG for {n_iterations} iterations...")
    gng.fit(points, n_iterations=n_iterations, callback=callback)

    # Save final frame
    nodes, edges = gng.get_graph()
    create_frame(ax, points, nodes, edges, n_iterations, bg_image)
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    # Save GIF
    if frames:
        # Add extra copies of final frame
        frames.extend([frames[-1]] * 10)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # ms per frame
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GNG on double ring data")
    parser.add_argument("--image", type=str, default="double_ring.png", help="Shape image path")
    parser.add_argument("-n", "--n-samples", type=int, default=2000, help="Number of samples")
    parser.add_argument(
        "--iterations", type=int, default=5000, help="Number of training iterations"
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of GIF frames")
    parser.add_argument("--output-gif", type=str, default="gng_growth.gif", help="Output GIF path")
    parser.add_argument(
        "--output-final", type=str, default="gng_final.png", help="Output final image path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_experiment(
        image_path=args.image,
        n_samples=args.n_samples,
        n_iterations=args.iterations,
        gif_frames=args.frames,
        output_gif=args.output_gif,
        output_final=args.output_final,
        seed=args.seed,
    )
