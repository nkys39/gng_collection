"""Test GNG Efficient (UG only, no Lazy Error) on triple ring data.

Tests the optimized GNG implementation with only Uniform Grid optimization,
which produces results equivalent to standard GNG.

Output:
    - gng_efficient_ug_only_triple_ring_final.png: Final network state
    - gng_efficient_ug_only_triple_ring_growth.gif: Training animation
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add paths for imports
sys.path.insert(
    0, str(Path(__file__).parents[2] / "algorithms" / "gng_efficient" / "python")
)
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import GNGEfficient, GNGEfficientParams
from sampler import sample_from_image


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
    bg_image: np.ndarray | None = None,
    extra_info: str = "",
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
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")

    title = f"GNG Efficient (UG only) - Iteration {iteration} ({len(nodes)} nodes)"
    if extra_info:
        title += f" [{extra_info}]"
    ax.set_title(title)
    ax.legend(loc="upper right")


def run_experiment(
    image_path: str = "triple_ring.png",
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "gng_efficient_ug_only_triple_ring_growth.gif",
    output_final: str = "gng_efficient_ug_only_triple_ring_final.png",
    seed: int = 42,
) -> None:
    """Run GNG Efficient (UG only) experiment with visualization."""
    np.random.seed(seed)

    # Generate shape image if not exists
    if not Path(image_path).exists():
        print(f"Generating shape image: {image_path}")
        shapes_dir = Path(__file__).parents[2] / "data" / "2d" / "shapes"
        sys.path.insert(0, str(shapes_dir))
        from generate_shape import generate_triple_ring

        generate_triple_ring(image_path)

    # Load background image
    bg_image = np.array(Image.open(image_path).convert("RGB"))

    # Sample points from image
    print(f"Sampling {n_samples} points from {image_path}...")
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Setup GNG Efficient with UG only (no Lazy Error)
    # This produces results equivalent to standard GNG
    params = GNGEfficientParams(
        max_nodes=100,
        lambda_=100,      # Standard GNG default
        eps_b=0.08,       # Standard GNG default
        eps_n=0.008,      # Standard GNG default
        alpha=0.5,        # Standard GNG default
        beta=0.005,       # Standard GNG default (decay rate)
        max_age=100,      # Standard GNG default
        # Optimization parameters
        use_uniform_grid=True,   # Enable UG for speedup
        use_lazy_error=False,    # Disable Lazy Error for standard GNG equivalence
        h_t=0.1,
        h_rho=1.5,
    )
    model = GNGEfficient(n_dim=2, params=params, seed=seed)

    # Collect frames for GIF
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            extra_info = f"{len(edges)} edges"
            create_frame(ax, points, nodes, edges, iteration, bg_image, extra_info)
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges")

    # Train
    print(f"Training GNG Efficient (UG only) for {n_iterations} iterations...")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}, beta={params.beta}")
    print(f"Optimizations: uniform_grid={params.use_uniform_grid}, lazy_error={params.use_lazy_error}")
    model.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame
    nodes, edges = model.get_graph()
    extra_info = f"{len(edges)} edges"
    create_frame(ax, points, nodes, edges, n_iterations, bg_image, extra_info)
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    # Save GIF
    if frames:
        frames.extend([frames[-1]] * 10)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print(f"\nFinal result: {len(nodes)} nodes, {len(edges)} edges")
    print("Done!")


if __name__ == "__main__":
    run_experiment()
