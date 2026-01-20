"""Test Neural Gas on single ring data with GIF visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "neural_gas" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import NeuralGas, NeuralGasParams
from sampler import sample_from_image


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

    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3, label="Data")

    # Plot edges (from Competitive Hebbian Learning)
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "r-",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5, label="Nodes")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(f"Neural Gas - Iteration {iteration} ({len(nodes)} nodes, {len(edges)} edges)")
    ax.legend(loc="upper right")


def run_experiment(
    image_path: str = "single_ring.png",
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "ng_single_ring_growth.gif",
    output_final: str = "ng_single_ring_final.png",
    seed: int = 42,
    n_nodes: int = 50,
) -> None:
    """Run Neural Gas experiment."""
    np.random.seed(seed)

    if not Path(image_path).exists():
        shapes_dir = Path(__file__).parents[2] / "data" / "2d" / "shapes"
        sys.path.insert(0, str(shapes_dir))
        from generate_shape import generate_single_ring
        generate_single_ring(image_path)

    bg_image = np.array(Image.open(image_path).convert("RGB"))
    print(f"Sampling {n_samples} points from {image_path}...")
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)

    params = NeuralGasParams(
        n_nodes=n_nodes,
        lambda_initial=n_nodes / 2,
        lambda_final=0.1,
        eps_initial=0.5,
        eps_final=0.005,
        max_age=50,
        use_chl=True,
    )
    ng = NeuralGas(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(model, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = model.get_graph()
            create_frame(ax, points, nodes, edges, iteration, bg_image)
            fig.canvas.draw()
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges")

    print(f"Training Neural Gas for {n_iterations} iterations...")
    print(f"Nodes: {n_nodes}, lambda: {params.lambda_initial}->{params.lambda_final}")
    ng.train(points, n_iterations=n_iterations, callback=callback)

    nodes, edges = ng.get_graph()
    create_frame(ax, points, nodes, edges, n_iterations, bg_image)
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_final}")

    if frames:
        frames.extend([frames[-1]] * 10)
        frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                      duration=100, loop=0)
        print(f"Saved: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    run_experiment()
