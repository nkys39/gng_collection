"""Test LBG (Linde-Buzo-Gray) on triple ring data with GIF visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "lbg" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import LindeBuzoGray, LBGParams
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

    ax.scatter(points[:, 0], points[:, 1], c="skyblue", s=3, alpha=0.3, label="Data")

    # LBG has no edges, just plot nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5, label="Nodes")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")

    title = f"LBG - Epoch {iteration} ({len(nodes)} nodes)"
    if extra_info:
        title += f" [{extra_info}]"
    ax.set_title(title)
    ax.legend(loc="upper right")


def run_experiment(
    image_path: str = "triple_ring.png",
    n_samples: int = 1500,
    n_iterations: int = 100,  # LBG uses epochs, not individual iterations
    gif_frames: int = 50,
    output_gif: str = "lbg_triple_ring_growth.gif",
    output_final: str = "lbg_triple_ring_final.png",
    seed: int = 42,
) -> None:
    """Run LBG experiment with visualization."""
    np.random.seed(seed)

    if not Path(image_path).exists():
        print(f"Generating shape image: {image_path}")
        shapes_dir = Path(__file__).parents[2] / "data" / "2d" / "shapes"
        sys.path.insert(0, str(shapes_dir))
        from generate_shape import generate_triple_ring
        generate_triple_ring(image_path)

    bg_image = np.array(Image.open(image_path).convert("RGB"))

    print(f"Sampling {n_samples} points from {image_path}...")
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # LBG parameters
    params = LBGParams(
        n_nodes=50,
        max_epochs=n_iterations,
        convergence_threshold=1e-6,
        use_utility=False,
    )
    model = LindeBuzoGray(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(m, epoch):
        if epoch % frame_interval == 0 or epoch == n_iterations - 1:
            nodes, edges = m.get_graph()
            create_frame(ax, points, nodes, edges, epoch, bg_image)
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Epoch {epoch}: {len(nodes)} nodes")

    print(f"Training LBG for {n_iterations} epochs...")
    print(f"Parameters: n_nodes={params.n_nodes}")
    model.train(points, n_iterations=n_iterations, callback=callback)

    nodes, edges = model.get_graph()
    create_frame(ax, points, nodes, edges, model.n_learning, bg_image)
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

    if frames:
        frames.extend([frames[-1]] * 10)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # Slower for batch learning
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LBG on triple ring data")
    parser.add_argument("--image", type=str, default="triple_ring.png")
    parser.add_argument("-n", "--n-samples", type=int, default=1500)
    parser.add_argument("--iterations", type=int, default=100)  # Epochs for LBG
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output-gif", type=str, default="lbg_triple_ring_growth.gif")
    parser.add_argument("--output-final", type=str, default="lbg_triple_ring_final.png")
    parser.add_argument("--seed", type=int, default=42)

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
