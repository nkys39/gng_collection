"""Test GNG with trajectory visualization on triple ring data."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "gng" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))
sys.path.insert(0, str(Path(__file__).parents[2] / "python"))

from model import GrowingNeuralGas, GNGParams
from sampler import sample_from_image
from trajectory import TrajectoryTracker, create_frame_with_trajectory


def run_experiment(
    image_path: str = "triple_ring.png",
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "gng_trajectory.gif",
    output_final: str = "gng_trajectory_final.png",
    seed: int = 42,
    trajectory_history: int = 30,
) -> None:
    """Run GNG experiment with trajectory visualization.

    Args:
        image_path: Path to shape image.
        n_samples: Number of points to sample.
        n_iterations: Number of training iterations.
        gif_frames: Number of frames in output GIF.
        output_gif: Output GIF path.
        output_final: Output final image path.
        seed: Random seed.
        trajectory_history: Number of positions to keep in trajectory.
    """
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

    # GNG parameters
    params = GNGParams(
        max_nodes=100,
        lambda_=100,
        eps_b=0.08,
        eps_n=0.008,
        alpha=0.5,
        beta=0.005,
        max_age=100,
    )
    model = GrowingNeuralGas(n_dim=2, params=params, seed=seed)

    # Trajectory tracker
    tracker = TrajectoryTracker(max_history=trajectory_history, fade_alpha=True)

    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(m, iteration):
        nodes, edges = m.get_graph()

        # Update trajectory tracker
        tracker.update(nodes)

        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            title = f"GNG with Trajectory - Iteration {iteration} ({len(nodes)} nodes)"
            create_frame_with_trajectory(
                ax, points, nodes, edges, tracker,
                iteration, bg_image, title,
                show_trajectory=True,
                trajectory_color="blue",
            )
            fig.canvas.draw()

            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges")

    print(f"Training GNG for {n_iterations} iterations with trajectory visualization...")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}")
    model.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame
    nodes, edges = model.get_graph()
    title = f"GNG with Trajectory - Final ({len(nodes)} nodes)"
    create_frame_with_trajectory(
        ax, points, nodes, edges, tracker,
        n_iterations, bg_image, title,
        show_trajectory=True,
        trajectory_color="blue",
    )
    plt.savefig(output_final, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved final result: {output_final}")

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
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GNG with trajectory visualization")
    parser.add_argument("--image", type=str, default="triple_ring.png")
    parser.add_argument("-n", "--n-samples", type=int, default=1500)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output-gif", type=str, default="gng_trajectory.gif")
    parser.add_argument("--output-final", type=str, default="gng_trajectory_final.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectory-history", type=int, default=30)

    args = parser.parse_args()

    run_experiment(
        image_path=args.image,
        n_samples=args.n_samples,
        n_iterations=args.iterations,
        gif_frames=args.frames,
        output_gif=args.output_gif,
        output_final=args.output_final,
        seed=args.seed,
        trajectory_history=args.trajectory_history,
    )
