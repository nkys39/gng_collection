"""Test DD-GNG on triple ring data with GIF visualization.

Demonstrates DD-GNG dynamic density control:
- Sets attention region around middle ring
- Higher node density in attention region

Usage:
    python test_ddgng_triple_ring.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "algorithms" / "dd_gng" / "python"))
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

from model import DynamicDensityGNG, DDGNGParams
from sampler import sample_from_image


def create_frame(
    ax,
    points: np.ndarray,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    iteration: int,
    strengths: np.ndarray | None = None,
    bg_image: np.ndarray | None = None,
    attention_regions: list | None = None,
    extra_info: str = "",
) -> None:
    """Create a single frame for visualization.

    Args:
        ax: Matplotlib axes.
        points: Sample points array.
        nodes: Node positions array.
        edges: List of edge tuples.
        iteration: Current iteration number.
        strengths: Optional strength values for nodes.
        bg_image: Optional background image.
        attention_regions: Optional list of attention regions to draw.
        extra_info: Extra info to show in title.
    """
    ax.clear()

    if bg_image is not None:
        ax.imshow(bg_image, extent=[0, 1, 1, 0], alpha=0.5)

    # Draw attention regions
    if attention_regions:
        for region in attention_regions:
            rect = plt.Rectangle(
                (region.center[0] - region.size[0], region.center[1] - region.size[1]),
                region.size[0] * 2,
                region.size[1] * 2,
                fill=False,
                edgecolor="orange",
                linewidth=2,
                linestyle="--",
                label="Attention Region",
            )
            ax.add_patch(rect)

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

    # Plot nodes with strength-based coloring
    if strengths is not None:
        # Color by strength: higher strength = darker/warmer color
        scatter = ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=strengths,
            cmap="YlOrRd",
            s=50,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
    else:
        ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=50, zorder=5, label="Nodes")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect("equal")

    title = f"DD-GNG - Iteration {iteration} ({len(nodes)} nodes)"
    if extra_info:
        title += f" [{extra_info}]"
    ax.set_title(title)
    ax.legend(loc="upper right")


def run_experiment(
    image_path: str = "triple_ring.png",
    n_samples: int = 1500,
    n_iterations: int = 5000,
    gif_frames: int = 100,
    output_gif: str = "ddgng_triple_ring_growth.gif",
    output_final: str = "ddgng_triple_ring_final.png",
    seed: int = 42,
) -> None:
    """Run DD-GNG experiment with visualization.

    Args:
        image_path: Path to shape image.
        n_samples: Number of points to sample.
        n_iterations: Number of training iterations.
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
        from generate_shape import generate_triple_ring

        generate_triple_ring(image_path)

    # Load background image
    bg_image = np.array(Image.open(image_path).convert("RGB"))

    # Sample points from image
    print(f"Sampling {n_samples} points from {image_path}...")
    points = sample_from_image(image_path, n_samples=n_samples, seed=seed)
    print(f"Sampled {len(points)} points")

    # Setup DD-GNG with parameters
    params = DDGNGParams(
        max_nodes=100,
        lambda_=100,
        eps_b=0.08,
        eps_n=0.008,
        alpha=0.5,
        beta=0.005,
        chi=0.005,
        max_age=100,
        utility_k=1000.0,
        kappa=10,
        # DD-GNG specific
        strength_power=4,
        strength_scale=4.0,
        use_strength_learning=True,
        use_strength_insertion=True,
    )
    model = DynamicDensityGNG(n_dim=2, params=params, seed=seed)

    # Add attention region around middle ring (approximately)
    # The middle ring is roughly at x=0.5, y=0.35 with radius ~0.15
    model.add_attention_region(
        center=[0.5, 0.35],
        size=[0.18, 0.18],
        strength=5.0,  # High strength bonus
    )

    print("Added attention region around middle ring (center=[0.5, 0.35], size=[0.18, 0.18])")

    # Collect frames for GIF
    frames = []
    frame_interval = max(1, n_iterations // gif_frames)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    def callback(m, iteration):
        if iteration % frame_interval == 0 or iteration == n_iterations - 1:
            nodes, edges = m.get_graph()
            strengths = m.get_node_strengths()
            extra_info = f"{m.n_removals} removed"
            create_frame(
                ax,
                points,
                nodes,
                edges,
                iteration,
                strengths=strengths,
                bg_image=bg_image,
                attention_regions=m.attention_regions,
                extra_info=extra_info,
            )
            fig.canvas.draw()

            # Convert to PIL Image
            img = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            )
            frames.append(img.convert("RGB"))
            print(
                f"Iteration {iteration}: {len(nodes)} nodes, {len(edges)} edges, "
                f"max_strength={strengths.max():.1f}"
            )

    # Train
    print(f"Training DD-GNG for {n_iterations} iterations...")
    print(f"Parameters: lambda={params.lambda_}, eps_b={params.eps_b}")
    model.train(points, n_iterations=n_iterations, callback=callback)

    # Save final frame
    nodes, edges = model.get_graph()
    strengths = model.get_node_strengths()
    extra_info = f"{model.n_removals} removed"
    create_frame(
        ax,
        points,
        nodes,
        edges,
        n_iterations,
        strengths=strengths,
        bg_image=bg_image,
        attention_regions=model.attention_regions,
        extra_info=extra_info,
    )
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
            duration=100,
            loop=0,
        )
        print(f"Saved GIF: {output_gif}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DD-GNG on triple ring data")
    parser.add_argument(
        "--image", type=str, default="triple_ring.png", help="Shape image path"
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=1500, help="Number of samples"
    )
    parser.add_argument(
        "--iterations", type=int, default=5000, help="Number of training iterations"
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of GIF frames")
    parser.add_argument(
        "--output-gif",
        type=str,
        default="ddgng_triple_ring_growth.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--output-final",
        type=str,
        default="ddgng_triple_ring_final.png",
        help="Output final image path",
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
