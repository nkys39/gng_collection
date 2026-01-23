"""Generate comparison GIFs for utility tracking experiments.

Creates side-by-side GIF animations comparing:
1. Distance metrics comparison (GNG-U Squared vs Euclidean)
2. GNG-U utility_k sweep (squared and euclidean)
3. GNG-U2 utility_k sweep (squared and euclidean)
4. AiS-GNG utility_k sweep (squared and euclidean)
5. Algorithm comparison at Squared k=1.3 (GNG-U vs GNG-U2 vs AiS-GNG)
6. Algorithm comparison at Euclidean k=50 (GNG-U vs GNG-U2 vs AiS-GNG)

Usage:
    python generate_comparison_gifs.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_module_from_path(module_name: str, file_path: Path):
    """Load a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load models
_models_dir = Path(__file__).parent / "models"
_algo_dir = Path(__file__).parents[2] / "algorithms"

# GNG-U (original squared distance)
_gngu_mod = load_module_from_path("gngu_model", _algo_dir / "gng_u" / "python" / "model.py")
GrowingNeuralGasU = _gngu_mod.GrowingNeuralGasU
GNGUParams = _gngu_mod.GNGUParams

# GNG-U Euclidean (comparison variant)
_gngu_euclidean_mod = load_module_from_path("gngu_euclidean", _models_dir / "gngu_euclidean.py")
GNGUEuclidean = _gngu_euclidean_mod.GNGUEuclidean
GNGUEuclideanParams = _gngu_euclidean_mod.GNGUEuclideanParams

# GNG-U2 (original Euclidean distance)
_gngu2_mod = load_module_from_path("gngu2_model", _algo_dir / "gng_u2" / "python" / "model.py")
GNGU2 = _gngu2_mod.GNGU2
GNGU2Params = _gngu2_mod.GNGU2Params

# GNG-U2 Squared (comparison variant)
_gngu2_squared_mod = load_module_from_path("gngu2_squared", _models_dir / "gngu2_squared.py")
GNGU2Squared = _gngu2_squared_mod.GNGU2Squared
GNGU2SquaredParams = _gngu2_squared_mod.GNGU2SquaredParams

# AiS-GNG (original Euclidean distance)
_aisgng_mod = load_module_from_path("aisgng_model", _algo_dir / "ais_gng" / "python" / "model.py")
AiSGNG = _aisgng_mod.AiSGNG
AiSGNGParams = _aisgng_mod.AiSGNGParams

# AiS-GNG Squared (comparison variant)
_aisgng_squared_mod = load_module_from_path("aisgng_squared", _models_dir / "aisgng_squared.py")
AiSGNGSquared = _aisgng_squared_mod.AiSGNGSquared
AiSGNGSquaredParams = _aisgng_squared_mod.AiSGNGSquaredParams


@dataclass
class TrackingConfig:
    """Tracking experiment configuration."""
    orbit_center: tuple[float, float] = (0.5, 0.5)
    orbit_radius: float = 0.25
    ring_r_inner: float = 0.08
    ring_r_outer: float = 0.12
    total_frames: int = 120
    samples_per_frame: int = 50
    seed: int = 42


def generate_ring_samples(center, r_inner, r_outer, n_samples, rng):
    """Generate random samples from a ring shape."""
    samples = []
    while len(samples) < n_samples:
        theta = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(rng.uniform(r_inner**2, r_outer**2))
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append([x, y])
    return np.array(samples)


def count_nodes_in_ring(nodes, ring_center, r_inner, r_outer):
    """Count nodes inside and outside the ring."""
    if len(nodes) == 0:
        return 0, 0
    distances = np.sqrt(np.sum((nodes - ring_center) ** 2, axis=1))
    inside = np.sum((distances >= r_inner) & (distances <= r_outer))
    outside = len(nodes) - inside
    return int(inside), int(outside)


def draw_frame(ax, ring_center, r_inner, r_outer, samples, nodes, edges,
               orbit_center, orbit_radius, title, inside, outside, removals):
    """Draw a single frame."""
    ax.clear()

    # Draw orbit path
    theta = np.linspace(0, 2 * np.pi, 100)
    orbit_x = orbit_center[0] + orbit_radius * np.cos(theta)
    orbit_y = orbit_center[1] + orbit_radius * np.sin(theta)
    ax.plot(orbit_x, orbit_y, "g--", alpha=0.3, linewidth=1)

    # Draw ring
    ring_outer_x = ring_center[0] + r_outer * np.cos(theta)
    ring_outer_y = ring_center[1] + r_outer * np.sin(theta)
    ring_inner_x = ring_center[0] + r_inner * np.cos(theta)
    ring_inner_y = ring_center[1] + r_inner * np.sin(theta)
    ax.fill(ring_outer_x, ring_outer_y, color="skyblue", alpha=0.3)
    ax.fill(ring_inner_x, ring_inner_y, color="white")

    # Draw samples
    ax.scatter(samples[:, 0], samples[:, 1], c="skyblue", s=3, alpha=0.5)

    # Draw edges
    for i, j in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
                "r-", linewidth=1.5, alpha=0.7)

    # Draw nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], c="red", s=40, zorder=5)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nIn:{inside} Out:{outside} Rem:{removals}", fontsize=10)


def generate_comparison_gif(
    models_config: list[tuple[str, type, Any]],
    config: TrackingConfig,
    output_path: str,
    ncols: int = 2,
):
    """Generate a comparison GIF with multiple models side-by-side.

    Args:
        models_config: List of (name, model_class, params) tuples
        config: Tracking configuration
        output_path: Output GIF path
        ncols: Number of columns in the grid
    """
    n_models = len(models_config)
    nrows = (n_models + ncols - 1) // ncols

    # Initialize models
    models = []
    for name, model_class, params in models_config:
        model = model_class(n_dim=2, params=params, seed=config.seed)
        models.append({"name": name, "model": model, "removals": 0})

    orbit_center = np.array(config.orbit_center)
    rng = np.random.default_rng(config.seed)

    # Create figure
    fig_width = 5 * ncols
    fig_height = 5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), facecolor="white")
    if n_models == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    frames = []

    print(f"Generating {config.total_frames} frames for {n_models} models...")

    for frame in range(config.total_frames):
        # Calculate ring position
        angle = (frame / config.total_frames) * 2 * np.pi
        ring_center = orbit_center + config.orbit_radius * np.array([
            np.cos(angle), np.sin(angle)
        ])

        # Generate samples (same for all models)
        samples = generate_ring_samples(
            ring_center, config.ring_r_inner, config.ring_r_outer,
            config.samples_per_frame, rng
        )

        # Train each model and draw
        for idx, m in enumerate(models):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col]

            # Train
            for sample in samples:
                m["model"].partial_fit(sample)

            # Get state
            nodes, edges = m["model"].get_graph()
            inside, outside = count_nodes_in_ring(
                nodes, ring_center, config.ring_r_inner, config.ring_r_outer
            )
            m["removals"] = m["model"].n_removals

            # Draw
            draw_frame(ax, ring_center, config.ring_r_inner, config.ring_r_outer,
                      samples, nodes, edges, orbit_center, config.orbit_radius,
                      m["name"], inside, outside, m["removals"])

        # Hide unused axes
        for idx in range(n_models, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].axis("off")

        plt.tight_layout()
        fig.canvas.draw()

        # Convert to PIL Image
        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        )
        frames.append(img.convert("RGB"))

        if (frame + 1) % 30 == 0:
            print(f"  Frame {frame + 1}/{config.total_frames}")

    plt.close(fig)

    # Save GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
        )
        print(f"Saved: {output_path}")


def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    config = TrackingConfig()

    # Common parameters for squared distance models (GNG-U style)
    common_squared = {
        "max_nodes": 50,
        "lambda_": 20,
        "eps_b": 0.15,
        "eps_n": 0.01,
        "alpha": 0.5,
        "beta": 0.01,
        "max_age": 30,
    }

    # Common parameters for Euclidean distance models (GNG-U2/AiS-GNG style)
    common_euclidean = {
        **common_squared,
        "chi": 0.01,
        "kappa": 10,
    }

    # AiS-GNG specific parameters (squared distance)
    common_aisgng_squared = {
        **common_squared,
        "chi": 0.01,
        "kappa": 10,
        "theta_ais_min_sq": 0.0009,  # 0.03^2
        "theta_ais_max_sq": 0.0225,  # 0.15^2
    }

    # AiS-GNG specific parameters (Euclidean distance)
    common_aisgng_euclidean = {
        **common_euclidean,
        "theta_ais_min": 0.03,
        "theta_ais_max": 0.15,
    }

    # ===================================================================
    # 1. GNG-U: Distance metric comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("1. GNG-U: Distance Metric Comparison")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("GNG-U Squared (k=1.3)", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=1.3)),
            ("GNG-U Euclidean (k=50)", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=50)),
        ],
        config,
        str(output_dir / "gngu_distance_metric_comparison.gif"),
        ncols=2,
    )

    # ===================================================================
    # 2. GNG-U: Squared distance utility_k comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("2. GNG-U: Squared Distance - utility_k Comparison")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("GNG-U Sq k=0.5", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=0.5)),
            ("GNG-U Sq k=1.3", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=1.3)),
            ("GNG-U Sq k=5.0", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=5.0)),
            ("GNG-U Sq k=20.0", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=20.0)),
        ],
        config,
        str(output_dir / "gngu_squared_utility_k.gif"),
        ncols=2,
    )

    # ===================================================================
    # 3. GNG-U: Euclidean distance utility_k comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("3. GNG-U: Euclidean Distance - utility_k Comparison")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("GNG-U Euc k=20", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=20)),
            ("GNG-U Euc k=50", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=50)),
            ("GNG-U Euc k=100", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=100)),
            ("GNG-U Euc k=500", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=500)),
        ],
        config,
        str(output_dir / "gngu_euclidean_utility_k.gif"),
        ncols=2,
    )

    # ===================================================================
    # 4. GNG-U2: Squared distance utility_k comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("4. GNG-U2: Squared Distance - utility_k Comparison")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("GNG-U2 Sq k=0.5", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=0.5)),
            ("GNG-U2 Sq k=1.3", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=1.3)),
            ("GNG-U2 Sq k=5.0", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=5.0)),
            ("GNG-U2 Sq k=20.0", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=20.0)),
        ],
        config,
        str(output_dir / "gngu2_squared_utility_k.gif"),
        ncols=2,
    )

    # ===================================================================
    # 5. GNG-U2: Euclidean distance utility_k comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("5. GNG-U2: Euclidean Distance - utility_k Comparison")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("GNG-U2 Euc k=20", GNGU2, GNGU2Params(**common_euclidean, utility_k=20)),
            ("GNG-U2 Euc k=50", GNGU2, GNGU2Params(**common_euclidean, utility_k=50)),
            ("GNG-U2 Euc k=100", GNGU2, GNGU2Params(**common_euclidean, utility_k=100)),
            ("GNG-U2 Euc k=500", GNGU2, GNGU2Params(**common_euclidean, utility_k=500)),
        ],
        config,
        str(output_dir / "gngu2_euclidean_utility_k.gif"),
        ncols=2,
    )

    # ===================================================================
    # 6. AiS-GNG: Squared distance utility_k comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("6. AiS-GNG: Squared Distance - utility_k Comparison")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("AiS-GNG Sq k=0.5", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=0.5)),
            ("AiS-GNG Sq k=1.3", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=1.3)),
            ("AiS-GNG Sq k=5.0", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=5.0)),
            ("AiS-GNG Sq k=20.0", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=20.0)),
        ],
        config,
        str(output_dir / "aisgng_squared_utility_k.gif"),
        ncols=2,
    )

    # ===================================================================
    # 7. AiS-GNG: Euclidean distance utility_k comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("7. AiS-GNG: Euclidean Distance - utility_k Comparison")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("AiS-GNG Euc k=20", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=20)),
            ("AiS-GNG Euc k=50", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=50)),
            ("AiS-GNG Euc k=100", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=100)),
            ("AiS-GNG Euc k=500", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=500)),
        ],
        config,
        str(output_dir / "aisgng_euclidean_utility_k.gif"),
        ncols=2,
    )

    # ===================================================================
    # 8. Algorithm Comparison: Squared k=1.3
    # ===================================================================
    print("\n" + "=" * 60)
    print("8. Algorithm Comparison: Squared Distance (k=1.3)")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("GNG-U Sq k=1.3", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=1.3)),
            ("GNG-U2 Sq k=1.3", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=1.3)),
            ("AiS-GNG Sq k=1.3", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=1.3)),
        ],
        config,
        str(output_dir / "algorithm_comparison_squared_k1.3.gif"),
        ncols=3,
    )

    # ===================================================================
    # 9. Algorithm Comparison: Euclidean k=50
    # ===================================================================
    print("\n" + "=" * 60)
    print("9. Algorithm Comparison: Euclidean Distance (k=50)")
    print("=" * 60)
    generate_comparison_gif(
        [
            ("GNG-U Euc k=50", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=50)),
            ("GNG-U2 Euc k=50", GNGU2, GNGU2Params(**common_euclidean, utility_k=50)),
            ("AiS-GNG Euc k=50", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=50)),
        ],
        config,
        str(output_dir / "algorithm_comparison_euclidean_k50.gif"),
        ncols=3,
    )

    print("\n" + "=" * 60)
    print("All GIFs generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
