"""Generate line graphs showing node count transitions over iterations.

Creates line graphs showing:
- Number of nodes inside/outside the distribution over time
- Average values as horizontal dashed lines

Usage:
    python generate_line_graphs.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import importlib.util

import numpy as np
import matplotlib.pyplot as plt


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
_gngu_mod = load_module_from_path("gngu_model_lg", _algo_dir / "gng_u" / "python" / "model.py")
GrowingNeuralGasU = _gngu_mod.GrowingNeuralGasU
GNGUParams = _gngu_mod.GNGUParams

# GNG-U Euclidean (comparison variant)
_gngu_euclidean_mod = load_module_from_path("gngu_euclidean_lg", _models_dir / "gngu_euclidean.py")
GNGUEuclidean = _gngu_euclidean_mod.GNGUEuclidean
GNGUEuclideanParams = _gngu_euclidean_mod.GNGUEuclideanParams

# GNG-U2 (original Euclidean distance)
_gngu2_mod = load_module_from_path("gngu2_model_lg", _algo_dir / "gng_u2" / "python" / "model.py")
GNGU2 = _gngu2_mod.GNGU2
GNGU2Params = _gngu2_mod.GNGU2Params

# GNG-U2 Squared (comparison variant)
_gngu2_squared_mod = load_module_from_path("gngu2_squared_lg", _models_dir / "gngu2_squared.py")
GNGU2Squared = _gngu2_squared_mod.GNGU2Squared
GNGU2SquaredParams = _gngu2_squared_mod.GNGU2SquaredParams

# AiS-GNG (original Euclidean distance)
_aisgng_mod = load_module_from_path("aisgng_model_lg", _algo_dir / "ais_gng" / "python" / "model.py")
AiSGNG = _aisgng_mod.AiSGNG
AiSGNGParams = _aisgng_mod.AiSGNGParams

# AiS-GNG Squared (comparison variant)
_aisgng_squared_mod = load_module_from_path("aisgng_squared_lg", _models_dir / "aisgng_squared.py")
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


def run_tracking_experiment(model_class, params, config: TrackingConfig):
    """Run tracking experiment and return per-frame node counts."""
    model = model_class(n_dim=2, params=params, seed=config.seed)

    orbit_center = np.array(config.orbit_center)
    rng = np.random.default_rng(config.seed)

    inside_counts = []
    outside_counts = []

    for frame in range(config.total_frames):
        # Calculate ring position
        angle = (frame / config.total_frames) * 2 * np.pi
        ring_center = orbit_center + config.orbit_radius * np.array([
            np.cos(angle), np.sin(angle)
        ])

        # Generate and train
        samples = generate_ring_samples(
            ring_center, config.ring_r_inner, config.ring_r_outer,
            config.samples_per_frame, rng
        )
        for sample in samples:
            model.partial_fit(sample)

        # Count nodes
        nodes, _ = model.get_graph()
        inside, outside = count_nodes_in_ring(
            nodes, ring_center, config.ring_r_inner, config.ring_r_outer
        )
        inside_counts.append(inside)
        outside_counts.append(outside)

    return np.array(inside_counts), np.array(outside_counts)


def plot_comparison_line_graph(
    models_config: list[tuple[str, type, Any]],
    config: TrackingConfig,
    output_path: str,
    title: str,
):
    """Generate a line graph comparing multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_config)))

    for idx, (name, model_class, params) in enumerate(models_config):
        print(f"  Running: {name}")
        inside, outside = run_tracking_experiment(model_class, params, config)

        color = colors[idx]
        frames = np.arange(1, config.total_frames + 1)

        # Inside nodes
        axes[0].plot(frames, inside, color=color, alpha=0.7, label=name)
        axes[0].axhline(y=np.mean(inside), color=color, linestyle='--', alpha=0.5)

        # Outside nodes
        axes[1].plot(frames, outside, color=color, alpha=0.7, label=name)
        axes[1].axhline(y=np.mean(outside), color=color, linestyle='--', alpha=0.5)

    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Nodes Inside Distribution')
    axes[0].set_title('Nodes Inside (Higher is Better)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1, config.total_frames)

    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Nodes Outside Distribution')
    axes[1].set_title('Nodes Outside (Lower is Better)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, config.total_frames)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    config = TrackingConfig()

    # Common parameters
    common_squared = {
        "max_nodes": 50,
        "lambda_": 20,
        "eps_b": 0.15,
        "eps_n": 0.01,
        "alpha": 0.5,
        "beta": 0.01,
        "max_age": 30,
    }

    common_euclidean = {
        **common_squared,
        "chi": 0.01,
        "kappa": 10,
    }

    common_aisgng_squared = {
        **common_squared,
        "chi": 0.01,
        "kappa": 10,
        "theta_ais_min_sq": 0.0009,
        "theta_ais_max_sq": 0.0225,
    }

    common_aisgng_euclidean = {
        **common_euclidean,
        "theta_ais_min": 0.03,
        "theta_ais_max": 0.15,
    }

    # 1. GNG-U Squared utility_k comparison
    print("\n1. GNG-U Squared utility_k comparison")
    plot_comparison_line_graph(
        [
            ("k=0.5", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=0.5)),
            ("k=1.3", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=1.3)),
            ("k=5.0", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=5.0)),
            ("k=20.0", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=20.0)),
        ],
        config,
        str(output_dir / "gngu_squared_line_graph.png"),
        "GNG-U Squared Distance - utility_k Comparison",
    )

    # 2. GNG-U Euclidean utility_k comparison
    print("\n2. GNG-U Euclidean utility_k comparison")
    plot_comparison_line_graph(
        [
            ("k=20", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=20)),
            ("k=50", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=50)),
            ("k=100", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=100)),
            ("k=500", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=500)),
        ],
        config,
        str(output_dir / "gngu_euclidean_line_graph.png"),
        "GNG-U Euclidean Distance - utility_k Comparison",
    )

    # 3. GNG-U2 Squared utility_k comparison
    print("\n3. GNG-U2 Squared utility_k comparison")
    plot_comparison_line_graph(
        [
            ("k=0.5", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=0.5)),
            ("k=1.3", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=1.3)),
            ("k=5.0", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=5.0)),
            ("k=20.0", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=20.0)),
        ],
        config,
        str(output_dir / "gngu2_squared_line_graph.png"),
        "GNG-U2 Squared Distance - utility_k Comparison",
    )

    # 4. GNG-U2 Euclidean utility_k comparison
    print("\n4. GNG-U2 Euclidean utility_k comparison")
    plot_comparison_line_graph(
        [
            ("k=20", GNGU2, GNGU2Params(**common_euclidean, utility_k=20)),
            ("k=50", GNGU2, GNGU2Params(**common_euclidean, utility_k=50)),
            ("k=100", GNGU2, GNGU2Params(**common_euclidean, utility_k=100)),
            ("k=500", GNGU2, GNGU2Params(**common_euclidean, utility_k=500)),
        ],
        config,
        str(output_dir / "gngu2_euclidean_line_graph.png"),
        "GNG-U2 Euclidean Distance - utility_k Comparison",
    )

    # 5. AiS-GNG Squared utility_k comparison
    print("\n5. AiS-GNG Squared utility_k comparison")
    plot_comparison_line_graph(
        [
            ("k=0.5", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=0.5)),
            ("k=1.3", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=1.3)),
            ("k=5.0", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=5.0)),
            ("k=20.0", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=20.0)),
        ],
        config,
        str(output_dir / "aisgng_squared_line_graph.png"),
        "AiS-GNG Squared Distance - utility_k Comparison",
    )

    # 6. AiS-GNG Euclidean utility_k comparison
    print("\n6. AiS-GNG Euclidean utility_k comparison")
    plot_comparison_line_graph(
        [
            ("k=20", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=20)),
            ("k=50", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=50)),
            ("k=100", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=100)),
            ("k=500", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=500)),
        ],
        config,
        str(output_dir / "aisgng_euclidean_line_graph.png"),
        "AiS-GNG Euclidean Distance - utility_k Comparison",
    )

    # 7. Algorithm Comparison: Squared k=1.3
    print("\n7. Algorithm Comparison: Squared k=1.3")
    plot_comparison_line_graph(
        [
            ("GNG-U", GrowingNeuralGasU, GNGUParams(**common_squared, utility_k=1.3)),
            ("GNG-U2", GNGU2Squared, GNGU2SquaredParams(**common_euclidean, utility_k=1.3)),
            ("AiS-GNG", AiSGNGSquared, AiSGNGSquaredParams(**common_aisgng_squared, utility_k=1.3)),
        ],
        config,
        str(output_dir / "algorithm_comparison_squared_line_graph.png"),
        "Algorithm Comparison - Squared Distance (k=1.3)",
    )

    # 8. Algorithm Comparison: Euclidean k=50
    print("\n8. Algorithm Comparison: Euclidean k=50")
    plot_comparison_line_graph(
        [
            ("GNG-U", GNGUEuclidean, GNGUEuclideanParams(**common_euclidean, utility_k=50)),
            ("GNG-U2", GNGU2, GNGU2Params(**common_euclidean, utility_k=50)),
            ("AiS-GNG", AiSGNG, AiSGNGParams(**common_aisgng_euclidean, utility_k=50)),
        ],
        config,
        str(output_dir / "algorithm_comparison_euclidean_line_graph.png"),
        "Algorithm Comparison - Euclidean Distance (k=50)",
    )

    print("\nAll line graphs generated!")


if __name__ == "__main__":
    main()
