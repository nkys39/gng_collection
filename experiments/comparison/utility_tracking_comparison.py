"""Utility-based tracking comparison experiment.

Compares GNG-U variants with different distance metrics and utility_k values.

Evaluation metrics:
- nodes_inside: Nodes within the true ring distribution (good)
- nodes_outside: Nodes outside the true ring distribution (bad - poor tracking)
- n_removals: Number of utility-based node removals

Usage:
    python utility_tracking_comparison.py
    python utility_tracking_comparison.py --utility-k-sweep
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
    sys.modules[module_name] = module  # Register in sys.modules for dataclass
    spec.loader.exec_module(module)
    return module


# Load comparison models
_models_dir = Path(__file__).parent / "models"
_gngu_euclidean_mod = load_module_from_path("gngu_euclidean", _models_dir / "gngu_euclidean.py")
_gngu2_squared_mod = load_module_from_path("gngu2_squared", _models_dir / "gngu2_squared.py")
_aisgng_squared_mod = load_module_from_path("aisgng_squared", _models_dir / "aisgng_squared.py")

# Load original models
_algo_dir = Path(__file__).parents[2] / "algorithms"
_gngu_mod = load_module_from_path("gngu_model", _algo_dir / "gng_u" / "python" / "model.py")
_gngu2_mod = load_module_from_path("gngu2_model", _algo_dir / "gng_u2" / "python" / "model.py")
_aisgng_mod = load_module_from_path("aisgng_model", _algo_dir / "ais_gng" / "python" / "model.py")

# Export classes
GNGUEuclidean = _gngu_euclidean_mod.GNGUEuclidean
GNGUEuclideanParams = _gngu_euclidean_mod.GNGUEuclideanParams
GNGU2Squared = _gngu2_squared_mod.GNGU2Squared
GNGU2SquaredParams = _gngu2_squared_mod.GNGU2SquaredParams
AiSGNGSquared = _aisgng_squared_mod.AiSGNGSquared
AiSGNGSquaredParams = _aisgng_squared_mod.AiSGNGSquaredParams
GrowingNeuralGasU = _gngu_mod.GrowingNeuralGasU
GNGUParams = _gngu_mod.GNGUParams
GrowingNeuralGasU2 = _gngu2_mod.GrowingNeuralGasU2
GNGU2Params = _gngu2_mod.GNGU2Params
AiSGNG = _aisgng_mod.AiSGNG
AiSGNGParams = _aisgng_mod.AiSGNGParams


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


def generate_ring_samples(
    center: np.ndarray,
    r_inner: float,
    r_outer: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random samples from a ring shape."""
    samples = []
    while len(samples) < n_samples:
        theta = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(rng.uniform(r_inner**2, r_outer**2))
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append([x, y])
    return np.array(samples)


def count_nodes_in_ring(
    nodes: np.ndarray,
    ring_center: np.ndarray,
    r_inner: float,
    r_outer: float,
) -> tuple[int, int]:
    """Count nodes inside and outside the ring.

    Returns:
        (nodes_inside, nodes_outside)
    """
    if len(nodes) == 0:
        return 0, 0

    distances = np.sqrt(np.sum((nodes - ring_center) ** 2, axis=1))
    inside = np.sum((distances >= r_inner) & (distances <= r_outer))
    outside = len(nodes) - inside
    return int(inside), int(outside)


def run_tracking_experiment(
    model_class: type,
    params: Any,
    config: TrackingConfig,
) -> dict:
    """Run a single tracking experiment.

    Returns:
        Dictionary with results:
        - nodes_inside_history: list of nodes inside ring at each frame
        - nodes_outside_history: list of nodes outside ring at each frame
        - n_removals: total utility removals
        - final_nodes: number of nodes at end
    """
    rng = np.random.default_rng(config.seed)
    orbit_center = np.array(config.orbit_center)

    model = model_class(n_dim=2, params=params, seed=config.seed)

    nodes_inside_history = []
    nodes_outside_history = []

    for frame in range(config.total_frames):
        # Calculate ring center position
        angle = (frame / config.total_frames) * 2 * np.pi
        ring_center = orbit_center + config.orbit_radius * np.array([
            np.cos(angle), np.sin(angle)
        ])

        # Generate and train on samples
        samples = generate_ring_samples(
            ring_center, config.ring_r_inner, config.ring_r_outer,
            config.samples_per_frame, rng
        )
        for sample in samples:
            model.partial_fit(sample)

        # Count nodes inside/outside ring
        nodes, _ = model.get_graph()
        inside, outside = count_nodes_in_ring(
            nodes, ring_center, config.ring_r_inner, config.ring_r_outer
        )
        nodes_inside_history.append(inside)
        nodes_outside_history.append(outside)

    return {
        "nodes_inside_history": nodes_inside_history,
        "nodes_outside_history": nodes_outside_history,
        "n_removals": model.n_removals,
        "final_nodes": model.n_nodes,
        "avg_inside": np.mean(nodes_inside_history),
        "avg_outside": np.mean(nodes_outside_history),
    }


def run_comparison_euclidean_vs_squared():
    """Compare Euclidean vs Squared distance metrics."""
    config = TrackingConfig()

    # Common tracking parameters
    common_params = {
        "max_nodes": 50,
        "lambda_": 20,
        "eps_b": 0.15,
        "eps_n": 0.01,
        "alpha": 0.5,
        "beta": 0.01,
        "max_age": 30,
    }

    experiments = [
        # GNG-U (Squared) with original utility_k=1.3
        ("GNG-U (squared, k=1.3)", GrowingNeuralGasU, GNGUParams(
            **common_params, utility_k=1.3
        )),
        # GNG-U (Euclidean) with adjusted utility_k
        ("GNG-U (euclidean, k=1000)", GNGUEuclidean, GNGUEuclideanParams(
            **common_params, chi=0.01, utility_k=1000.0, kappa=10
        )),
        ("GNG-U (euclidean, k=100)", GNGUEuclidean, GNGUEuclideanParams(
            **common_params, chi=0.01, utility_k=100.0, kappa=10
        )),
        ("GNG-U (euclidean, k=10)", GNGUEuclidean, GNGUEuclideanParams(
            **common_params, chi=0.01, utility_k=10.0, kappa=10
        )),
        # GNG-U2 (Euclidean) original
        ("GNG-U2 (euclidean, k=1000)", GrowingNeuralGasU2, GNGU2Params(
            **common_params, chi=0.01, utility_k=1000.0, kappa=10
        )),
        ("GNG-U2 (euclidean, k=100)", GrowingNeuralGasU2, GNGU2Params(
            **common_params, chi=0.01, utility_k=100.0, kappa=10
        )),
        ("GNG-U2 (euclidean, k=10)", GrowingNeuralGasU2, GNGU2Params(
            **common_params, chi=0.01, utility_k=10.0, kappa=10
        )),
        # GNG-U2 (Squared)
        ("GNG-U2 (squared, k=1.3)", GNGU2Squared, GNGU2SquaredParams(
            **common_params, chi=0.01, utility_k=1.3, kappa=10
        )),
        ("GNG-U2 (squared, k=5)", GNGU2Squared, GNGU2SquaredParams(
            **common_params, chi=0.01, utility_k=5.0, kappa=10
        )),
    ]

    results = {}
    print("=" * 70)
    print("Comparison: Euclidean vs Squared Distance Metrics")
    print("=" * 70)

    for name, model_class, params in experiments:
        print(f"\nRunning: {name}...")
        result = run_tracking_experiment(model_class, params, config)
        results[name] = result
        print(f"  Avg inside: {result['avg_inside']:.1f}, "
              f"Avg outside: {result['avg_outside']:.1f}, "
              f"Removals: {result['n_removals']}")

    return results


def run_utility_k_sweep():
    """Sweep utility_k values to find optimal."""
    config = TrackingConfig()

    common_params = {
        "max_nodes": 50,
        "lambda_": 20,
        "eps_b": 0.15,
        "eps_n": 0.01,
        "alpha": 0.5,
        "beta": 0.01,
        "max_age": 30,
        "chi": 0.01,
        "kappa": 10,
    }

    results = {"euclidean": [], "squared": []}

    # Sweep utility_k for Euclidean distance
    utility_k_values_euclidean = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    print("=" * 70)
    print("Utility_k Sweep (Euclidean Distance)")
    print("=" * 70)
    print(f"{'utility_k':>10} | {'Avg Inside':>10} | {'Avg Outside':>11} | {'Removals':>8}")
    print("-" * 50)

    for k in utility_k_values_euclidean:
        params = GNGUEuclideanParams(**common_params, utility_k=k)
        result = run_tracking_experiment(GNGUEuclidean, params, config)
        results["euclidean"].append({
            "utility_k": k,
            "avg_inside": result["avg_inside"],
            "avg_outside": result["avg_outside"],
            "n_removals": result["n_removals"],
        })
        print(f"{k:>10} | {result['avg_inside']:>10.1f} | {result['avg_outside']:>11.1f} | {result['n_removals']:>8}")

    # Sweep utility_k for Squared distance (GNG-U original)
    utility_k_values_squared = [0.5, 1.0, 1.3, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    print("\n" + "=" * 70)
    print("Utility_k Sweep (Squared Distance) - GNG-U original")
    print("=" * 70)
    print(f"{'utility_k':>10} | {'Avg Inside':>10} | {'Avg Outside':>11} | {'Removals':>8}")
    print("-" * 50)

    common_params_squared = {
        "max_nodes": 50,
        "lambda_": 20,
        "eps_b": 0.15,
        "eps_n": 0.01,
        "alpha": 0.5,
        "beta": 0.01,
        "max_age": 30,
    }

    for k in utility_k_values_squared:
        params = GNGUParams(**common_params_squared, utility_k=k)
        result = run_tracking_experiment(GrowingNeuralGasU, params, config)
        results["squared"].append({
            "utility_k": k,
            "avg_inside": result["avg_inside"],
            "avg_outside": result["avg_outside"],
            "n_removals": result["n_removals"],
        })
        print(f"{k:>10} | {result['avg_inside']:>10.1f} | {result['avg_outside']:>11.1f} | {result['n_removals']:>8}")

    return results


def plot_results(results: dict, output_path: str = "comparison_results.png"):
    """Plot comparison results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(results.keys())
    avg_inside = [results[n]["avg_inside"] for n in names]
    avg_outside = [results[n]["avg_outside"] for n in names]
    n_removals = [results[n]["n_removals"] for n in names]

    # Plot 1: Average nodes inside
    ax1 = axes[0]
    bars1 = ax1.barh(names, avg_inside, color="green", alpha=0.7)
    ax1.set_xlabel("Average Nodes Inside Ring")
    ax1.set_title("Tracking Quality (higher is better)")
    ax1.invert_yaxis()

    # Plot 2: Average nodes outside
    ax2 = axes[1]
    bars2 = ax2.barh(names, avg_outside, color="red", alpha=0.7)
    ax2.set_xlabel("Average Nodes Outside Ring")
    ax2.set_title("Tracking Error (lower is better)")
    ax2.invert_yaxis()

    # Plot 3: Utility removals
    ax3 = axes[2]
    bars3 = ax3.barh(names, n_removals, color="blue", alpha=0.7)
    ax3.set_xlabel("Total Utility Removals")
    ax3.set_title("Node Cleanup Activity")
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_utility_k_sweep(results: dict, output_path: str = "utility_k_sweep.png"):
    """Plot utility_k sweep results for both distance types."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, (dist_type, data) in enumerate(results.items()):
        k_values = [r["utility_k"] for r in data]
        avg_inside = [r["avg_inside"] for r in data]
        avg_outside = [r["avg_outside"] for r in data]
        n_removals = [r["n_removals"] for r in data]

        color = "b" if dist_type == "euclidean" else "C1"  # blue or orange
        title_suffix = f"({dist_type.capitalize()} Distance)"

        # Plot 1: Nodes inside vs utility_k
        ax1 = axes[row, 0]
        ax1.semilogx(k_values, avg_inside, "-o", color=color, linewidth=2, markersize=8)
        ax1.set_xlabel("utility_k")
        ax1.set_ylabel("Avg Nodes Inside Ring")
        ax1.set_title(f"Tracking Quality {title_suffix}")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Nodes outside vs utility_k
        ax2 = axes[row, 1]
        ax2.semilogx(k_values, avg_outside, "-o", color="r", linewidth=2, markersize=8)
        ax2.set_xlabel("utility_k")
        ax2.set_ylabel("Avg Nodes Outside Ring")
        ax2.set_title(f"Tracking Error {title_suffix}")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Removals vs utility_k
        ax3 = axes[row, 2]
        ax3.semilogx(k_values, n_removals, "-o", color="g", linewidth=2, markersize=8)
        ax3.set_xlabel("utility_k")
        ax3.set_ylabel("Total Utility Removals")
        ax3.set_title(f"Node Removals {title_suffix}")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Utility tracking comparison")
    parser.add_argument("--utility-k-sweep", action="store_true",
                        help="Run utility_k parameter sweep")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    if args.utility_k_sweep:
        results = run_utility_k_sweep()
        plot_utility_k_sweep(results, str(output_dir / "utility_k_sweep.png"))
    else:
        results = run_comparison_euclidean_vs_squared()
        plot_results(results, str(output_dir / "comparison_results.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
