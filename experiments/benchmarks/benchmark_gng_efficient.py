"""Benchmark: Standard GNG vs GNG Efficient optimizations.

Compares execution time and result quality between:
1. Standard GNG (baseline)
2. GNG Efficient with Uniform Grid only
3. GNG Efficient with Lazy Error only
4. GNG Efficient with both optimizations

This reproduces the experiments from Fišer et al. (2013) Tables 3-8.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add paths for imports
algorithms_dir = Path(__file__).parents[2] / "algorithms"
sys.path.insert(0, str(Path(__file__).parents[2] / "data" / "2d"))

# Import standard GNG
sys.path.insert(0, str(algorithms_dir / "gng" / "python"))
from model import GrowingNeuralGas, GNGParams

# Remove gng from path and import GNG Efficient
sys.path.remove(str(algorithms_dir / "gng" / "python"))
sys.path.insert(0, str(algorithms_dir / "gng_efficient" / "python"))

# Need to reimport with different module name
import importlib
if "model" in sys.modules:
    del sys.modules["model"]
    # Also remove any submodules
    to_remove = [k for k in sys.modules if k.startswith("model.") or k in ["lazy_heap", "uniform_grid"]]
    for k in to_remove:
        del sys.modules[k]

from model import GNGEfficient, GNGEfficientParams

from sampler import sample_from_image


def generate_random_data(n_samples: int, n_dim: int = 2, seed: int = 42) -> np.ndarray:
    """Generate random uniform data in [0, 1]^n_dim."""
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, n_dim))


def run_standard_gng(
    data: np.ndarray,
    max_nodes: int,
    n_iterations: int,
    seed: int = 42,
) -> tuple[float, int, int]:
    """Run standard GNG and return (time, n_nodes, n_edges)."""
    # Use standard GNG parameters
    params = GNGParams(
        max_nodes=max_nodes,
        lambda_=100,
        eps_b=0.05,
        eps_n=0.006,
        alpha=0.5,
        beta=0.0005,  # Standard GNG: decay RATE (small value)
        max_age=100,
    )
    gng = GrowingNeuralGas(n_dim=data.shape[1], params=params, seed=seed)

    start_time = time.perf_counter()
    gng.train(data, n_iterations=n_iterations)
    elapsed = time.perf_counter() - start_time

    return elapsed, gng.n_nodes, gng.n_edges


def run_gng_efficient(
    data: np.ndarray,
    max_nodes: int,
    n_iterations: int,
    use_uniform_grid: bool = True,
    use_lazy_error: bool = True,
    seed: int = 42,
) -> tuple[float, int, int]:
    """Run GNG Efficient variant and return (time, n_nodes, n_edges)."""
    # Use paper's Table 2 parameters
    params = GNGEfficientParams(
        max_nodes=max_nodes,
        lambda_=100,
        eps_b=0.05,
        eps_n=0.006,
        alpha=0.5,
        beta=0.9995,  # Paper: decay FACTOR (close to 1)
        max_age=100,
        use_uniform_grid=use_uniform_grid,
        use_lazy_error=use_lazy_error,
        h_t=0.1,
        h_rho=1.5,
    )
    gng = GNGEfficient(n_dim=data.shape[1], params=params, seed=seed)

    start_time = time.perf_counter()
    gng.train(data, n_iterations=n_iterations)
    elapsed = time.perf_counter() - start_time

    return elapsed, gng.n_nodes, gng.n_edges


def run_benchmark(
    max_nodes_list: list[int],
    n_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """Run benchmark for different max_nodes values."""
    print(f"Generating {n_samples} random data points...")
    data = generate_random_data(n_samples, n_dim=2, seed=seed)

    results = {
        "max_nodes": [],
        "standard_gng": {"time": [], "nodes": [], "edges": []},
        "ug_only": {"time": [], "nodes": [], "edges": []},
        "lazy_only": {"time": [], "nodes": [], "edges": []},
        "ug_lazy": {"time": [], "nodes": [], "edges": []},
    }

    for max_nodes in max_nodes_list:
        # n_iterations = max_nodes * lambda to ensure all nodes are created
        n_iterations = max_nodes * 100
        print(f"\n{'='*60}")
        print(f"max_nodes={max_nodes}, n_iterations={n_iterations}")
        print("=" * 60)

        results["max_nodes"].append(max_nodes)

        # Standard GNG
        print("Running Standard GNG...", end=" ", flush=True)
        t, n, e = run_standard_gng(data, max_nodes, n_iterations, seed)
        results["standard_gng"]["time"].append(t)
        results["standard_gng"]["nodes"].append(n)
        results["standard_gng"]["edges"].append(e)
        print(f"{t:.3f}s, {n} nodes, {e} edges")

        # GNG Efficient with Uniform Grid only
        print("Running GNG Efficient (UG only)...", end=" ", flush=True)
        t, n, e = run_gng_efficient(
            data, max_nodes, n_iterations, use_uniform_grid=True, use_lazy_error=False, seed=seed
        )
        results["ug_only"]["time"].append(t)
        results["ug_only"]["nodes"].append(n)
        results["ug_only"]["edges"].append(e)
        print(f"{t:.3f}s, {n} nodes, {e} edges")

        # GNG Efficient with Lazy Error only
        print("Running GNG Efficient (Lazy only)...", end=" ", flush=True)
        t, n, e = run_gng_efficient(
            data, max_nodes, n_iterations, use_uniform_grid=False, use_lazy_error=True, seed=seed
        )
        results["lazy_only"]["time"].append(t)
        results["lazy_only"]["nodes"].append(n)
        results["lazy_only"]["edges"].append(e)
        print(f"{t:.3f}s, {n} nodes, {e} edges")

        # GNG Efficient with both optimizations
        print("Running GNG Efficient (UG+Lazy)...", end=" ", flush=True)
        t, n, e = run_gng_efficient(
            data, max_nodes, n_iterations, use_uniform_grid=True, use_lazy_error=True, seed=seed
        )
        results["ug_lazy"]["time"].append(t)
        results["ug_lazy"]["nodes"].append(n)
        results["ug_lazy"]["edges"].append(e)
        print(f"{t:.3f}s, {n} nodes, {e} edges")

    return results


def print_summary(results: dict) -> None:
    """Print benchmark summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Time comparison table
    print("\n### Execution Time (seconds)")
    print(f"{'max_nodes':>10} | {'Standard':>10} | {'UG only':>10} | {'Lazy only':>10} | {'UG+Lazy':>10} | {'Speedup':>8}")
    print("-" * 75)

    for i, max_nodes in enumerate(results["max_nodes"]):
        std_time = results["standard_gng"]["time"][i]
        ug_time = results["ug_only"]["time"][i]
        lazy_time = results["lazy_only"]["time"][i]
        both_time = results["ug_lazy"]["time"][i]
        speedup = std_time / both_time if both_time > 0 else float("inf")

        print(
            f"{max_nodes:>10} | {std_time:>10.3f} | {ug_time:>10.3f} | {lazy_time:>10.3f} | {both_time:>10.3f} | {speedup:>7.1f}x"
        )

    # Node count comparison
    print("\n### Final Node Count")
    print(f"{'max_nodes':>10} | {'Standard':>10} | {'UG only':>10} | {'Lazy only':>10} | {'UG+Lazy':>10}")
    print("-" * 60)

    for i, max_nodes in enumerate(results["max_nodes"]):
        std_nodes = results["standard_gng"]["nodes"][i]
        ug_nodes = results["ug_only"]["nodes"][i]
        lazy_nodes = results["lazy_only"]["nodes"][i]
        both_nodes = results["ug_lazy"]["nodes"][i]

        print(
            f"{max_nodes:>10} | {std_nodes:>10} | {ug_nodes:>10} | {lazy_nodes:>10} | {both_nodes:>10}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GNG vs GNG Efficient")
    parser.add_argument(
        "--max-nodes",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500],
        help="List of max_nodes values to test",
    )
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of data samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = run_benchmark(
        max_nodes_list=args.max_nodes,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    print_summary(results)
