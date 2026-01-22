"""Measure computation time for all algorithms."""

import subprocess
import sys
from pathlib import Path


def measure_algorithm(algo_name: str, code: str) -> tuple[float, int, int]:
    """Run timing code and extract results."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent)
    )
    if result.returncode != 0:
        print(f"Error for {algo_name}: {result.stderr}")
        return 0, 0, 0

    lines = result.stdout.strip().split("\n")
    time_ms = float(lines[0])
    nodes = int(lines[1])
    edges = int(lines[2])
    return time_ms, nodes, edges


def main():
    results = []

    # GNG
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "gng" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import GrowingNeuralGas, GNGParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = GNGParams(max_nodes=100, lambda_=100, eps_b=0.08, eps_n=0.008, alpha=0.5, beta=0.005, max_age=100)
model = GrowingNeuralGas(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("GNG", code)
    results.append(("GNG", t, n, e))

    # GNG-U
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "gng_u" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import GrowingNeuralGasU, GNGUParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = GNGUParams(max_nodes=100, lambda_=100, eps_b=0.08, eps_n=0.008, alpha=0.5, beta=0.005, max_age=100)
model = GrowingNeuralGasU(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("GNG-U", code)
    results.append(("GNG-U", t, n, e))

    # GNG-T
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "gng_t" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import GrowingNeuralGasT, GNGTParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = GNGTParams(max_nodes=100, lambda_=100, eps_b=0.08, eps_n=0.008, alpha=0.5, beta=0.005, max_age=100)
model = GrowingNeuralGasT(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("GNG-T", code)
    results.append(("GNG-T", t, n, e))

    # GNG-D
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "gng_d" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import GrowingNeuralGasD, GNGDParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = GNGDParams(max_nodes=100, lambda_=100, eps_b=0.08, eps_n=0.008, alpha=0.5, beta=0.005, update_topology_every=10)
model = GrowingNeuralGasD(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("GNG-D", code)
    results.append(("GNG-D", t, n, e))

    # SOM
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "som" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import SelfOrganizingMap, SOMParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = SOMParams(grid_height=10, grid_width=10)
model = SelfOrganizingMap(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("SOM", code)
    results.append(("SOM", t, n, e))

    # Neural Gas
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "neural_gas" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import NeuralGas, NeuralGasParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = NeuralGasParams(n_nodes=100, use_chl=True)
model = NeuralGas(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("Neural Gas", code)
    results.append(("Neural Gas", t, n, e))

    # GCS
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "gcs" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import GrowingCellStructures, GCSParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = GCSParams(max_nodes=100, lambda_=100, eps_b=0.08, eps_n=0.008, alpha=0.5, beta=0.005)
model = GrowingCellStructures(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("GCS", code)
    results.append(("GCS", t, n, e))

    # HCL
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "hcl" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import HardCompetitiveLearning, HCLParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = HCLParams(n_nodes=100)
model = HardCompetitiveLearning(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("HCL", code)
    results.append(("HCL", t, n, e))

    # LBG
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "lbg" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import LindeBuzoGray, LBGParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = LBGParams(n_nodes=100, max_epochs=50)
model = LindeBuzoGray(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("LBG", code)
    results.append(("LBG", t, n, e))

    # Growing Grid
    code = '''
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parents[1] / "algorithms" / "growing_grid" / "python"))
sys.path.insert(0, str(Path(".").resolve().parents[1] / "data" / "2d"))
from model import GrowingGrid, GrowingGridParams
from sampler import sample_from_image
X = sample_from_image("triple_ring.png", n_samples=1500, seed=42)
params = GrowingGridParams(initial_height=2, initial_width=2, max_nodes=100)
model = GrowingGrid(n_dim=2, params=params, seed=42)
start = time.perf_counter()
model.train(X, n_iterations=5000)
elapsed = (time.perf_counter() - start) * 1000
print(f"{elapsed:.1f}")
print(model.n_nodes)
print(model.n_edges)
'''
    t, n, e = measure_algorithm("Growing Grid", code)
    results.append(("Growing Grid", t, n, e))

    print("=== Python計算時間 (5,000 iterations) ===")
    print(f"{'Algorithm':<15} {'Time (ms)':>12} {'Nodes':>8} {'Edges':>8}")
    print("-" * 47)
    for name, t, n, e in results:
        print(f"{name:<15} {t:>12.0f} {n:>8} {e:>8}")


if __name__ == "__main__":
    main()
