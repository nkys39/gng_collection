# GNG Collection

Growing Neural Gas (GNG) およびその関連アルゴリズムのコレクションリポジトリです。
各アルゴリズムのリファクタリング、2D/3Dデータへの適用テスト、新しいアイデアの実験を行います。

## アルゴリズム概要

### 成長型ネットワーク

| アルゴリズム | 特徴 |
|-------------|------|
| **GNG** | ノードを動的に追加、エッジ年齢に基づくトポロジー学習 |
| **GNG-U** | GNG + Utility基準でノード削除、非定常分布に対応 |
| **GNG-T** | GNG + 明示的Delaunay三角形分割によるトポロジー管理 |
| **GCS** | 三角メッシュ（単体複体）構造を維持しながら成長 |
| **Growing Grid** | 矩形グリッド構造を維持しながら行/列を追加 |

### 固定ノード数ネットワーク

| アルゴリズム | 特徴 |
|-------------|------|
| **SOM** | 固定グリッド、近傍関数（Gaussian）で周辺ノードも更新 |
| **Neural Gas** | ランク（距離順位）ベースの近傍関数、CHL でエッジ学習 |
| **HCL** | 最もシンプル、勝者ノードのみ更新（Winner-Take-All） |
| **LBG** | バッチ学習、k-means類似、収束まで反復 |

## 可視化サンプル

### GNG (Growing Neural Gas)

動的にノードを追加してトポロジーを学習。エッジ年齢が閾値を超えると削除。

| トリプルリング | トラッキング |
|:-------------:|:-----------:|
| ![GNG Triple Ring](experiments/2d_visualization/samples/gng/python/triple_ring_growth.gif) | ![GNG Tracking](experiments/2d_visualization/samples/gng/python/tracking.gif) |

### GNG-U (GNG with Utility)

非定常分布に対応。Utility（有用度）が低いノードを削除し、分布変化に追従。

| トリプルリング | トラッキング |
|:-------------:|:-----------:|
| ![GNG-U Triple Ring](experiments/2d_visualization/samples/gng_u/python/triple_ring_growth.gif) | ![GNG-U Tracking](experiments/2d_visualization/samples/gng_u/python/tracking.gif) |

### GNG-T (GNG with Delaunay Triangulation)

明示的なDelaunay三角形分割でトポロジーを管理。厳密な三角形メッシュ構造を保証。

| トリプルリング | トラッキング |
|:-------------:|:-----------:|
| ![GNG-T Triple Ring](experiments/2d_visualization/samples/gng_t/python/triple_ring_growth.gif) | ![GNG-T Tracking](experiments/2d_visualization/samples/gng_t/python/tracking.gif) |

### SOM (Self-Organizing Map)

固定グリッド構造でトポロジーを保存。近傍関数がガウシアンで周辺ノードも同時に更新。

| トリプルリング | トラッキング |
|:-------------:|:-----------:|
| ![SOM Triple Ring](experiments/2d_visualization/samples/som/python/triple_ring_growth.gif) | ![SOM Tracking](experiments/2d_visualization/samples/som/python/tracking.gif) |

### Neural Gas

ランクベースの近傍関数で全ノードを更新。CHL（Competitive Hebbian Learning）でエッジを学習。

| トリプルリング | トラッキング |
|:-------------:|:-----------:|
| ![NG Triple Ring](experiments/2d_visualization/samples/neural_gas/python/triple_ring_growth.gif) | ![NG Tracking](experiments/2d_visualization/samples/neural_gas/python/tracking.gif) |

### GCS (Growing Cell Structures)

三角メッシュ構造を維持しながら成長。常に単体複体（simplicial complex）を保持。

| トリプルリング | トラッキング |
|:-------------:|:-----------:|
| ![GCS Triple Ring](experiments/2d_visualization/samples/gcs/python/triple_ring_growth.gif) | ![GCS Tracking](experiments/2d_visualization/samples/gcs/python/tracking.gif) |

### HCL (Hard Competitive Learning)

最もシンプルな競合学習。勝者ノードのみを更新（Winner-Take-All）。トポロジー学習なし。

| トリプルリング |
|:-------------:|
| ![HCL Triple Ring](experiments/2d_visualization/samples/hcl/python/triple_ring_growth.gif) |

### LBG (Linde-Buzo-Gray)

バッチ学習ベースのベクトル量子化。各エポックで全データを処理し重心を計算。

| トリプルリング |
|:-------------:|
| ![LBG Triple Ring](experiments/2d_visualization/samples/lbg/python/triple_ring_growth.gif) |

### Growing Grid

自己成長するグリッド構造。高エラー領域の境界に行/列を追加。

| トリプルリング |
|:-------------:|
| ![GG Triple Ring](experiments/2d_visualization/samples/growing_grid/python/triple_ring_growth.gif) |

## 対応言語

- Python
- C++

## ディレクトリ構成

```
gng_collection/
├── algorithms/          # 各アルゴリズム実装
│   ├── _template/       # 新アルゴリズム用テンプレート
│   ├── gng/             # 標準GNG
│   ├── gng_u/           # GNG-U (Utility)
│   ├── som/             # Self-Organizing Map
│   ├── neural_gas/      # Neural Gas
│   ├── gcs/             # Growing Cell Structures
│   ├── hcl/             # Hard Competitive Learning
│   ├── lbg/             # Linde-Buzo-Gray
│   ├── growing_grid/    # Growing Grid
│   └── gng_t/           # GNG-T (Delaunay Triangulation)
├── experiments/         # 実験・アイデア試行
│   └── 2d_visualization/
│       ├── _templates/  # テストテンプレート
│       └── samples/     # 出力サンプル
├── data/                # サンプルデータ
├── references/          # 参照資料・元コード
│   ├── notes/           # アルゴリズムノート
│   └── original_code/   # リファレンス実装
└── python/              # Python共通設定・コア
```

## アルゴリズム一覧

| アルゴリズム | Python | C++ | 説明 |
|-------------|:------:|:---:|------|
| GNG         | ✓      | ✓   | Growing Neural Gas - 動的トポロジー学習 |
| GNG-U       | ✓      | -   | GNG with Utility - 非定常分布対応 |
| GNG-T       | ✓      | -   | GNG with Delaunay - 明示的三角形分割 |
| SOM         | ✓      | -   | Self-Organizing Map - 固定グリッド |
| Neural Gas  | ✓      | -   | ランクベース競合学習 |
| GCS         | ✓      | -   | Growing Cell Structures - メッシュ構造 |
| HCL         | ✓      | -   | Hard Competitive Learning - 勝者のみ更新 |
| LBG         | ✓      | -   | Linde-Buzo-Gray - バッチベクトル量子化 |
| Growing Grid| ✓      | -   | 自己成長グリッド構造 |

## セットアップ

### Python

```bash
# 依存パッケージ
pip install numpy matplotlib pillow scipy
```

### C++

```bash
cd experiments/2d_visualization/cpp
mkdir build && cd build
cmake ..
make
```

## 使い方

### GNG

```python
from algorithms.gng.python.model import GrowingNeuralGas, GNGParams

params = GNGParams(max_nodes=50, lambda_=100)
gng = GrowingNeuralGas(n_dim=2, params=params)
gng.train(X, n_iterations=5000)
nodes, edges = gng.get_graph()
```

### GNG-T (Delaunay Triangulation)

```python
from algorithms.gng_t.python.model import GrowingNeuralGasT, GNGTParams

params = GNGTParams(max_nodes=50, lambda_=100, update_topology_every=10)
gng_t = GrowingNeuralGasT(n_dim=2, params=params)
gng_t.train(X, n_iterations=5000)
nodes, edges = gng_t.get_graph()
triangles = gng_t.get_triangles()  # Delaunay三角形を取得
```

### SOM

```python
from algorithms.som.python.model import SelfOrganizingMap, SOMParams

params = SOMParams(grid_height=10, grid_width=10)
som = SelfOrganizingMap(n_dim=2, params=params)
som.train(X, n_iterations=5000)
nodes, edges = som.get_graph()
```

### Neural Gas

```python
from algorithms.neural_gas.python.model import NeuralGas, NeuralGasParams

params = NeuralGasParams(n_nodes=50, use_chl=True)
ng = NeuralGas(n_dim=2, params=params)
ng.train(X, n_iterations=5000)
nodes, edges = ng.get_graph()
```

### HCL

```python
from algorithms.hcl.python.model import HardCompetitiveLearning, HCLParams

params = HCLParams(n_nodes=50)
hcl = HardCompetitiveLearning(n_dim=2, params=params)
hcl.train(X, n_iterations=5000)
nodes, edges = hcl.get_graph()  # edges is empty (no topology)
```

### LBG

```python
from algorithms.lbg.python.model import LindeBuzoGray, LBGParams

params = LBGParams(n_nodes=50, max_epochs=100)
lbg = LindeBuzoGray(n_dim=2, params=params)
lbg.train(X)  # Batch learning
nodes, edges = lbg.get_graph()
```

### Growing Grid

```python
from algorithms.growing_grid.python.model import GrowingGrid, GrowingGridParams

params = GrowingGridParams(initial_height=2, initial_width=2, max_nodes=100)
gg = GrowingGrid(n_dim=2, params=params)
gg.train(X, n_iterations=5000)
nodes, edges = gg.get_graph()
```

## テストの実行

```bash
cd experiments/2d_visualization

# 各アルゴリズムのテスト（トリプルリング）
python test_gng_triple_ring.py
python test_gngu_triple_ring.py
python test_gngt_triple_ring.py
python test_som_triple_ring.py
python test_ng_triple_ring.py
python test_gcs_triple_ring.py
python test_hcl_triple_ring.py
python test_lbg_triple_ring.py
python test_gg_triple_ring.py

# トラッキングテスト
python test_gng_tracking.py
python test_gngu_tracking.py
python test_gngt_tracking.py
python test_som_tracking.py
python test_ng_tracking.py
python test_gcs_tracking.py

# 軌跡可視化
python test_gng_trajectory.py
```

## 新しいアルゴリズムの追加

1. `algorithms/_template/` をコピー
2. `experiments/2d_visualization/_templates/` のテストテンプレートを使用
3. テスト実行後、出力を `samples/[algorithm]/python/` に保存

詳細は [CLAUDE.md](CLAUDE.md) を参照してください。

## 参照元について

各アルゴリズムの詳細は `references/notes/` を参照してください。

- **GNG**: Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
- **GNG-U**: Fritzke, B. (1997). "Some Competitive Learning Methods"
- **GNG-T**: Martinetz, T. & Schulten, K. (1994). "Topology representing networks" (Delaunay approach)
- **SOM**: Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps"
- **Neural Gas**: Martinetz, T. and Schulten, K. (1991). "A Neural-Gas Network Learns Topologies"
- **GCS**: Fritzke, B. (1994). "Growing cell structures - a self-organizing network"
- **HCL**: Rumelhart, D. E., & Zipser, D. (1985). "Feature discovery by competitive learning"
- **LBG**: Linde, Y., Buzo, A., & Gray, R. (1980). "An Algorithm for Vector Quantizer Design"
- **Growing Grid**: Fritzke, B. (1995). "Growing Grid - a self-organizing network"
- **demogng.de**: https://www.demogng.de/ (リファレンス実装)

## License

MIT License
