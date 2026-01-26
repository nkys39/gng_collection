# GNG Collection

Growing Neural Gas (GNG) およびその関連アルゴリズムのコレクションリポジトリです。
各アルゴリズムのリファクタリング、2D/3Dデータへの適用テスト、新しいアイデアの実験を行います。

## アルゴリズム概要

### 成長型ネットワーク

| アルゴリズム | 特徴 |
|-------------|------|
| **GNG** | ノードを動的に追加、エッジ年齢に基づくトポロジー学習 |
| **GNG-U** | GNG + Utility基準でノード削除、非定常分布に対応 |
| **GNG-U2** | GNG-U改良版、κ間隔Utilityチェックでリアルタイム処理対応 |
| **AiS-GNG** | GNG-U2 + Add-if-Silentルール、高密度位相構造の高速生成 |
| **GNG-T** | GNG + ヒューリスティック三角形分割（四角形探索・交差点探索） |
| **GNG-D** | GNG + 明示的Delaunay三角形分割（scipy.spatial.Delaunay） |
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

| Python (5K iter) | C++ (50K iter) | トラッキング |
|:----------------:|:--------------:|:-----------:|
| ![GNG Python](experiments/2d_visualization/samples/gng/python/triple_ring_growth.gif) | ![GNG C++](experiments/2d_visualization/samples/gng/cpp/triple_ring_growth.gif) | ![GNG Tracking](experiments/2d_visualization/samples/gng/python/tracking.gif) |

### GNG-U (GNG with Utility)

非定常分布に対応。Utility（有用度）が低いノードを削除し、分布変化に追従。

| Python | C++ | トラッキング |
|:------:|:---:|:-----------:|
| ![GNG-U Python](experiments/2d_visualization/samples/gng_u/python/triple_ring_growth.gif) | ![GNG-U C++](experiments/2d_visualization/samples/gng_u/cpp/triple_ring_growth.gif) | ![GNG-U Tracking](experiments/2d_visualization/samples/gng_u/python/tracking.gif) |

### GNG-U2 (GNG with Utility - Variant 2)

GNG-Uの改良版。κ間隔でUtilityチェックを行い、非定常分布への追従性を向上。AiS-GNGのベースアルゴリズム。

| Python | C++ | トラッキング |
|:------:|:---:|:-----------:|
| ![GNG-U2 Python](experiments/2d_visualization/samples/gng_u2/python/triple_ring_growth.gif) | ![GNG-U2 C++](experiments/2d_visualization/samples/gng_u2/cpp/triple_ring_growth.gif) | ![GNG-U2 Tracking](experiments/2d_visualization/samples/gng_u2/python/tracking.gif) |

### AiS-GNG (Add-if-Silent Rule-Based GNG)

GNG-U2をベースに、ネオコグニトロンの「Add-if-Silent」ルールを導入。従来のGNGでは累積誤差に基づきλ間隔でノードを追加するが、AiS-GNGでは有用な入力を直接ノードとして即座に追加することで、高密度な位相構造を素早く生成する。

2つの論文に基づく3つのバリアントを実装：

| バリアント | 論文 | Add-if-Silent条件 | AM機能 |
|-----------|------|------------------|:------:|
| **RO-MAN 2023** | IEEE RO-MAN 2023 | `\|\|v - h\|\| < θ_AiS` (単一閾値) | - |
| **SMC 2023** | IEEE SMC 2023 | `θ_min < \|\|v - h\|\| < θ_max` (範囲閾値) | - |
| **SMC 2023 + AM** | IEEE SMC 2023 | `θ_min < \|\|v - h\|\| < θ_max` (範囲閾値) | ✓ |

**Add-if-Silent条件の違い：**
- **RO-MAN 2023（単一閾値）**: 入力が両勝者ノードからθ_AiS以内なら追加。シンプルだが、既存ノードに近すぎる入力も追加される可能性がある。
- **SMC 2023（範囲閾値）**: 入力が両勝者ノードから[θ_min, θ_max]の範囲内なら追加。θ_minにより冗長ノードを防止。

**Amount of Movement (AM)：**
SMC 2023論文で導入された動的オブジェクト検出機能。各ノードの移動量を `AM_i(t) = γ_AM × AM_i(t-1) + ||h_i(t) - h_i(t-1)||` で追跡し、AM > θ_AM のノードを「移動中」と分類する。

#### AiS-GNG SMC 2023 (範囲閾値)

範囲閾値 `θ_min < ||v - h|| < θ_max` で判定。本リポジトリのデフォルト実装。

| Python | C++ | トラッキング |
|:------:|:---:|:-----------:|
| ![AiS-GNG Python](experiments/2d_visualization/samples/ais_gng/python/triple_ring_growth.gif) | ![AiS-GNG C++](experiments/2d_visualization/samples/ais_gng/cpp/triple_ring_growth.gif) | ![AiS-GNG Tracking](experiments/2d_visualization/samples/ais_gng/python/tracking.gif) |

#### AiS-GNG RO-MAN 2023 (単一閾値)

単一閾値 `||v - h|| < θ_AiS` で判定。最初に発表された基本版。

| Python | C++ | トラッキング |
|:------:|:---:|:-----------:|
| ![AiS-GNG RO-MAN Python](experiments/2d_visualization/samples/ais_gng_roman/python/triple_ring_growth.gif) | ![AiS-GNG RO-MAN C++](experiments/2d_visualization/samples/ais_gng_roman/cpp/triple_ring_growth.gif) | ![AiS-GNG RO-MAN Tracking](experiments/2d_visualization/samples/ais_gng_roman/python/tracking.gif) |

#### AiS-GNG-AM SMC 2023 (移動量追跡)

範囲閾値 + Amount of Movement機能。ノード移動量を色で可視化（青=静止、赤=移動）。動的オブジェクトのセグメンテーションが可能。

| Python | C++ | トラッキング |
|:------:|:---:|:-----------:|
| ![AiS-GNG-AM Python](experiments/2d_visualization/samples/ais_gng_am/python/triple_ring_growth.gif) | ![AiS-GNG-AM C++](experiments/2d_visualization/samples/ais_gng_am/cpp/triple_ring_growth.gif) | ![AiS-GNG-AM Tracking](experiments/2d_visualization/samples/ais_gng_am/python/tracking.gif) |

### GNG-T (GNG with Triangulation)

ヒューリスティックな三角形分割（四角形探索・交差点探索）でメッシュ構造を改善。Kubota & Satomi (2008) に基づく実装。

| Python (5K iter) | C++ (50K iter) | トラッキング |
|:----------------:|:--------------:|:-----------:|
| ![GNG-T Python](experiments/2d_visualization/samples/gng_t/python/triple_ring_growth.gif) | ![GNG-T C++](experiments/2d_visualization/samples/gng_t/cpp/triple_ring_growth.gif) | ![GNG-T Tracking](experiments/2d_visualization/samples/gng_t/python/tracking.gif) |

### GNG-D (GNG with explicit Delaunay)

scipy.spatial.Delaunay による明示的な三角形分割でトポロジーを管理。厳密な幾何学的メッシュ構造を保証。

| トリプルリング | トラッキング |
|:-------------:|:-----------:|
| ![GNG-D Triple Ring](experiments/2d_visualization/samples/gng_d/python/triple_ring_growth.gif) | ![GNG-D Tracking](experiments/2d_visualization/samples/gng_d/python/tracking.gif) |

### SOM (Self-Organizing Map)

固定グリッド構造でトポロジーを保存。近傍関数がガウシアンで周辺ノードも同時に更新。

| Python | C++ | トラッキング |
|:------:|:---:|:-----------:|
| ![SOM Python](experiments/2d_visualization/samples/som/python/triple_ring_growth.gif) | ![SOM C++](experiments/2d_visualization/samples/som/cpp/triple_ring_growth.gif) | ![SOM Tracking](experiments/2d_visualization/samples/som/python/tracking.gif) |

### Neural Gas

ランクベースの近傍関数で全ノードを更新。CHL（Competitive Hebbian Learning）でエッジを学習。

| Python | C++ | トラッキング |
|:------:|:---:|:-----------:|
| ![NG Python](experiments/2d_visualization/samples/neural_gas/python/triple_ring_growth.gif) | ![NG C++](experiments/2d_visualization/samples/neural_gas/cpp/triple_ring_growth.gif) | ![NG Tracking](experiments/2d_visualization/samples/neural_gas/python/tracking.gif) |

### GCS (Growing Cell Structures)

三角メッシュ構造を維持しながら成長。常に単体複体（simplicial complex）を保持。

| Python | C++ | トラッキング |
|:------:|:---:|:-----------:|
| ![GCS Python](experiments/2d_visualization/samples/gcs/python/triple_ring_growth.gif) | ![GCS C++](experiments/2d_visualization/samples/gcs/cpp/triple_ring_growth.gif) | ![GCS Tracking](experiments/2d_visualization/samples/gcs/python/tracking.gif) |

### HCL (Hard Competitive Learning)

最もシンプルな競合学習。勝者ノードのみを更新（Winner-Take-All）。トポロジー学習なし。

| Python | C++ |
|:------:|:---:|
| ![HCL Python](experiments/2d_visualization/samples/hcl/python/triple_ring_growth.gif) | ![HCL C++](experiments/2d_visualization/samples/hcl/cpp/triple_ring_growth.gif) |

### LBG (Linde-Buzo-Gray)

バッチ学習ベースのベクトル量子化。各エポックで全データを処理し重心を計算。

| Python | C++ |
|:------:|:---:|
| ![LBG Python](experiments/2d_visualization/samples/lbg/python/triple_ring_growth.gif) | ![LBG C++](experiments/2d_visualization/samples/lbg/cpp/triple_ring_growth.gif) |

### Growing Grid

自己成長するグリッド構造。高エラー領域の境界に行/列を追加。

| Python | C++ |
|:------:|:---:|
| ![GG Python](experiments/2d_visualization/samples/growing_grid/python/triple_ring_growth.gif) | ![GG C++](experiments/2d_visualization/samples/growing_grid/cpp/triple_ring_growth.gif) |

## 3D可視化サンプル

床面（XZ平面）と壁面（XY平面）が直角に接続したL字型形状での3Dトポロジー学習。

### GNG

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![GNG 3D](experiments/3d_pointcloud/samples/gng/python/floor_wall_growth.gif) | ![GNG 3D Final](experiments/3d_pointcloud/samples/gng/python/floor_wall_final.png) |

### GNG-U

Utility付きGNG。低利用ノードを削除して非定常分布に対応。

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![GNG-U 3D](experiments/3d_pointcloud/samples/gng_u/python/floor_wall_growth.gif) | ![GNG-U 3D Final](experiments/3d_pointcloud/samples/gng_u/python/floor_wall_final.png) |

### GNG-U2

κ間隔でUtilityチェックを行う改良版。AiS-GNGのベース。

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![GNG-U2 3D](experiments/3d_pointcloud/samples/gng_u2/python/floor_wall_growth.gif) | ![GNG-U2 3D Final](experiments/3d_pointcloud/samples/gng_u2/python/floor_wall_final.png) |

### AiS-GNG

Add-if-Silentルール付きGNG。高密度位相構造を高速生成。

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![AiS-GNG 3D](experiments/3d_pointcloud/samples/ais_gng/python/floor_wall_growth.gif) | ![AiS-GNG 3D Final](experiments/3d_pointcloud/samples/ais_gng/python/floor_wall_final.png) |

### GNG-T

ヒューリスティック三角形分割。エッジ交差を検出して削除。

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![GNG-T 3D](experiments/3d_pointcloud/samples/gng_t/python/floor_wall_growth.gif) | ![GNG-T 3D Final](experiments/3d_pointcloud/samples/gng_t/python/floor_wall_final.png) |

### GNG-D

明示的Delaunay三角形分割。scipy.spatialを使用。

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![GNG-D 3D](experiments/3d_pointcloud/samples/gng_d/python/floor_wall_growth.gif) | ![GNG-D 3D Final](experiments/3d_pointcloud/samples/gng_d/python/floor_wall_final.png) |

### GCS

単体複体（simplicial complex）構造を維持しながら成長。3Dでは四面体から開始。

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![GCS 3D](experiments/3d_pointcloud/samples/gcs/floor_wall_growth.gif) | ![GCS 3D Final](experiments/3d_pointcloud/samples/gcs/floor_wall_final.png) |

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
│   ├── gng_u2/          # GNG-U2 (Utility V2 - κ間隔チェック)
│   ├── ais_gng/         # AiS-GNG (Add-if-Silent Rule)
│   ├── gng_t/           # GNG-T (Triangulation - Kubota 2008)
│   ├── gng_d/           # GNG-D (explicit Delaunay)
│   ├── som/             # Self-Organizing Map
│   ├── neural_gas/      # Neural Gas
│   ├── gcs/             # Growing Cell Structures
│   ├── hcl/             # Hard Competitive Learning
│   ├── lbg/             # Linde-Buzo-Gray
│   └── growing_grid/    # Growing Grid
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
| GNG-U       | ✓      | ✓   | GNG with Utility - 非定常分布対応 |
| GNG-U2      | ✓      | ✓   | GNG with Utility V2 - κ間隔Utilityチェック |
| AiS-GNG     | ✓      | ✓   | Add-if-Silent GNG - 高密度位相構造の高速生成 |
| GNG-T       | ✓      | ✓   | GNG with Triangulation - ヒューリスティック三角形分割 |
| GNG-D       | ✓      | -   | GNG with Delaunay - 明示的三角形分割（※scipy依存） |
| SOM         | ✓      | ✓   | Self-Organizing Map - 固定グリッド |
| Neural Gas  | ✓      | ✓   | ランクベース競合学習 |
| GCS         | ✓      | ✓   | Growing Cell Structures - メッシュ構造 |
| HCL         | ✓      | ✓   | Hard Competitive Learning - 勝者のみ更新 |
| LBG         | ✓      | ✓   | Linde-Buzo-Gray - バッチベクトル量子化 |
| Growing Grid| ✓      | ✓   | 自己成長グリッド構造 |

## 計算時間

トリプルリングデータ（1,500サンプル）での計算時間比較（可視化なし、純粋な学習時間）。

### Python実装 (5,000イテレーション)

| アルゴリズム | 計算時間 [ms] | ノード数 | エッジ数 |
|-------------|-------------:|--------:|--------:|
| GNG         | 496          | 52      | 60      |
| GNG-U       | 381          | 27      | 26      |
| GNG-T       | 1,042        | 53      | 132     |
| GNG-D       | 2,074        | 52      | 135     |
| SOM         | 144          | 100     | 180     |
| Neural Gas  | 213          | 100     | 163     |
| GCS         | 516          | 53      | 103     |
| HCL         | 62           | 100     | 0       |
| LBG         | 453          | 100     | 0       |
| Growing Grid| 164          | 96      | 172     |

### C++実装 (5,000イテレーション)

| アルゴリズム | 計算時間 [ms] | ノード数 | エッジ数 |
|-------------|-------------:|--------:|--------:|
| GNG         | 51           | 51      | 63      |
| GNG-U       | 85           | 30      | 29      |
| GNG-T       | 139          | 53      | 102     |
| SOM         | 311          | 100     | 180     |
| Neural Gas  | 508          | 50      | 55      |
| GCS         | 97           | 53      | 103     |
| HCL         | 88           | 50      | 0       |
| LBG         | 457          | 50      | 0       |
| Growing Grid| 259          | 99      | 178     |

### C++実装 (50,000イテレーション)

| アルゴリズム | 計算時間 [ms] | ノード数 | エッジ数 |
|-------------|-------------:|--------:|--------:|
| GNG         | 1,380        | 150     | 292     |
| GNG-T       | 11,431       | 150     | 347     |

※ C++のNeural Gas, HCL, LBGはノード数50で測定（Pythonは100）。
※ C++実装の一部はコールバック処理のオーバーヘッドを含む。GNG系アルゴリズムはPythonの約5〜10倍高速。

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

### GNG-U2 (κ間隔Utilityチェック)

```python
from algorithms.gng_u2.python.model import GrowingNeuralGasU2, GNGU2Params

params = GNGU2Params(
    max_nodes=100,
    kappa=10,         # Utilityチェック間隔（GNG-U2の特徴）
    utility_k=1000.0, # Utility閾値
)
gng_u2 = GrowingNeuralGasU2(n_dim=2, params=params)
gng_u2.train(X, n_iterations=5000)
nodes, edges = gng_u2.get_graph()
print(f"Utility removals: {gng_u2.n_removals}")
```

### AiS-GNG (Add-if-Silent Rule)

```python
from algorithms.ais_gng.python.model import AiSGNG, AiSGNGParams

params = AiSGNGParams(
    max_nodes=100,
    theta_ais_min=0.02,  # Add-if-Silentの最小距離閾値
    theta_ais_max=0.10,  # Add-if-Silentの最大距離閾値
)
ais_gng = AiSGNG(n_dim=2, params=params)
ais_gng.train(X, n_iterations=5000)
nodes, edges = ais_gng.get_graph()
print(f"AiS additions: {ais_gng.n_ais_additions}")  # Add-if-Silentによる追加数
```

### GNG-T (Triangulation - Kubota 2008)

```python
from algorithms.gng_t.python.model import GrowingNeuralGasT, GNGTParams

params = GNGTParams(max_nodes=50, lambda_=100, max_age=100)
gng_t = GrowingNeuralGasT(n_dim=2, params=params)
gng_t.train(X, n_iterations=5000)
nodes, edges = gng_t.get_graph()
triangles = gng_t.get_triangles()  # 三角形を取得
```

### GNG-D (explicit Delaunay)

```python
from algorithms.gng_d.python.model import GrowingNeuralGasD, GNGDParams

params = GNGDParams(max_nodes=50, lambda_=100, update_topology_every=10)
gng_d = GrowingNeuralGasD(n_dim=2, params=params)
gng_d.train(X, n_iterations=5000)
nodes, edges = gng_d.get_graph()
triangles = gng_d.get_triangles()  # Delaunay三角形を取得
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
python test_gngu2_triple_ring.py
python test_aisgng_triple_ring.py
python test_gngt_triple_ring.py
python test_gngd_triple_ring.py
python test_som_triple_ring.py
python test_ng_triple_ring.py
python test_gcs_triple_ring.py
python test_hcl_triple_ring.py
python test_lbg_triple_ring.py
python test_gg_triple_ring.py

# トラッキングテスト
python test_gng_tracking.py
python test_gngu_tracking.py
python test_gngu2_tracking.py
python test_aisgng_tracking.py
python test_gngt_tracking.py
python test_gngd_tracking.py
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

## 実装バリアント

GNG, GCS, GNG-T には2つの実装バリアントがあります：

| バリアント | ファイル | 特徴 | 適したケース |
|-----------|---------|------|-------------|
| demogng準拠 | `model.py` | 誤差ベース・適応的 | 不均一密度データ、既存実装との互換性 |
| Kubotalab準拠 | `model_kubota.py` | 幾何学ベース・均等 | メッシュ品質重視、論文再現 |

**主な違い**: ノード挿入時の隣接ノード選択方法
- **demogng版**: 最大エラー近傍を選択 → データ密度に適応的
- **Kubotalab版**: 最長エッジ近傍を選択 → 均等なノード配置

詳細は [references/notes/variant_comparison.md](references/notes/variant_comparison.md) を参照してください。

### Kubotalab版の使用例

```python
from algorithms.gng.python.model_kubota import GNGKubota, GNGKubotaParams
from algorithms.gcs.python.model_kubota import GCSKubota, GCSKubotaParams
from algorithms.gng_t.python.model_kubota import GNGTKubota, GNGTKubotaParams
```

### Kubotalab版の可視化サンプル

**トリプルリング（静的分布） - Python**

| GNG Kubota | GCS Kubota | GNG-T Kubota |
|:----------:|:----------:|:------------:|
| ![GNG Kubota](experiments/2d_visualization/samples/gng_kubota/python/triple_ring_growth.gif) | ![GCS Kubota](experiments/2d_visualization/samples/gcs_kubota/python/triple_ring_growth.gif) | ![GNG-T Kubota](experiments/2d_visualization/samples/gng_t_kubota/python/triple_ring_growth.gif) |

**トリプルリング（静的分布） - C++**

| GNG Kubota | GCS Kubota | GNG-T Kubota |
|:----------:|:----------:|:------------:|
| ![GNG Kubota C++](experiments/2d_visualization/samples/gng_kubota/cpp/triple_ring_growth.gif) | ![GCS Kubota C++](experiments/2d_visualization/samples/gcs_kubota/cpp/triple_ring_growth.gif) | ![GNG-T Kubota C++](experiments/2d_visualization/samples/gng_t_kubota/cpp/triple_ring_growth.gif) |

**トラッキング（動的分布）**

| GNG Kubota | GCS Kubota | GNG-T Kubota |
|:----------:|:----------:|:------------:|
| ![GNG Kubota Tracking](experiments/2d_visualization/samples/gng_kubota/python/tracking.gif) | ![GCS Kubota Tracking](experiments/2d_visualization/samples/gcs_kubota/python/tracking.gif) | ![GNG-T Kubota Tracking](experiments/2d_visualization/samples/gng_t_kubota/python/tracking.gif) |

## 参照元について

各アルゴリズムの詳細は `references/notes/` を参照してください。

- **GNG**: Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies" (NIPS'94)
- **GNG-U**: Fritzke, B. (1997). "Some Competitive Learning Methods"
- **GNG-U2**: Toda, Y., et al. (2016). "Real-time 3D point cloud segmentation using Growing Neural Gas with Utility" (IEEE ICRA 2016)
- **AiS-GNG (RO-MAN)**: Shoji, M., Obo, T., & Kubota, N. (2023). "Add-if-Silent Rule-Based Growing Neural Gas for High-Density Topological Structure of Unknown Objects" (IEEE RO-MAN 2023, pp. 2492-2498)
- **AiS-GNG-AM (SMC)**: Shoji, M., Obo, T., & Kubota, N. (2023). "Add-if-Silent Rule-Based Growing Neural Gas with Amount of Movement for High-Density Topological Structure Generation of Dynamic Object" (IEEE SMC 2023, pp. 3040-3047)
- **GNG-T**: Kubota, N. & Satomi, M. (2008). "Growing Neural Gas with Triangulation"
- **GNG-D**: Martinetz & Schulten (1994) の明示的Delaunay手法を応用
- **SOM**: Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps"
- **Neural Gas**: Martinetz, T. and Schulten, K. (1991). "A Neural-Gas Network Learns Topologies"
- **GCS**: Fritzke, B. (1994). "Growing cell structures - a self-organizing network"
- **HCL**: Rumelhart, D. E., & Zipser, D. (1985). "Feature discovery by competitive learning"
- **LBG**: Linde, Y., Buzo, A., & Gray, R. (1980). "An Algorithm for Vector Quantizer Design"
- **Growing Grid**: Fritzke, B. (1995). "Growing Grid - a self-organizing network"
- **demogng.de**: https://www.demogng.de/ (リファレンス実装)
- **Kubotalab論文**: 久保田直行, 里見将志 (2008). "自己増殖型ニューラルネットワークと教師無し分類学習"

## License

MIT License
