# GNG Efficient 実験結果

Fišer et al. (2013) "Growing Neural Gas Efficiently" の再現実装の実験結果です。

## 概要

GNG Efficient は、標準GNGアルゴリズムに2つの最適化を適用した高速版です：

1. **Uniform Grid**: 空間をセルに分割し、最近傍探索をO(1)平均で実行
2. **Lazy Error Evaluation**: エラー減衰を遅延計算し、毎ステップO(n)の操作を回避

## 実験パラメータ

### 論文デフォルト (Table 2)

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| λ (lambda) | 200 | ノード挿入間隔 |
| ε_b | 0.05 | 勝者学習率 |
| ε_n | 0.0006 | 近傍学習率 |
| β | 0.9995 | エラー減衰率（毎ステップ） |
| α | 0.95 | ノード挿入時のエラー減衰率 |
| A_max | 200 | 最大エッジ年齢 |
| h_t | 0.1 | グリッド密度閾値 |
| h_ρ | 1.5 | グリッド拡張係数 |

---

## 2D実験

### Triple Ring テスト（静的分布）

3つの同心円リングからサンプリングした点群にGNG Efficientを適用。

**パラメータ**:
- max_nodes: 100
- lambda: 100
- n_iterations: 5000

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![Growth](2d/triple_ring_growth.gif) | ![Final](2d/triple_ring_final.png) |

**結果**: 51ノード、64エッジで3つのリング構造を学習。

---

### Tracking テスト（動的分布）

移動するリング分布をオンライン学習で追跡。

**パラメータ**:
- max_nodes: 50
- lambda: 20
- total_frames: 120
- samples_per_frame: 50

| 追跡アニメーション |
|:------------------:|
| ![Tracking](2d/tracking.gif) |

**結果**: 移動する分布に適応し、リング構造を維持しながら追跡。

---

## 3D実験

### Floor & Wall テスト（L字型3D表面）

床面（XZ平面）と壁面（XY平面）が直角に接続したL字型形状での3Dトポロジー学習。

**パラメータ**:
- max_nodes: 150
- lambda: 100
- n_iterations: 8000

| 成長過程 | 最終状態 |
|:--------:|:--------:|
| ![Growth](3d/floor_wall_growth.gif) | ![Final](3d/floor_wall_final.png) |

**結果**: 81ノード、163エッジで床と壁の両面を覆うトポロジーを学習。

---

## 最適化の効果（ベンチマーク実験）

### 本リポジトリでの実測結果

2D一様分布（50,000点）に対してPythonで実測した結果：

| max_nodes | Standard GNG | UG only | 高速化 | エッジ数 (Std/UG) |
|-----------|-------------|---------|--------|-------------------|
| 500 | 45.4s | 10.3s | **4.4x** | 1298 / 1298 ✓ |
| 1000 | 177.0s | 20.8s | **8.5x** | 2631 / 2604 ✓ |

**観察結果**:
- **Uniform Grid (UG)**: ノード数が増えるほど効果大。1000ノードで約8.5倍高速化
- **Lazy Error**: 結果のエッジ数が異なる（論文のβ解釈の違いによる）

### ベンチマークの実行

```bash
cd experiments/benchmarks
python benchmark_gng_efficient.py --max-nodes 500 1000 --n-samples 50000
```

### 論文の報告値（参考）

論文によると、50,000ノードでの性能比較（C++実装）：

| バリアント | 時間 (秒) | 高速化率 |
|-----------|----------|---------|
| Original (線形探索 + 通常エラー) | 8,294 | 1x |
| UG (Uniform Grid のみ) | 4,814 | 1.7x |
| Err (遅延エラー のみ) | 3,418 | 2.4x |
| **UG+Err (両方)** | **10.67** | **777x** |

**注意**: 論文の数百倍の高速化は、オリジナル実装がO(n)の単純な線形探索を使用していたことによる。
本リポジトリの標準GNGは既にハッシュベースのエッジ管理を使用しているため、差は小さい。

## 参考文献

- Fišer, D., Faigl, J., & Kulich, M. (2013). "Growing Neural Gas Efficiently". Neurocomputing.
- Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies". NIPS.

## ファイル構成

```
experiments/gng_efficient/
├── README.md           # このファイル
├── 2d/
│   ├── triple_ring_final.png
│   ├── triple_ring_growth.gif
│   └── tracking.gif
└── 3d/
    ├── floor_wall_final.png
    └── floor_wall_growth.gif
```

## 実行方法

```bash
# 2D Triple Ring
cd experiments/2d_visualization
python test_gng_efficient_triple_ring.py

# 2D Tracking
python test_gng_efficient_tracking.py

# 3D Floor & Wall
cd experiments/3d_pointcloud
python test_gng_efficient_floor_wall.py
```
