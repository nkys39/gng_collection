# 多角形塗りつぶし実験

GNGで検出されるクリーク構造の可視化実験。極大クリーク検出と三角形分割による多角形塗りつぶしを実装。

## アルゴリズム

### 極大クリーク検出: Bron-Kerbosch法

**極大クリーク**: これ以上頂点を追加できないクリーク（完全部分グラフ）

```
K4 = {A,B,C,D} が存在する場合:
  ✓ K4 が検出される（極大）
  ✗ 内部のK3は検出されない（K4に拡張可能なので極大ではない）
```

#### 基本版
```python
def bron_kerbosch(R, P, X):
    if P と X が両方空:
        R は極大クリーク → 出力

    for v in P:
        bron_kerbosch(R ∪ {v}, P ∩ N(v), X ∩ N(v))
        P = P \ {v}
        X = X ∪ {v}
```

#### Pivot最適化版
```python
def bron_kerbosch_pivot(R, P, X):
    if P と X が両方空:
        R は極大クリーク → 出力

    # Pivot選択（P∪Xで最も隣接数が多い頂点）
    pivot = argmax_{v ∈ P∪X} |N(v) ∩ P|

    # Pivotの非隣接頂点のみ探索（枝刈り）
    for v in P - N(pivot):
        bron_kerbosch_pivot(R ∪ {v}, P ∩ N(v), X ∩ N(v))
        ...
```

### 描画: 極大クリーク → 三角形分割

```
K4 → 2つの三角形に分割（Fan Triangulation）
K5 → 3つの三角形に分割
Kn → (n-2)つの三角形に分割
```

## 結果

### GNG トリプルリング

![GNG Triple Ring](samples/gng_triple_ring_growth.gif)

### アルゴリズム比較

#### GNGグラフ（疎なグラフ）での結果

| 項目 | 基本版 | Pivot版 |
|------|--------|---------|
| 検出数 | 11個 (K3のみ) | 11個 (K3のみ) |
| 実行時間 | 0.134 ms | 0.203 ms |
| 速度比 | 1.0x | **0.66x (遅い)** |

**考察**: 疎なグラフでは、Pivot選択のオーバーヘッドが枝刈りの効果を上回る。

#### 密なグラフでの結果（test_dense_graph.py）

| テスト | ノード数 | エッジ密度 | 検出クリーク | 基本版 | Pivot版 | 速度向上 |
|--------|---------|-----------|-------------|--------|---------|---------|
| 複数クリーク | 12 | - | K3:1, K4:1, K5:1 | 0.038ms | 0.031ms | **1.21x** |
| ランダム(p=0.3) | 30 | 30% | K3:63, K4:12 | 0.385ms | 0.406ms | 0.95x |
| ランダム(p=0.5) | 30 | 50% | K3:15, K4:122, K5:48, K6:7 | 1.615ms | 1.067ms | **1.51x** |
| ランダム(p=0.7) | 25 | 70% | K5:26, K6:96, K7:61, K8:14 | 6.665ms | 1.485ms | **4.49x** |
| 完全グラフK10 | 10 | 100% | K10:1 | 1.054ms | 0.046ms | **22.67x** |

**結論**: グラフが密になるほどPivot最適化の効果が顕著。

#### 推奨

| グラフの密度 | 基本版 | Pivot版 | 推奨 |
|-------------|--------|---------|------|
| 疎（GNG等） | 高速 | オーバーヘッド | 基本版 |
| 中程度(p≈0.3) | 同等 | 同等 | どちらでも |
| 密(p≧0.5) | 遅い | **高速** | Pivot版 |
| 完全グラフ | 非常に遅い | **非常に高速** | Pivot版 |

## 使い方

```bash
cd experiments/polygon_filling

# GNGトリプルリングテスト（GIF生成）
python test_gng_triple_ring.py

# 密なグラフでのアルゴリズム比較
python test_dense_graph.py
```

### 出力ファイル

```
samples/
├── gng_triple_ring_final.png      # 最終結果画像
├── gng_triple_ring_growth.gif     # 成長アニメーション
└── algorithm_comparison.txt       # アルゴリズム比較結果
```

### カスタマイズ

```python
from polygon_utils import bron_kerbosch_pivot, decompose_to_triangles

# 極大クリーク検出
result = bron_kerbosch_pivot(edges_per_node, active_node_ids, min_size=3)

# サイズ別カウント
print(result.count_by_size())  # {3: 11, 4: 2, ...}

# 三角形に分割
triangles = decompose_to_triangles(result.cliques)
```

## ファイル構成

```
experiments/polygon_filling/
├── README.md                 # このファイル
├── polygon_utils.py          # クリーク検出・三角形分割ユーティリティ
├── test_gng_triple_ring.py   # GNGトリプルリングテスト（GIF生成）
├── test_dense_graph.py       # 密なグラフでのアルゴリズム比較
└── samples/                  # 出力結果
    ├── gng_triple_ring_final.png
    ├── gng_triple_ring_growth.gif
    └── algorithm_comparison.txt
```

## 参考文献

- Bron, C., & Kerbosch, J. (1973). "Algorithm 457: finding all cliques of an undirected graph"
- Tomita, E., Tanaka, A., & Takahashi, H. (2006). "The worst-case time complexity for generating all maximal cliques"
