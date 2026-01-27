# GNG-DT (Growing Neural Gas with Different Topologies)

## 概要

GNG-DT は、複数の異なるトポロジー（エッジ構造）を同時に学習するGNG実装です。
位置、色、法線など異なる属性に基づく独立したエッジ構造を管理します。

## 出典

- Toda, Y., et al. (2022). "Learning of Point Cloud Data by Growing Neural Gas with Different Topologies"
- 参照実装: references/original_code/toda_gngdt/

## 標準GNGとの違い

| 特性 | 標準GNG | GNG-DT |
|------|--------|--------|
| エッジ構造 | 単一 | 複数（位置、色、法線） |
| 勝者選択 | 全属性を使用 | 位置のみを使用 |
| エッジ接続 | 勝者間で常に接続 | 属性ごとに閾値判定 |
| 法線計算 | なし | PCAによる自動計算 |

## アルゴリズム

### キーコンセプト

1. **位置ベースの勝者選択**:
   - 距離計算には位置情報のみを使用
   - `s1 = argmin_i ||v^pos - h^pos_i||`

2. **属性ごとの独立したエッジ**:
   - 位置エッジ: 標準GNGと同様（年齢管理）
   - 色エッジ: 色差 < τ^col で接続
   - 法線エッジ: 法線内積 > τ^nor で接続

3. **PCA法線計算**:
   - 近傍ノードの位置から共分散行列を計算
   - 最小固有値に対応する固有ベクトル = 法線

### パラメータ

```python
@dataclass
class GNGDTParams:
    max_nodes: int = 100      # 最大ノード数
    lambda_: int = 200        # ノード挿入間隔（オリジナル: ramda = 200）
    eps_b: float = 0.05       # 勝者学習率（オリジナル: e1 = 0.05）
    eps_n: float = 0.0005     # 近傍学習率（オリジナル: e2 = 0.0005）
    alpha: float = 0.5        # 分割時誤差減衰
    beta: float = 0.005       # 全体誤差減衰
    max_age: int = 88         # 最大エッジ年齢（オリジナル: MAX_AGE = 88）
    tau_color: float = 0.05   # 色閾値（オリジナル: cthv = 0.05）
    tau_normal: float = 0.998 # 法線閾値（|内積| > 0.998、オリジナル: nthv = 0.998）
```

### オリジナルコードのパラメータ（toda_gngdt/gng.c）

```c
// gng.c:144-148
net->cthv = 0.05;     // 色閾値
net->nthv = 0.998;    // 法線閾値（fabs(dot) > nthv）

// gng.c:590
const int MAX_AGE = 88;

// gng.c:975, 981
const int ramda = 200;
learn_epoch(net, v, dmax, 0.05, 0.0005, 1);  // e1=0.05, e2=0.0005
```

## 使用例

```python
from algorithms.gng_dt.python.model import GrowingNeuralGasDT, GNGDTParams

# パラメータ設定
params = GNGDTParams(
    max_nodes=150,
    tau_normal=0.95,  # cos(18°) 程度
)

# 3D点群データ
points = np.random.rand(2000, 3)

# 学習
gng = GrowingNeuralGasDT(params=params)
gng.train(points, n_iterations=8000)

# 複数トポロジーの取得
nodes, pos_edges, color_edges, normal_edges = gng.get_multi_graph()

# 法線ベクトルの取得
normals = gng.get_node_normals()
```

## 応用例

1. **3D点群セグメンテーション**:
   - 法線トポロジーで平面を分離
   - 色トポロジーで同色領域を抽出

2. **環境認識**:
   - 床と壁の分離（法線の向きで判別）
   - 走行可能領域の判定

3. **物体認識**:
   - 色と法線の組み合わせで物体を分類

## 関連アルゴリズム

- `gng/`: 標準GNG
- `gng_t/`: 三角形分割（法線計算に関連）
