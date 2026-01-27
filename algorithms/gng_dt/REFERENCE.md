# GNG-DT (Growing Neural Gas with Different Topologies)

## 概要

GNG-DT は、複数の異なるトポロジー（エッジ構造）を同時に学習するGNG実装です。
位置、色、法線など異なる属性に基づく独立したエッジ構造を管理します。

## 出典

- Toda, Y., et al. (2022). "Learning of Point Cloud Data by Growing Neural Gas with Different Topologies"
- 参照実装: references/original_code/toda_gngdt/gng_livox/src/gng.c

## オリジナルコードの構造

### ノード配列 (node[i][j])
```c
node[i][0-2]: 位置 (x, y, z)
node[i][3]:   色（LDIMまで）
node[i][4-6]: 法線ベクトル (nx, ny, nz)
node[i][7]:   PCAの残差 r
```

### エッジ配列
```c
edge[i][j]:   位置ベースエッジ（年齢管理付き）
cedge[i][j]:  色ベースエッジ
nedge[i][j]:  法線ベースエッジ
pedge[i][j]:  走行可能判定エッジ
age[i][j]:    エッジ年齢
```

## アルゴリズムの詳細

### 1. 勝者選択（位置のみ使用）
```c
// gng.c:914-937
for (j = 0; j < 3; j++)  // 位置の3次元のみ
    dis[i] += (net->node[i][j] - v[t][j]) * (net->node[i][j] - v[t][j]);
```

### 2. 色エッジの更新（s1-s2間）
```c
// gng.c:618-630
dis = 0.0;
for (i = 3; i < LDIM; i++) {
    dis += (net->node[s1][i] - net->node[s2][i]) * ...;
}
if (dis < net->cthv * net->cthv) {
    net->cedge[s1][s2] = 1;  // 色が近い場合に接続
}
```

### 3. 法線の内積計算（PCA更新前）
```c
// gng.c:632-635
dis = 0.0;
for (i = 4; i < 7; i++) {
    dis += net->node[s1][i] * net->node[s2][i];  // 法線の内積
}
```

### 4. PCAによる法線更新（毎回実行）
```c
// gng.c:712-728
if (ect > 1) {
    r = pca(s_ele, cog, ect, ev1);  // s1と近傍位置からPCA
    // 法線を正規化してnode[4-6]に格納
    for (j = 0; j < 3; j++) {
        net->node[s1][j + 4] = ev1[j] * dis1;
    }
}
```

### 5. 法線エッジの更新（s1-s2間のみ）
```c
// gng.c:741-748
// PCA更新「前」の内積を使用
if (fabs(dis) > net->nthv) {
    net->nedge[s1][s2] = 1;  // |内積| > 0.998 で接続
    net->nedge[s2][s1] = 1;
}
```

## 標準GNGとの違い

| 特性 | 標準GNG | GNG-DT |
|------|--------|--------|
| エッジ構造 | 単一 | 複数（位置、色、法線） |
| 勝者選択 | 全属性を使用 | 位置のみを使用 |
| エッジ接続 | 勝者間で常に接続 | 属性ごとに閾値判定 |
| 法線計算 | なし | PCAによる**毎回**更新 |
| nedge更新 | - | **s1-s2間のみ** |

## パラメータ

```python
@dataclass
class GNGDTParams:
    max_nodes: int = 100      # 最大ノード数
    lambda_: int = 200        # ノード挿入間隔（オリジナル: ramda = 200）
    eps_b: float = 0.05       # 勝者学習率（オリジナル: e1 = 0.05）
    eps_n: float = 0.0005     # 近傍学習率（オリジナル: e2 = 0.0005）
    alpha: float = 0.5        # 分割時誤差減衰
    beta: float = 0.0005      # 全体誤差減衰（オリジナル: dise = 0.0005）
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
