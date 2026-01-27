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

## オリジナルの追加機能

### 1. 距離閾値によるノード追加（DIS_THV）
```c
// gng.c:844, 953-957
#define DIS_THV 0.5
if(mindis > DIS_THV*DIS_THV && net->node_n < GNGN-2){
    add_new_node_distance(net, v[t]);  // 入力位置に2ノード追加
    return 0;
}
```
勝者ノードが入力から遠い場合（距離 > 0.5）、入力位置に2つの新ノードを追加。

### 2. THV条件付きノード追加
```c
// gng.c:18, 986-990
#define THV 0.001*0.001  // = 0.000001
total_error /= (double)ramda;
if (net->node_n < GNGN && total_error > THV){
    node_add_gng(net);
}
```
lambda_回の学習後、平均誤差が閾値（THV）を超えた場合のみノードを追加。

### 3. Utility削除（node_add_gng内）
```c
// gng.c:450, 461-463, 544-549
for (i = 1; i < net->node_n; i++) {  // i=1から開始（node[0]は除外）
    if(net->gng_u[i]*1000000.0 < 100.0){  // u < 0.0001
        delete_list[delete_num++] = i;
    }
}
if (net->node_n > 10 && min_err < THV){
    for(int i=0;i<delete_num;i++){
        if(delete_list[i] > net->node_n-2) break;  // インデックス有効性チェック
        node_delete(net, delete_list[i]);
    }
}
```
ノード追加時、低Utility（< 0.0001）のノードを削除。
**注意**: ループがi=1から開始するため、node[0]はdelete_listに追加されない。

### 4. ramda/2でのdelete_node_gngu
```c
// gng.c:980-984
for (int i1 = 0; i1 < ramda; i1++) {
    if (i1 != ramda/2)
        learn_epoch(net, v, dmax, 0.05, 0.0005, 1);
    else
        learn_epoch(net, v, dmax, 0.05, 0.0005, 2);  // flag=2でdelete_node_gngu
}
```
lambda_/2回目（100回目）に低Utilityノードを削除。

### 5. 連鎖ノード削除
```c
// gng.c:379-386, 427-429
if(net->edge[del_num][i] == 1){
    net->edge_ct[i]--;
    if(net->edge_ct[i] == 0){
        del_list[del_ct++] = i;  // 孤立ノードをリストに追加
    }
}
for(int i=0;i<del_ct;i++){
    node_delete(net, del_list[i]);  // 再帰的に削除
}
```
ノード削除時、孤立した近傍ノードも再帰的に削除。

### 6. 新ノードの法線初期化
```c
// gng.c:483-485
for (i = 0; i < DIM; i++)
    net->node[r][i] = 0.5 * (net->node[q][i] + net->node[f][i]);
```
新ノードの全属性（位置、色、法線）をq,fの平均で初期化。

### 7. PCA位置収集タイミング
```c
// gng.c:637-641: s1の更新前位置を収集
for (int c = 0; c < 3; c++) {
    cog[c] = net->node[s1][c];
    s_ele[ect][c] = net->node[s1][c];
}

// gng.c:650-652: s1を更新
net->node[s1][i] += e1 * (v[i] - net->node[s1][i]);

// gng.c:657-658, 663-666: 近傍を更新後、その位置を収集
net->node[i][j] += e2 * (v[j] - net->node[i][j]);
s_ele[ect][c] = net->node[i][c];  // 更新後の位置
```
PCAはs1の**更新前**位置 + 近傍の**更新後**位置で計算。

### 8. cedge常時更新
```c
// gng.c:618-630: 常に実行（条件なし）
double dis = 0.0;
for (i = 3; i < LDIM; i++) {
    dis += (net->node[s1][i] - net->node[s2][i]) * ...;
}
if (dis < net->cthv * net->cthv) {
    net->cedge[s1][s2] = 1;
}
```
cedgeはノードの保存された色属性に基づき常に更新。

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
    dis_thv: float = 0.5      # 距離閾値（オリジナル: DIS_THV = 0.5）
    thv: float = 0.000001     # ノード追加誤差閾値（オリジナル: THV = 0.001*0.001）
```

### オリジナルコードのパラメータ（toda_gngdt/gng.c）

```c
// gng.h:11-18
#define GNGN 1000       // 最大ノード数
#define DIM 11          // ノードの次元数
#define LDIM 4          // 学習次元数（x,y,z + 1属性）
#define THV 0.001*0.001 // ノード追加閾値

// gng.c:144-148
net->cthv = 0.05;     // 色閾値
net->nthv = 0.998;    // 法線閾値（fabs(dot) > nthv）

// gng.c:590, 844
const int MAX_AGE = 88;
#define DIS_THV 0.5

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

## Python実装の差異

以下はオリジナルCコードとPython実装の構造的な差異です（アルゴリズム動作への影響は微小）：

### 1. エラー減衰率
- **オリジナル**: `dise[i]`（ノードごと）、すべて0.0005で初期化
- **Python**: `params.beta`（グローバル）
- **影響**: すべてのdise[i]が同値のため実質同等

### 2. discount_err_gngのタイミング
- **オリジナル**: gng_learn()内、近傍更新ループ内で全ノード処理
- **Python**: _gng_learn()の後に_discount_errors()を呼び出し
- **影響**: エラー/ユーティリティはノード追加判定にのみ使用されるため微小

### 3. 新ノードの法線初期化
- **オリジナル**: 親ノードの法線の単純平均（正規化なし）
- **Python**: 平均後に単位ベクトルに正規化
- **影響**: PCAで毎回再計算されるため実質的影響なし

### 4. 色次元
- **オリジナル**: 1次元（node[3]、LDIM=4）
- **Python**: 3次元（RGB）
- **影響**: より一般的な用途に対応可能
