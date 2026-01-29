# GNG-DT (GNG with Different Topologies)

## 出典

- Author: Toda
- Organization: 首都大学東京 Kubota研究室
- 取得日: 2025-01-27

## ライセンス

(要確認)

## 説明

GNG-DT (GNG with Different Topologies) は、複数の異なるトポロジー（エッジ構造）を同時に学習するGNG実装です。
ROS2環境でLivox LiDARデータを処理し、位置・色・法線・走行可能性の4種類の独立したエッジを管理します。
これにより、異なる属性に基づく複数のグラフ構造を同時に獲得できます。

## ファイル構成

```
toda_gngdt/
└── gng_livox/
    ├── CMakeLists.txt
    ├── package.xml
    ├── config/
    │   ├── parameter.yaml
    │   └── oic_parameter*.yaml
    ├── launch/
    │   ├── gng_launch.py
    │   ├── gng_livox_launch.py
    │   └── ...
    ├── rviz/
    │   └── test.rviz
    └── src/
        ├── gng.c           # GNGコアアルゴリズム
        ├── gng.h
        ├── gng_livox.cpp   # ROS2ノード
        ├── icp.c           # ICPアルゴリズム
        ├── icp.h
        ├── malloc.h
        ├── mt.h            # メルセンヌ・ツイスタ
        ├── parameter.h
        ├── pca.h           # 主成分分析
        └── rnd.h
```

## アルゴリズム概要

### マルチエッジ構造

4種類のエッジを独立に管理:

```c
int edge[GNGN][GNGN];       // 位置ベースのエッジ
int edgeC[GNGN][GNGN];      // 色ベースのエッジ
int edgeN[GNGN][GNGN];      // 法線ベースのエッジ
int edgeT[GNGN][GNGN];      // 走行可能判定エッジ
```

### 主要機能

1. **GNG-U (Utility)**: 非定常分布への適応
2. **PCAクラスタ特徴量**: 各ノードの局所的な形状を計算
3. **走行可能判定**: 地形の走行可能性を判定
4. **ICP統合**: 点群の位置合わせ

## ノード構造

```c
struct gng {
    double node[GNGN][DIM];      // ノード位置
    double nodeColor[GNGN][3];   // ノード色
    double nodeNorm[GNGN][3];    // ノード法線

    int edge[GNGN][GNGN];        // 位置エッジ
    int edgeC[GNGN][GNGN];       // 色エッジ
    int edgeN[GNGN][GNGN];       // 法線エッジ
    int edgeT[GNGN][GNGN];       // 走行可能エッジ

    int age[GNGN][GNGN];         // エッジ年齢
    double gng_err[GNGN];        // 積算誤差
    double gng_u[GNGN];          // Utility値

    // PCA関連
    double cov[GNGN][DIM][DIM];  // 共分散行列
    double eigenVal[GNGN][DIM];  // 固有値
    double eigenVec[GNGN][DIM][DIM]; // 固有ベクトル
    double pcaRatio[GNGN][DIM];  // PCA比率

    // 走行可能判定
    int traversable[GNGN];
};
```

## パラメータ設定 (YAML)

```yaml
gng_livox:
  ros__parameters:
    # GNGパラメータ
    max_node: 500
    lambda: 100
    eps_b: 0.1
    eps_n: 0.01
    alpha: 0.5
    beta: 0.005
    max_age: 100

    # Utilityパラメータ
    utility_k: 1000000

    # 走行可能判定パラメータ
    traversable_angle_threshold: 0.5
    traversable_height_threshold: 0.1
```

## 主要関数

### GNGコア (gng.c)

| 関数 | 説明 |
|-----|------|
| `init_gng()` | GNGネットワークの初期化 |
| `gng_main()` | メイン学習ループ |
| `find_winner()` | 勝者ノード探索 |
| `update_node()` | ノード位置の更新 |
| `update_edge()` | エッジの更新 |
| `add_node()` | ノード追加 |
| `delete_node()` | ノード削除 |
| `update_utility()` | Utility値の更新 |

### PCA機能 (pca.h)

| 関数 | 説明 |
|-----|------|
| `calc_covariance()` | 共分散行列の計算 |
| `calc_eigenvalue()` | 固有値・固有ベクトルの計算 |
| `calc_pca_ratio()` | PCA比率の計算 |

### 走行可能判定

```c
void calc_traversable(struct gng* net) {
    for (int i = 0; i < net->node_n; i++) {
        // 法線と鉛直方向の角度
        double angle = acos(net->nodeNorm[i][2]);

        // 高さ変動
        double height_var = net->eigenVal[i][2];

        // 判定
        if (angle < threshold_angle && height_var < threshold_height) {
            net->traversable[i] = 1;
        } else {
            net->traversable[i] = 0;
        }
    }
}
```

## ROS2統合 (gng_livox.cpp)

### トピック

| トピック | 型 | 説明 |
|---------|---|------|
| `/livox/lidar` | PointCloud2 | 入力点群 |
| `/gng/nodes` | MarkerArray | ノード可視化 |
| `/gng/edges` | MarkerArray | エッジ可視化 |
| `/gng/traversable` | PointCloud2 | 走行可能領域 |

### 使用例

```bash
# ビルド
colcon build --packages-select gng_livox

# 起動
ros2 launch gng_livox gng_livox_launch.py
```

## 依存関係

- ROS2 (Humble推奨)
- PCL (Point Cloud Library)
- Eigen3
- Livox SDK

## 参考にすべきポイント

1. **マルチエッジ**: 複数の属性で独立にエッジを管理
2. **PCAによる形状特徴**: 局所的な形状を固有値で表現
3. **走行可能判定**: 法線と高さ変動による判定
4. **ROS2統合**: PointCloud2との連携

## 関連アルゴリズム

- `algorithms/gng_u/`: GNG-U機能
- `algorithms/gng_t/`: 三角形分割機能（法線計算部分）
