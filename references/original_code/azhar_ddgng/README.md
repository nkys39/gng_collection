# DD-GNG (Depth Data GNG)

## 出典

- Author: Azhar Aulia Saputra
- Organization: 首都大学東京 Kubota研究室
- Original Author: Naoyuki Kubota (gng.cpp, 2012)
- Created: 2020/05/31
- 取得日: 2025-01-27

## ライセンス

Copyright 2012 首都大学東京 (gng.cpp)
Copyright 2020 Azhar Aulia Saputra (main.cpp, extDisplay.cpp)

## 説明

DD-GNG (Depth Data GNG) は、デプスセンサーからのリアルタイム3D点群データを処理するためのGNG拡張実装です。
ODEシミュレータと連携したロボットビジョンシステムの一部として設計されています。

## アルゴリズム概要

### 基本機能
- **GNG-U (Utility)**: 非定常分布への適応（`gng_u[]`）
- **法線ベクトル計算**: 各ノードのサーフェス法線を計算（`calc_node_normal_vector()`）
- **サーフェス分類**: ノードを平面/エッジ/コーナーに分類（`normTriangle[]`）

### 三角形分割機能
- **四角形検出**: `search_square()` - 四角形の対角線を追加
- **五角形検出**: `search_pentagon()` - 五角形の対角線を追加
- **六角形検出**: `search_hexagon()` - 六角形の対角線を追加
- **三角形探索**: `gng_triangle_search()` - 三角形メッシュの生成

### 学習機能
- **FCMベースのローカル学習**: `gng_learn_local_fcm()` - Fuzzy C-Meansに基づく近傍学習
- **強度ベースの学習**: `strength[]` による重み付け学習

## ファイル構成

```
azhar_ddgng/
└── DepthSensor_Buggy/
    ├── gng.cpp           # GNGコアアルゴリズム (Kubota)
    ├── gng.hpp           # GNG構造体定義
    ├── main.cpp          # メインエントリポイント (Azhar)
    ├── extDisplay.cpp    # OpenGL可視化
    ├── extDisplay.hpp
    ├── surfMatching.cpp  # サーフェスマッチング
    ├── surfMatching.hpp
    ├── projection.cpp    # 射影変換・ベクトル演算
    ├── projection.h
    ├── mesh.cpp          # メッシュ処理
    ├── mesh.hpp
    ├── malloc.cpp        # メモリ管理
    ├── malloc.h
    ├── rnd.cpp           # 乱数生成
    ├── rnd.h
    ├── main.h            # 定数定義
    └── environment.cpp   # 環境設定
```

## 主要な構造体

```c
struct gng {
    double node[GNGN][DIM];      // ノード位置
    int edge[GNGN][GNGN];        // エッジ（1:あり、0:なし）
    int edgeTriangle[GNGN][GNGN]; // 三角形分割用エッジ
    double normTriangle[GNGN][DIM]; // 法線/サーフェスタイプ
    int normAge[GNGN];           // 法線の年齢
    int age[GNGN][GNGN];         // エッジの年齢
    double gng_err[GNGN];        // 積算誤差
    double gng_u[GNGN];          // Utility値
    double strength[GNGN];       // ノード強度
    double normVect[GNGN][5];    // 法線ベクトル
    // ... その他多数のフィールド
};
```

## サーフェス分類

`normTriangle[i][0]` の値によるノード分類:

| 値 | 分類 | 説明 |
|----|------|------|
| 0 | 平面 (floor) | 法線が安定した平面領域 |
| 1 | エッジ | 法線変化が大きい境界領域 |
| 2 | コーナー | 法線が収束しない領域 |
| 3 | 未分類 | 接続数不足など |
| 4 | 安定平面 | 長期間平面として検出 |
| 5 | 安定エッジ | 長期間エッジとして検出 |
| 6 | 安定コーナー | 長期間コーナーとして検出 |

## デフォルトパラメータ

```c
#define GNGN 500           // 最大ノード数
#define DIM 3              // 次元数
#define R_STRE 0.1         // 強度計算の閾値

// gng_main内
const int ramda[4] = {1500, 200, 200, 200};  // 学習サンプル数
const int MAX_AGE[4] = {88, 40, 30, 15};     // 最大エッジ年齢（層別）
net->K = 1000000;          // Utilityの係数
net->udrate = 0.001;       // Utility減衰率
```

## 主要関数

| 関数 | 説明 |
|-----|------|
| `init_gng()` | GNGネットワークの初期化 |
| `gng_main()` | メイン学習ループ |
| `gng_learn_local_fcm()` | FCMベースのローカル学習 |
| `node_add_gng()` | 最大誤差位置にノード挿入 |
| `node_add_gng2()` | 強度を考慮したノード挿入 |
| `node_delete()` | ノード削除 |
| `calc_age()` | エッジ年齢の更新と古いエッジ削除 |
| `gng_triangulation()` | 三角形分割の実行 |
| `calc_node_normal_vector()` | 法線ベクトルとサーフェス分類 |
| `gng_classification()` | クラスタリング |

## 依存ライブラリ

- OpenGL / GLUT
- ODE (Open Dynamics Engine)
- drawstuff (ODE付属)

## 参考にすべきポイント

1. **法線計算アルゴリズム**: `calc_node_normal_vector()` - 隣接ノードから法線を推定
2. **サーフェス分類ロジック**: 法線の安定性に基づく分類
3. **三角形分割**: 四角形/五角形/六角形の検出と対角線追加
4. **FCM学習**: メンバーシップ関数を用いた重み付け学習
5. **Cognitive Map生成**: 認知マップ用の別レイヤーGNG

## 関連アルゴリズム

- `algorithms/gng_u/`: GNG-U機能
- `algorithms/gng_t/`: 三角形分割機能
