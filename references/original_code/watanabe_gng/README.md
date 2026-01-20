# Watanabe GNG Implementation

## 出典

- Author: watanabe
- Created: 2024/05/16
- 取得日: 2025-01-20

## ライセンス

(要確認)

## 説明

C++によるGNG (Growing Neural Gas) 実装。
Eigenライブラリを使用し、テンプレートにより2D/3Dベクトルに対応。

## 特徴

- テンプレートによる次元の切り替え（`Eigen::Vector2f`, `Eigen::Vector3f`）
- 固定サイズ配列によるノード管理（最大100ノード）
- 隣接行列によるエッジ管理
- フレームごとの学習回数制御

## デフォルトパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| MAX_NODE_NUM | 100 | 最大ノード数 |
| START_NODE_NUM | 2 | 初期ノード数 |
| LAMBDA | 100 | ノード追加周期 |
| LEARNRATE_S1 | 0.08 | 勝者ノードの学習率 |
| LEARNRATE_S2 | 0.008 | 近傍ノードの学習率 |
| BETA | 0.005 | グローバルエラー減衰率 |
| ALFA | 0.5 | 分割時のエラー減衰率 |
| MAX_EDGE_AGE | 100 | 最大エッジ年齢 |

## 使用方法

```cpp
#include "GrowingNeuralGas.hpp"

using PointT = Eigen::Vector2f;  // or Eigen::Vector3f

std::vector<PointT> sample_data;
// ... データ準備 ...

GNG::GrowingNeuralGas<PointT> gng(2);  // 引数: 次元数
gng.gngTrain(sample_data);
```

## 対応アルゴリズム

- `algorithms/gng/` のリファクタリング元として使用
