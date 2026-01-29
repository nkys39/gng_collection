# DBL-GNG (Distributed Batch Learning GNG)

## 出典

- Author: Corner
- Created: 2024/06/21
- 取得日: 2025-01-27

## ライセンス

(要確認)

## 説明

DBL-GNG (Distributed Batch Learning GNG) は、バッチ学習に基づくGNG実装です。
オンライン学習ではなく、バッチ単位でネットワークを更新することで効率的な学習を実現します。

## ファイル構成

```
corner_dbl_gng/
├── README.md
├── DBL_GNG.py         # メインのバッチ学習GNG
├── FCM_BL_GNG.py      # Fuzzy C-Means統合版
└── standard_gng.py    # 標準GNG参照実装
```

## アルゴリズム概要

### DBL_GNG.py

バッチ学習による効率的なGNG更新:

1. **分散初期化**: 複数の開始点からネットワークを初期化
2. **バッチ学習**: 全データに対して一括で更新量を計算
3. **行列演算**: NumPyの行列演算による高速化
4. **エッジ管理**: 重要度に基づくエッジの追加/削除

### FCM_BL_GNG.py

Fuzzy C-Means (FCM) をバッチ学習GNGに統合した実装。

### standard_gng.py

比較用の標準GNG実装。

## DBL_GNGクラス

### 初期化パラメータ

```python
class DBL_GNG():
    def __init__(self,
                 feature_number,      # 特徴量の次元数
                 maxNodeLength,       # 最大ノード数
                 L1=0.5,              # 勝者ノード学習率 (alpha)
                 L2=0.01,             # 近傍ノード学習率 (beta)
                 errorNodeFactor=0.5, # エラー減衰率 (delta)
                 newNodeFactor=0.5):  # 新ノードエラー係数 (rho)
```

### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `initializeDistributedNode(data, n)` | n個の開始点から分散初期化 |
| `resetBatch()` | バッチ更新用の累積変数をリセット |
| `batchLearning(X)` | バッチデータXで学習（累積のみ） |
| `updateNetwork()` | 累積した更新を適用 |
| `addNewNode()` | 最大誤差位置にノード挿入 |
| `cutEdge()` | 重要度の低いエッジを削除 |
| `removeIsolatedNodes()` | 孤立ノードを削除 |
| `removeNonActivatedNodes()` | 非活性ノードを削除 |

### バッチ学習の流れ

```python
gng = DBL_GNG(feature_number=2, maxNodeLength=68)
gng.initializeDistributedNode(data, number_of_starting_points=10)

for epoch in range(20):
    gng.resetBatch()           # 累積変数をリセット
    gng.batchLearning(data)    # バッチ全体で更新量を計算
    gng.updateNetwork()        # 更新を適用
    gng.addNewNode()           # ノード追加
```

### 内部変数

```python
self.W          # ノード位置 [N, feature_number]
self.C          # エッジリスト [E, 2]
self.E          # ノード積算誤差 [N]
self.Delta_W_1  # 勝者ノード更新累積
self.Delta_W_2  # 近傍ノード更新累積
self.A_1        # 勝者ノード活性化回数
self.A_2        # 近傍ノード活性化回数
self.S          # エッジ重要度行列
```

## バッチ学習の利点

1. **計算効率**: 行列演算による並列化
2. **安定性**: バッチ平均による更新のため振動が少ない
3. **GPU対応**: NumPy/CuPyによる高速化が容易

## バッチ学習 vs オンライン学習

| 項目 | バッチ学習 | オンライン学習 |
|-----|----------|--------------|
| 更新頻度 | エポックごと | サンプルごと |
| 計算効率 | 高い（行列演算） | 低い（逐次） |
| メモリ | 累積変数が必要 | 少ない |
| 収束 | 安定 | 速いが不安定 |
| 非定常対応 | 遅い | 速い |

## 使用例

```python
import numpy as np

# データ読み込み
data = np.loadtxt("dataset/Aggregation.txt")

# GNG初期化
gng = DBL_GNG(feature_number=2, maxNodeLength=68)
gng.initializeDistributedNode(data, number_of_starting_points=10)

# 学習ループ
for epoch in range(20):
    gng.resetBatch()
    gng.batchLearning(data)
    gng.updateNetwork()
    gng.addNewNode()

# 結果: gng.W (ノード), gng.C (エッジ)
```

## 参考にすべきポイント

1. **分散初期化**: 複数の開始点から効率的に初期化
2. **行列演算による学習**: `batchLearning()`内の一括計算
3. **エッジ重要度**: `self.S`による重要度管理
4. **適応的ノード追加**: `np.quantile()`によるアウトライア検出

## 関連アルゴリズム

このバッチ学習アプローチは `algorithms/` には未実装。
実装候補として有望。
