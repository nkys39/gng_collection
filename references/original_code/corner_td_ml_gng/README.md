# TD-ML-GNG (Top-Down Multi-Layer GNG)

## 出典

- Author: kubota / Corner
- Created: 2024/10/02
- 取得日: 2025-01-27

## ライセンス

(要確認)

## 説明

TD-ML-GNG (Top-Down Multi-Layer GNG) は、階層的なGNG構造を持つ実装です。
上位層から下位層へトップダウンに制御を行い、複数の解像度でデータを表現します。

## ファイル構成

```
corner_td_ml_gng/
├── README.md
├── Final_TD_ML_GNG.py    # メイン実装
└── TD_ML_GNG_FlowChart.pdf  # フローチャート
```

## アルゴリズム概要

### 階層構造

```
Layer 0 (Top)    : 粗い表現 (少ないノード、低学習率)
    ↓
Layer 1 (Middle) : 中間表現
    ↓
Layer 2 (Bottom) : 細かい表現 (多いノード、高学習率)
```

### 主要機能

1. **階層的学習**: 上位層から下位層へ親子関係を維持
2. **Add-if-Silent**: 距離が閾値を超えた場合に新ノード追加
3. **階層的分割 (Hierarchical Split)**: 密度が低い領域で親ノードを分割
4. **親子リンク管理**: 各ノードが親レイヤーのノードを参照

## クラス構造

### GNGクラス

```python
class GNG:
    def __init__(self,
                 feature_number=2,    # 特徴量次元
                 maxNodeLength=68,    # 最大ノード数
                 L1=0.5,              # 勝者学習率 (alpha)
                 L2=0.01,             # 近傍学習率 (beta)
                 newNodeFreq=50,      # ノード追加間隔 (gamma)
                 maxAge=25,           # 最大エッジ年齢 (theta)
                 newNodeFactor=0.5):  # エラー減衰率 (rho)
```

### TDMLGNGクラス

```python
class TDMLGNG:
    def __init__(self, featureNumber, maxLayerNodes):
        # maxLayerNodes: 各層の最大ノード数リスト
        # 例: [5, 333, 1000] → Layer0: 5, Layer1: 333, Layer2: 1000
```

## ノード配列の構造

```python
# W配列: [N, feature_number + 4]
# 各ノードの構造:
# [feature_0, feature_1, ..., feature_n, ERROR, W2, ACTIVATION, PARENT_INDEX]

ERROR_INDEX = -4       # 積算誤差
W2_INDEX = -3          # ||w||^2 (距離計算の高速化用)
ACTIVATION_INDEX = -2  # 活性化回数
PARENT_INDEX = -1      # 親ノードのインデックス
```

## 層ごとのパラメータ

上位層ほど学習率が低く、更新頻度が低く設定:

```python
for l, M in enumerate(maxLayerNodes):
    alpha = L1 / pow(10, L - l - 1)  # 上位層ほど小さい
    beta = L2 / pow(10, L - l - 1)   # 上位層ほど小さい
    gamma = newNodeFreq * (L - l)     # 上位層ほど間隔が長い
    theta = maxAge * (L - l)          # 上位層ほど寿命が長い
```

## 主要メソッド

### GNGクラス

| メソッド | 説明 |
|---------|------|
| `pushData(x, parent_idx)` | データ入力と学習 |
| `findNearestNodes(x, subNetworkID)` | 最近傍ノード探索 |
| `addIfSilent(x, s1)` | 距離が大きい場合にノード追加 |
| `addNewNode()` | 最大誤差位置にノード挿入 |
| `hirahicalSplit()` | 階層的分割 |
| `hirahicalUpdate()` | 親リンクの更新 |
| `updateParentLink(nodes_idx)` | 指定ノードの親リンク更新 |

### TDMLGNGクラス

| メソッド | 説明 |
|---------|------|
| `pushData(x)` | 全層へのデータ入力 |
| `display(data)` | 全層の可視化 |
| `log_network_stats()` | ネットワーク統計のログ出力 |

## Add-if-Silent

距離が閾値を超えた場合に新ノードを追加:

```python
def addIfSilent(x, s1):
    # 距離と接続エッジの平均距離を比較
    dist = getConnectedDistance(s1, s2)
    z = d_s1 - np.mean(dist)
    p = max(0, math.tanh(z / (np.max(dist) + eps)))

    if p > random.random():
        # 新ノード追加
        q3 = len(self.W)
        new_w = np.append(x, [0, 0, 1, -1])
        self.W = np.vstack((self.W, new_w))
        self.c = np.vstack((self.c, [s1, q3, 0]))
```

## 階層的分割 (Hierarchical Split)

子レイヤーの密度が低い親ノードを分割:

```python
def hirahicalSplit():
    for i in range(len(self.W)):
        particles_idx = np.argwhere(child_layer.W[:, PARENT_INDEX] == i)
        particles = child_layer.W[particles_idx, :ERROR_INDEX]

        # 密度を計算
        dist = np.linalg.norm(samplePoint - particles, axis=1)
        density = np.exp(-dist**2 / (2 * radius**2))

        if np.mean(density) < 0.5:
            # 最も遠い点に新しい親ノードを追加
            x = particles[np.argmax(dist)]
            self.addParentNode(x, i)
```

## 使用例

```python
# データ読み込み
data = np.loadtxt("dataset/D31.txt")[:, :2]

# 階層GNG初期化: 3層構造
tdmlgng = TDMLGNG(featureNumber=2, maxLayerNodes=[5, 333, 1000])

# 学習
for epoch in range(100):
    np.random.shuffle(data)
    for x in data:
        tdmlgng.pushData(x)
    tdmlgng.display(data)
```

## 参考にすべきポイント

1. **階層構造の設計**: 親子レイヤーのリンク管理
2. **Add-if-Silent**: 距離ベースの適応的ノード追加
3. **階層的分割**: 密度に基づく親ノードの分割
4. **層別パラメータ**: 上位/下位層で異なるパラメータ設定
5. **高速距離計算**: `W2_INDEX`による`||w||^2`のキャッシュ

## 関連アルゴリズム

- `algorithms/ais_gng/`: Add-if-Silent機能
- 階層的GNGは `algorithms/` には未実装。実装候補として有望。
