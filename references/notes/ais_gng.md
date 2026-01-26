# AiS-GNG (Add-if-Silent Rule-Based Growing Neural Gas)

## 概要

AiS-GNG (Add-if-Silent Rule-Based Growing Neural Gas) は、GNG-U (Growing Neural Gas with Utility) を拡張し、Add-if-Silentルールを導入することで、高密度な位相構造を素早く生成できるアルゴリズムです。

## 実装バリアント

本リポジトリでは3つのバリアントを提供しています：

| ファイル | 論文 | 閾値 | AM機能 | 説明 |
|---------|------|------|:------:|------|
| `model_roman.py` | RO-MAN 2023 | 単一 θ_AiS | - | 基本版、シンプル |
| `model.py` | SMC 2023 (部分) | 範囲 [θ_min, θ_max] | - | 範囲閾値版 |
| `model_am.py` | SMC 2023 (完全) | 範囲 [θ_min, θ_max] | ✓ | 動的オブジェクト対応 |

```python
# RO-MAN 2023 版（単一閾値）
from algorithms.ais_gng.python.model_roman import AiSGNGRoman, AiSGNGRomanParams

# SMC 2023 範囲閾値版（AM未実装）
from algorithms.ais_gng.python.model import AiSGNG, AiSGNGParams

# SMC 2023 完全版（AM機能付き）
from algorithms.ais_gng.python.model_am import AiSGNGAM, AiSGNGAMParams
```

## 論文情報

### 主要論文

1. **RO-MAN 2023** - 基本版
   - Shoji, M., Obo, T., & Kubota, N.
   - "Add-if-Silent Rule-Based Growing Neural Gas for High-Density Topological Structure of Unknown Objects"
   - IEEE RO-MAN 2023, pp. 2492-2498
   - **特徴**: 単一閾値 θ_AiS による Add-if-Silent ルール

2. **SMC 2023 (拡張版: AiS-GNG-AM)** - 動的オブジェクト対応版
   - Shoji, M., Obo, T., & Kubota, N.
   - "Add-if-Silent Rule-Based Growing Neural Gas with Amount of Movement for High-Density Topological Structure Generation of Dynamic Object"
   - IEEE SMC 2023, pp. 3040-3047
   - **特徴**: 範囲閾値 [θ_min, θ_max] + Amount of Movement (AM) による動的オブジェクト検出

### ベースとなる手法

- **GNG-U2**: Toda, Y., et al. (2016). "Real-time 3D point cloud segmentation using Growing Neural Gas with Utility"
- **Add-if-Silent Rule**: Fukushima, K. (2014). "Add-if-Silent Rule for Training Multi-layered Convolutional Network Neocognitron"

## アルゴリズムの特徴

### 従来のGNGの問題点

従来のGNGでは、累積誤差に基づいてλ入力ごとにノードを追加します。しかし、以下の問題があります：

1. **疎サンプリング領域**: 遠くの物体など、サンプル密度が低い領域では、ノードが密に生成されない
2. **ノード追加の遅延**: 累積誤差に基づくため、新しい領域へのノード追加が遅い

### Add-if-Silentルール

ネオコグニトロンの学習ルールに基づく概念：
> 「有用な入力に反応するニューロンがなければ、その位置に新しいニューロンを追加する」

#### RO-MAN 2023 版（単一閾値）

```
IF ||v_t - h_s1|| < θ_AiS AND ||v_t - h_s2|| < θ_AiS THEN
    新しいノード r を追加
```

- 入力が両勝者ノードから θ_AiS **以内** なら追加
- シンプルだが、既存ノードに近すぎる入力も追加される可能性

#### SMC 2023 版（範囲閾値）

```
IF θ_min < ||v_t - h_s1|| < θ_max AND θ_min < ||v_t - h_s2|| < θ_max THEN
    新しいノード r を追加:
    - h_r = v_t (入力をそのまま参照ベクトルとして使用)
    - E_r = 0.5 * (E_s1 + E_s2)
    - U_r = 0.5 * (U_s1 + U_s2)
    - エッジ: r-s1, r-s2 を接続
```

- **最小閾値 θ_min** により、既存ノードに近すぎる入力は追加されない（冗長ノード防止）
- **最大閾値 θ_max** により、遠すぎる入力も追加されない

### アルゴリズムの流れ

```
1. 2つのノードをランダム位置に生成
2. 各入力に対して:
   a. 最近傍ノード s1, s2 を探索
   b. s1-s2 間にエッジを追加
   c. Add-if-Silent条件をチェック → 条件満たせば入力を新ノードとして追加
   d. 累積誤差更新: E_s1 += ||v_t - h_s1||
   e. Utility更新: U_s1 += ||v_t - h_s2|| - ||v_t - h_s1||
   f. 参照ベクトル更新（勝者と近傍）
   g. エッジ年齢更新、古いエッジ削除
   h. κ間隔でUtility基準チェック → E_max/U_min > k なら最小Utilityノードを削除
   i. 誤差・Utility減衰
3. λ間隔で累積誤差に基づくノード追加（通常のGNG方式）
```

## パラメータ

### 基本パラメータ（GNG-U2共通）

| パラメータ | 論文の値 | 説明 |
|-----------|---------|------|
| λ | 300 | ノード追加間隔（通常のGNG方式） |
| κ | 10 | Utility基準チェック間隔 |
| η₁ (eps_b) | 0.08 | 勝者ノードの学習率 |
| η₂ (eps_n) | 0.008 | 近傍ノードの学習率 |
| AgeMax | 88 | エッジの最大年齢 |
| α | 0.5 | 分割時の誤差減衰率 |
| β | 0.005 | 誤差の全体減衰率 |
| χ | 0.005 | Utilityの減衰率 |
| k | 1000 | Utility基準の閾値 |

### AiS-GNG固有パラメータ

| パラメータ | 論文の値 | 説明 |
|-----------|---------|------|
| θ_AiS | 0.50 | tolerance領域の半径（RO-MAN版） |
| θ_AiS_min | 0.25 | tolerance領域の最小半径（SMC版） |
| θ_AiS_max | 0.50 | tolerance領域の最大半径（SMC版） |

**注意**: 論文の値は3D環境（メートル単位）向けです。2D [0,1]範囲では適切にスケーリングが必要です。

### 2D実装での推奨値

Triple ring (リング幅 ~0.08):
- θ_min: 0.02
- θ_max: 0.10

Tracking:
- θ_min: 0.01
- θ_max: 0.05

## GNG-Uとの違い

| 機能 | GNG-U | AiS-GNG |
|------|-------|---------|
| ノード追加方法 | λ間隔で累積誤差ベース | Add-if-Silentルール + λ間隔 |
| 高密度構造生成 | 遅い（疎領域で問題） | 速い（直接追加） |
| Utility基準チェック | λ間隔 | κ間隔（より頻繁） |
| 誤差の計算 | 二乗距離 | ユークリッド距離 |

## Amount of Movement (AM) 機能 (SMC 2023)

SMC 2023 論文では、動的オブジェクト検出のために各ノードの移動量を追跡します。

### AM の計算

```
AM_i(t) = γ_AM * AM_i(t-1) + ||h_i(t) - h_i(t-1)||
```

- `γ_AM`: 減衰率（0.95 など）、過去の移動を徐々に忘れる
- `||h_i(t) - h_i(t-1)||`: 現在の移動量

### 動的オブジェクト検出

```
IF AM_i > θ_AM THEN
    ノード i は「移動中」と分類
```

### 使用例

```python
from algorithms.ais_gng.python.model_am import AiSGNGAM, AiSGNGAMParams

params = AiSGNGAMParams(
    am_decay=0.95,      # AM減衰率
    am_threshold=0.01,  # 移動判定閾値
)
gng = AiSGNGAM(n_dim=2, params=params)
gng.train(X, n_iterations=5000)

# 移動量取得
movements = gng.get_node_movements()

# 移動ノードのマスク
moving_mask = gng.get_moving_nodes_mask()

# グラフを移動/静止部分に分割
moving_nodes, moving_edges, static_nodes, static_edges = gng.segment_by_movement()
```

## 応用

- 3Dポイントクラウド処理
- 未知物体の位相構造学習
- 移動ロボットの環境認識
- **動的オブジェクトの追跡・セグメンテーション**（AiS-GNG-AM版）

## 実装のポイント

1. **距離の計算**: 論文ではユークリッド距離（二乗ではない）を使用
2. **θ_AiSのスケーリング**: データのスケールに応じて調整が必要
3. **ラベリング**: 元論文では3D環境でオブジェクトカテゴリラベルを使用するが、2Dでは省略可能
4. **Utility基準**: κ間隔でE_max/U_min > kをチェック

## 参考文献

1. Fritzke, B. (1995). "A growing neural gas network learns topologies"
2. Fritzke, B. (1999). "Be Busy and Unique — or Be History"
3. Toda, Y., et al. (2016). "Real-time 3D point cloud segmentation using Growing Neural Gas with Utility"
4. Fukushima, K. (2014). "Add-if-Silent Rule for Training Multi-layered Convolutional Network Neocognitron"
