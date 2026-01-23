# AiS-GNG (Add-if-Silent Rule-Based Growing Neural Gas)

## 概要

AiS-GNG (Add-if-Silent Rule-Based Growing Neural Gas) は、GNG-U (Growing Neural Gas with Utility) を拡張し、Add-if-Silentルールを導入することで、高密度な位相構造を素早く生成できるアルゴリズムです。

## 論文情報

### 主要論文

1. **RO-MAN 2023**
   - Shoji, M., Obo, T., & Kubota, N.
   - "Add-if-Silent Rule-Based Growing Neural Gas for High-Density Topological Structure of Unknown Objects"
   - IEEE RO-MAN 2023, pp. 2492-2498

2. **SMC 2023 (拡張版: AiS-GNG-AM)**
   - Shoji, M., Obo, T., & Kubota, N.
   - "Add-if-Silent Rule-Based Growing Neural Gas with Amount of Movement for High-Density Topological Structure Generation of Dynamic Object"
   - IEEE SMC 2023, pp. 3040-3047

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

AiS-GNGでは、以下の条件を満たす場合、入力データを直接新しいノードとして追加します：

```
IF θ_min < ||v_t - h_s1|| < θ_max AND θ_min < ||v_t - h_s2|| < θ_max THEN
    新しいノード r を追加:
    - h_r = v_t (入力をそのまま参照ベクトルとして使用)
    - E_r = 0.5 * (E_s1 + E_s2)
    - U_r = 0.5 * (U_s1 + U_s2)
    - エッジ: r-s1, r-s2 を接続
```

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

## 応用

- 3Dポイントクラウド処理
- 未知物体の位相構造学習
- 移動ロボットの環境認識
- 動的オブジェクトの追跡（AiS-GNG-AM版）

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
