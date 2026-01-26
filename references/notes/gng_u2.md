# GNG-U2 (Growing Neural Gas with Utility - Variant 2)

## 概要

GNG-U2は、GNG-U (Fritzke 1997/1999) を改良したアルゴリズムで、3Dポイントクラウドのリアルタイムセグメンテーションを目的として開発されました。

## 論文情報

- **Toda, Y., Kubota, N., Kuno, Y., & Yamada, K. (2016)**
- "Real-time 3D point cloud segmentation using Growing Neural Gas with Utility"
- IEEE International Conference on Robotics and Automation (ICRA) 2016
- pp. 3256-3262

## GNG-Uからの主な改良点

### 1. κ間隔でのUtilityチェック

従来のGNG-UではUtility基準をλ間隔（ノード挿入時）にのみチェックしていましたが、GNG-U2ではκ間隔（κ=10）で独立してチェックします。

```
従来: λ=300 → 300回に1回チェック
GNG-U2: κ=10 → 10回に1回チェック（30倍頻繁）
```

これにより、不要なノードを素早く除去でき、非定常分布への追従性が向上します。

### 2. ユークリッド距離の使用

誤差とUtilityの計算に二乗距離ではなくユークリッド距離を使用します。

```
GNG-U: E += ||v - h||²
GNG-U2: E += ||v - h||
```

### 3. 分離したUtility減衰率

誤差減衰率βとは別に、Utility減衰率χを設定できます（実用上は同じ値を使用することが多い）。

## アルゴリズム

```
1. 2つのノードをランダム位置に生成
2. 入力 v に対して:
   a. 最近傍ノード s1, s2 を探索
   b. s1-s2 間にエッジを追加
   c. 誤差更新: E_s1 += ||v - h_s1||
   d. Utility更新: U_s1 += ||v - h_s2|| - ||v - h_s1||
   e. 参照ベクトル更新（勝者と近傍）
   f. エッジ年齢更新、古いエッジ削除
   g. κ間隔でUtility基準チェック → E_max/U_min > k なら最小Utilityノード削除
   h. 誤差・Utility減衰
3. λ間隔で累積誤差に基づくノード追加
```

## パラメータ

| パラメータ | 論文の値 | 説明 |
|-----------|---------|------|
| λ | 300 | ノード追加間隔 |
| κ | 10 | Utility基準チェック間隔 |
| η₁ (eps_b) | 0.08 | 勝者ノードの学習率 |
| η₂ (eps_n) | 0.008 | 近傍ノードの学習率 |
| g_max (max_age) | 88 | エッジの最大年齢 |
| α | 0.5 | 分割時の誤差減衰率 |
| β | 0.005 | 誤差の減衰率 |
| χ | 0.005 | Utilityの減衰率 |
| k | 1000 | Utility基準の閾値 |

## 2D実装での推奨値

3D論文の値は3D環境向けなので、2D [0,1]範囲では以下を推奨：

### 静的分布（トリプルリング）
```python
params = GNGU2Params(
    max_nodes=100,
    lambda_=100,      # 調整済み
    kappa=10,
    eps_b=0.08,
    eps_n=0.008,
    max_age=100,
    utility_k=1000.0,
)
```

### 動的分布（追跡）
```python
params = GNGU2Params(
    max_nodes=50,
    lambda_=20,       # より頻繁にノード挿入
    kappa=5,          # より頻繁にUtilityチェック
    eps_b=0.15,       # 高い学習率
    eps_n=0.01,
    max_age=30,       # 短いエッジ寿命
    utility_k=500.0,  # 低い閾値
)
```

## 派生アルゴリズム

GNG-U2は以下のアルゴリズムのベースとなっています：

- **AiS-GNG** (Shoji et al. 2023): Add-if-Silentルールを追加
- **AiS-GNG-AM** (Shoji et al. 2023): 移動量を考慮した動的オブジェクト追跡

## 応用分野

- 3Dポイントクラウドセグメンテーション
- リアルタイムロボットビジョン
- 移動ロボットの環境認識
- オンライン形状学習

## 参考文献

1. Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
2. Fritzke, B. (1999). "Be Busy and Unique — or Be History—The Utility Criterion"
3. Toda, Y., et al. (2016). "Real-time 3D point cloud segmentation using Growing Neural Gas with Utility"
