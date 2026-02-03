# DD-GNG (Dynamic Density Growing Neural Gas)

## 概要

DD-GNG（Dynamic Density Growing Neural Gas）は、GNG-Uをベースに**動的密度制御**機能を追加した拡張アルゴリズムです。注目領域（障害物、オブジェクトなど）でノード密度を自動的に増加させ、詳細なトポロジー構造を生成できます。

## 論文情報

- **著者**: Azhar Aulia Saputra, Wei Hong Chin, Yuichiro Toda, Naoyuki Takesue, Naoyuki Kubota
- **タイトル**: "Dynamic Density Topological Structure Generation for Real-Time Ladder Affordance Detection"
- **会議**: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2019
- **DOI**: 10.1109/IROS40897.2019.8967713

## 主要な特徴

### 1. ノード強度 (Strength)

各ノードは強度値 δ を持ち、注目領域内のノードは高い強度を持ちます：

```
δ = 1 + Σ(strength_bonus for each containing attention region)
```

### 2. 強度重み付きノード挿入

ノード挿入時の優先度計算に強度を使用：

```python
priority = error * (scale * strength)^power
```

これにより、注目領域で優先的にノードが挿入され、高いノード密度が実現されます。

### 3. 強度重み付き学習

学習率を強度で調整：

```python
effective_eps_b = eps_b / strength
```

高強度ノードは学習が遅くなり、位置が安定します。

### 4. 動的サンプリング（オプション）

注目領域から優先的にサンプリングを行い、その領域での学習を加速：

```python
model.train_with_density_sampling(data, attention_sampling_ratio=0.5)
```

## アルゴリズム

### Algorithm 1: Dynamic Topological Structure (論文より)

```
1. 初期化: 2ノードをランダム位置に生成
2. ループ:
   a. センサーデータ取得
   b. 障害物検出時は、その領域から優先的にサンプリング
   c. GNG-Uメインプロセス（strength考慮）
   d. ノード数 > 閾値 なら:
      - 三角形分割
      - セグメンテーション
      - 推測障害物の検出
      - 障害物領域のノード強度を増加
```

### Algorithm 2: Generate Strength (論文より)

```python
def calculate_strength(node_position, obstacles):
    δ = 1
    for obstacle in obstacles:
        A_min = obstacle.position - obstacle.size / 2
        A_max = obstacle.position + obstacle.size / 2
        if A_min < node_position < A_max:
            δ += 1
    return δ
```

## パラメータ

### 基本パラメータ（GNG-U2継承）

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| max_nodes | 100 | 最大ノード数 |
| lambda_ | 300 | ノード挿入間隔 |
| eps_b | 0.08 | 勝者学習率 (η1) |
| eps_n | 0.008 | 近傍学習率 (η2) |
| alpha | 0.5 | 分割時誤差減衰率 |
| beta | 0.005 | 全体誤差減衰率 |
| chi | 0.005 | Utility減衰率 |
| max_age | 88 | 最大エッジ年齢 |
| utility_k | 1000.0 | Utility閾値 |
| kappa | 10 | Utilityチェック間隔 |

### DD-GNG固有パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| strength_power | 4 | 強度の指数（論文では4） |
| strength_scale | 4.0 | 強度のスケール係数 |
| use_strength_learning | True | 学習率への強度適用 |
| use_strength_insertion | True | ノード挿入への強度適用 |

## 使用例

```python
from algorithms.dd_gng.python import DynamicDensityGNG, DDGNGParams

# パラメータ設定
params = DDGNGParams(
    max_nodes=150,
    lambda_=100,
    eps_b=0.1,
    strength_power=4,
    strength_scale=4.0,
)

# モデル作成
model = DynamicDensityGNG(n_dim=3, params=params, seed=42)

# 注目領域を追加（例: 障害物周辺）
model.add_attention_region(
    center=[0.5, 0.0, 0.1],  # 領域の中心
    size=[0.4, 0.08, 0.08],  # 領域のサイズ（半径）
    strength=5.0,            # 強度ボーナス
)

# 学習
model.train(data, n_iterations=8000)

# または動的サンプリング付き学習
model.train_with_density_sampling(
    data,
    n_iterations=8000,
    attention_sampling_ratio=0.5,  # 50%を注目領域から
)

# 結果取得
nodes, edges = model.get_graph()
strengths = model.get_node_strengths()
```

## GNG-U2との違い

| 機能 | GNG-U2 | DD-GNG |
|-----|--------|--------|
| ノード強度 | なし | あり（δ） |
| 動的密度制御 | なし | あり |
| 強度重み付き挿入 | なし | error * strength^4 |
| 強度重み付き学習 | なし | eps_b / strength |
| 注目領域設定 | なし | add_attention_region() |

## 応用例

論文では4脚ロボットの**リアルタイム梯子検出**に使用：

1. **環境認識**: 3D点群データからトポロジカル構造を生成
2. **障害物検出**: セグメンテーションにより推測される障害物を検出
3. **動的密度制御**: 障害物（梯子）周辺でノード密度を増加
4. **把持点検出**: 梯子のラングの位置と把持可能領域を特定

## リファレンス実装

- `references/original_code/azhar_ddgng/`: Kubota研究室のオリジナル実装
- `algorithms/dd_gng/python/model.py`: 本リポジトリの実装

## 関連アルゴリズム

- **GNG-U**: Utility付きGNG（非定常分布対応）
- **GNG-U2**: GNG-U改良版（κ間隔Utilityチェック）
- **AiS-GNG**: Add-if-Silentルール付きGNG
- **GNG-DT**: 異なるトポロジー構造を持つGNG（3D空間知覚）
