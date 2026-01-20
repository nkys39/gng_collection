# Growing Neural Gas (GNG)

## 概要

Growing Neural Gas (GNG)は、Fritzkeによって1995年に提案された教師なし学習アルゴリズムである。
入力データの位相構造を保持しながら、ノード数を自動的に調整してベクトル量子化を行う。

## 元論文

- Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
  Advances in Neural Information Processing Systems 7 (NIPS 1994)

## アルゴリズム

1. 2つのノードで初期化
2. 入力信号 ξ に対して:
   - 最近傍ノード s1 と次近傍ノード s2 を見つける
   - s1 のエラーを増加: `error_s1 += ||ξ - w_s1||²`
   - s1 を入力方向に移動: `w_s1 += ε_b * (ξ - w_s1)`
   - s1 の直接隣接ノードも移動: `w_n += ε_n * (ξ - w_n)`
   - s1-s2 間のエッジを作成/リセット (age = 0)
   - s1 に接続する全エッジの age を増加
   - max_age を超えたエッジを削除
   - 孤立ノードを削除
3. λ 回の学習ごとに:
   - 最大エラーノード q を見つける
   - q の隣接ノードの中で最大エラーノード f を見つける
   - q-f 間に新ノード r を挿入
   - エラーを分配: `error_q *= α, error_f *= α, error_r = error_q`
4. 全ノードのエラーを減衰: `error *= (1 - β)`

## パラメータ

| パラメータ | 説明 | 典型値 |
|-----------|------|--------|
| max_nodes | 最大ノード数 | 100 |
| λ (lambda_) | ノード挿入間隔 | 100 |
| ε_b (eps_b) | 勝者ノードの学習率 | 0.05-0.2 |
| ε_n (eps_n) | 隣接ノードの学習率 | 0.005-0.02 |
| α (alpha) | ノード挿入時のエラー減衰 | 0.5 |
| β (beta) | 全体のエラー減衰率 | 0.0005-0.005 |
| max_age | エッジの最大寿命 | 50-100 |

## demogng.de 実装との比較

demogng.de の実装を参照して以下の点を確認:

### エラー計算
demogng.de では二乗距離を使用:
```javascript
// demogng.de (vbnn.js)
bmu.error += mindist;  // mindist は二乗距離
```

本実装でも同様に二乗距離を使用:
```python
dist_sq = np.sum((sample - self.nodes[s1_id].weight) ** 2)
self.nodes[s1_id].error += dist_sq
```

### デフォルトパラメータ (demogng.de)
- nodes: 100
- lambda: 300
- epsilonB: 0.05
- epsilonN: 0.006
- alpha: 0.5
- amax: 50
- beta: 0.0005

## 特徴

- **自動的なトポロジー学習**: Competitive Hebbian Learning によりエッジを形成
- **動的なノード数**: データの複雑さに応じてノードが増減
- **分布追従性**: エッジの age 管理により分布の変化に対応可能
- **非定常分布**: 基本的には定常分布向けだが、パラメータ調整でトラッキングにも対応

## 派生アルゴリズムとの関係

- **GNG-U**: Utility を追加して非定常分布への対応を強化
- **Neural Gas**: GNG の前身、エッジなし・ノード数固定
- **SOM**: グリッド構造固定、GNG は自由なトポロジー
- **GCS**: 三角形メッシュによるトポロジー学習
