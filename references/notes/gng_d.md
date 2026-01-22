# GNG-D (Growing Neural Gas with explicit Delaunay)

## 概要

GNG-D は標準GNGの Competitive Hebbian Learning (CHL) を scipy.spatial.Delaunay による明示的な Delaunay 三角形分割に置き換えた変種である。これにより、厳密な幾何学的三角形メッシュ構造を保証する。

**注意**: これは GNG-T (Kubota & Satomi 2008) とは異なるアルゴリズムである。GNG-T はCHLベースのGNGにヒューリスティックな三角形分割（四角形探索・交差点探索）を追加した手法である。

## 理論的背景

### Martinetz & Schulten (1994) の2つのアプローチ

元論文「Topology representing networks」では、Delaunay 三角形分割を形成する2つの方法が提案されている：

1. **CHL (Competitive Hebbian Learning)** - オンライン、インクリメンタル
   - 勝者と次点の間にエッジを作成
   - データ分布に沿った「誘導Delaunay」を形成
   - 標準GNG、GNG-Tで使用

2. **明示的Delaunay計算** - オフライン、幾何学的
   - ノード位置から直接Delaunay三角形分割を計算
   - 純粋な幾何学的三角形分割
   - **GNG-Dで使用**

### CHL vs 明示的Delaunay

| 特性 | CHL (標準GNG/GNG-T) | 明示的Delaunay (GNG-D) |
|-----|---------------------|----------------------|
| エッジ決定 | データ駆動 | 幾何学駆動 |
| 計算量 | O(1) per step | O(n log n) per update |
| トポロジー | 誘導Delaunay（近似） | 厳密Delaunay |
| データがない領域 | エッジなし | エッジあり（凸包内） |
| エッジaging | あり (max_age) | なし |

## アルゴリズム

### 標準GNGとの違い

1. **エッジ管理の変更**
   - CHL による勝者-次点間エッジ作成を廃止
   - scipy.spatial.Delaunay で三角形分割を計算
   - max_age パラメータ不要

2. **トポロジー更新タイミング**
   - `update_topology_every` 回ごとに再計算
   - ノード追加時にも強制更新

3. **近傍の定義**
   - 標準GNG: エッジで接続されたノード
   - GNG-D: Delaunay 三角形で隣接するノード

### 学習アルゴリズム

```
1. 入力信号 ξ に対して:
   - 最近傍ノード（勝者）を見つける
   - 勝者のエラーを増加: error += ||ξ - w_winner||²
   - 勝者を入力方向に移動: w += ε_b * (ξ - w)
   - Delaunay近傍も移動: w_n += ε_n * (ξ - w_n)

2. update_topology_every 回ごとに:
   - Delaunay三角形分割を再計算
   - エッジと近傍リストを更新

3. λ 回ごとに:
   - 最大エラーノード q を見つける
   - q の近傍で最大エラーの f を見つける
   - q と f の中間に新ノードを挿入
   - トポロジーを強制更新
```

## パラメータ

| パラメータ | 説明 | 典型値 |
|-----------|------|--------|
| max_nodes | 最大ノード数 | 100 |
| λ (lambda_) | ノード挿入間隔 | 100 |
| ε_b (eps_b) | 勝者の学習率 | 0.05 |
| ε_n (eps_n) | 近傍の学習率 | 0.006 |
| α (alpha) | エラー減衰（挿入時） | 0.5 |
| β (beta) | エラー減衰（毎ステップ） | 0.0005 |
| update_topology_every | 三角形分割更新間隔 | 10 |

### パラメータ調整のヒント

- **update_topology_every**:
  - 小さい値 → 正確なトポロジー、高い計算コスト
  - 大きい値 → 効率的だがトポロジーが遅延
  - トラッキング用途: 5-10
  - 静的分布: 10-50

## 利点

1. **厳密な三角形メッシュ**
   - 交差するエッジがない
   - 正しい三角形構造を保証

2. **サーフェス再構築に最適**
   - 3Dスキャンデータの処理
   - メッシュ生成の前処理

3. **可視化が美しい**
   - 三角形を塗りつぶして表示可能
   - Voronoi図との双対関係を活用可能

## 欠点

1. **計算コストが高い**
   - Delaunay計算は O(n log n)
   - 大量ノードで遅くなる

2. **データがない領域にもエッジ**
   - 凸包内全体に三角形ができる
   - 穴がある分布では不適切なエッジが発生

3. **2次元以上が必要**
   - 1次元データには使用不可
   - scipy.spatial.Delaunay の制約

## 使用場面

### 適している
- メッシュ生成・サーフェス再構築
- 厳密な三角形構造が必要な場合
- 可視化で三角形を表示したい場合
- 凸形状のデータ

### 適していない
- 穴がある分布（リング、ドーナツ形状）
- 高次元データ
- リアルタイム処理（計算コスト）
- 1次元データ

## GNG-T との違い

| 特性 | GNG-D | GNG-T (Kubota 2008) |
|-----|-------|---------------------|
| ベース | 独自 | GNG + CHL |
| 三角形分割 | scipy.spatial.Delaunay | ヒューリスティック |
| エッジaging | なし | あり (max_age) |
| 初期化 | 2ノード | 3ノード（2D単体） |
| 交差エッジ | 発生しない | 交差点探索で除去 |

## 実装の注意点

### Delaunay計算の失敗

点が共線（同一直線上）の場合、Delaunay計算が失敗する可能性がある。
フォールバックとして最近傍接続を実装している。

```python
try:
    tri = Delaunay(positions)
except Exception:
    self._fallback_connect_nearest(active_ids)
```

### 凸包外のノード

Delaunay三角形分割は凸包内のみを対象とする。
凸包外のノードは近傍に含まれない可能性がある。

## 参考文献

1. Martinetz, T. & Schulten, K. (1994). "Topology representing networks"
   Neural Networks, 7(3), 507-522.

2. Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
   Advances in Neural Information Processing Systems 7.

3. scipy.spatial.Delaunay ドキュメント
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
