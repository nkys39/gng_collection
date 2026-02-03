# GNG Efficiently (Fišer et al., 2013) 再現実装ノート

**実装状況**: ✅ 完了（Python/C++）

## 論文情報

- **タイトル**: Growing Neural Gas Efficiently
- **著者**: Daniel Fišer, Jan Faigl, Miroslav Kulich
- **出典**: Neurocomputing, 2013
- **所属**: Czech Technical University in Prague

## 論文の概要

GNGアルゴリズムの2つの最も時間がかかる操作を特定し、それぞれに対する最適化テクニックを提案：

| 操作 | 計算時間の割合 | 最適化手法 |
|------|---------------|-----------|
| 最近傍探索 (NN Search) | 約48% | Uniform Grid (UG) |
| エラー処理 (Error Handling) | 約51% | 遅延評価 + Lazy Heap |

**結果**: 両最適化を組み合わせることで、オリジナルGNGより数百倍の高速化を実現。

---

## 1. アルゴリズムの詳細

### 1.1 代替的なGNG定式化（Algorithm 3）

オリジナルGNGと等価だが、最適化を適用しやすい形式に再構成：

```
gng():
  1. Gを2つのランダムノードで初期化
  2. c ← 0  (サイクルカウンタ)
  3. while 停止条件が満たされない:
  4.   for s ← 0 to λ-1:  (適応ステップ)
  5.     ξ ← ランダム入力信号
  6.     gng_adapt(c, s, ξ)
  7.   gng_new_node(c)  (ノード挿入)
  8.   c ← c + 1

gng_adapt(c, s, ξ):
  1. ν, μ ← 2最近傍ノード(ξ)
  2. inc_error(c, s, ν, ||w_ν - ξ||²)
  3. w_ν ← w_ν + ε_b(ξ - w_ν)
  4. w_n ← w_n + ε_n(ξ - w_n), ∀n ∈ N_ν
  5. νとμ間にエッジがなければ作成
  6. A_ν,μ ← 0
  7. foreach n in N_ν:
  8.   A_n,ν ← A_n,ν + 1
  9.   if A_n,ν > A_max:
  10.    エッジ削除、孤立ノードも削除
  11. dec_all_error(β)

gng_new_node(c):
  1. q, f ← largest_error(c)  (最大エラーノードとその最大エラー隣接)
  2. w_r ← (w_q + w_f) / 2
  3. q-f間エッジを削除、r-qとr-fエッジを作成
  4. dec_error(c, q, α)
  5. dec_error(c, f, α)
  6. set_error(c, r, (E_q + E_f) / 2)
```

### 1.2 最適化1: Uniform Grid (UG) による最近傍探索

#### 概念
- 空間を均一なセルに分割してインデックス化
- ノード位置からセル座標を計算: `p = floor((w - o) / l)`
  - `o`: グリッド原点
  - `l`: セル辺長

#### 探索手順
1. 入力信号ξの属するセルCを特定
2. 境界距離 `b = min_i min(|ξ_i - o_i - p_i*l|, |ξ_i - o_i - (p_i+1)*l|)` を計算
3. セルC内で2最近傍を探索
4. 見つからなければ `b ← b + l` として近隣セルに探索範囲を拡大
5. 2最近傍が見つかるまで繰り返し

#### Growing Uniform Grid
動的にグリッドサイズを調整：
- `h_d`: グリッド密度（セルあたり平均ノード数）
- `h_t`: 許容最大密度閾値（デフォルト: 0.1）
- `h_ρ`: 拡張係数（デフォルト: 1.5）

```
if h_d > h_t:
    新グリッドを構築（h_ρ倍のセル数）
    全ノードを再挿入
```

### 1.3 最適化2: エラー処理の遅延評価

#### 問題点
- `dec_all_error(β)`: 毎ステップ全ノードのエラーを減衰 → O(n)
- `largest_error()`: 最大エラーノードを線形探索 → O(n)

#### 解決策: サイクルカウンタによる遅延計算

各ノードνが「最後にエラーが更新されたサイクル」`C_ν`を保持。
実際のエラー値は必要時に計算：

```
E_ν,c_j = β^((c_j - c_0)λ) * E_ν,c_0 + Σ β^(λ-s_i) * v_i
```

#### 新しいエラー処理関数（Algorithm 4）

```
fix_error(c, ν):
  E_ν ← β^(λ(c - C_ν)) * E_ν
  C_ν ← c

inc_error*(c, s, ν, v):
  fix_error(c, ν)
  E_ν ← β^(λ-s) * E_ν + v
  ヒープ内のνを更新

dec_error*(c, ν, α):
  fix_error(c, ν)
  E_ν ← α * E_ν
  ヒープ内のνを更新

set_error*(c, ν, v):
  E_ν ← v
  C_ν ← c
  ヒープにνを挿入

dec_all_error*(β):
  nop()  // 何もしない

largest_error*(c):
  q ← ヒープからトップノード
  f ← arg max_{n ∈ N_q} E_n
  return q, f
```

#### Lazy Heap
- 挿入・更新を遅延リストLに追加
- `top`操作時にまとめて処理
- サイクルカウンタが異なるノードは`fix_error`で補正

---

## 2. 実験パラメータ（論文Table 2）

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| ε_b | 0.05 | 勝者学習率 |
| ε_n | 0.0006 | 近傍学習率 |
| λ | 200 | ノード挿入間隔 |
| β | 0.9995 | エラー減衰率（毎ステップ） |
| α | 0.95 | ノード挿入時のエラー減衰率 |
| A_max | 200 | 最大エッジ年齢 |
| h_t | 0.1 | グリッド密度閾値 |
| h_ρ | 1.5 | グリッド拡張係数 |

---

## 3. 再現実装計画

### 3.1 ディレクトリ構造

```
algorithms/
└── gng_efficient/
    ├── python/
    │   ├── __init__.py
    │   ├── model.py           # メインアルゴリズム
    │   ├── uniform_grid.py    # Uniform Grid実装
    │   └── lazy_heap.py       # Lazy Heap実装
    └── cpp/
        ├── gng_efficient.hpp
        ├── uniform_grid.hpp
        └── lazy_heap.hpp
```

### 3.2 実装フェーズ

#### Phase 1: 基本構造（Python）

1. **Uniform Grid クラス**
   ```python
   class UniformGrid:
       def __init__(self, origin, cell_size, grid_dims):
           ...
       def insert(self, node): ...
       def remove(self, node): ...
       def update(self, node, old_pos, new_pos): ...
       def find_two_nearest(self, point) -> Tuple[Node, Node]: ...
   ```

2. **Growing Uniform Grid クラス**
   ```python
   class GrowingUniformGrid(UniformGrid):
       def __init__(self, h_t=0.1, h_rho=1.5):
           ...
       def check_and_rebuild(self): ...
   ```

3. **Lazy Heap クラス**
   ```python
   class LazyHeap:
       def __init__(self):
           self.heap = []  # pairing heap or binary heap
           self.pending_list = []
       def insert(self, node): ...
       def update(self, node): ...
       def top(self, current_cycle) -> Node: ...
   ```

#### Phase 2: GNG Efficient 本体

```python
class GNGEfficient:
    def __init__(self, params: GNGEfficientParams):
        self.grid = GrowingUniformGrid(params.h_t, params.h_rho)
        self.error_heap = LazyHeap()
        self.cycle = 0
        self.beta_powers = self._precompute_beta_powers()

    def _precompute_beta_powers(self):
        """β^i を事前計算（i = 0 to λ）"""
        return [self.params.beta ** i for i in range(self.params.lambda_ + 1)]

    def adapt(self, input_signal, step):
        """gng_adapt(c, s, ξ)"""
        ...

    def insert_node(self):
        """gng_new_node(c)"""
        ...

    def fix_error(self, node):
        """エラー値を現在サイクルに補正"""
        ...
```

#### Phase 3: C++実装

Python実装と同等のC++版を作成。

### 3.3 パラメータクラス

```python
@dataclass
class GNGEfficientParams:
    # 標準GNGパラメータ
    max_nodes: int = 100
    lambda_: int = 200
    eps_b: float = 0.05
    eps_n: float = 0.0006
    alpha: float = 0.95
    beta: float = 0.9995
    max_age: int = 200

    # Uniform Grid パラメータ
    h_t: float = 0.1      # 密度閾値
    h_rho: float = 1.5    # 拡張係数

    # 最適化フラグ（比較用）
    use_uniform_grid: bool = True
    use_lazy_error: bool = True
```

---

## 4. 検証テスト計画

### 4.1 正確性検証（結果の同一性）

**目的**: 最適化版が通常GNGと同じ結果を生成することを確認

```python
def test_correctness():
    """同一シードで通常GNGと結果比較"""
    np.random.seed(42)

    # 通常GNG
    gng_orig = GrowingNeuralGas(params_orig)
    gng_orig.fit(data)

    # 最適化GNG
    np.random.seed(42)
    gng_eff = GNGEfficient(params_eff)
    gng_eff.fit(data)

    # ノード位置の比較
    assert_nodes_equal(gng_orig.nodes, gng_eff.nodes)
    # エッジ構造の比較
    assert_edges_equal(gng_orig.edges, gng_eff.edges)
```

### 4.2 パフォーマンステスト

**目的**: 論文のTable 3-8を再現

```python
def test_performance():
    """ノード数別の計算時間を測定"""
    node_counts = [1000, 5000, 10000, 25000, 50000]

    results = {
        'Orig': [],      # 線形NN探索 + 通常エラー処理
        'UG': [],        # Uniform Grid + 通常エラー処理
        'Err': [],       # 線形NN探索 + 遅延エラー処理
        'UG+Err': [],    # 両方の最適化
    }

    for n_nodes in node_counts:
        for variant in results.keys():
            time = measure_time(variant, n_nodes, data)
            results[variant].append(time)

    plot_performance_comparison(results)
```

### 4.3 標準テスト（本リポジトリ準拠）

#### 4.3.1 Triple Ring テスト

```bash
experiments/2d_visualization/test_gng_efficient_triple_ring.py
```

| 項目 | 値 |
|------|-----|
| max_nodes | 100 |
| lambda_ | 100 |
| n_iterations | 5000 |
| 出力 | triple_ring_final.png, triple_ring_growth.gif |

#### 4.3.2 Tracking テスト

```bash
experiments/2d_visualization/test_gng_efficient_tracking.py
```

| 項目 | 値 |
|------|-----|
| max_nodes | 50 |
| lambda_ | 20 |
| total_frames | 120 |
| 出力 | tracking.gif |

#### 4.3.3 3D Floor & Wall テスト

```bash
experiments/3d_pointcloud/test_gng_efficient_floor_wall.py
```

| 項目 | 値 |
|------|-----|
| max_nodes | 150 |
| n_iterations | 8000 |
| 出力 | floor_wall_final.png, floor_wall_growth.gif |

### 4.4 スケーラビリティテスト

**目的**: 大規模ノード数での性能評価

```python
def test_scalability():
    """100,000ノードまでのスケーラビリティ"""
    node_counts = [10000, 25000, 50000, 100000]

    for n_nodes in node_counts:
        gng = GNGEfficient(params)
        start = time.time()
        gng.fit(data, max_nodes=n_nodes)
        elapsed = time.time() - start

        print(f"{n_nodes} nodes: {elapsed:.2f}s")
```

---

## 5. 実装の注意点

### 5.1 β^i の事前計算

```python
# 論文Algorithm 4のコメント:
# "Note that all powers of β can be pre-computed into
# an array and the expensive processor operation can be avoided."

self.beta_powers = np.array([
    self.params.beta ** i
    for i in range(self.params.lambda_ + 1)
])
```

### 5.2 エッジ年齢の違い

論文の代替定式化では、エッジ年齢が実質的に1から開始（0ではなく）：

> "The difference between these two formulations is that the age of
> the activated edge is effectively set to value 1 instead of 0."

### 5.3 αパラメータの調整

代替定式化では dec_all_error の実行タイミングが異なるため：

> "This difference can be compensated by a selection of the
> parameter α so that α = β·α_orig."

### 5.4 Lazy Heap の正当性

論文Section 5.2で証明：
- Lemma 1: C_ν ≤ c ならば E_ν,c ≤ E_ν,C_ν
- Lemma 2: top操作で最終的にC_μ = cとなるノードが得られる
- Theorem 3: そのノードが最大エラーを持つ

---

## 6. 期待される成果

### 6.1 性能向上（論文の結果に基づく）

| ノード数 | Orig (秒) | UG+Err (秒) | 高速化率 |
|---------|-----------|-------------|---------|
| 1,000 | 1.3 | 0.12 | 10x |
| 5,000 | 46 | 0.77 | 60x |
| 10,000 | 190 | 1.68 | 113x |
| 25,000 | 1,259 | 4.59 | 274x |
| 50,000 | 8,294 | 10.67 | 777x |

### 6.2 成果物

1. **Python実装**: `algorithms/gng_efficient/python/`
2. **C++実装**: `algorithms/gng_efficient/cpp/`
3. **テストスクリプト**: `experiments/2d_visualization/test_gng_efficient_*.py`
4. **サンプル出力**: `experiments/2d_visualization/samples/gng_efficient/`
5. **性能比較レポート**: `experiments/benchmarks/gng_efficient_benchmark.py`

---

## 7. スケジュール案

| フェーズ | タスク | 優先度 |
|---------|--------|--------|
| 1 | Uniform Grid (Python) | 高 |
| 2 | Lazy Heap (Python) | 高 |
| 3 | GNGEfficient 本体 (Python) | 高 |
| 4 | 正確性テスト | 高 |
| 5 | 標準テスト (triple_ring, tracking) | 中 |
| 6 | パフォーマンステスト | 中 |
| 7 | C++実装 | 中 |
| 8 | 3Dテスト | 低 |
| 9 | ドキュメント整備 | 低 |

---

## 8. 実装詳細ノート（論文準拠ポイント）

実装時に確認した論文準拠の重要ポイント：

### 8.1 Uniform Grid の初期化（Section 4.2）

論文の記述：
> "The growing uniform grid starts with a single cell encapsulating an axis aligned bounding box of the input signals."

**実装**: `train()`の開始時に入力データの最小/最大座標を計算し、その境界ボックスで単一セルのグリッドを初期化。

```python
# model.py - train()
min_coords = np.min(data, axis=0)
max_coords = np.max(data, axis=0)
self._grid.initialize_with_bounds(min_coords, max_coords)
```

### 8.2 エッジ年齢の初期化（Algorithm 3, Step 6）

論文の記述：
> "A_ν,μ ← 0"

その後、Step 8で全ての隣接エッジ年齢をインクリメント：
> "A_n,ν ← A_n,ν + 1"

**実装**: `_add_edge()`でエッジ年齢を0に設定。これにより新規作成エッジは隣接ループ後に1になる。

```python
def _add_edge(self, node1: int, node2: int) -> None:
    """Per Algorithm 3, step 6: A_ν,μ ← 0"""
    # Reset age to 0 (will be incremented to 1 in the neighbor loop)
    self.edges[node1, node2] = 0
    self.edges[node2, node1] = 0
```

### 8.3 ステップカウンタの追跡（Algorithm 4）

`inc_error*(c, s, ν, v)`には現在のステップ`s`が必要（`β^(λ-s)`の計算用）。

**実装**: `adapt()`の開始時に`step = n_learning % lambda`を計算。

```python
def adapt(self, input_signal: np.ndarray) -> None:
    self.step = self.n_learning % self.params.lambda_
    # ... inc_error uses self.step for β^(λ-s)
```

### 8.4 β^i の事前計算（Algorithm 4 コメント）

論文の記述：
> "Note that all powers of β can be pre-computed into an array and the expensive processor operation can be avoided."

**実装**: 初期化時に`β^0`から`β^λ`まで事前計算。

```python
self._beta_powers = np.array([
    self.params.beta ** i for i in range(self.params.lambda_ + 1)
])
```

---

## 9. 参考文献

- [1] B. Fritzke, "A growing neural gas network learns topologies," NIPS 1995
- [26] M. L. Fredman et al., "The pairing heap," Algorithmica 1986
- [28] Fermat Library: http://www.danfis.cz (論文著者の実装)
