# Claude Code プロジェクトガイド

このリポジトリはGrowing Neural Gas (GNG)とその関連アルゴリズムのコレクションです。

## リポジトリ構造

```
gng_collection/
├── algorithms/           # アルゴリズム実装
│   ├── _template/        # 新アルゴリズム用テンプレート
│   ├── gng/              # 標準GNG
│   ├── gng_u/            # GNG-U (Utility)
│   ├── gng_u2/           # GNG-U2 (Utility V2 - κ間隔チェック)
│   ├── ais_gng/          # AiS-GNG (Add-if-Silent Rule)
│   ├── gng_t/            # GNG-T (Triangulation - Kubota 2008)
│   ├── gng_d/            # GNG-D (explicit Delaunay)
│   ├── som/              # Self-Organizing Map
│   ├── neural_gas/       # Neural Gas
│   ├── gcs/              # Growing Cell Structures
│   ├── hcl/              # Hard Competitive Learning
│   ├── lbg/              # Linde-Buzo-Gray
│   └── growing_grid/     # Growing Grid
├── experiments/          # 実験コード
│   └── 2d_visualization/
│       ├── _templates/   # テストテンプレート
│       │   ├── triple_ring.py
│       │   └── tracking.py
│       ├── samples/      # 出力サンプル（git管理対象）
│       │   ├── gng/
│       │   │   ├── python/
│       │   │   └── cpp/
│       │   ├── gng_u/
│       │   ├── gng_u2/
│       │   ├── ais_gng/
│       │   ├── gng_t/
│       │   │   ├── python/
│       │   │   └── cpp/
│       │   ├── som/
│       │   ├── neural_gas/
│       │   └── gcs/
│       └── test_*.py     # テストスクリプト
├── data/2d/              # 2Dデータ関連
│   ├── sampler.py
│   └── shapes/
├── references/           # 参照資料
│   ├── notes/            # アルゴリズムノート
│   └── original_code/    # リファレンス実装
└── python/               # 共通Pythonコード
```

## 新しいアルゴリズムの追加手順

### 1. アルゴリズム実装

```bash
# テンプレートをコピー
cp -r algorithms/_template algorithms/[algorithm_name]

# model.pyを編集
# - クラス名を変更 (AlgorithmName -> YourAlgorithm)
# - パラメータクラスを更新
# - アルゴリズム固有のロジックを実装
```

### 2. テストの作成

```bash
cd experiments/2d_visualization

# トリプルリングテスト
cp _templates/triple_ring.py test_[algorithm]_triple_ring.py
# - import文を更新
# - クラス名を更新
# - 出力ファイル名を更新

# トラッキングテスト
cp _templates/tracking.py test_[algorithm]_tracking.py
# - 同様に更新
```

### 3. テストの実行

```bash
cd experiments/2d_visualization

# トリプルリングテスト
python test_[algorithm]_triple_ring.py

# トラッキングテスト
python test_[algorithm]_tracking.py
```

### 4. サンプルの保存

```bash
# ディレクトリ作成
mkdir -p samples/[algorithm]/python

# 出力を移動（ファイル名は統一形式に）
mv [algorithm]_triple_ring_final.png samples/[algorithm]/python/triple_ring_final.png
mv [algorithm]_triple_ring_growth.gif samples/[algorithm]/python/triple_ring_growth.gif
mv [algorithm]_tracking.gif samples/[algorithm]/python/tracking.gif

# gitに追加（-f が必要）
git add -f samples/[algorithm]/
```

### 5. リファレンスノートの作成

```bash
# references/notes/[algorithm].md に以下を記載:
# - アルゴリズムの概要
# - 元論文の情報
# - 標準GNGとの違い
# - パラメータの説明
```

## 標準テストパラメータ

### トリプルリングテスト（静的分布）
```python
params = Params(
    max_nodes=100,
    lambda_=100,
    eps_b=0.08,
    eps_n=0.008,
    alpha=0.5,
    beta=0.005,
    max_age=100,
)
n_iterations = 5000
n_samples = 1500
```

### トラッキングテスト（動的分布）
```python
params = Params(
    max_nodes=50,
    lambda_=20,       # より頻繁にノード挿入
    eps_b=0.15,       # より高い学習率
    eps_n=0.01,
    alpha=0.5,
    beta=0.01,        # より速い減衰
    max_age=30,       # より短いエッジ寿命
)
total_frames = 120
samples_per_frame = 50
```

## .gitignore について

`experiments/**/*.gif` と `experiments/**/*.png` は除外されていますが、
`experiments/**/samples/` は例外として追跡されます。

出力ファイルをgitに追加する場合は `-f` フラグを使用:
```bash
git add -f experiments/2d_visualization/samples/
```

## 実装済みアルゴリズム

| アルゴリズム | Python | C++ | 説明 |
|-------------|:------:|:---:|------|
| GNG         | ✓      | ✓   | 標準 Growing Neural Gas |
| GNG (Kubota)| ✓      | -   | Kubotalab論文準拠版（最長エッジ方式） |
| GNG-U       | ✓      | ✓   | Utility付きGNG（非定常分布対応） |
| GNG-U2      | ✓      | -   | GNG-U改良版（κ間隔Utilityチェック、AiS-GNGのベース） |
| AiS-GNG     | ✓      | -   | Add-if-Silentルール付きGNG（高密度位相構造の高速生成） |
| GNG-T       | ✓      | ✓   | ヒューリスティック三角形分割（Kubota 2008） |
| GNG-T (Kubota)| ✓    | -   | Kubotalab論文完全準拠版（γ式使用） |
| GNG-D       | ✓      | -   | 明示的Delaunay三角形分割（scipy.spatial） |
| SOM         | ✓      | ✓   | Self-Organizing Map |
| Neural Gas  | ✓      | ✓   | ランクベース競合学習 |
| GCS         | ✓      | ✓   | Growing Cell Structures |
| GCS (Kubota)| ✓      | -   | Kubotalab論文準拠版（最長エッジ方式） |
| HCL         | ✓      | ✓   | Hard Competitive Learning（勝者のみ更新） |
| LBG         | ✓      | ✓   | Linde-Buzo-Gray（バッチ学習） |
| Growing Grid| ✓      | ✓   | 自己成長グリッド |

### 実装バリアント

GNG, GCS, GNG-Tには2つのバリアントがあります：

1. **demogng版** (`model.py`)
   - demogng.de等の参照実装に基づく
   - ノード挿入時に「最大誤差の隣接ノード」を選択
   - 実用的で高速

2. **Kubotalab論文版** (`model_kubota.py`)
   - Kubota & Satomi (2008) に完全準拠
   - ノード挿入時に「最長エッジの隣接ノード」を選択
   - GNG-Tでは交差判定にγ式を使用

```python
# demogng版
from algorithms.gng.python.model import GrowingNeuralGas

# Kubotalab論文版
from algorithms.gng.python.model_kubota import GrowingNeuralGasKubota
```

## よく使うコマンド

```bash
# テスト実行
cd experiments/2d_visualization
python test_gng_triple_ring.py
python test_gng_tracking.py
python test_gngu_triple_ring.py
python test_gngu_tracking.py
python test_gngu2_triple_ring.py
python test_gngu2_tracking.py
python test_aisgng_triple_ring.py
python test_aisgng_tracking.py
python test_gngt_triple_ring.py
python test_gngt_tracking.py
python test_gngd_triple_ring.py
python test_gngd_tracking.py
python test_som_triple_ring.py
python test_som_tracking.py
python test_ng_triple_ring.py
python test_ng_tracking.py
python test_gcs_triple_ring.py
python test_gcs_tracking.py
python test_hcl_triple_ring.py
python test_lbg_triple_ring.py
python test_gg_triple_ring.py

# C++ビルド
cd experiments/2d_visualization/cpp
mkdir -p build && cd build
cmake .. && make

# C++テスト実行 + 可視化
./test_gng_triple_ring && python ../visualize_results.py
./test_gng_tracking && python ../visualize_results.py --tracking
```

## 派生アルゴリズム候補

今後実装予定の派生手法:

- **SOINN** - Self-Organizing Incremental Neural Network
- **E-SOINN** - Enhanced SOINN
- **A-GNG** - Adaptive GNG
- **ITM** - Instantaneous Topological Map

## 参照資料

- demogng.de の実装詳細: `references/notes/demogng_reference.md`
