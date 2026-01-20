# GNG Collection

Growing Neural Gas (GNG) およびその派生アルゴリズムのコレクションリポジトリです。
各アルゴリズムのリファクタリング、2D/3Dデータへの適用テスト、新しいアイデアの実験を行います。

## 可視化サンプル

### GNG (Growing Neural Gas)

| シングルリング | トラッキング |
|:-------------:|:-----------:|
| ![GNG Single Ring](experiments/2d_visualization/samples/gng/python/single_ring_growth.gif) | ![GNG Tracking](experiments/2d_visualization/samples/gng/python/tracking.gif) |

### GNG-U (GNG with Utility)

非定常分布に対応したGNG。不要なノードをユーティリティ基準で除去します。

| シングルリング | トラッキング |
|:-------------:|:-----------:|
| ![GNG-U Single Ring](experiments/2d_visualization/samples/gng_u/python/single_ring_growth.gif) | ![GNG-U Tracking](experiments/2d_visualization/samples/gng_u/python/tracking.gif) |

## 対応言語

- Python
- C++

## ディレクトリ構成

```
gng_collection/
├── algorithms/          # 各アルゴリズム実装
│   ├── _template/       # 新アルゴリズム用テンプレート
│   ├── gng/             # 標準GNG
│   └── gng_u/           # GNG-U (Utility)
├── experiments/         # 実験・アイデア試行
│   └── 2d_visualization/
│       ├── _templates/  # テストテンプレート
│       └── samples/     # 出力サンプル
├── data/                # サンプルデータ
├── references/          # 参照資料・元コード
│   ├── notes/           # アルゴリズムノート
│   └── original_code/   # リファレンス実装
└── python/              # Python共通設定・コア
```

## アルゴリズム一覧

| アルゴリズム | Python | C++ | 説明 |
|-------------|:------:|:---:|------|
| GNG         | ✓      | ✓   | 標準 Growing Neural Gas |
| GNG-U       | ✓      | -   | Utility付きGNG（非定常分布対応） |
| SOINN       | -      | -   | Self-Organizing Incremental Neural Network |
| E-SOINN     | -      | -   | Enhanced SOINN |

## セットアップ

### Python

```bash
# 依存パッケージ
pip install numpy matplotlib pillow
```

### C++

```bash
cd experiments/2d_visualization/cpp
mkdir build && cd build
cmake ..
make
```

## 使い方

### Python

```python
import numpy as np
from algorithms.gng.python.model import GrowingNeuralGas, GNGParams

# データ準備
X = np.random.rand(1000, 2)

# GNGの作成と学習
params = GNGParams(max_nodes=50, lambda_=100)
gng = GrowingNeuralGas(n_dim=2, params=params)
gng.train(X, n_iterations=5000)

# グラフ構造の取得
nodes, edges = gng.get_graph()
```

### GNG-U (非定常分布対応)

```python
from algorithms.gng_u.python.model import GrowingNeuralGasU, GNGUParams

params = GNGUParams(
    max_nodes=50,
    utility_k=1.3,  # ユーティリティ閾値
)
gng_u = GrowingNeuralGasU(n_dim=2, params=params)

# オンライン学習
for sample in streaming_data:
    gng_u.partial_fit(sample)
```

## テストの実行

```bash
cd experiments/2d_visualization

# GNGテスト
python test_gng_single_ring.py
python test_gng_tracking.py

# GNG-Uテスト
python test_gngu_single_ring.py
python test_gngu_tracking.py
```

## 新しいアルゴリズムの追加

1. `algorithms/_template/` をコピー
2. `experiments/2d_visualization/_templates/` のテストテンプレートを使用
3. テスト実行後、出力を `samples/[algorithm]/python/` に保存

詳細は [CLAUDE.md](CLAUDE.md) を参照してください。

## 参照元について

各アルゴリズムの `REFERENCE.md` に論文情報や元コードの出典を記載しています。
元コードのスナップショットは `references/original_code/` に保存されています。

- **GNG**: Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
- **GNG-U**: Fritzke, B. (1997). "Some Competitive Learning Methods"

## License

MIT License
