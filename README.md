# GNG Collection

Growing Neural Gas (GNG) およびその派生アルゴリズムのコレクションリポジトリです。
各アルゴリズムのリファクタリング、2D/3Dデータへの適用テスト、新しいアイデアの実験を行います。

## 対応言語

- Python
- C++

## ディレクトリ構成

```
gng_collection/
├── python/              # Python共通設定・コア
├── cpp/                 # C++共通設定・コア
├── algorithms/          # 各アルゴリズム実装
│   ├── gng/             # 標準GNG
│   ├── gng_u/           # GNG-U (Utility)
│   └── ...
├── experiments/         # 実験・アイデア試行
├── notebooks/           # Jupyter notebooks
├── data/                # サンプルデータ
└── references/          # 参照資料・元コード
```

## アルゴリズム一覧

| アルゴリズム | Python | C++ | 説明 |
|-------------|:------:|:---:|------|
| GNG         | -      | -   | 標準 Growing Neural Gas |
| GNG-U       | -      | -   | Utility付きGNG |
| SOINN       | -      | -   | Self-Organizing Incremental Neural Network |
| E-SOINN     | -      | -   | Enhanced SOINN |

※ 実装状況は `algorithms/README.md` を参照

## セットアップ

### Python

```bash
cd python
pip install -e .
```

### C++

```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

## 使い方

### Python

```python
from gng.algorithms.gng import GNG

model = GNG()
model.fit(data)
```

### C++

```cpp
#include <gng/algorithms/gng.hpp>

gng::GNG model;
model.fit(data);
```

## 実験

`experiments/` ディレクトリで自由に実験できます。

- `2d_visualization/` - 2Dデータでの可視化実験
- `3d_pointcloud/` - 3D点群への適用実験
- `sandbox/` - 自由な試行錯誤

## 参照元について

各アルゴリズムの `REFERENCE.md` に論文情報や元コードの出典を記載しています。
元コードのスナップショットは `references/original_code/` に保存されています。

## License

MIT License
