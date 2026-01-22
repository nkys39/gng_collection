# Data

テスト・実験用のサンプルデータを格納するディレクトリです。

## ディレクトリ構成

```
data/
├── 2d/     # 2Dサンプルデータ
└── 3d/     # 3Dサンプルデータ（点群など）
```

## データ形式

### 2Dデータ
- CSV形式: `x,y` の2列
- NumPy形式: `.npy` ファイル

### 3Dデータ
- CSV形式: `x,y,z` の3列
- NumPy形式: `.npy` ファイル
- PLY形式: 点群データ
- PCD形式: Point Cloud Data

## 注意事項

- 大きなファイル（>10MB）は Git LFS を使用するか `.gitignore` に追加
- 外部データセットはダウンロードスクリプトを用意

## 2Dデータ生成ツール

### 形状画像の生成

```bash
cd data/2d/shapes
python generate_shape.py --shape double_ring --output double_ring.png
```

### 画像から点群をサンプリング

```bash
cd data/2d
python sampler.py shapes/double_ring.png -n 2000 -o points.npy
```

オプション:
- `-n`: サンプル数（デフォルト: 1000）
- `-o`: 出力ファイル（.npy または .csv）
- `--seed`: 乱数シード
- `--color`: サンプリング対象の色（hex、デフォルト: 87CEEB）

### GNGテスト（GIF出力付き）

```bash
cd experiments/2d_visualization
python test_gng_double_ring.py --iterations 5000 --frames 100
```

出力:
- `gng_growth.gif` - 学習過程のアニメーション
- `gng_final.png` - 最終結果

## サンプルデータの生成（コード例）

```python
import numpy as np

# 2D: 3クラスタ
np.random.seed(42)
c1 = np.random.randn(100, 2) * 0.5 + [0, 0]
c2 = np.random.randn(100, 2) * 0.5 + [3, 0]
c3 = np.random.randn(100, 2) * 0.5 + [1.5, 2.5]
data_2d = np.vstack([c1, c2, c3])
np.save('data/2d/clusters_3.npy', data_2d)
```
