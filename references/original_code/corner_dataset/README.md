# Corner Dataset

## 説明

GNGアルゴリズムのテスト用データセットコレクションです。
クラスタリング評価に広く使用される標準的なデータセットを含みます。

## データセット一覧

| ファイル | サンプル数 | クラスタ数 | 説明 |
|---------|-----------|-----------|------|
| Aggregation.txt | 788 | 7 | 様々な形状のクラスタ |
| D31.txt | 3100 | 31 | 31個の均一クラスタ |
| H72D.txt | 14400 | 72 | 72個の高密度クラスタ |
| R15.txt | 600 | 15 | 15個のリング状クラスタ |
| pathbased.txt | 300 | 3 | パス状のクラスタ |
| spiral.txt | 312 | 3 | 螺旋状のクラスタ |

## データ形式

各ファイルはスペース区切りのテキスト形式:

```
x y [label]
```

- `x`: X座標
- `y`: Y座標
- `label`: クラスタラベル（オプション）

## 使用例

### Python

```python
import numpy as np

# データ読み込み
data = []
with open("dataset/Aggregation.txt", "r") as f:
    for line in f:
        data.append(np.array(line.split(), dtype=np.float32))
data = np.vstack(data)

# 座標のみ抽出（ラベルを除く）
X = data[:, :2]

# ラベル抽出（ある場合）
if data.shape[1] > 2:
    labels = data[:, 2].astype(int)
```

### C++

```cpp
#include <fstream>
#include <vector>
#include <Eigen/Dense>

std::vector<Eigen::Vector2d> loadData(const std::string& filename) {
    std::vector<Eigen::Vector2d> data;
    std::ifstream file(filename);
    double x, y;
    while (file >> x >> y) {
        data.push_back(Eigen::Vector2d(x, y));
    }
    return data;
}
```

## 各データセットの特徴

### Aggregation

- 7つの異なる形状のクラスタ
- 密度の異なる領域を含む
- GNGの形状適応能力をテスト

### D31

- 31個の均一なクラスタ
- クラスタ間の距離がほぼ等しい
- 多数クラスタの分離能力をテスト

### H72D

- 72個の高密度クラスタ
- 大規模データセット（14400点）
- スケーラビリティのテスト

### R15

- 15個のリング状クラスタ
- 入れ子構造を含む
- 非凸形状の処理能力をテスト

### pathbased

- 3つのパス状クラスタ
- 細長い形状
- 経路状構造の学習をテスト

### spiral

- 3つの螺旋状クラスタ
- 交差しない螺旋
- 複雑な形状の学習をテスト

## 出典

これらのデータセットは以下の研究で使用されています:

- Clustering datasets: http://cs.joensuu.fi/sipu/datasets/
- Gionis, A., Mannila, H., & Tsaparas, P. (2007). Clustering aggregation.

## 可視化例

```python
import matplotlib.pyplot as plt

data = np.loadtxt("dataset/Aggregation.txt")
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='tab10', s=5)
plt.title("Aggregation Dataset")
plt.show()
```
