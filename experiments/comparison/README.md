# Utility-based GNG Comparison Experiments

GNG-UとGNG-U2の距離計算方式による追従性の違いを比較する実験。

## 背景

GNG-UとGNG-U2では距離計算方式が異なるため、utility_kパラメータの値を直接比較できない：

| アルゴリズム | 距離計算 | 論文のutility_k |
|------------|---------|----------------|
| GNG-U (Fritzke/demogng.de) | 二乗距離 ||x-w||² | 1.3 |
| GNG-U2 (Toda et al. 2016) | ユークリッド距離 ||x-w|| | 1000 |

## 実験モデル

公平な比較のため、距離計算方式を統一したバリアントを作成：

### Euclidean距離統一
- `gngu_euclidean.py`: GNG-U + ユークリッド距離 + κ間隔チェック
- `model.py` (GNG-U2): 元のユークリッド距離版

### Squared距離統一
- `model.py` (GNG-U): 元の二乗距離版
- `gngu2_squared.py`: GNG-U2 + 二乗距離
- `aisgng_squared.py`: AiS-GNG + 二乗距離

## 評価指標

追従テスト（移動リング）での評価：

1. **nodes_inside**: リング分布内のノード数（多いほど良い）
2. **nodes_outside**: リング分布外のノード数（少ないほど良い = 追従性が良い）
3. **n_removals**: Utility削除回数

## 使い方

```bash
cd experiments/comparison

# 距離計算方式の比較
python utility_tracking_comparison.py

# utility_kパラメータのスイープ
python utility_tracking_comparison.py --utility-k-sweep
```

## ディレクトリ構造

```
comparison/
├── models/
│   ├── __init__.py
│   ├── gngu_euclidean.py     # GNG-U (ユークリッド距離版)
│   ├── gngu2_squared.py      # GNG-U2 (二乗距離版)
│   └── aisgng_squared.py     # AiS-GNG (二乗距離版)
├── results/                   # 実験結果出力（git管理外）
│   ├── comparison_results.png
│   └── utility_k_sweep.png
├── utility_tracking_comparison.py  # 比較実験スクリプト
└── README.md
```

## 期待される結果

- 同じ距離計算方式で統一した場合、GNG-U2のκ間隔チェックの効果が明確になる
- 最適なutility_k値は距離計算方式に依存する
- 二乗距離ではk≈1〜5、ユークリッド距離ではk≈10〜100が適切な範囲と予想
