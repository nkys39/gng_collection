# Experiments

実験・アイデア試行用ディレクトリです。

## ディレクトリ構成

```
experiments/
├── 2d_visualization/    # 2Dデータでの可視化実験
├── 3d_pointcloud/       # 3D点群への適用実験
└── sandbox/             # 自由な試行錯誤
```

## 使い方

### 新しい実験の追加

1. 適切なサブディレクトリに実験用ファイルを作成
2. 実験内容を説明するREADMEを追加（推奨）
3. 再現可能なように乱数シードを固定

### 命名規則

```
YYYYMMDD_experiment_name/
├── README.md
├── experiment.py (or .cpp)
└── results/
```

## 注意事項

- 大きなデータファイルは `.gitignore` に追加
- 成果物は `results/` サブディレクトリに保存
- 実験が成功したら、`algorithms/` への統合を検討
