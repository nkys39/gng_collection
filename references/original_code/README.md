# Original Code Reference

参照元コードのスナップショットを保存するディレクトリです。

## 収録コード一覧

| ディレクトリ | アルゴリズム | 言語 | 作者/出典 | 主な特徴 |
|-------------|------------|------|----------|---------|
| `watanabe_gng` | 標準GNG | C++/Eigen | watanabe | テンプレートベース、基本実装 |
| `azhar_ddgng` | DD-GNG | C++ | Azhar (Kubota研) | デプスセンサー用、法線計算、GNG-U |
| `corner_dbl_gng` | DBL-GNG | Python | Corner | バッチ学習、FCM統合 |
| `corner_td_ml_gng` | TD-ML-GNG | Python | kubota/Corner | 階層的GNG、Add-if-Silent |
| `miya_gng` | HNSW-GNG | C++/Eigen | miya | 高速近傍探索、分散共分散行列 |
| `toda_gngdt` | GNG-DT | C/ROS2 | Toda (Kubota研) | LiDAR用、地形分類 |
| `corner_dataset` | - | - | - | テスト用データセット |

## アルゴリズム分類

### 基本GNG系
- **watanabe_gng**: 標準GNG（Fritzkeオリジナル準拠）
- **corner_dbl_gng/standard_gng.py**: Python版標準GNG

### GNG-U (Utility) 系
- **azhar_ddgng**: GNG-U + 法線計算 + サーフェス分類
- **toda_gngdt**: GNG-U + マルチエッジ

### バッチ学習系
- **corner_dbl_gng/DBL_GNG.py**: Distributed Batch Learning GNG
- **corner_dbl_gng/FCM_BL_GNG.py**: Fuzzy C-Means + Batch Learning GNG

### 階層的GNG系
- **corner_td_ml_gng**: Top-Down Multi-Layer GNG

### 高速化系
- **miya_gng**: HNSW (Hierarchical Navigable Small World) 統合

## 機能マトリクス

| 機能 | watanabe | azhar | dbl_gng | td_ml_gng | miya | toda |
|-----|:--------:|:-----:|:-------:|:---------:|:----:|:----:|
| 標準GNG | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GNG-U | - | ✓ | - | - | - | ✓ |
| バッチ学習 | - | - | ✓ | - | - | - |
| 階層構造 | - | - | - | ✓ | - | - |
| Add-if-Silent | - | - | - | ✓ | ✓ | - |
| HNSW高速化 | - | - | - | - | ✓ | - |
| 三角形分割 | - | ✓ | - | - | - | - |
| 法線計算 | - | ✓ | - | - | - | ✓ |
| FCM学習 | - | ✓ | ✓ | - | - | - |
| 分散共分散 | - | - | - | - | ✓ | - |
| ROS2対応 | - | - | - | - | - | ✓ |

## 参照ガイド

### 新しいアルゴリズムを実装したい場合

| 目的 | 参照すべきコード |
|-----|----------------|
| 基本的なGNG実装 | `watanabe_gng` |
| バッチ学習GNG | `corner_dbl_gng/DBL_GNG.py` |
| 階層的GNG | `corner_td_ml_gng` |
| 大規模データ対応 | `miya_gng` (HNSW) |
| リアルタイム3D点群 | `azhar_ddgng`, `toda_gngdt` |
| 法線/サーフェス分類 | `azhar_ddgng` |

### algorithms/ との対応

| original_code | algorithms/ |
|--------------|-------------|
| watanabe_gng | gng/ |
| azhar_ddgng (GNG-U部分) | gng_u/, gng_u2/ |
| toda_gngdt (GNG-T部分) | gng_t/ |

## ディレクトリ構成

```
original_code/
├── README.md                    # このファイル
├── watanabe_gng/               # 標準GNG (C++)
│   ├── README.md
│   └── original/
├── azhar_ddgng/                # DD-GNG (C++)
│   ├── README.md
│   └── DepthSensor_Buggy/
├── corner_dbl_gng/             # バッチ学習GNG (Python)
│   ├── README.md
│   ├── DBL_GNG.py
│   ├── FCM_BL_GNG.py
│   └── standard_gng.py
├── corner_td_ml_gng/           # 階層的GNG (Python)
│   ├── README.md
│   └── Final_TD_ML_GNG.py
├── miya_gng/                   # HNSW-GNG (C++)
│   ├── README.md
│   ├── GNG.hpp
│   └── main.cpp
├── toda_gngdt/                 # ROS2 GNG (C)
│   ├── README.md
│   └── gng_livox/
└── corner_dataset/             # テストデータセット
    ├── README.md
    └── dataset/
```

## 注意事項

- ライセンスが許可する範囲で使用すること
- 元コードの著作権表示を保持すること
- 各ディレクトリのREADME.mdを参照して出典を確認すること
