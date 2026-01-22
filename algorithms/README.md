# Algorithms

GNGおよび派生アルゴリズムの実装一覧です。

## 実装状況

| アルゴリズム | Python | C++ | 説明 | 年 |
|-------------|:------:|:---:|------|:---:|
| [GNG](./gng/) | WIP | - | Growing Neural Gas | 1995 |
| GNG-U | - | - | GNG with Utility | 2006 |
| GNG-T | - | - | GNG with Topology Learning | - |
| SOINN | - | - | Self-Organizing Incremental NN | 2006 |
| E-SOINN | - | - | Enhanced SOINN | 2010 |
| A-SOINN | - | - | Adjusted SOINN | - |
| GWR | - | - | Grow When Required | 2002 |
| IGNG | - | - | Incremental GNG | - |

※ WIP = Work In Progress, `-` = 未実装

## ディレクトリ構成

各アルゴリズムは以下の構成に従います：

```
algorithm_name/
├── REFERENCE.md          # 参照元情報（必須）
├── python/
│   ├── __init__.py
│   ├── model.py          # メイン実装
│   └── tests/
│       └── test_model.py
└── cpp/
    ├── CMakeLists.txt
    ├── algorithm_name.hpp
    ├── algorithm_name.cpp
    └── tests/
        └── test_algorithm_name.cpp
```

## 新しいアルゴリズムの追加

1. `algorithms/` に新しいディレクトリを作成
2. `REFERENCE.md` に論文・元コードの情報を記載
3. Python/C++のいずれか（または両方）を実装
4. テストを追加
5. このREADMEの実装状況を更新

## REFERENCE.md テンプレート

```markdown
# [Algorithm Name]

## 論文
- Author(s). (Year). "Title"
- DOI/URL:

## 元コード
- Repository:
- License:
- 取得日:

## 概要
[アルゴリズムの簡単な説明]

## 主な特徴
- 特徴1
- 特徴2

## 変更履歴
- YYYY-MM-DD: 初期実装
```
