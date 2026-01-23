# 実装バリアント比較: demogng版 vs Kubotalab版

## 概要

GNG, GCS, GNG-Tには2つの実装バリアントがあります。両者の主な違いは**ノード挿入時の隣接ノード選択方法**です。

| バリアント | ファイル | ノード挿入時のf選択 | 交差判定（GNG-T） |
|-----------|---------|-------------------|------------------|
| demogng準拠 | `model.py` | 最大エラー近傍 | CCW法 |
| Kubotalab準拠 | `model_kubota.py` | 最長エッジ近傍 | γ式 |

## 詳細な違い

### 1. ノード挿入時のf選択（Step 8.ii / Step 5.ii）

GNGのノード挿入ステップでは、最大誤差ノード `q` の隣接ノードから `f` を選び、`q-f` 間に新ノード `r` を挿入します。

#### demogng版
```python
# 最大エラーを持つ隣接ノードを選択
f_id = max(neighbors, key=lambda n: self.nodes[n].error)
```

**根拠**: demogng.de等の参照実装に基づく。誤差が大きい領域を優先的に細分化する。

#### Kubotalab版
```python
# 最長エッジで接続された隣接ノードを選択
f_id = max(neighbors, key=lambda n: np.sum((self.nodes[q_id].weight - self.nodes[n].weight) ** 2))
```

**根拠**: Kubota & Satomi (2008) の論文記述に準拠。原文では「qに直接連結されたユニットのうち，qと最も長いエッジで結ばれたユニットf」と記載。

### 2. 交差判定（GNG-Tのみ）

GNG-Tの三角形分割では、新しいエッジが既存エッジと交差するかを判定します。

#### demogng版: CCW法
```python
def ccw(A, B, C):
    """Counter-Clockwise test"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def edges_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
```

#### Kubotalab版: γ式
```python
def edges_intersect_gamma(B, D, C, E):
    """論文 Section 2.5.2 の γ式"""
    gamma1 = (C[0] - E[0]) * (D[1] - C[1]) + (C[1] - E[1]) * (C[0] - D[0])
    gamma2 = (C[0] - E[0]) * (B[1] - C[1]) + (C[1] - E[1]) * (C[0] - B[0])
    gamma3 = (B[0] - D[0]) * (C[1] - B[1]) + (B[1] - D[1]) * (B[0] - C[0])
    gamma4 = (B[0] - D[0]) * (E[1] - B[1]) + (B[1] - D[1]) * (B[0] - E[0])
    return (gamma1 * gamma2 <= 0) and (gamma3 * gamma4 <= 0)
```

**根拠**: 論文 Section 2.5.2「交点探索」に記載された式をそのまま実装。

## 特性比較

### demogng版

| 項目 | 評価 |
|------|------|
| **設計思想** | 誤差ベース（適応的） |
| **ノード配置** | データ密度に応じて不均一 |
| **近似精度** | 高密度領域で高い |
| **メッシュ品質** | 領域により不均一 |
| **実績** | 広く使われている参照実装 |

**適したケース**:
- データ密度に偏りがある場合
- 近似精度を重視する場合
- 既存実装との互換性が必要な場合

### Kubotalab版

| 項目 | 評価 |
|------|------|
| **設計思想** | 幾何学ベース（均等） |
| **ノード配置** | より均等な間隔 |
| **近似精度** | 全体的に均一 |
| **メッシュ品質** | 幾何学的に良好 |
| **実績** | 論文の厳密な再現 |

**適したケース**:
- 均等なメッシュ構造が必要な場合
- 三角形メッシュの品質が重要な場合（GNG-T, GCS）
- 論文との再現性・比較実験
- CAD/メッシュ生成への応用

## 使い分けガイド

```
目的は何ですか？
│
├─ 論文の再現・比較実験
│   └─ → Kubotalab版
│
├─ メッシュ品質が重要（CAD, 3Dモデリング）
│   └─ → Kubotalab版
│
├─ データ密度が不均一で高精度近似が必要
│   └─ → demogng版
│
├─ 既存システムとの互換性
│   └─ → demogng版
│
└─ 特に要件なし / 一般的な用途
    └─ → どちらでも可（demogng版が標準）
```

## 視覚的な違い

実際のところ、多くのケースで視覚的な差は小さいです。トリプルリングのような均一密度データでは、両者の結果はほぼ同等になります。

差が顕著になるケース:
- 密度が大きく異なる領域が混在するデータ
- 細長い形状や複雑なトポロジー
- 三角形メッシュの品質が重要な場合

## 参考文献

- **demogng.de**: https://www.demogng.de/
  - JavaScriptによるGNG/GNG-T実装
  - 多くの実装の参照元

- **Kubotalab論文**:
  - 久保田直行, 里見将志 (2008). "自己増殖型ニューラルネットワークと教師無し分類学習"
  - GNG-Tの詳細なアルゴリズム記述を含む

## 実装ファイル

| アルゴリズム | demogng版 | Kubotalab版 |
|------------|-----------|-------------|
| GNG | `algorithms/gng/python/model.py` | `algorithms/gng/python/model_kubota.py` |
| GCS | `algorithms/gcs/python/model.py` | `algorithms/gcs/python/model_kubota.py` |
| GNG-T | `algorithms/gng_t/python/model.py` | `algorithms/gng_t/python/model_kubota.py` |

C++実装:
| アルゴリズム | demogng版 | Kubotalab版 |
|------------|-----------|-------------|
| GNG | `algorithms/gng/cpp/gng.hpp` | `algorithms/gng/cpp/gng_kubota.hpp` |
| GCS | `algorithms/gcs/cpp/gcs.hpp` | `algorithms/gcs/cpp/gcs_kubota.hpp` |
| GNG-T | `algorithms/gng_t/cpp/gng_t.hpp` | `algorithms/gng_t/cpp/gng_t_kubota.hpp` |
