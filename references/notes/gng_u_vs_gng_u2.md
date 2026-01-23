# GNG-U vs GNG-U2: 比較ノート

## 概要

GNG-U (Growing Neural Gas with Utility) には主に2つのバリアントがあります：

1. **GNG-U (Fritzke 1997/1999)**
   - 原著: Fritzke, B. (1999). "Be Busy and Unique — or Be History—The Utility Criterion"
   - GNGにUtility基準を追加し、非定常分布に対応

2. **GNG-U2 (Toda et al. 2016)**
   - 原著: Toda, Y., et al. (2016). "Real-time 3D point cloud segmentation using Growing Neural Gas with Utility"
   - IEEE ICRA 2016
   - GNG-Uを改良し、より頻繁なUtilityチェックでリアルタイム処理に対応

## 主な違い

### 1. Utility基準のチェックタイミング

| 項目 | GNG-U | GNG-U2 |
|-----|-------|--------|
| チェック間隔 | λ間隔（ノード挿入時） | κ間隔（独立した頻度） |
| 典型値 | λ=100〜300 | κ=10 |
| チェック頻度 | 低い | 高い（10〜30倍） |

**GNG-U:**
```python
# ノード挿入時にのみチェック
if iteration % lambda_ == 0:
    insert_node()
    check_utility_criterion()  # λ間隔でチェック
```

**GNG-U2:**
```python
# ノード挿入とは独立してチェック
if iteration % kappa == 0:
    check_utility_criterion()  # κ間隔でチェック（より頻繁）

if iteration % lambda_ == 0:
    insert_node()
```

### 2. 距離の計算方法

| 項目 | GNG-U | GNG-U2 |
|-----|-------|--------|
| 誤差計算 | 二乗距離 ||x - w||² | ユークリッド距離 ||x - w|| |
| Utility計算 | 二乗距離差 | ユークリッド距離差 |

**GNG-U (demogng.de準拠):**
```python
dist_sq = np.sum((x - w) ** 2)
error += dist_sq  # 二乗距離
utility += dist2_sq - dist1_sq  # 二乗距離の差
```

**GNG-U2:**
```python
dist = np.sqrt(np.sum((x - w) ** 2))
error += dist  # ユークリッド距離
utility += dist2 - dist1  # ユークリッド距離の差
```

### 3. 減衰率パラメータ

| 項目 | GNG-U | GNG-U2 |
|-----|-------|--------|
| 誤差減衰率 | β | β |
| Utility減衰率 | β（共通） | χ（別パラメータ） |

**GNG-U:**
```python
error -= beta * error
utility -= beta * utility  # 同じβを使用
```

**GNG-U2:**
```python
error -= beta * error
utility -= chi * utility  # 別のχを使用可能
```

## パラメータ比較

### GNG-U (demogng.de)

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| λ | 100 | ノード挿入間隔 |
| η1 (eps_b) | 0.08 | 勝者学習率 |
| η2 (eps_n) | 0.008 | 近傍学習率 |
| AgeMax | 100 | 最大エッジ年齢 |
| α | 0.5 | 分割時誤差減衰 |
| β | 0.005 | 誤差/Utility減衰率 |
| k | 1.3 | Utility閾値 |

### GNG-U2 (Toda et al. 2016)

| パラメータ | 論文値 | 説明 |
|-----------|--------|------|
| λ | 300 | ノード挿入間隔 |
| κ | 10 | Utilityチェック間隔 |
| η1 (eps_b) | 0.08 | 勝者学習率 |
| η2 (eps_n) | 0.008 | 近傍学習率 |
| g_max (max_age) | 88 | 最大エッジ年齢 |
| α | 0.5 | 分割時誤差減衰 |
| β | 0.005 | 誤差減衰率 |
| χ | 0.005 | Utility減衰率 |
| k | 1000 | Utility閾値 |

## 用途の違い

### GNG-U
- 一般的な非定常分布追跡
- 概念的にシンプル
- demogng.de等のリファレンス実装で広く使用

### GNG-U2
- リアルタイム3Dポイントクラウド処理
- 高速な分布変化への対応
- AiS-GNGのベースアルゴリズム

## AiS-GNGとの関係

AiS-GNG (Add-if-Silent Rule-Based GNG) は **GNG-U2をベースに** 以下を追加しています：

1. **Add-if-Silentルール**: 入力が両勝者から適切な距離にある場合、直接ノードとして追加
2. **κ間隔でのUtilityチェック**: GNG-U2から継承
3. **ユークリッド距離**: GNG-U2から継承

```
GNG (Fritzke 1995)
    ↓
GNG-U (Fritzke 1997/1999) - Utility基準追加
    ↓
GNG-U2 (Toda et al. 2016) - κ間隔チェック、ユークリッド距離
    ↓
AiS-GNG (Shoji et al. 2023) - Add-if-Silentルール追加
```

## 実装の選択指針

| 要件 | 推奨アルゴリズム |
|-----|----------------|
| シンプルな非定常分布追跡 | GNG-U |
| 高速な分布変化への対応 | GNG-U2 |
| 高密度位相構造の高速生成 | AiS-GNG |
| リアルタイム3D処理 | GNG-U2 or AiS-GNG |

## 参考文献

1. Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
2. Fritzke, B. (1997). "Some Competitive Learning Methods"
3. Fritzke, B. (1999). "Be Busy and Unique — or Be History—The Utility Criterion"
4. Toda, Y., et al. (2016). "Real-time 3D point cloud segmentation using Growing Neural Gas with Utility"
5. Shoji, M., et al. (2023). "Add-if-Silent Rule-Based Growing Neural Gas for High-Density Topological Structure of Unknown Objects"
