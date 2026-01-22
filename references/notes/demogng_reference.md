# DemoGNG.de 実装リファレンス

## 概要

DemoGNGは Dr. Bernd Fritzke によって作成された競合学習手法のシミュレータです。
本ドキュメントは demogng.de のJavaScript実装（rhaschke/demogng）を調査した結果をまとめたものです。

- **公式サイト**: https://www.demogng.de/
- **GitHubソース**: https://github.com/rhaschke/demogng
- **ソースファイル**: js/vbnn.js

## 実装されているアルゴリズム

| アルゴリズム | 関数名 | 説明 |
|-------------|--------|------|
| GNG | `adaptGNG()` | Growing Neural Gas |
| GNG-U | `adaptGNG()` + utility | GNG with Utility |
| NG | `adaptNG()` | Neural Gas |
| NG-CHL | `adaptNG()` + CHL | Neural Gas with Competitive Hebbian Learning |
| SOM | `adaptSOM()` | Self-Organizing Map |
| GG | `adaptGG()` | Growing Grid |
| HCL | `adaptHCL()` | Hard Competitive Learning |
| CHL | `adaptCHL()` | Competitive Hebbian Learning |
| LBG | `adaptLBG()` | Linde-Buzo-Gray |
| LBG-U | `adaptLBG()` + utility | LBG with Utility |
| ITM | `adaptITM()` | Instantaneous Topological Map |

---

## GNG (Growing Neural Gas)

### アルゴリズムステップ

1. **勝者・次勝者の検索**: 入力信号との二乗距離が最小の2ノードを特定
2. **エラー更新**: `bmu.error += mindist` （二乗距離を加算）
3. **勝者移動**: `weight += eps_b * (signal - weight)`
4. **近傍移動**: `weight += eps_n * (signal - weight)`
5. **エッジ老化**:
   - 勝者の全エッジの年齢を+1
   - 勝者-次勝者間のエッジを年齢1に設定（作成またはリセット）
   - max_ageを超えたエッジを削除
6. **孤立ノード削除**: エッジのないノードを削除
7. **エラー減衰**: `error *= (1 - beta)` を全ノードに適用
8. **ノード挿入**: lambda回ごとに最大エラーノードの近傍に新ノードを挿入

### パラメータ

| パラメータ | 説明 | 典型値 |
|-----------|------|--------|
| `eps_b` | 勝者学習率 | 0.05-0.2 |
| `eps_n` | 近傍学習率 | 0.006-0.02 |
| `max_age` | エッジ最大年齢 | 50-200 |
| `lambda` | ノード挿入間隔 | 100-500 |
| `alpha` | 挿入時エラー減衰 | 0.5 |
| `beta` | グローバルエラー減衰 | 0.0005-0.005 |

---

## GNG-U (GNG with Utility)

### Utilityの計算

```javascript
// 初期化
node.utility = 0;

// 更新（各学習ステップ）
bmu.utility += min2dist - mindist;  // 距離の差（二乗ではない）

// 減衰
node.utility *= (1 - beta);
```

### ノード削除基準

```javascript
// 最小utility × utilfac < 最大error の場合、最小utilityノードを削除
if (luu.utility * glob.gng_utilfac < meu.error) {
    removeNode(luu);
}
```

### 重要な違い

- **Utility初期値**: 0（エラーは1で初期化）
- **Utility更新**: `min2dist - mindist`（距離の差、二乗距離ではない）
- **削除基準**: `min_utility * k < max_error`

---

## SOM (Self-Organizing Map)

### グリッド距離

demogng.deでは**マンハッタン距離**を使用:

```javascript
d = Math.abs(x - bmu.x) + Math.abs(y - bmu.y);
```

### 近傍関数

Gaussian関数:

```javascript
var hrs = Math.exp(-d * d / (2 * sigma * sigma));
```

### パラメータ減衰

指数減衰:

```javascript
// 学習率
var epsilon = eps_i * Math.pow(eps_f / eps_i, t / t_max);

// 近傍半径
var sigma = sigma_i * Math.pow(sigma_f / sigma_i, t / t_max);
```

---

## Neural Gas

### ランクベース近傍関数

```javascript
// ノードを距離でソート
units.sort(function(a, b) { return a.dist - b.dist; });

// k = ランク（0が最近傍）
var h = Math.exp(-k / lambda);
```

### パラメータ減衰

```javascript
var lambda = lambda_i * Math.pow(lambda_f / lambda_i, t / t_max);
var epsilon = eps_i * Math.pow(eps_f / eps_i, t / t_max);
```

### CHL (Competitive Hebbian Learning)

```javascript
// 勝者のエッジを老化
for (edge in bmu.edges) {
    edge.age += 1;
}

// 勝者-次勝者間のエッジを作成/リセット
edge(bmu, bmu2).age = 0;

// 古いエッジを削除
removeEdgesOlderThan(max_age);
```

---

## HCL (Hard Competitive Learning)

勝者のみを更新する最もシンプルな競合学習:

```javascript
// 勝者検索
var bmu = findBMU(signal);

// 勝者のみ更新
bmu.weight += epsilon * (signal - bmu.weight);
```

---

## LBG (Linde-Buzo-Gray)

バッチ学習ベースのベクトル量子化:

```javascript
// 各データポイントを最近傍ノードに割り当て
for (data in dataset) {
    bmu = findBMU(data);
    bmu.assigned.push(data);
}

// 各ノードを割り当てられたデータの重心に移動
for (node in nodes) {
    if (node.assigned.length > 0) {
        node.weight = centroid(node.assigned);
    }
}
```

### LBG-U (LBG with Utility)

Utilityに基づいてノードを削除/追加:

- 低Utilityノードを削除
- 高エラー領域に新ノードを追加

---

## Growing Grid

自己成長するグリッド構造:

- 初期: 2×2グリッド
- 成長: 最大エラーノードの周辺に行/列を追加
- グリッド構造を維持

---

## 可視化機能

### Voronoi図

```javascript
var diagram = voronoi.compute(sites, bbox);
```

各ノードの担当領域を可視化。

### ノード軌跡

ノードの移動履歴を保存し、軌跡として描画可能。

---

## 参考文献

- Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies" - NIPS
- Fritzke, B. (1997). "Some Competitive Learning Methods" - Technical Report
- Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps"
- Martinetz, T. and Schulten, K. (1991). "A Neural-Gas Network Learns Topologies"

## 調査日

2026-01-20
