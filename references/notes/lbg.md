# Linde-Buzo-Gray (LBG) Algorithm

## 概要

Linde-Buzo-Gray (LBG) アルゴリズムは、ベクトル量子化のためのバッチ学習アルゴリズムである。
K-means クラスタリングと本質的に同じアルゴリズムで、Lloyd アルゴリズムとも呼ばれる。

## 元論文

- Linde, Y., Buzo, A., & Gray, R. (1980). "An Algorithm for Vector Quantizer Design"
  IEEE Transactions on Communications, 28(1), 84-95.

## アルゴリズム

1. N 個のコードブックベクトルをランダムに初期化
2. 収束するまで繰り返し:
   - **割り当てステップ**: 各データ点を最近傍のコードブックに割り当て
   - **更新ステップ**: 各コードブックを割り当てられた点の重心に移動
3. 歪み（量子化誤差）の変化が閾値以下になったら終了

## パラメータ

| パラメータ | 説明 | 典型値 |
|-----------|------|--------|
| n_nodes | コードブック数（固定） | 50-100 |
| max_epochs | 最大エポック数 | 100 |
| convergence_threshold | 収束判定閾値 | 1e-6 |
| use_utility | utility管理を使用するか | false |
| utility_threshold | utility閾値（LBG-U用） | 0.01 |

## demogng.de 実装

demogng.de では LBG は以下の特徴:

```javascript
// バッチ更新：全データを一度に処理
for (var i = 0; i < signals.length; i++) {
    var bmu = findBMU(signals[i]);
    bmu.sum.add(signals[i]);
    bmu.count++;
}
// 重心に移動
for (var j = 0; j < nodes.length; j++) {
    if (nodes[j].count > 0) {
        nodes[j].pos = nodes[j].sum.scale(1.0 / nodes[j].count);
    }
}
```

## バッチ学習 vs オンライン学習

### バッチ学習（LBG）
- 全データを一度に処理
- 収束が保証される
- メモリ使用量が大きい
- 非定常分布に不向き

### オンライン学習（GNG, SOM等）
- データを1つずつ処理
- 逐次的に適応可能
- メモリ効率が良い
- 非定常分布に対応可能

## LBG-U（Utility付きLBG）

demogng.de には LBG-U も実装されている:

- 割り当て数（utility）が低いノードを検出
- utility閾値以下のノードを高エラー領域に再配置
- dead units 問題を緩和

```python
# 本実装での LBG-U
if node.utility < threshold:
    # 高エラーノード近くに再配置
    node.position = high_error_node.position + noise
```

## 特徴

### 利点
- **収束保証**: 歪みが単調減少
- **最適性**: 局所最適解に収束
- **シンプル**: 実装が容易

### 欠点
- **バッチ処理必須**: 全データをメモリに保持
- **初期値依存**: 初期配置で結果が変わる
- **非定常分布に不向き**: オンライン適応が困難
- **dead units**: 一部のコードブックが使われなくなる

## 使用場面

- 静的なデータセットのベクトル量子化
- 画像圧縮のコードブック設計
- 音声認識の特徴量クラスタリング
- データの初期クラスタリング

## トラッキングへの適用

LBG は本質的にバッチアルゴリズムだが、以下の方法でトラッキングに使用可能:

1. **スライディングウィンドウ**: 最新のN個のデータでバッチ学習
2. **定期的再学習**: 一定間隔で全体を再学習
3. **オンライン近似**: 小さな学習率で逐次更新（本来の LBG ではない）

本実装の `partial_fit` メソッドは3番目のアプローチを採用。
