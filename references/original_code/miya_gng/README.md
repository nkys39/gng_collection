# HNSW-GNG (Hierarchical Navigable Small World GNG)

## 出典

- Author: miya
- 取得日: 2025-01-27

## ライセンス

(要確認)

## 説明

HNSW-GNG は、HNSW (Hierarchical Navigable Small World) による高速近傍探索を統合したGNG実装です。
大規模データセットでの勝者ノード探索を高速化し、分散共分散行列による適応的な距離計算を行います。

## ファイル構成

```
miya_gng/
├── README.md
├── GNG.hpp     # ヘッダーオンリー実装
└── main.cpp    # 使用例
```

## アルゴリズム概要

### 主要機能

1. **HNSW統合**: 階層的近傍グラフによる高速な勝者ノード探索
2. **分散共分散行列**: 各ノードに分散共分散行列を保持
3. **純度 (Purity)**: ノードの信頼性を表す指標
4. **Add-if-Silent**: 距離が大きい場合にノード追加

## クラス構造

### Nodeクラス

```cpp
struct Node {
    Eigen::VectorXd reference_vector;  // 参照ベクトル
    double purity;                      // ノードの純度
    Eigen::MatrixXd vcm;               // 分散共分散行列
    double squared_radius;              // 分散の指標 (trace)
    Eigen::MatrixXd inv_vcm;           // 分散共分散行列の逆行列

    std::vector<std::pair<Node*, Edge*>> neighbors;  // 隣接ノード
    HNSWNode* hnsw_node;               // 対応するHNSWノード
};
```

### Edgeクラス

```cpp
struct Edge {
    double length;  // エッジの長さ（未使用）
    double weight;  // エッジの重み（年齢の代わり）
};
```

### HNSWNodeクラス

```cpp
struct HNSWNode {
    Node* gng_node;            // 対応するGNGノード
    int distance;              // 検索時の距離（ホップ数）
    double euclidean_distance; // 入力ベクトルとのユークリッド距離
    int time_stamp;            // ノード追加時のタイムスタンプ
    double level;              // ノードの連続レベル
};
```

## GNGクラス

### 初期化パラメータ

```cpp
class GNG {
public:
    int new_node_interval = 300;         // ノード追加間隔 (lambda)
    int max_nodes = 100;                 // 最大ノード数
    double eta1 = 0.05;                  // 勝者ノード学習率
    double eta2 = 0.006;                 // 近傍ノード学習率
    double node_removal_threshold = 2;   // ノード削除閾値
    double edge_removal_threshold = 0.5; // エッジ削除閾値
};
```

### HNSWパラメータ

```cpp
class HNSW {
public:
    int ef_upper = 2;   // 上層検索時のビーム幅
    int ef_lower = 10;  // 下層検索時のビーム幅
    double mL = 0.5;    // 上層遷移確率パラメータ
};
```

## 主要メソッド

### GNGクラス

| メソッド | 説明 |
|---------|------|
| `initialize()` | 2ノードで初期化 |
| `selectWinners(input_vector)` | 勝者ノード選択 |
| `updateEdges(winners)` | エッジの更新（重みベース） |
| `updateReferenceVectors(winners, input)` | 参照ベクトル・分散共分散行列の更新 |
| `findNodesUandF()` | 最大誤差ノードとその隣接ノードを探索 |
| `addNode(u, f)` | uとfの中点にノード追加 |
| `addIfSilent(winners, input)` | 距離が大きい場合にノード追加 |
| `removeNodes(node)` | ノード削除 |
| `run(input_vector_with_label)` | 1ステップの学習 |

### HNSWクラス

| メソッド | 説明 |
|---------|------|
| `insert(gng_node)` | GNGノードをHNSWに挿入 |
| `hierarchicalBeamSearch(input)` | 階層的ビームサーチで近傍探索 |

## 分散共分散行列の更新

各ノードで分散共分散行列を維持:

```cpp
void updateReferenceVectors(Winners& winners, const Eigen::VectorXd& input) {
    Eigen::VectorXd error_vector = input - winners.s1->reference_vector;

    // 参照ベクトルの更新
    winners.s1->reference_vector += eta1 * error_vector;

    // 分散共分散行列の更新
    winners.s1->vcm += eta1 * (error_vector * error_vector.transpose() - winners.s1->vcm);
}
```

## Add-if-Silent

分散に基づく閾値でノード追加:

```cpp
void addIfSilent(Winners& winners, const Eigen::VectorXd& input) {
    double euclidean_distance = (input - winners.s1->reference_vector).squaredNorm();
    double threshold = 4.0 * winners.s1->squared_radius;  // 2σ

    if (euclidean_distance > threshold) {
        // 新ノード追加
        Node* new_node = new Node{input, 1.0, winners.s1->vcm, ...};
        nodes.insert(new_node);
        hnsw.insert(new_node);
    }
}
```

## HNSWによる近傍探索

階層的なビームサーチによる高速探索:

```cpp
Winners hierarchicalBeamSearch(const Eigen::VectorXd& input_vector) {
    // 上層から下層へ探索
    int current_level = log(node_count) * mL;

    while (true) {
        // ビームサーチ
        for (auto neighbor : current_node->gng_node->neighbors) {
            if (neighbor level >= current_level) {
                // 距離計算・候補追加
            }
        }

        if (current_level == 0) break;
        current_level--;  // 下層へ
    }
}
```

## エッジの重みベース管理

年齢の代わりに重みを使用:

```cpp
// 勝者ノード間のエッジ: 重みをリセット
edge->weight = 1.0;

// その他のエッジ: 重みを減衰
edge->weight += eta2 * (0.0 - edge->weight);

// 閾値以下で削除
if (edge->weight < edge_removal_threshold) {
    checkEdgeExists(neighbor);
}
```

## 依存ライブラリ

- Eigen (線形代数)
- Boost.Heap (pairing_heap)

## 参考にすべきポイント

1. **HNSW統合**: GNGのエッジを利用したHNSW構築
2. **分散共分散行列**: 各ノードの局所的な分布を学習
3. **重みベースのエッジ管理**: 年齢ではなく重みで管理
4. **ビームサーチ**: 効率的な近傍探索

## 計算量

| 操作 | 標準GNG | HNSW-GNG |
|-----|--------|----------|
| 勝者探索 | O(N) | O(log N) |
| ノード追加 | O(1) | O(log N) |

## 関連アルゴリズム

- `algorithms/ais_gng/`: Add-if-Silent機能
- HNSW統合は `algorithms/` には未実装。大規模データ対応として有望。
