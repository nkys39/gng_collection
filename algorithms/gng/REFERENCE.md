# GNG (Growing Neural Gas)

## 論文

- Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies"
- Advances in Neural Information Processing Systems 7 (NIPS 1994)
- URL: https://proceedings.neurips.cc/paper/1994/file/d56b9fc4b0f1be8571f8e2db7ae49e87-Paper.pdf

## 元コード

### watanabe_gng (C++)

- 場所: `references/original_code/watanabe_gng/`
- Author: watanabe
- Created: 2024/05/16
- License: (要確認)
- 特徴:
  - Eigenライブラリ使用
  - テンプレートによる2D/3D対応
  - 固定サイズ配列によるノード管理

## 概要

Growing Neural Gas (GNG) は、入力データの位相構造を学習する自己組織化ネットワークです。
競合学習とHebbianルールに基づいてノードとエッジを動的に追加・削除します。

## 主な特徴

- オンライン学習（逐次的にデータを処理）
- ノード数の自動調整
- 位相構造の学習
- ノイズに対するロバスト性

## アルゴリズムの流れ

1. 2つのノードで初期化
2. 入力サンプルに対して最近傍の2ノード(s1, s2)を探索
3. s1のローカルエラーを更新
4. s1とその近傍をサンプル方向に移動
5. s1-s2間のエッジ年齢を0にリセット（なければ作成）
6. s1に接続するエッジの年齢をインクリメント
7. 最大年齢を超えたエッジを削除
8. λステップごとに、最大エラーのノード付近に新ノードを挿入
9. 全ノードのエラーを減衰

## ハイパーパラメータ

| パラメータ | 説明 | 典型値 |
|-----------|------|--------|
| λ (lambda) | ノード挿入間隔 | 100-300 |
| ε_b | 勝者ノードの学習率 | 0.05-0.3 |
| ε_n | 近傍ノードの学習率 | 0.001-0.01 |
| α | エラー減衰率（分割時） | 0.5 |
| β | グローバルエラー減衰率 | 0.0005-0.001 |
| a_max | エッジの最大年齢 | 50-200 |

## 変更履歴

- (初期実装時に記載)
