#!/usr/bin/env python3
"""密なグラフでの極大クリーク検出テスト

K4, K5などの大きなクリークを含む密なグラフで、
Bron-Kerbosch法の基本版とPivot最適化版を比較。
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np

from experiments.polygon_filling.polygon_utils import (
    bron_kerbosch_basic,
    bron_kerbosch_pivot,
    compare_algorithms,
    decompose_to_triangles,
    print_comparison,
)


def create_dense_random_graph(
    n_nodes: int,
    edge_probability: float,
    seed: int = 42,
) -> tuple[dict[int, set[int]], list[int]]:
    """密なランダムグラフを生成（Erdős–Rényi model）

    Args:
        n_nodes: ノード数
        edge_probability: エッジ存在確率
        seed: 乱数シード

    Returns:
        (edges_per_node, active_node_ids)
    """
    rng = np.random.default_rng(seed)
    edges_per_node: dict[int, set[int]] = {i: set() for i in range(n_nodes)}

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_probability:
                edges_per_node[i].add(j)
                edges_per_node[j].add(i)

    active_ids = list(range(n_nodes))
    return edges_per_node, active_ids


def create_complete_graph(n_nodes: int) -> tuple[dict[int, set[int]], list[int]]:
    """完全グラフを生成

    Args:
        n_nodes: ノード数

    Returns:
        (edges_per_node, active_node_ids)
    """
    edges_per_node: dict[int, set[int]] = {i: set() for i in range(n_nodes)}

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edges_per_node[i].add(j)

    active_ids = list(range(n_nodes))
    return edges_per_node, active_ids


def create_multiple_cliques_graph() -> tuple[dict[int, set[int]], list[int]]:
    """複数のクリークを含むグラフを生成

    構成:
    - K5 (ノード 0-4)
    - K4 (ノード 5-8)
    - K3 (ノード 9-11)
    - いくつかの接続エッジ
    """
    n_nodes = 12
    edges_per_node: dict[int, set[int]] = {i: set() for i in range(n_nodes)}

    def add_clique(nodes: list[int]):
        for i in nodes:
            for j in nodes:
                if i != j:
                    edges_per_node[i].add(j)

    # K5
    add_clique([0, 1, 2, 3, 4])
    # K4
    add_clique([5, 6, 7, 8])
    # K3
    add_clique([9, 10, 11])

    # クリーク間の接続（極大性を壊さない程度に）
    edges_per_node[4].add(5)
    edges_per_node[5].add(4)
    edges_per_node[8].add(9)
    edges_per_node[9].add(8)

    return edges_per_node, list(range(n_nodes))


def main():
    """メイン関数"""
    print("=" * 60)
    print("密なグラフでの極大クリーク検出テスト")
    print("=" * 60)

    # テスト1: 複数クリークグラフ（既知の構造）
    print("\n" + "=" * 60)
    print("テスト1: 複数クリークグラフ（K5 + K4 + K3）")
    print("=" * 60)

    edges, active_ids = create_multiple_cliques_graph()
    n_edges = sum(len(neighbors) for neighbors in edges.values()) // 2
    print(f"ノード数: {len(active_ids)}, エッジ数: {n_edges}")

    comparison = compare_algorithms(edges, active_ids, min_size=3, n_runs=100)
    print_comparison(comparison)

    # 三角形分割
    result = bron_kerbosch_pivot(edges, active_ids, min_size=3)
    triangles = decompose_to_triangles(result.cliques)
    print(f"\n三角形分割後: {len(triangles)}個の三角形")

    # テスト2: ランダム密グラフ（中程度）
    print("\n" + "=" * 60)
    print("テスト2: ランダム密グラフ（n=30, p=0.3）")
    print("=" * 60)

    edges, active_ids = create_dense_random_graph(30, 0.3, seed=42)
    n_edges = sum(len(neighbors) for neighbors in edges.values()) // 2
    print(f"ノード数: {len(active_ids)}, エッジ数: {n_edges}")

    comparison = compare_algorithms(edges, active_ids, min_size=3, n_runs=20)
    print_comparison(comparison)

    # テスト3: ランダム密グラフ（高密度）
    print("\n" + "=" * 60)
    print("テスト3: ランダム密グラフ（n=30, p=0.5）")
    print("=" * 60)

    edges, active_ids = create_dense_random_graph(30, 0.5, seed=42)
    n_edges = sum(len(neighbors) for neighbors in edges.values()) // 2
    print(f"ノード数: {len(active_ids)}, エッジ数: {n_edges}")

    comparison = compare_algorithms(edges, active_ids, min_size=3, n_runs=20)
    print_comparison(comparison)

    # テスト4: ランダム密グラフ（非常に高密度）
    print("\n" + "=" * 60)
    print("テスト4: ランダム密グラフ（n=25, p=0.7）")
    print("=" * 60)

    edges, active_ids = create_dense_random_graph(25, 0.7, seed=42)
    n_edges = sum(len(neighbors) for neighbors in edges.values()) // 2
    print(f"ノード数: {len(active_ids)}, エッジ数: {n_edges}")

    comparison = compare_algorithms(edges, active_ids, min_size=3, n_runs=10)
    print_comparison(comparison)

    # テスト5: 完全グラフ
    print("\n" + "=" * 60)
    print("テスト5: 完全グラフ K10")
    print("=" * 60)

    edges, active_ids = create_complete_graph(10)
    n_edges = sum(len(neighbors) for neighbors in edges.values()) // 2
    print(f"ノード数: {len(active_ids)}, エッジ数: {n_edges}")

    comparison = compare_algorithms(edges, active_ids, min_size=3, n_runs=100)
    print_comparison(comparison)

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
