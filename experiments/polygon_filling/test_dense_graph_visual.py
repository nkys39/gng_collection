#!/usr/bin/env python3
"""密なグラフでの極大クリーク検出テスト（可視化付き）

複数のクリーク構造を含むグラフを可視化。
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from experiments.polygon_filling.polygon_utils import (
    bron_kerbosch_pivot,
    decompose_to_triangles,
)


# 色の定義（クリークサイズ別）
CLIQUE_COLORS = {
    3: ("#90EE90", "#228B22"),      # lightgreen, forestgreen
    4: ("#87CEEB", "#0000CD"),       # skyblue, mediumblue
    5: ("#FFD700", "#FF8C00"),       # gold, darkorange
    6: ("#FF6347", "#8B0000"),       # tomato, darkred
    7: ("#DA70D6", "#800080"),       # orchid, purple
    8: ("#00CED1", "#008B8B"),       # darkturquoise, darkcyan
}


def create_multiple_cliques_graph_with_positions():
    """複数のクリークを含むグラフを生成（位置情報付き）

    構成:
    - K5 (ノード 0-4): 左側
    - K4 (ノード 5-8): 中央
    - K3 (ノード 9-11): 右側
    """
    # ノード位置
    positions = np.array([
        # K5 (五角形配置)
        [0.15, 0.5],   # 0
        [0.25, 0.8],   # 1
        [0.35, 0.65],  # 2
        [0.35, 0.35],  # 3
        [0.25, 0.2],   # 4
        # K4 (四角形配置)
        [0.50, 0.7],   # 5
        [0.65, 0.7],   # 6
        [0.65, 0.3],   # 7
        [0.50, 0.3],   # 8
        # K3 (三角形配置)
        [0.80, 0.65],  # 9
        [0.90, 0.5],   # 10
        [0.80, 0.35],  # 11
    ])

    n_nodes = 12
    edges_per_node: dict[int, set[int]] = {i: set() for i in range(n_nodes)}

    def add_clique(nodes: list[int]):
        for i in nodes:
            for j in nodes:
                if i != j:
                    edges_per_node[i].add(j)

    # クリーク作成
    add_clique([0, 1, 2, 3, 4])  # K5
    add_clique([5, 6, 7, 8])      # K4
    add_clique([9, 10, 11])       # K3

    # クリーク間の接続
    edges_per_node[4].add(5)
    edges_per_node[5].add(4)
    edges_per_node[8].add(9)
    edges_per_node[9].add(8)

    return positions, edges_per_node


def create_random_dense_graph_with_positions(n_nodes: int, p: float, seed: int = 42):
    """ランダム密グラフを生成（位置情報付き）"""
    rng = np.random.default_rng(seed)

    # ノード位置（円形配置）
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = np.column_stack([
        0.5 + 0.4 * np.cos(angles),
        0.5 + 0.4 * np.sin(angles),
    ])

    # 少しランダムにずらす
    positions += rng.normal(0, 0.03, positions.shape)
    positions = np.clip(positions, 0.05, 0.95)

    # エッジ生成
    edges_per_node: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p:
                edges_per_node[i].add(j)
                edges_per_node[j].add(i)

    return positions, edges_per_node


def draw_graph_with_cliques(
    ax,
    positions: np.ndarray,
    edges_per_node: dict[int, set[int]],
    title: str = "",
):
    """グラフとクリークを描画"""
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)

    n_nodes = len(positions)
    active_ids = list(range(n_nodes))

    # クリーク検出
    result = bron_kerbosch_pivot(edges_per_node, active_ids, min_size=3)
    cliques_by_size = result.by_size()

    # 統計表示用テキスト
    stats_text = []
    for size in sorted(cliques_by_size.keys()):
        count = len(cliques_by_size[size])
        stats_text.append(f"K{size}: {count}")

    # クリークを描画（大きいものから）
    for size in sorted(cliques_by_size.keys(), reverse=True):
        cliques = cliques_by_size[size]
        if not cliques:
            continue

        triangles = decompose_to_triangles(cliques)
        facecolor, edgecolor = CLIQUE_COLORS.get(size, ("#CCCCCC", "gray"))

        for tri in triangles:
            coords = positions[list(tri)]
            patch = MplPolygon(
                coords,
                fill=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=0.5,
                linewidth=1,
            )
            ax.add_patch(patch)

    # エッジ描画
    drawn_edges = set()
    for i, neighbors in edges_per_node.items():
        for j in neighbors:
            edge = tuple(sorted([i, j]))
            if edge not in drawn_edges:
                drawn_edges.add(edge)
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color="gray",
                    linewidth=0.5,
                    alpha=0.5,
                    zorder=1,
                )

    # ノード描画
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c="red",
        s=50,
        zorder=3,
        edgecolors="black",
        linewidths=0.5,
    )

    # ノード番号
    for i, (x, y) in enumerate(positions):
        ax.annotate(
            str(i),
            (x, y),
            fontsize=6,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
        )

    # 統計テキスト
    ax.text(
        0.02, 0.98, ", ".join(stats_text),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = []
    for size in sorted(cliques_by_size.keys()):
        if cliques_by_size[size]:
            facecolor, _ = CLIQUE_COLORS.get(size, ("#CCCCCC", "gray"))
            legend_elements.append(
                Patch(facecolor=facecolor, edgecolor='black', alpha=0.6, label=f'K{size}')
            )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)


def main():
    """メイン関数"""
    print("Dense Graph Clique Visualization")
    print("=" * 60)

    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    # 4つのグラフを2x2で配置
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. 複数クリーク（既知の構造）
    print("\n1. Multiple Cliques (K5 + K4 + K3)")
    positions, edges = create_multiple_cliques_graph_with_positions()
    draw_graph_with_cliques(axes[0, 0], positions, edges, "Multiple Cliques (K5 + K4 + K3)")

    # 2. ランダム密グラフ (p=0.3)
    print("2. Random Graph (n=20, p=0.3)")
    positions, edges = create_random_dense_graph_with_positions(20, 0.3, seed=42)
    draw_graph_with_cliques(axes[0, 1], positions, edges, "Random Graph (n=20, p=0.3)")

    # 3. ランダム密グラフ (p=0.5)
    print("3. Random Graph (n=15, p=0.5)")
    positions, edges = create_random_dense_graph_with_positions(15, 0.5, seed=42)
    draw_graph_with_cliques(axes[1, 0], positions, edges, "Random Graph (n=15, p=0.5)")

    # 4. ランダム密グラフ (p=0.7)
    print("4. Random Graph (n=12, p=0.7)")
    positions, edges = create_random_dense_graph_with_positions(12, 0.7, seed=42)
    draw_graph_with_cliques(axes[1, 1], positions, edges, "Random Graph (n=12, p=0.7)")

    plt.tight_layout()

    # 保存
    output_path = output_dir / "dense_graph_cliques.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    plt.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
