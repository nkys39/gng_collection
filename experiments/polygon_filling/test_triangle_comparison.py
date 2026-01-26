#!/usr/bin/env python3
"""三角形検出方法の比較テスト

極大クリーク方式 vs エッジベース方式（Delaunay + GNGエッジフィルタ）
の検出数と可視化を比較する。
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "2d"))
sys.path.insert(0, str(project_root / "algorithms" / "gng" / "python"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon as MplPolygon

from model import GrowingNeuralGas, GNGParams
from sampler import sample_triple_ring
from experiments.polygon_filling.polygon_utils import (
    bron_kerbosch_pivot,
    decompose_to_triangles,
    get_edge_based_triangles,
    get_all_graph_triangles,
    compare_triangle_methods,
    print_triangle_comparison,
)


# Triple ring geometry
TRIPLE_RING_PARAMS = [
    (0.50, 0.23, 0.06, 0.14),
    (0.27, 0.68, 0.06, 0.14),
    (0.73, 0.68, 0.06, 0.14),
]


def draw_ring_outlines(ax, alpha: float = 0.3) -> None:
    """リングの輪郭を描画"""
    theta = np.linspace(0, 2 * np.pi, 100)
    for cx, cy, inner_r, outer_r in TRIPLE_RING_PARAMS:
        outer_x = cx + outer_r * np.cos(theta)
        outer_y = cy + outer_r * np.sin(theta)
        inner_x = cx + inner_r * np.cos(theta)
        inner_y = cy + inner_r * np.sin(theta)
        ax.fill(outer_x, outer_y, color="lightblue", alpha=alpha)
        ax.fill(inner_x, inner_y, color="white")


def draw_triangles(ax, nodes, triangles, facecolor, edgecolor, alpha=0.4):
    """三角形を描画"""
    for tri in triangles:
        coords = nodes[list(tri)]
        patch = MplPolygon(
            coords,
            fill=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=0.5,
        )
        ax.add_patch(patch)


def draw_graph(ax, nodes, edges, node_color="red", edge_color="gray"):
    """グラフを描画"""
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            color=edge_color,
            linewidth=0.5,
            zorder=1,
        )
    ax.scatter(nodes[:, 0], nodes[:, 1], c=node_color, s=15, zorder=2)


def get_graph_data(gng):
    """GNGからグラフデータを取得"""
    nodes, edges = gng.get_graph()

    # ノードIDからインデックスへのマッピング
    id_to_idx = {}
    idx = 0
    for node in gng.nodes:
        if node.id != -1:
            id_to_idx[node.id] = idx
            idx += 1

    # edges_per_nodeをインデックスベースに変換
    edges_per_node: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    for node_id, neighbors in gng.edges_per_node.items():
        if gng.nodes[node_id].id == -1:
            continue
        node_idx = id_to_idx[node_id]
        for neighbor_id in neighbors:
            if gng.nodes[neighbor_id].id == -1:
                continue
            neighbor_idx = id_to_idx[neighbor_id]
            edges_per_node[node_idx].add(neighbor_idx)

    return nodes, edges, edges_per_node


def visualize_comparison(ax_clique, ax_edge, gng, iteration):
    """2つの方法を並べて可視化"""
    nodes, edges, edges_per_node = get_graph_data(gng)

    if len(nodes) < 3:
        for ax in [ax_clique, ax_edge]:
            ax.clear()
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
        return 0, 0

    # 極大クリーク方式
    result = bron_kerbosch_pivot(edges_per_node, list(range(len(nodes))), min_size=3)
    triangles_clique = decompose_to_triangles(result.cliques)

    # グラフ構造から直接検出（エッジで囲まれた三角形を全て検出）
    triangles_edge = get_all_graph_triangles(edges_per_node)

    # 左: 極大クリーク方式
    ax_clique.clear()
    ax_clique.set_xlim(-0.05, 1.05)
    ax_clique.set_ylim(-0.05, 1.05)
    ax_clique.set_aspect("equal")
    ax_clique.set_title(f"Maximal Clique (n={len(triangles_clique)})")
    draw_ring_outlines(ax_clique)
    draw_triangles(ax_clique, nodes, triangles_clique, "lightgreen", "green")
    draw_graph(ax_clique, nodes, edges)

    # 右: エッジベース方式
    ax_edge.clear()
    ax_edge.set_xlim(-0.05, 1.05)
    ax_edge.set_ylim(-0.05, 1.05)
    ax_edge.set_aspect("equal")
    ax_edge.set_title(f"All Graph Triangles (n={len(triangles_edge)})")
    draw_ring_outlines(ax_edge)
    draw_triangles(ax_edge, nodes, triangles_edge, "lightyellow", "orange")
    draw_graph(ax_edge, nodes, edges)

    return len(triangles_clique), len(triangles_edge)


def main():
    print("=" * 60)
    print("三角形検出方法の比較: 極大クリーク vs エッジベース")
    print("=" * 60)

    # パラメータ
    params = GNGParams(
        max_nodes=100,
        lambda_=100,
        eps_b=0.08,
        eps_n=0.008,
        alpha=0.5,
        beta=0.005,
        max_age=100,
    )
    n_iterations = 5000
    n_samples = 1500
    seed = 42

    # データ生成
    data = sample_triple_ring(n_samples=n_samples, seed=seed)

    # GNG初期化
    gng = GrowingNeuralGas(n_dim=2, params=params, seed=seed)

    # フレーム収集
    frames = []
    frame_interval = 100

    print(f"\nTraining: {n_iterations} iterations")

    fig, (ax_clique, ax_edge) = plt.subplots(1, 2, figsize=(14, 7))

    np.random.seed(seed)

    for i in range(n_iterations):
        idx = np.random.randint(0, n_samples)
        gng.partial_fit(data[idx])

        if i % frame_interval == 0 or i == n_iterations - 1:
            n_clique, n_edge = visualize_comparison(ax_clique, ax_edge, gng, i)

            fig.suptitle(f"iter={i}, nodes={gng.n_nodes}", fontsize=14)

            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            frames.append(image.copy())

            if i % 500 == 0:
                print(f"  iter={i}: nodes={gng.n_nodes}, clique={n_clique}, edge={n_edge}")

    # 最終比較
    print(f"\nTraining complete: {gng.n_nodes} nodes")
    nodes, edges, edges_per_node = get_graph_data(gng)
    comparison = compare_triangle_methods(nodes, edges_per_node, n_runs=10)
    print_triangle_comparison(comparison)

    # 出力
    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    # 最終画像
    final_path = output_dir / "triangle_comparison_final.png"
    fig.savefig(final_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {final_path}")

    # GIF
    gif_path = output_dir / "triangle_comparison_growth.gif"
    print(f"Generating GIF... ({len(frames)} frames)")

    fig_anim, ax_anim = plt.subplots(figsize=(14, 7))

    def update(frame_idx):
        ax_anim.clear()
        ax_anim.imshow(frames[frame_idx])
        ax_anim.axis("off")
        return []

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig_anim, update, frames=len(frames), interval=200, blit=True)
    anim.save(gif_path, writer=PillowWriter(fps=5))
    print(f"Saved: {gif_path}")

    plt.close("all")
    print("\nDone!")


if __name__ == "__main__":
    main()
