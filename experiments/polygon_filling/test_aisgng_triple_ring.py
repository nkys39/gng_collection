#!/usr/bin/env python3
"""AiS-GNGトリプルリング多角形塗りつぶしテスト

AiS-GNGはノードを密に生成するため、K4, K5などの
大きなクリークが検出されやすい。
"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートとデータディレクトリをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "2d"))
sys.path.insert(0, str(project_root / "algorithms" / "ais_gng" / "python"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon as MplPolygon

from model import AiSGNG, AiSGNGParams
from sampler import sample_triple_ring
from experiments.polygon_filling.polygon_utils import (
    bron_kerbosch_pivot,
    compare_algorithms,
    decompose_to_triangles,
    print_comparison,
)


# Triple ring geometry (matching existing tests)
TRIPLE_RING_PARAMS = [
    (0.50, 0.23, 0.06, 0.14),  # top center
    (0.27, 0.68, 0.06, 0.14),  # bottom left
    (0.73, 0.68, 0.06, 0.14),  # bottom right
]

# 色の定義（クリークサイズ別）
CLIQUE_COLORS = {
    3: ("#90EE90", "green"),      # lightgreen
    4: ("#87CEEB", "blue"),       # skyblue
    5: ("#FFD700", "orange"),     # gold
    6: ("#FF6347", "red"),        # tomato
    7: ("#DA70D6", "purple"),     # orchid
    8: ("#00CED1", "darkturquoise"),
}


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


def draw_cliques_by_size(
    ax,
    nodes: np.ndarray,
    cliques_by_size: dict[int, list[tuple[int, ...]]],
    alpha: float = 0.4,
) -> None:
    """クリークをサイズ別に色分けして描画（三角形分割）"""

    # 大きいクリークから描画（背面に）
    for size in sorted(cliques_by_size.keys(), reverse=True):
        cliques = cliques_by_size[size]
        if not cliques:
            continue

        # 三角形に分割
        triangles = decompose_to_triangles(cliques)

        facecolor, edgecolor = CLIQUE_COLORS.get(size, ("#CCCCCC", "gray"))

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


def draw_graph(
    ax,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    node_color: str = "red",
    edge_color: str = "gray",
    node_size: float = 10,
    edge_width: float = 0.3,
) -> None:
    """グラフ（ノードとエッジ）を描画"""
    # エッジ
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            color=edge_color,
            linewidth=edge_width,
            zorder=1,
        )

    # ノード
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=node_color,
        s=node_size,
        zorder=2,
    )


def add_legend(ax, cliques_by_size: dict[int, int]) -> None:
    """凡例を追加"""
    from matplotlib.patches import Patch

    legend_elements = []
    for size in sorted(cliques_by_size.keys()):
        count = cliques_by_size[size]
        if count > 0:
            facecolor, _ = CLIQUE_COLORS.get(size, ("#CCCCCC", "gray"))
            legend_elements.append(
                Patch(facecolor=facecolor, edgecolor='black', alpha=0.6,
                      label=f'K{size}: {count}個')
            )

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def visualize_frame(
    ax,
    gng: AiSGNG,
    iteration: int,
) -> dict:
    """1フレームを描画し、統計情報を返す"""
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"AiS-GNG Polygon Filling (iter={iteration}, nodes={gng.n_nodes})")

    # リング輪郭
    draw_ring_outlines(ax)

    # グラフ取得
    nodes, edges = gng.get_graph()
    if len(nodes) < 3:
        return {"triangles": 0, "cliques": {}}

    # ノードIDからグラフインデックスへのマッピング
    id_to_idx = {}
    idx = 0
    for node in gng.nodes:
        if node.id != -1:
            id_to_idx[node.id] = idx
            idx += 1

    # edges_per_node をグラフインデックスベースに変換
    edges_per_node_idx: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    for node_id, neighbors in gng.edges_per_node.items():
        if gng.nodes[node_id].id == -1:
            continue
        node_idx = id_to_idx[node_id]
        for neighbor_id in neighbors:
            if gng.nodes[neighbor_id].id == -1:
                continue
            neighbor_idx = id_to_idx[neighbor_id]
            edges_per_node_idx[node_idx].add(neighbor_idx)

    # クリーク検出
    result = bron_kerbosch_pivot(
        edges_per_node_idx,
        list(range(len(nodes))),
        min_size=3,
    )

    cliques_by_size = result.by_size()
    count_by_size = result.count_by_size()

    stats = {
        "cliques": count_by_size,
        "triangles": sum(
            len(cliques) * (size - 2)
            for size, cliques in cliques_by_size.items()
        ),
    }

    # クリークをサイズ別に描画
    if result.cliques:
        draw_cliques_by_size(ax, nodes, cliques_by_size)

    # グラフ描画
    draw_graph(ax, nodes, edges)

    # 凡例追加
    add_legend(ax, count_by_size)

    return stats


def main():
    """メイン関数"""
    print("AiS-GNG トリプルリング多角形塗りつぶしテスト")
    print("=" * 60)

    # パラメータ（AiS-GNG用、より多くのノードを生成）
    params = AiSGNGParams(
        max_nodes=150,         # より多くのノード
        lambda_=100,
        kappa=10,
        eps_b=0.05,            # やや小さい学習率
        eps_n=0.005,
        alpha=0.5,
        beta=0.005,
        chi=0.005,
        max_age=50,            # 短めのエッジ寿命
        utility_k=1000,
        theta_ais_min=0.015,   # 小さい閾値でより密に
        theta_ais_max=0.06,
    )
    n_iterations = 8000
    n_samples = 1500
    seed = 42

    # データ生成
    data = sample_triple_ring(n_samples=n_samples, seed=seed)

    # AiS-GNG初期化
    gng = AiSGNG(n_dim=2, params=params, seed=seed)

    # アニメーション用フレーム収集
    frames = []
    frame_interval = 100

    print(f"\n学習開始: {n_iterations}イテレーション")
    print(f"パラメータ: max_nodes={params.max_nodes}, theta_ais=[{params.theta_ais_min}, {params.theta_ais_max}]")

    # 学習ループ
    fig, ax = plt.subplots(figsize=(10, 10))

    np.random.seed(seed)

    for i in range(n_iterations):
        idx = np.random.randint(0, n_samples)
        gng.partial_fit(data[idx])

        if i % frame_interval == 0 or i == n_iterations - 1:
            stats = visualize_frame(ax, gng, i)

            # フレームを画像として保存
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            frames.append(image.copy())

            if i % 500 == 0:
                clique_str = ", ".join(
                    f"K{k}:{v}" for k, v in sorted(stats["cliques"].items())
                )
                print(f"  iter={i}: nodes={gng.n_nodes}, triangles={stats['triangles']}, cliques=[{clique_str}]")

    # 最終結果表示
    print(f"\n学習完了: {gng.n_nodes}ノード")

    # アルゴリズム比較
    print("\n" + "=" * 60)

    # ノードIDからグラフインデックスへのマッピング
    id_to_idx = {}
    idx = 0
    for node in gng.nodes:
        if node.id != -1:
            id_to_idx[node.id] = idx
            idx += 1

    n_nodes = len(id_to_idx)
    edges_per_node_idx: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for node_id, neighbors in gng.edges_per_node.items():
        if gng.nodes[node_id].id == -1:
            continue
        node_idx = id_to_idx[node_id]
        for neighbor_id in neighbors:
            if gng.nodes[neighbor_id].id == -1:
                continue
            neighbor_idx = id_to_idx[neighbor_id]
            edges_per_node_idx[node_idx].add(neighbor_idx)

    comparison = compare_algorithms(
        edges_per_node_idx,
        list(range(n_nodes)),
        min_size=3,
        n_runs=10,
    )
    print_comparison(comparison)

    # 出力ディレクトリ
    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    # 最終フレームを保存
    final_path = output_dir / "aisgng_triple_ring_final.png"
    fig.savefig(final_path, dpi=150, bbox_inches="tight")
    print(f"\n最終画像を保存: {final_path}")

    # GIFアニメーション保存
    gif_path = output_dir / "aisgng_triple_ring_growth.gif"

    print(f"GIFを生成中... ({len(frames)}フレーム)")

    # 新しいfigureでアニメーション作成
    fig_anim, ax_anim = plt.subplots(figsize=(10, 10))

    def update(frame_idx):
        ax_anim.clear()
        ax_anim.imshow(frames[frame_idx])
        ax_anim.axis("off")
        return []

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(
        fig_anim,
        update,
        frames=len(frames),
        interval=200,
        blit=True,
    )
    anim.save(gif_path, writer=PillowWriter(fps=5))
    print(f"GIFを保存: {gif_path}")

    plt.close("all")

    print("\n完了!")


if __name__ == "__main__":
    main()
