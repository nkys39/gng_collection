#!/usr/bin/env python3
"""GNGトリプルリング多角形塗りつぶしテスト

極大クリーク検出と三角形分割による多角形塗りつぶしの可視化実験。
Bron-Kerbosch法の基本版とPivot最適化版の比較も行う。
"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートとデータディレクトリをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "2d"))
sys.path.insert(0, str(project_root / "algorithms" / "gng" / "python"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon

from model import GrowingNeuralGas, GNGParams
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


def draw_triangles(
    ax,
    nodes: np.ndarray,
    triangles: list[tuple[int, int, int]],
    facecolor: str = "lightgreen",
    edgecolor: str = "green",
    alpha: float = 0.4,
    linewidth: float = 0.5,
) -> list:
    """三角形を描画"""
    patches = []
    for tri in triangles:
        coords = nodes[list(tri)]
        patch = MplPolygon(
            coords,
            fill=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def draw_graph(
    ax,
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    node_color: str = "red",
    edge_color: str = "gray",
    node_size: float = 15,
    edge_width: float = 0.5,
) -> tuple:
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
    scatter = ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=node_color,
        s=node_size,
        zorder=2,
    )
    return scatter


def visualize_frame(
    ax,
    gng: GrowingNeuralGas,
    iteration: int,
    show_triangles: bool = True,
) -> dict:
    """1フレームを描画し、統計情報を返す"""
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"GNG Polygon Filling (iter={iteration}, nodes={gng.n_nodes})")

    # リング輪郭
    draw_ring_outlines(ax)

    # グラフ取得
    nodes, edges = gng.get_graph()
    if len(nodes) < 3:
        return {"triangles": 0, "cliques": {}}

    # 極大クリーク検出
    active_ids = [i for i, node in enumerate(gng.nodes) if node.id != -1]

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

    stats = {
        "cliques": result.count_by_size(),
        "triangles": 0,
    }

    # 三角形に分割して描画
    if show_triangles and result.cliques:
        triangles = decompose_to_triangles(result.cliques)
        stats["triangles"] = len(triangles)
        draw_triangles(ax, nodes, triangles)

    # グラフ描画
    draw_graph(ax, nodes, edges)

    return stats


def run_comparison(gng: GrowingNeuralGas) -> dict:
    """アルゴリズム比較を実行"""
    # ノードIDからグラフインデックスへのマッピング
    id_to_idx = {}
    idx = 0
    for node in gng.nodes:
        if node.id != -1:
            id_to_idx[node.id] = idx
            idx += 1

    # edges_per_node をグラフインデックスベースに変換
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

    return compare_algorithms(
        edges_per_node_idx,
        list(range(n_nodes)),
        min_size=3,
        n_runs=10,
    )


def main():
    """メイン関数"""
    print("GNG トリプルリング多角形塗りつぶしテスト")
    print("=" * 50)

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

    # アニメーション用フレーム収集
    frames = []
    frame_interval = 100  # 100イテレーションごとにフレーム保存

    print(f"\n学習開始: {n_iterations}イテレーション")

    # 学習ループ
    fig, ax = plt.subplots(figsize=(8, 8))

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
    print("\n" + "=" * 50)
    comparison = run_comparison(gng)
    print_comparison(comparison)

    # 最終フレームを保存
    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    final_path = output_dir / "gng_triple_ring_final.png"
    fig.savefig(final_path, dpi=150, bbox_inches="tight")
    print(f"\n最終画像を保存: {final_path}")

    # GIFアニメーション保存
    gif_path = output_dir / "gng_triple_ring_growth.gif"

    print(f"GIFを生成中... ({len(frames)}フレーム)")

    # 新しいfigureでアニメーション作成
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

    def update(frame_idx):
        ax_anim.clear()
        ax_anim.imshow(frames[frame_idx])
        ax_anim.axis("off")
        return []

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

    # 比較結果をファイルに保存
    comparison_path = output_dir / "algorithm_comparison.txt"
    with open(comparison_path, "w") as f:
        f.write("Bron-Kerbosch アルゴリズム比較結果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"ノード数: {gng.n_nodes}\n")
        f.write(f"エッジ数: {gng.n_edges}\n\n")

        f.write("【基本版】\n")
        f.write(f"  検出数: {comparison['basic']['total']}個\n")
        for size, count in sorted(comparison['basic']['cliques'].items()):
            f.write(f"    K{size}: {count}個\n")
        f.write(f"  平均実行時間: {comparison['basic']['avg_time_ms']:.3f} ms\n\n")

        f.write("【Pivot最適化版】\n")
        f.write(f"  検出数: {comparison['pivot']['total']}個\n")
        for size, count in sorted(comparison['pivot']['cliques'].items()):
            f.write(f"    K{size}: {count}個\n")
        f.write(f"  平均実行時間: {comparison['pivot']['avg_time_ms']:.3f} ms\n\n")

        f.write("【比較結果】\n")
        f.write(f"  検出結果一致: {'Yes' if comparison['same_result'] else 'No'}\n")
        f.write(f"  速度向上率: {comparison['speedup']:.2f}x\n")

    print(f"比較結果を保存: {comparison_path}")

    print("\n完了!")


if __name__ == "__main__":
    main()
