#!/usr/bin/env python3
"""AiS-GNGでの最小サイクル検出テスト

極大クリーク方式と最小サイクル方式の比較。
AiS-GNGは密なグラフを生成するため、より多くの多角形が検出される。
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    decompose_to_triangles,
    detect_minimal_cycles,
)


# Triple ring geometry
TRIPLE_RING_PARAMS = [
    (0.50, 0.23, 0.06, 0.14),
    (0.27, 0.68, 0.06, 0.14),
    (0.73, 0.68, 0.06, 0.14),
]

# サイズ別の色（最大12角形まで対応）
CYCLE_COLORS = {
    3: ("#90EE90", "#228B22"),   # green
    4: ("#87CEEB", "#0000CD"),   # blue
    5: ("#FFD700", "#FF8C00"),   # gold
    6: ("#FF6347", "#8B0000"),   # red
    7: ("#DA70D6", "#800080"),   # purple
    8: ("#00CED1", "#008B8B"),   # cyan
    9: ("#F0E68C", "#BDB76B"),   # khaki
    10: ("#DDA0DD", "#9932CC"),  # plum
    11: ("#98FB98", "#006400"),  # pale green
    12: ("#FFA07A", "#CD5C5C"),  # light salmon
}


def draw_ring_outlines(ax, alpha: float = 0.3) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    for cx, cy, inner_r, outer_r in TRIPLE_RING_PARAMS:
        outer_x = cx + outer_r * np.cos(theta)
        outer_y = cy + outer_r * np.sin(theta)
        inner_x = cx + inner_r * np.cos(theta)
        inner_y = cy + inner_r * np.sin(theta)
        ax.fill(outer_x, outer_y, color="lightblue", alpha=alpha)
        ax.fill(inner_x, inner_y, color="white")


def draw_cycles(ax, nodes, cycles, alpha=0.4):
    """サイクルをサイズ別に色分けして描画"""
    for cycle in cycles:
        size = len(cycle)
        facecolor, edgecolor = CYCLE_COLORS.get(size, ("#CCCCCC", "gray"))
        coords = nodes[cycle]
        patch = MplPolygon(
            coords,
            fill=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=0.5,
        )
        ax.add_patch(patch)


def draw_cliques(ax, nodes, cliques, alpha=0.4):
    """クリークをサイズ別に色分けして描画（三角形分割）"""
    for clique in cliques:
        size = len(clique)
        facecolor, edgecolor = CYCLE_COLORS.get(size, ("#CCCCCC", "gray"))
        # 三角形に分割して描画
        triangles = decompose_to_triangles([clique])
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
    nodes, edges = gng.get_graph()
    id_to_idx = {}
    idx = 0
    for node in gng.nodes:
        if node.id != -1:
            id_to_idx[node.id] = idx
            idx += 1

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


def add_legend(ax, count_by_size: dict[int, int], title: str = ""):
    from matplotlib.patches import Patch
    legend_elements = []
    for size in sorted(count_by_size.keys()):
        count = count_by_size[size]
        if count > 0:
            facecolor, _ = CYCLE_COLORS.get(size, ("#CCCCCC", "gray"))
            legend_elements.append(
                Patch(facecolor=facecolor, edgecolor='black', alpha=0.6,
                      label=f'{size}-gon: {count}')
            )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, title=title)


def visualize_comparison(ax_clique, ax_cycle, gng, iteration, max_cycle_size=12):
    """極大クリークと最小サイクルを並べて可視化"""
    nodes, edges, edges_per_node = get_graph_data(gng)

    if len(nodes) < 3:
        for ax in [ax_clique, ax_cycle]:
            ax.clear()
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
        return {}, {}, 0.0, 0.0

    # 極大クリーク方式
    result_clique = bron_kerbosch_pivot(edges_per_node, list(range(len(nodes))), min_size=3)
    clique_time = result_clique.elapsed_time

    # 最小サイクル方式
    result_cycle = detect_minimal_cycles(nodes, edges_per_node, max_cycle_size=max_cycle_size)
    cycle_time = result_cycle.elapsed_time

    # 左: 極大クリーク方式
    ax_clique.clear()
    ax_clique.set_xlim(-0.05, 1.05)
    ax_clique.set_ylim(-0.05, 1.05)
    ax_clique.set_aspect("equal")
    ax_clique.set_title(f"Maximal Cliques (n={result_clique.total_count()})\n{clique_time*1000:.2f} ms")
    draw_ring_outlines(ax_clique)
    draw_cliques(ax_clique, nodes, result_clique.cliques)
    draw_graph(ax_clique, nodes, edges)
    add_legend(ax_clique, result_clique.count_by_size(), "Clique Size")

    # 右: 最小サイクル方式
    ax_cycle.clear()
    ax_cycle.set_xlim(-0.05, 1.05)
    ax_cycle.set_ylim(-0.05, 1.05)
    ax_cycle.set_aspect("equal")
    ax_cycle.set_title(f"Minimal Cycles (n={result_cycle.total_count()})\n{cycle_time*1000:.2f} ms")
    draw_ring_outlines(ax_cycle)
    draw_cycles(ax_cycle, nodes, result_cycle.cycles)
    draw_graph(ax_cycle, nodes, edges)
    add_legend(ax_cycle, result_cycle.count_by_size(), "Cycle Size")

    return result_clique.count_by_size(), result_cycle.count_by_size(), clique_time, cycle_time


def main():
    print("=" * 70)
    print("AiS-GNG 最小サイクル検出テスト: 極大クリーク vs 最小サイクル")
    print("=" * 70)

    # 検出閾値（最大サイクルサイズ）
    max_cycle_size = 12
    print(f"\n検出閾値: max_cycle_size = {max_cycle_size}")
    print("  → {max_cycle_size}角形までを検出対象とする")

    params = AiSGNGParams(
        max_nodes=300,
        lambda_=100,
        kappa=10,
        eps_b=0.05,
        eps_n=0.005,
        alpha=0.5,
        beta=0.005,
        chi=0.005,
        max_age=50,
        utility_k=1000,
        theta_ais_min=0.015,
        theta_ais_max=0.06,
    )
    n_iterations = 20000
    n_samples = 1500
    seed = 42

    data = sample_triple_ring(n_samples=n_samples, seed=seed)
    gng = AiSGNG(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = 100

    print(f"\nTraining: {n_iterations} iterations")
    print(f"Parameters: max_nodes={params.max_nodes}, theta_ais=[{params.theta_ais_min}, {params.theta_ais_max}]")

    fig, (ax_clique, ax_cycle) = plt.subplots(1, 2, figsize=(14, 7))

    np.random.seed(seed)

    # 計算時間の累積
    total_clique_time = 0.0
    total_cycle_time = 0.0
    n_measurements = 0

    for i in range(n_iterations):
        idx = np.random.randint(0, n_samples)
        gng.partial_fit(data[idx])

        if i % frame_interval == 0 or i == n_iterations - 1:
            clique_counts, cycle_counts, clique_time, cycle_time = visualize_comparison(
                ax_clique, ax_cycle, gng, i, max_cycle_size
            )

            total_clique_time += clique_time
            total_cycle_time += cycle_time
            n_measurements += 1

            fig.suptitle(f"AiS-GNG: iter={i}, nodes={gng.n_nodes}", fontsize=14)

            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            frames.append(image.copy())

            if i % 1000 == 0:
                clique_str = ", ".join(f"K{k}:{v}" for k, v in sorted(clique_counts.items()))
                cycle_str = ", ".join(f"{k}:{v}" for k, v in sorted(cycle_counts.items()))
                print(f"  iter={i}: nodes={gng.n_nodes}")
                print(f"    Cliques: [{clique_str}] ({clique_time*1000:.2f} ms)")
                print(f"    Cycles:  [{cycle_str}] ({cycle_time*1000:.2f} ms)")

    print(f"\nTraining complete: {gng.n_nodes} nodes")

    # 最終統計
    nodes, edges, edges_per_node = get_graph_data(gng)
    result_clique = bron_kerbosch_pivot(edges_per_node, list(range(len(nodes))), min_size=3)
    result_cycle = detect_minimal_cycles(nodes, edges_per_node, max_cycle_size=max_cycle_size)

    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)

    print(f"\n【極大クリーク方式】")
    print(f"  検出総数: {result_clique.total_count()}")
    for size, count in sorted(result_clique.count_by_size().items()):
        print(f"    K{size} (完全{size}角形): {count}個")
    print(f"  計算時間: {result_clique.elapsed_time * 1000:.3f} ms")
    print(f"  平均計算時間: {total_clique_time / n_measurements * 1000:.3f} ms")

    print(f"\n【最小サイクル方式】")
    print(f"  検出総数: {result_cycle.total_count()}")
    max_detected = 0
    for size, count in sorted(result_cycle.count_by_size().items()):
        print(f"    {size}角形: {count}個")
        if count > 0:
            max_detected = size
    print(f"  最大検出: {max_detected}角形")
    print(f"  計算時間: {result_cycle.elapsed_time * 1000:.3f} ms")
    print(f"  平均計算時間: {total_cycle_time / n_measurements * 1000:.3f} ms")

    print(f"\n【検出閾値について】")
    print(f"  max_cycle_size = {max_cycle_size}")
    print(f"  この値より大きいサイクルは検出されません。")
    print(f"  大きな値にすると計算時間が増加します。")

    # 出力
    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    final_path = output_dir / "aisgng_minimal_cycles_final.png"
    fig.savefig(final_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {final_path}")

    gif_path = output_dir / "aisgng_minimal_cycles_growth.gif"
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
