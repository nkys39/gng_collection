#!/usr/bin/env python3
"""BFS版 vs CCW版 最小サイクル検出の比較

AiS-GNGで2つのアルゴリズムを並べて可視化。
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
from experiments.polygon_filling.polygon_utils import detect_minimal_cycles


# Triple ring geometry
TRIPLE_RING_PARAMS = [
    (0.50, 0.23, 0.06, 0.14),
    (0.27, 0.68, 0.06, 0.14),
    (0.73, 0.68, 0.06, 0.14),
]

# サイズ別の色
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


def visualize_comparison(ax_bfs, ax_ccw, gng, iteration, max_cycle_size=12):
    """BFS版とCCW版を並べて可視化"""
    nodes, edges, edges_per_node = get_graph_data(gng)

    if len(nodes) < 3:
        for ax in [ax_bfs, ax_ccw]:
            ax.clear()
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
        return {}, {}, 0.0, 0.0

    # BFS版
    result_bfs = detect_minimal_cycles(nodes, edges_per_node, max_cycle_size=max_cycle_size, use_simple=True)
    bfs_time = result_bfs.elapsed_time

    # CCW版
    result_ccw = detect_minimal_cycles(nodes, edges_per_node, max_cycle_size=max_cycle_size, use_simple=False)
    ccw_time = result_ccw.elapsed_time

    # 左: BFS版
    ax_bfs.clear()
    ax_bfs.set_xlim(-0.05, 1.05)
    ax_bfs.set_ylim(-0.05, 1.05)
    ax_bfs.set_aspect("equal")
    ax_bfs.set_title(f"BFS (n={result_bfs.total_count()})\n{bfs_time*1000:.2f} ms")
    draw_ring_outlines(ax_bfs)
    draw_cycles(ax_bfs, nodes, result_bfs.cycles)
    draw_graph(ax_bfs, nodes, edges)
    add_legend(ax_bfs, result_bfs.count_by_size(), "BFS")

    # 右: CCW版
    ax_ccw.clear()
    ax_ccw.set_xlim(-0.05, 1.05)
    ax_ccw.set_ylim(-0.05, 1.05)
    ax_ccw.set_aspect("equal")
    ax_ccw.set_title(f"CCW (n={result_ccw.total_count()})\n{ccw_time*1000:.2f} ms")
    draw_ring_outlines(ax_ccw)
    draw_cycles(ax_ccw, nodes, result_ccw.cycles)
    draw_graph(ax_ccw, nodes, edges)
    add_legend(ax_ccw, result_ccw.count_by_size(), "CCW")

    return result_bfs.count_by_size(), result_ccw.count_by_size(), bfs_time, ccw_time


def main():
    print("=" * 70)
    print("BFS版 vs CCW版 最小サイクル検出比較（AiS-GNG 300ノード）")
    print("=" * 70)

    max_cycle_size = 12

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

    print(f"\nTraining: {n_iterations} iterations, max_nodes={params.max_nodes}")

    fig, (ax_bfs, ax_ccw) = plt.subplots(1, 2, figsize=(14, 7))

    np.random.seed(seed)

    total_bfs_time = 0.0
    total_ccw_time = 0.0
    n_measurements = 0

    for i in range(n_iterations):
        idx = np.random.randint(0, n_samples)
        gng.partial_fit(data[idx])

        if i % frame_interval == 0 or i == n_iterations - 1:
            bfs_counts, ccw_counts, bfs_time, ccw_time = visualize_comparison(
                ax_bfs, ax_ccw, gng, i, max_cycle_size
            )

            total_bfs_time += bfs_time
            total_ccw_time += ccw_time
            n_measurements += 1

            fig.suptitle(f"AiS-GNG: iter={i}, nodes={gng.n_nodes}", fontsize=14)

            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            frames.append(image.copy())

            if i % 2000 == 0:
                bfs_str = ", ".join(f"{k}:{v}" for k, v in sorted(bfs_counts.items()))
                ccw_str = ", ".join(f"{k}:{v}" for k, v in sorted(ccw_counts.items()))
                print(f"  iter={i}: nodes={gng.n_nodes}")
                print(f"    BFS: [{bfs_str}] ({bfs_time*1000:.2f} ms)")
                print(f"    CCW: [{ccw_str}] ({ccw_time*1000:.2f} ms)")

    print(f"\nTraining complete: {gng.n_nodes} nodes")

    # 最終統計
    nodes, edges, edges_per_node = get_graph_data(gng)
    result_bfs = detect_minimal_cycles(nodes, edges_per_node, max_cycle_size=max_cycle_size, use_simple=True)
    result_ccw = detect_minimal_cycles(nodes, edges_per_node, max_cycle_size=max_cycle_size, use_simple=False)

    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)

    print(f"\n【BFS版（簡易版）】")
    print(f"  検出総数: {result_bfs.total_count()}")
    for size, count in sorted(result_bfs.count_by_size().items()):
        print(f"    {size}角形: {count}個")
    print(f"  計算時間: {result_bfs.elapsed_time * 1000:.3f} ms")
    print(f"  平均計算時間: {total_bfs_time / n_measurements * 1000:.3f} ms")

    print(f"\n【CCW版（反時計回り探索）】")
    print(f"  検出総数: {result_ccw.total_count()}")
    for size, count in sorted(result_ccw.count_by_size().items()):
        print(f"    {size}角形: {count}個")
    print(f"  計算時間: {result_ccw.elapsed_time * 1000:.3f} ms")
    print(f"  平均計算時間: {total_ccw_time / n_measurements * 1000:.3f} ms")

    print(f"\n【比較】")
    print(f"  検出数差: CCW版が {result_ccw.total_count() - result_bfs.total_count()} 個多い")
    print(f"  速度比: BFS版は CCW版の {total_ccw_time / total_bfs_time:.2f} 倍高速")

    # 出力
    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    final_path = output_dir / "bfs_vs_ccw_final.png"
    fig.savefig(final_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {final_path}")

    gif_path = output_dir / "bfs_vs_ccw_growth.gif"
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
