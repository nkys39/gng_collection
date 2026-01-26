#!/usr/bin/env python3
"""最小サイクル検出テスト

極大クリーク方式と最小サイクル方式の比較。
最小サイクル方式は「エッジで囲まれた全ての領域」を検出できる。
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
    detect_minimal_cycles,
)


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


def add_legend(ax, count_by_size: dict[int, int]):
    from matplotlib.patches import Patch
    legend_elements = []
    for size in sorted(count_by_size.keys()):
        count = count_by_size[size]
        if count > 0:
            facecolor, _ = CYCLE_COLORS.get(size, ("#CCCCCC", "gray"))
            legend_elements.append(
                Patch(facecolor=facecolor, edgecolor='black', alpha=0.6,
                      label=f'{size}-cycle: {count}')
            )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def visualize_comparison(ax_clique, ax_cycle, gng, iteration):
    """極大クリークと最小サイクルを並べて可視化"""
    nodes, edges, edges_per_node = get_graph_data(gng)

    if len(nodes) < 3:
        for ax in [ax_clique, ax_cycle]:
            ax.clear()
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
        return {}, {}

    # 極大クリーク方式
    result_clique = bron_kerbosch_pivot(edges_per_node, list(range(len(nodes))), min_size=3)
    triangles_clique = decompose_to_triangles(result_clique.cliques)

    # 最小サイクル方式
    result_cycle = detect_minimal_cycles(nodes, edges_per_node, max_cycle_size=8)

    # 左: 極大クリーク方式（三角形のみ）
    ax_clique.clear()
    ax_clique.set_xlim(-0.05, 1.05)
    ax_clique.set_ylim(-0.05, 1.05)
    ax_clique.set_aspect("equal")
    ax_clique.set_title(f"Clique-based (n={len(triangles_clique)})")
    draw_ring_outlines(ax_clique)
    # クリークベースは三角形のみ
    draw_cycles(ax_clique, nodes, [list(t) for t in triangles_clique])
    draw_graph(ax_clique, nodes, edges)
    add_legend(ax_clique, {3: len(triangles_clique)})

    # 右: 最小サイクル方式（全サイズ）
    ax_cycle.clear()
    ax_cycle.set_xlim(-0.05, 1.05)
    ax_cycle.set_ylim(-0.05, 1.05)
    ax_cycle.set_aspect("equal")
    ax_cycle.set_title(f"Minimal Cycles (n={result_cycle.total_count()})")
    draw_ring_outlines(ax_cycle)
    draw_cycles(ax_cycle, nodes, result_cycle.cycles)
    draw_graph(ax_cycle, nodes, edges)
    add_legend(ax_cycle, result_cycle.count_by_size())

    return result_clique.count_by_size(), result_cycle.count_by_size()


def main():
    print("=" * 60)
    print("最小サイクル検出テスト: 極大クリーク vs 最小サイクル")
    print("=" * 60)

    params = GNGParams(
        max_nodes=80,
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

    data = sample_triple_ring(n_samples=n_samples, seed=seed)
    gng = GrowingNeuralGas(n_dim=2, params=params, seed=seed)

    frames = []
    frame_interval = 100

    print(f"\nTraining: {n_iterations} iterations")

    fig, (ax_clique, ax_cycle) = plt.subplots(1, 2, figsize=(14, 7))

    np.random.seed(seed)

    for i in range(n_iterations):
        idx = np.random.randint(0, n_samples)
        gng.partial_fit(data[idx])

        if i % frame_interval == 0 or i == n_iterations - 1:
            clique_counts, cycle_counts = visualize_comparison(ax_clique, ax_cycle, gng, i)

            fig.suptitle(f"iter={i}, nodes={gng.n_nodes}", fontsize=14)

            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            frames.append(image.copy())

            if i % 500 == 0:
                clique_str = ", ".join(f"{k}:{v}" for k, v in sorted(clique_counts.items()))
                cycle_str = ", ".join(f"{k}:{v}" for k, v in sorted(cycle_counts.items()))
                print(f"  iter={i}: nodes={gng.n_nodes}")
                print(f"    Clique: [{clique_str}]")
                print(f"    Cycle:  [{cycle_str}]")

    print(f"\nTraining complete: {gng.n_nodes} nodes")

    # 最終統計
    nodes, edges, edges_per_node = get_graph_data(gng)
    result_clique = bron_kerbosch_pivot(edges_per_node, list(range(len(nodes))), min_size=3)
    result_cycle = detect_minimal_cycles(nodes, edges_per_node, max_cycle_size=8)

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"\nClique-based (triangles only):")
    print(f"  Total: {len(decompose_to_triangles(result_clique.cliques))}")
    print(f"  Time: {result_clique.elapsed_time * 1000:.3f} ms")

    print(f"\nMinimal Cycles (all faces):")
    print(f"  Total: {result_cycle.total_count()}")
    for size, count in sorted(result_cycle.count_by_size().items()):
        print(f"    {size}-cycle: {count}")
    print(f"  Time: {result_cycle.elapsed_time * 1000:.3f} ms")

    # 出力
    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    final_path = output_dir / "minimal_cycles_final.png"
    fig.savefig(final_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {final_path}")

    gif_path = output_dir / "minimal_cycles_growth.gif"
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
