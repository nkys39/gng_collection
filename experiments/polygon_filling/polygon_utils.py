"""多角形検出ユーティリティ（実験用）

極大クリーク検出とメッシュ三角形分割のためのユーティリティ。

アルゴリズム:
- Bron-Kerbosch法（基本版）: 全極大クリークを列挙
- Bron-Kerbosch法（Pivot最適化版）: 枝刈りで高速化
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CliqueResult:
    """クリーク検出結果"""
    cliques: list[tuple[int, ...]]
    elapsed_time: float  # 秒

    def by_size(self) -> dict[int, list[tuple[int, ...]]]:
        """サイズ別に分類"""
        result: dict[int, list[tuple[int, ...]]] = {}
        for clique in self.cliques:
            size = len(clique)
            if size not in result:
                result[size] = []
            result[size].append(clique)
        return result

    def count_by_size(self) -> dict[int, int]:
        """サイズ別カウント"""
        return {size: len(cliques) for size, cliques in self.by_size().items()}

    def total_count(self) -> int:
        """総数"""
        return len(self.cliques)


def bron_kerbosch_basic(
    edges_per_node: dict[int, set[int]],
    active_node_ids: list[int],
    min_size: int = 3,
) -> CliqueResult:
    """Bron-Kerbosch法（基本版）で極大クリークを検出

    Args:
        edges_per_node: ノードIDから隣接ノードIDの集合へのマップ
        active_node_ids: アクティブなノードIDのリスト
        min_size: 最小クリークサイズ（デフォルト3）

    Returns:
        CliqueResult: 検出されたクリークと実行時間
    """
    results: list[tuple[int, ...]] = []

    def recurse(R: set[int], P: set[int], X: set[int]) -> None:
        if not P and not X:
            if len(R) >= min_size:
                results.append(tuple(sorted(R)))
            return

        for v in list(P):
            neighbors_v = edges_per_node.get(v, set())
            recurse(
                R | {v},
                P & neighbors_v,
                X & neighbors_v,
            )
            P = P - {v}
            X = X | {v}

    start_time = time.perf_counter()
    active_set = set(active_node_ids)
    recurse(set(), active_set, set())
    elapsed = time.perf_counter() - start_time

    return CliqueResult(cliques=results, elapsed_time=elapsed)


def bron_kerbosch_pivot(
    edges_per_node: dict[int, set[int]],
    active_node_ids: list[int],
    min_size: int = 3,
) -> CliqueResult:
    """Bron-Kerbosch法（Pivot最適化版）で極大クリークを検出

    Pivot選択により不要な再帰を枝刈りして高速化。

    Args:
        edges_per_node: ノードIDから隣接ノードIDの集合へのマップ
        active_node_ids: アクティブなノードIDのリスト
        min_size: 最小クリークサイズ（デフォルト3）

    Returns:
        CliqueResult: 検出されたクリークと実行時間
    """
    results: list[tuple[int, ...]] = []

    def recurse(R: set[int], P: set[int], X: set[int]) -> None:
        if not P and not X:
            if len(R) >= min_size:
                results.append(tuple(sorted(R)))
            return

        # Pivot選択: P∪Xの中で最も隣接数が多い頂点
        union_px = P | X
        if not union_px:
            return

        pivot = max(union_px, key=lambda v: len(edges_per_node.get(v, set()) & P))
        pivot_neighbors = edges_per_node.get(pivot, set())

        # Pivotの非隣接頂点のみ探索（枝刈り）
        for v in P - pivot_neighbors:
            neighbors_v = edges_per_node.get(v, set())
            recurse(
                R | {v},
                P & neighbors_v,
                X & neighbors_v,
            )
            P = P - {v}
            X = X | {v}

    start_time = time.perf_counter()
    active_set = set(active_node_ids)
    recurse(set(), active_set, set())
    elapsed = time.perf_counter() - start_time

    return CliqueResult(cliques=results, elapsed_time=elapsed)


def decompose_to_triangles(
    cliques: list[tuple[int, ...]],
) -> list[tuple[int, int, int]]:
    """極大クリークを三角形に分割（Fan Triangulation）

    Args:
        cliques: 極大クリークのリスト

    Returns:
        三角形（3頂点タプル）のリスト
    """
    triangles: list[tuple[int, int, int]] = []

    for clique in cliques:
        n = len(clique)
        if n < 3:
            continue
        elif n == 3:
            triangles.append((clique[0], clique[1], clique[2]))
        else:
            # Fan triangulation: 最初の頂点を中心に扇状に分割
            v0 = clique[0]
            for i in range(1, n - 1):
                triangles.append((v0, clique[i], clique[i + 1]))

    return triangles


def order_convex_hull(
    points: np.ndarray,
    vertex_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """頂点を凸包順に並べ替え（2D用）

    Args:
        points: 全頂点の座標配列 (n_nodes, 2)
        vertex_ids: 並べ替える頂点IDのタプル

    Returns:
        凸包順に並べ替えた頂点IDのタプル
    """
    if len(vertex_ids) <= 3:
        return vertex_ids

    # 頂点座標を取得
    coords = points[list(vertex_ids)]

    # 重心を計算
    centroid = coords.mean(axis=0)

    # 重心からの角度でソート
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)

    return tuple(vertex_ids[i] for i in sorted_indices)


def compare_algorithms(
    edges_per_node: dict[int, set[int]],
    active_node_ids: list[int],
    min_size: int = 3,
    n_runs: int = 5,
) -> dict:
    """基本版とPivot版の比較

    Args:
        edges_per_node: 隣接リスト
        active_node_ids: アクティブノードID
        min_size: 最小クリークサイズ
        n_runs: 計測回数（平均を取る）

    Returns:
        比較結果の辞書
    """
    # 複数回実行して平均時間を計測
    basic_times = []
    pivot_times = []

    for _ in range(n_runs):
        result_basic = bron_kerbosch_basic(edges_per_node, active_node_ids, min_size)
        result_pivot = bron_kerbosch_pivot(edges_per_node, active_node_ids, min_size)
        basic_times.append(result_basic.elapsed_time)
        pivot_times.append(result_pivot.elapsed_time)

    # 最後の結果を使用
    result_basic = bron_kerbosch_basic(edges_per_node, active_node_ids, min_size)
    result_pivot = bron_kerbosch_pivot(edges_per_node, active_node_ids, min_size)

    avg_basic = sum(basic_times) / len(basic_times)
    avg_pivot = sum(pivot_times) / len(pivot_times)

    return {
        "basic": {
            "cliques": result_basic.count_by_size(),
            "total": result_basic.total_count(),
            "avg_time_ms": avg_basic * 1000,
        },
        "pivot": {
            "cliques": result_pivot.count_by_size(),
            "total": result_pivot.total_count(),
            "avg_time_ms": avg_pivot * 1000,
        },
        "speedup": avg_basic / avg_pivot if avg_pivot > 0 else float("inf"),
        "same_result": (
            result_basic.count_by_size() == result_pivot.count_by_size()
        ),
    }


def print_comparison(comparison: dict) -> None:
    """比較結果を出力"""
    print("=" * 50)
    print("Bron-Kerbosch アルゴリズム比較")
    print("=" * 50)

    print("\n【基本版】")
    print(f"  検出数: {comparison['basic']['total']}個")
    for size, count in sorted(comparison['basic']['cliques'].items()):
        print(f"    K{size}: {count}個")
    print(f"  平均実行時間: {comparison['basic']['avg_time_ms']:.3f} ms")

    print("\n【Pivot最適化版】")
    print(f"  検出数: {comparison['pivot']['total']}個")
    for size, count in sorted(comparison['pivot']['cliques'].items()):
        print(f"    K{size}: {count}個")
    print(f"  平均実行時間: {comparison['pivot']['avg_time_ms']:.3f} ms")

    print("\n【比較結果】")
    print(f"  検出結果一致: {'Yes' if comparison['same_result'] else 'No'}")
    print(f"  速度向上率: {comparison['speedup']:.2f}x")
    print("=" * 50)


# =============================================================================
# エッジベースの三角形検出（GNGエッジで囲まれた領域を塗りつぶす）
# =============================================================================

def get_edge_based_triangles(
    nodes: np.ndarray,
    edges_per_node: dict[int, set[int]],
) -> list[tuple[int, int, int]]:
    """GNGエッジで囲まれた三角形を検出

    Delaunay三角形分割を行い、3辺全てがGNGエッジとして存在する
    三角形のみを抽出する。

    Args:
        nodes: ノード座標 (n_nodes, 2)
        edges_per_node: 隣接リスト

    Returns:
        三角形のリスト [(a, b, c), ...]
    """
    from scipy.spatial import Delaunay

    if len(nodes) < 3:
        return []

    # エッジセットを作成（高速検索用）
    edge_set: set[tuple[int, int]] = set()
    for node_id, neighbors in edges_per_node.items():
        for neighbor_id in neighbors:
            edge = tuple(sorted([node_id, neighbor_id]))
            edge_set.add(edge)

    # Delaunay三角形分割
    try:
        tri = Delaunay(nodes)
    except Exception:
        return []

    # GNGエッジで囲まれた三角形のみを抽出
    valid_triangles: list[tuple[int, int, int]] = []
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])

        # 3辺全てがGNGエッジとして存在するか確認
        edge_ab = tuple(sorted([a, b]))
        edge_bc = tuple(sorted([b, c]))
        edge_ca = tuple(sorted([c, a]))

        if edge_ab in edge_set and edge_bc in edge_set and edge_ca in edge_set:
            valid_triangles.append((a, b, c))

    return valid_triangles


def get_all_graph_triangles(
    edges_per_node: dict[int, set[int]],
) -> list[tuple[int, int, int]]:
    """グラフ内の全ての三角形（3サイクル）を検出

    Delaunayを使わず、純粋にグラフ構造から三角形を検出。
    クリーク検出と同じ結果になるが、K3のみを効率的に検出。

    Args:
        edges_per_node: 隣接リスト

    Returns:
        三角形のリスト [(a, b, c), ...]
    """
    triangles: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    for a in edges_per_node:
        neighbors_a = edges_per_node[a]
        for b in neighbors_a:
            if b <= a:
                continue
            neighbors_b = edges_per_node.get(b, set())
            # aとbの共通隣接ノードを探す
            common = neighbors_a & neighbors_b
            for c in common:
                if c <= b:
                    continue
                tri = tuple(sorted([a, b, c]))
                if tri not in seen:
                    seen.add(tri)
                    triangles.append((a, b, c))

    return triangles


@dataclass
class TriangleResult:
    """三角形検出結果"""
    triangles: list[tuple[int, int, int]]
    method: str  # "clique", "delaunay", "graph"
    elapsed_time: float

    def count(self) -> int:
        return len(self.triangles)


def compare_triangle_methods(
    nodes: np.ndarray,
    edges_per_node: dict[int, set[int]],
    n_runs: int = 5,
) -> dict:
    """三角形検出方法の比較

    Args:
        nodes: ノード座標
        edges_per_node: 隣接リスト
        n_runs: 計測回数

    Returns:
        比較結果
    """
    active_ids = list(edges_per_node.keys())

    # 方法1: 極大クリーク → 三角形分割
    clique_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = bron_kerbosch_pivot(edges_per_node, active_ids, min_size=3)
        triangles_clique = decompose_to_triangles(result.cliques)
        clique_times.append(time.perf_counter() - start)

    # 方法2: Delaunay + GNGエッジフィルタ（scipyが利用可能な場合のみ）
    triangles_delaunay = []
    delaunay_times = [0.0]
    try:
        for _ in range(n_runs):
            start = time.perf_counter()
            triangles_delaunay = get_edge_based_triangles(nodes, edges_per_node)
            delaunay_times.append(time.perf_counter() - start)
        delaunay_times = delaunay_times[1:]  # 最初の0.0を除去
    except ImportError:
        pass  # scipyがない場合はスキップ

    # 方法3: グラフ構造から直接検出
    graph_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        triangles_graph = get_all_graph_triangles(edges_per_node)
        graph_times.append(time.perf_counter() - start)

    return {
        "clique": {
            "count": len(triangles_clique),
            "avg_time_ms": sum(clique_times) / len(clique_times) * 1000,
            "triangles": triangles_clique,
        },
        "delaunay": {
            "count": len(triangles_delaunay),
            "avg_time_ms": sum(delaunay_times) / len(delaunay_times) * 1000 if delaunay_times else 0,
            "triangles": triangles_delaunay,
        },
        "graph": {
            "count": len(triangles_graph),
            "avg_time_ms": sum(graph_times) / len(graph_times) * 1000,
            "triangles": triangles_graph,
        },
    }


def print_triangle_comparison(comparison: dict) -> None:
    """三角形検出方法の比較結果を出力"""
    print("=" * 50)
    print("三角形検出方法の比較")
    print("=" * 50)

    print("\n【極大クリーク → 三角形分割】")
    print(f"  検出数: {comparison['clique']['count']}個")
    print(f"  実行時間: {comparison['clique']['avg_time_ms']:.3f} ms")

    print("\n【Delaunay + GNGエッジフィルタ】")
    print(f"  検出数: {comparison['delaunay']['count']}個")
    print(f"  実行時間: {comparison['delaunay']['avg_time_ms']:.3f} ms")

    print("\n【グラフ構造から直接検出】")
    print(f"  検出数: {comparison['graph']['count']}個")
    print(f"  実行時間: {comparison['graph']['avg_time_ms']:.3f} ms")

    print("=" * 50)


# =============================================================================
# 最小サイクル検出（平面グラフの面検出）
# =============================================================================

def get_minimal_cycles(
    nodes: np.ndarray,
    edges_per_node: dict[int, set[int]],
    max_cycle_size: int = 10,
) -> list[list[int]]:
    """平面グラフの最小サイクル（面）を検出

    各有向エッジから、反時計回りに次のエッジを辿って面を構成する。

    Args:
        nodes: ノード座標 (n_nodes, 2)
        edges_per_node: 隣接リスト
        max_cycle_size: 最大サイクルサイズ（無限ループ防止）

    Returns:
        サイクルのリスト [[v0, v1, v2, ...], ...]
    """
    if len(nodes) < 3:
        return []

    def angle_from(v_from: int, v_to: int) -> float:
        """v_fromからv_toへの角度を計算"""
        dx = nodes[v_to, 0] - nodes[v_from, 0]
        dy = nodes[v_to, 1] - nodes[v_from, 1]
        return np.arctan2(dy, dx)

    def next_edge_ccw(v_prev: int, v_curr: int) -> int:
        """v_prevからv_currに来た時、反時計回りで次のエッジの先を返す"""
        neighbors = edges_per_node.get(v_curr, set())
        if not neighbors:
            return -1

        # v_prevからv_currへの入射角度
        incoming_angle = angle_from(v_curr, v_prev)

        # 各隣接ノードへの角度を計算し、反時計回りで最も近いものを選ぶ
        best_next = -1
        best_angle_diff = float('inf')

        for neighbor in neighbors:
            if neighbor == v_prev and len(neighbors) > 1:
                continue  # 来た道は避ける（ただし唯一の隣接なら戻る）

            outgoing_angle = angle_from(v_curr, neighbor)
            # 反時計回りの角度差（0〜2π）
            angle_diff = (outgoing_angle - incoming_angle) % (2 * np.pi)
            if angle_diff < 0.0001:  # ほぼ同じ方向は避ける
                angle_diff = 2 * np.pi

            if angle_diff < best_angle_diff:
                best_angle_diff = angle_diff
                best_next = neighbor

        return best_next

    # 全ての有向エッジを列挙
    directed_edges: set[tuple[int, int]] = set()
    for u, neighbors in edges_per_node.items():
        for v in neighbors:
            directed_edges.add((u, v))

    # 既に使用した有向エッジを追跡
    used_edges: set[tuple[int, int]] = set()
    cycles: list[list[int]] = []

    for start_u, start_v in directed_edges:
        if (start_u, start_v) in used_edges:
            continue

        # このエッジから始まるサイクルを探索
        cycle = [start_u]
        prev_node = start_u
        curr_node = start_v

        while len(cycle) < max_cycle_size:
            cycle.append(curr_node)

            if curr_node == start_u:
                # サイクル完成
                break

            next_node = next_edge_ccw(prev_node, curr_node)
            if next_node == -1:
                break

            prev_node = curr_node
            curr_node = next_node

        # 有効なサイクルか確認
        if len(cycle) >= 3 and cycle[-1] == start_u:
            cycle = cycle[:-1]  # 最後の重複を除去

            # サイクルを正規化（最小頂点から開始）
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])

            # 逆方向のサイクルを避けるため、2番目の頂点でソート方向を決定
            if len(normalized) >= 2:
                # 既に追加済みか確認
                is_new = True
                for existing in cycles:
                    if set(existing) == set(normalized):
                        is_new = False
                        break

                if is_new:
                    cycles.append(list(normalized))

                    # 使用したエッジをマーク
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        used_edges.add((u, v))

    return cycles


def get_minimal_cycles_simple(
    nodes: np.ndarray,
    edges_per_node: dict[int, set[int]],
    max_cycle_size: int = 8,
) -> list[list[int]]:
    """簡易版: BFSで最小サイクルを検出

    各エッジについて、そのエッジを含む最小サイクルを探す。

    Args:
        nodes: ノード座標 (n_nodes, 2)
        edges_per_node: 隣接リスト
        max_cycle_size: 最大サイクルサイズ

    Returns:
        サイクルのリスト
    """
    from collections import deque

    cycles: list[tuple[int, ...]] = []
    seen_cycles: set[tuple[int, ...]] = set()

    # 全エッジを列挙
    all_edges: list[tuple[int, int]] = []
    for u, neighbors in edges_per_node.items():
        for v in neighbors:
            if u < v:
                all_edges.append((u, v))

    for edge_u, edge_v in all_edges:
        # edge_u -> edge_v を含む最小サイクルをBFSで探索
        # edge_vからedge_uへの最短パス（edge_u-edge_v以外のエッジを使用）を探す

        queue: deque[tuple[int, list[int]]] = deque()
        queue.append((edge_v, [edge_u, edge_v]))
        visited: set[int] = {edge_v}

        found = False
        while queue and not found:
            curr, path = queue.popleft()

            if len(path) > max_cycle_size:
                continue

            for neighbor in edges_per_node.get(curr, set()):
                if neighbor == edge_u and len(path) >= 3:
                    # サイクル発見
                    cycle = path
                    # 正規化
                    min_idx = cycle.index(min(cycle))
                    normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                    # 逆順も確認
                    reversed_norm = tuple(reversed(normalized))
                    reversed_norm = tuple(
                        list(reversed_norm)[-1:] + list(reversed_norm)[:-1]
                    )

                    if normalized not in seen_cycles and reversed_norm not in seen_cycles:
                        cycles.append(normalized)
                        seen_cycles.add(normalized)
                    found = True
                    break

                if neighbor not in visited and neighbor != edge_u:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return [list(c) for c in cycles]


@dataclass
class CycleResult:
    """サイクル検出結果"""
    cycles: list[list[int]]
    elapsed_time: float

    def by_size(self) -> dict[int, list[list[int]]]:
        """サイズ別に分類"""
        result: dict[int, list[list[int]]] = {}
        for cycle in self.cycles:
            size = len(cycle)
            if size not in result:
                result[size] = []
            result[size].append(cycle)
        return result

    def count_by_size(self) -> dict[int, int]:
        """サイズ別カウント"""
        return {size: len(cycles) for size, cycles in self.by_size().items()}

    def total_count(self) -> int:
        return len(self.cycles)


def detect_minimal_cycles(
    nodes: np.ndarray,
    edges_per_node: dict[int, set[int]],
    max_cycle_size: int = 8,
) -> CycleResult:
    """最小サイクルを検出（時間計測付き）"""
    start = time.perf_counter()
    cycles = get_minimal_cycles_simple(nodes, edges_per_node, max_cycle_size)
    elapsed = time.perf_counter() - start
    return CycleResult(cycles=cycles, elapsed_time=elapsed)
