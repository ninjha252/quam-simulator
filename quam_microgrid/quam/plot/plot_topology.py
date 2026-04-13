#!/usr/bin/env python3
"""
plot_topology.py - Draw simple topology diagrams for QuAM runs.

Usage:
  python3 quam/plot_topology.py --topology ring --nodes MG0 MG1 MG2 --output outputs/topology_ring.png
"""

from __future__ import annotations

import argparse
import random
import math
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from .network import build_topology


def _positions_for_topology(name: str, nodes: List[str], g: nx.Graph, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    if name in ("ring", "star"):
        return nx.circular_layout(g)
    if name == "mesh":
        return nx.spring_layout(g, seed=seed)
    if name == "two_cluster_bridge":
        mid = len(nodes) // 2
        a = nodes[:mid]
        b = nodes[mid:]
        pos: Dict[str, Tuple[float, float]] = {}
        # cluster A around (-1, 0)
        for i, n in enumerate(a):
            angle = 2.0 * math.pi * (i / max(1, len(a)))
            pos[n] = (-1.0 + 0.5 * math.cos(angle), 0.5 * math.sin(angle))
        # cluster B around (1, 0)
        for i, n in enumerate(b):
            angle = 2.0 * math.pi * (i / max(1, len(b)))
            pos[n] = (1.0 + 0.5 * math.cos(angle), 0.5 * math.sin(angle))
        return pos
    return nx.spring_layout(g, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot QuAM topology diagram")
    parser.add_argument("--topology", required=True, help="ring, star, mesh, two_cluster_bridge")
    parser.add_argument("--nodes", nargs="+", required=True, help="Node names")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--seed", type=int, default=42, help="Layout seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    g = build_topology(args.topology, args.nodes, rng)
    pos = _positions_for_topology(args.topology, args.nodes, g, seed=args.seed)

    # Draw non-edges in red (faint) and actual edges in green for clarity.
    all_pairs = [(args.nodes[i], args.nodes[j]) for i in range(len(args.nodes)) for j in range(i + 1, len(args.nodes))]
    all_g = nx.Graph()
    all_g.add_nodes_from(args.nodes)
    all_g.add_edges_from(all_pairs)
    real_edges = {tuple(sorted(e)) for e in g.edges()}
    non_edges = [e for e in all_pairs if tuple(sorted(e)) not in real_edges]

    plt.figure(figsize=(6, 4))
    if non_edges:
        nx.draw_networkx_edges(all_g, pos, edgelist=non_edges, width=1.0, edge_color="#d62728", alpha=0.2)
    nx.draw_networkx_edges(g, pos, width=2.2, edge_color="#2ca02c", alpha=0.9)
    nx.draw_networkx_nodes(g, pos, node_size=900, node_color="#6baed6", edgecolors="#1f77b4")
    nx.draw_networkx_labels(g, pos, font_size=10, font_color="#111111")
    plt.title(f"Topology: {args.topology} (green=connected, red=not connected)")
    legend_handles = [
        Line2D([0], [0], color="#2ca02c", lw=2.2, label="Connected"),
        Line2D([0], [0], color="#d62728", lw=1.2, alpha=0.6, label="Not connected"),
    ]
    plt.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
