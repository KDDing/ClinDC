"""PPI network helpers for ClinDC's essential-hypertension case study."""

from __future__ import annotations

import collections
import gzip
from pathlib import Path

import numpy as np


def load_independent_ppi(path: Path, min_score: float):
    graph: dict[str, set[str]] = collections.defaultdict(set)
    input_edges = high_confidence = 0
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, mode="rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            input_edges += 1
            try:
                score = float(fields[4])
            except ValueError:
                continue
            if score < min_score:
                continue
            high_confidence += 1
            a, b = fields[1], fields[3]
            if not a.isdigit() or not b.isdigit() or a == b:
                continue
            graph[a].add(b)
            graph[b].add(a)
    retained_edges = sum(map(len, graph.values())) // 2
    return graph, {
        "hippie_input_rows": input_edges,
        "hippie_score_ge_threshold_rows": high_confidence,
        "retained_unique_edges": retained_edges,
        "retained_nodes": len(graph),
    }


def multisource_distances(graph: dict[str, set[str]], sources: set[str]):
    distances = {node: 0 for node in sources if node in graph}
    queue = collections.deque(distances)
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, ()):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def degree_bins(graph: dict[str, set[str]], bins: int = 10):
    nodes = np.asarray(list(graph), dtype=object)
    degrees = np.asarray([len(graph[x]) for x in nodes], dtype=float)
    cuts = np.unique(np.quantile(np.log1p(degrees), np.linspace(0, 1, bins + 1)))
    assignments = np.digitize(np.log1p(degrees), cuts[1:-1], right=True)
    pools: dict[int, list[str]] = collections.defaultdict(list)
    node_bin = {}
    for node, assignment in zip(nodes.tolist(), assignments.tolist()):
        pools[assignment].append(node)
        node_bin[node] = assignment
    return node_bin, pools
