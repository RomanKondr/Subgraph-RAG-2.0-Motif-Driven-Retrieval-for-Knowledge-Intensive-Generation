from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
from collections import defaultdict

Triplet3 = Tuple[str, str, str]
Triplet4 = Tuple[str, str, str, float]

@dataclass
class MotifUnit:
    motif_id: str
    motif_type: str              # "wedge" | "triangle"
    nodes: Tuple[str, str, str]  # always 3 nodes
    edges: List[Triplet4]        # list of (h,r,t,score)
    score: float                 # rank score
    meta: Dict                   # extras (anchor hits, etc.)

def _as_triplet4(t) -> Triplet4:
    if isinstance(t, (list, tuple)):
        if len(t) >= 4:
            h, r, ta, sc = t[0], t[1], t[2], float(t[3])
            return str(h), str(r), str(ta), float(sc)
        if len(t) == 3:
            h, r, ta = t
            return str(h), str(r), str(ta), 0.0
    if isinstance(t, dict):
        h = str(t.get("h") or t.get("head") or t.get("subj") or "")
        r = str(t.get("r") or t.get("rel")  or t.get("pred") or "")
        ta = str(t.get("t") or t.get("tail") or t.get("obj")  or "")
        sc = float(t.get("score", 0.0))
        return h, r, ta, sc
    raise TypeError(f"Unsupported triplet format: {type(t)}")

def _pairkey(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)

def _edgekey(e: Triplet4) -> str:
    return f"{e[0]}|{e[1]}|{e[2]}"

def build_candidate_graph(scored_triplets: Sequence, max_edges: int = 300) -> List[Triplet4]:
    out: List[Triplet4] = []
    for t in scored_triplets[:max_edges]:
        out.append(_as_triplet4(t))
    return out

def enumerate_wedges(edges: Sequence[Triplet4], anchor: Set[str], max_edges_per_node: int = 10) -> List[MotifUnit]:
    inc: Dict[str, List[Triplet4]] = defaultdict(list)
    for e in edges:
        h, _, t, _ = e
        inc[h].append(e)
        inc[t].append(e)

    for n in list(inc.keys()):
        inc[n].sort(key=lambda x: x[3], reverse=True)
        inc[n] = inc[n][:max_edges_per_node]

    motifs: List[MotifUnit] = []
    for center, elist in inc.items():
        if len(elist) < 2:
            continue
        for i in range(len(elist)):
            for j in range(i + 1, len(elist)):
                e1, e2 = elist[i], elist[j]

                def other(edge: Triplet4, c: str) -> str:
                    return edge[2] if edge[0] == c else edge[0]

                a = other(e1, center)
                b = center
                c = other(e2, center)
                if a == b or b == c or a == c:
                    continue
                nodes = (a, b, c)

                anchor_hits = sum(1 for n in nodes if n in anchor)
                if anchor and anchor_hits == 0:
                    continue

                s_max = max(e1[3], e2[3])
                s_min = min(e1[3], e2[3])
                spread = s_max - s_min
                score = s_max + (0.15 * anchor_hits) + (0.25 * spread)

                motif_id = f"wedge:{a}|{b}|{c}:{_edgekey(e1)}+{_edgekey(e2)}"
                motifs.append(
                    MotifUnit(
                        motif_id=motif_id,
                        motif_type="wedge",
                        nodes=nodes,
                        edges=sorted([e1, e2], key=lambda x: x[3], reverse=True),
                        score=score,
                        meta={"center": center, "anchor_hits": anchor_hits, "s_max": s_max, "spread": spread},
                    )
                )
    motifs.sort(key=lambda m: m.score, reverse=True)
    return motifs

def enumerate_triangles(edges: Sequence[Triplet4], anchor: Set[str]) -> List[MotifUnit]:
    best_pair: Dict[Tuple[str, str], Triplet4] = {}
    neigh: Dict[str, Set[str]] = defaultdict(set)

    for e in edges:
        h, _, t, _ = e
        if not h or not t:
            continue
        pk = _pairkey(h, t)
        if pk not in best_pair or e[3] > best_pair[pk][3]:
            best_pair[pk] = e
        neigh[h].add(t)
        neigh[t].add(h)

    nodes = list(neigh.keys())
    deg = {u: len(neigh[u]) for u in nodes}
    ordered = sorted(nodes, key=lambda u: (deg[u], u))
    rank = {u: i for i, u in enumerate(ordered)}

    forward: Dict[str, Set[str]] = defaultdict(set)
    for u in nodes:
        for v in neigh[u]:
            if rank[u] < rank[v]:
                forward[u].add(v)

    motifs: List[MotifUnit] = []
    for u in nodes:
        for v in forward[u]:
            common = forward[u].intersection(forward[v])
            for w in common:
                tri_nodes = tuple(sorted([u, v, w]))
                a, b, c = tri_nodes

                anchor_hits = sum(1 for n in tri_nodes if n in anchor)
                if anchor and anchor_hits == 0:
                    continue

                e_ab = best_pair.get(_pairkey(a, b))
                e_ac = best_pair.get(_pairkey(a, c))
                e_bc = best_pair.get(_pairkey(b, c))
                if not (e_ab and e_ac and e_bc):
                    continue

                s_max = max(e_ab[3], e_ac[3], e_bc[3])
                s_min = min(e_ab[3], e_ac[3], e_bc[3])
                spread = s_max - s_min
                score = s_max + (0.20 * anchor_hits) + (0.25 * spread) + 0.10

                motif_id = f"tri:{a}|{b}|{c}"
                motifs.append(
                    MotifUnit(
                        motif_id=motif_id,
                        motif_type="triangle",
                        nodes=(a, b, c),
                        edges=sorted([e_ab, e_ac, e_bc], key=lambda x: x[3], reverse=True),
                        score=score,
                        meta={"anchor_hits": anchor_hits, "s_max": s_max, "spread": spread},
                    )
                )
    motifs.sort(key=lambda m: m.score, reverse=True)
    return motifs

def build_motif_units(
    scored_triplets: Sequence,
    anchor_entities: Optional[Iterable[str]] = None,
    motif_types: Sequence[str] = ("wedge", "triangle"),
    candidate_edges: int = 300,
    max_edges_per_node: int = 10,
    max_units: int = 2000,
) -> List[MotifUnit]:
    anchor = set(anchor_entities or [])
    edges = build_candidate_graph(scored_triplets, max_edges=candidate_edges)

    motifs: List[MotifUnit] = []
    if "wedge" in motif_types:
        motifs.extend(enumerate_wedges(edges, anchor=anchor, max_edges_per_node=max_edges_per_node))
    if "triangle" in motif_types:
        motifs.extend(enumerate_triangles(edges, anchor=anchor))

    motifs.sort(key=lambda m: m.score, reverse=True)
    return motifs[:max_units]

def select_and_expand_motifs(
    motifs: Sequence[MotifUnit],
    unit_topk: int = 50,
    edge_budget: int = 35,
    edges_per_unit_cap: int = 3,
) -> Tuple[List[MotifUnit], List[Triplet4]]:
    picked_units: List[MotifUnit] = list(motifs[:unit_topk])
    out_edges: List[Triplet4] = []
    seen: Set[str] = set()

    for m in picked_units:
        for e in m.edges[:edges_per_unit_cap]:
            k = _edgekey(e)
            if k in seen:
                continue
            seen.add(k)
            out_edges.append(e)
            if len(out_edges) >= edge_budget:
                return picked_units, out_edges

    return picked_units, out_edges
