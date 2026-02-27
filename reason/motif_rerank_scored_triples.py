#!/usr/bin/env python3
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set
import torch

Trip = Tuple[str, str, str, float]

def build_adj(trips: List[Trip]) -> Dict[str, Set[str]]:
    adj = defaultdict(set)
    for h, r, t, s in trips:
        if not h or not t or h == t:
            continue
        adj[h].add(t)
        adj[t].add(h)
    return adj

def triangle_edge_score(adj: Dict[str, Set[str]], u: str, v: str) -> int:
    nu = adj.get(u)
    nv = adj.get(v)
    if not nu or not nv:
        return 0
    if len(nu) > len(nv):
        nu, nv = nv, nu
    return sum(1 for x in nu if x in nv)

def eligible_nodes(adj: Dict[str, Set[str]], q: Set[str], hops: int) -> Set[str]:
    q = {x for x in q if x}
    if hops <= 0:
        return q
    out = set(q)
    frontier = set(q)
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            nxt.update(adj.get(u, ()))
        nxt -= out
        out |= nxt
        frontier = nxt
        if not frontier:
            break
    return out

def rerank_one(item: Dict[str, Any], topk_graph: int, lam: float, mode: str, anchor_hops: int) -> List[Trip]:
    scored_triples: List[Trip] = item.get("scored_triples", [])
    if not scored_triples:
        return scored_triples

    base = scored_triples[:topk_graph]
    adj = build_adj(base)

    qents = set(item.get("q_entity_in_graph", []) or [])
    elig = eligible_nodes(adj, qents, anchor_hops) if anchor_hops >= 0 else set()

    if mode == "triangles":
        vals = []
        for h, r, t, s in base:
            if not h or not t or h == t:
                continue
            if anchor_hops >= 0 and (h not in elig and t not in elig):
                continue
            vals.append(triangle_edge_score(adj, h, t))
        max_m = max(vals) if vals else 0

        def motif(h: str, t: str) -> float:
            if max_m <= 0:
                return 0.0
            if anchor_hops >= 0 and (h not in elig and t not in elig):
                return 0.0
            return triangle_edge_score(adj, h, t) / max_m

    elif mode == "wedge":
        deg = {u: len(nbrs) for u, nbrs in adj.items()}
        wedge = {u: (d * (d - 1) // 2 if d >= 2 else 0) for u, d in deg.items()}

        vals = []
        for h, r, t, s in base:
            if not h or not t or h == t:
                continue
            if anchor_hops >= 0 and (h not in elig and t not in elig):
                continue
            vals.append(wedge.get(h, 0) + wedge.get(t, 0))
        max_m = max(vals) if vals else 0

        def motif(h: str, t: str) -> float:
            if max_m <= 0:
                return 0.0
            if anchor_hops >= 0 and (h not in elig and t not in elig):
                return 0.0
            return (wedge.get(h, 0) + wedge.get(t, 0)) / max_m

    else:
        raise ValueError(f"unknown mode: {mode}")

    base2: List[Trip] = []
    for h, r, t, s in base:
        new_s = float(s) + lam * motif(h, t)
        base2.append((h, r, t, float(new_s)))
    base2.sort(key=lambda x: x[3], reverse=True)
    return base2 + scored_triples[topk_graph:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--topk_graph", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--mode", choices=["triangles", "wedge"], default="triangles")
    ap.add_argument("--anchor_hops", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    data: Dict[str, Dict[str, Any]] = torch.load(args.in_path, map_location="cpu")

    out: Dict[str, Dict[str, Any]] = {}
    for i, (qid, item) in enumerate(data.items()):
        if args.limit > 0 and i >= args.limit:
            break
        new_item = dict(item)
        new_item["scored_triples"] = rerank_one(
            new_item,
            topk_graph=args.topk_graph,
            lam=args.lam,
            mode=args.mode,
            anchor_hops=args.anchor_hops,
        )
        out[qid] = new_item
        if (i + 1) % 200 == 0:
            print(f"processed {i+1} questions")

    torch.save(out, args.out_path, _use_new_zipfile_serialization=False)
    print("saved:", args.out_path, "questions:", len(out))

if __name__ == "__main__":
    main()
