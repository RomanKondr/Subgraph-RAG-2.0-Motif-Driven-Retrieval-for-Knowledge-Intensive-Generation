#!/usr/bin/env python3
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import torch

Trip = Tuple[str, str, str, float]

def comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0

def motif_rerank_one(scored_triples: List[Trip], topk_graph: int, lam: float) -> List[Trip]:
    """
    v0: wedge-based motif proxy.
    scored_triples items are (head, relation, tail, score).
    """
    if not scored_triples:
        return scored_triples

    base = scored_triples[:topk_graph]

    # Undirected adjacency built from top-K edges
    adj = defaultdict(set)
    for h, r, t, s in base:
        if not h or not t or h == t:
            continue
        adj[h].add(t)
        adj[t].add(h)

    deg = {u: len(nbrs) for u, nbrs in adj.items()}
    wedge = {u: comb2(d) for u, d in deg.items()}

    def edge_motif(h: str, t: str) -> int:
        return wedge.get(h, 0) + wedge.get(t, 0)

    motif_vals = [edge_motif(h, t) for (h, r, t, s) in base]
    max_m = max(motif_vals) if motif_vals else 0

    def norm_m(x: int) -> float:
        return (x / max_m) if max_m > 0 else 0.0

    reranked: List[Trip] = []
    for h, r, t, s in scored_triples:
        m = norm_m(edge_motif(h, t))
        new_s = float(s) + lam * m
        reranked.append((h, r, t, float(new_s)))

    reranked.sort(key=lambda x: x[3], reverse=True)
    return reranked

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Input .pth scored_triples file")
    ap.add_argument("--out_path", required=True, help="Output .pth file")
    ap.add_argument("--topk_graph", type=int, default=300, help="Top-K edges to define motif stats")
    ap.add_argument("--lam", type=float, default=0.15, help="Motif bonus weight")
    ap.add_argument("--limit", type=int, default=-1, help="Limit number of questions (debug)")
    args = ap.parse_args()

    data: Dict[str, Dict[str, Any]] = torch.load(args.in_path, map_location="cpu")

    out: Dict[str, Dict[str, Any]] = {}
    for i, (qid, item) in enumerate(data.items()):
        if args.limit > 0 and i >= args.limit:
            break
        new_item = dict(item)
        st = item.get("scored_triples", [])
        new_item["scored_triples"] = motif_rerank_one(st, topk_graph=args.topk_graph, lam=args.lam)
        out[qid] = new_item
        if (i + 1) % 200 == 0:
            print(f"processed {i+1} questions")

    torch.save(out, args.out_path)
    print("saved:", args.out_path, "questions:", len(out))

if __name__ == "__main__":
    main()
