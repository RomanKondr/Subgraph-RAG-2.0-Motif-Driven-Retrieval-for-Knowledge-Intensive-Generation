#!/usr/bin/env python3
import argparse, json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import torch

def norm_ent(x): return str(x)

def undirected_key(a,b):
    a=norm_ent(a); b=norm_ent(b)
    return (a,b) if a<=b else (b,a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="retrieve/data_files/webqsp/gpt_triples.pth")
    ap.add_argument("--out_dir", default="reason/motif_index")
    ap.add_argument("--max_per_anchor", type=int, default=2000)
    ap.add_argument("--max_deg", type=int, default=300, help="cap neighbor list to avoid blowup")
    ap.add_argument("--modes", default="tri,wedge")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.in_path, map_location="cpu")  # dict qid -> list[(h,r,t)]

    # Pool triples globally
    triples = []
    for qid, lst in data.items():
        for h,r,t in lst:
            h=norm_ent(h); r=str(r); t=norm_ent(t)
            if h and t and r:
                triples.append((h,r,t))

    # Build undirected adjacency + pair->directed triples
    nbrs = defaultdict(set)                     # node -> set(neighbors)
    pair2trip = defaultdict(list)               # (u,v undirected) -> list of directed triples
    for h,r,t in triples:
        nbrs[h].add(t); nbrs[t].add(h)
        pair2trip[undirected_key(h,t)].append((h,r,t))

    # neighbor cap for motif extraction
    nbrs_list = {u: list(ns)[:args.max_deg] for u,ns in nbrs.items()}

    want_tri = "tri" in args.modes
    want_wedge = "wedge" in args.modes

    tokens = defaultdict(list)

    # WEDGE: (a - u - b), u is center/anchor
    if want_wedge:
        for u, ns in nbrs_list.items():
            if len(ns) < 2: 
                continue
            for a,b in combinations(ns, 2):
                if len(tokens[u]) >= args.max_per_anchor:
                    break
                tokens[u].append({"type":"WEDGE","u":u,"a":a,"b":b})

    # TRI: triangle (u,v,w) where v,w in N(u) and edge(v,w) exists
    if want_tri:
        for u, ns in nbrs_list.items():
            if len(ns) < 2:
                continue
            ns_set = set(ns)
            for v in ns:
                # candidates w are neighbors of v that also lie in N(u)
                for w in nbrs[v].intersection(ns_set):
                    if v >= w:
                        continue
                    if len(tokens[u]) >= args.max_per_anchor:
                        break
                    # ensure edge(v,w) exists in pooled graph (it does if pair2trip has it)
                    if undirected_key(v,w) in pair2trip:
                        tokens[u].append({"type":"TRI","u":u,"v":v,"w":w})

    # Write token index: anchor -> tokens
    idx_path = out_dir / "webqsp_tokens_global.jsonl"
    with idx_path.open("w", encoding="utf-8") as f:
        for anchor, lst in tokens.items():
            f.write(json.dumps({"anchor":anchor,"tokens":lst}) + "\n")

    # Write pair2trip map for expansion (jsonl shards)
    # (keep as jsonl so it's streamable and doesn't explode RAM in retrieval)
    pair_path = out_dir / "webqsp_pair2trip.jsonl"
    with pair_path.open("w", encoding="utf-8") as f:
        for (u,v), lst in pair2trip.items():
            f.write(json.dumps({"u":u,"v":v,"triples":lst}) + "\n")

    meta = {
        "n_questions": len(data),
        "n_pooled_triples": len(triples),
        "n_anchors": len(tokens),
        "max_per_anchor": args.max_per_anchor,
        "max_deg": args.max_deg,
        "modes": args.modes,
        "idx_path": str(idx_path),
        "pair2trip_path": str(pair_path),
    }
    (out_dir / "webqsp_motif_index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("WROTE:")
    print(" ", idx_path, "anchors:", len(tokens))
    print(" ", pair_path, "pairs:", len(pair2trip))
    print(" ", out_dir / "webqsp_motif_index_meta.json")

if __name__ == "__main__":
    main()
