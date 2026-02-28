from collections import defaultdict
from itertools import combinations

def undirected_key(a, b):
    a = str(a); b = str(b)
    return (a, b) if a <= b else (b, a)

def local_motif_expand(scored_triplets, k_base=300, k_anchor=50, max_added=300):
    """
    scored_triplets: list of (h,r,t,score) sorted desc by score
    Returns: list of extra triples (h,r,t,score) to augment baseline
    Strategy:
      - Build undirected adjacency on entities from top k_base triples
      - Extract TRI + WEDGE motifs around anchors from top k_anchor triples
      - Expand motifs into edge pairs, then add *existing* triples from those pairs
    """
    base = scored_triplets[:k_base]

    # pair -> directed triples present in base
    pair2trip = defaultdict(list)
    nbrs = defaultdict(set)
    for h, r, t, s in base:
        h = str(h); t = str(t)
        pair2trip[undirected_key(h, t)].append((h, r, t, float(s)))
        nbrs[h].add(t); nbrs[t].add(h)

    # anchors from top k_anchor triples
    anchors = set()
    for h, r, t, s in scored_triplets[:k_anchor]:
        if h: anchors.add(str(h))
        if t: anchors.add(str(t))

    added = []
    seen = set()

    def add_pair(u, v):
        key = undirected_key(u, v)
        for tri in pair2trip.get(key, []):
            h, r, t, s = tri
            k = (h, r, t)
            if k in seen:
                continue
            seen.add(k)
            added.append((h, r, t, -0.5))  # mark as motif-added

    # WEDGES + TRIANGLES centered at anchor u
    for u in anchors:
        ns = list(nbrs.get(u, []))
        if len(ns) < 2:
            continue

        # wedges: (a - u - b)
        for a, b in combinations(ns, 2):
            add_pair(u, a)
            add_pair(u, b)
            if len(added) >= max_added:
                return added[:max_added]

        # triangles: u with neighbors v,w s.t. (v,w) exists
        ns_set = set(ns)
        for v in ns:
            for w in nbrs.get(v, set()).intersection(ns_set):
                if v >= w:
                    continue
                add_pair(v, w)
                if len(added) >= max_added:
                    return added[:max_added]

    return added[:max_added]
