import json
from pathlib import Path
from typing import Dict, List, Tuple

Triple = Tuple[str, str, str]

def _pair_key(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)

class MotifIndex:
    def __init__(self, tokens_path: str, pair2trip_path: str):
        self.tokens_by_anchor: Dict[str, List[dict]] = {}
        self.pair2trip: Dict[Tuple[str, str], List[Triple]] = {}

        for line in Path(tokens_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            self.tokens_by_anchor[obj["anchor"]] = obj["tokens"]

        for line in Path(pair2trip_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            u, v = obj["u"], obj["v"]
            triples = [tuple(t) for t in obj["triples"]]
            self.pair2trip[_pair_key(u, v)] = triples

    def get_tokens(self, anchors: List[str], top_tokens: int = 200) -> List[dict]:
        toks: List[dict] = []
        for a in anchors:
            toks.extend(self.tokens_by_anchor.get(a, []))
        toks.sort(key=lambda t: 2 if t.get("type") == "TRI" else 1, reverse=True)
        return toks[:top_tokens]

    def expand_tokens(self, toks: List[dict], top_triples: int = 300) -> List[Triple]:
        out: List[Triple] = []
        seen = set()

        def add_pair(u: str, v: str):
            key = _pair_key(u, v)
            for h, r, t in self.pair2trip.get(key, []):
                tri = (h, r, t)
                if tri in seen:
                    continue
                seen.add(tri)
                out.append(tri)

        for tok in toks:
            tp = tok.get("type")
            if tp == "TRI":
                u, v, w = tok["u"], tok["v"], tok["w"]
                add_pair(u, v); add_pair(u, w); add_pair(v, w)
            elif tp == "WEDGE":
                a, u, b = tok["a"], tok["u"], tok["b"]
                add_pair(a, u); add_pair(u, b)

            if len(out) >= top_triples:
                break

        return out[:top_triples]
