import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from .prepare_prompts import unique_preserve_order
from motif_retriever import MotifIndex
from motif_token_learned import augment_scored_triples_learned
from motif_triangle_retriever import augment_scored_triples_triangles
from motif_token_ranked import augment_scored_triples_ranked
from local_motif import local_motif_expand
from motif_units import build_motif_units, select_and_expand_motifs


def get_subgraphs(dataset_name, split):
    input_file = os.path.join("rmanluo", f"RoG-{dataset_name}")
    return load_dataset(input_file, split=split)


def extract_reasoning_paths(text):
    pattern = r"Reasoning Paths:(.*?)\n\nQuestion:"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reasoning_paths = match.group(1).strip()
        return reasoning_paths
    else:
        return None


def add_good_triplets_from_rog(data):
    print("Adding good triplets from ROG...")
    total_good_triplets = 0
    total_good_triplets_in_graph = 0
    total_good_triplets_not_in_graph = 0
    for idx, each_qa in enumerate(tqdm(data)):
        all_paths = extract_reasoning_paths(each_qa["input"]).split("\n")
        data[idx]["good_paths_rog"] = all_paths
        all_good_triplets = []
        for each_path in all_paths:
            each_path = each_path.split(" -> ")
            good_triplets = []
            i = 0
            while i < len(each_path):
                if i + 2 < len(each_path):
                    triplet = (each_path[i], each_path[i + 1], each_path[i + 2])
                    temp_triplet = (each_path[i + 2], each_path[i + 1], each_path[i])
                    total_good_triplets += 1
                    # if triplet in each_qa["graph"] or temp_triplet in each_qa["graph"]:
                    #     total_good_triplets_in_graph += 1
                    # else:
                    #     total_good_triplets_not_in_graph += 1
                    good_triplets.append(triplet)
                i += 2
            all_good_triplets.extend(good_triplets)
        data[idx]["good_triplets_rog"] = unique_preserve_order(all_good_triplets)
    return data


def add_gt_if_not_present(triple_score_dict):
    st = [','.join(list(each)[:3]) for each in triple_score_dict['scored_triples']]
    tt = [','.join(list(each)[:3]) for each in triple_score_dict['target_relevant_triples']]
    for each in tt:
        if each in st:
            continue
        else:
            # put at the beginning
            triple_score_dict["scored_triples"].insert(0, tuple(each.split(',')))
    return triple_score_dict["scored_triples"]


def add_scored_triplets(data, score_dict_path, prompt_mode):
    print("Adding scored triplets...")
    new_data = []
    cnt = 0
    triple_score_dict = torch.load(score_dict_path, weights_only=False)

    running_baselines = False
    if 'triples' in triple_score_dict[next(iter(triple_score_dict))]:
        running_baselines = True
        for k, v in tqdm(triple_score_dict.items()):
            triple_score_dict[k]['scored_triples'] = v['triples']

    for each_qa in tqdm(data):
        if each_qa["id"] in triple_score_dict:
            if 'gt' in prompt_mode:
                scored_triples = add_gt_if_not_present(triple_score_dict[each_qa["id"]])
            else:
                scored_triples = triple_score_dict[each_qa["id"]]["scored_triples"]
            each_qa['scored_triplets'] = scored_triples
            new_data.append(each_qa)
        else:
            print(f"Triplets not found for {each_qa['id']}")
            if running_baselines:
                each_qa['scored_triplets'] = [('', '', '')]
                new_data.append(each_qa)
            elif 'gt' not in prompt_mode:
                raise ValueError
            else:
                cnt += 1
    print(f"Triplets not found for {cnt} questions")
    return new_data


def sample_random_triplets(data, num_triplets, seed=0):
    print(f"Sampling {num_triplets} random triplets...")
    np.random.seed(seed)
    for idx, each_qa in enumerate(tqdm(data)):
        all_triplets = np.array(each_qa["graph"])
        sampled_triplets = np.random.permutation(all_triplets)[:num_triplets]
        data[idx][f"sampled_triplets_{num_triplets}"] = sampled_triplets.tolist()
    return data


def get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode, seed=0, triplets_to_sample=[50, 100, 200, 300], retriever='baseline', motif_tokens_path=None, motif_pair2trip_path=None, top_tokens=200, top_triples=300, motif_quota=0):
    with open(pred_file_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    print("Loading subgraphs...")
    subgraphs = get_subgraphs(dataset_name, split)

    print("Adding subgraphs to data...")
    data = []
    for i, each_qa in enumerate(tqdm(raw_data)):
        assert each_qa["id"] == subgraphs[i]["id"]
        each_qa["graph"] = [tuple(each) for each in subgraphs[i]["graph"]]
        each_qa['a_entity'] = subgraphs[i]['a_entity']
        data.append(each_qa)
    # data = raw_data

    data = add_good_triplets_from_rog(data)
    data = add_scored_triplets(data, score_dict_path, prompt_mode)
    # MOTIF_UNITS_PACKING_IN_DATA: keep prepare_prompts.py upstream; pack inside retrieval
    if retriever == "motif_units" and int(motif_quota) > 0:
        # prompt budget from prompt_mode (e.g., scored_100)
        try:
            budget = int(prompt_mode.split("_")[1])
        except Exception:
            budget = 100

        for each_qa in data:
            base = each_qa.get("scored_triplets", []) or []
            # build motif edges if not already present
            if "motif_triplets" not in each_qa or not each_qa.get("motif_triplets"):
                anchor = each_qa.get("a_entity", []) or each_qa.get("q_entity_in_graph", []) or []
                motifs = build_motif_units(
                    base,
                    anchor_entities=anchor,
                    motif_types=("wedge", "triangle"),
                    candidate_edges=800,       # deeper pool helps "tractor beam"
                    max_edges_per_node=50,
                    max_units=2000,
                )
                picked_units, motif_edges = select_and_expand_motifs(
                    motifs,
                    unit_topk=80,
                    edge_budget=int(motif_quota),  # match quota directly
                    edges_per_unit_cap=3,
                )
                each_qa["motif_units"] = [m.__dict__ for m in picked_units]
                each_qa["motif_triplets"] = [(h, r, t, sc) for (h, r, t, sc) in motif_edges]

            motif_edges = each_qa.get("motif_triplets", []) or []
            motif_3 = [(t[0], t[1], t[2]) for t in motif_edges if len(t) >= 3]
            motif_3 = unique_preserve_order(motif_3)[:min(int(motif_quota), budget)]

            used = set(motif_3)

            # baseline fill keeps original ordering; support both 3- and 4-tuples
            base_3 = []
            base_4_map = {}
            for t in base:
                if len(t) >= 4:
                    base_4_map[(t[0], t[1], t[2])] = t
                    base_3.append((t[0], t[1], t[2]))
                elif len(t) >= 3:
                    base_4_map[(t[0], t[1], t[2])] = (t[0], t[1], t[2], -2.0)
                    base_3.append((t[0], t[1], t[2]))

            base_3 = unique_preserve_order(base_3)
            base_fill_3 = [t for t in base_3 if t not in used]
            remaining = budget - len(motif_3)
            if remaining < 0:
                remaining = 0
            base_fill_3 = base_fill_3[:remaining]

            # build final scored_triplets list as 4-tuples, motif first then baseline
            packed_4 = []
            seen = set()
            for (h, r, t) in motif_3:
                if (h, r, t) in seen:
                    continue
                seen.add((h, r, t))
                packed_4.append((h, r, t, 1e9))  # very high score so it stays in front if anyone sorts later

            for (h, r, t) in base_fill_3:
                if (h, r, t) in seen:
                    continue
                seen.add((h, r, t))
                packed_4.append(base_4_map.get((h, r, t), (h, r, t, -2.0)))

            each_qa["scored_triplets"] = packed_4
            each_qa["motif_edge_count"] = len(motif_3)
            each_qa["baseline_edge_count"] = len(base_fill_3)
            each_qa["retrieved_triplets"] = [list(x[:3]) for x in packed_4]


    # -------------------------------
    # SubgraphRAG 2.0: motif units (wedges+triangles) as atomic retrieval objects
    # Attaches:
    #   each_qa["motif_units"]   : list[dict]
    #   each_qa["motif_triplets"]: list[(h,r,t,score)]
    # -------------------------------
    if retriever == "motif_units":
        for each_qa in data:
            anchor = each_qa.get("a_entity", None)
            if anchor is None:
                anchor = each_qa.get("q_entity_in_graph", []) or []
            motifs = build_motif_units(
                each_qa.get("scored_triplets", []),
                anchor_entities=anchor,
                motif_types=("wedge", "triangle"),
                candidate_edges=800,
                max_edges_per_node=50,
                max_units=2000,
            )
            picked_units, motif_edges = select_and_expand_motifs(
                motifs,
                unit_topk=50,
                edge_budget=800,   # pool; prompt quota will pick first motif_quota
                edges_per_unit_cap=3,
            )
            each_qa["motif_units"] = [m.__dict__ for m in picked_units]
            each_qa["motif_triplets"] = [(h, r, t, sc) for (h, r, t, sc) in motif_edges]


    if retriever == "motif_tokens":
        assert motif_tokens_path is not None and motif_pair2trip_path is not None
        data = add_motif_triplets(data, motif_tokens_path, motif_pair2trip_path, top_tokens=top_tokens, top_triples=top_triples)
        for motif_learn_debug_i, each_qa in enumerate(data):
            base = each_qa.get("scored_triplets", [])
            motif = each_qa.get("motif_triplets", [])

            # keep only top baseline by score (already sorted in score dict)
            base = base[:300]

            # motif triples are 3-tuples -> wrap as 4-tuples with dummy score
            motif_scored = [(h, r, t, -1.0) for (h, r, t) in motif]

            seen = set()
            merged = []
            for item in base + motif_scored:
                h, r, t, s = item
                key = (h, r, t)
                if key in seen:
                    continue
                seen.add(key)
                merged.append((h, r, t, s))

            each_qa["scored_triplets"] = merged
    if retriever == "motif_token_ranked":
        for each_qa in data:
            q = each_qa.get("question") or each_qa.get("query") or ""
            base = each_qa.get("scored_triplets", [])
            merged = augment_scored_triples_ranked(
                question=q,
                scored_triplets=base,
                base_keep=300,
                cand_pool=300,
                top_tokens=100,
                top_triples=100,
            )
            if motif_learn_debug_i % 50 == 0:
                base300 = base[:300]
                print("LEARN_DEBUG i", motif_learn_debug_i, "base300", len(base300), "merged", len(merged), "added", len(merged)-len(base300))
            each_qa["scored_triplets"] = merged

    if retriever == "motif_triangles":
        for each_qa in data:
            base = each_qa.get("scored_triplets", [])
            merged = augment_scored_triples_triangles(
                scored_triplets=base,
                q_entity_in_graph=each_qa.get("q_entity_in_graph", []),
                base_keep=250,
                cand_pool=2000,
                tri_edges_budget=150,
                max_triangles=80,
            )
            each_qa["scored_triplets"] = merged

    if retriever == "motif_token_learned":
        ckpt = "motif_learn_run3/scorer.pt"
        for each_qa in data:
            q = each_qa.get("question") or each_qa.get("query") or ""
            base = each_qa.get("scored_triplets", [])
            merged = augment_scored_triples_learned(
                question=q,
                scored_triplets=base,
                ckpt_path=ckpt,
                q_entity_in_graph=each_qa.get("q_entity_in_graph", []),
                base_keep=300,
                cand_pool=1000,
                top_tokens=200,
                top_triples=300,
            )
            each_qa["scored_triplets"] = merged

    if retriever == "local_motif":
        # purely question-local motif closure on top-K scored_triplets
        for each_qa in data:
            base = each_qa.get("scored_triplets", [])
            # add motif-closed edges/triples derived from base itself
            extra = local_motif_expand(base, k_base=300, k_anchor=50, max_added=300)

            seen = set()
            merged = []
            for item in base[:300] + extra:
                h, r, t, s = item
                key = (h, r, t)
                if key in seen:
                    continue
                seen.add(key)
                merged.append((h, r, t, s))
            each_qa["scored_triplets"] = merged
    # for num_triplets in triplets_to_sample:
    #     data = sample_random_triplets(data, num_triplets, seed)

    return data

def add_motif_triplets(
    data,
    motif_tokens_path: str,
    motif_pair2trip_path: str,
    top_tokens: int = 200,
    top_triples: int = 300,
):
    idx = MotifIndex(motif_tokens_path, motif_pair2trip_path)
    print("Adding motif-expanded triplets...")

    for each_qa in data:
        anchors = set()

        # Use entities already present in scored_triplets as anchors (best cheap default)
        for h, r, t, _ in each_qa.get("scored_triplets", [])[:50]:
            if h: anchors.add(str(h))
            if t: anchors.add(str(t))

        # fallback: use RoG triplets if scored_triplets empty
        if not anchors:
            for h, r, t in each_qa.get("good_triplets_rog", []):
                if h: anchors.add(str(h))
                if t: anchors.add(str(t))

        anchors = list(anchors)
        toks = idx.get_tokens(anchors, top_tokens=top_tokens)
        motif_triples = idx.expand_tokens(toks, top_triples=top_triples)

        each_qa["motif_triplets"] = motif_triples

    return data
