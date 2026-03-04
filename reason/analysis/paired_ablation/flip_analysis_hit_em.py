#!/usr/bin/env python3
import json
from pathlib import Path

A_DET = Path("results/KGQA/webqsp/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/eval_N500_tri_a0_lam005_topKonly_fp036/scored_100-sys_icl_dc-0.36-thres_0.0-test-detailed_eval_result.jsonl")
B_DET = Path("results/KGQA/webqsp/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/eval_N500_tri_a0_lam005_NOTopK_fp036_paired_from_saved/scored_100-sys_icl_dc-0.36-thres_0.0-test-detailed_eval_result.jsonl")

def load_map(p: Path):
    m={}
    for line in p.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line: continue
        obj=json.loads(line)
        m[str(obj["id"])] = obj
    return m

a=load_map(A_DET); b=load_map(B_DET)
common=sorted(set(a).intersection(b))
print("A_DET:", A_DET)
print("B_DET:", B_DET)
print("common:", len(common))

def as_hit(o): return int(o.get("hit",0))==1
def as_em(o):  return float(o.get("acc",0.0))>=1.0

def flips(metric):
    cw=wc=same=0
    ex_cw=[]; ex_wc=[]
    for k in common:
        ac=metric(a[k]); bc=metric(b[k])
        if ac==bc:
            same+=1
        elif ac and (not bc):
            cw+=1
            if len(ex_cw)<5:
                ex_cw.append((k,a[k].get("ground_truth"),a[k].get("hit"),a[k].get("acc"),b[k].get("hit"),b[k].get("acc")))
        else:
            wc+=1
            if len(ex_wc)<5:
                ex_wc.append((k,b[k].get("ground_truth"),a[k].get("hit"),a[k].get("acc"),b[k].get("hit"),b[k].get("acc")))
    return cw,wc,same,ex_cw,ex_wc

cw,wc,same,ex_cw,ex_wc = flips(as_hit)
print("\nHIT flips (hit==1):")
print("topK-only correct -> non-topK wrong:", cw)
print("topK-only wrong -> non-topK correct:", wc)
print("same:", same)

cw2,wc2,same2,_,_ = flips(as_em)
print("\nEM flips (acc==1.0):")
print("topK-only EM -> non-topK not-EM:", cw2)
print("topK-only not-EM -> non-topK EM:", wc2)
print("same:", same2)

print("\nExamples HIT: topK-only correct -> non-topK wrong (up to 5)")
for k,gt,ahit,aacc,bhit,bacc in ex_cw:
    print("-", k, "| gt:", gt, "| topK(hit,acc):", ahit, aacc, "| nonTop(hit,acc):", bhit, bacc)

print("\nExamples HIT: topK-only wrong -> non-topK correct (up to 5)")
for k,gt,ahit,aacc,bhit,bacc in ex_wc:
    print("-", k, "| gt:", gt, "| topK(hit,acc):", ahit, aacc, "| nonTop(hit,acc):", bhit, bacc)
