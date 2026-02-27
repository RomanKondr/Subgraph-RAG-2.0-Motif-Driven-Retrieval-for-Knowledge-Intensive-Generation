# Heuristic motif reranking: locked-down conclusion (WebQSP test, N=500)

Compared settings (same 500 questions, fp=0.36):
- **TopK-only triangles a0 lam=0.05**: Hit@1 89.0, MacroF1 60.4703, EM 41.2, TW 11.0, no-ans 13.0%, Hit 86.8
- **Non-topK triangles a0 lam=0.05**: Hit@1 90.2, MacroF1 61.0385, EM 41.2, TW 9.8, no-ans 13.6%, Hit 88.0

Paired flips (per-example detailed eval):
- HIT flips (hit==1): **7 regressions** (topK correct→nonTop wrong) vs **13 fixes** (topK wrong→nonTop correct) → **net +6 hits** for non-topK
- EM flips (acc==1.0): **11 gained / 11 lost** → net 0 (consistent with equal EM)

Conclusion:
TopK-only reranking is neutral/negative here. Standard reranking is better on paired N=500.
