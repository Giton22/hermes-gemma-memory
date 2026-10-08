"""Free screen: does a cross-encoder reranker (over the fused top-N passages) put more answer evidence into what
Hermes delivers? Same coverage metric as screen_coverage.py; the reranker runs here on GPU for speed only."""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

import eval_longmemeval as E  # noqa: E402
import lme_select  # noqa: E402
from bench_longmemeval import pairs  # noqa: E402
from gemma_memory import recall  # noqa: E402
from gemma_memory.provider import DEFAULTS, PASSAGE_KINDS, _turn_text  # noqa: E402


def main(model_name, depth):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, device="cuda")
    data = [q for q in lme_select.select(json.load(open(os.path.join(E.DATA, "longmemeval_s_cleaned.json"),
                                                         encoding="utf-8")), "dev") if not q["question_id"].endswith("_abs")]
    emb = E.Embeddings("google/embeddinggemma-2", "egm2")
    budget = 10000 - 800
    res = {"fused": [], "reranked": [], "blend": []}
    for q in data:
        sessions = {sid: [(_turn_text(u, a, DEFAULTS["max_chars"]), h) for u, a, h in pairs(s)]
                    for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"])}
        store = E.build_store(q, sessions, emb, 768)
        qvec = emb.get(q["question"], E.QUERY_PREFIX, 768)
        fused = store.fused(q["question"], qvec, candidates=depth, kinds=PASSAGE_KINDS)
        head = fused[:depth]
        scores = ce.predict([(q["question"], r["text"]) for r in head], batch_size=64, show_progress_bar=False)
        reranked = [head[i] for i in np.argsort(-scores)] + fused[depth:]
        ce_rank = {r["id"]: k for k, r in enumerate(reranked)}
        blend = sorted(fused, key=lambda r: 1 / (60 + fused.index(r)) + 1 / (60 + ce_rank[r["id"]]), reverse=True)
        for name, ranked in (("fused", fused), ("reranked", reranked), ("blend", blend)):
            picked = recall.select(ranked, budget=budget, assistant_chars=0)
            delivered = E.hermes_spill(recall.render(picked))
            res[name].append(E.evidence_coverage(q, sessions, delivered)["evidence_all"])
        store.close()
    print(f"{model_name}, top {depth} reranked, {len(data)} dev questions: all answer sessions delivered (%)")
    for name, v in res.items():
        print(f"  {name:9} {100 * np.mean(v):5.1f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60)
