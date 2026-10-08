"""Free pre-screen for recall settings: share of dev questions whose every answer session reaches the text Hermes
delivers (after spill), through the shipped prefetch() (plugin arms). No LLM calls; end-to-end runs confirm the
best few. Coverage is necessary, not sufficient: it can't see whether the reader uses the evidence.

    python tools/screen_coverage.py "plugin" "plugin:recall_unit=passage" ...
"""

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
from gemma_memory.provider import DEFAULTS, _turn_text  # noqa: E402


def main(arms):
    data = [q for q in lme_select.select(json.load(open(os.path.join(E.DATA, "longmemeval_s_cleaned.json"),
                                                         encoding="utf-8")), "dev") if not q["question_id"].endswith("_abs")]
    emb = E.Embeddings("google/embeddinggemma-2", "egm2")
    res = {a: {"all": [], "any": [], "chars": []} for a in arms}
    by_type = {a: {} for a in arms}
    for q in data:
        sessions = {sid: [(_turn_text(u, a, DEFAULTS["max_chars"]), h) for u, a, h in pairs(s)]
                    for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"])}
        store = E.build_store(q, sessions, emb, 768)
        for a in arms:
            delivered = E.hermes_spill(E.recall(a, q, store, emb, 768))
            c = E.evidence_coverage(q, sessions, delivered)
            res[a]["all"].append(c["evidence_all"])
            res[a]["any"].append(c["evidence_any"])
            res[a]["chars"].append(len(delivered))
            by_type[a].setdefault(q["question_type"], []).append(c["evidence_all"])
        store.close()
    print(f"{len(data)} dev questions: all / any answer sessions delivered (%), delivered chars")
    for a in arms:
        r = res[a]
        types = "  ".join(f"{t[:12]} {100 * np.mean(v):4.0f}" for t, v in sorted(by_type[a].items()))
        print(f"  {100 * np.mean(r['all']):5.1f} {100 * np.mean(r['any']):5.1f} {np.mean(r['chars']):6.0f}  {a}\n        {types}")


if __name__ == "__main__":
    main(sys.argv[1:])
