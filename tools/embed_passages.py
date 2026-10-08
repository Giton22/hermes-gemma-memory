"""Embed the passages (gemma_memory.passages.split_turn) of every turn in a split's haystacks into their own cache
(bench-data/emb-egm2-passages.npz). One GPU job: don't run it next to embed_server.py on the same card.

    python tools/embed_passages.py --split dev
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
if os.environ.get("HERMES_AGENT_PATH"):
    sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

import lme_select  # noqa: E402
from bench_longmemeval import DATA, Embeddings, pairs  # noqa: E402
from gemma_memory.embedder import DOC_PREFIX  # noqa: E402
from gemma_memory.passages import split_turn  # noqa: E402
from gemma_memory.provider import DEFAULTS, _turn_text  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    data = lme_select.select(json.load(open(os.path.join(DATA, "longmemeval_s_cleaned.json"), encoding="utf-8")),
                             args.split)
    texts = {p for q in data for s in q["haystack_sessions"] for u, a, _ in pairs(s)
             for p in split_turn(_turn_text(u, a, DEFAULTS["max_chars"]))}
    print(len(texts), "passages", flush=True)
    emb = Embeddings("google/embeddinggemma-2", "egm2-passages")
    emb.fill(sorted(texts), DOC_PREFIX, save_every=600)
    print("cached passage vectors:", len(emb.vecs), flush=True)
