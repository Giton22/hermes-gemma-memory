"""Embed the user's side of every LongMemEval-S turn into the bench cache (key expansion for the retrieval arms).
One GPU job: don't run it next to embed_server.py on the same card."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
if os.environ.get("HERMES_AGENT_PATH"):
    sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

from bench_longmemeval import DATA, Embeddings, pairs  # noqa: E402
from gemma_memory.embedder import DOC_PREFIX  # noqa: E402
from gemma_memory.provider import DEFAULTS, _turn_text  # noqa: E402


def user_side(turn_text):
    return turn_text.split("\nAssistant: ")[0]


if __name__ == "__main__":
    data = json.load(open(os.path.join(DATA, "longmemeval_s_cleaned.json"), encoding="utf-8"))
    texts = {user_side(_turn_text(u, a, DEFAULTS["max_chars"]))
             for q in data for s in q["haystack_sessions"] for u, a, _ in pairs(s)}
    print(len(texts), "user-side keys", flush=True)
    emb = Embeddings("google/embeddinggemma-2", "egm2")
    emb.fill(sorted(texts), DOC_PREFIX)
    emb.save()
    print("cached vectors:", len(emb.vecs), flush=True)
