"""Embed the extracted facts (bench-data/facts.jsonl) into bench-data/emb-egm2-facts.npz. One GPU job."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
if os.environ.get("HERMES_AGENT_PATH"):
    sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

from bench_longmemeval import DATA, Embeddings  # noqa: E402
from gemma_memory.embedder import DOC_PREFIX  # noqa: E402

if __name__ == "__main__":
    name = os.environ.get("FACTS", "facts")
    texts = {f for line in open(os.path.join(DATA, f"{name}.jsonl"), encoding="utf-8") for f in json.loads(line)["facts"]}
    print(len(texts), "facts", flush=True)
    emb = Embeddings("google/embeddinggemma-2", "egm2-facts")
    url = os.environ.get("EMBED_URL")  # use a running tools/embed_server.py instead of a second model on the GPU
    if url:
        import numpy as np
        from gemma_memory.embedder import Embedder
        server = Embedder(url, "embeddinggemma-2", 0, timeout=300)
        todo = sorted(t for t in texts if emb.key(DOC_PREFIX + t) not in emb.vecs)
        for i in range(0, len(todo), 64):
            for t, v in zip(todo[i:i + 64], server.documents(todo[i:i + 64])):
                emb.vecs[emb.key(DOC_PREFIX + t)] = np.asarray(v, dtype=np.float16)
            if i % 6400 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        emb.save()
    else:
        emb.fill(sorted(texts), DOC_PREFIX, save_every=300)
    print("cached fact vectors:", len(emb.vecs), flush=True)
