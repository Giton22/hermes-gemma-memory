"""Mem0 (open-source) on LoCoMo: each conversation ingested once, the way tools/eval_mem0.py does LongMemEval
(two utterances per add(), the session date in the text since the OSS add() takes no timestamp), then a top-k search
for every question of that conversation. The first speaker is "user", the second "assistant", each line named.

LLM calls go through tools/llm_proxy.py (counted under "mem0"); vectors from tools/embed_server.py. Writes
bench-data/eval-locomo/mem0-recall.jsonl ({id, recall}), which tools/eval_locomo.py reads as the mem0 arm.
Run with .venv-mem0:

    python tools/locomo_mem0.py --convs 0
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request

os.environ.setdefault("MEM0_TELEMETRY", "false")  # Mem0 reports usage to PostHog by default
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from locomo_common import DATA, OUT_DIR, conversations, questions_of, sessions  # noqa: E402

PROXY = "http://127.0.0.1:8098"


def proxy_stats(tag):
    with urllib.request.urlopen(PROXY + "/stats", timeout=10) as r:
        return json.load(r).get(tag, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default="0", help="conversation indices, e.g. 0,1,2 or all")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--embed-url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "mem0-recall.jsonl"))
    args = ap.parse_args()
    from mem0 import Memory

    convs = conversations()
    picked = range(len(convs)) if args.convs == "all" else map(int, args.convs.split(","))
    done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")} if os.path.exists(args.out) else set()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ci in picked:
        qs = [q for q in questions_of(convs, ci) if q["id"] not in done]
        if not qs:
            continue
        tmp = tempfile.mkdtemp(prefix="mem0-")
        t0, before = time.time(), proxy_stats("mem0")
        try:
            m = Memory.from_config({
                "llm": {"provider": "openai", "config": {"model": "proxy-model", "api_key": "via-proxy",
                                                         "openai_base_url": PROXY + "/t/mem0/v1", "temperature": 0}},
                "embedder": {"provider": "openai", "config": {"model": "embeddinggemma-2", "api_key": "local",
                                                              "openai_base_url": args.embed_url,
                                                              "embedding_dims": 768}},
                # Qdrant (Mem0's default store, local files) keeps Mem0's hybrid BM25 + vector search.
                "vector_store": {"provider": "qdrant", "config": {"collection_name": "locomo", "path": tmp,
                                                                  "embedding_model_dims": 768, "on_disk": True}},
                "history_db_path": os.path.join(tmp, "history.db"),
            })
            conv = convs[ci]["conversation"]
            first = conv["speaker_a"]
            adds = errors = 0
            for date, utts in sessions(conv):
                for i in range(0, len(utts), 2):
                    pair = [{"role": "user" if u["speaker"] == first else "assistant", "content": u["line"]}
                            for u in utts[i:i + 2]]
                    pair[0] = {**pair[0], "content": f"[Conversation date: {date}] {pair[0]['content']}"}
                    for attempt in range(4):
                        try:
                            m.add(pair, user_id="u")
                            adds += 1
                            break
                        except Exception as e:  # bad JSON from the LLM etc.: retry, then skip the pair
                            if attempt == 3:
                                errors += 1
                                print("  add failed:", str(e)[:150], flush=True)
                            time.sleep(2 ** attempt)
            ingest = proxy_stats("mem0")
            with open(args.out, "a", encoding="utf-8") as f:
                for q in qs:
                    res = m.search(q["question"], top_k=args.top_k, filters={"user_id": "u"})
                    results = res.get("results", res) if isinstance(res, dict) else res
                    lines = [f"- {r.get('memory', '')}" for r in results]
                    f.write(json.dumps({"id": q["id"], "recall": "## Recalled facts\n" + "\n".join(lines)
                                        if lines else ""}, ensure_ascii=False) + "\n")
            cost = {k: ingest.get(k, 0) - before.get(k, 0) for k in ("requests", "errors", "in", "cached_in", "out",
                                                                       "micro_usd")}
            print(f"conversation {ci}: {adds} adds ({errors} failed), {len(qs)} questions recalled, "
                  f"{time.time() - t0:.0f}s, ingestion {cost} = ${cost['micro_usd'] / 1e6:.3f}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
