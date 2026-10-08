"""Mem0 (open-source) on LongMemEval-S, ingested the way mem0ai/memory-benchmarks does it: sessions in date order,
one add() per user+assistant exchange with the session's timestamp, then a top-k search per question.

Same models as the other arms: MiMo flash for Mem0's fact extraction, EmbeddingGemma 2 (tools/embed_server.py on
--embed-url) for its vectors. Writes bench-data/eval/mem0-recall.jsonl (resumable), which eval_longmemeval.py
injects as the mem0 arm. Run with .venv-mem0:

    python tools/embed_server.py --port 8099          # in the GPU venv
    python tools/eval_mem0.py --per-type 15 --workers 12
"""

import argparse
import collections
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

os.environ.setdefault("MEM0_TELEMETRY", "false")  # Mem0 reports usage to PostHog by default
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "bench-data")
OUT = os.path.join(DATA, "eval", "mem0-recall.jsonl")
BASE = os.environ.get("MIMO_BASE_URL", "")  # your MiMo Token Plan endpoint


def api_key():
    key = os.environ.get("XIAOMI_TOKEN_PLAN_API_KEY")
    if not key and sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            key = winreg.QueryValueEx(h, "XIAOMI_TOKEN_PLAN_API_KEY")[0]
    return key


USAGE = collections.Counter()
_usage_lock = threading.Lock()


def no_thinking():
    """Mem0 calls the OpenAI client itself: switch MiMo's reasoning off on every call (like the other arms' reader)
    and count the tokens it spends."""
    from openai.resources.chat import completions
    original = completions.Completions.create

    def create(self, *a, **kw):
        kw.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
        out = original(self, *a, **kw)
        u = getattr(out, "usage", None)
        if u:
            with _usage_lock:
                USAGE["calls"] += 1
                USAGE["in"] += u.prompt_tokens or 0
                USAGE["out"] += u.completion_tokens or 0
        return out
    completions.Completions.create = create


def ts(date_str):
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
    return int(datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc).timestamp())


def exchanges(session, date):
    """User+assistant pairs (mem0-benchmarks CHUNK_SIZE = 2). The OSS SDK refuses add(timestamp=...), so the
    session date goes into the text, where the extraction LLM can read it."""
    out, cur = [], []
    for m in session:
        cur.append({"role": m["role"], "content": m["content"]})
        if m["role"] == "assistant":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    out = [p for p in out if all(x["content"].strip() for x in p)]
    for p in out:
        p[0] = {**p[0], "content": f"[Conversation date: {date}] {p[0]['content']}"}
    return out


def sample(data, per_type, split="all"):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import lme_select
    return lme_select.select(data, split, per_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--per-type", type=int, default=15)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--embed-url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--limit", type=int, default=0, help="pilot: only the first N sampled questions")
    ap.add_argument("--out", default=OUT, help="recall file; e.g. bench-data/eval/mem0ds-recall.jsonl")
    args = ap.parse_args()

    no_thinking()
    from mem0 import Memory
    key = api_key()
    data = sample(json.load(open(os.path.join(DATA, "longmemeval_s_cleaned.json"), encoding="utf-8")), args.per_type, args.split)
    if args.limit:
        data = data[:args.limit]
    done = set()
    out_path = args.out
    if os.path.exists(out_path):
        done = {json.loads(l)["question_id"] for l in open(out_path, encoding="utf-8")}
    todo = [q for q in data if q["question_id"] not in done]
    print(f"{len(todo)} questions to ingest ({len(done)} done)", flush=True)
    lock = threading.Lock()

    def one(q):
        tmp = tempfile.mkdtemp(prefix="mem0-")
        try:
            m = Memory.from_config({
                # Through tools/llm_proxy.py: same MiMo model, reasoning off, tokens counted under "mem0".
                "llm": {"provider": "openai", "config": {"model": args.model, "api_key": "via-proxy",
                                                         "openai_base_url": "http://127.0.0.1:8098/t/mem0/v1",
                                                         "temperature": 0}},
                "embedder": {"provider": "openai", "config": {"model": "embeddinggemma-2", "api_key": "local",
                                                              "openai_base_url": args.embed_url,
                                                              "embedding_dims": 768}},
                # Qdrant (Mem0's default store, local files) keeps Mem0's hybrid BM25 + vector search; faiss drops it.
                "vector_store": {"provider": "qdrant", "config": {"collection_name": "lme", "path": tmp,
                                                                  "embedding_model_dims": 768, "on_disk": True}},
                "history_db_path": os.path.join(tmp, "history.db"),
            })
            sessions = sorted(zip(q["haystack_dates"], q["haystack_sessions"]), key=lambda x: ts(x[0]))
            t0, adds, errors = time.time(), 0, 0
            for date, s in sessions:
                for pair in exchanges(s, date):
                    for attempt in range(4):
                        try:
                            m.add(pair, user_id="u")
                            adds += 1
                            break
                        except Exception as e:  # rate limit / bad JSON from the LLM: retry, then skip the pair
                            if attempt == 3:
                                errors += 1
                                print("  add failed:", str(e)[:120], flush=True)
                            if "not supported" in str(e):
                                raise  # a config error, not a transient one: stop instead of retrying 260 times
                            time.sleep(2 ** attempt)
            res = m.search(q["question"], top_k=args.top_k, filters={"user_id": "u"})
            results = res.get("results", res) if isinstance(res, dict) else res
            mems = [{"memory": r.get("memory", ""), "created_at": r.get("created_at"), "score": r.get("score")}
                    for r in results]
            row = {"question_id": q["question_id"], "memories": mems, "adds": adds, "add_errors": errors,
                   "seconds": round(time.time() - t0), "usage_total_so_far": dict(USAGE)}
            with lock, open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return row
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, q) for q in todo]), 1):
            try:
                r = fut.result()
                print(f"  {i}/{len(todo)} {r['question_id']}: {r['adds']} adds ({r['add_errors']} failed), "
                      f"{len(r['memories'])} memories, {r['seconds']}s, tokens so far {dict(USAGE)}", flush=True)
            except Exception as e:
                print("  question failed:", repr(e)[:200], flush=True)


if __name__ == "__main__":
    main()
