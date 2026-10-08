"""Hindsight (embedded, local Postgres) on LongMemEval-S, ingested the way vectorize-io/agent-memory-benchmark does
it: one document per session (its turns as JSON) with the session timestamp and a context line, then recall with
budget "high" and the question date. Recall is saved at two context budgets from one ingestion:

  published  max_tokens 32768 facts + 16384 raw chunks (their LongMemEval setting)
  matched    under Hermes' 10,000-char prefetch spill cap, like gemma-memory's budget

LLM calls go through tools/llm_proxy.py (MiMo flash, reasoning off, counted); embeddings come from
tools/embed_server.py (EmbeddingGemma 2), like the Mem0 arm. Writes bench-data/eval/hindsight-recall.jsonl.
Run with .venv-hindsight:

    python tools/eval_hindsight.py --per-type 3 --limit 1
"""

import argparse
import collections
import json
import os
import sys
import random
import re
import time
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "bench-data")
OUT = os.path.join(DATA, "eval", "hindsight-recall.jsonl")
PROXY = "http://127.0.0.1:8098"

os.environ.update({
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "openai",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL": "http://127.0.0.1:8099/v1",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL": "embeddinggemma-2",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": "local",
})

BUDGETS = {"published": dict(max_tokens=32768, include_chunks=True, max_chunk_tokens=16384),
           # Under Hermes' 10,000-char prefetch spill cap, like gemma-memory's budget (~9k chars delivered).
           "matched": dict(max_tokens=1400, include_chunks=True, max_chunk_tokens=600)}


def parse_date(s):
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", s).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


def sample(data, per_type, split="all"):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import lme_select
    return lme_select.select(data, split, per_type)


def proxy_stats():
    with urllib.request.urlopen(PROXY + "/stats", timeout=10) as r:
        return json.load(r).get("hindsight", {})


def as_text(resp):
    """Facts (with their dates) then raw chunks, as one memory block."""
    d = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
    lines = []
    for r in d.get("results") or []:
        when = r.get("occurred_start") or r.get("mentioned_at") or ""
        lines.append(f"- [{str(when)[:10]}] {r.get('text', '')}".replace("[] ", ""))
    chunks = d.get("chunks") or {}
    chunk_texts = [c.get("text", "") for c in (chunks.values() if isinstance(chunks, dict) else chunks)]
    out = "## Recalled facts\n" + "\n".join(lines) if lines else ""
    if chunk_texts:
        out += "\n\n## Recalled conversation excerpts\n" + "\n---\n".join(chunk_texts)
    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="mimo-v2.6-flash")
    args = ap.parse_args()

    from hindsight import HindsightClient, HindsightServer
    data = sample(json.load(open(os.path.join(DATA, "longmemeval_s_cleaned.json"), encoding="utf-8")), args.per_type, args.split)
    if args.limit:
        data = data[:args.limit]
    done = {json.loads(l)["question_id"] for l in open(OUT, encoding="utf-8")} if os.path.exists(OUT) else set()
    todo = [q for q in data if q["question_id"] not in done]
    print(f"{len(todo)} questions to ingest ({len(done)} done)", flush=True)

    server = HindsightServer(llm_provider="openai", llm_api_key="via-proxy", llm_model=args.model,
                             llm_base_url=PROXY + "/t/hindsight/v1")
    server.start(timeout=300)  # embedded Postgres can take a while on a busy machine; the default is 30 s
    try:
        client = HindsightClient(base_url=server.url)
        for i, q in enumerate(todo, 1):
            t0, before = time.time(), proxy_stats()
            bank = re.sub(r"[^a-zA-Z0-9_-]", "_", q["question_id"]) + f"_{int(time.time())}"  # fresh bank per run
            client.create_bank(bank_id=bank)
            items = []
            for sid, date, turns in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
                dt = parse_date(date)
                items.append({
                    "content": json.dumps([{"role": t["role"], "content": t["content"]} for t in turns]),
                    "timestamp": dt.isoformat(),
                    "context": f"Session {sid} - you are the assistant in this conversation - happened on "
                               f"{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC.",
                    "document_id": f"{bank}_{sid}",
                })
            for j in range(0, len(items), 10):  # synchronous retain: returns once the facts are stored
                client.retain_batch(bank_id=bank, items=items[j:j + 10])
            qts = parse_date(q["question_date"]).isoformat()
            recall = {name: as_text(client.recall(bank_id=bank, query=q["question"][:1900], budget="high",
                                                  query_timestamp=qts, **kw)) for name, kw in BUDGETS.items()}
            after = proxy_stats()
            cost = {k: after.get(k, 0) - before.get(k, 0) for k in ("requests", "errors", "in", "out")}
            row = {"question_id": q["question_id"], "recall": recall, "llm": cost, "seconds": round(time.time() - t0)}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {i}/{len(todo)} {q['question_id']}: {row['seconds']}s, llm {cost}, recall chars "
                  f"{ {k: len(v) for k, v in recall.items()} }", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
