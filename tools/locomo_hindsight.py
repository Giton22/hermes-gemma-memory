"""Hindsight (embedded, local Postgres) on LoCoMo: each conversation ingested once, the way tools/eval_hindsight.py
does LongMemEval (one document per session with its timestamp and a context line), then recall for every question
of that conversation at the published budget and at the budget matched to Hermes' spill cap.

LLM calls go through tools/llm_proxy.py (counted under "hindsight"); vectors from tools/embed_server.py. Writes
bench-data/eval-locomo/hindsight-recall.jsonl ({id, recall: {published, matched}}). Run with .venv-hindsight:

    python tools/locomo_hindsight.py --convs 0
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_hindsight import BUDGETS, as_text  # noqa: E402  (sets the embedding env vars too)
from locomo_common import OUT_DIR, conversations, last_date, parse_when, questions_of, sessions  # noqa: E402

PROXY = "http://127.0.0.1:8098"


def proxy_stats():
    with urllib.request.urlopen(PROXY + "/stats", timeout=10) as r:
        return json.load(r).get("hindsight", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default="0", help="conversation indices, e.g. 0,1,2 or all")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "hindsight-recall.jsonl"))
    args = ap.parse_args()
    from hindsight import HindsightClient, HindsightServer

    convs = conversations()
    picked = range(len(convs)) if args.convs == "all" else map(int, args.convs.split(","))
    done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")} if os.path.exists(args.out) else set()
    os.makedirs(OUT_DIR, exist_ok=True)
    server = HindsightServer(llm_provider="openai", llm_api_key="via-proxy", llm_model="proxy-model",
                             llm_base_url=PROXY + "/t/hindsight/v1")
    server.start(timeout=300)
    try:
        client = HindsightClient(base_url=server.url)
        for ci in picked:
            qs = [q for q in questions_of(convs, ci) if q["id"] not in done]
            if not qs:
                continue
            conv = convs[ci]["conversation"]
            t0, before = time.time(), proxy_stats()
            bank = re.sub(r"[^a-zA-Z0-9_-]", "_", convs[ci]["sample_id"]) + f"_{int(time.time())}"
            client.create_bank(bank_id=bank)
            items = []
            for n, (date, utts) in enumerate(sessions(conv), 1):
                dt = parse_when(date)
                items.append({
                    "content": json.dumps([{"speaker": u["speaker"], "text": u["line"].split(": ", 1)[1]}
                                           for u in utts], ensure_ascii=False),
                    "timestamp": dt.isoformat(),
                    "context": f"Session {n} of a conversation between {conv['speaker_a']} and {conv['speaker_b']}, "
                               f"held on {dt.strftime('%Y-%m-%d %H:%M')}.",
                    "document_id": f"{bank}_s{n}",
                })
            for j in range(0, len(items), 10):  # synchronous retain: returns once the facts are stored
                client.retain_batch(bank_id=bank, items=items[j:j + 10])
            # Retain returns before Hindsight's background consolidation (LLM-written observations) is done; recall
            # waits until its LLM traffic has been idle for 90 s, so every question sees the finished memory.
            last, idle_since = proxy_stats().get("requests", 0), time.time()
            while time.time() - idle_since < 90:
                time.sleep(15)
                now = proxy_stats().get("requests", 0)
                if now != last:
                    last, idle_since = now, time.time()
            ingest = proxy_stats()
            qts = parse_when(last_date(conv)).isoformat()
            with open(args.out, "a", encoding="utf-8") as f:
                for q in qs:
                    recall = {name: as_text(client.recall(bank_id=bank, query=q["question"][:1900], budget="high",
                                                          query_timestamp=qts, **kw)) for name, kw in BUDGETS.items()}
                    f.write(json.dumps({"id": q["id"], "recall": recall}, ensure_ascii=False) + "\n")
            cost = {k: ingest.get(k, 0) - before.get(k, 0) for k in ("requests", "errors", "in", "cached_in", "out",
                                                                       "micro_usd")}
            print(f"conversation {ci}: {len(items)} sessions, {len(qs)} questions recalled, {time.time() - t0:.0f}s, "
                  f"ingestion {cost} = ${cost['micro_usd'] / 1e6:.3f}", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
