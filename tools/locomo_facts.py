"""gemma-memory's optional fact layer on LoCoMo: one gemma_memory.facts call per session (what on_session_end does),
cached in bench-data/eval-locomo/facts-<tag>.jsonl ({conv, session, date, facts}). Any OpenRouter model, reasoning
off; the same model the other systems use for their own extraction keeps the comparison fair.

    python tools/locomo_facts.py --model anthropic/claude-haiku-5.5 --tag haiku
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
if os.environ.get("HERMES_AGENT_PATH"):
    sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

import eval_longmemeval  # noqa: E402
from gemma_memory import facts  # noqa: E402
from locomo_common import OUT_DIR, conversations, parse_when, sessions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--tag", default="haiku")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = os.path.join(OUT_DIR, f"facts-{args.tag}.jsonl")
    done = {(d["conv"], d["session"]) for d in map(json.loads, open(out, encoding="utf-8"))} if os.path.exists(out) else set()
    chat = facts.openai_chat(eval_longmemeval.OPENROUTER, args.model, eval_longmemeval._env("OPENROUTER_API_KEY"),
                             extra_body={"reasoning": {"enabled": False, "exclude": True}})
    todo = []
    for ci, c in enumerate(conversations()):
        for n, (date, utts) in enumerate(sessions(c["conversation"]), 1):
            if (ci, n) not in done:
                todo.append((ci, n, date, "\n".join(u["line"] for u in utts)))
    print(f"{len(todo)} sessions to extract", flush=True)
    lock, total = threading.Lock(), 0

    def one(ci, n, date, text):
        found = facts.extract(text, parse_when(date).strftime("%Y-%m-%d"), chat)
        with lock, open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"conv": ci, "session": n, "date": date, "facts": found}, ensure_ascii=False) + "\n")
        return len(found)

    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(one, *t) for t in todo]):
            try:
                total += fut.result()
            except Exception as e:
                print("  error:", repr(e)[:150], flush=True)
    print(f"done: {total} facts", flush=True)


if __name__ == "__main__":
    main()
