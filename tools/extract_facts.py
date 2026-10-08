"""Extract facts (gemma_memory.facts) for every conversation in the selected LongMemEval questions' histories, once
per conversation, cached in bench-data/facts.jsonl. MiMo flash, reasoning off, through the Token Plan.

    python tools/extract_facts.py --split dev --per-type 10
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

import eval_longmemeval  # noqa: E402
import lme_select  # noqa: E402
from eval_longmemeval import BASE, DATA, DEEPSEEK, api_key  # noqa: E402
from gemma_memory import facts  # noqa: E402



def conversation_text(session, assistant_chars=1500):
    """The session as the extractor sees it: user turns whole, assistant replies cut to keep the prompt small."""
    lines = []
    for t in session:
        text = t["content"] if t["role"] == "user" else t["content"][:assistant_chars]
        lines.append(f"{'User' if t['role'] == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="mimo-v2.6-flash", help="mimo-v2.6-flash or deepseek-flash")
    ap.add_argument("--out", default=os.path.join(DATA, "facts.jsonl"))
    args = ap.parse_args()
    data = lme_select.select(json.load(open(os.path.join(DATA, "longmemeval_s_cleaned.json"), encoding="utf-8")),
                             args.split, args.per_type)
    sessions = {}
    for q in data:
        for sid, date, s in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
            sessions[sid] = (date, s)
    OUT = args.out
    done = {json.loads(l)["session_id"] for l in open(OUT, encoding="utf-8")} if os.path.exists(OUT) else set()
    todo = [sid for sid in sessions if sid not in done]
    print(f"{len(data)} questions, {len(sessions)} conversations, {len(todo)} to extract", flush=True)
    if "/" in args.model and not args.model.endswith(":free"):  # OpenRouter, paid: reasoning off as with DeepSeek's API
        chat = facts.openai_chat(eval_longmemeval.OPENROUTER, args.model, eval_longmemeval._env("OPENROUTER_API_KEY"),
                                 extra_body={"reasoning": {"enabled": False, "exclude": True},
                                             **({"provider": eval_longmemeval.OR_DEEPSEEK}
                                                if "deepseek" in args.model else {})})
    elif "/" in args.model:  # OpenRouter free models: paced by the harness's chat()
        def chat(prompt):
            return eval_longmemeval.chat("", args.model, [{"role": "user", "content": prompt}], 4000)[0]
    elif args.model.startswith("deepseek"):
        chat = facts.openai_chat(DEEPSEEK, args.model, os.environ["DEEPSEEK_API_KEY"],
                                 extra_body={"thinking": {"type": "disabled"}})
    else:
        chat = facts.openai_chat(BASE, args.model, api_key(), extra_body={"thinking": {"type": "disabled"}})
    lock, n = threading.Lock(), 0

    stop = threading.Event()  # out of credit or the free daily cap: the rest waits for the next run

    def one(sid):
        if stop.is_set():
            return 0
        date, s = sessions[sid]
        try:
            found = facts.extract(conversation_text(s), date, chat)
        except RuntimeError as e:
            if any(w in str(e).lower() for w in ("per-day", "quota", "insufficient")):
                stop.set()
            raise
        with lock, open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps({"session_id": sid, "date": date, "facts": found}, ensure_ascii=False) + "\n")
        return len(found)

    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, sid) for sid in todo]), 1):
            try:
                n += fut.result()
            except Exception as e:
                print("  error:", repr(e)[:150], flush=True)
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} conversations, {n} facts", flush=True)


if __name__ == "__main__":
    main()
