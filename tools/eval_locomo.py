"""LoCoMo (snap-research, 10 long two-person conversations): held-out check of what was tuned on LongMemEval.

Nothing here was tuned on LoCoMo. Per conversation the turns are stored once (consecutive utterances paired, shared
photos as their captions, each session under its date) and every question of that conversation asks the same store.
Reader and Hermes-style injection are the same as tools/eval_longmemeval.py. Each answer is judged twice:

  strict   LongMemEval's answer-check prompt
  lenient  mem0ai/memory-benchmarks' LoCoMo judge (Apache-2.0): partial credit, date tolerance; how published
           LoCoMo "J" scores are made

Arms: none, keyword (top 5 by FTS), plugin (GemmaMemoryProvider.prefetch, shipped defaults, overrides as in
eval_longmemeval), full (the whole conversation in the prompt: the context-window ceiling).
Categories 1-4 (5 = adversarial, excluded as in published results). Embeddings from tools/embed_server.py.

    python tools/eval_locomo.py run --per-cat 50 --arms none,keyword,plugin,full
"""

import argparse
import collections
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

from eval_longmemeval import (JUDGE, SYSTEM, api_key, build_memory_context_block, chat,  # noqa: E402
                              compose_user_api_content, fit_under_cap)
import rejudge_decider  # noqa: E402
from gemma_memory.passages import split_turn  # noqa: E402
from gemma_memory.embedder import Embedder  # noqa: E402
from gemma_memory.provider import DEFAULTS, GemmaMemoryProvider  # noqa: E402
from gemma_memory.store import Store  # noqa: E402

DATA = os.path.join(ROOT, "bench-data")
OUT_DIR = os.path.join(DATA, "eval-locomo")
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

LENIENT_JUDGE = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


# The lenient judge's rules alone, as Decider instructions (its yes = CORRECT).
LENIENT_RULES = ("Answer yes (CORRECT) or no (WRONG) for the model response against the correct answer, by these "
                 "rules:\n" +LENIENT_JUDGE.split("## Rules", 1)[1].split("## Question", 1)[0].strip())


def parse_when(s):
    """'1:56 pm on 8 May, 2023' -> timestamp."""
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").timestamp()


def utterance(t):
    text = t.get("text", "")
    if t.get("blip_caption"):
        text += f" [shares a photo: {t['blip_caption']}]"
    return f"{t['speaker']}: {text}"


def conversation_turns(conv):
    """[(session_no, timestamp, text)] with consecutive utterances paired, like the plugin pairs user+assistant."""
    out = []
    sessions = sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
    for n in sessions:
        ts = parse_when(conv[f"session_{n}_date_time"])
        utts = conv[f"session_{n}"]
        for i in range(0, len(utts), 2):
            out.append((n, ts, "\n".join(utterance(t) for t in utts[i:i + 2])))
    return out


def full_text(conv):
    lines = []
    sessions = sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
    for n in sessions:
        lines.append(f"### Conversation on {conv[f'session_{n}_date_time']}")
        lines += [utterance(t) for t in conv[f"session_{n}"]]
    return "## Earlier conversations (complete)\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--arms", default="none,keyword,plugin,full")
    ap.add_argument("--per-cat", type=int, default=50)
    ap.add_argument("--reader", default="mimo-v2.6-flash")
    ap.add_argument("--judge", default="mimo-v2.6-pro")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for result files (e.g. -haiku)")
    ap.add_argument("--convs", default="all", help="conversation indices to score, e.g. 0 or 0,1,2")
    ap.add_argument("--facts", default="", help="LoCoMo facts file (tools/locomo_facts.py) for use_facts arms")
    args = ap.parse_args()
    arms = args.arms.split(",")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = lambda arm: os.path.join(OUT_DIR, re.sub(r"[^A-Za-z0-9_.=-]", "_", arm) + f"{args.tag}.jsonl")  # noqa: E731

    convs = json.load(open(os.path.join(DATA, "locomo10.json"), encoding="utf-8"))
    questions = []
    for ci, c in enumerate(convs):
        for qi, q in enumerate(c["qa"]):
            if q.get("category") in CATEGORY and "answer" in q:
                questions.append({**q, "conv": ci, "id": f"{c['sample_id']}-{qi}"})
    rng, by = random.Random(0), collections.defaultdict(list)
    for q in questions:
        by[q["category"]].append(q)
    questions = [q for cat in sorted(by) for q in (rng.sample(by[cat], min(args.per_cat, len(by[cat])))
                                                   if args.per_cat else by[cat])]
    if args.convs != "all":
        keep = {int(c) for c in args.convs.split(",")}
        questions = [q for q in questions if q["conv"] in keep]

    if args.cmd == "report":
        return report(questions, arms, path)

    key = api_key() if not ((args.reader.count("/") or args.reader.startswith("proxy:")) and args.judge == "decider") else ""
    decider_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if args.judge == "decider" and not decider_key:
        sys.exit("--judge decider needs PERPLEXITY_API_KEY")
    facts = {}  # (conversation, session number) -> [fact]
    if args.facts:
        for line in open(args.facts, encoding="utf-8"):
            d = json.loads(line)
            facts[(d["conv"], d["session"])] = d["facts"]
    saved = {}  # another system's recall, by question id (tools/locomo_mem0.py, tools/locomo_hindsight.py)
    for a in arms:
        system = a.split("-")[0]
        if system in ("mem0", "hindsight") and system not in saved:
            saved[system] = {json.loads(l)["id"]: json.loads(l)["recall"]
                             for l in open(os.path.join(OUT_DIR, f"{system}-recall.jsonl"), encoding="utf-8")}
    embedder = Embedder(args.embed_url, "embeddinggemma-2", 768)
    stores, provider_cfg = {}, {**DEFAULTS, "dims": 768}
    for ci in sorted({q["conv"] for q in questions}):  # one store per conversation, embedded once
        turns = conversation_turns(convs[ci]["conversation"])
        texts = [t for _, _, t in turns]
        texts += [p for t in texts for p in split_turn(t)]  # stored like the plugin: each turn, then its passages
        texts += [f for n in sorted({n for n, _, _ in turns}) for f in facts.get((ci, n), [])]
        vec = {}
        for j in range(0, len(texts), 32):
            for t, v in zip(texts[j:j + 32], embedder.documents(texts[j:j + 32])):
                vec[t] = v
        store = Store(":memory:", model="bench", dims=768)
        for n, ts, text in turns:
            turn = store.add("turn", text, session_id=f"s{n}", created_at=ts, vec=vec[text])
            for p in split_turn(text):
                store.add("passage", p, session_id=f"s{n}", created_at=ts, parent=turn, vec=vec[p])
        for n, ts in sorted({(n, ts) for n, ts, _ in turns}):  # facts stored after their session, like at session end
            for f in facts.get((ci, n), []):
                store.add("fact", f, session_id=f"s{n}", created_at=ts, vec=vec[f])
        stores[ci] = store
        print(f"conversation {ci}: {len(turns)} turns stored", flush=True)
    qvecs = {}
    qs = [q["question"] for q in questions]
    for j in range(0, len(qs), 32):
        for text, v in zip(qs[j:j + 32], [embedder.query(t) for t in qs[j:j + 32]]):
            qvecs[text] = v

    class Cached:
        def query(self, text, *, timeout=None):
            return qvecs[text]

    def recall(arm, q):
        store = stores[q["conv"]]
        if arm == "none":
            return ""
        if arm == "full":
            return full_text(convs[q["conv"]]["conversation"])
        if arm == "mem0":  # same treatment as gemma-memory: whole lines in their own order, under the spill cap
            return fit_under_cap(saved["mem0"][q["id"]])
        if arm.startswith("hindsight-"):  # hindsight-matched / hindsight-published, fitted the same way
            return fit_under_cap(saved["hindsight"][q["id"]][arm.split("-", 1)[1]])
        if arm == "keyword":
            hits = [r for r in store.search(q["question"], None, limit=40) if r["kind"] == "turn"][:5]
            return "## Recalled from earlier conversations\n" + "\n".join(r["text"] for r in hits) if hits else ""
        if arm == "plugin" or arm.startswith("plugin:"):
            overrides = {}
            for part in filter(None, arm[len("plugin:"):].split("+")):
                k, v = part.split("=")
                overrides[k] = (v.lower() in ("1", "true", "yes")) if isinstance(DEFAULTS[k], bool) else type(DEFAULTS[k])(v)
            p = GemmaMemoryProvider({**provider_cfg, **overrides})
            p._store, p._embedder = store, Cached()
            p._bind("eval-new-session")
            return p.prefetch(q["question"])
        raise ValueError(arm)

    done = {a: {json.loads(l)["id"] for l in open(path(a), encoding="utf-8")} if os.path.exists(path(a)) else set()
            for a in arms}
    lock = threading.Lock()

    def last_date(ci):
        conv = convs[ci]["conversation"]
        n = max(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
        return conv[f"session_{n}_date_time"]

    def one(q):
        for arm in arms:
            if q["id"] in done[arm]:
                continue
            memory = recall(arm, q)
            if arm == "full":  # the ceiling: everything in the prompt, not a Hermes memory provider (no spill)
                user = q["question"] + "\n\n" + build_memory_context_block(memory)
            else:
                user = compose_user_api_content(q["question"], memory) or q["question"]
            hyp, _ = chat(key, args.reader, [{"role": "system", "content": SYSTEM.format(date=last_date(q["conv"]))},
                                            {"role": "user", "content": user}],
                          800 if args.no_think else 8000, thinking=not args.no_think)
            gold = str(q["answer"]).split(";")[0].strip() if q["category"] == 3 else str(q["answer"])
            row = {"id": q["id"], "category": CATEGORY[q["category"]], "question": q["question"], "answer": gold,
                   "memory_chars": len(memory), "hypothesis": hyp}
            if args.judge == "decider":  # Decider v1.1 with each judge's rules as its instructions
                row["strict_p"] = rejudge_decider.decide(decider_key, row)
                row["lenient_p"] = rejudge_decider.decide(decider_key, {**row, "rule": LENIENT_RULES})
                row["strict"], row["lenient"] = row["strict_p"] >= 0.5, row["lenient_p"] >= 0.5
            else:
                strict, _ = chat(key, args.judge, [{"role": "user", "content":
                                                    JUDGE["default"].format(q["question"], gold, hyp)}], 10)
                lenient, _ = chat(key, args.judge, [{"role": "user", "content": LENIENT_JUDGE.format(
                    question=q["question"], answer=gold, response=hyp)}], 200)
                row["strict"] = "yes" in strict.lower()
                row["lenient"] = ('"CORRECT"' in lenient or "label\": \"CORRECT" in lenient
                                  or bool(re.search(r'label"?\s*:\s*"?CORRECT', lenient)))
            with lock, open(path(arm), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    todo = [q for q in questions if any(q["id"] not in done[a] for a in arms)]
    print(f"{len(todo)} questions x {len(arms)} arms", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, q) for q in todo]), 1):
            try:
                fut.result()
            except Exception as e:
                print("  error:", repr(e)[:200], flush=True)
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} questions, {time.time() - t0:.0f}s", flush=True)
    report(questions, arms, path)


def report(questions, arms, path):
    ids = {q["id"] for q in questions}
    res = {}
    for a in arms:
        if os.path.exists(path(a)):
            res[a] = {r["id"]: r for r in map(json.loads, open(path(a), encoding="utf-8")) if r["id"] in ids}
    arms = [a for a in arms if res.get(a)]
    for judge in ("strict", "lenient"):
        print(f"\nLoCoMo accuracy, {judge} judge (%)\n{'category':>12} | " + " | ".join(f"{a:>10}" for a in arms))
        for cat in list(CATEGORY.values()) + ["ALL"]:
            cells = []
            for a in arms:
                rows = [r for r in res[a].values() if cat == "ALL" or r["category"] == cat]
                cells.append(f"{100 * np.mean([r[judge] for r in rows]):10.1f}" if rows else "         -")
            print(f"{cat:>12} | " + " | ".join(cells))
    for a in arms:
        print(f"  {a}: avg injected {np.mean([r['memory_chars'] for r in res[a].values()]):.0f} chars, "
              f"answered {len(res[a])}/{len(ids)}")
    if len(arms) > 1:
        print("\nPaired difference, strict judge (row minus column), points with bootstrap 95% CI:")
        rng = np.random.default_rng(0)
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                common = sorted(set(res[a]) & set(res[b]))
                d = np.array([int(res[b][k]["strict"]) - int(res[a][k]["strict"]) for k in common])
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                print(f"  {b} vs {a}: {100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]  (n={len(d)})")


if __name__ == "__main__":
    main()
