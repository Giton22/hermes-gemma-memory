"""LongMemEval-S retrieval benchmark for gemma-memory, through the plugin's own Store and turn format.

Embeds every unique turn once with EmbeddingGemma 2 (sentence-transformers, GPU if available; cached under
bench-data/), then for each question builds that question's store from its ~50 haystack sessions and scores
what search returns. No LLM involved: this measures retrieval, LongMemEval's own retrieval metric.

    python tools/bench_longmemeval.py --dims 768 [--limit 100] [--query-prompt "task: question answering | query: "]
"""

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("HERMES_AGENT_PATH"):  # the provider module imports Hermes
    sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

from gemma_memory.embedder import DOC_PREFIX, QUERY_PREFIX  # noqa: E402
from gemma_memory import recall  # noqa: E402
from gemma_memory.provider import DEFAULTS, _turn_text  # noqa: E402
from gemma_memory.store import Store  # noqa: E402

DATA = os.path.join(ROOT, "bench-data")


def pairs(session):
    """(user, assistant, has_answer) for each exchange, the unit the plugin stores (sync_turn)."""
    out, i = [], 0
    while i < len(session):
        if session[i]["role"] == "user":
            user = session[i]
            nxt = session[i + 1] if i + 1 < len(session) and session[i + 1]["role"] == "assistant" else None
            out.append((user["content"], nxt["content"] if nxt else "",
                        bool(user.get("has_answer") or (nxt and nxt.get("has_answer")))))
            i += 2 if nxt else 1
        else:  # assistant without a preceding user message
            out.append(("", session[i]["content"], bool(session[i].get("has_answer"))))
            i += 1
    return out


def parse_date(s):
    return datetime.strptime(s.split(" (")[0] + s.split(")")[-1], "%Y/%m/%d %H:%M").timestamp()


class Embeddings:
    """sha1(text) -> vector cache on disk, filled in batches by sentence-transformers."""

    def __init__(self, model_name, tag):
        self.path = os.path.join(DATA, f"emb-{tag}.npz")
        self.vecs = {}
        if os.path.exists(self.path):
            z = np.load(self.path)
            self.vecs = dict(zip(z["keys"].tolist(), z["vecs"]))
        self.model_name, self.model = model_name, None

    @staticmethod
    def key(text):
        return hashlib.sha1(text.encode()).hexdigest()

    def fill(self, texts, prefix, char_budget=48_000, save_every=60):
        """Embed missing texts, shortest first, in chunks of about ``char_budget`` characters (so a batch of
        8,000-character turns stays as small in GPU memory as a batch of short ones). Saved after every chunk."""
        todo = sorted({t for t in texts if self.key(prefix + t) not in self.vecs}, key=len)
        if not todo:
            return
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=dev,
                                             model_kwargs={"torch_dtype": torch.bfloat16} if dev == "cuda" else {})
            print(f"model on {dev}", flush=True)
        t0 = time.time()
        last_save, i = time.time(), 0
        while i < len(todo):
            batch = max(1, min(64, char_budget // max(1, len(todo[min(i + 63, len(todo) - 1)]))))
            chunk = todo[i:i + batch]
            vecs = self.model.encode([prefix + t for t in chunk], batch_size=batch, normalize_embeddings=True)
            for t, v in zip(chunk, vecs):
                self.vecs[self.key(prefix + t)] = v.astype(np.float16)
            i += len(chunk)
            if time.time() - last_save > save_every or i == len(todo):
                self.save()
                last_save = time.time()
                print(f"  embedded {i}/{len(todo)} ({time.time() - t0:.0f}s, batch {batch})", flush=True)

    def get(self, text, prefix, dims):
        v = self.vecs[self.key(prefix + text)].astype(np.float32)[:dims]
        return v / (np.linalg.norm(v) or 1.0)

    def save(self):
        keys = list(self.vecs)
        np.savez(self.path, keys=np.array(keys), vecs=np.stack([self.vecs[k] for k in keys]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(DATA, "longmemeval_s_cleaned.json"))
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (0 = all)")
    ap.add_argument("--query-prompt", default=QUERY_PREFIX)
    ap.add_argument("--budget", type=int, default=DEFAULTS["recall_budget"])
    ap.add_argument("--assistant-chars", type=int, default=DEFAULTS["assistant_chars"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = json.load(open(args.data, encoding="utf-8"))
    if args.limit:
        data = data[:args.limit]
    max_chars = DEFAULTS["max_chars"]
    sessions = {}
    for q in data:
        for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            sessions[sid] = [(_turn_text(u, a, max_chars), h) for u, a, h in pairs(s)]

    emb = Embeddings(args.model, "egm2")
    emb.fill([t for s in sessions.values() for t, _ in s], DOC_PREFIX)
    emb.fill([q["question"] for q in data], args.query_prompt)
    emb.save()

    ks = (1, 3, 5, 10)
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    t0 = time.time()
    for q in data:
        qtype = q["question_type"] + ("_abs" if q["question_id"].endswith("_abs") else "")
        store = Store(":memory:", model="bench", dims=args.dims)
        owner, answer_turn = {}, set()
        for sid, date in zip(q["haystack_session_ids"], q["haystack_dates"]):
            for text, has_answer in sessions[sid]:
                item = store.add("turn", text, session_id=sid, created_at=parse_date(date),
                                 vec=emb.get(text, DOC_PREFIX, args.dims))
                owner[item] = sid
                if has_answer:
                    answer_turn.add(item)
        qvec = emb.get(q["question"], args.query_prompt, args.dims)
        gold = set(q["answer_session_ids"])
        fused = store.fused(q["question"], qvec, candidates=DEFAULTS["candidates"])
        for mode, vec in (("vector", qvec), ("keyword", None), ("fused", "fused")):
            ranked = ([r["id"] for r in fused] if vec == "fused" else
                      [r["id"] for r in store.search(q["question"], vec, limit=max(ks), min_similarity=-1.0)])
            if qtype.endswith("_abs"):
                continue
            for k in ks:
                got = {owner[i] for i in ranked[:k]}
                stats[(mode, qtype)][f"any@{k}"].append(bool(got & gold))
                stats[(mode, qtype)][f"all@{k}"].append(gold <= got)
            if answer_turn:
                stats[(mode, qtype)]["turn@5"].append(bool(set(ranked[:5]) & answer_turn))
        # What automatic recall (prefetch) injects with the plugin's budget.
        injected = recall.select(fused, budget=args.budget, assistant_chars=args.assistant_chars)
        stats[("prefetch", qtype)]["n"].append(len(injected))
        if not qtype.endswith("_abs"):
            got = {owner[r["id"]] for r in injected}
            stats[("prefetch", qtype)]["hit"].append(bool(got & gold))
            stats[("prefetch", qtype)]["all"].append(gold <= got)
        store.close()
    print(f"\nsearched {len(data)} questions in {time.time() - t0:.0f}s, dims {args.dims}, "
          f"query prompt {args.query_prompt!r}\n")

    def table(mode, cols):
        types = sorted({t for m, t in stats if m == mode})
        rows = [(t, stats[(mode, t)]) for t in types]
        total = collections.defaultdict(list)
        for _, s in rows:
            for c in cols:
                total[c] += s.get(c, [])
        print(f"{mode:>8} | " + " | ".join(f"{c:>7}" for c in cols) + " |   n")
        for name, s in rows + [("ALL", total)]:
            vals = [s.get(c, []) for c in cols]
            n = max((len(v) for v in vals), default=0)
            cells = [f"{np.mean(v) * (1 if c == 'n' else 100):7.1f}" if v else "      -" for c, v in zip(cols, vals)]
            print(f"{name[:26]:>26} | " + " | ".join(cells) + f" | {n:3d}")
        print()

    cols = ["any@1", "any@5", "all@5", "all@10", "turn@5"]
    print("Session recall (% of questions; any = an answer session found, all = every answer session found)")
    table("vector", cols)
    table("keyword", cols)
    table("fused", cols)
    print(f"Automatic recall, budget {args.budget} chars, assistant side {args.assistant_chars} chars "
          "(n = items injected, hit / all = % with an / every answer session among them)")
    table("prefetch", ["n", "hit", "all"])
    if args.out:
        json.dump({f"{m}|{t}": {c: float(np.mean(v)) for c, v in s.items() if v} for (m, t), s in stats.items()},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
