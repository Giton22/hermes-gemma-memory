"""Second, independent judge: re-grade saved answers with Perplexity's Decider v1.1 (a decision model from another
family than the MiMo reader/judge), using LongMemEval's own per-type grading rules as its instructions.

For each results file: MiMo-pro accuracy, Decider accuracy (p >= 0.5), agreement, and how many Decider verdicts are
unsure (0.2 < p < 0.8). Verdicts are cached in bench-data/decider/. Key from PERPLEXITY_API_KEY (environment only).

    python tools/rejudge_decider.py bench-data/eval/plugin-spill.jsonl bench-data/eval/mem0-fit-spill.jsonl ...
    python tools/rejudge_decider.py --compare --split test --per-type 2 plugin mem0-fit hindsight-published-fit
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "bench-data", "decider")
URL = "https://api.perplexity.ai/v1/decisions"
MODEL = "pplx-decider-v1.1-27b"

_BASE = ("Answer yes if the model response contains the correct answer. If the response is equivalent to the correct "
         "answer or contains all the intermediate steps to get the correct answer, also answer yes. If the response "
         "only contains a subset of the information required by the answer, answer no.")
RULES = {  # LongMemEval's grading rules (src/evaluation/evaluate_qa.py), as instructions
    "default": _BASE,
    "temporal-reasoning": _BASE + " Do not penalize off-by-one errors for the number of days: if the question asks "
                                  "for a number of days/weeks/months and the response is off by one, it is still correct.",
    "knowledge-update": "Answer yes if the model response contains the correct answer. If the response contains some "
                        "previous information along with an updated answer, it is correct as long as the updated answer "
                        "is the required answer.",
    "single-session-preference": "The correct_answer is a rubric for a desired personalized response. Answer yes if "
                                 "the response satisfies it. It does not need to reflect every point in the rubric; it is "
                                 "correct as long as it recalls and uses the user's personal information correctly.",
    "abstention": "The question is unanswerable and correct_answer explains why. Answer yes if the model response "
                  "correctly identifies the question as unanswerable: it may say the information is incomplete, or "
                  "that some other information is given but the asked information is not.",
}


def _key(row):
    return hashlib.sha1(json.dumps([row["question"], str(row["answer"]), row["hypothesis"]]).encode()).hexdigest()


def decide(api_key, row):
    rule = row.get("rule") or RULES.get(row.get("type", "default"), _BASE)  # "rule": a benchmark's own wording
    body = {"model": MODEL, "state": {"question": row["question"], "correct_answer": str(row["answer"]),
                                      "model_response": row["hypothesis"]},
            "questions": {"correct": {"type": "noul", "instructions": rule}}}
    for attempt in range(6):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["answers"]["correct"]["noul"]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]}")
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError("gave up")


def judge_file(path, api_key, workers=8):
    """{row id: (row, p_correct)} for every row in a results file, using and filling the cache."""
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, "verdicts.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            d = json.loads(line)
            cache[d["key"]] = d["p"]
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    lock = threading.Lock()

    def one(row):
        k = _key(row)
        if k not in cache:
            p = decide(api_key, row)
            with lock:
                cache[k] = p
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": k, "p": p}) + "\n")
        return row, cache[k]

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, rows))
    return {r.get("question_id") or r.get("id"): (r, p) for r, p in out}


def mimo_verdict(row):
    return row["correct"] if "correct" in row else row["strict"]


def summarize(path, judged):
    m = np.array([mimo_verdict(r) for r, _ in judged.values()], dtype=float)
    p = np.array([p for _, p in judged.values()])
    d = (p >= 0.5).astype(float)
    po = (m == d).mean()
    pe = m.mean() * d.mean() + (1 - m.mean()) * (1 - d.mean())
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    unsure = ((p > 0.2) & (p < 0.8)).sum()
    print(f"{os.path.basename(path):60} n={len(p):4d}  MiMo {100 * m.mean():5.1f}  Decider {100 * d.mean():5.1f}  "
          f"agree {100 * po:5.1f}%  kappa {kappa:.2f}  unsure {unsure}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--compare", nargs="*", help="arms (eval_longmemeval names) to compare paired under Decider")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-type", type=int, default=0)
    ap.add_argument("--tag", default="-spill")
    args = ap.parse_args()
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        sys.exit("PERPLEXITY_API_KEY is not set")
    for f in args.files:
        summarize(f, judge_file(f, api_key))
    if args.compare:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import re
        import lme_select
        data = json.load(open(os.path.join(ROOT, "bench-data", "longmemeval_s_cleaned.json"), encoding="utf-8"))
        ids = {q["question_id"] for q in lme_select.select(data, args.split, args.per_type)}
        res = {}
        for arm in args.compare:
            f = os.path.join(ROOT, "bench-data", "eval", re.sub(r"[^A-Za-z0-9_.=-]", "_", arm) + f"{args.tag}.jsonl")
            res[arm] = {k: v for k, v in judge_file(f, api_key).items() if k in ids}
        common = sorted(set.intersection(*(set(v) for v in res.values())))
        print(f"\nDecider v1.1 verdicts on {len(common)} shared questions ({args.split}, per-type {args.per_type or 'all'}):")
        for arm in args.compare:
            d = [res[arm][k][1] >= 0.5 for k in common]
            m = [mimo_verdict(res[arm][k][0]) for k in common]
            print(f"  {arm:28} Decider {100 * np.mean(d):5.1f}%   MiMo-pro {100 * np.mean(m):5.1f}%")
        rng = np.random.default_rng(0)
        base = args.compare[0]
        for arm in args.compare[1:]:
            diff = np.array([int(res[base][k][1] >= 0.5) - int(res[arm][k][1] >= 0.5) for k in common])
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  {base} vs {arm}: {100 * diff.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]  "
                  f"({(diff > 0).sum()} won, {(diff < 0).sum()} lost)")


if __name__ == "__main__":
    main()
