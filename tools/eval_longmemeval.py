"""End-to-end LongMemEval-S: does recall make the answers better?

Per question and arm, the reader model gets exactly what Hermes would send: the question with the arm's recalled
memory appended the way Hermes' compose_user_api_content() does it. The judge grades with LongMemEval's official prompts.
Same reader and judge for every arm; answers are saved as they come (resumable).

Arms:
  none     no memory
  keyword  keyword (FTS5) recall, top 5, same injection format
  gemma    the plugin's real prefetch() over its Store, with the cached EmbeddingGemma 2 vectors

    python tools/eval_longmemeval.py run --arms none,keyword,gemma --per-type 2      # pilot
    python tools/eval_longmemeval.py run --arms none,keyword,gemma                   # all 500
    python tools/eval_longmemeval.py report --arms none,keyword,gemma

Needs HERMES_AGENT_PATH, bench-data/ from bench_longmemeval.py, and XIAOMI_TOKEN_PLAN_API_KEY.
"""

import argparse
import collections
import hashlib
import http.client
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

from agent.memory_manager import build_memory_context_block  # noqa: E402
from bench_longmemeval import DATA, Embeddings, pairs, parse_date  # noqa: E402
import lme_select  # noqa: E402
import rejudge_decider  # noqa: E402
from gemma_memory.embedder import DOC_PREFIX, QUERY_PREFIX  # noqa: E402
from gemma_memory.passages import split_turn  # noqa: E402
from gemma_memory.provider import DEFAULTS, GemmaMemoryProvider, _turn_text, _when  # noqa: E402
from gemma_memory.store import Store  # noqa: E402

BASE = os.environ.get("MIMO_BASE_URL", "")  # your MiMo Token Plan endpoint
EVAL_DIR = os.path.join(DATA, "eval")

def hermes_spill(memory, source="memory prefetch"):
    """What Hermes' MemoryManager.prefetch_all() does to a provider's recall before injecting it: anything over
    hooks.output_spill.max_chars (10,000 by default) becomes a head/tail preview plus a file path. Hermes' own
    function and config; spill files go under bench-data instead of HERMES_HOME."""
    from tools.hook_output_spill import get_spill_config, spill_if_oversized
    cfg = {**get_spill_config(), "directory": os.path.join(DATA, "spill")}
    return spill_if_oversized(memory, session_id="eval", source=source, config=cfg) if memory else memory


def compose_user_api_content(content, ext_prefetch_cache, plugin_user_context=""):
    """Hermes' agent/turn_context.py, minus its heavy imports: the question, then the fenced recall block.
    The recall goes through Hermes' spill first, as in prefetch_all()."""
    ext_prefetch_cache = hermes_spill(ext_prefetch_cache)
    fenced = build_memory_context_block(ext_prefetch_cache) if ext_prefetch_cache else ""
    return content + "\n\n" + fenced if fenced else None


# LongMemEval's official answer-check prompts (src/evaluation/evaluate_qa.py, MIT).
_CHECK = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
          "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
          "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
          "only contains a subset of the information required by the answer, answer no. ")
JUDGE = {
    "default": _CHECK + "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
                        "Answer yes or no only.",
    "temporal-reasoning": _CHECK + "In addition, do not penalize off-by-one errors for the number of days. If the "
        "question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
        "predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\n"
        "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes "
        "if the response contains the correct answer. Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be considered as correct as long as the "
        "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs "
        "the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a "
        "response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer "
        "no. The model does not need to reflect all the points in the rubric. The response is correct as long as it "
        "recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel "
        "Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer "
        "yes if the model correctly identifies the question as unanswerable. The model could say that the "
        "information is incomplete, or some other information is given but the asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as "
        "unanswerable? Answer yes or no only.",
}
SYSTEM = "You are a helpful personal assistant. The current date is {date}. Answer the user's question."


def api_key():
    key = os.environ.get("XIAOMI_TOKEN_PLAN_API_KEY")
    if not key and sys.platform == "win32":  # set with setx after this process started
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            key = winreg.QueryValueEx(h, "XIAOMI_TOKEN_PLAN_API_KEY")[0]
    if not key:
        sys.exit("XIAOMI_TOKEN_PLAN_API_KEY is not set")
    return key


DEEPSEEK = "https://api.deepseek.com/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
# DeepSeek models only from DeepSeek itself: other OpenRouter providers serve fp8/fp4 copies, which would not match
# the results run on DeepSeek's own API.
OR_DEEPSEEK = {"order": ["DeepSeek"], "allow_fallbacks": False}
_OR_GAP = 3.2  # OpenRouter free models: 20 requests a minute per account, whoever sends them
_or_lock, _or_next = threading.Lock(), [0.0]


def _env(name):
    key = os.environ.get(name)
    if not key and sys.platform == "win32":  # set with setx after this process started
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
                key = winreg.QueryValueEx(h, name)[0]
        except OSError:
            pass
    return key or ""


def _or_wait():
    with _or_lock:
        now = time.time()
        wait, _or_next[0] = max(0.0, _or_next[0] - now), max(now, _or_next[0]) + _OR_GAP
    time.sleep(wait)


def chat(key, model, messages, max_tokens, thinking=False):
    """One chat completion. MiMo models via the Token Plan; deepseek-* via DeepSeek (key: DEEPSEEK_API_KEY), with
    reasoning effort high when thinking, low otherwise; vendor/model ids (e.g. nvidia/...:free) via OpenRouter
    (key: OPENROUTER_API_KEY), paced under its free-model rate limit."""
    if model.startswith("proxy:"):  # through tools/llm_proxy.py (e.g. its claude-cli route), counted as "reader"
        base, key = "http://127.0.0.1:8098/t/reader/v1", "via-proxy"
        body = {"model": model[len("proxy:"):], "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    elif "/" in model:
        base, key = OPENROUTER, _env("OPENROUTER_API_KEY")
        if "deepseek" in model:  # the same settings as DeepSeek's own API below, so results from both mix
            body = {"model": model, "messages": messages, "max_tokens": max(max_tokens, 6000),
                    "reasoning": {"effort": "high" if thinking else "low", "exclude": True},
                    "provider": OR_DEEPSEEK}
        else:
            body = {"model": model, "messages": messages,
                    "max_tokens": max(max_tokens, 6000) if thinking else max_tokens,
                    "reasoning": {"enabled": True} if thinking else {"enabled": False, "exclude": True}}
            if not thinking:
                body["temperature"] = 0
    elif model.startswith("deepseek"):
        base, key = DEEPSEEK, os.environ.get("DEEPSEEK_API_KEY", "")
        body = {"model": model, "messages": messages, "max_tokens": max(max_tokens, 6000),  # reasoning counts too
                "reasoning_effort": "high" if thinking else "low"}
    else:
        base = BASE
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "thinking": {"type": "enabled" if thinking else "disabled"}}
        if not thinking:
            body["temperature"] = 0
    data = json.dumps(body).encode()
    for attempt in range(8):
        if base == OPENROUTER and model.endswith(":free"):
            _or_wait()
        req = urllib.request.Request(base + "/chat/completions", data=data, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "X-Title": "hermes-gemma-memory eval"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.load(r)
            if "choices" in out:  # OpenRouter can answer 200 with an upstream error instead
                return (out["choices"][0]["message"].get("content") or "").strip(), out.get("usage", {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            low = detail.lower()
            if e.code not in (408, 409, 429, 500, 502, 503, 504) or "quota" in low or "insufficient" in low \
                    or "per-day" in low:  # out of credit or the free daily cap: retrying won't help
                raise RuntimeError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            pass
        time.sleep(min(60, 2 ** attempt + random.random()))
    raise RuntimeError("gave up after retries")


class _CachedQuery:
    """Stands in for the plugin's Embedder: returns the cached EmbeddingGemma 2 query vector."""

    def __init__(self, emb, dims):
        self.emb, self.dims = emb, dims

    def query(self, text, *, timeout=None):
        return self.emb.get(text, QUERY_PREFIX, self.dims).tolist()


def evidence_coverage(q, sessions, delivered):
    """Diagnostics only (labels never reach retrieval): which answer sessions' answer-bearing turns are in the text
    Hermes actually delivers. A turn counts when the opening of its user side (or, for an assistant-only answer, of
    its assistant side) appears; trimming only cuts the ends of assistant text, a spill drops everything."""
    found = []
    for sid in q["answer_session_ids"]:
        turns = [t for t, has in sessions.get(sid, []) if has] or [t for t, _ in sessions.get(sid, [])]
        # Any passage of an answer turn (its opening 100 chars) in the delivered text: works for whole-turn and
        # passage recall alike, and a span trimmed off the end of an assistant reply counts as missing.
        found.append(any(p[:100] in delivered for t in turns for p in split_turn(t) if len(p) > 20))
    return {"evidence_any": any(found), "evidence_all": bool(found) and all(found)}


_PASSAGE_EMB = None


def passage_embeddings():
    """Cached passage vectors (tools/embed_passages.py); None until that cache exists."""
    global _PASSAGE_EMB
    if _PASSAGE_EMB is None and os.path.exists(os.path.join(DATA, "emb-egm2-passages.npz")):
        _PASSAGE_EMB = Embeddings("google/embeddinggemma-2", "egm2-passages")
    return _PASSAGE_EMB


_FACTS = None


def session_facts():
    """(facts by session id from tools/extract_facts.py, their vector cache), or ({}, None) before extraction."""
    global _FACTS
    if _FACTS is None:
        name = os.environ.get("FACTS", "facts")  # facts.jsonl (MiMo-extracted) or e.g. facts-ds.jsonl (DeepSeek)
        path, cache = os.path.join(DATA, f"{name}.jsonl"), os.path.join(DATA, "emb-egm2-facts.npz")
        if os.path.exists(path) and os.path.exists(cache):
            _FACTS = ({json.loads(l)["session_id"]: json.loads(l)["facts"] for l in open(path, encoding="utf-8")},
                      Embeddings("google/embeddinggemma-2", "egm2-facts"))
        else:
            _FACTS = ({}, None)
    return _FACTS


def build_store(q, sessions, emb, dims):
    """The question's haystack stored the way the plugin stores it: each turn, then its passages, and the
    conversation's facts when they were extracted (only arms with use_facts rank them)."""
    store = Store(":memory:", model="bench", dims=dims)
    pemb = passage_embeddings()
    fact_map, femb = session_facts()
    for sid, date in zip(q["haystack_session_ids"], q["haystack_dates"]):
        for f in fact_map.get(sid, []) if femb is not None else []:
            if femb.key(DOC_PREFIX + f) in femb.vecs:
                store.add("fact", f, session_id=sid, created_at=parse_date(date), vec=femb.get(f, DOC_PREFIX, dims))
    for sid, date in zip(q["haystack_session_ids"], q["haystack_dates"]):
        ts = parse_date(date)
        for text, _ in sessions[sid]:
            turn = store.add("turn", text, session_id=sid, created_at=ts, vec=emb.get(text, DOC_PREFIX, dims))
            if pemb is not None:
                for p in split_turn(text):
                    store.add("passage", p, session_id=sid, created_at=ts, parent=turn,
                              vec=pemb.get(p, DOC_PREFIX, dims))
    return store


def _fmt(hits):
    """The plugin's prefetch() format, for arms that pick items another way."""
    if not hits:
        return ""
    lines = [f"- [#{r['id']} {r['kind']}, {_when(r['created_at'])}] {r['text']}" for r in hits]
    return "## Recalled from earlier conversations\n" + "\n".join(lines)


X_DEFAULTS = {"k": 5, "n": 20, "wv": 1.0, "wk": 1.0, "wu": 0.0, "cap": 0, "nb": 0, "budget": 0, "order": "rel",
              "ast": 0}


def _trim_assistant(text, limit):
    """Injected text only: the assistant's side of a turn cut to ``limit`` chars (the user's side is kept whole)."""
    sep = "\nAssistant: "
    if not limit or sep not in text:
        return text
    user, assistant = text.split(sep, 1)
    return user + sep + (assistant if len(assistant) <= limit else assistant[:limit].rstrip() + " …")


def parse_x(arm):
    """'x:k=10+nb=1+order=time' -> options for xrecall (unknown keys are an error, not silently ignored)."""
    opts = dict(X_DEFAULTS)
    for part in filter(None, arm[2:].split("+")):  # "+" between options: "," separates arms
        key, val = part.split("=")
        if key not in opts:
            raise ValueError(f"unknown option {key!r} in {arm}")
        opts[key] = type(opts[key])(val) if not isinstance(opts[key], str) else val
    return opts


def _stamp(ts):
    return time.strftime("%Y-%m-%d (%a) %H:%M", time.localtime(ts))


def _user_key_ranking(store, qvec, emb, dims, n):
    """Key expansion: every turn also indexed by the user's side alone; best n item ids by that similarity."""
    rows = [store.get(i) for i in range(1, store.stats()["total"] + 1)]
    keys = np.stack([emb.get(r["text"].split("\nAssistant: ")[0], DOC_PREFIX, dims) for r in rows])
    sims = keys @ np.asarray(qvec, dtype=np.float32)
    return [{**rows[j], "id": rows[j]["id"]} for j in np.argsort(-sims)[:n]]


def xrecall(q, store, qvec, o, emb=None, dims=768):
    """Configurable retrieval: vector + keyword candidates fused by weighted reciprocal rank, an optional per-session
    cap, neighbouring turns of each hit for context, and a character budget or item count. Output groups the turns
    by conversation, each under its date."""
    query = q["question"]
    vec = store.search(query, qvec, limit=o["n"], min_similarity=-1.0, max_gap=1.0)
    kw = store.search(query, None, limit=o["n"]) if o["wk"] > 0 else []
    score, item = collections.Counter(), {}
    uk = _user_key_ranking(store, qvec, emb, dims, o["n"]) if o["wu"] > 0 else []
    for weight, ranking in ((o["wv"], vec), (o["wk"], kw), (o["wu"], uk)):
        for rank, r in enumerate(ranking):
            score[r["id"]] += weight / (60 + rank)
            item[r["id"]] = r
    picked, per_session, chars = [], collections.Counter(), 0
    for item_id, _ in score.most_common():
        r = item[item_id]
        if o["cap"] and per_session[r["session_id"]] >= o["cap"]:
            continue
        ids = [item_id]
        for d in range(1, o["nb"] + 1):  # neighbours: adjacent turns of the same conversation
            for nid in (item_id - d, item_id + d):
                n = store.get(nid)
                if n and n["session_id"] == r["session_id"]:
                    ids.append(nid)
        new = [i for i in ids if i not in picked]
        cost = sum(len(_trim_assistant(store.get(i)["text"], o["ast"])) for i in new)
        if o["budget"] and chars + cost > o["budget"] and picked:
            break
        picked += new
        chars += cost
        per_session[r["session_id"]] += 1
        if not o["budget"] and per_session.total() >= o["k"]:
            break
    if not picked:
        return ""
    rows = [store.get(i) for i in picked]
    best = {}
    for rank, r in enumerate(rows):
        best.setdefault(r["session_id"], rank)
    sessions = sorted({r["session_id"] for r in rows},
                      key=(lambda s: next(r["created_at"] for r in rows if r["session_id"] == s))
                      if o["order"] == "time" else best.get)
    out = ["## Recalled from earlier conversations"]
    for s in sessions:
        turns = sorted((r for r in rows if r["session_id"] == s), key=lambda r: r["id"])
        out.append(f"### Conversation on {_stamp(turns[0]['created_at'])}")
        out += [_trim_assistant(t["text"], o["ast"]) for t in turns]
    return "\n".join(out)


_SAVED = {}
FIT_CHARS = 10000 - 800  # what gemma-memory allows itself under Hermes' spill cap (provider.SPILL_MARGIN)


def fit_under_cap(text, limit=FIT_CHARS):
    """Another system's recall given gemma-memory's treatment: whole lines in its own ranking order until the next
    one would pass the cap Hermes spills at, so nothing reaches the model as a spilled preview."""
    out, used = [], 0
    for line in text.splitlines():
        if used + len(line) + 1 > limit:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def saved_recall(arm, question_id):
    """Recall another memory system produced for this question (tools/eval_hindsight.py, tools/eval_mem0.py run in
    their own environments). Missing = that system hasn't ingested this question: an error, never an empty memory."""
    system = arm.split("-")[0]
    if system not in _SAVED:
        path = os.path.join(EVAL_DIR, f"{system}-recall.jsonl")
        _SAVED[system] = {json.loads(l)["question_id"]: json.loads(l) for l in open(path, encoding="utf-8")}
    row = _SAVED[system][question_id]
    if system == "hindsight":  # hindsight-published / hindsight-matched [/ ...-fit]
        variant = arm.split("-", 1)[1]
        text = row["recall"][variant.removesuffix("-fit")]
        return fit_under_cap(text) if variant.endswith("-fit") else text
    # Mem0's created_at is ingestion time (OSS add() takes no timestamp); dates live in the facts themselves.
    lines = [f"- {m['memory']}" for m in row["memories"]]
    text = "## Recalled facts\n" + "\n".join(lines) if lines else ""
    return fit_under_cap(text) if arm.endswith("-fit") else text


def recall(arm, q, store, emb, dims):
    """The memory text the arm injects for this question ("" = none)."""
    k = DEFAULTS["top_k"]
    if arm == "none":
        return ""
    if arm.startswith("hindsight-") or arm.split("-")[0] in ("mem0", "mem0ds"):  # mem0ds: ingested via DeepSeek
        return saved_recall(arm, q["question_id"])
    if arm == "plugin" or arm.startswith("plugin:"):  # the shipped code path: GemmaMemoryProvider.prefetch()
        overrides = {}
        for part in filter(None, arm[len("plugin:"):].split("+")):
            key, val = part.split("=")
            if key not in DEFAULTS:
                raise ValueError(f"unknown plugin setting {key!r}")
            overrides[key] = (val.lower() in ("1", "true", "yes")) if isinstance(DEFAULTS[key], bool) else type(DEFAULTS[key])(val)
        p = GemmaMemoryProvider({**DEFAULTS, "dims": dims, **overrides})
        p._store, p._embedder = store, _CachedQuery(emb, dims)
        p._bind("eval-new-session")
        return p.prefetch(q["question"])
    if arm.startswith("x:"):
        return xrecall(q, store, emb.get(q["question"], QUERY_PREFIX, dims), parse_x(arm), emb, dims)
    if arm == "keyword":
        return _fmt(store.search(q["question"], None, limit=k))
    if arm == "holographic":  # Hermes' bundled provider, passive mode: auto_extract at each session end
        import tempfile
        from plugins.memory.holographic import HolographicMemoryProvider
        tmp = tempfile.mkdtemp(prefix="holo-")
        p = HolographicMemoryProvider(config={"db_path": os.path.join(tmp, "facts.db"), "auto_extract": "true"})
        p.initialize("eval", hermes_home=tmp)
        for s in q["haystack_sessions"]:
            p.on_session_end([{"role": t["role"], "content": t["content"]} for t in s])
        out = p.prefetch(q["question"])
        p.shutdown()
        return out
    qvec = emb.get(q["question"], QUERY_PREFIX, dims)
    if arm == "gemma-top5":  # plain vector top 5: no similarity floor, no gap filter
        return _fmt(store.search(q["question"], qvec, limit=k, min_similarity=-1.0, max_gap=1.0))
    if arm == "hybrid":  # vector top 20 + keyword top 20, reciprocal rank fusion, top 5
        vec = store.search(q["question"], qvec, limit=20, min_similarity=-1.0, max_gap=1.0)
        kw = store.search(q["question"], None, limit=20)
        score, item = collections.Counter(), {}
        for ranking in (vec, kw):
            for rank, r in enumerate(ranking):
                score[r["id"]] += 1.0 / (60 + rank)
                item[r["id"]] = r
        return _fmt([item[i] for i, _ in score.most_common(k)])
    if arm == "gemma":
        p = GemmaMemoryProvider({**DEFAULTS, "dims": dims})
        p._store, p._embedder = store, _CachedQuery(emb, dims)
        p._bind("eval-new-session")  # a fresh chat: nothing in the haystack is in context
        return p.prefetch(q["question"])
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--arms", default="none,keyword,gemma")
    ap.add_argument("--per-type", type=int, default=0, help="pilot: N questions per type (0 = all 500)")
    ap.add_argument("--reader", default="mimo-v2.6-flash")
    ap.add_argument("--judge", default="decider", help="decider (Perplexity Decider v1.1, PERPLEXITY_API_KEY) or a MiMo model")
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for result files (e.g. a settings variant)")
    ap.add_argument("--think", action="store_true", help="reader thinks before answering (reasoning on)")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all",
                    help="dev = tune on these; test = held out, scored only for final numbers")
    args = ap.parse_args()
    arms = args.arms.split(",")
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = lambda arm: os.path.join(EVAL_DIR, re.sub(r"[^A-Za-z0-9_.=-]", "_", arm) + f"{args.tag}.jsonl")  # noqa: E731

    data = json.load(open(os.path.join(DATA, "longmemeval_s_cleaned.json"), encoding="utf-8"))
    qtype = lambda q: "abstention" if q["question_id"].endswith("_abs") else q["question_type"]  # noqa: E731
    data = lme_select.select(data, args.split, args.per_type)

    if args.cmd == "report":
        return report(data, arms, path, qtype)

    key = api_key()
    decider_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if args.judge == "decider" and not decider_key:
        sys.exit("--judge decider needs PERPLEXITY_API_KEY")
    sessions = {}
    for q in data:
        for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            sessions[sid] = [(_turn_text(u, a, DEFAULTS["max_chars"]), h) for u, a, h in pairs(s)]
    emb = Embeddings("google/embeddinggemma-2", "egm2")
    missing = [q["question"] for q in data if emb.key(QUERY_PREFIX + q["question"]) not in emb.vecs]
    if missing:
        sys.exit(f"{len(missing)} query vectors missing: run bench_longmemeval.py first")

    done = {arm: {json.loads(l)["question_id"] for l in open(path(arm), encoding="utf-8")} if os.path.exists(path(arm))
            else set() for arm in arms}
    lock = threading.Lock()
    usage = collections.Counter()

    def one(q):
        store = build_store(q, sessions, emb, args.dims) if any(a != "none" for a in arms) else None
        rows = []
        for arm in arms:
            if q["question_id"] in done[arm]:
                continue
            memory = recall(arm, q, store, emb, args.dims)
            delivered = hermes_spill(memory)
            coverage = evidence_coverage(q, sessions, delivered)
            user = compose_user_api_content(q["question"], memory, "") or q["question"]
            hyp, u1 = chat(key, args.reader, [{"role": "system", "content": SYSTEM.format(date=q["question_date"])},
                                             {"role": "user", "content": user}], 8000 if args.think else 800,
                           thinking=args.think)
            t = qtype(q)
            row = {"question_id": q["question_id"], "type": t, "question": q["question"], "answer": q["answer"],
                   "memory_chars": len(memory), "delivered_chars": len(delivered), **coverage, "hypothesis": hyp,
                   "reader_usage": u1}
            if args.judge == "decider":  # Perplexity Decider v1.1 with LongMemEval's rules (tools/rejudge_decider.py)
                p_correct = rejudge_decider.decide(decider_key, row)
                row.update({"verdict": f"decider p={p_correct:.3f}", "correct": p_correct >= 0.5,
                            "decider_p": p_correct, "judge_usage": {}})
            else:
                prompt = JUDGE.get(t, JUDGE["default"]).format(q["question"], q["answer"], hyp)
                verdict, u2 = chat(key, args.judge, [{"role": "user", "content": prompt}], 10)
                row.update({"verdict": verdict, "correct": "yes" in verdict.lower(), "judge_usage": u2})
            rows.append((arm, row))
        if store:
            store.close()
        with lock:
            for arm, row in rows:
                with open(path(arm), "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                usage["reader_in"] += row["reader_usage"].get("prompt_tokens", 0)
                usage["reader_out"] += row["reader_usage"].get("completion_tokens", 0)
                usage["judge_in"] += row["judge_usage"].get("prompt_tokens", 0)
                usage["judge_out"] += row["judge_usage"].get("completion_tokens", 0)
        return len(rows)

    todo = [q for q in data if any(q["question_id"] not in done[a] for a in arms)]
    print(f"{len(todo)} questions x {len(arms)} arms to run ({len(data) - len(todo)} already done)", flush=True)
    t0, n = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, q) for q in todo]), 1):
            try:
                n += fut.result()
            except Exception as e:  # keep going; a rerun retries what's missing
                print("  error:", e, flush=True)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} questions, {time.time() - t0:.0f}s, tokens {dict(usage)}", flush=True)
    print("tokens used:", dict(usage), "total", sum(usage.values()))
    report(data, arms, path, qtype)


def report(data, arms, path, qtype):
    ids = {q["question_id"] for q in data}
    res = {}
    for arm in arms:
        if os.path.exists(path(arm)):
            rows = [json.loads(l) for l in open(path(arm), encoding="utf-8")]
            res[arm] = {r["question_id"]: r for r in rows if r["question_id"] in ids}
    arms = [a for a in arms if res.get(a)]
    types = sorted({qtype(q) for q in data})
    print("\nAccuracy (% judged correct)\n" + f"{'type':>26} | " + " | ".join(f"{a:>8}" for a in arms) + " |   n")
    for t in types + ["ALL", "TASK-AVG"]:
        cells = []
        for a in arms:
            rows = [r for r in res[a].values() if t in ("ALL", "TASK-AVG") or r["type"] == t]
            if t == "TASK-AVG":
                per = [np.mean([r["correct"] for r in rows if r["type"] == tt]) for tt in types
                       if any(r["type"] == tt for r in rows)]
                cells.append(f"{100 * np.mean(per):8.1f}")
            else:
                cells.append(f"{100 * np.mean([r['correct'] for r in rows]):8.1f}" if rows else "       -")
        n = len([q for q in data if t in ("ALL", "TASK-AVG") or qtype(q) == t])
        print(f"{t:>26} | " + " | ".join(cells) + f" | {n:3d}")
    for a in arms:
        mem = [r["memory_chars"] for r in res[a].values()]
        print(f"  {a}: avg injected {np.mean(mem):.0f} chars, answered {len(res[a])}/{len(ids)}")
    if len(arms) > 1:
        print("\nPaired difference (row minus column), percentage points with bootstrap 95% CI:")
        rng = np.random.default_rng(0)
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                common = sorted(set(res[a]) & set(res[b]))
                d = np.array([int(res[b][k]["correct"]) - int(res[a][k]["correct"]) for k in common])
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                print(f"  {b} vs {a}: {100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]  "
                      f"({(d > 0).sum()} won, {(d < 0).sum()} lost, n={len(d)})")


if __name__ == "__main__":
    main()
