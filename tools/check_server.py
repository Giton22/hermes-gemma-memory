"""Check a running embeddings server (llama.cpp on the NAS) against the vectors the benchmarks used.

The benchmarks embedded with sentence-transformers (bf16/fp32 weights); the NAS serves a Q8_0 GGUF through
llama-server. This sends the plugin's own requests (gemma_memory.embedder) and reports:
  - agreement: cosine between the server's vector and the benchmark's for the same LongMemEval questions and turns
  - ranking: whether each question still finds the same best turns among a shared pool
  - images: whether three drawn pictures match their descriptions (needs --mmproj on the server)
  - speed: one query (what a recall waits for) and a batch of turns (what storing a turn costs)

    python tools/check_server.py --url http://<server-ip>:8091/v1
"""

import argparse
import base64
import io
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
if os.environ.get("HERMES_AGENT_PATH"):
    sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

import lme_select  # noqa: E402
from bench_longmemeval import DATA, Embeddings, pairs  # noqa: E402
from gemma_memory.embedder import DOC_PREFIX, QUERY_PREFIX, EmbedError, Embedder  # noqa: E402
from gemma_memory.provider import DEFAULTS, _turn_text  # noqa: E402


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def drawings():
    from PIL import Image, ImageDraw
    out = {}
    for name, draw in (("a red circle", lambda d: d.ellipse((40, 40, 216, 216), fill=(220, 30, 30))),
                       ("a blue square", lambda d: d.rectangle((48, 48, 208, 208), fill=(30, 60, 220))),
                       ("a green triangle", lambda d: d.polygon([(128, 30), (226, 220), (30, 220)], fill=(30, 170, 60)))):
        img = Image.new("RGB", (256, 256), "white")
        draw(ImageDraw.Draw(img))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        out[name] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=20, help="questions (each brings its best turns to the pool)")
    ap.add_argument("--model", default=DEFAULTS["model"], help="model name the server expects (Ollama needs it)")
    args = ap.parse_args()
    server = Embedder(args.url, args.model, 768, timeout=120)

    ref = Embeddings("google/embeddinggemma-2", "egm2")
    data = lme_select.select(json.load(open(os.path.join(DATA, "longmemeval_s_cleaned.json"), encoding="utf-8")),
                             "test", 0)[: args.n]
    questions = [q["question"] for q in data]
    turns = []
    for q in data:  # the answer-bearing turns plus a few others, as a shared pool
        ts = [_turn_text(u, a, DEFAULTS["max_chars"]) for s in q["haystack_sessions"] for u, a, _ in pairs(s)]
        hit = [_turn_text(u, a, DEFAULTS["max_chars"]) for s in q["haystack_sessions"] for u, a, h in pairs(s) if h]
        turns += hit[:2] + ts[:3]
    turns = [t for t in dict.fromkeys(turns) if ref.key(DOC_PREFIX + t) in ref.vecs]
    print(f"{len(questions)} questions, {len(turns)} turns with benchmark vectors", flush=True)

    t0 = time.time()
    try:
        server_q = [server.query(q) for q in questions]
    except EmbedError as e:
        sys.exit(f"server unreachable or refused the request: {e}")
    q_times = []
    for q in questions[:5]:
        t = time.time()
        server.query(q)
        q_times.append(time.time() - t)
    server_d = []
    d_t = time.time()
    for i in range(0, len(turns), 8):
        server_d += server.documents(turns[i:i + 8])
    d_time = time.time() - d_t
    ref_q = [ref.vecs[ref.key(QUERY_PREFIX + q)] for q in questions]
    ref_d = [ref.vecs[ref.key(DOC_PREFIX + t)] for t in turns]

    agree = [cos(a, b) for a, b in zip(server_q + server_d, ref_q + ref_d)]
    print(f"\nagreement with the benchmark vectors: mean cosine {statistics.mean(agree):.4f}, "
          f"min {min(agree):.4f}  (Q8_0 vs full weights: expect > 0.98)")

    S, R = np.asarray(server_q) @ np.asarray(server_d).T, np.asarray(ref_q, dtype=np.float32) @ np.asarray(ref_d, dtype=np.float32).T
    top1 = np.mean(S.argmax(1) == R.argmax(1)) * 100
    top5 = np.mean([len(set(np.argsort(-S[i])[:5]) & set(np.argsort(-R[i])[:5])) / 5 for i in range(len(S))]) * 100
    print(f"ranking over the {len(turns)}-turn pool: same best turn {top1:.0f}%, top-5 overlap {top5:.0f}%")

    try:
        pics = drawings()
        img = server.images(list(pics.values()))
        txt = server.query  # images are compared with queries, as recall does
        right = sum(int(np.argmax([cos(v, txt(name)) for name in pics]) == i) for i, v in enumerate(img))
        print(f"images: {right}/3 drawings matched their description, vectors {len(img[0])}-dim")
    except EmbedError as e:
        print(f"images: not embedded ({e}); start llama-server with its mmproj for picture memory")

    print(f"\nspeed: one query {statistics.median(q_times) * 1000:.0f} ms (recall waits for this; timeout "
          f"{DEFAULTS['prefetch_timeout']:.0f} s), storing turns {d_time / len(turns) * 1000:.0f} ms each "
          f"(~8 rows per turn with passages)")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
