"""Can the memory recall a picture from a text question? A synthetic, download-free test of the image path.

Images are generated from fixed templates with seeded random values (receipts, error screens, Wi-Fi notes,
boarding passes, shape scenes). Each is sent through GemmaMemoryProvider.sync_turn() as a chat image with only a
vague caption ("look at this"), so the facts exist only in the pixels, amid a real LongMemEval history (text
distractors). For each image a natural question asks for one fact in it.

  retrieval  is the image among what prefetch() delivers (after Hermes' spill)?  pixels vs caption-only control
  reading    reader gets prefetch()'s text plus the recalled images attached (a vision-capable Hermes opening the
             file), the judge checks the answer  (MiMo flash is multimodal)

    python tools/eval_images.py --n 40            # needs tools/embed_server.py (8099) and the MiMo key
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

import lme_select  # noqa: E402
from eval_longmemeval import DATA, JUDGE, SYSTEM, api_key, chat, compose_user_api_content, hermes_spill  # noqa: E402
from bench_longmemeval import pairs  # noqa: E402
from gemma_memory.provider import DEFAULTS, GemmaMemoryProvider, _save_image  # noqa: E402

FONT = ImageFont.truetype("arial.ttf", 28)
SMALL = ImageFont.truetype("arial.ttf", 22)
SHOPS = ["HARDWARE STORE", "GARDEN CENTER", "PET SUPPLIES", "BIKE SHOP", "BAKERY", "PHARMACY"]
ITEMS = ["drill bits", "screws", "potting soil", "dog food", "inner tube", "sourdough", "plasters", "paint brush",
         "chain oil", "bird seed", "croissants", "vitamins"]
CITIES = ["Vienna", "Lisbon", "Oslo", "Prague", "Dublin", "Madrid", "Zurich", "Athens"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black"]
SHAPES = ["circle", "square", "triangle"]


def _png(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _doc(lines, bg="white", fg="black"):
    im = Image.new("RGB", (640, 80 + 46 * len(lines)), bg)
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((30, 30 + i * 46), line, fill=fg, font=FONT)
    return im


def make_case(i, rng):
    """(data_uri, question, answer) for one generated image."""
    kind = i % 5
    if kind == 0:
        shop, (a, b) = rng.choice(SHOPS), rng.sample(ITEMS, 2)
        pa, pb = rng.randint(2, 40) + 0.99, rng.randint(2, 40) + 0.49
        im = _doc([shop, f"{a:<16}{pa:6.2f}", f"{b:<16}{pb:6.2f}", f"TOTAL EUR {pa + pb:8.2f}"])
        return _png(im), f"How much did the {a} cost on that {shop.lower()} receipt I showed you?", f"{pa:.2f} EUR"
    if kind == 1:
        code, port = rng.randint(100, 999), rng.randint(1024, 65000)
        svc = rng.choice(["nginx", "postgres", "redis", "jellyfin", "immich"])
        im = _doc([f"{svc} failed to start", f"Error E{code}: address already in use", f"bind 0.0.0.0:{port}"],
                  bg="black", fg="lime")
        return _png(im), f"Which port was {svc} complaining about in that error screenshot?", str(port)
    if kind == 2:
        ssid = f"{rng.choice(['Attic', 'Garage', 'Studio', 'Cellar'])}-{rng.randint(10, 99)}"
        chan = rng.choice([1, 6, 11, 36, 44, 149])
        im = _doc(["Wi-Fi settings", f"SSID: {ssid}", f"Channel: {chan}", "Security: WPA3"])
        return _png(im), f"What channel is the {ssid} Wi-Fi network on, from the settings picture?", str(chan)
    if kind == 3:
        city, gate = rng.choice(CITIES), f"{rng.choice('ABCD')}{rng.randint(1, 40)}"
        im = _doc(["BOARDING PASS", f"To: {city.upper()}", f"Gate {gate}", f"Seat {rng.randint(1, 30)}{rng.choice('ACDF')}"],
                  bg="lightyellow")
        return _png(im), f"What gate was on my boarding pass to {city}?", gate
    color, shape, bg = rng.choice(COLORS), rng.choice(SHAPES), rng.choice(["white", "lightgray", "beige"])
    count = rng.randint(2, 5)
    im = Image.new("RGB", (512, 384), bg)
    d = ImageDraw.Draw(im)
    for k in range(count):
        x, y = 40 + k * 95, 150
        box = (x, y, x + 80, y + 80)
        if shape == "circle":
            d.ellipse(box, fill=color)
        elif shape == "square":
            d.rectangle(box, fill=color)
        else:
            d.polygon([(x + 40, y), (x, y + 80), (x + 80, y + 80)], fill=color)
    d.text((20, 20), "sketch for the logo", fill="black", font=SMALL)
    return _png(im), f"How many {color} {shape}s were in the logo sketch I sent?", str(count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--embed-url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--reader", default="mimo-v2.6-flash")
    ap.add_argument("--judge", default="mimo-v2.6-pro")
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()
    rng = random.Random(7)
    cases = [make_case(i, rng) for i in range(args.n)]
    # A real history as distractors: one held-out LongMemEval haystack.
    q0 = lme_select.select(json.load(open(os.path.join(DATA, "longmemeval_s_cleaned.json"), encoding="utf-8")), "test")[0]
    results = {}
    for mode in ("pixels", "caption-only"):
        home = tempfile.mkdtemp(prefix=f"img-{mode}-")
        p = GemmaMemoryProvider({**DEFAULTS, "base_url": args.embed_url})
        p.initialize("hist", hermes_home=home)
        for sid, s in zip(q0["haystack_session_ids"], q0["haystack_sessions"]):
            p.on_session_switch(sid)
            for u, a, _ in pairs(s):
                p.sync_turn(u, a, session_id=sid)
        p._worker.join()
        for i, (uri, _, _) in enumerate(cases):
            sid = f"img{i}"
            p.on_session_switch(sid)
            msgs = [{"role": "user", "content": [{"type": "text", "text": "look at this"},
                                                 {"type": "image_url", "image_url": {"url": uri}}]},
                    {"role": "assistant", "content": "Got it, I'll keep that in mind."}]
            if mode == "caption-only":  # control: same turn, the picture's file kept but no pixel vector
                p.sync_turn("look at this", "Got it, I'll keep that in mind.", session_id=sid)
                p._worker.join()
                path = _save_image(uri, p._images_dir)
                text = f"Image shared in a conversation: look at this [file: {path}]"
                item = p._store.add("image", text, session_id=sid)
                p._store.set_vec(item, p._embedder.documents([text])[0])  # the words only, never the pixels
            else:
                p.sync_turn("look at this", "Got it, I'll keep that in mind.", session_id=sid, messages=msgs)
        p._worker.join()
        p.on_session_switch("ask", reset=True)
        rows = []
        for i, (uri, question, answer) in enumerate(cases):
            block = hermes_spill(p.prefetch(question))
            files = [ln.split("[file: ")[1].rstrip("]") for ln in block.splitlines() if "[file: " in ln]
            want = _save_image(uri, p._images_dir)  # content-hash name: the file sync_turn saved
            rows.append({"i": i, "question": question, "answer": answer, "found": str(want) in block,
                         "block": block, "files": files})
        results[mode] = rows
        print(f"{mode}: image delivered for {sum(r['found'] for r in rows)}/{len(rows)} questions", flush=True)
        p.shutdown()
    if args.retrieval_only:
        return
    key = api_key()
    for mode, rows in results.items():
        correct = 0
        for r in rows:
            user_text = compose_user_api_content(r["question"], r["block"]) or r["question"]
            content = [{"type": "text", "text": user_text}] + [
                {"type": "image_url", "image_url": {"url": _file_uri(f)}} for f in r["files"][:4] if os.path.exists(f)]
            hyp, _ = chat(key, args.reader, [{"role": "system", "content": SYSTEM.format(date="2026-10-07")},
                                            {"role": "user", "content": content}], 8000, thinking=True)
            verdict, _ = chat(key, args.judge, [{"role": "user", "content":
                                                 JUDGE["default"].format(r["question"], r["answer"], hyp)}], 10)
            r["hypothesis"], r["correct"] = hyp, "yes" in verdict.lower()
            correct += r["correct"]
        print(f"{mode}: answered correctly {correct}/{len(rows)}", flush=True)
    json.dump(results, open(os.path.join(DATA, "eval", "images.json"), "w", encoding="utf-8"), indent=1)


def _file_uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


if __name__ == "__main__":
    main()
