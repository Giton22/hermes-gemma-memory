"""LoCoMo data access shared by the LoCoMo runners (no numpy or Hermes imports, so every venv can use it)."""

import json
import os
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "bench-data")
OUT_DIR = os.path.join(DATA, "eval-locomo")
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}  # 5 = adversarial, excluded


def conversations():
    return json.load(open(os.path.join(DATA, "locomo10.json"), encoding="utf-8"))


def parse_when(s):
    """'1:56 pm on 8 May, 2023' -> datetime."""
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")


def line(t):
    text = t.get("text", "")
    if t.get("blip_caption"):
        text += f" [shares a photo: {t['blip_caption']}]"
    return f"{t['speaker']}: {text}"


def sessions(conv):
    """[(date string, [{speaker, line}])] in session order."""
    nums = sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
    return [(conv[f"session_{n}_date_time"], [{"speaker": t["speaker"], "line": line(t)} for t in conv[f"session_{n}"]])
            for n in nums]


def last_date(conv):
    return sessions(conv)[-1][0]


def questions_of(convs, ci):
    """Every scored question (categories 1-4) of conversation ci, with the ids tools/eval_locomo.py uses."""
    c = convs[ci]
    return [{**q, "conv": ci, "id": f"{c['sample_id']}-{qi}"} for qi, q in enumerate(c["qa"])
            if q.get("category") in CATEGORY and "answer" in q]
