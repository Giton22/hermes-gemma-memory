"""What gets injected before a reply: ranked items cut to a character budget, grouped by conversation under its date.

Measured on LongMemEval-S (dev half, tools/eval_longmemeval.py): recalling more conversations with the assistant's
side trimmed beat a fixed top 5 by ~11 points; the user's own words carry most personal facts, the assistant's long
answers mostly fill space. The best-ranked few turns can be kept whole (``full_turns``), so an answer that sits deep
in the assistant's reply is still there when it's the closest match.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List

from . import dates

SEP = "\nAssistant: "
HEADER = "## Recalled from earlier conversations"


def trim_assistant(text: str, limit: int) -> str:
    """The assistant's side of a turn cut to ``limit`` chars; the user's side is kept whole. 0 = no trim."""
    if not limit or SEP not in text:
        return text
    user, assistant = text.split(SEP, 1)
    return user + SEP + (assistant if len(assistant) <= limit else assistant[:limit].rstrip() + " …")


def select(ranked: Iterable[dict], *, budget: int, assistant_chars: int, full_turns: int = 0,
           per_session: int = 0) -> List[dict]:
    """Best-first items until the next one would pass ``budget`` characters (always at least one). The first
    ``full_turns`` picks keep their whole text; the rest are trimmed. Each picked row gets ``shown``: its text as
    it will be injected."""
    picked, used, count, seen = [], 0, {}, set()
    for row in ranked:
        if row["id"] in seen:  # an expansion (see expand) may repeat an item ranked later on its own
            continue
        seen.add(row["id"])
        sid = row.get("session_id", "")
        if per_session and count.get(sid, 0) >= per_session:
            continue
        shown = row["text"] if len(picked) < full_turns else trim_assistant(row["text"], assistant_chars)
        if picked and used + len(shown) > budget:
            break
        picked.append({**row, "shown": shown})
        used += len(shown)
        count[sid] = count.get(sid, 0) + 1
    return picked


def expand(ranked: Iterable[dict], get, children, *, with_question: bool, context: int) -> Iterable[dict]:
    """Each ranked passage followed by the context the reader needs to use it: the user's words of the same turn
    when the hit is the assistant's reply (an answer is ambiguous without its question), and ``context`` neighbouring
    passages of that turn on each side. ``get(id)`` -> row, ``children(turn_id)`` -> passage ids in order."""
    for row in ranked:
        yield row
        if row.get("kind") != "passage" or not row.get("parent"):
            continue
        siblings = children(row["parent"])
        pos = siblings.index(row["id"]) if row["id"] in siblings else -1
        extra = []
        if with_question and row["text"].startswith("Assistant: "):
            extra += [i for i in siblings if (get(i) or {}).get("text", "").startswith("User: ")][:1]
        if context and pos >= 0:
            extra += [siblings[j] for j in range(max(0, pos - context), min(len(siblings), pos + context + 1)) if j != pos]
        for i in extra:
            r = get(i)
            if r:
                yield r


def stamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d (%a) %H:%M", time.localtime(ts))


def render(rows: List[dict], *, assistant_chars: int = 0, anchor_dates: bool = False) -> str:
    """Items grouped by conversation (best conversation first), turns in their original order, each group under the
    date it happened; notes and images keep their id so the agent can refer to them (memory_forget)."""
    if not rows:
        return ""
    first: Dict[str, int] = {}
    for rank, r in enumerate(rows):
        first.setdefault(r.get("session_id", ""), rank)
    out = [HEADER]
    for sid in sorted(first, key=first.get):
        group = sorted((r for r in rows if r.get("session_id", "") == sid), key=lambda r: r["id"])
        out.append(f"### Conversation on {stamp(group[0]['created_at'])}")
        prev = None
        for r in group:
            kind = r.get("kind")
            if kind == "passage":  # spans of turns: mark where text between them was left out
                if prev is not None and r["id"] != prev + 1:
                    out.append("…")
                out.append(dates.anchor(r["text"], r["created_at"]) if anchor_dates else r["text"])
                prev = r["id"]
                continue
            prev = None
            if kind == "fact":  # derived from this conversation (gemma_memory.facts)
                out.append(f"Fact: {r['text']}")
                continue
            text = r.get("shown") or trim_assistant(r["text"], assistant_chars)
            if anchor_dates:
                text = dates.anchor(text, r["created_at"])
            out.append(text if kind == "turn" else f"[#{r['id']} {kind}] {text}")
    return "\n".join(out)
