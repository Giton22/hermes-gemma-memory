"""A stored turn split into short passages, one speaker each, so recall can deliver the answer-bearing span instead
of a whole turn: more conversations fit under Hermes' prefetch cap, and a fact deep in a long assistant reply can be
found without keeping the whole reply."""

from __future__ import annotations

import re
from typing import List

SEP = "\nAssistant: "
TARGET = 350  # characters per passage, roughly a few sentences
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _chunks(text: str, target: int) -> List[str]:
    """Consecutive sentences packed up to ``target`` chars; a single overlong sentence is cut at word boundaries."""
    out: List[str] = []
    cur = ""
    for piece in filter(None, (p.strip() for p in _SENTENCE.split(text))):
        while len(piece) > target:
            cut = piece.rfind(" ", 0, target)
            cut = cut if cut > target // 2 else target
            if cur:
                out.append(cur)
                cur = ""
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if cur and len(cur) + 1 + len(piece) > target:
            out.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        out.append(cur)
    return out


def split_turn(turn_text: str, target: int = TARGET) -> List[str]:
    """'User: …\\nAssistant: …' -> ['User: …', …, 'Assistant: …', …]. Other texts (notes) -> speaker-less chunks."""
    if not turn_text.startswith("User: "):
        return _chunks(turn_text, target)
    user, _, assistant = turn_text[len("User: "):].partition(SEP)
    return ([f"User: {c}" for c in _chunks(user, target)] +
            ([f"Assistant: {c}" for c in _chunks(assistant, target)] if assistant else []))
