"""Past Hermes conversations as the turns sync_turn() would have stored: each user message with the assistant's
final reply to it (tool calls, tool results and in-between assistant steps left out), at the user message's time."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple


def _text(content: Any) -> str:
    """A stored message's text; content saved as a JSON list of parts keeps only its text parts."""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    if isinstance(content, str) and content.startswith("[") and '"type"' in content:
        try:
            return _text(json.loads(content))
        except ValueError:
            pass
    return content or ""


def turns(messages: Iterable[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    """(user text, final assistant reply, user message timestamp) in order; a user message with no reply is dropped."""
    out: List[Tuple[str, str, float]] = []
    user, reply, when = None, "", 0.0
    for m in messages:
        if m.get("_compressed_summary"):  # a compression summary, not something said
            continue
        role = m.get("role")
        if role == "user":
            if user is not None and reply.strip():
                out.append((user, reply, when))
            user, reply, when = _text(m.get("content")), "", float(m.get("timestamp") or 0.0)
        elif role == "assistant" and user is not None:
            text = _text(m.get("content"))
            if text.strip():
                reply = text  # the last assistant text of the turn is its answer
    if user is not None and reply.strip():
        out.append((user, reply, when))
    return out
