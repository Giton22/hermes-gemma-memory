"""Optional derived memory: a few short, dated facts per conversation, extracted by an LLM once per session.

Facts are an index into the evidence, not a replacement for it: each is stored next to the conversation's passages
(same session, its date), ranked with them, and the passages stay. One call per conversation, at session end, keeps
the cost far below per-exchange extraction (Mem0). Any OpenAI-compatible chat endpoint works.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Callable, List

PROMPT = """You maintain long-term memory for an assistant. Below is one conversation between the user and the \
assistant, held on {date}.

Write the facts worth remembering about the user and their life: preferences, plans, decisions, possessions, people \
and relationships, events and when they happened, numbers, places, and anything the user explicitly asked to be \
remembered. Also note concrete recommendations or answers the assistant gave that the user may ask about later.

Rules:
- One fact per line, self-contained (name people and things; no "it" or "they" without a referent).
- Turn relative or partial dates into calendar dates using the conversation date ({date}); keep the user's own \
wording when unsure ("around early March 2023").
- Keep exact names, titles, numbers and amounts.
- Do not invent anything that is not in the conversation. Skip small talk.
- At most {max_facts} facts. Return a JSON list of strings and nothing else.

Conversation:
{conversation}"""


def parse_facts(text: str, max_facts: int) -> List[str]:
    """The JSON list from a model reply (tolerating code fences or stray prose around it)."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and s.strip()][:max_facts]


def extract(conversation: str, date: str, chat: Callable[[str], str], max_facts: int = 12) -> List[str]:
    return parse_facts(chat(PROMPT.format(date=date, conversation=conversation, max_facts=max_facts)), max_facts)


def openai_chat(base_url: str, model: str, api_key: str = "", timeout: float = 120.0,
                extra_body: dict | None = None) -> Callable[[str], str]:
    """A prompt -> reply function for any OpenAI-compatible /chat/completions endpoint, with retries."""
    url = base_url.rstrip("/") + "/chat/completions"

    def chat(prompt: str) -> str:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": 4000, **(extra_body or {})}
        for attempt in range(6):
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json",
                                                  **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.load(r)["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"{url}: no answer after retries")

    return chat
