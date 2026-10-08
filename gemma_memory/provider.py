"""The MemoryProvider: embeds every turn, recalls the closest past turns and notes before each reply."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt, spawn_context_thread

from . import facts, history, passages, recall
from .embedder import DOC_PREFIX, QUERY_PREFIX, Embedder, EmbedError
from .store import Store

logger = logging.getLogger(__name__)

NAME = "gemma-memory"
TURN_KINDS = ("turn", "note", "memory", "image")  # what recall ranks in each recall_unit mode
PASSAGE_KINDS = ("passage", "note", "memory", "image")
SPILL_MARGIN = 800  # under Hermes' spill cap: room for the per-conversation headers recall.render() adds
CONFIG_KEY = "gemma-memory"  # plugins.gemma-memory in config.yaml

DEFAULTS: Dict[str, Any] = {
    "base_url": "http://localhost:8080/v1",
    "model": "embeddinggemma-2",
    "dims": 768,
    # Recall before each reply, tuned on LongMemEval-S (dev half; see tools/eval_longmemeval.py): vector and keyword
    # rankings fused by rank, then as many conversations as fit the budget, the assistant's side trimmed.
    "recall_budget": 9000,  # characters injected; capped under Hermes' prefetch spill limit (see _budget)
    "assistant_chars": 800,  # per recalled turn; the user's side is kept whole
    "full_turns": 0,  # the best-ranked N turns are recalled whole
    "per_conversation": 0,  # at most N recalled items per conversation (0 = no cap), for breadth across chats
    "with_question": False,  # a recalled reply passage brings the user's words of its turn
    "context_passages": 0,  # neighbouring passages of the same turn, each side of a hit
    "anchor_dates": False,  # "last weekend" -> "last weekend [2023-05-20/21]", from when the message was written
    # Optional derived memory: a few dated facts per conversation from an LLM, once per session (gemma_memory.facts).
    "use_facts": False,
    "facts_base_url": "",  # OpenAI-compatible chat endpoint; key from GEMMA_MEMORY_FACTS_API_KEY
    "facts_model": "",
    "candidates": 20,  # depth of each ranking before fusion
    "keyword_weight": 1.0,  # 0 = vector only
    "recall_unit": "passage",  # spans of turns (measured +8 pts over whole turns, LongMemEval dev); "turn" = whole turns
    "top_k": 8,  # memory_recall tool default
    "query_prefix": QUERY_PREFIX,
    "doc_prefix": DOC_PREFIX,
    "prefetch_timeout": 3.0,
    "max_chars": 8000,  # per stored turn; EmbeddingGemma 2 reads 8K tokens
    # Keys, tokens and passwords are masked before anything is stored (Hermes' own egress redactor): stored text
    # is later recalled into prompts for a remote model, and facts are extracted by one.
    "redact_secrets": True,
}

_RECALL_SCHEMA = {
    "name": "memory_recall",
    "description": (
        "Search long-term memory: every past conversation turn and saved note, by meaning and keywords. "
        "The closest few are already recalled before each reply; use this to look further or for something specific."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "limit": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": ["query"],
    },
}
_SAVE_SCHEMA = {
    "name": "memory_note",
    "description": "Save a note to long-term memory (a fact, decision or preference worth finding later).",
    "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
}
_FORGET_SCHEMA = {
    "name": "memory_forget",
    "description": "Delete one item from long-term memory by its id (from memory_recall), e.g. when the user asks.",
    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
}


def load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        user = cfg_get(load_config_readonly(), "plugins", CONFIG_KEY, default={}) or {}
    except Exception:
        user = {}
    return {**DEFAULTS, **{k: v for k, v in user.items() if v not in (None, "")}}


# Said in words rather than as KEY=value: "my wifi password is Sunflower2024". The value needs a digit or symbol,
# so "the password is correct" stays.
_SPOKEN_SECRET = re.compile(
    r"(?i)\b((?:wi-?fi\s+)?(?:password|passphrase|passcode|pin(?:\s+code)?)\s*(?:is|was|:|=)\s*)"
    r"(?=[^\s,;]*[\d@#$%^&*_+=~/\\-])([^\s,;]{4,}?)([.,;!?]?(?:\s|$))")


# Used only if Hermes' redactor can't be imported (a Hermes version that moved it): the common shapes.
_FALLBACK_SECRETS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\b(?:sk-(?:or-v1-|proj-|ant-)?|pplx-|ghp_|gho_|github_pat_|xox[abpr]-|AIza|hf_)[A-Za-z0-9_\-]{16,}"),
     "[REDACTED KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "[REDACTED JWT]"),
    (re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/-]{20,}=*"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*\s*[=:]\s*)[\"']?[^\s\"']{4,}[\"']?"),
     r"\1***"),
    (re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]+@"), r"\1***@"),
]
_warned_fallback = False


def redact(text: str) -> str:
    """Secrets masked with Hermes' egress redactor (vendor-prefixed keys, JWTs, private keys, credential
    assignments, bearer tokens, URL credentials), plus passwords given in words. Hermes' redactor fails closed: if
    it breaks, a placeholder is stored instead. If this Hermes has none, the plugin's own patterns are used."""
    global _warned_fallback
    try:
        from agent.redact import redact_for_egress
    except ImportError:
        if not _warned_fallback:
            logger.warning("gemma-memory: Hermes' redactor not found; using the plugin's own secret patterns")
            _warned_fallback = True
        for pattern, repl in _FALLBACK_SECRETS:
            text = pattern.sub(repl, text)
        return _SPOKEN_SECRET.sub(lambda m: m.group(1) + "***" + m.group(3), text)
    return _SPOKEN_SECRET.sub(lambda m: m.group(1) + "***" + m.group(3), redact_for_egress(text))


def _turn_text(user: str, assistant: str, max_chars: int) -> str:
    text = f"User: {user.strip()}\nAssistant: {assistant.strip()}"
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


_IMAGE_PART_TYPES = ("image_url", "input_image")
_DATA_URI = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", re.IGNORECASE | re.DOTALL)
_FILE_TAG = re.compile(r"\[file: ([^\]]+)\]\s*$")
MAX_IMAGES_PER_TURN = 4


def _turn_images(messages: Optional[List[Dict[str, Any]]]) -> List[str]:
    """data: URIs of the images in the newest turn (from the last user message on). Web links are not fetched."""
    if not messages:
        return []
    start = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
    found = []
    for m in messages[start:]:
        content = m.get("content")
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict) or part.get("type") not in _IMAGE_PART_TYPES:
                continue
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if isinstance(url, str) and _DATA_URI.match(url):
                found.append(url)
    return found[:MAX_IMAGES_PER_TURN]


def _save_image(data_uri: str, folder: Path) -> Path:
    """Keep a copy (Hermes clears its own image cache), named by content hash so a resent image is stored once."""
    mime, b64 = _DATA_URI.match(data_uri).groups()
    raw = base64.b64decode(b64)
    ext = {"image/jpeg": "jpg", "image/svg+xml": "svg"}.get(mime.lower(), mime.split("/")[1].lower())
    path = folder / f"{hashlib.sha1(raw).hexdigest()[:20]}.{ext}"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return path


def _file_data_uri(path: str) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    ext = p.suffix.lstrip(".").lower()
    mime = {"jpg": "image/jpeg", "svg": "image/svg+xml"}.get(ext, f"image/{ext}")
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


class GemmaMemoryProvider(MemoryProvider):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config if config is not None else load_config()
        self._store: Optional[Store] = None
        self._embedder: Optional[Embedder] = None
        self._writes = True
        self._session_id = ""
        self._in_context: set[str] = set()  # sessions whose turns the model can already see
        self._compressed_at: Dict[str, float] = {}  # turns older than this left the context window
        self._last_recall: Optional[RecallStatus] = None
        self._worker: Optional[threading.Thread] = None
        self._backfill_lock = threading.Lock()

    # -- identity and setup -------------------------------------------------

    @property
    def name(self) -> str:
        return NAME

    def is_available(self) -> bool:
        return True  # SQLite is built in; the embedding server is checked lazily and failures degrade to keywords

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "base_url", "description": "OpenAI-compatible embeddings URL (llama-server: http://<host>:8080/v1)",
             "default": DEFAULTS["base_url"], "required": True},
            {"key": "model", "description": "Embedding model name on that server", "default": DEFAULTS["model"]},
            {"key": "dims", "description": "Vector size (smaller = less storage, slightly worse recall)",
             "default": str(DEFAULTS["dims"]), "choices": ["768", "512", "256", "128"]},
            {"key": "api_key", "description": "API key, if the server needs one", "secret": True,
             "env_var": "GEMMA_MEMORY_API_KEY"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from hermes_cli.config import save_config
        values = {k: v for k, v in values.items() if k != "api_key"}
        if "dims" in values:
            values["dims"] = int(values["dims"])
        save_config({"plugins": {CONFIG_KEY: values}}, merge_existing=True)

    def get_status_config(self) -> Dict[str, Any]:
        return {k: self._config.get(k) for k in ("base_url", "model", "dims")}

    def initialize(self, session_id: str, **kwargs) -> None:
        import os
        from hermes_constants import get_hermes_home
        home = Path(kwargs.get("hermes_home") or get_hermes_home())
        c = self._config
        self._store = Store(home / "gemma-memory" / "memory.db", model=str(c["model"]), dims=int(c["dims"]))
        self._images_dir = home / "gemma-memory" / "images"
        self._embedder = Embedder(
            str(c["base_url"]), str(c["model"]), int(c["dims"]),
            api_key=os.environ.get("GEMMA_MEMORY_API_KEY", ""),
            query_prefix=str(c["query_prefix"]), doc_prefix=str(c["doc_prefix"]))
        # Cron runs, subagents and flushes read memory but don't write their turns into it.
        self._writes = kwargs.get("agent_context", "primary") == "primary"
        self._bind(session_id, kwargs.get("parent_session_id", ""))
        # The search index loads first, so the first recall of the session doesn't wait for it. The CLI passes
        # background=False and does its own maintenance, in the foreground with progress.
        if kwargs.get("background", True):
            self._run(lambda: (self._store.warm(), self._redact_stored(), self._split_old_turns(), self._backfill()),
                      "gemma-memory-backfill")

    def backup_paths(self) -> List[str]:
        return []  # everything lives under HERMES_HOME

    def system_prompt_block(self) -> str:
        return ("# Long-term memory\nEvery conversation with the user is kept permanently and automatically, across "
                "sessions. Before each reply the most relevant past turns and notes are recalled (under \"Recalled from "
                "earlier conversations\"); they come from this permanent store, so they will be found again in any "
                "future session. Do not copy recalled details into other memory just to keep them. Use memory_recall to "
                "look further, memory_note to save something worth finding later that was not said in a conversation, "
                "memory_forget to delete an item the user wants gone.")

    # -- per turn -----------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall = None
        if not self._store or is_trivial_prompt(query):
            return ""
        c = self._config
        ranked = self._search(query, timeout=float(c["prefetch_timeout"]))
        if c["with_question"] or int(c["context_passages"]):
            ranked = recall.expand(ranked, self._store.get, self._store.children,
                                   with_question=bool(c["with_question"]), context=int(c["context_passages"]))
        picked = recall.select(ranked, budget=self._budget(), assistant_chars=int(c["assistant_chars"]),
                               full_turns=int(c["full_turns"]), per_session=int(c["per_conversation"]))
        if not picked:
            return ""
        sources = {r.get("parent") or r["id"] for r in picked}  # passages of one turn are one memory
        self._last_recall = RecallStatus(provider_label="Gemma memory", count=len(sources))
        return recall.render(picked, assistant_chars=int(c["assistant_chars"]), anchor_dates=bool(c["anchor_dates"]))

    def recall_status(self) -> Optional[RecallStatus]:
        return self._last_recall

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[List[Dict[str, Any]]] = None) -> None:
        if not (self._store and self._writes) or is_trivial_prompt(user_content) or not (assistant_content or "").strip():
            return
        text = _turn_text(user_content, assistant_content, int(self._config["max_chars"]))
        sid = session_id or self._session_id
        images = _turn_images(messages)
        self._run(lambda: self._add("turn", text, sid), "gemma-memory-sync")
        for uri in images:
            self._run(lambda uri=uri: self._add_image(uri, user_content, sid), "gemma-memory-image")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """With use_facts: extract this conversation's facts once (one LLM call), in the background."""
        if self._store and self._writes and self._config["use_facts"] and self._config["facts_base_url"]:
            sid = self._session_id
            self._run(lambda: self._extract_facts(sid), "gemma-memory-facts")

    def _extract_facts(self, session_id: str) -> int:
        """Facts for one conversation, stored next to its passages. A session that already has facts is skipped."""
        import os
        if self._store.session_items(session_id, "fact"):
            return 0
        turns = self._store.session_items(session_id, "turn")
        if not turns:
            return 0
        c = self._config
        chat = facts.openai_chat(str(c["facts_base_url"]), str(c["facts_model"]),
                                 os.environ.get("GEMMA_MEMORY_FACTS_API_KEY", ""))
        when = time.strftime("%Y-%m-%d", time.localtime(turns[0]["created_at"]))
        found = facts.extract("\n".join(recall.trim_assistant(t["text"], 1500) for t in turns), when, chat)
        found = [self._clean(f) for f in found]
        ids = [self._store.add("fact", f, session_id=session_id, created_at=turns[0]["created_at"]) for f in found]
        try:
            for i, v in zip(ids, self._embedder.documents(found)):
                self._store.set_vec(i, v)
        except EmbedError:
            pass  # the backfill embeds them
        return len(ids)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        # Turns stored before now are about to drop out of the window: from here on they're worth recalling.
        self._compressed_at[self._session_id] = time.time()
        return ""

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        self._bind(new_session_id, parent_session_id)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror the built-in MEMORY.md / USER.md so those entries are searchable too."""
        if not (self._store and self._writes):
            return
        previous = (metadata or {}).get("previous_content")
        if action in ("replace", "remove") and previous:
            self._store.delete_text("memory", self._clean(previous))
        if action in ("add", "replace") and content:
            self._run(lambda: self._add("memory", content, self._session_id), "gemma-memory-mirror")

    def shutdown(self) -> None:
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=5.0)
        if self._store:
            self._store.close()
            self._store = None

    # -- tools --------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_RECALL_SCHEMA, _SAVE_SCHEMA, _FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        from tools.registry import tool_error
        if not self._store:
            return tool_error("gemma-memory is not initialized")
        try:
            if tool_name == "memory_recall":
                found = self._search(str(args["query"]), int(args.get("limit") or 8), include_current=True)
                return json.dumps({"results": [{**r, "date": _when(r.pop("created_at"))} for r in found]},
                                  ensure_ascii=False)
            if tool_name == "memory_note":
                content = str(args["content"]).strip()
                if not content:
                    return tool_error("content is empty")
                return json.dumps({"id": self._add("note", content, self._session_id), "status": "saved"})
            if tool_name == "memory_forget":
                return json.dumps({"deleted": self._store.delete(int(args["id"]))})
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))
        return tool_error(f"Unknown tool: {tool_name}")

    # -- internals ----------------------------------------------------------

    def _budget(self) -> int:
        """recall_budget, kept under Hermes' prefetch spill cap (hooks.output_spill.max_chars, 10,000 by default):
        Hermes replaces a longer recall with a 1,000-character preview and a file path, so going over loses
        nearly everything. Room is left for the per-conversation headers."""
        budget = int(self._config["recall_budget"])
        try:
            from tools.hook_output_spill import get_spill_config
            spill = get_spill_config()
            if spill.get("enabled", True):
                budget = min(budget, int(spill.get("max_chars") or 10000) - SPILL_MARGIN)
        except Exception:  # outside Hermes (tests, tools): the configured budget as is
            pass
        return max(1000, budget)

    def _bind(self, session_id: str, parent_session_id: str = "") -> None:
        self._session_id = session_id
        self._in_context = {s for s in (session_id, parent_session_id) if s}

    def _skip(self, session_id: str, created_at: float) -> bool:
        """True for turns the model already sees: this chat's (and its branch parent's), unless compressed away."""
        return session_id in self._in_context and created_at > self._compressed_at.get(session_id, float("-inf"))

    def _search(self, query: str, limit: int = 0, *, timeout: Optional[float] = None, include_current: bool = False):
        """Best-first items for ``query`` (all fused candidates, or the first ``limit``)."""
        qvec = None
        try:
            qvec = self._embedder.query(query, timeout=timeout)
        except EmbedError as exc:
            logger.warning("gemma-memory: embedding server unreachable, keyword search only (%s)", exc)
        c = self._config
        unit = PASSAGE_KINDS if c["recall_unit"] == "passage" else TURN_KINDS
        if c["use_facts"]:
            unit = unit + ("fact",)
        ranked = self._store.fused(query, qvec, candidates=int(c["candidates"]),
                                   weights=(1.0, float(c["keyword_weight"])),
                                   skip=(lambda s, t: False) if include_current else self._skip, kinds=unit)
        return ranked[:limit] if limit else ranked

    def _add(self, kind: str, text: str, session_id: str) -> int:
        """Store first, embed second: text is never lost to a down server, the backfill embeds it later.
        A turn also gets its passages (gemma_memory.passages), linked to it, for passage-level recall."""
        text = self._clean(text)
        item_id = self._store.add(kind, text, session_id=session_id)
        rows = [(item_id, text)]
        if kind == "turn":
            created = self._store.get(item_id)["created_at"]
            rows += [(self._store.add("passage", p, session_id=session_id, created_at=created, parent=item_id), p)
                     for p in passages.split_turn(text)]
        try:
            for (row_id, _), vec in zip(rows, self._embedder.documents([t for _, t in rows])):
                self._store.set_vec(row_id, vec)
        except EmbedError as exc:
            logger.warning("gemma-memory: stored #%d without a vector for now (%s)", item_id, exc)
            return item_id
        self._backfill()
        return item_id

    def _clean(self, text: str) -> str:
        return redact(text) if self._config["redact_secrets"] and text else text

    def _redact_stored(self) -> int:
        """Once per store: mask secrets in rows saved before redaction existed (or while it was off). A changed
        row loses its vector, and a changed turn its passages; the backfill rebuilds both from the clean text."""
        if not self._config["redact_secrets"] or self._store.get_flag("redacted"):
            return 0
        changed, after = 0, 0
        while rows := self._store.texts(after=after):
            for item_id, kind, text in rows:
                clean = redact(text)
                if clean != text:
                    if kind == "turn":
                        for child in self._store.children(item_id):
                            self._store.delete(child)
                    self._store.set_text(item_id, clean)
                    changed += 1
            after = rows[-1][0]
        self._store.set_flag("redacted")
        if changed:
            logger.info("gemma-memory: masked secrets in %d stored items", changed)
        return changed

    def import_session(self, session_id: str, messages: List[Dict[str, Any]]) -> int:
        """A past conversation (Hermes' stored messages) as the turns sync_turn() would have kept, at their original
        times, secrets masked, trivial prompts skipped. Text only: _backfill() embeds them afterwards. A session that
        already has turns here (imported before, or remembered live) is left alone. Returns turns stored."""
        if not self._store or self._store.session_items(session_id, "turn"):
            return 0
        stored = 0
        for user, reply, when in history.turns(messages):
            if is_trivial_prompt(user):
                continue
            text = self._clean(_turn_text(user, reply, int(self._config["max_chars"])))
            turn_id = self._store.add("turn", text, session_id=session_id, created_at=when or None)
            for p in passages.split_turn(text):
                self._store.add("passage", p, session_id=session_id, created_at=when or None, parent=turn_id)
            stored += 1
        return stored

    def _split_old_turns(self) -> int:
        """Turns stored before passages existed get theirs (text now; the backfill embeds them)."""
        done = 0
        while self._store and (rows := self._store.turns_without_passages()):
            for turn_id, sid, text, created in rows:
                for p in passages.split_turn(text) or [text]:
                    self._store.add("passage", p, session_id=sid, created_at=created, parent=turn_id)
                done += 1
        return done

    def _add_image(self, data_uri: str, user_content: str, session_id: str) -> int:
        """An image of a turn: saved to disk, its text (the user's words and the file) searchable by keyword, its
        vector from the pixels. If the server can't embed images yet, the backfill retries the image later."""
        path = _save_image(data_uri, self._images_dir)
        words = self._clean((user_content or "").strip()[:500])
        text = f"Image shared in a conversation{': ' + words if words else ''} [file: {path}]"
        item_id = self._store.add("image", text, session_id=session_id)
        try:
            self._store.set_vec(item_id, self._embedder.images([data_uri])[0])
        except EmbedError as exc:
            logger.warning("gemma-memory: image #%d stored without a vector for now (%s)", item_id, exc)
        return item_id

    def _embed_rows(self, rows) -> List[Optional[List[float]]]:
        """Vectors for pending (id, text, kind) rows: images from their saved file, everything else from text."""
        vecs: List[Optional[List[float]]] = [None] * len(rows)
        texts = [(i, t) for i, (_, t, kind) in enumerate(rows) if kind != "image"]
        if texts:
            for (i, _), v in zip(texts, self._embedder.documents([t for _, t in texts])):
                vecs[i] = v
        for i, (_, t, kind) in enumerate(rows):
            if kind == "image":
                m = _FILE_TAG.search(t)
                uri = _file_data_uri(m.group(1)) if m else None
                if uri:
                    vecs[i] = self._embedder.images([uri])[0]
        return vecs

    def _backfill(self, batch: int = 16, limit: int = 0) -> int:
        """Embed rows stored while the server was down, or with another model or size, at most ``limit`` (0 = all)
        per call. Returns rows done."""
        if not self._backfill_lock.acquire(blocking=False):
            return 0
        done = 0
        try:
            after = 0  # one pass in id order: a row that can't be embedded (an image whose file is gone) is skipped
            while self._store and (rows := self._store.pending(batch, after)):
                for (item_id, _, _), vec in zip(rows, self._embed_rows(rows)):
                    if vec is not None:
                        self._store.set_vec(item_id, vec)
                        done += 1
                after = rows[-1][0]
                if limit and done >= limit:
                    break
        except EmbedError as exc:
            logger.info("gemma-memory: backfill paused (%s)", exc)
        finally:
            self._backfill_lock.release()
        return done

    def _run(self, fn, name: str) -> None:
        """One background writer at a time, keeping turn order; never blocks the reply."""
        previous = self._worker

        def job():
            if previous and previous.is_alive():
                previous.join()
            try:
                fn()
            except Exception:
                logger.exception("gemma-memory: %s failed", name)

        self._worker = spawn_context_thread(job, name=name)
        self._worker.start()
