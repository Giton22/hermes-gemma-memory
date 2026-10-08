"""`hermes gemma-memory status|backfill|search <query>|import`."""

import collections
import json
import time
from datetime import datetime

SKIP_SOURCES = ("cron", "subagent", "kanban", "tool")  # agents' own sessions: read memory, never write it


def _provider():
    from hermes_constants import get_hermes_home

    from .gemma_memory.provider import GemmaMemoryProvider
    provider = GemmaMemoryProvider()
    # No background work: each command does what it needs in the foreground, so none waits on a hidden backfill.
    provider.initialize("cli", hermes_home=str(get_hermes_home()), agent_context="cli", background=False)
    return provider


def gemma_memory_command(args):
    sub = getattr(args, "gemma_memory_command", None)
    if sub not in ("status", "backfill", "search", "import"):
        print("Usage: hermes gemma-memory <status|backfill|search QUERY|import>")
        return
    provider = _provider()
    try:
        if sub == "status":
            stats = provider._store.stats()
            try:
                provider._embedder.query("ping", timeout=5)
                stats["server"] = "ok"
            except Exception as exc:
                stats["server"] = f"unreachable: {exc}"
                hint = _server_hint(str(exc))
                if hint:
                    stats["hint"] = hint
            print(json.dumps({**provider.get_status_config(), **stats}, indent=2, ensure_ascii=False))
        elif sub == "import":
            _import(provider, args)
        elif sub == "backfill":
            provider._redact_stored()
            provider._split_old_turns()
            _embed(provider)
        else:
            for r in provider._search(" ".join(args.query), 10, include_current=True):
                print(f"#{r['id']} {r['kind']} {r.get('similarity', '-')}: {r['text'][:160]!r}")
    finally:
        provider.shutdown()


def _server_hint(error: str) -> str:
    """What to do about the embedding-server failures people actually hit."""
    e = error.lower()
    if "unknown model architecture" in e or "gemma-embedding2" in e:
        return ("The server's llama.cpp is older than EmbeddingGemma 2 support (2026-10-06): update the image "
                "(ghcr.io/ggml-org/llama.cpp:server) or the binary. Ollama bundles an older llama.cpp too.")
    if "mlx" in e:
        return "Ollama runs EmbeddingGemma 2 only with MLX (Apple Silicon/NVIDIA): use llama.cpp's server instead."
    if any(w in e for w in ("refused", "timed out", "no route", "name or service", "getaddrinfo", "unreachable")):
        return "Nothing answers at base_url: is the embedding server running, and is the address right?"
    if "404" in e:
        return "base_url should end in /v1 (the plugin adds /embeddings)."
    return ""


def _embed(provider):
    """Embed everything waiting, a small batch at a time, with progress. Safe to interrupt: the rest is embedded
    later (by Hermes in the background, or another run)."""
    total = provider._store.stats()["not_embedded"]
    if not total:
        print("Nothing waiting to be embedded.")
        return
    print(f"Embedding {total} items (Ctrl+C is safe: the rest is embedded later)...", flush=True)
    t0, done = time.time(), 0
    while done < total:
        n = provider._backfill(batch=8, limit=48)
        if not n:
            break
        done += n
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  {done}/{total} embedded, about {(total - done) / rate / 60:.1f} min left", flush=True)
    left = provider._store.stats()["not_embedded"]
    print(f"Done in {time.time() - t0:.0f}s." + (f" {left} not embedded (server down?): run "
                                                  "hermes gemma-memory backfill later." if left else ""))


def _import(provider, args):
    """Past conversations from Hermes' session store into memory, then embedded."""
    from hermes_state import SessionDB

    since = datetime.strptime(args.since, "%Y-%m-%d").timestamp() if args.since else 0.0
    db = SessionDB(read_only=True)
    sessions, offset = [], 0
    while page := db.list_sessions_rich(limit=200, offset=offset, include_children=True, include_archived=True):
        sessions += page
        offset += len(page)
    skipped = collections.Counter(s.get("source") or "?" for s in sessions if s.get("source") in SKIP_SOURCES)
    old = [s for s in sessions if s.get("source") not in SKIP_SOURCES
           and (s.get("started_at") or s.get("last_active") or 0) < since]
    picked = [s for s in sessions if s.get("source") not in SKIP_SOURCES
              and (s.get("started_at") or s.get("last_active") or 0) >= since]
    reasons = [f"{n} {src}" for src, n in skipped.most_common()] + ([f"{len(old)} before {args.since}"] if old else [])
    print(f"{len(picked)} conversations to read" + (f" (skipped: {', '.join(reasons)})" if reasons else ""))
    imported, turns = [], 0
    for i, s in enumerate(picked, 1):
        n = provider.import_session(s["id"], db.get_messages(s["id"]))
        if n:
            imported.append(s["id"])
            turns += n
        if i % 25 == 0 or i == len(picked):
            print(f"  {i}/{len(picked)} read, {turns} turns from {len(imported)} new conversations", flush=True)
    _embed(provider)
    if args.facts:
        if not (provider._config["use_facts"] and provider._config["facts_base_url"]):
            print("--facts: set use_facts and facts_base_url first.")
            return
        for j, sid in enumerate(imported, 1):
            provider._extract_facts(sid)
            if j % 10 == 0 or j == len(imported):
                print(f"  facts: {j}/{len(imported)} conversations", flush=True)


def register_cli(subparser) -> None:
    subs = subparser.add_subparsers(dest="gemma_memory_command")
    subs.add_parser("status", help="Item counts, model, and whether the embedding server answers")
    subs.add_parser("backfill", help="Embed items stored while the server was down or under another model")
    search = subs.add_parser("search", help="Search memory like the agent does")
    search.add_argument("query", nargs="+")
    imp = subs.add_parser("import", help="Remember past conversations from Hermes' session history")
    imp.add_argument("--since", help="only conversations started on or after this date (YYYY-MM-DD)")
    imp.add_argument("--facts", action="store_true", help="also extract facts (use_facts; one LLM call each)")
    subparser.set_defaults(func=gemma_memory_command)
