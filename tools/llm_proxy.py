"""OpenAI-compatible pass-through to MiMo, DeepSeek or OpenRouter for the memory systems under test (Mem0, Hindsight, ...).

Every request gets the same treatment whoever sends it: reasoning switched off, the Token Plan key added (clients
send any placeholder key), and the token usage tallied per client tag (the path prefix /t/<tag>/v1). GET /stats
returns the tallies, so each system's ingestion cost is measured, not estimated.

    python tools/llm_proxy.py --port 8098
    # client base_url: http://127.0.0.1:8098/t/mem0/v1
"""

import argparse
import collections
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("MIMO_BASE_URL", "")  # your MiMo Token Plan endpoint
CLAUDE = shutil.which("claude") or "claude"


def _text(content):
    """A message's text, whether a string or a list of content parts."""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def api_key():
    key = os.environ.get("XIAOMI_TOKEN_PLAN_API_KEY")
    if not key and sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            key = winreg.QueryValueEx(h, "XIAOMI_TOKEN_PLAN_API_KEY")[0]
    return key


def env_key(name):
    key = os.environ.get(name)
    if not key and sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            key = winreg.QueryValueEx(h, name)[0]
    return key


def cache_prefix(messages):
    """Mark the end of the system prompt (each system's fixed instructions) as cacheable. Anthropic then bills
    a repeat of that prefix at a tenth of the input price; the model sees exactly the same text either way."""
    for m in messages:
        if m.get("role") == "system":
            content = m["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if content and isinstance(content[-1], dict) and content[-1].get("type") == "text":
                content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
            m["content"] = content
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--concurrency", type=int, default=6, help="max requests upstream at once")
    ap.add_argument("--upstream", choices=["mimo", "deepseek", "openrouter", "claude-cli"], default="mimo",
                    help="deepseek: api.deepseek.com, key from DEEPSEEK_API_KEY, reasoning effort low; "
                         "openrouter: key from OPENROUTER_API_KEY, every request sent to --model; "
                         "claude-cli: `claude -p` on the signed-in subscription, --model e.g. haiku")
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5",
                    help="openrouter: the model every system's calls go to, whatever model name they send")
    args = ap.parse_args()
    global UPSTREAM
    if args.upstream == "deepseek":
        UPSTREAM, key = "https://api.deepseek.com/v1", os.environ["DEEPSEEK_API_KEY"]
    elif args.upstream == "openrouter":
        UPSTREAM, key = "https://openrouter.ai/api/v1", env_key("OPENROUTER_API_KEY")
    elif args.upstream == "claude-cli":
        UPSTREAM, key = "claude -p", ""
    else:
        key = api_key()
    stats = collections.defaultdict(collections.Counter)
    lock = threading.Lock()
    upstream = threading.BoundedSemaphore(args.concurrency)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self):
            parts = self.path.split("/")  # /t/<tag>/v1/...
            if len(parts) > 3 and parts[1] == "t":
                return parts[2], "/" + "/".join(parts[4:])
            return "default", self.path.replace("/v1", "", 1)

        def do_GET(self):
            if self.path == "/stats":
                with lock:
                    return self._send(200, json.dumps(stats).encode())
            tag, rest = self._route()
            self._forward("GET", rest, None, tag)

        def do_POST(self):
            tag, rest = self._route()
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if rest.endswith("/chat/completions") and args.upstream == "claude-cli":
                return self._claude_cli(json.loads(body), tag)
            if rest.endswith("/chat/completions"):
                req = json.loads(body)
                if args.upstream == "openrouter":  # one model for every system; reasoning off; cache the prefix
                    req.pop("thinking", None)
                    req["model"] = args.model
                    req["reasoning"] = {"enabled": False, "exclude": True}
                    req["usage"] = {"include": True}  # the dollar cost of each call, tallied below
                    if "anthropic/" in args.model:
                        cache_prefix(req["messages"])
                elif args.upstream == "deepseek":  # its client's model name is passed through; reasoning kept minimal
                    req.pop("thinking", None)  # MiMo's switch, which some clients add; not DeepSeek's
                    req["thinking"] = {"type": "disabled"}  # extraction needs no reasoning; output is most of the bill
                    req["max_tokens"] = max(int(req.get("max_tokens") or 0), 4000)
                else:
                    req["thinking"] = {"type": "disabled"}
                if req.get("stream"):  # tally needs the usage block; ask for it in the final chunk
                    req.setdefault("stream_options", {})["include_usage"] = True
                body = json.dumps(req).encode()
            self._forward("POST", rest, body, tag)

        def _claude_cli(self, req, tag):
            """One chat completion through Claude Code's headless mode on the signed-in subscription (`claude -p`,
            first-party): the system messages become its system prompt (replacing Claude Code's own), tools, MCP
            and settings are off, and the rest is the prompt. Temperature can't be set there (its default is used)."""
            system = "\n\n".join(_text(m.get("content")) for m in req["messages"] if m.get("role") == "system")
            rest_msgs = [m for m in req["messages"] if m.get("role") != "system"]
            if len(rest_msgs) == 1:
                prompt = _text(rest_msgs[0].get("content"))
            else:  # earlier turns (a client's retry with feedback), as a transcript
                prompt = "\n\n".join(f"[{m.get('role')}]\n{_text(m.get('content'))}" for m in rest_msgs)
            for attempt in range(6):
                with upstream, tempfile.TemporaryDirectory() as tmp:
                    sp = os.path.join(tmp, "system.txt")
                    with open(sp, "w", encoding="utf-8") as f:
                        f.write(system or "You are a helpful assistant.")
                    cmd = [CLAUDE, "-p", "--model", args.model, "--system-prompt-file", sp, "--tools", "",
                           "--output-format", "json", "--no-session-persistence", "--setting-sources", "",
                           "--strict-mcp-config"]
                    try:
                        run = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, cwd=tmp,
                                             timeout=600)
                        out = json.loads(run.stdout.decode("utf-8", errors="replace"))
                    except (subprocess.TimeoutExpired, ValueError) as e:
                        out = {"is_error": True, "result": f"proxy: {e!r}"[:300]}
                if not out.get("is_error") and out.get("result") is not None:
                    break
                with lock:
                    stats[tag]["retries"] += 1
                time.sleep(min(60, 2 ** attempt) + random.random())
            u = out.get("usage") or {}
            usage = {"prompt_tokens": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                     + (u.get("cache_read_input_tokens") or 0), "completion_tokens": u.get("output_tokens") or 0}
            ok = not out.get("is_error") and out.get("result") is not None
            with lock:
                s = stats[tag]
                s["requests"] += 1
                s["errors"] += int(not ok)
                s["in"] += usage["prompt_tokens"]
                s["cached_in"] += u.get("cache_read_input_tokens") or 0
                s["out"] += usage["completion_tokens"]
                s["micro_usd"] += int((out.get("total_cost_usd") or 0) * 1e6)
            if not ok:
                return self._send(502, json.dumps({"error": {"message": str(out.get("result"))[:300]}}).encode())
            body = {"id": "cli-" + str(out.get("session_id", "")), "object": "chat.completion", "created": int(time.time()),
                    "model": args.model, "usage": {**usage, "total_tokens": sum(usage.values())},
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": out["result"]}}]}
            self._send(200, json.dumps(body).encode())

        def _forward(self, method, rest, body, tag):
            # Rate limits (429) and upstream hiccups are retried here, with growing waits, and at most
            # args.concurrency requests are upstream at once. Systems differ in how they handle a failed
            # LLM call (Mem0 skips the exchange, Hindsight aborts the batch), so none of them should see one.
            for attempt in range(10):
                req = urllib.request.Request(UPSTREAM + rest, data=body, method=method,
                                             headers={"Authorization": f"Bearer {key}",
                                                      "Content-Type": "application/json"})
                with upstream:
                    try:
                        with urllib.request.urlopen(req, timeout=300) as r:
                            data, code, ctype = r.read(), r.status, r.headers.get("Content-Type", "application/json")
                    except urllib.error.HTTPError as e:
                        data, code, ctype = e.read(), e.code, "application/json"
                    except Exception as e:
                        data, code, ctype = json.dumps({"error": {"message": str(e)}}).encode(), 502, "application/json"
                if code not in (429, 500, 502, 503, 504):
                    break
                with lock:
                    stats[tag]["retries"] += 1
                time.sleep(min(60, 2 ** attempt) + random.random())
            usage = None
            if code == 200 and rest.endswith("/chat/completions"):
                if b"data:" in data[:20]:  # streamed: usage is in the last data chunk
                    for line in data.decode(errors="replace").splitlines()[::-1]:
                        if line.startswith("data:") and '"usage"' in line:
                            usage = json.loads(line[5:]).get("usage")
                            break
                else:
                    usage = json.loads(data).get("usage")
            with lock:
                s = stats[tag]
                s["requests"] += 1
                s["errors"] += int(code != 200)
                if usage:
                    s["in"] += usage.get("prompt_tokens") or 0
                    s["out"] += usage.get("completion_tokens") or 0
                    s["cached_in"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    s["micro_usd"] += int((usage.get("cost") or 0) * 1e6)
            self._send(code, data, ctype)

        def log_message(self, *a):
            pass

    print(f"proxy on 127.0.0.1:{args.port} -> {UPSTREAM}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
