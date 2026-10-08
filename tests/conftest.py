import base64
import hashlib
import json
import math
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

# The provider imports Hermes (agent.memory_provider): point HERMES_AGENT_PATH at a hermes-agent checkout.
if os.environ.get("HERMES_AGENT_PATH"):
    sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])

DIMS = 768


def fake_embedding(text: str):
    """Bag of words hashed into 768 buckets: shared words -> similar vectors, like a (very dumb) real model."""
    vec = [0.0] * DIMS
    for word in re.findall(r"\w+", text.lower().split("|")[-1].split(":", 1)[-1]):
        h = hashlib.sha1(word.encode()).digest()
        vec[int.from_bytes(h[:4], "little") % 128] += 1.0  # leading dims carry it all, like Matryoshka
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeServer:
    """An OpenAI-compatible /v1/embeddings endpoint that can be switched off."""

    def __init__(self):
        self.up, self.calls, self.images = True, [], True
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.calls.append(body)
                if not server.up or self.path != "/v1/embeddings":
                    self.send_response(503)
                    self.end_headers()
                    return
                inputs = []
                for t in body["input"]:
                    if isinstance(t, dict):  # llama-server multimodal shape: {"content": [{"type": "image_url", ...}]}
                        if not server.images:
                            self.send_response(400)
                            self.end_headers()
                            return
                        url = t["content"][0]["image_url"]["url"]
                        t = base64.b64decode(url.split(",", 1)[1]).decode()  # test "pixels" are words
                    inputs.append(t)
                data = [{"index": i, "embedding": fake_embedding(t)} for i, t in enumerate(inputs)]
                out = json.dumps({"data": data}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.httpd.shutdown()
