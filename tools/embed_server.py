"""Stand-in embeddings server: EmbeddingGemma 2 through sentence-transformers, behind an OpenAI-style
/v1/embeddings (the same request shapes as llama-server, images included). For trying the model before a
llama.cpp build supports it; CPU is fine.

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -U sentence-transformers transformers pillow
    python tools/embed_server.py --port 8091
"""

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_item(item):
    """A string, or {"content": [{"type": "text"|"image_url", ...}]} -> what SentenceTransformer.encode takes."""
    if isinstance(item, str):
        return item
    from PIL import Image
    text, images = [], []
    for part in item.get("content", []):
        if part.get("type") == "text":
            text.append(part["text"])
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if not url.startswith("data:"):
                raise ValueError("only data: URIs are supported")
            images.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
            text.append("<|image|>")
        else:
            raise ValueError(f"unsupported content part {part.get('type')!r}")
    if not images:
        return " ".join(text)
    return {"text": " ".join(text), "image": images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--device", default="auto", help="auto, cuda (ROCm/NVIDIA) or cpu")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = SentenceTransformer(args.model, device=dev,
                                model_kwargs={"torch_dtype": torch.bfloat16} if dev == "cuda" else {})
    lock = threading.Lock()
    print(f"ready on {args.host}:{args.port} ({dev})", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            out = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            self._send(200, {"status": "ok"} if self.path == "/health" else {"data": [{"id": args.model}]})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/embeddings", "/embeddings"):
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                raw = req["input"]
                items = [parse_item(i) for i in (raw if isinstance(raw, list) else [raw])]
                t = time.time()
                with lock:
                    vecs = model.encode(items, normalize_embeddings=True)
                print(f"{len(items)} item(s) in {time.time() - t:.2f}s", flush=True)
            except Exception as exc:
                return self._send(400, {"error": str(exc)})
            self._send(200, {"object": "list", "model": args.model,
                             "data": [{"object": "embedding", "index": i, "embedding": v.tolist()}
                                      for i, v in enumerate(vecs)]})

        def log_message(self, *a):
            pass

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
