"""Text -> vector through any OpenAI-compatible /v1/embeddings server (Ollama, llama.cpp, LM Studio, vLLM)."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import List, Sequence

# EmbeddingGemma's task prompts: queries and stored text are embedded differently.
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "


class EmbedError(RuntimeError):
    pass


def truncate(vec: Sequence[float], dims: int) -> List[float]:
    """Matryoshka: the first ``dims`` values are a usable embedding once re-normalized."""
    head = list(vec[:dims]) if dims else list(vec)
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]


class Embedder:
    def __init__(self, base_url: str, model: str, dims: int, *, api_key: str = "",
                 query_prefix: str = QUERY_PREFIX, doc_prefix: str = DOC_PREFIX, timeout: float = 10.0):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model, self.dims, self.timeout = model, dims, timeout
        self.api_key, self.query_prefix, self.doc_prefix = api_key, query_prefix, doc_prefix

    def query(self, text: str, *, timeout: float | None = None) -> List[float]:
        return self._embed([self.query_prefix + text], timeout)[0]

    def documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed([self.doc_prefix + t for t in texts], None)

    def images(self, data_uris: Sequence[str]) -> List[List[float]]:
        """Images into the same space as text (EmbeddingGemma 2 with its vision encoder, e.g. llama-server --mmproj).
        Images take no task prefix. Request shape: llama-server's multimodal /v1/embeddings input."""
        return self._embed([{"content": [{"type": "image_url", "image_url": {"url": u}}]} for u in data_uris], None)

    def _embed(self, inputs: List, timeout: float | None) -> List[List[float]]:
        body = json.dumps({"model": self.model, "input": inputs}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as exc:  # the server's own reason (e.g. an unknown model) is in the body
            detail = exc.read().decode("utf-8", errors="replace")[:300] if exc.fp else ""
            raise EmbedError(f"{self.url}: HTTP {exc.code} {detail}".strip()) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise EmbedError(f"{self.url}: {exc}") from exc
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        if len(rows) != len(inputs):
            raise EmbedError(f"{self.url}: expected {len(inputs)} embeddings, got {len(rows)}")
        vecs = [truncate(r["embedding"], self.dims) for r in rows]
        if self.dims and any(len(v) != self.dims for v in vecs):
            raise EmbedError(f"{self.model} returns fewer than {self.dims} dimensions")
        return vecs
