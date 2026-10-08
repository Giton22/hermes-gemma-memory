"""SQLite store: text, its embedding, and an FTS5 index. Search is hybrid (vectors + keywords, fused by rank)."""

from __future__ import annotations

import heapq
import re
import sqlite3
import threading
import time
from array import array
from operator import mul
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

try:  # Fast path when Hermes' environment has numpy; plain Python is fine for one person's history.
    import numpy as _np
except ImportError:  # pragma: no cover - depends on the environment
    _np = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,              -- turn | passage | note | memory | image
    session_id TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    model TEXT,                      -- model+dims of vec; NULL vec = not embedded yet
    dims INTEGER,
    vec BLOB
);
CREATE INDEX IF NOT EXISTS items_pending ON items(model, dims);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(text, content='items', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TABLE IF NOT EXISTS flags (name TEXT PRIMARY KEY);  -- one-time migrations done on this store
"""

_KEYWORD_BONUS = 0.02  # a shared rare word (a name, an IP, a port) tips close calls; it never adds a result
_RRF_K = 60  # reciprocal rank fusion constant (Cormack et al.)
_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)
_STOPWORDS = frozenset("""
about after again all also and any are because been before being but can could did does doing done each few for
from had has have having her here hers him his how its just like may mine more most much must not now off once
only other our ours out over own same she should some such than that the their theirs them then there these they
this those through too under until very was way were what when where which while who whom why will with would
you your yours what's where's how's i'm it's don't can't
""".split())


def _fts_query(text: str) -> str:
    words = [w for w in dict.fromkeys(w.lower() for w in _WORD_RE.findall(text)) if w not in _STOPWORDS][:24]
    return " OR ".join(f'"{w}"' for w in words)


class Store:
    def __init__(self, path: str | Path, *, model: str, dims: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model, self.dims = model, dims
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(items)")}
        if "parent" not in cols:  # databases from before passages: add the column in place
            self._conn.execute("ALTER TABLE items ADD COLUMN parent INTEGER")
        self._conn.execute("CREATE INDEX IF NOT EXISTS items_parent ON items(parent)")
        self._vecs: Optional[_VecIndex] = None  # rows embedded with the current model; built on first search
        self._meta_cache: Optional[Dict[int, tuple]] = None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes -------------------------------------------------------------

    def add(self, kind: str, text: str, *, session_id: str = "", vec: Optional[Sequence[float]] = None,
            created_at: Optional[float] = None, parent: Optional[int] = None) -> int:
        blob = array("f", vec).tobytes() if vec is not None else None
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO items(kind, session_id, text, created_at, model, dims, vec, parent) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (kind, session_id, text, created_at or time.time(),
                 self.model if blob else None, self.dims if blob else None, blob, parent))
            if blob and self._vecs is not None:
                self._vecs.put(cur.lastrowid, vec)
            if self._meta_cache is not None:
                row = self._conn.execute("SELECT session_id, created_at, kind FROM items WHERE id=?",
                                         (cur.lastrowid,)).fetchone()
                self._meta_cache[cur.lastrowid] = tuple(row)
            return cur.lastrowid

    def set_vec(self, item_id: int, vec: Sequence[float]) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE items SET model=?, dims=?, vec=? WHERE id=?",
                               (self.model, self.dims, array("f", vec).tobytes(), item_id))
            if self._vecs is not None:
                self._vecs.put(item_id, vec)

    def set_text(self, item_id: int, text: str) -> bool:
        """Replace a row's text; its vector is dropped (the backfill re-embeds it) and the keyword index follows."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT text FROM items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                return False
            self._conn.execute("INSERT INTO items_fts(items_fts, rowid, text) VALUES ('delete', ?, ?)", (item_id, row[0]))
            self._conn.execute("UPDATE items SET text=?, vec=NULL, model=NULL, dims=NULL WHERE id=?", (text, item_id))
            self._conn.execute("INSERT INTO items_fts(rowid, text) VALUES (?, ?)", (item_id, text))
            if self._vecs is not None:
                self._vecs.drop(item_id)
            return True

    def texts(self, after: int = 0, limit: int = 500) -> List[tuple]:
        """(id, kind, text) of rows with ids > after, in id order."""
        with self._lock:
            return self._conn.execute("SELECT id, kind, text FROM items WHERE id > ? ORDER BY id LIMIT ?",
                                      (after, limit)).fetchall()

    def get_flag(self, name: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM flags WHERE name=?", (name,)).fetchone() is not None

    def set_flag(self, name: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO flags(name) VALUES (?)", (name,))

    def delete(self, item_id: int) -> bool:
        """Remove an item and anything derived from it (a turn's passages)."""
        with self._lock, self._conn:
            children = [r[0] for r in self._conn.execute("SELECT id FROM items WHERE parent=?", (item_id,))]
            gone = self._conn.execute("DELETE FROM items WHERE id=? OR parent=?", (item_id, item_id)).rowcount > 0
            for i in [item_id, *children]:
                if self._vecs is not None:
                    self._vecs.drop(i)
                if self._meta_cache is not None:
                    self._meta_cache.pop(i, None)
            return gone

    def session_items(self, session_id: str, kind: str) -> List[dict]:
        """One conversation's items of a kind, in stored order."""
        with self._lock:
            rows = self._conn.execute("SELECT id, kind, session_id, text, created_at, parent FROM items "
                                      "WHERE session_id=? AND kind=? ORDER BY id", (session_id, kind)).fetchall()
        return [_row(r) for r in rows]

    def children(self, parent: int) -> List[int]:
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT id FROM items WHERE parent=? ORDER BY id", (parent,))]

    def turns_without_passages(self, limit: int = 64) -> List[tuple]:
        """(id, session_id, text, created_at) of turns stored before passages existed, oldest first."""
        with self._lock:
            return self._conn.execute(
                "SELECT t.id, t.session_id, t.text, t.created_at FROM items t WHERE t.kind='turn' AND NOT EXISTS "
                "(SELECT 1 FROM items p WHERE p.parent=t.id) ORDER BY t.id LIMIT ?", (limit,)).fetchall()

    def delete_text(self, kind: str, text: str) -> int:
        with self._lock:
            ids = [r[0] for r in self._conn.execute("SELECT id FROM items WHERE kind=? AND text=?", (kind, text))]
        return sum(self.delete(i) for i in ids)

    # -- reads --------------------------------------------------------------

    def get(self, item_id: int) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT id, kind, session_id, text, created_at, parent FROM items WHERE id=?",
                                     (item_id,)).fetchone()
        return _row(row) if row else None

    def pending(self, limit: int = 32, after: int = 0) -> List[tuple]:
        """(id, text, kind) of rows not yet embedded with the current model and size, oldest first, ids > after."""
        with self._lock:
            return self._conn.execute(
                "SELECT id, text, kind FROM items WHERE id > ? AND (vec IS NULL OR model IS NOT ? OR dims IS NOT ?) "
                "ORDER BY id LIMIT ?", (after, self.model, self.dims, limit)).fetchall()

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            kinds = dict(self._conn.execute("SELECT kind, COUNT(*) FROM items GROUP BY kind").fetchall())
            waiting = self._conn.execute(
                "SELECT COUNT(*) FROM items WHERE vec IS NULL OR model IS NOT ? OR dims IS NOT ?",
                (self.model, self.dims)).fetchone()[0]
        return {"total": total, "by_kind": kinds, "not_embedded": waiting, "model": self.model, "dims": self.dims}

    def search(self, query: str, qvec: Optional[Sequence[float]], *, limit: int = 5, min_similarity: float = 0.6,
               max_gap: float = 1.0, skip: Callable[[str, float], bool] = lambda s, t: False) -> List[dict]:
        """Meaning first: rows at least ``min_similarity`` to the query and within ``max_gap`` of the best one,
        with a small bonus for sharing a rare word. ``qvec`` None (embedding server down) = keywords only.
        ``skip(session_id, created_at)`` drops rows the caller already has in context."""
        meta = self._meta()
        keyword_hits = self._keyword_hits(query, limit * 8)
        if qvec is None:
            ids = [i for i in keyword_hits if i in meta and not skip(*meta[i][:2])][:limit]
            return [item for item in map(self.get, ids) if item]
        sims: Dict[int, float] = {}
        for item_id, sim in self._nearest(qvec, limit * 8):
            if sim < min_similarity:
                break
            if item_id in meta and not skip(*meta[item_id][:2]):
                sims[item_id] = sim
        if not sims:
            return []
        best = max(sims.values())
        scores = {i: s + (_KEYWORD_BONUS if i in keyword_hits else 0.0) for i, s in sims.items() if s >= best - max_gap}
        out = []
        for item_id in sorted(scores, key=scores.get, reverse=True)[:limit]:
            item = self.get(item_id)
            if item:
                item["similarity"] = round(sims[item_id], 3)
                out.append(item)
        return out

    def fused(self, query: str, qvec: Optional[Sequence[float]], *, candidates: int = 20,
              weights: Sequence[float] = (1.0, 1.0), skip: Callable[[str, float], bool] = lambda s, t: False,
              extra: Sequence[tuple] = (), kinds: Optional[Sequence[str]] = None) -> List[dict]:
        """Best-first rows by weighted reciprocal rank fusion of the vector ranking and the keyword ranking (each
        ``candidates`` deep), plus any ``extra`` (weight, [ids]) rankings. No similarity floor: what reaches the
        prompt is decided by the caller's budget. ``qvec`` None (embedding server down) = keywords only."""
        meta = self._meta()
        keep = lambda i: i in meta and not skip(*meta[i][:2]) and (kinds is None or meta[i][2] in kinds)  # noqa: E731
        vector = [i for i, _ in self._nearest(qvec, candidates * 4) if keep(i)][:candidates] if qvec is not None else []
        keyword = [i for i in self._keyword_hits(query, candidates * 4) if keep(i)][:candidates]
        score: Dict[int, float] = {}
        for weight, ranking in ((weights[0], vector), (weights[1], keyword), *extra):
            for rank, item_id in enumerate(ranking):
                if keep(item_id):
                    score[item_id] = score.get(item_id, 0.0) + weight / (_RRF_K + rank)
        return [row for row in (self.get(i) for i in sorted(score, key=score.get, reverse=True)) if row]

    def _keyword_hits(self, query: str, limit: int) -> List[int]:
        fts = _fts_query(query)
        if not fts:
            return []
        with self._lock:
            return [i for (i,) in self._conn.execute(
                "SELECT rowid FROM items_fts WHERE items_fts MATCH ? ORDER BY bm25(items_fts) LIMIT ?", (fts, limit))]

    def warm(self) -> None:
        """Load the search index now (a background thread at startup) rather than on the first recall."""
        self._meta()
        self._index()

    def _meta(self) -> Dict[int, tuple]:
        """id -> (session_id, created_at, kind), kept in memory and updated by add() and delete()."""
        with self._lock:
            if self._meta_cache is None:
                self._meta_cache = {r[0]: (r[1], r[2], r[3]) for r in self._conn.execute(
                    "SELECT id, session_id, created_at, kind FROM items")}
            return self._meta_cache

    def _index(self) -> "_VecIndex":
        with self._lock:
            if self._vecs is None:
                index = _VecIndex(self.dims)
                rows = self._conn.execute("SELECT id, substr(vec, 1, ?) FROM items WHERE vec IS NOT NULL AND model=? "
                                          "AND dims=?", (_PREFIX * 4, self.model, self.dims)).fetchall()
                index.load(rows)
                self._vecs = index
            return self._vecs

    def _full(self, ids: Sequence[int]) -> Dict[int, array]:
        out: Dict[int, array] = {}
        with self._lock:
            for i in range(0, len(ids), 500):  # SQLite caps bound parameters
                chunk = list(ids[i:i + 500])
                out.update((r[0], _floats(r[1])) for r in self._conn.execute(
                    f"SELECT id, vec FROM items WHERE id IN ({','.join('?' * len(chunk))})", chunk))
        return out

    def _nearest(self, qvec: Sequence[float], k: int) -> List[tuple]:
        """(id, cosine) best first. Stored and query vectors are unit length, so cosine = dot product. A first pass
        over every row's first _PREFIX dims (Matryoshka-trained vectors rank well cut short) picks _RESCORE rows;
        their full vectors, read back from SQLite, give the exact order."""
        with self._lock:
            shortlist = self._index().shortlist(qvec, max(k, _RESCORE))
        if len(qvec) <= _PREFIX:
            return shortlist[:k]
        full = self._full([i for i, _ in shortlist])
        if _np is not None and full:
            ids = list(full)
            sims = _np.frombuffer(b"".join(full[i].tobytes() for i in ids), dtype=_np.float32).reshape(
                len(ids), -1) @ _np.asarray(qvec, dtype=_np.float32)
            scored = zip(ids, sims.tolist())
        else:
            q = array("f", qvec)
            scored = ((i, sum(map(mul, v, q))) for i, v in full.items())
        return sorted(scored, key=lambda p: p[1], reverse=True)[:k]


_PREFIX = 256  # dims scanned for every row (Matryoshka); with _RESCORE it matched exact search on real vectors:
_RESCORE = 1000  # 100% of the top 20, 99.95% of the top 80 (LongMemEval queries over 100k facts)


def _floats(blob: bytes) -> array:
    v = array("f")
    v.frombytes(blob)
    return v


def _prefix(vec: Sequence[float]) -> array:
    """The first _PREFIX dims at unit length, as Matryoshka truncation is meant to be compared."""
    p = array("f", vec[:_PREFIX])
    n = sum(x * x for x in p) ** 0.5 or 1.0
    return array("f", (x / n for x in p))


def _headroom(n: int) -> int:
    return max(1024, n // 4)  # spare rows for new items; grows by a quarter, not double


class _VecIndex:
    """Every embedded row's vector prefix in memory: a numpy matrix grown in place (rows of deleted items are
    zeroed and skipped), or plain arrays without numpy. About 1 KB per row."""

    def __init__(self, dims: int):
        self.width = min(dims, _PREFIX)
        self.ids: List[int] = []
        self.pos: Dict[int, int] = {}
        self.rows: list = []  # plain-Python rows
        self.mat = _np.zeros((1024, self.width), dtype=_np.float32) if _np is not None else None

    def load(self, rows: Sequence[tuple]) -> None:
        """Bulk start: (id, prefix bytes) pairs, read without the rest of each vector."""
        if self.mat is None or not rows:
            for item_id, blob in rows:
                self.put(item_id, _floats(blob))
            return
        self.ids = [r[0] for r in rows]
        self.pos = {i: j for j, i in enumerate(self.ids)}
        self.mat = _np.zeros((len(rows) + _headroom(len(rows)), self.width), dtype=_np.float32)
        part = self.mat[:len(rows)]
        part[:] = _np.frombuffer(b"".join(r[1] for r in rows), dtype=_np.float32).reshape(len(rows), self.width)
        part /= _np.maximum(_np.linalg.norm(part, axis=1, keepdims=True), 1e-12)

    def put(self, item_id: int, vec: Sequence[float]) -> None:
        p = _prefix(vec)
        j = self.pos.get(item_id)
        if j is None:
            j = self.pos[item_id] = len(self.ids)
            self.ids.append(item_id)
            if self.mat is None:
                self.rows.append(p)
            elif j >= len(self.mat):
                self.mat = _np.concatenate([self.mat, _np.zeros((_headroom(j), self.width), dtype=_np.float32)])
        if self.mat is not None:
            self.mat[j] = p
        else:
            self.rows[j] = p

    def drop(self, item_id: int) -> None:
        j = self.pos.pop(item_id, None)
        if j is None:
            return
        self.ids[j] = -1
        if self.mat is not None:
            self.mat[j] = 0
        else:
            self.rows[j] = None

    def shortlist(self, qvec: Sequence[float], n: int) -> List[tuple]:
        """(id, prefix cosine) of the n best rows, best first."""
        if not self.pos:
            return []
        q = _prefix(qvec)
        if self.mat is not None:
            sims = self.mat[:len(self.ids)] @ _np.frombuffer(q.tobytes(), dtype=_np.float32)
            top = _np.argpartition(-sims, n)[:n] if n < len(sims) else _np.arange(len(sims))
            top = top[_np.argsort(-sims[top])]
            return [(self.ids[j], float(sims[j])) for j in top.tolist() if self.ids[j] >= 0]
        return heapq.nlargest(n, ((self.ids[j], sum(map(mul, r, q))) for j, r in enumerate(self.rows)
                                  if r is not None), key=lambda p: p[1])


def _row(row) -> dict:
    return {"id": row[0], "kind": row[1], "session_id": row[2], "text": row[3], "created_at": row[4],
            "parent": row[5] if len(row) > 5 else None}
