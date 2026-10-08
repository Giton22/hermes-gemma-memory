"""The vector index: a prefix scan plus full-vector re-ranking must give exact nearest neighbours, stay in step with
adds, re-embeds and deletes, and behave the same with and without numpy."""

import random

import pytest

from gemma_memory import store as store_mod

DIMS = 768


def unit(rng):
    """Matryoshka-like: most of the signal in the leading dims, as in EmbeddingGemma (isotropic random vectors
    would make any prefix scan look bad; tools/ checks on real vectors gave 100% of the exact top 20)."""
    v = [rng.gauss(0, 1) * (1.0 if d < 256 else 0.15) for d in range(DIMS)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def exact(vecs, q, k):
    return [i for i, _ in sorted(((i, sum(a * b for a, b in zip(v, q))) for i, v in vecs.items()),
                                 key=lambda p: p[1], reverse=True)[:k]]


@pytest.fixture(params=["numpy", "plain"])
def np_mode(request, monkeypatch):
    if request.param == "numpy":
        pytest.importorskip("numpy")
    else:
        monkeypatch.setattr(store_mod, "_np", None)
    monkeypatch.setattr(store_mod, "_RESCORE", 50)  # small, so 300 rows exercise the shortlist
    return request.param


def test_nearest_matches_exact_search(tmp_path, np_mode):
    rng = random.Random(1)
    s = store_mod.Store(tmp_path / "m.db", model="m", dims=DIMS)
    vecs = {s.add("passage", f"row {n}", vec=v): v for n, v in enumerate(unit(rng) for _ in range(300))}
    for _ in range(5):
        q = unit(rng)
        assert [i for i, _ in s._nearest(q, 10)] == exact(vecs, q, 10)


def test_index_follows_adds_reembeds_and_deletes(tmp_path, np_mode):
    rng = random.Random(2)
    s = store_mod.Store(tmp_path / "m.db", model="m", dims=DIMS)
    first = s.add("passage", "first", vec=unit(rng))
    s.warm()  # index built: later changes must update it in place
    target = unit(rng)
    added = s.add("passage", "added after warm", vec=target)
    assert s._nearest(target, 1)[0][0] == added

    s.set_vec(first, target)  # re-embedded: now the closest match too
    assert {i for i, _ in s._nearest(target, 2)} == {first, added}

    s.delete(added)
    assert [i for i, _ in s._nearest(target, 5)] == [first]
    assert added not in s._meta()

    reopened = store_mod.Store(tmp_path / "m.db", model="m", dims=DIMS)  # same answer from a cold load
    assert [i for i, _ in reopened._nearest(target, 5)] == [first]


def test_index_grows_past_its_spare_rows(tmp_path, np_mode):
    rng = random.Random(3)
    s = store_mod.Store(tmp_path / "m.db", model="m", dims=DIMS)
    s.warm()
    vecs = {s.add("passage", f"row {n}", vec=v): v for n, v in enumerate(unit(rng) for _ in range(1100))}
    q = unit(rng)
    assert [i for i, _ in s._nearest(q, 5)] == exact(vecs, q, 5)
