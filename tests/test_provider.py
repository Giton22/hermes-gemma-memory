import json
import math

import pytest

from gemma_memory.embedder import Embedder, truncate
from gemma_memory.provider import DEFAULTS, GemmaMemoryProvider


def make(server, tmp_path, session="s1", **config):
    p = GemmaMemoryProvider({**DEFAULTS, "base_url": server.url, **config})
    p.initialize(session, hermes_home=str(tmp_path))
    return p


def settle(p):
    if p._worker:
        p._worker.join(5)


def test_truncate_is_unit_length_and_short():
    v = truncate([3.0, 4.0, 12.0], 2)
    assert v == pytest.approx([0.6, 0.8]) and math.isclose(sum(x * x for x in v), 1.0)


def test_query_and_documents_use_task_prompts(server):
    e = Embedder(server.url, "embeddinggemma", 256)
    assert len(e.query("hi there")) == 256
    e.documents(["a", "b"])
    assert server.calls[0]["input"] == ["task: search result | query: hi there"]
    assert server.calls[1]["input"] == ["title: none | text: a", "title: none | text: b"]


def test_a_past_chat_is_recalled_in_a_new_one(server, tmp_path):
    p = make(server, tmp_path)
    p.sync_turn("My NAS runs TrueNAS Scale with a ZFS pool called tank", "Noted, tank on TrueNAS Scale.", session_id="s1")
    settle(p)
    # Same chat: the turn is still in the context window, so recalling it would only duplicate it.
    assert p.prefetch("which ZFS pool is on my NAS?", session_id="s1") == ""
    p.on_session_switch("s2", reset=True)
    block = p.prefetch("which ZFS pool is on my NAS?", session_id="s2")
    assert "tank" in block and p.recall_status().count == 1
    p.shutdown()


def test_compressed_turns_of_this_chat_become_recallable(server, tmp_path):
    p = make(server, tmp_path)
    p.sync_turn("deploy target is the homelab box hermes01", "ok, hermes01 it is", session_id="s1")
    settle(p)
    assert p.prefetch("what is the deploy target", session_id="s1") == ""
    p.on_pre_compress([])
    assert "hermes01" in p.prefetch("what is the deploy target", session_id="s1")
    p.shutdown()


def test_branch_does_not_recall_its_parent(server, tmp_path):
    p = make(server, tmp_path)
    p.sync_turn("the wifi password policy is rotation every 90 days", "understood", session_id="s1")
    settle(p)
    p.on_session_switch("s1-branch", parent_session_id="s1")
    assert p.prefetch("wifi password rotation policy", session_id="s1-branch") == ""
    p.shutdown()


def test_server_down_keeps_text_and_backfills_later(server, tmp_path):
    p = make(server, tmp_path)
    server.up = False
    p.sync_turn("my favourite editor is helix", "helix noted", session_id="s1")
    settle(p)
    stats = p._store.stats()
    assert stats["not_embedded"] == stats["total"] == 1 + stats["by_kind"]["passage"]  # the turn and its passages
    p.on_session_switch("s2")
    assert "helix" in p.prefetch("favourite editor helix", session_id="s2")  # keyword fallback
    server.up = True
    p.sync_turn("I use fish as my shell", "fish it is", session_id="s2")
    settle(p)
    assert p._store.stats()["not_embedded"] == 0
    p.shutdown()


def test_changing_model_or_size_reembeds_on_start(server, tmp_path):
    p = make(server, tmp_path)
    p.sync_turn("the backup job runs nightly at 3am", "noted", session_id="s1")
    settle(p)
    p.shutdown()
    q = make(server, tmp_path, session="s2", dims=128, model="embeddinggemma2")
    settle(q)
    stats = q._store.stats()
    assert stats["not_embedded"] == 0 and server.calls[-1]["model"] == "embeddinggemma2"
    assert "3am" in q.prefetch("when does the backup job run", session_id="s2")
    q.shutdown()


def test_builtin_memory_writes_are_mirrored(server, tmp_path):
    p = make(server, tmp_path)
    p.on_memory_write("add", "user", "Prefers answers in British English")
    settle(p)
    p.on_memory_write("replace", "user", "Prefers answers in Slovenian",
                      {"previous_content": "Prefers answers in British English"})
    settle(p)
    texts = [r["text"] for r in p._store.search("prefers answers", None, limit=10)]
    assert texts == ["Prefers answers in Slovenian"]
    p.shutdown()


def test_tools_note_recall_forget(server, tmp_path):
    p = make(server, tmp_path)
    saved = json.loads(p.handle_tool_call("memory_note", {"content": "Router admin page is at 192.0.2.1"}))
    found = json.loads(p.handle_tool_call("memory_recall", {"query": "router admin page"}))["results"]
    assert found[0]["id"] == saved["id"] and found[0]["kind"] == "note"
    assert json.loads(p.handle_tool_call("memory_forget", {"id": saved["id"]})) == {"deleted": True}
    assert json.loads(p.handle_tool_call("memory_recall", {"query": "router admin page"}))["results"] == []
    assert "error" in json.loads(p.handle_tool_call("memory_recall", {}))
    p.shutdown()


@pytest.mark.parametrize("ctx", ["cron", "subagent", "flush"])
def test_background_agents_read_but_dont_write(server, tmp_path, ctx):
    p = GemmaMemoryProvider({**DEFAULTS, "base_url": server.url})
    p.initialize("s1", hermes_home=str(tmp_path), agent_context=ctx)
    p.sync_turn("something long enough to store", "a reply", session_id="s1")
    p.on_memory_write("add", "memory", "a fact")
    settle(p)
    assert p._store.stats()["total"] == 0
    p.shutdown()


def test_trivial_turns_are_skipped(server, tmp_path):
    p = make(server, tmp_path)
    p.sync_turn("thanks!", "You're welcome.", session_id="s1")
    settle(p)
    assert p._store.stats()["total"] == 0 and p.prefetch("ok") == ""
    p.shutdown()


def image_uri(words):
    return "data:image/png;base64," + __import__("base64").b64encode(words.encode()).decode()


def turn_with_image(words, text="look at this"):
    return [{"role": "user", "content": [{"type": "text", "text": text},
                                         {"type": "image_url", "image_url": {"url": image_uri(words)}}]},
            {"role": "assistant", "content": "Nice picture."}]


def test_images_are_saved_embedded_and_recalled_by_text(server, tmp_path):
    p = make(server, tmp_path)
    p.sync_turn("look at this", "Nice picture.", session_id="s1", messages=turn_with_image("red bicycle leaning on a fence"))
    settle(p)
    images = list((tmp_path / "gemma-memory" / "images").iterdir())
    assert len(images) == 1 and images[0].read_bytes() == b"red bicycle leaning on a fence"
    p.on_session_switch("s2", reset=True)
    found = json.loads(p.handle_tool_call("memory_recall", {"query": "red bicycle by the fence"}))["results"]
    assert found[0]["kind"] == "image" and str(images[0]) in found[0]["text"]
    p.shutdown()


def test_resent_image_is_stored_once_on_disk(server, tmp_path):
    p = make(server, tmp_path)
    for sid in ("s1", "s2"):
        p.sync_turn("again", "Same picture.", session_id=sid, messages=turn_with_image("blue mug"))
    settle(p)
    assert len(list((tmp_path / "gemma-memory" / "images").iterdir())) == 1
    p.shutdown()


def test_image_waits_for_a_server_that_can_embed_images(server, tmp_path):
    server.images = False  # e.g. llama-server started without --mmproj
    p = make(server, tmp_path)
    p.sync_turn("look at this", "Nice picture.", session_id="s1", messages=turn_with_image("green tent"))
    settle(p)
    assert p._store.stats()["not_embedded"] == 1  # the image; the text turn is embedded
    server.images = True
    assert p._backfill() == 1 and p._store.stats()["not_embedded"] == 0
    p.shutdown()


def test_web_image_links_are_not_fetched(server, tmp_path):
    p = make(server, tmp_path)
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]
    p.sync_turn("what is this?", "A chart.", session_id="s1", messages=msgs)
    settle(p)
    assert "image" not in p._store.stats()["by_kind"]
    p.shutdown()



def test_turns_get_passages_and_passage_recall_finds_the_span(server, tmp_path):
    p = make(server, tmp_path, recall_unit="passage", recall_budget=1000)
    filler = "Here is a general overview of storage options and some background reading. " * 40
    p.sync_turn("what NAS should I buy?", filler + "For your case the Synology DS923 with four bays fits best.",
                session_id="s1")
    settle(p)
    kinds = p._store.stats()["by_kind"]
    assert kinds["turn"] == 1 and kinds["passage"] >= 3
    p.on_session_switch("s2", reset=True)
    block = p.prefetch("which Synology DS923 four bays NAS", session_id="s2")
    assert "DS923" in block and len(block) < len(filler)  # the span, not the whole reply


def test_deleting_a_turn_deletes_its_passages(server, tmp_path):
    p = make(server, tmp_path)
    p.sync_turn("my bike is a Brompton", "Nice folding bike.", session_id="s1")
    settle(p)
    turn = next(i for i in range(1, 20) if (r := p._store.get(i)) and r["kind"] == "turn")
    assert p._store.children(turn) and p._store.delete(turn)
    assert p._store.stats()["total"] == 0


def test_turns_stored_before_passages_get_them(server, tmp_path):
    p = make(server, tmp_path)
    p._store.add("turn", "User: old turn about tents\nAssistant: Bring a green tent.", session_id="s0")
    assert p._split_old_turns() == 1 and p._store.stats()["by_kind"]["passage"] == 2
    p.shutdown()


def test_facts_are_extracted_once_per_session_and_recalled(server, tmp_path, monkeypatch):
    from gemma_memory import facts
    calls = []

    def fake_chat_factory(base_url, model, api_key="", **kw):
        def chat(prompt):
            calls.append(prompt)
            return '["The user keeps a bonsai named Kenji, watered on Sundays."]'
        return chat

    monkeypatch.setattr(facts, "openai_chat", fake_chat_factory)
    p = make(server, tmp_path, use_facts=True, facts_base_url="http://llm.invalid/v1", facts_model="m")
    p.sync_turn("my little tree needs water", "Noted, water it weekly.", session_id="s1")
    settle(p)
    p.on_session_end([])
    settle(p)
    p.on_session_end([])  # a second end of the same session: no second extraction
    settle(p)
    assert len(calls) == 1 and p._store.stats()["by_kind"]["fact"] == 1
    p.on_session_switch("s2", reset=True)
    assert "Fact: The user keeps a bonsai named Kenji" in p.prefetch("bonsai named Kenji watering day", session_id="s2")
    p.shutdown()


KEY = "sk-or-v1-" + "0123456789abcdef" * 4


def stored_texts(p):
    return [t for _, _, t in p._store.texts()]


def test_secrets_are_masked_before_storing(server, tmp_path):
    p = make(server, tmp_path)
    p.sync_turn(f"use this key {KEY} for the router, wifi password is Sunflower2024",
                "Saved: the router at 192.0.2.1 uses that key.", session_id="s1")
    p.handle_tool_call("memory_note", {"content": f"DB_PASSWORD=Hunter2xyz and {KEY}"})
    settle(p)
    texts = stored_texts(p)
    assert texts and not any(KEY in t or "Sunflower2024" in t or "Hunter2xyz" in t for t in texts)
    assert any("192.0.2.1" in t for t in texts)  # ordinary details stay
    assert not p._store._keyword_hits("Sunflower2024", 5)  # nor in the keyword index
    p.shutdown()


def test_secrets_stored_before_redaction_are_masked_once(server, tmp_path):
    p = make(server, tmp_path, redact_secrets=False)
    p.sync_turn(f"the openrouter key is {KEY}", "noted, the openrouter key", session_id="s1")
    settle(p)
    assert any(KEY in t for t in stored_texts(p))
    p.shutdown()

    p = make(server, tmp_path)  # restarted with redaction on: old rows cleaned, passages and vectors rebuilt
    settle(p)
    texts = stored_texts(p)
    assert not any(KEY in t for t in texts)
    assert p._store.stats()["not_embedded"] == 0
    assert [r["kind"] for r in p._store.session_items("s1", "passage")]
    p.on_session_switch("s2", reset=True)
    assert "openrouter key" in p.prefetch("what is my openrouter key?", session_id="s2")
    assert p._store.get_flag("redacted")
    p.shutdown()


def test_redaction_can_be_turned_off(server, tmp_path):
    p = make(server, tmp_path, redact_secrets=False)
    p.handle_tool_call("memory_note", {"content": f"key {KEY}"})
    settle(p)
    assert any(KEY in t for t in stored_texts(p))
    p.shutdown()


def test_history_pairs_each_question_with_its_final_answer():
    from gemma_memory import history
    msgs = [
        {"role": "user", "content": "what's the weather in the pool room?", "timestamp": 100.0},
        {"role": "assistant", "content": "", "tool_calls": "[...]", "timestamp": 101.0},
        {"role": "tool", "content": "21C", "timestamp": 102.0},
        {"role": "assistant", "content": "It's 21C in the pool room.", "timestamp": 103.0},
        {"role": "user", "content": "summary", "timestamp": 150.0, "_compressed_summary": 1},
        {"role": "user", "content": '[{"type": "text", "text": "and tomorrow?"}, {"type": "image_url"}]',
         "timestamp": 200.0},
        {"role": "assistant", "content": "Sunny tomorrow.", "timestamp": 201.0},
        {"role": "user", "content": "never answered", "timestamp": 300.0},
    ]
    assert history.turns(msgs) == [("what's the weather in the pool room?", "It's 21C in the pool room.", 100.0),
                                   ("and tomorrow?", "Sunny tomorrow.", 200.0)]


def test_past_conversations_are_imported_once_with_their_dates(server, tmp_path):
    p = make(server, tmp_path)
    old = [{"role": "user", "content": f"the boiler service is booked for March, key {KEY}", "timestamp": 1.7e9},
           {"role": "assistant", "content": "Noted: boiler service in March.", "timestamp": 1.7e9 + 5},
           {"role": "user", "content": "thanks", "timestamp": 1.7e9 + 10},
           {"role": "assistant", "content": "You're welcome!", "timestamp": 1.7e9 + 11}]
    assert p.import_session("old-1", old) == 1  # "thanks" is trivial
    assert p.import_session("old-1", old) == 0  # already there
    p._backfill()
    turn = p._store.session_items("old-1", "turn")[0]
    assert turn["created_at"] == 1.7e9 and KEY not in turn["text"]
    assert p._store.session_items("old-1", "passage") and p._store.stats()["not_embedded"] == 0
    p.on_session_switch("now", reset=True)
    block = p.prefetch("when is the boiler service?", session_id="now")
    assert "boiler service" in block and "2023-11-14" in block  # recalled under its original date
    p.shutdown()


def test_secrets_are_masked_even_without_hermes_redactor(monkeypatch):
    import builtins

    from gemma_memory import provider as prov
    real_import = builtins.__import__

    def no_redact(name, *a, **kw):
        if name == "agent.redact":
            raise ImportError(name)
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", no_redact)
    text = (f"key {KEY}, export API_TOKEN=abc123def456, db postgres://me:S3cret@db:5432/x, "
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345, wifi password is Sunflower2024, server 192.0.2.10")
    out = prov.redact(text)
    for secret in (KEY, "abc123def456", "S3cret", "abcdefghijklmnopqrstuvwxyz012345", "Sunflower2024"):
        assert secret not in out
    assert "192.0.2.10" in out


def test_status_explains_common_server_failures():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("gm_cli", os.path.join(os.path.dirname(__file__), "..", "cli.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert "2026-10-06" in cli._server_hint("HTTP 500 error loading model: unknown model architecture: 'gemma-embedding2'")
    assert "MLX" in cli._server_hint("this model requires MLX support")
    assert "running" in cli._server_hint("<urlopen error [Errno 111] Connection refused>")
    assert cli._server_hint("something else") == ""
