# gemma-memory

Long-term memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Every conversation is kept
and the relevant parts are recalled before each reply, so Hermes remembers what you told it weeks ago, in any
chat. Embeddings run locally with [EmbeddingGemma 2](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)
on a CPU; no conversation leaves your network unless you turn on the optional fact layer.

## How it works

- **Storing.** After each reply, the exchange is saved to a local SQLite database, whole and split into short
  passages. Each passage is embedded by EmbeddingGemma 2. Keys, tokens and passwords are masked first.
  Pictures sent in a chat are saved and embedded from their pixels into the same space as text.
- **Recalling.** Before each reply, your message is matched against memory by meaning (vectors) and by words
  (keyword search), and the two rankings are combined. The best passages, up to about 9,000 characters, are
  attached to your message, grouped by conversation and date. No tool call is needed. Turns from the current
  chat are left out: the model already sees them.
- **Facts (optional).** At the end of a conversation, one LLM call writes a few short, dated facts about it
  ("The user's sister moved to Lisbon in March 2025"). They are stored next to the passages and help with questions that
  combine several conversations or need dates worked out.
- **Robust.** Text is stored before it is embedded. If the embedding server is down, recall falls back to keyword
  search and the missing vectors are filled in later. Changing the model or size re-embeds everything.

The agent also gets three tools: `memory_recall` (search further), `memory_note` (save something on purpose)
and `memory_forget` (delete an item). Cron jobs and subagents read memory but don't write to it.

## Setup

### 1. Run an embedding server

EmbeddingGemma 2 needs llama.cpp's server **from 2026-10-06 or newer** (tested with b11467). Older builds fail with
`unknown model architecture: 'gemma-embedding2'`. Ollama can't serve it on a CPU yet.

With Docker (on TrueNAS: Apps → Discover → ⋮ → Install via YAML):

```yaml
services:
  embeddings:
    image: ghcr.io/ggml-org/llama.cpp:server
    command: >
      -hf ggml-org/embeddinggemma-2-GGUF:Q8_0
      --embedding --host 0.0.0.0 --port 8080
      -c 8192 -b 8192 -ub 8192
    environment:
      LLAMA_CACHE: /models
    volumes:
      - /path/to/models:/models
    ports:
      - "8091:8080"
    restart: unless-stopped
```

The first start downloads about 870 MB (the model plus the image encoder for pictures). On a 4-thread CPU a
search takes about 0.2 s.

### 2. Install the plugin

```bash
hermes plugins install Giton22/hermes-gemma-memory
```

Accept numpy when asked: search is much faster with it.

### 3. Configure it

In Hermes' `config.yaml`:

```yaml
memory:
  provider: gemma-memory
plugins:
  gemma-memory:
    base_url: http://<server-ip>:8091/v1
```

Restart Hermes and check:

```bash
hermes gemma-memory status    # "server": "ok"
```

### 4. Bring in past conversations (optional)

```bash
hermes gemma-memory import                # all past chats, with their original dates
hermes gemma-memory import --since 2026-01-01
```

Safe to run again: conversations already in memory are skipped.

## Options (`plugins.gemma-memory`)

| Key | Default | |
|---|---|---|
| `base_url` | `http://localhost:8080/v1` | OpenAI-compatible embeddings endpoint |
| `recall_budget` | `9000` | Characters recalled before each reply; kept under Hermes' prefetch limit |
| `use_facts` | `false` | Write a few dated facts per conversation (one LLM call each) |
| `facts_base_url` / `facts_model` | | Any OpenAI-compatible chat endpoint for facts, local or cloud; key in `GEMMA_MEMORY_FACTS_API_KEY` |
| `redact_secrets` | `true` | Mask keys, tokens and passwords before storing |
| `dims` | `768` | Vector size: 768, 512, 256 or 128 |

Set `GEMMA_MEMORY_API_KEY` if your embedding server needs a key.

## Commands

```bash
hermes gemma-memory status           # counts, model, server check
hermes gemma-memory search <words>   # what the agent would find
hermes gemma-memory import           # past conversations
hermes gemma-memory backfill         # embed anything stored while the server was down
```

## Good to know

- Recalled text is attached to your message, so it stays in that chat's history and is sent again with later
  messages until Hermes compresses them. Lower `recall_budget` to save tokens.
- With `use_facts`, finished conversations are sent to the configured LLM. Point it at a local model to keep
  everything at home.
- Memory lives in `$HERMES_HOME/gemma-memory/` (database and pictures), unencrypted, per profile. Secret masking
  is pattern-based: it catches common keys and passwords, not everything.
- Questions that need pieces from several conversations are the hardest case; the fact layer helps most there.

## License

MIT. EmbeddingGemma 2 is under Google's Gemma terms.
