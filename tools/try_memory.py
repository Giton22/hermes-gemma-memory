"""Try the provider end to end against a real embeddings server: store past turns, ask paraphrased questions,
print exactly what prefetch() would inject into the prompt.

    HERMES_AGENT_PATH=... python tools/try_memory.py http://127.0.0.1:8099/v1 [recall_budget]
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.environ["HERMES_AGENT_PATH"])
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma_memory.provider import DEFAULTS, GemmaMemoryProvider  # noqa: E402

PAST = [  # (session, user, assistant): a made-up homelab owner; addresses are RFC 5737 / example.com placeholders
    ("a", "My NAS runs a ZFS pool called tank", "Got it: ZFS, pool tank."),
    ("a", "App data goes in tank/apps, one dataset per app", "Noted, tank/apps/<app> per app."),
    ("a", "The NAS is at 192.0.2.10 and its web UI is on port 8080", "OK, UI at 192.0.2.10:8080."),
    ("b", "I expose my recipe app, wiki and chat UI through a tunnel on example.com", "Understood, three hostnames via the tunnel."),
    ("b", "The smart-home hub runs on a separate box, 192.0.2.20", "Noted, the hub is not on the NAS."),
    ("c", "Notes is my phone app for the assistant, written in Swift", "Notes: a Swift client."),
    ("c", "Release builds are split per device type and the in-app updater picks the right one", "Per-device builds, updater chooses."),
    ("c", "Push notifications go through a self-hosted relay, end-to-end encrypted to the phone's key", "E2E push through the relay."),
    ("d", "I hate long answers, keep it short and skip the preamble", "Will keep replies brief."),
    ("d", "I write commit messages in plain English, no conventional-commit prefixes", "Plain commit subjects, noted."),
    ("e", "The NAS has no proper GPU, everything runs on the CPU", "CPU-only inference then."),
    ("e", "Embeddings come from a llama.cpp server on port 8091", "llama.cpp on :8091 for embeddings."),
    ("f", "I mostly cook Italian food, risotto and pasta e fagioli", "Italian dishes, got it."),
    ("f", "I'm allergic to peanuts", "I'll keep peanuts out of suggestions."),
    ("g", "The password manager is self-hosted on port 8090", "Password manager on :8090."),
    ("g", "Backups: the pool replicates to an external USB disk every Sunday night", "Weekly replication on Sundays."),
    ("h", "My desktop has a gaming GPU, I run local models there", "Local models on the desktop GPU."),
    ("h", "I prefer fish shell on Linux boxes", "fish on Linux, noted."),
]

QUESTIONS = [
    "where should I put the files for a new app on the server?",       # tank/apps dataset
    "what's the IP of my storage box?",                                  # 192.0.2.10
    "which of my services can be reached from outside the house?",      # the tunnel
    "how do notifications reach my phone?",                              # the push relay
    "what should I make for dinner?",                                    # Italian food + peanut allergy
    "can I run a big model on the nas with a graphics card?",            # no GPU
    "how often is my data copied off the nas?",                          # Sunday replication
    "what's the password manager setup?",                                # self-hosted, port 8090
    "what is the capital of Australia?",                                 # nothing relevant
    "explain how TCP congestion control works",                          # nothing relevant
]


def main():
    url = sys.argv[1]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 1500  # small, so the printout stays readable
    home = tempfile.mkdtemp()
    p = GemmaMemoryProvider({**DEFAULTS, "base_url": url, "model": "embeddinggemma-2", "dims": 768,
                             "recall_budget": budget})
    p.initialize("a", hermes_home=home)
    t = time.time()
    for session, user, assistant in PAST:
        p.on_session_switch(session)
        p.sync_turn(user, assistant, session_id=session)
    p._worker.join()
    print(f"stored {p._store.stats()['total']} turns in {time.time() - t:.1f}s\n")
    p.on_session_switch("now", reset=True)
    for q in QUESTIONS:
        t = time.time()
        block = p.prefetch(q)  # exactly what Hermes would inject
        ms = (time.time() - t) * 1000
        print(f"Q: {q}  ({ms:.0f} ms)")
        for line in block.splitlines()[1:]:
            print("   " + line[:110])
        print()
    p.shutdown()


if __name__ == "__main__":
    main()
