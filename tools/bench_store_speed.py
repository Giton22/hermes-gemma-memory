"""How long one recall search takes as the store grows: random unit vectors and short texts, timed with and without
numpy (Hermes' environment may not have it). Search quality is measured elsewhere; this is only speed and memory.

    python tools/bench_store_speed.py --items 20000,100000
"""

import argparse
import os
import random
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma_memory import store as store_mod  # noqa: E402

WORDS = "alpha beta gamma delta port server backup photo trip sister dentist budget router python garden".split()


def unit(rng, dims):
    v = [rng.gauss(0, 1) for _ in range(dims)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="20000,100000")
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--queries", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(0)
    pool = [unit(rng, args.dims) for _ in range(500)]  # reused vectors: building 100k random ones is the slow part
    for n in map(int, args.items.split(",")):
        with tempfile.TemporaryDirectory() as d:
            s = store_mod.Store(os.path.join(d, "m.db"), model="m", dims=args.dims)
            t0 = time.time()
            for i in range(n):
                s.add("passage", " ".join(rng.choices(WORDS, k=12)), session_id=f"s{i // 50}",
                      vec=pool[i % len(pool)])
            build = time.time() - t0
            np_saved = store_mod._np
            for label, np_mod in (("numpy", np_saved), ("plain", None)):
                if label == "numpy" and np_mod is None:
                    continue
                store_mod._np = np_mod
                s2 = store_mod.Store(os.path.join(d, "m.db"), model="m", dims=args.dims)  # cold: loads vectors
                tracemalloc.start()
                t0 = time.time()
                s2.fused("backup router port", pool[0])
                cold = time.time() - t0
                mem = tracemalloc.get_traced_memory()[1] / 1e6
                tracemalloc.stop()
                times = []
                for q in range(args.queries):
                    t0 = time.time()
                    s2.fused("sister trip photo", pool[q + 1])
                    times.append(time.time() - t0)
                print(f"{n:>7} items  {label:>5}: first search {cold:6.2f}s (peak {mem:5.0f} MB), "
                      f"then {sorted(times)[len(times) // 2] * 1000:7.0f} ms per search", flush=True)
                s2.close()
            store_mod._np = np_saved
            s.close()
        print(f"        (built in {build:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
