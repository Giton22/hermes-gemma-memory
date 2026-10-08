"""Which LongMemEval-S questions a run uses: one definition for every harness, so arms are always paired.

split   dev / test: a fixed 50/50 by question-id hash (tune on dev; test is held out for final numbers)
per_type  N questions per type (abstention counted as its own type), seeded, taken after the split
"""

import collections
import hashlib
import random


def qtype(q):
    return "abstention" if q["question_id"].endswith("_abs") else q["question_type"]


def in_split(question_id, split):
    if split == "all":
        return True
    return int(hashlib.md5(question_id.encode()).hexdigest(), 16) % 2 == (0 if split == "dev" else 1)


def select(data, split="all", per_type=0):
    data = [q for q in data if in_split(q["question_id"], split)]
    if not per_type:
        return data
    rng, by = random.Random(0), collections.defaultdict(list)
    for q in data:
        by[qtype(q)].append(q)
    return [q for t in sorted(by) for q in rng.sample(by[t], min(per_type, len(by[t])))]
