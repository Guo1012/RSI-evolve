"""Reward primitive library P (paper Table 3-2).  Every primitive maps (prompt, response[, group]) -> [0,1].

Primitives only read D_train prompts/labels and policy outputs.  `judge` is the frozen initial policy
pi_0 (mean token log-prob); it is never used by the fitness or audit evaluators, which use the
programmatic task checkers on other splits.  `selfscore` is the *current* policy's own log-prob: a
deliberately risky, endogenous signal kept in the library to see whether the search learns to avoid it.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..constraints import Constraint
from ..data import answers_match, extract_answer
from ..modeling import is_degenerate
from ..synth import substantive, valid_math_chain

PRIMS = ["correct", "partial", "format", "steps", "consistency", "judge", "length", "repeat", "novelty", "selfscore"]
CATEGORY = {"correct": "correctness", "partial": "correctness", "format": "format", "steps": "process",
            "consistency": "process", "judge": "judge", "length": "regularizer", "repeat": "regularizer",
            "novelty": "diversity", "selfscore": "risk"}
CORRECTNESS = {"correct", "partial"}


def _grams(t, n=3):
    w = t.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def _rep_rate(t):
    w = t.split()
    if len(w) < 8:
        return 0.0
    g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1.0 - len(set(g)) / len(g)


def _lp_norm(lp):
    # mean log-prob ~[-3, 0] -> (0,1); fixed affine squash, no data-dependent fitting
    return 1.0 / (1.0 + math.exp(-(lp + 1.2) / 0.35))


def compute(items: list[dict], groups: list[list[dict]], judge_lp=None, self_lp=None) -> dict[str, np.ndarray]:
    """items[i] = D_train item, groups[i] = K samples {'text','finished'}; *_lp = flat lists (i*K+k)."""
    n, K = len(items), len(groups[0])
    P = {p: np.zeros((n, K), dtype=np.float32) for p in PRIMS}
    for i, (it, grp) in enumerate(zip(items, groups)):
        texts = [s["text"] for s in grp]
        if it["task_type"] == "if":
            cons = [Constraint.from_dict(c) for c in it["constraints"]]
            kinds = [c.kind for c in cons]
            checks = [[c.check(t) for c in cons] for t in texts]
            maj = Counter(tuple(c) for c in checks).most_common(1)[0][0]
        else:
            preds = [extract_answer(t) for t in texts]
            vals = [p for p in preds if p is not None]
            maj = Counter(vals).most_common(1)[0][0] if vals else None
        lens = [len(t.split()) for t in texts]
        ref = max(1.0, float(np.median(lens)))
        grams = [_grams(t) for t in texts]
        for k, s in enumerate(grp):
            t = s["text"]
            if it["task_type"] == "if":
                P["correct"][i, k] = float(all(checks[k]))
                P["partial"][i, k] = sum(checks[k]) / len(checks[k])
                P["format"][i, k] = float(s["finished"])
                P["steps"][i, k] = float(substantive(t, kinds))
                P["consistency"][i, k] = float(tuple(checks[k]) == maj)
            else:
                ok = answers_match(preds[k], it["answer"])
                P["correct"][i, k] = float(ok)
                P["partial"][i, k] = float(preds[k] is not None) * 0.5 + 0.5 * float(ok)
                P["format"][i, k] = float(s["finished"] and "answer is" in t.lower())
                P["steps"][i, k] = float(valid_math_chain(t))
                P["consistency"][i, k] = float(preds[k] is not None and preds[k] == maj)
            P["length"][i, k] = 1.0 - min(1.0, abs(lens[k] - ref) / ref)
            P["repeat"][i, k] = 0.0 if is_degenerate(t, s["finished"]) else 1.0 - min(1.0, 2 * _rep_rate(t))
            others = [grams[j] for j in range(K) if j != k]
            sim = max((len(grams[k] & o) / max(1, len(grams[k] | o)) for o in others), default=0.0)
            P["novelty"][i, k] = 1.0 - sim
            if judge_lp is not None:
                P["judge"][i, k] = _lp_norm(judge_lp[i * K + k])
            if self_lp is not None:
                P["selfscore"][i, k] = _lp_norm(self_lp[i * K + k])
    return P
