"""Five disjoint data splits (paper Table 3-1).

  D_train : prompts for recursive self-training (constrained alpaca instructions + GSM8K train)
  D_fit   : outer-loop fitness only
  D_sel   : final selection only (read once, after the search is frozen)
  D_test  : official IFEval (541) + GSM8K test -- reported only
  D_audit : post-hoc reward audit only

IF prompts are built from alpaca instructions (+1..3 verifiable constraints); held-out task pools come
from the same alpaca rows the earlier pilot reserved, so no task text crosses splits.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache

from .. import data as D
from ..constraints import render_prompt, sample_constraints
from ..paths import DATA

SPLIT_DIR = DATA / "rga"


def _if_items(tasks, tag, seed):
    g = random.Random(f"{seed}-{tag}")
    out = []
    for i, t in enumerate(tasks):
        cons = sample_constraints(g, g.choice([1, 2, 2, 3]))
        out.append({"id": f"{tag}-if-{i}", "task_type": "if", "prompt": render_prompt(t, cons),
                    "constraints": [c.to_dict() for c in cons]})
    return out


def _math_items(qs, tag):
    return [{"id": f"{tag}-m-{i}", "task_type": "math", "prompt": D.math_prompt(q["question"]),
             "answer": q["answer"]} for i, q in enumerate(qs)]


def build(seed: int = 2026, n_train_if: int = 1200, n_train_math: int = 600,
          n_fit=(120, 80), n_sel=(120, 80), n_audit=(100, 100)) -> dict:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if (SPLIT_DIR / "manifest.json").exists():
        return json.loads((SPLIT_DIR / "manifest.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    # held-out open-ended tasks reserved by rsi.data.prepare (never in alpaca_train)
    held = [x["task"] for x in D.load_split("if_dev")] + [x["task"] for x in D.load_split("if_test")]
    rng.shuffle(held)
    fit_t, sel_t = held[:n_fit[0]], held[n_fit[0]:n_fit[0] + n_sel[0]]
    alpaca = [r for r in D.load_split("alpaca_train") if len(r["response"].split()) >= 20]
    rng.shuffle(alpaca)
    audit_t = [r["prompt"] for r in alpaca[:n_audit[0]]]
    train_t = [r["prompt"] for r in alpaca[n_audit[0]:n_audit[0] + n_train_if]]
    gsm_tr = D.load_split("gsm_train")
    rng.shuffle(gsm_tr)
    spare = D.gsm_spare()  # gsm8k test items unused by gsm_dev/gsm_test
    splits = {
        "train": _if_items(train_t, "train", seed) + _math_items(gsm_tr[:n_train_math], "train"),
        "fit": _if_items(fit_t, "fit", seed) + _math_items(spare[:n_fit[1]], "fit"),
        "sel": _if_items(sel_t, "sel", seed) + _math_items(spare[n_fit[1]:n_fit[1] + n_sel[1]], "sel"),
        "audit": _if_items(audit_t, "audit", seed) + _math_items(gsm_tr[n_train_math:n_train_math + n_audit[1]], "audit"),
        # math test = the 300 gsm8k test items of gsm_dev + gsm_test (disjoint from spare)
        "test_math": _math_items(D.load_split("gsm_dev") + D.load_split("gsm_test"), "test"),
    }
    for k, v in splits.items():
        D.write_jsonl(SPLIT_DIR / f"{k}.jsonl", v)
    man = {k: len(v) for k, v in splits.items()} | {"seed": seed, "test_if": "google/IFEval (541)"}
    (SPLIT_DIR / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


@lru_cache(maxsize=None)
def split(name: str) -> tuple:
    return tuple(D.read_jsonl(SPLIT_DIR / f"{name}.jsonl"))


@lru_cache(maxsize=1)
def ifeval_docs() -> tuple:
    from datasets import load_dataset
    return tuple(dict(r) for r in load_dataset("google/IFEval", split="train"))
