"""Final selection on D_sel (new seeds S_sel), sealed test (IFEval + GSM8K test), post-hoc audit on D_audit.

Order is fixed: every method's search is frozen before D_sel is read; D_test / D_audit results are
computed after all selections and never fed back.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..constraints import Constraint
from ..data import answers_match, extract_answer
from ..ifeval_official import score as ifeval_score
from . import data as RD
from . import primitives as PR
from .genome import Genome
from .inner import Evaluator, Ledger, _h


def select(ev: Evaluator, cfg, ledger: Ledger, method: str, finalists: list[Genome], log=print) -> dict:
    out_dir = ev.dir / "final" / method
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "selection.json"
    if f.exists():
        res = json.loads(f.read_text(encoding="utf-8"))
        for row in res["candidates"]:  # re-charge the (cached) selection work for this invocation's ledger
            ledger.add(method, "select", len(cfg["seeds_sel"]) * cfg["T"], row.get("seconds", 0.0), row["desc"])
        return res
    rows = []
    for g in finalists:
        import time
        t0 = time.time()
        per = []
        for s in cfg["seeds_sel"]:
            ad = out_dir / f"{_h(g.key())}_s{s}.pt"
            per.append(ev.run_seed(g, s, "high", RD.split("sel"), keep_adapter=ad))
        sec = time.time() - t0
        rows.append({"genome": g.to_dict(), "desc": g.describe(), "size": g.size(),
                     "J_sel": float(np.mean([p["J"] for p in per])),
                     "if_sel": float(np.mean([p["if_acc"] for p in per])),
                     "math_sel": float(np.mean([p["math_acc"] for p in per])),
                     "J_sel_seeds": [p["J"] for p in per], "seconds": round(sec, 1),
                     "adapters": [str(out_dir / f"{_h(g.key())}_s{s}.pt") for s in cfg["seeds_sel"]]})
        ledger.add(method, "select", len(cfg["seeds_sel"]) * cfg["T"], sec, g.describe())
        log(f"[{method}/select] J_sel={rows[-1]['J_sel']:.4f} if={rows[-1]['if_sel']:.3f} math={rows[-1]['math_sel']:.3f} | {g.describe()}")
    win = max(rows, key=lambda r: (r["J_sel"], -r["size"]))
    res = {"method": method, "candidates": rows, "chosen": win}
    f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------------- sealed test
def test_model(ev: Evaluator, cfg, adapter: str | None, tag: str) -> dict:
    f = ev.dir / "test" / f"{tag}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    f.parent.mkdir(parents=True, exist_ok=True)
    be = ev.be
    if adapter:
        be.attach()
        be.load_lora_state(torch.load(adapter))
    try:
        docs = list(RD.ifeval_docs())
        outs = be.generate([d["prompt"] for d in docs], max_new=cfg["max_new_test"], seed=0)
        resp = [o[0]["text"] for o in outs]
        m_if, rows = ifeval_score(docs, resp)
        maths = list(RD.split("test_math"))
        mo = be.generate([it["prompt"] for it in maths], max_new=cfg["max_new"], seed=0)
        ok = [answers_match(extract_answer(o[0]["text"]), it["answer"]) for it, o in zip(maths, mo)]
    finally:
        be.release()
    res = {"tag": tag, "ifeval_prompt_strict": m_if["prompt_level_strict_acc"],
           "ifeval_inst_strict": m_if["inst_level_strict_acc"], "ifeval_prompt_loose": m_if["prompt_level_loose_acc"],
           "gsm8k": float(np.mean(ok)), "ifeval_finish": float(np.mean([o[0]["finished"] for o in outs])),
           "prompt_ok": [bool(r["prompt_level_strict_acc"]) for r in rows], "gsm_ok": [bool(x) for x in ok]}
    f.write_text(json.dumps(res), encoding="utf-8")
    return res


# ---------------------------------------------------------------------- audit
def _ngr(t, n):
    w = t.split()
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def _bleu(hyp, refs):
    if not hyp.split():
        return 0.0
    ps = []
    for n in range(1, 5):
        h = Counter(_ngr(hyp, n))
        if not h:
            ps.append(1e-9)
            continue
        mx = Counter()
        for r in refs:
            for k, v in Counter(_ngr(r, n)).items():
                mx[k] = max(mx[k], v)
        ps.append((sum(min(v, mx[k]) for k, v in h.items()) + 1) / (sum(h.values()) + 1))
    hl = len(hyp.split())
    rl = min((abs(len(r.split()) - hl), len(r.split())) for r in refs)[1]
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(1, hl))
    return bp * math.exp(sum(math.log(p) for p in ps) / 4)


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def audit_samples(ev: Evaluator, cfg, adapter: str | None, tag: str):
    f = ev.dir / "audit" / f"samples_{tag}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return d["groups"], {k: np.array(v) for k, v in d["P"].items()}, np.array(d["J"])
    f.parent.mkdir(parents=True, exist_ok=True)
    be = ev.be
    items = list(RD.split("audit"))
    if adapter:
        be.attach()
        be.load_lora_state(torch.load(adapter))
    try:
        prompts = [it["prompt"] for it in items]
        groups = be.generate(prompts, n=cfg["K"], temperature=cfg["temperature"], max_new=cfg["max_new"], seed=777)
        fp = [p for p in prompts for _ in range(cfg["K"])]
        fr = [s["text"] for g in groups for s in g]
        judge = be.mean_logprob(fp, fr, use_adapter=False)
        selfs = be.mean_logprob(fp, fr, use_adapter=True) if adapter else judge
    finally:
        be.release()
    P = PR.compute(items, groups, judge, selfs)
    J = np.zeros_like(P["correct"])
    for i, (it, g) in enumerate(zip(items, groups)):
        for k, s in enumerate(g):  # independent task checker (not a reward primitive call)
            if it["task_type"] == "if":
                J[i, k] = float(all(Constraint.from_dict(c).check(s["text"]) for c in it["constraints"]))
            else:
                J[i, k] = float(answers_match(extract_answer(s["text"]), it["answer"]))
    f.write_text(json.dumps({"groups": groups, "P": {k: v.tolist() for k, v in P.items()}, "J": J.tolist()}),
                 encoding="utf-8")
    return groups, P, J


def audit(ev: Evaluator, cfg, g: Genome, adapter: str, tag: str, q: float = 0.75) -> dict:
    groups0, P0, J0 = audit_samples(ev, cfg, None, "pi0")
    groups, P, J = audit_samples(ev, cfg, adapter, tag)
    R, R0 = g.reward(P), g.reward(P0)
    thr = np.quantile(R, q)
    hi = R >= thr
    afr = float(1 - J[hi].mean()) if hi.any() else None
    rhos = [r for r in (_spearman(R[i], J[i]) for i in range(len(R))) if r is not None]
    texts = [[s["text"] for s in gr] for gr in groups]
    toks = [w for gr in texts for t in gr for w in _ngr(t, 2)]
    sb = np.mean([np.mean([_bleu(gr[k], gr[:k] + gr[k + 1:]) for k in range(len(gr))]) for gr in texts])
    return {"tag": tag, "Gap": float((R.mean() - R0.mean()) - (J.mean() - J0.mean())),
            "dR": float(R.mean() - R0.mean()), "dJ": float(J.mean() - J0.mean()),
            "AFR": afr, "AFR_pi0_same_reward": float(1 - J0[R0 >= np.quantile(R0, q)].mean()),
            "spearman": float(np.mean(rhos)) if rhos else None, "n_spearman": len(rhos),
            "distinct2": len(set(toks)) / max(1, len(toks)), "self_bleu": float(sb),
            "pass1": float(J.mean()), "passk": float(J.max(1).mean()),
            "unfinished": float(1 - np.mean([[s["finished"] for s in gr] for gr in groups]))}
