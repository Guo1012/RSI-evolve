"""Inner loop: recursive self-training under a candidate reward, and its evaluation (paper §3.1-3.2).

pi_t = U(pi_{t-1}, D_t),  D_t = rejection-sampled by R_g from K samples per D_train prompt     (3-1)
F_s(g) = J(pi_T; D_fit) - lam_h * H(pi_T; D_fit) - lam_c * |g|                                (3-2)
F_hat(g) = mean_s F_s - beta * std_s F_s

Exact caches (never change results, only save GPU time):
  * round-1 samples come from pi_0, identical for every candidate with the same seed -> generated once;
  * the adapter after round t depends only on the selected training data -> keyed by a hash chain of
    the selected (prompt, sample) ids; identical selections reuse the trained adapter and the
    downstream samples.  (Different reward expressions often select identical data -- paper §4.1.2.)
Budget is charged per genome-level evaluation in standardised rounds (Eq. 4-1), cache or not.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..constraints import Constraint
from ..data import answers_match, extract_answer, read_jsonl, write_jsonl
from . import data as RD
from . import primitives as PR
from .genome import Genome


class Ledger:
    """Standardised inner-training work C_inner (Eq. 4-1) + wall-clock per phase."""

    def __init__(self, path: Path):
        # rebuilt on every invocation: a resumed run replays its (deterministic) decisions from the
        # result cache, so the ledger is regenerated rather than appended twice
        self.path = Path(path)
        self.path.write_text("", encoding="utf-8")
        self.rows = []

    def add(self, method: str, phase: str, rounds: float, seconds: float, note: str = ""):
        r = {"method": method, "phase": phase, "rounds": rounds, "seconds": round(seconds, 1), "note": note,
             "t": time.strftime("%H:%M:%S")}
        self.rows.append(r)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def used(self, method: str, phases=("init", "search")) -> float:
        return sum(r["rounds"] for r in self.rows if r["method"] == method and r["phase"] in phases)


class Evaluator:
    def __init__(self, be, cfg: dict, run_dir: Path, log=print):
        self.be, self.cfg, self.log = be, cfg, log
        self.dir = Path(run_dir)
        (self.dir / "cache").mkdir(parents=True, exist_ok=True)
        self.train = RD.split("train")
        self.results_path = self.dir / "results.jsonl"
        self.results = {(r["key"], r["fid"]): r for r in read_jsonl(self.results_path)}

    # ------------------------------------------------------------ prompts
    def prompts(self, seed: int, t: int, fid: str) -> list[dict]:
        c = self.cfg
        rng = random.Random(f"prompts-{seed}-{t}")
        items = rng.sample(list(self.train), c["Q"])
        return items[:int(c["Q"] * c["rho"])] if fid == "low" else items

    # ------------------------------------------------------------ sampling + primitives (cached)
    def _sample(self, items, seed, t, chain_key):
        f = self.dir / "cache" / f"samples_{_h(chain_key)}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            return d["groups"], {k: np.array(v, dtype=np.float32) for k, v in d["P"].items()}
        c, be = self.cfg, self.be
        prompts = [it["prompt"] for it in items]
        groups = be.generate(prompts, n=c["K"], temperature=c["temperature"], max_new=c["max_new"],
                             seed=int(_h(chain_key), 16) % (2 ** 31))
        flat_p = [p for p in prompts for _ in range(c["K"])]
        flat_r = [s["text"] for g in groups for s in g]
        judge = be.mean_logprob(flat_p, flat_r, use_adapter=False)
        selfs = be.mean_logprob(flat_p, flat_r, use_adapter=True) if be.peft is not None else judge
        P = PR.compute(items, groups, judge, selfs)
        f.write_text(json.dumps({"groups": groups, "P": {k: v.tolist() for k, v in P.items()}}), encoding="utf-8")
        return groups, P

    def round1(self, seed: int):
        """pi_0 samples for the full prompt list of `seed` (LF uses a prefix of the same list)."""
        if not hasattr(self, "_r1"):
            self._r1 = {}
        if seed not in self._r1:
            items = self.prompts(seed, 1, "high")
            self._r1[seed] = (items, *self._sample(items, seed, 1, f"{self.be.path}|pi0|{seed}|r1"))
        return self._r1[seed]

    def signature(self, g: Genome, seed: int) -> str:
        """Behavioural signature: which pi_0 samples g selects on the low-fidelity prompt subset.

        Rejection sampling only uses the *ranking* induced by R_g, so e.g. positive rescaling of a
        one-module reward is a no-op.  Two genomes with the same signature yield identical low-fidelity
        training data, hence identical low-fidelity scores; such children are skipped as duplicates.
        """
        items, groups, P = self.round1(seed)
        n = int(self.cfg["Q"] * self.cfg["rho"])
        rows, _ = self.select(g, items[:n], groups[:n], {k: v[:n] for k, v in P.items()}, seed, 1)
        return _h(json.dumps([(r["id"], r["k"]) for r in rows]))

    # ------------------------------------------------------------ selection (rejection sampling)
    def select(self, g: Genome, items, groups, P, seed, t):
        R = g.reward(P)
        noise = np.random.default_rng(seed * 1000 + t).random(R.shape) * 1e-6  # seeded tie-breaking
        Rn = R + noise
        best = Rn.argmax(1)
        bestR = Rn[np.arange(len(items)), best]
        n_keep = max(1, int(round(self.cfg["keep_frac"] * len(items))))
        keep = np.argsort(-bestR)[:n_keep]
        rows = [{"prompt": items[i]["prompt"], "response": groups[i][best[i]]["text"],
                 "id": items[i]["id"], "k": int(best[i])} for i in sorted(keep)]
        stats = {"kept": n_keep, "R_mean_kept": float(R[keep, best[keep]].mean()),
                 "correct_kept": float(P["correct"][keep, best[keep]].mean()),
                 "finished_kept": float(P["format"][keep, best[keep]].mean())}
        return rows, stats

    # ------------------------------------------------------------ one seed
    def run_seed(self, g: Genome, seed: int, fid: str, eval_items, keep_adapter: Path | None = None) -> dict:
        c, be = self.cfg, self.be
        T = c["T"] if fid == "high" else c["Ts"]
        items1, groups1, P1 = self.round1(seed)
        if fid == "low":
            n = int(c["Q"] * c["rho"])
            items1, groups1, P1 = items1[:n], groups1[:n], {k: v[:n] for k, v in P1.items()}
        chain = f"{be.path}|{seed}|{fid}"
        be.attach()
        trace = []
        try:
            for t in range(1, T + 1):
                if t == 1:
                    items, groups, P = items1, groups1, P1
                else:
                    items = self.prompts(seed, t, fid)
                    groups, P = self._sample(items, seed, t, chain + "|samples")
                rows, st = self.select(g, items, groups, P, seed, t)
                chain = chain + "|" + _h(json.dumps([(r["id"], r["k"]) for r in rows]))
                ad = self.dir / "cache" / f"adapter_{_h(chain)}.pt"
                if ad.exists():
                    be.load_lora_state(torch.load(ad))
                    st["train"] = "cache"
                else:
                    st["train"] = be.train(rows, epochs=c["epochs"], lr=c["lr"], seed=seed * 100 + t)
                    torch.save(be.lora_state(), ad)
                st["proxy_R_all"] = float(g.reward(P).mean())  # mean proxy reward of this round's samples
                trace.append(st)
            ev = self.eval_items(eval_items, tag=_h(chain))
            if keep_adapter:
                keep_adapter.parent.mkdir(parents=True, exist_ok=True)
                torch.save(be.lora_state(), keep_adapter)
        finally:
            be.release()
        return {"seed": seed, "trace": trace, **ev, "chain": _h(chain)}

    # ------------------------------------------------------------ task metrics J and risk H
    def eval_items(self, items, tag: str, use_cache=True) -> dict:
        f = self.dir / "cache" / f"eval_{tag}_{_h(json.dumps([i['id'] for i in items]))}.json"
        if use_cache and f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        c = self.cfg
        outs = self.be.generate([it["prompt"] for it in items], max_new=c["max_new"], seed=0)
        rows = []
        for it, o in zip(items, outs):
            s = o[0]
            if it["task_type"] == "if":
                ok = all(Constraint.from_dict(x).check(s["text"]) for x in it["constraints"])
            else:
                ok = answers_match(extract_answer(s["text"]), it["answer"])
            rows.append({"id": it["id"], "type": it["task_type"], "ok": bool(ok), "finished": s["finished"],
                         "long": s["tokens"] >= 0.9 * c["max_new"], "rep": PR._rep_rate(s["text"]),
                         "words": len(s["text"].split()), "text": s["text"]})
        r = metrics(rows)
        r["rows"] = rows
        f.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        return r

    # ------------------------------------------------------------ genome-level evaluation (Eq. 3-2)
    def evaluate(self, g: Genome, fid: str, seeds: list[int], method: str, ledger: Ledger, phase: str) -> dict:
        c = self.cfg
        key = g.key()
        T = c["T"] if fid == "high" else c["Ts"]
        rounds = len(seeds) * T * (1.0 if fid == "high" else c["rho"])
        cached = self.results.get((key, fid))
        t0 = time.time()
        if cached is None or cached["seeds"] != seeds:
            per = [self.run_seed(g, s, fid, RD.split("fit")) for s in seeds]
            gpu_s = time.time() - t0
            F = [p["J"] - c["lam_h"] * p["H"] - c["lam_c"] * g.size() for p in per]
            res = {"key": key, "fid": fid, "seeds": seeds, "genome": g.to_dict(), "desc": g.describe(),
                   "F_s": F, "F_hat": float(np.mean(F) - c["beta"] * np.std(F)),
                   "J": float(np.mean([p["J"] for p in per])), "H": float(np.mean([p["H"] for p in per])),
                   "if_acc": float(np.mean([p["if_acc"] for p in per])),
                   "math_acc": float(np.mean([p["math_acc"] for p in per])),
                   "trace": [p["trace"] for p in per], "first_by": method, "seconds": round(gpu_s, 1)}
            self.results[(key, fid)] = res
            with open(self.results_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        res = self.results[(key, fid)]
        # charged in full to every method that requests it (cross-method physical reuse is not a discount)
        ledger.add(method, phase, rounds, res.get("seconds", 0.0), f"{fid} {g.describe()}")
        if self.log:
            self.log(f"[{method}/{phase}] {fid} F={res['F_hat']:+.4f} J={res['J']:.3f} H={res['H']:.3f} "
                     f"if={res['if_acc']:.3f} math={res['math_acc']:.3f} | {g.describe()}")
        return res


def metrics(rows) -> dict:
    ifr = [r for r in rows if r["type"] == "if"]
    mr = [r for r in rows if r["type"] == "math"]
    if_acc = sum(r["ok"] for r in ifr) / max(1, len(ifr))
    math_acc = sum(r["ok"] for r in mr) / max(1, len(mr))
    H = (np.mean([r["long"] for r in rows]) + np.mean([min(1.0, 2 * r["rep"]) for r in rows])
         + np.mean([not r["finished"] for r in rows])) / 3
    return {"J": 0.5 * if_acc + 0.5 * math_acc, "if_acc": if_acc, "math_acc": math_acc, "H": float(H),
            "unfinished": float(np.mean([not r["finished"] for r in rows])),
            "mean_words": float(np.mean([r["words"] for r in rows]))}


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
