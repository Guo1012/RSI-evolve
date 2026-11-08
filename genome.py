"""Reward genome g = (M, w, G, phi) and genetic operators (paper §3.3-3.4).

  R_g(x, y) = sum_{i in M} gate_i(x, y) * w_i * phi_i(m_i(x, y))                     (3-4)

Legality: >=1 correctness module, |M| <= L_MAX, w in [-W_MAX, W_MAX], gates reference enabled modules
and form a DAG.  Every operator output goes through the same deterministic `repair` + canonical
rounding, so identical rewards hash identically and evaluations are reused from the cache.
"""
from __future__ import annotations

import copy
import json
import random

import numpy as np

from .primitives import CATEGORY, CORRECTNESS, PRIMS

L_MAX = 6
W_MAX = 2.0
PHIS = ["id", "rank", "z", "bin"]
JUDGE_LIKE = {"judge", "selfscore"}


class Genome:
    def __init__(self, modules: dict | None = None, gates: list | None = None):
        self.modules = {k: dict(v) for k, v in (modules or {}).items()}   # prim -> {"w": float, "phi": str}
        self.gates = [tuple(g) for g in (gates or [])]                      # (target, source, threshold)

    # ------------------------------------------------------------ identity
    def to_dict(self):
        return {"modules": {k: self.modules[k] for k in sorted(self.modules)},
                "gates": sorted([list(g) for g in self.gates])}

    @staticmethod
    def from_dict(d):
        return Genome(d["modules"], [tuple(g) for g in d["gates"]])

    def key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def size(self) -> int:
        return len(self.modules)

    def describe(self) -> str:
        parts = []
        for k in sorted(self.modules):
            m = self.modules[k]
            gs = [f"[{s}>={t:.2f}]" for (tg, s, t) in self.gates if tg == k]
            parts.append(f"{m['w']:+.2f}*{m['phi']}({k}){''.join(gs)}")
        return " ".join(parts)

    def descriptor(self) -> tuple[int, int]:
        """MAP-Elites cell: (#modules, judge-like |w| share binned to 4)."""
        tot = sum(abs(m["w"]) for m in self.modules.values())
        jw = sum(abs(m["w"]) for k, m in self.modules.items() if k in JUDGE_LIKE)
        ratio = jw / tot if tot > 0 else 0.0
        return (self.size(), min(3, int(ratio * 4)))

    # ------------------------------------------------------------ reward
    def reward(self, P: dict[str, np.ndarray]) -> np.ndarray:
        """P[prim] has shape (n_prompts, K); returns R of the same shape."""
        R = np.zeros_like(next(iter(P.values())), dtype=np.float32)
        for k, m in self.modules.items():
            x = _phi(P[k], m["phi"])
            gate = np.ones_like(x)
            for (tg, s, t) in self.gates:
                if tg == k:
                    gate *= (P[s] >= t).astype(np.float32)
            R += gate * m["w"] * x
        return R


def _phi(x, phi):
    if phi == "id":
        return x
    if phi == "rank":
        K = x.shape[1]
        return x.argsort(1).argsort(1).astype(np.float32) / max(1, K - 1)
    if phi == "z":
        z = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        return np.clip(z, -2, 2) / 4 + 0.5
    if phi == "bin":
        return (x >= 0.5).astype(np.float32)
    raise KeyError(phi)


# ---------------------------------------------------------------- repair / canonical form
def repair(g: Genome, rng: random.Random | None = None) -> Genome:
    g = copy.deepcopy(g)
    for m in g.modules.values():
        m["w"] = round(max(-W_MAX, min(W_MAX, m["w"])) / 0.05) * 0.05
        m["w"] = round(m["w"], 2)
        if m["phi"] not in PHIS:
            m["phi"] = "id"
    g.modules = {k: m for k, m in g.modules.items() if abs(m["w"]) > 1e-9 or k in CORRECTNESS}
    if not (set(g.modules) & CORRECTNESS):
        g.modules["correct"] = {"w": 1.0, "phi": "id"}
    while len(g.modules) > L_MAX:  # drop the weakest non-correctness module (deterministic)
        drop = min((k for k in g.modules if k not in CORRECTNESS), key=lambda k: (abs(g.modules[k]["w"]), k))
        del g.modules[drop]
    gates = []
    for (tg, s, t) in sorted(g.gates):
        if tg in g.modules and s in g.modules and tg != s:
            t = round(max(0.05, min(0.95, t)) / 0.05) * 0.05
            gates.append((tg, s, round(t, 2)))
    # at most one gate per (target, source); keep a DAG by dropping edges that close a cycle
    seen, dag = set(), []
    for e in gates:
        if (e[0], e[1]) in seen:
            continue
        if not _reaches(dag, e[0], e[1]):  # adding source->target must not create target ~> source
            dag.append(e)
            seen.add((e[0], e[1]))
    g.gates = dag
    return g


def _reaches(edges, start, goal):
    """Is there a path start -> goal following source->target edges (edge = (target, source, t))?"""
    adj = {}
    for tg, s, _ in edges:
        adj.setdefault(s, []).append(tg)
    stack, seen = [start], set()
    while stack:
        u = stack.pop()
        if u == goal:
            return True
        if u in seen:
            continue
        seen.add(u)
        stack += adj.get(u, [])
    return False


# ---------------------------------------------------------------- initialisation
def manual_seeds() -> list[Genome]:
    """Fixed human seeds (paper §3.4.1)."""
    return [repair(Genome({"correct": {"w": 1.0, "phi": "id"}})),
            repair(Genome({"correct": {"w": 1.0, "phi": "id"}, "format": {"w": 0.5, "phi": "id"}})),
            repair(Genome({"correct": {"w": 1.0, "phi": "id"}, "length": {"w": 0.5, "phi": "id"}},
                          [("length", "correct", 0.5)]))]


def random_genome(rng: random.Random) -> Genome:
    n = rng.randint(1, 4)
    mods = {rng.choice(sorted(CORRECTNESS)): {"w": rng.uniform(0.2, W_MAX), "phi": rng.choice(PHIS)}}
    while len(mods) < n:
        k = rng.choice(PRIMS)
        if k not in mods:
            mods[k] = {"w": rng.uniform(-1.0, W_MAX), "phi": rng.choice(PHIS)}
    gates = []
    if len(mods) > 1 and rng.random() < 0.4:
        tg = rng.choice(sorted(mods))
        s = rng.choice(sorted(k for k in mods if k != tg))
        gates.append((tg, s, rng.uniform(0.3, 0.9)))
    return repair(Genome(mods, gates))


# ---------------------------------------------------------------- variation
def mutate(g: Genome, rng: random.Random, sigma_w=0.3, sigma_t=0.1, p_struct=0.3, structural=True) -> Genome:
    c = copy.deepcopy(g)
    for m in c.modules.values():                       # parameter mutation: truncated Gaussian (via clipping)
        m["w"] += rng.gauss(0, sigma_w)
    c.gates = [(tg, s, t + rng.gauss(0, sigma_t)) for (tg, s, t) in c.gates]
    if structural and rng.random() < p_struct:
        op = rng.choice(["add", "delete", "replace", "gate_add", "gate_del", "phi"])
        free = [k for k in PRIMS if k not in c.modules]
        if op == "add" and free and len(c.modules) < L_MAX:
            c.modules[rng.choice(free)] = {"w": rng.uniform(-1.0, W_MAX), "phi": rng.choice(PHIS)}
        elif op == "delete" and len(c.modules) > 1:
            del c.modules[rng.choice(sorted(c.modules))]
        elif op == "replace" and free:
            old = rng.choice(sorted(c.modules))
            c.modules[rng.choice(free)] = c.modules.pop(old)
            c.gates = [(tg, s, t) for (tg, s, t) in c.gates if old not in (tg, s)]
        elif op == "gate_add" and len(c.modules) > 1:
            tg = rng.choice(sorted(c.modules))
            s = rng.choice(sorted(k for k in c.modules if k != tg))
            c.gates.append((tg, s, rng.uniform(0.3, 0.9)))
        elif op == "gate_del" and c.gates:
            c.gates.pop(rng.randrange(len(c.gates)))
        elif op == "phi":
            k = rng.choice(sorted(c.modules))
            c.modules[k]["phi"] = rng.choice(PHIS)
    return repair(c)


def crossover(a: Genome, b: Genome, rng: random.Random, p_one_side=0.5, p_convex=0.5) -> Genome:
    """Module-level crossover: align by primitive id; gates travel with their target module."""
    mods, src = {}, {}
    for k in sorted(set(a.modules) | set(b.modules)):
        if k in a.modules and k in b.modules:
            pa = rng.random() < 0.5
            m = dict((a if pa else b).modules[k])
            if rng.random() < p_convex:
                lam = rng.random()
                m["w"] = lam * a.modules[k]["w"] + (1 - lam) * b.modules[k]["w"]
            mods[k], src[k] = m, (a if pa else b)
        elif rng.random() < p_one_side:
            par = a if k in a.modules else b
            mods[k], src[k] = dict(par.modules[k]), par
    gates = [gt for k, par in src.items() for gt in par.gates if gt[0] == k]
    return repair(Genome(mods, gates))


def distance(a: Genome, b: Genome) -> float:
    sa, sb = set(a.modules), set(b.modules)
    jac = 1 - len(sa & sb) / max(1, len(sa | sb))
    common = sa & sb
    wd = np.mean([abs(a.modules[k]["w"] - b.modules[k]["w"]) / (2 * W_MAX) for k in common]) if common else 1.0
    return 0.7 * jac + 0.3 * float(wd)
