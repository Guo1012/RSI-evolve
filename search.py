"""Outer loop: budgeted genetic reward search (Algorithm 3-1), baselines and ablation.

Methods (paper §4.4, minimal-viable subset):
  B0     correctness-only reward
  B1     hand-designed composite reward, weights chosen on D_fit by a fixed 2x2 grid
  B2     random search in the same legal space (shared initial population, then history-free proposals)
  GA     selection + module-level crossover + parameter/structural mutation + fitness sharing + MAP-Elites
  GA-noX ablation: GA with crossover probability 0
All share the primitive library, inner training, data splits, fidelity levels, evaluation rules and the
same initial population; every method is charged the full initial-population cost.
"""
from __future__ import annotations

import math
import random

import numpy as np

from .genome import Genome, crossover, distance, manual_seeds, mutate, random_genome, repair
from .inner import Evaluator, Ledger


def initial_population(N: int, seed: int) -> list[Genome]:
    rng = random.Random(f"init-{seed}")
    pop, keys = [], set()
    for g in manual_seeds() + [random_genome(rng) for _ in range(20 * N)]:
        if g.key() not in keys:
            pop.append(g)
            keys.add(g.key())
        if len(pop) == N:
            break
    return pop


def manual_b1() -> list[Genome]:
    """B1: structure fixed from the task definition; 2x2 grid over (format weight, regulariser weight)."""
    out = []
    for wf in (0.25, 0.75):
        for wr in (0.25, 0.75):
            out.append(repair(Genome({"correct": {"w": 1.0, "phi": "id"}, "format": {"w": wf, "phi": "id"},
                                      "length": {"w": wr, "phi": "id"}, "repeat": {"w": wr, "phi": "id"}},
                                     [("length", "correct", 0.5)])))
    return out


def _rank_scores(vals):
    order = np.argsort(np.argsort(vals))
    return (order + 1) / len(vals)  # positive normalised ranks in (0, 1]


def survivor_selection(cands: list[tuple[Genome, dict]], N: int, E: int, sigma: float = 0.4):
    """E elites by raw F_hat; remaining slots greedily by rank-score / niche density (fitness sharing)."""
    cands = sorted(cands, key=lambda c: (-c[1]["F_hat"], c[0].size(), c[0].key()))
    chosen = cands[:E]
    rest = cands[E:]
    while len(chosen) < N and rest:
        allc = chosen + rest
        rs = _rank_scores([c[1]["F_hat"] for c in allc])
        scores = []
        for j, c in enumerate(rest):
            niche = 1.0 + sum(max(0.0, 1 - distance(c[0], o[0]) / sigma) for o in chosen)
            scores.append(rs[len(chosen) + j] / niche)
        b = int(np.argmax(scores))
        chosen.append(rest.pop(b))
    return chosen


class Search:
    def __init__(self, ev: Evaluator, cfg: dict, ledger: Ledger, log=print):
        self.ev, self.cfg, self.ledger, self.log = ev, cfg, ledger, log
        self.SH = cfg["seeds_high"]
        self.sL = [cfg["seeds_high"][0]]

    def hf(self, g, method, phase):
        return self.ev.evaluate(g, "high", self.SH, method, self.ledger, phase)

    def lf(self, g, method):
        return self.ev.evaluate(g, "low", self.sL, method, self.ledger, "search")

    def cost_hf(self):
        return len(self.SH) * self.cfg["T"]

    def cost_lf(self):
        return self.cfg["Ts"] * self.cfg["rho"]

    # ------------------------------------------------------------------ evolutionary / random search
    def run(self, method: str, pc: float, random_proposals: bool = False) -> dict:
        c = self.cfg
        N, E, B = c["N"], c["E"], c["B_search"]
        rng = random.Random(f"{method}-{c['run_seed']}")
        pop = [(g, self.hf(g, method, "init")) for g in initial_population(N, c["run_seed"])]
        archive: dict = {}
        history = {g.key() for g, _ in pop}
        sigs = {self.ev.signature(g, self.sL[0]) for g, _ in pop}
        n_dup = 0
        traj = []

        def upd_archive(items):
            for g, r in items:
                d = g.descriptor()
                if d not in archive or r["F_hat"] > archive[d][1]["F_hat"]:
                    archive[d] = (g, r)

        upd_archive(pop)
        best = max(pop, key=lambda x: x[1]["F_hat"])
        traj.append({"gen": 0, "used": self.ledger.used(method), "best_F": best[1]["F_hat"], "best": best[0].describe()})
        for gen in range(1, c["G"] + 1):
            # ---- propose up to N-E new, distinct children
            children, tries = [], 0
            while len(children) < N - E and tries < 10 * N:
                tries += 1
                if random_proposals:
                    ch = random_genome(rng)
                else:
                    p1 = self._tournament(pop, rng)
                    if rng.random() < pc:
                        ch = crossover(p1[0], self._tournament(pop, rng)[0], rng)
                        ch = mutate(ch, rng, p_struct=c["p_struct"])
                    else:
                        ch = mutate(p1[0], rng, p_struct=c["p_struct"])
                if ch.key() in history:
                    continue
                history.add(ch.key())
                sig = self.ev.signature(ch, self.sL[0])
                if sig in sigs:  # behaviourally identical to an evaluated reward: free, uncharged skip
                    n_dup += 1
                    continue
                sigs.add(sig)
                children.append(ch)
            if not children:
                self.log(f"[{method}] gen {gen}: no new candidates -> stop")
                break
            # ---- low-fidelity screening (ranking within this batch only)
            lf = []
            for ch in children:
                if self.ledger.used(method) + self.cost_lf() > B:
                    break
                lf.append((ch, self.lf(ch, method)))
            if not lf:
                break
            m = math.ceil(c["eta"] * len(lf))
            ranked = sorted(lf, key=lambda x: -x[1]["F_hat"])
            promoted = []
            for ch, _ in ranked[:m]:
                if self.ledger.used(method) + self.cost_hf() > B:
                    break
                promoted.append((ch, self.hf(ch, method, "search")))
            pop = survivor_selection(pop + promoted, N, E)
            upd_archive(promoted)
            best = max(archive.values(), key=lambda x: x[1]["F_hat"])
            traj.append({"gen": gen, "used": self.ledger.used(method), "best_F": best[1]["F_hat"],
                         "best": best[0].describe(), "n_lf": len(lf), "n_promoted": len(promoted),
                         "n_behavioural_duplicates": n_dup, "tries": tries,
                         "lineage_ops": "crossover+mutation" if pc > 0 and not random_proposals else
                         ("random" if random_proposals else "mutation")})
            self.log(f"[{method}] gen {gen}: used {self.ledger.used(method):.1f}/{B} rounds, best F={best[1]['F_hat']:+.4f} {best[0].describe()}")
            if self.ledger.used(method) + self.cost_lf() > B:
                break
        finals = sorted(archive.values(), key=lambda x: (-x[1]["F_hat"], x[0].size()))
        return {"method": method, "trajectory": traj, "finalists": [g for g, _ in finals[:c["K_sel"]]],
                "archive": [{"cell": list(g.descriptor()), "F_hat": r["F_hat"], "desc": g.describe()} for g, r in finals],
                "population": [{"F_hat": r["F_hat"], "desc": g.describe()} for g, r in pop]}

    def _tournament(self, pop, rng):
        k = min(self.cfg["k_tour"], len(pop))
        return max(rng.sample(pop, k), key=lambda x: x[1]["F_hat"])

    # ------------------------------------------------------------------ fixed / hand-designed baselines
    def run_fixed(self, method: str, genomes: list[Genome]) -> dict:
        res = [(g, self.hf(g, method, "search")) for g in genomes]
        res.sort(key=lambda x: (-x[1]["F_hat"], x[0].size()))
        return {"method": method, "trajectory": [{"gen": 0, "used": self.ledger.used(method),
                                                   "best_F": res[0][1]["F_hat"], "best": res[0][0].describe()}],
                "finalists": [g for g, _ in res[:self.cfg["K_sel"]]],
                "archive": [{"F_hat": r["F_hat"], "desc": g.describe()} for g, r in res]}
