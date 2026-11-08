"""Run the full protocol for one base model (resumable: every evaluation is cached under runs/rga/<name>).

    python -m rsi.rga.run --model qwen            # Qwen3-1.7B-Base
    python -m rsi.rga.run --model gemma           # Gemma 4 E2B-it
    python -m rsi.rga.run --model qwen --smoke    # tiny end-to-end check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..paths import ROOT, RUNS
from . import data as RD
from . import final as FN
from .backend import Backend
from .genome import Genome, manual_seeds
from .inner import Evaluator, Ledger
from .search import Search, manual_b1

MODELS = {"qwen": "models/Qwen3-1.7B-Base", "gemma": "models/gemma-4-E2B-it"}

CONFIG = {
    # inner loop (Eq. 3-1, 3-2)
    "Q": 384, "K": 4, "T": 2, "Ts": 1, "rho": 0.5, "keep_frac": 0.5, "epochs": 2, "lr": 2e-4,
    "temperature": 0.8, "max_new": 320, "max_new_test": 640,
    "lam_h": 0.2, "lam_c": 0.01, "beta": 0.5,
    # outer loop (Algorithm 3-1, Table 4-1, minimal-viable scale)
    "N": 6, "E": 2, "G": 4, "k_tour": 3, "pc": 0.6, "p_struct": 0.3, "eta": 0.5,
    "seeds_high": [11, 12], "seeds_sel": [21, 22], "K_sel": 2, "run_seed": 1,
}
CONFIG["B_search"] = (CONFIG["N"] * len(CONFIG["seeds_high"]) * CONFIG["T"]
                      + CONFIG["G"] * ((CONFIG["N"] - CONFIG["E"]) * CONFIG["rho"] * CONFIG["Ts"]
                                       + 2 * len(CONFIG["seeds_high"]) * CONFIG["T"]))  # = 24 + 4*(2+8) = 64

SMOKE = {"Q": 16, "K": 2, "N": 3, "E": 1, "G": 1, "K_sel": 1, "max_new": 96, "max_new_test": 96,
         "seeds_high": [11], "seeds_sel": [21]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--methods", default="B0,B1,GA,GA-noX,B2")
    a = ap.parse_args()
    cfg = dict(CONFIG)
    name = a.model
    if a.smoke:
        cfg.update(SMOKE)
        cfg["B_search"] = 1e9
        name += "_smoke"
    run_dir = RUNS / "rga" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== RGA run {name}  cfg={json.dumps(cfg)}")
    RD.build()
    be = Backend(str((ROOT / MODELS[a.model]).resolve()), log=log)
    if a.smoke:
        _shrink()
    ev = Evaluator(be, cfg, run_dir, log=log)
    ledger = Ledger(run_dir / "ledger.jsonl")
    S = Search(ev, cfg, ledger, log=log)
    methods = a.methods.split(",")
    out = {}
    for m in methods:
        t0 = time.time()
        if m == "B0":
            r = S.run_fixed("B0", [manual_seeds()[0]])
        elif m == "B1":
            r = S.run_fixed("B1", manual_b1())
        elif m == "GA":
            r = S.run("GA", pc=cfg["pc"])
        elif m == "GA-noX":
            r = S.run("GA-noX", pc=0.0)
        elif m == "B2":
            r = S.run("B2", pc=0.0, random_proposals=True)
        else:
            raise KeyError(m)
        r["finalists"] = [g.to_dict() for g in r["finalists"]]
        r["search_rounds"] = ledger.used(m)
        out[m] = r
        (run_dir / "methods").mkdir(exist_ok=True)
        (run_dir / "methods" / f"{m}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"### {m} search done: {ledger.used(m):.1f} rounds, {time.time() - t0:.0f}s")
    # ---- all searches frozen -> final selection on D_sel
    sel = {m: FN.select(ev, cfg, ledger, m, [Genome.from_dict(d) for d in out[m]["finalists"]], log=log)
           for m in methods}
    # ---- sealed test + audit (after every selection is frozen)
    tests = {"pi0": FN.test_model(ev, cfg, None, "pi0")}
    audits = {}
    for m in methods:
        ch = sel[m]["chosen"]
        tests[m] = [FN.test_model(ev, cfg, ad, f"{m}_s{s}") for ad, s in zip(ch["adapters"], cfg["seeds_sel"])]
        audits[m] = FN.audit(ev, cfg, Genome.from_dict(ch["genome"]), ch["adapters"][0], m)
        log(f"[{m}/test] " + " | ".join(f"IFEval={t['ifeval_prompt_strict']:.3f} GSM8K={t['gsm8k']:.3f}" for t in tests[m])
            + f" | audit Gap={audits[m]['Gap']:+.3f} AFR={audits[m]['AFR']}")
    summary = {"model": a.model, "config": cfg, "methods": {}, "pi0_test": tests["pi0"],
               "ledger_rows": len(ledger.rows)}
    for m in methods:
        summary["methods"][m] = {
            "search_rounds": ledger.used(m), "select_rounds": ledger.used(m, ("select",)),
            "gpu_hours": sum(r["seconds"] for r in ledger.rows if r["method"] == m) / 3600,
            "trajectory": out[m]["trajectory"], "chosen": sel[m]["chosen"], "selection": sel[m]["candidates"],
            "test": tests[m], "audit": audits[m]}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=== done")


def _shrink():
    """--smoke: tiny evaluation sets (6 IF + 6 math items per split, 8 IFEval prompts)."""
    full_split, full_docs = RD.split, RD.ifeval_docs

    def small(name):
        rows = full_split(name)
        if name == "train":
            return rows
        return tuple([r for r in rows if r["task_type"] == "if"][:6] + [r for r in rows if r["task_type"] == "math"][:6])

    RD.split = small
    RD.ifeval_docs = lambda: full_docs()[:8]


if __name__ == "__main__":
    main()
