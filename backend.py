"""Model backend shared by every candidate of a run.

One base model stays resident on the GPU.  Each candidate attaches a fresh LoRA adapter, trains it
for T rounds, is evaluated, and the adapter is removed again (``release``), so no candidate inherits
another candidate's parameters (paper §3.1: every candidate restarts from pi_0).

Supports Qwen3 base (ChatML prompt, <|endoftext|> terminator -- see rsi/data.py for why) and
Gemma 4 instruction models (native chat template, <end_of_turn> terminator).
"""
from __future__ import annotations

import gc
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.92)

LORA_TARGETS = r".*(language_model|model\.layers).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


class Backend:
    def __init__(self, path: str, lora_r: int = 16, log=print):
        self.path = str(path)
        self.log = log
        cfg = AutoConfig.from_pretrained(self.path)
        self.kind = "gemma" if "gemma" in cfg.model_type else "qwen"
        tc = getattr(cfg, "text_config", cfg)
        self.softcap = getattr(tc, "final_logit_softcapping", None)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.kind == "qwen":
            self.end_text = "<|endoftext|>"
            stops = {self.tok.convert_tokens_to_ids("<|endoftext|>"), self.tok.convert_tokens_to_ids("<|im_end|>")}
        else:
            # Gemma 4 turns are <|turn>role ... <turn|>; take the terminator ids from the model's own
            # generation config instead of assuming Gemma-3's <end_of_turn> (which maps to <unk> here)
            gen_eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
            end_id = next(i for i in gen_eos if i != self.tok.eos_token_id)
            self.end_text = self.tok.convert_ids_to_tokens(end_id)
            stops = set(gen_eos) | {self.tok.eos_token_id}
        self.stops = sorted(i for i in stops if i is not None and i >= 0)
        attn = "sdpa"
        if self.kind == "gemma":  # text decoder on SDPA; the (unused) audio/vision towers lack SDPA support
            attn = {"text_config": "sdpa", "audio_config": "eager", "vision_config": "eager", "": "sdpa"}
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16, attn_implementation=attn,
                                                          device_map={"": 0})
        self.model.config.use_cache = True
        self.lora_r = lora_r
        self.peft = None
        self._prefix_cache: dict[str, list[int]] = {}

    # ---------------------------------------------------------------- formatting
    def prefix(self, prompt: str) -> str:
        if self.kind == "qwen":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)

    def prefix_ids(self, prompt: str) -> list[int]:
        ids = self._prefix_cache.get(prompt)
        if ids is None:
            ids = self.tok(self.prefix(prompt), add_special_tokens=False)["input_ids"]
            self._prefix_cache[prompt] = ids
        return ids

    # ---------------------------------------------------------------- adapters
    def attach(self):
        from peft import LoraConfig, get_peft_model
        assert self.peft is None
        cfg = LoraConfig(r=self.lora_r, lora_alpha=2 * self.lora_r, lora_dropout=0.05, target_modules=LORA_TARGETS,
                         bias="none", task_type="CAUSAL_LM")
        self.peft = get_peft_model(self.model, cfg)
        for p in self.peft.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def lora_state(self) -> dict:
        from peft import get_peft_model_state_dict
        return {k: v.detach().to("cpu", torch.bfloat16) for k, v in get_peft_model_state_dict(self.peft).items()}

    def load_lora_state(self, sd: dict):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.peft, {k: v.float() for k, v in sd.items()})

    def release(self):
        if self.peft is not None:
            self.model = self.peft.unload()
            self.peft = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def active(self):
        return self.peft if self.peft is not None else self.model

    # ---------------------------------------------------------------- generation
    @torch.inference_mode()
    def generate(self, prompts: list[str], n: int = 1, temperature: float = 0.0, max_new: int = 320,
                 seed: int = 0, token_budget: int = 64_000, use_adapter: bool = True) -> list[list[dict]]:
        m = self.active
        m.eval()
        ctx = self.peft.disable_adapter() if (self.peft is not None and not use_adapter) else _Null()
        enc = [self.prefix_ids(p) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: -len(enc[i]))
        out: list[list[dict]] = [[] for _ in prompts]
        torch.manual_seed(seed)
        pad = self.tok.pad_token_id
        i = 0
        with ctx:
            while i < len(order):
                longest = len(enc[order[i]]) + max_new
                bs = max(1, min(192, token_budget // (longest * n)))
                idx = order[i:i + bs]
                i += bs
                L = max(len(enc[j]) for j in idx)
                ids = torch.full((len(idx), L), pad, dtype=torch.long)
                att = torch.zeros((len(idx), L), dtype=torch.long)
                for b, j in enumerate(idx):  # left padding
                    ids[b, L - len(enc[j]):] = torch.tensor(enc[j])
                    att[b, L - len(enc[j]):] = 1
                sample = temperature > 0
                g = m.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=max_new,
                               do_sample=sample, temperature=temperature if sample else None,
                               top_p=0.95 if sample else None, top_k=None, num_return_sequences=n,
                               eos_token_id=self.stops, pad_token_id=pad)
                new = g[:, L:].tolist()
                for b, seq in enumerate(new):
                    j = idx[b // n]
                    cut = next((k for k, s in enumerate(seq) if s in self.stops), None)
                    text = self.tok.decode(seq[:cut if cut is not None else len(seq)], skip_special_tokens=True)
                    out[j].append({"text": text.strip(), "finished": cut is not None,
                                   "tokens": cut if cut is not None else len(seq)})
        return out

    # ---------------------------------------------------------------- scoring
    def _hidden(self, ids, att):
        m = self.active
        base = m.get_base_model() if self.peft is not None else m
        return base, base.model(input_ids=ids, attention_mask=att).last_hidden_state

    def _logits(self, base, h):
        z = base.lm_head(h).float()
        if self.softcap:
            z = torch.tanh(z / self.softcap) * self.softcap
        return z

    @torch.inference_mode()
    def mean_logprob(self, prompts: list[str], responses: list[str], use_adapter: bool = True,
                     token_budget: int = 24_000) -> list[float]:
        """Mean per-token log-probability of each response (judge = frozen pi_0, self-score = current policy)."""
        ctx = self.peft.disable_adapter() if (self.peft is not None and not use_adapter) else _Null()
        feats = []
        for p, r in zip(prompts, responses):
            a = self.prefix_ids(p)
            b = self.tok(r + self.end_text, add_special_tokens=False)["input_ids"][:400]
            feats.append((a, b))
        order = sorted(range(len(feats)), key=lambda i: -(len(feats[i][0]) + len(feats[i][1])))
        res = [0.0] * len(feats)
        pad = self.tok.pad_token_id
        i = 0
        self.active.eval()
        with ctx:
            while i < len(order):
                Lmax = len(feats[order[i]][0]) + len(feats[order[i]][1])
                bs = max(1, token_budget // Lmax)
                idx = order[i:i + bs]
                i += bs
                ids = torch.full((len(idx), Lmax), pad, dtype=torch.long)
                att = torch.zeros_like(ids)
                lab = torch.full_like(ids, -100)
                for k, j in enumerate(idx):
                    a, b = feats[j]
                    x = a + b
                    ids[k, :len(x)] = torch.tensor(x)
                    att[k, :len(x)] = 1
                    lab[k, len(a):len(x)] = torch.tensor(b)
                ids, att, lab = ids.cuda(), att.cuda(), lab.cuda()
                base, h = self._hidden(ids, att)
                h, t = h[:, :-1], lab[:, 1:]
                msk = t != -100
                lp = torch.zeros(t.shape, device=t.device)
                hs, ts = h[msk], t[msk]
                vals = []
                for c in range(0, hs.shape[0], 2048):
                    z = self._logits(base, hs[c:c + 2048])
                    vals.append(torch.log_softmax(z, -1).gather(1, ts[c:c + 2048, None]).squeeze(1))
                lp[msk] = torch.cat(vals)
                means = (lp.sum(1) / msk.sum(1).clamp(min=1)).tolist()
                for k, j in enumerate(idx):
                    res[j] = means[k]
        return res

    # ---------------------------------------------------------------- training
    def train(self, rows: list[dict], epochs: float = 2, lr: float = 2e-4, token_budget: int = 8192,
              seed: int = 0, max_len: int = 768) -> dict:
        """Continue training the attached adapter on (prompt, response) rows; prompt tokens are masked."""
        assert self.peft is not None
        rng = random.Random(seed)
        torch.manual_seed(seed)
        feats = []
        for r in rows:
            a = self.prefix_ids(r["prompt"])
            b = self.tok(r["response"] + self.end_text, add_special_tokens=False)["input_ids"]
            ids = (a + b)[:max_len]
            lab = ([-100] * len(a) + b)[:max_len]
            if sum(x != -100 for x in lab) >= 2:
                feats.append((ids, lab))
        if not feats:
            return {"examples": 0, "steps": 0, "seconds": 0.0, "loss": None}
        m = self.peft
        m.train()
        base = m.get_base_model()
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        base.config.use_cache = False
        params = [p for p in m.parameters() if p.requires_grad]
        plan = []
        for e in range(math.ceil(epochs)):
            order = sorted(range(len(feats)), key=lambda i: len(feats[i][0]) + rng.random())
            batches, cur, cm = [], [], 0
            for j in order:
                L = len(feats[j][0])
                if cur and max(cm, L) * (len(cur) + 1) > token_budget:
                    batches.append(cur)
                    cur, cm = [], 0
                cur.append(j)
                cm = max(cm, L)
            if cur:
                batches.append(cur)
            rng.shuffle(batches)
            if e + 1 > epochs:
                batches = batches[:int(len(batches) * (epochs - e))]
            plan += batches
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        sched = get_cosine_schedule_with_warmup(opt, max(1, len(plan) // 20), len(plan))
        pad = self.tok.pad_token_id
        t0, losses = time.time(), []
        for idx in plan:
            L = max(len(feats[j][0]) for j in idx)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            lab = torch.full_like(ids, -100)
            att = torch.zeros_like(ids)
            for k, j in enumerate(idx):
                x, y = feats[j]
                ids[k, :len(x)] = torch.tensor(x)
                lab[k, :len(y)] = torch.tensor(y)
                att[k, :len(x)] = 1
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = base.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
                t = lab[:, 1:]
                msk = t != -100
                hs, ts = h[msk], t[msk]
                total = 0.0
                for c in range(0, hs.shape[0], 2048):
                    total = total + checkpoint(self._ce_sum, base, hs[c:c + 2048], ts[c:c + 2048], use_reentrant=False)
                loss = total / max(1, hs.shape[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        base.gradient_checkpointing_disable()
        base.config.use_cache = True
        del opt
        torch.cuda.empty_cache()
        return {"examples": len(feats), "steps": len(plan), "seconds": round(time.time() - t0, 1),
                "loss": sum(losses[-5:]) / len(losses[-5:])}

    def _ce_sum(self, base, h, t):
        return torch.nn.functional.cross_entropy(self._logits(base, h), t, reduction="sum")


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
