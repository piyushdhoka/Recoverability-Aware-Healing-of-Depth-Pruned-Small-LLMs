"""Healing: build a token-budgeted training mixture and fine-tune a pruned model in one of three scopes.

Scopes
  all_lora : LoRA on every block (attention + MLP projections)
  cut_lora : LoRA only on the kept blocks that border a removed span
  last_k   : full fine-tuning of the last k blocks (+ final norm); the tied embedding/head stays frozen
"""
import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence

from .modeling import chat_ids, get_blocks


# --------------------------------------------------------------------------- data
def tokenize_pool(tok, examples: list[dict], seq_len: int) -> list[dict]:
    """examples: teacher outputs {"messages", "response"} -> [{"input_ids", "labels", "n_tokens"}]."""
    out = []
    for ex in examples:
        ids, labels = chat_ids(tok, ex["messages"], ex["response"], max_len=seq_len)
        if any(l != -100 for l in labels):
            out.append({"input_ids": ids, "labels": labels, "n_tokens": len(ids)})
    return out


def build_mixture(tokenized: dict, tokens_per_pool: dict, seed: int) -> list[dict]:
    """Draw examples from each pool until its token quota is met (sampling without replacement first,
    then cycling), and shuffle. The realised token count per pool is within one example of the quota."""
    rng = random.Random(seed)
    mix = []
    for pool, quota in tokens_per_pool.items():
        data = tokenized.get(pool, [])
        if quota <= 0:
            continue
        if not data:
            raise ValueError(f"pool '{pool}' has a token quota of {quota:.0f} but no usable examples "
                             f"(check teacher QC / seq_len)")
        order, used, k = list(range(len(data))), 0, 0
        rng.shuffle(order)
        while used < quota:
            ex = data[order[k % len(order)]]
            mix.append(ex)
            used += ex["n_tokens"]
            k += 1
            if k % len(order) == 0:
                rng.shuffle(order)
    rng.shuffle(mix)
    return mix


def mixture_tokens(mix: list[dict]) -> int:
    return sum(ex["n_tokens"] for ex in mix)


def _collate(batch, pad_id):
    ids = pad_sequence([torch.tensor(b["input_ids"]) for b in batch], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([torch.tensor(b["labels"]) for b in batch], batch_first=True, padding_value=-100)
    mask = pad_sequence([torch.ones(len(b["input_ids"]), dtype=torch.long) for b in batch], batch_first=True)
    return ids, labels, mask


# --------------------------------------------------------------------------- scopes
def apply_scope(model, scope: str, cfg: dict, cut_adjacent: list[int]):
    """Returns the model to train (possibly PEFT-wrapped) and a short description."""
    hcfg = cfg["healing"]
    for p in model.parameters():
        p.requires_grad_(False)
    if scope in ("all_lora", "cut_lora"):
        from peft import LoraConfig, get_peft_model

        layers = None if scope == "all_lora" else cut_adjacent
        if scope == "cut_lora" and not layers:
            raise ValueError("cut_lora needs at least one cut-adjacent block")
        lcfg = LoraConfig(r=hcfg["lora_r"], lora_alpha=hcfg["lora_alpha"], lora_dropout=hcfg["lora_dropout"],
                          target_modules=hcfg["lora_targets"], layers_to_transform=layers,
                          layers_pattern="layers" if layers is not None else None, task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        return model, hcfg["lr_lora"], f"{scope} layers={layers or 'all'}"
    if scope == "last_k":
        blocks, _ = get_blocks(model)
        k = hcfg["last_k"]
        for b in blocks[-k:]:
            b.float()                         # fp32 master weights for the trained blocks
            for p in b.parameters():
                p.requires_grad_(True)
        norm = getattr(getattr(model, "model", model), "norm", None)
        if norm is not None:
            norm.float()
            for p in norm.parameters():
                p.requires_grad_(True)
        return model, hcfg["lr_full"], f"last_k k={k}"
    raise ValueError(f"unknown scope {scope}")


# --------------------------------------------------------------------------- training
def train(model, tok, mix: list[dict], lr: float, cfg: dict, seed: int, log=None) -> dict:
    hcfg = cfg["healing"]
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    if hcfg.get("gradient_checkpointing") and device.type == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=hcfg["weight_decay"])
    mb, accum = hcfg["micro_batch"], hcfg["grad_accum"]
    steps = max(1, math.ceil(len(mix) / (mb * accum)))
    warm = max(1, int(hcfg["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    use_amp = device.type == "cuda"
    model.train()
    seen_tokens, losses, step = 0, [], 0
    opt.zero_grad(set_to_none=True)
    for i in range(0, len(mix), mb):
        ids, labels, mask = _collate(mix[i:i + mb], tok.pad_token_id)
        ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
        (loss / accum).backward()
        seen_tokens += int(mask.sum())
        losses.append(float(loss))
        if (i // mb + 1) % accum == 0 or i + mb >= len(mix):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if log and step % 20 == 0:
                log.info(f"  step {step}/{steps} loss {sum(losses[-20:]) / len(losses[-20:]):.4f}")
    model.eval()
    if hcfg.get("gradient_checkpointing") and device.type == "cuda":
        model.gradient_checkpointing_disable()
    return {"steps": step, "tokens": seen_tokens, "final_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1),
            "n_trainable": sum(p.numel() for p in params)}


def finalize(model, scope: str, dtype):
    """Merge LoRA (if any) and return a plain model in the evaluation dtype."""
    if scope in ("all_lora", "cut_lora"):
        model = model.merge_and_unload()
    return model.to(dtype)
