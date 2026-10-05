"""Model loading, chat formatting and permanent depth pruning.

Pruning removes blocks from the ModuleList *and* renumbers each kept block's layer index and the
config, so KV-cache generation stays correct after pruning (a silent bug if skipped).
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import pick_device, torch_dtype

BLOCK_PATHS = ["model.layers", "model.decoder.layers", "transformer.h"]
FALLBACK_TEMPLATE = (  # only for tiny test models that ship without a chat template
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for batched generation
    if not getattr(tok, "chat_template", None):
        tok.chat_template = FALLBACK_TEMPLATE
    return tok


def load_model(model_id: str, dtype: str = "bfloat16", device: str | None = None):
    device = device or pick_device()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype(dtype))
    model.to(device).eval()
    return model


def chat_prompt(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def chat_ids(tok, messages: list[dict], response: str | None = None, max_len: int | None = None):
    """Token ids for a chat prompt (+ optional assistant response) and a loss mask over the response."""
    prompt_ids = tok(chat_prompt(tok, messages), add_special_tokens=False)["input_ids"]
    if response is None:
        return prompt_ids, None
    resp_ids = tok(response + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = prompt_ids + resp_ids
    labels = [-100] * len(prompt_ids) + resp_ids
    if max_len is not None and len(ids) > max_len:
        ids, labels = ids[:max_len], labels[:max_len]
    return ids, labels


def _resolve(obj, path):
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def get_blocks(model):
    for path in BLOCK_PATHS:
        try:
            blocks = _resolve(model, path)
        except AttributeError:
            continue
        if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
            return blocks, path
    raise RuntimeError("could not locate transformer blocks")


@dataclass
class PruneSpec:
    n_layers: int
    removed: list[int]
    kept: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.removed = sorted(int(i) for i in self.removed)
        self.kept = [i for i in range(self.n_layers) if i not in set(self.removed)]

    def cut_adjacent(self) -> list[int]:
        """New (post-pruning) indices of kept blocks that directly border a removed span."""
        rem = set(self.removed)
        adj = set()
        for new_idx, orig in enumerate(self.kept):
            if (orig - 1) in rem or (orig + 1) in rem:
                adj.add(new_idx)
        return sorted(adj)

    def spans(self) -> list[tuple[int, int]]:
        """Contiguous removed runs as (first, last) original indices."""
        out = []
        for i in self.removed:
            if out and i == out[-1][1] + 1:
                out[-1] = (out[-1][0], i)
            else:
                out.append((i, i))
        return out

    def to_dict(self):
        return {"n_layers": self.n_layers, "removed": self.removed, "kept": self.kept}


def prune_model(model, spec: PruneSpec):
    """Permanently remove spec.removed blocks (in place) and keep config/cache indices consistent."""
    blocks, path = get_blocks(model)
    assert len(blocks) == spec.n_layers, (len(blocks), spec.n_layers)
    kept = [blocks[i] for i in spec.kept]
    for new_idx, block in enumerate(kept):
        for mod in block.modules():
            if hasattr(mod, "layer_idx"):
                mod.layer_idx = new_idx
    parent, attr = path.rsplit(".", 1)
    setattr(_resolve(model, parent), attr, nn.ModuleList(kept))
    cfg = model.config
    cfg.num_hidden_layers = len(kept)
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = [cfg.layer_types[i] for i in spec.kept]
    return model


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    """Greedy batched generation; prompts are sorted by length to minimise padding."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            out[i] = tok.decode(new[j], skip_special_tokens=True).strip()
    return out
