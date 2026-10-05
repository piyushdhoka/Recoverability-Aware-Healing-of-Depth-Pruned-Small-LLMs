"""Block Influence (ShortGPT): BI_l = 1 - mean_t cos(h_in, h_out), measured with hooks on each block.

Hooks capture the block's true input/output (before the final norm), and the first position
(BOS / attention sink) is excluded, fixing the two pitfalls documented in our earlier paper.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .modeling import get_blocks


@torch.no_grad()
def block_influence(model, batches: list[torch.Tensor], skip_first: int = 1) -> np.ndarray:
    """batches: list of [B, T] input-id tensors (right-padded with attention handled by masks)."""
    blocks, _ = get_blocks(model)
    store = {}

    def pre(i):
        def fn(module, args, kwargs):
            store[("in", i)] = args[0] if args else kwargs["hidden_states"]
        return fn

    def post(i):
        def fn(module, args, kwargs, output):
            store[("out", i)] = output[0] if isinstance(output, (tuple, list)) else output
        return fn

    handles = [b.register_forward_pre_hook(pre(i), with_kwargs=True) for i, b in enumerate(blocks)]
    handles += [b.register_forward_hook(post(i), with_kwargs=True) for i, b in enumerate(blocks)]
    sums = np.zeros(len(blocks))
    count = 0
    try:
        for ids, mask in batches:
            ids, mask = ids.to(model.device), mask.to(model.device)
            store.clear()
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            valid = mask[:, skip_first:].bool()
            for i in range(len(blocks)):
                x = store[("in", i)][:, skip_first:].float()
                y = store[("out", i)][:, skip_first:].float()
                cos = F.cosine_similarity(x, y, dim=-1)
                sums[i] += (1 - cos)[valid].sum().item()
            count += valid.sum().item()
    finally:
        for h in handles:
            h.remove()
    return sums / max(count, 1)


def lowest_k(bi: np.ndarray, k: int, protect_first: bool = True) -> list[int]:
    """k lowest-influence blocks; block 0 is never removed (it maps embeddings into the residual stream)."""
    order = [int(i) for i in np.argsort(bi, kind="stable") if not (protect_first and i == 0)]
    return sorted(order[:k])
