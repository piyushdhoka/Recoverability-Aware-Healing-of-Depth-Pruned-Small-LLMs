"""Training-free baseline in the spirit of LinearPatch / Ghosted Layers: for every removed span, fit a
ridge-regularised linear map from the hidden state entering the span to the hidden state leaving it (on
the unpruned model), and apply it to the output of the kept block that precedes the cut.
"""
import torch

from .modeling import PruneSpec, get_blocks


@torch.no_grad()
def collect_span_states(model, batches, spec: PruneSpec, max_tokens: int = 16384, skip_first: int = 1):
    """On the UNPRUNED model: X = input to the first removed block, Y = output of the last one, per span."""
    blocks, _ = get_blocks(model)
    spans = spec.spans()
    store, X, Y = {}, {s: [] for s in spans}, {s: [] for s in spans}

    def pre(i):
        def fn(module, args, kwargs):
            store[("in", i)] = args[0] if args else kwargs["hidden_states"]
        return fn

    def post(i):
        def fn(module, args, kwargs, output):
            store[("out", i)] = output[0] if isinstance(output, (tuple, list)) else output
        return fn

    handles = []
    for a, b in spans:
        handles.append(blocks[a].register_forward_pre_hook(pre(a), with_kwargs=True))
        handles.append(blocks[b].register_forward_hook(post(b), with_kwargs=True))
    n = 0
    try:
        for ids, mask in batches:
            store.clear()
            model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False)
            valid = mask[:, skip_first:].bool().to(model.device)
            for a, b in spans:
                X[(a, b)].append(store[("in", a)][:, skip_first:][valid].float().cpu())
                Y[(a, b)].append(store[("out", b)][:, skip_first:][valid].float().cpu())
            n += int(valid.sum())
            if n >= max_tokens:
                break
    finally:
        for h in handles:
            h.remove()
    return {s: (torch.cat(X[s])[:max_tokens], torch.cat(Y[s])[:max_tokens]) for s in spans}


def fit_patches(states: dict, ridge_scale: float = 1e-2) -> dict:
    """W = argmin ||XW - Y||^2 + lam ||W - I||^2  ->  (X^T X + lam I) W = X^T Y + lam I."""
    patches = {}
    for span, (X, Y) in states.items():
        d = X.shape[1]
        XtX = X.T @ X
        lam = ridge_scale * float(torch.trace(XtX)) / d
        eye = torch.eye(d)
        patches[span] = torch.linalg.solve(XtX + lam * eye, X.T @ Y + lam * eye)
    return patches


def attach_patches(pruned_model, spec: PruneSpec, patches: dict) -> list:
    """Register output hooks on the kept block just before each cut; returns hook handles."""
    blocks, _ = get_blocks(pruned_model)
    handles = []
    for (a, _b), W in patches.items():
        if a == 0:
            raise ValueError("cannot patch a span that starts at block 0")
        new_idx = spec.kept.index(a - 1)
        Wd = W.to(device=pruned_model.device)

        def hook(module, args, output, Wd=Wd):
            h = output[0] if isinstance(output, (tuple, list)) else output
            h2 = (h.float() @ Wd).to(h.dtype)
            return (h2,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else h2

        handles.append(blocks[new_idx].register_forward_hook(hook))
    return handles
