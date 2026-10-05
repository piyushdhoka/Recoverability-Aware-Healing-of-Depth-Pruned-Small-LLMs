"""Model-level checks on a tiny random Llama (a few MB, CPU is fine)."""
import pytest
import torch

from rah.healing import apply_scope, build_mixture, finalize, mixture_tokens, tokenize_pool, train
from rah.influence import block_influence, lowest_k
from rah.linear_patch import attach_patches, collect_span_states, fit_patches
from rah.modeling import PruneSpec, chat_prompt, generate, get_blocks, load_model, load_tokenizer, prune_model

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CFG = {"healing": {"lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "lr_lora": 1e-3, "lr_full": 1e-4,
                   "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   "last_k": 1, "micro_batch": 2, "grad_accum": 1, "warmup_frac": 0.1, "weight_decay": 0.0,
                   "gradient_checkpointing": False, "seq_len": 64}}


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(TINY)


def fresh():
    return load_model(TINY, "float32", "cpu")


def test_prune_spec():
    s = PruneSpec(8, [5, 2, 3])
    assert s.removed == [2, 3, 5] and s.kept == [0, 1, 4, 6, 7]
    assert s.spans() == [(2, 3), (5, 5)]
    assert s.cut_adjacent() == [1, 2, 3]       # new indices of original blocks 1, 4, 6


def test_prune_renumbers_and_generation_still_works(tok):
    model = fresh()
    n = len(get_blocks(model)[0])
    spec = PruneSpec(n, [1])
    prune_model(model, spec)
    blocks, _ = get_blocks(model)
    assert len(blocks) == n - 1 == model.config.num_hidden_layers
    assert [b.self_attn.layer_idx for b in blocks] == list(range(n - 1))
    out = generate(model, tok, [chat_prompt(tok, [{"role": "user", "content": "hi"}])] * 3, 5, 2)
    assert len(out) == 3                       # KV-cache decoding works after pruning


def test_block_influence_and_lowest_k():
    model = fresh()
    ids = torch.randint(5, 100, (2, 16))
    bi = block_influence(model, [(ids, torch.ones_like(ids))])
    assert bi.shape == (len(get_blocks(model)[0]),) and (bi >= 0).all()
    assert 0 not in lowest_k(bi, 1)


@pytest.mark.parametrize("scope", ["all_lora", "cut_lora", "last_k"])
def test_heal_each_scope(tok, scope):
    model = fresh()
    n = len(get_blocks(model)[0])
    spec = PruneSpec(n, [1])
    prune_model(model, spec)
    exs = [{"messages": [{"role": "user", "content": f"q{i}"}], "response": "an answer"} for i in range(6)]
    tk = {"general": tokenize_pool(tok, exs, 64)}
    mix = build_mixture(tk, {"general": 60}, seed=0)
    assert mixture_tokens(mix) >= 60
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model, lr, _ = apply_scope(model, scope, CFG, spec.cut_adjacent())
    info = train(model, tok, mix, lr, CFG, seed=0)
    assert info["steps"] >= 1 and info["final_loss"] == info["final_loss"]   # finite
    model = finalize(model, scope, torch.float32)
    after = model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before if k in after), "weights did not change"


def test_linear_patch_identity_when_target_equals_input():
    model = fresh()
    n = len(get_blocks(model)[0])
    spec = PruneSpec(n, [1])
    ids = torch.randint(5, 100, (2, 16))
    states = collect_span_states(model, [(ids, torch.ones_like(ids))], spec)
    X, _ = states[(1, 1)]
    W = fit_patches({(1, 1): (X, X)})[(1, 1)]
    assert torch.allclose(W, torch.eye(X.shape[1]), atol=1e-3)
    prune_model(model, spec)
    hs = attach_patches(model, spec, {(1, 1): W})
    model(input_ids=ids)                       # forward with the patch hook attached
    for h in hs:
        h.remove()
