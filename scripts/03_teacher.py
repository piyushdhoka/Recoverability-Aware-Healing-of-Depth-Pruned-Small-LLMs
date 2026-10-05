"""Stage 3: generate healing targets with the UNPRUNED model (self-distillation) and quality-filter them."""
import _bootstrap  # noqa: F401

from rah.data.pools import passes_qc
from rah.modeling import chat_ids, chat_prompt, generate, load_model
from rah.pipeline import cli, log, tokenizer
from rah.utils import read_json, read_jsonl, timed, write_json, write_jsonl


def main():
    args, cfg, p = cli(__doc__)
    tok, model = tokenizer(cfg), None
    stats_path = p["results"] / "teacher_stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    for pool in cfg["pools"]:
        out = p["teacher"] / f"{pool}.jsonl"
        if out.exists():
            log.info(f"{pool}: exists, skipping")
            continue
        model = model or load_model(cfg["model_id"], cfg["dtype"])
        items = read_jsonl(p["pools"] / f"{pool}.jsonl")
        with timed(log, f"teacher {pool} ({len(items)} prompts)"):
            resps = generate(model, tok, [chat_prompt(tok, it["messages"]) for it in items],
                             cfg["teacher"]["max_new_tokens"][pool], cfg["teacher"]["batch_size"])
        passed = [{"id": it["id"], "pool": pool, "messages": it["messages"], "response": r}
                  for it, r in zip(items, resps)
                  if cfg["fixture_data"] or passes_qc(pool, r, it["check"])]   # fixtures: random model, no QC
        # examples longer than the training sequence length would lose their answer -> drop them here
        sl = cfg["healing"]["seq_len"]
        fits = [ex for ex in passed if len(chat_ids(tok, ex["messages"], ex["response"])[0]) <= sl]
        kept = fits[:cfg["data"]["pool_keep"]]
        stats[pool] = {"prompts": len(items), "passed_qc": len(passed), "too_long": len(passed) - len(fits),
                       "kept": len(kept), "pass_rate": len(kept) / max(len(items), 1)}
        if len(kept) < 0.5 * cfg["data"]["pool_keep"]:
            log.warning(f"{pool}: only {len(kept)} examples passed QC; consider raising data.pool_prompts")
        write_jsonl(out, kept)
        log.info(f"{pool}: kept {len(kept)}")
        write_json(stats_path, stats)


if __name__ == "__main__":
    main()
