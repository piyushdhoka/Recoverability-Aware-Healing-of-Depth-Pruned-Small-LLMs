# Capability-Level Recovery of Depth-Pruned Small Language Models under Fixed and Compute-Matched Healing Budgets

Depth pruning removes whole Transformer blocks from a language model to make it smaller and faster, but it damages
capabilities unevenly. A short fine-tuning run, called **healing**, decides what comes back. This repository holds the
code and all results of a study of that process in four small instruction-tuned models: Llama-3.2-1B, Qwen2.5-1.5B,
SmolLM2-1.7B and OLMo-2-1B.

We measure six capabilities: structured output, function calling, instruction following, math, safety refusal and
knowledge. We then ask whether planning the healing data from short pilot runs (**RAH**, recoverability-aware healing)
does better than spreading it uniformly or in proportion to the damage. We test this at a fixed healing budget and again
once the compute spent on planning is counted.

## Key findings

- **What breaks.** Function calling and math collapse in every model; what happens to the other capabilities depends on
  the model.
- **What is trained matters as much as the data.** Fully tuning only the last three blocks instead of adding low-rank
  adapters to all blocks ties on one model and loses 0.05–0.18 of mean retention on the other three.
- **At a fixed budget**, planning never significantly hurts mean retention and lifts the weakest capability (math) by
  0.17 on Llama and 0.14 on OLMo.
- **At equal total compute**, simply healing longer with uniform data matches planning on two models and beats it on the
  weakest capability on the other two (by 0.11 and 0.24). Planning helps only when the healing budget itself is fixed.

| Model | Uniform | Damage-proportional | RAH | RAH-capped | Uniform, longer heal |
|---|---|---|---|---|---|
| Llama-3.2-1B | 0.82 / 0.38 | 0.83 / 0.43 | 0.85 / 0.54 | 0.84 / 0.55 | 0.86 / 0.58 |
| Qwen2.5-1.5B | 0.62 / 0.21 | 0.61 / 0.19 | 0.64 / 0.19 | – | 0.65 / 0.29 |
| SmolLM2-1.7B | 0.77 / 0.15 | 0.76 / 0.14 | 0.77 / 0.10 | 0.78 / 0.10 | 0.79 / 0.34 |
| OLMo-2-1B | 0.74 / 0.21 | 0.77 / 0.31 | 0.75 / 0.29 | 0.76 / 0.35 | 0.78 / 0.37 |

*Mean retention (capped at 1) / weakest-capability retention, relative to the unpruned model, on the test split,
averaged over three seeds. The first four columns heal with 1M tokens; RAH also spends 3.48M tokens on pilots, and the
last column gives uniform healing that same 4.48M-token total. On Qwen, RAH-capped gives the same allocation as RAH.*

## What is in the repository

- `rah/` – the library: pruning, healing, recovery curves, allocation rules and evaluation
- `scripts/` – the experiment pipeline, one numbered script per stage
- `configs/` – settings for each model
- `data/` – the evaluation items and healing prompts
- `results/` – every run with per-item scores, and summary tables in `results/analysis/`

## License

The code is released under the [MIT License](LICENSE). The models and datasets it uses keep their own licenses.
