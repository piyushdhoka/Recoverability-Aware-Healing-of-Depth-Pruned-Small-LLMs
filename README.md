# Capability-Level Recovery of Depth-Pruned Small Language Models under Fixed and Compute-Matched Healing Budgets

Research code and results for a capability-level study of **healing** (short recovery fine-tuning) after **depth pruning**
(removing whole Transformer blocks) of small instruction-tuned language models.

We prune four 1.2–1.7B families with Block Influence, measure what breaks on six capabilities, fit recovery curves from
cheap pilot heals, and ask whether planning the healing data (**RAH**, recoverability-aware healing) beats uniform and
damage-proportional allocation, both at a fixed budget and once the pilot compute is counted.

| | |
|---|---|
| **Models** | Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, OLMo-2-0425-1B-Instruct |
| **Capabilities** | structured output (JSONSchemaBench), function calling (BFCL v3), instruction following (IFEval), math (GSM8K), safety refusal (XSTest), knowledge (MMLU) |
| **Healing** | LoRA (rank 16) on teacher-distilled pools, 1M-token budget, three seeds per core method |
| **Statistics** | item-level paired bootstrap, Holm correction, Kendall's *W*, seed standard deviations |

## Key findings

- **What breaks.** Function calling and math collapse on every family (chance-corrected retention ≤ 0.07); the order of
  the other capabilities is family-specific once chance levels are accounted for.
- **Scope matters as much as data.** Fully tuning the last three blocks ties all-block LoRA on Llama but loses
  0.05–0.18 of capped mean retention on the other three families; one 300k-token scope pilot detects this.
- **Fixed budget (1M tokens).** The planned allocation is never significantly worse than uniform healing on capped mean
  retention and raises the worst-retained capability (math) by +0.17 on Llama and +0.14 on OLMo. No difference from
  damage-proportional allocation survives multiple-comparison correction.
- **Compute-matched.** A uniform heal that also receives the 3.48M pilot tokens (4.48M total, three seeds) ties the best
  planned variant on Llama and OLMo and raises the worst-retained capability by 0.11 (Qwen) and 0.24 (SmolLM2): slowly
  recovering capabilities keep improving far beyond the pilot budgets. Planning pays off only when the healing budget
  itself is the binding constraint.

| Family | Uniform 1M | Damage-prop. 1M | RAH 1M | RAH-capped 1M | Uniform 4.48M |
|---|---|---|---|---|---|
| Llama-3.2-1B | 0.82 / 0.38 | 0.83 / 0.43 | 0.85 / 0.54 | 0.84 / 0.55 | 0.86 / 0.58 |
| Qwen2.5-1.5B | 0.62 / 0.21 | 0.61 / 0.19 | 0.64 / 0.19 | – | 0.65 / 0.29 |
| SmolLM2-1.7B | 0.77 / 0.15 | 0.76 / 0.14 | 0.77 / 0.10 | 0.78 / 0.10 | 0.79 / 0.34 |
| OLMo-2-1B | 0.74 / 0.21 | 0.77 / 0.31 | 0.75 / 0.29 | 0.76 / 0.35 | 0.78 / 0.37 |

*Capped mean retention / worst-case retention on the test split, three seeds each (retention = score relative to the
unpruned model; the capped mean counts values above 1 as 1). On Qwen, RAH-capped gives the same allocation as RAH.
Full tables: [`results/analysis/`](results/analysis).*

## Repository layout

```
rah-healing/
├── rah/                    core library
│   ├── modeling.py         model loading, chat formatting, permanent depth pruning
│   ├── influence.py        Block Influence scores
│   ├── healing.py          token-budgeted mixtures and LoRA / last-k healing
│   ├── recovery.py         saturating recovery curves + transfer matrix
│   ├── allocate.py         uniform, damage-proportional and RAH allocation rules
│   ├── proxy.py            pilot-free RAH-proxy
│   ├── linear_patch.py     training-free linear-patch baseline
│   ├── stats.py            bootstrap tests, rank agreement
│   ├── pipeline.py         glue shared by the stage scripts
│   ├── config.py, utils.py
│   ├── data/               evaluation items, healing pools, prompt templates
│   └── evaluation/         capability scorers and the evaluation runner
├── scripts/                numbered pipeline stages (see below) and run_all.{sh,ps1}
├── configs/                base.yaml + one file per model family (+ smoke.yaml for tests)
├── data/                   fixed evaluation items (eval/) and healing-pool prompts (pools/)
├── results/                every run as JSON, per family, plus cross-family analysis/
├── tests/                  unit tests (CPU only)
└── requirements.txt        pinned dependencies
```

## Setup

Python 3.12, an NVIDIA GPU with at least 8 GB (the Llama and Qwen pipelines ran on an 8 GB RTX 4060 laptop GPU), and
about 15 GB of disk. All models and datasets are public; no Hugging Face login is needed.

```bash
git clone https://github.com/piyushdhoka/Recoverability-Aware-Healing-of-Depth-Pruned-Small-LLMs.git rah-healing
cd rah-healing
python -m venv venv && source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_datasets.py                       # every dataset reachable?
pytest -q tests                                        # unit tests, CPU only
```

The results in this repository used transformers 4.56.2, PEFT 0.21.2 and lm-evaluation-harness 0.4.13 throughout
(PyTorch 2.5.1, 2.8.0 or 2.11.0 depending on the machine). Keep these pins when adding runs.

## Running the pipeline

Every stage takes `--model {llama,qwen,smollm,olmo}`, is resumable (finished runs are skipped), and writes to
`results/<model>/`. To run stages 0–7 for one family:

```bash
bash scripts/run_all.sh llama          # Windows: .\scripts\run_all.ps1 -Model llama   (add -From 5 to resume at stage 5)
```

| Stage | Script | What it does |
|---|---|---|
| 0 | `00_check_env.py` | record the environment, fail fast on missing pieces |
| 1 | `01_build_eval.py` | build the fixed dev/test evaluation items (model-independent, committed in `data/eval`) |
| 2 | `02_build_pools.py` | build the healing-pool prompts (model-independent, committed in `data/pools`) |
| 3 | `03_teacher.py` | generate targets with the unpruned model and quality-filter them (kept locally in `artifacts/`) |
| 4 | `04_diagnose.py` | evaluate the unpruned model, prune by Block Influence, measure damage, linear-patch baseline |
| 5 | `05_pilots.py` | 18 recovery pilots (6 pools × 3 budgets) and 3 scope pilots, on the dev split |
| 6 | `06_fit_allocate.py` | fit recovery curves and transfer matrix, choose the scope, compute every allocation |
| 7 | `07_main.py` | heal and evaluate on the test split: core methods × 3 seeds, variants, ablations, budget sweep |
| 8 | `08_analyze.py` | cross-family tables and figures in `results/analysis/` |
| 9 | `09_headline_tests.py` | paired bootstrap on the primary metrics with Holm correction |
| 10 | `10_compute_matched.py` | planned 1M heals versus the 4.48M-token uniform heal |

Stages 8–10 read every family and need no GPU. Extra seeds of the compute-matched heal:

```bash
python scripts/07_main.py --model qwen --only uniform --budgets 4480000 --sweep-seeds 1 2
```

Teacher data and adapters (`artifacts/`) are machine-local and not committed; move `artifacts/<model>/teacher/` with a
family if you continue its runs on another machine, so every seed heals on identical targets.

## Results layout

```
results/<model>/
├── diagnosis/      unpruned and pruned evaluations, Block Influence, prune specs, damage
├── pilots/         21 pilot heals (dev split)
├── fit/            recovery curves, transfer matrix, allocations of every method
├── main/           1M-token test runs: <method>_s<seed>.json with per-item records
└── budget/         budget sweep and compute-matched heals: uniform_b4480000_s<seed>.json
results/analysis/   main, damage, tests, headline_tests, compute_matched, budget tables (CSV + Markdown), figures
```

Each run file stores the allocation, training statistics, per-capability summary and every per-item score, so all
tables can be recomputed without a GPU.
