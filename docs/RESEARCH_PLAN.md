# Research Plan: Recoverability-Aware Healing of Depth-Pruned Small LLMs

**Working title:** *Most Damaged Is Not Most Recoverable: Recoverability-Aware Healing for Depth-Pruned Small Language Models*
**Target venue:** IEEE Access (regular paper, about 10–12 pages)
**Plan date:** 2026-10-05
**Hardware:** 1× RTX 4060 Laptop (8 GB), plus Kaggle (2× T4, about 30 GPU-h/week)
**Compute budget:** about 12 GPU-hours core, about 18–19 GPU-hours with a 50% buffer

---

## 1. One-paragraph summary

Removing whole Transformer blocks ("depth pruning") makes small LLMs cheaper to deploy, but different capabilities break by different amounts. Examples are structured JSON output, function calling, instruction following, math, safety refusal, and knowledge. The standard fix is a short "healing" fine-tune. Today the healing budget is either spread uniformly or allocated in proportion to how *damaged* each capability is (PASER, ICLR 2026). We hypothesise that **how damaged a capability is does not predict how recoverable it is**: some skills are badly hurt but cheap to restore, others barely come back. We propose **Recoverability-Aware Healing (RAH)**. RAH fits a small recovery curve per capability from cheap pilot runs. It then jointly chooses the healing data mixture, token budget and adaptation scope, to maximise recovered capability per healing token. We evaluate on two model families (Qwen2.5-1.5B, Llama-3.2-1B), with an optional third (Gemma-2-2B), on a single laptop GPU.

---

## 2. Research questions and hypotheses

| ID | Research question | Hypothesis | How we test it |
|---|---|---|---|
| RQ1 | After depth pruning, in what order and by how much do capabilities break in 1–2B instruct models? | Damage differs strongly between capabilities. The order is broadly consistent across model families. | Retention per capability at two pruning levels; Kendall's W across families |
| RQ2 | Is damage a good predictor of recoverability? | **H1:** No. Rank correlation between damage and recovery ceiling is weak (Spearman ρ < 0.5). | Fit recovery curves from pilots; correlate damage with ceiling *a_c* and speed *τ_c* |
| RQ3 | Does optimising the healing allocation from recovery curves beat existing allocation rules at equal budget? | **H2:** RAH beats uniform and damage-proportional (PASER-style) allocation on mean *and* worst-case retention. | Main comparison, 3 seeds, paired bootstrap |
| RQ4 | Does the best place to adapt (which layers) depend on the capability? | **H3:** Yes. For example, format skills recover with last-layer tuning, while math needs adapters around the cut. | Scope pilots per capability |
| RQ5 | Can a near-zero-cost proxy replace the pilots? | **H4:** Per-capability output divergence on 50 prompts predicts the RAH allocation closely. | RAH-proxy vs RAH |

---

## 3. Novelty and positioning

Verified by three literature scans on 2026-10-05.

**Closest prior work, and what it does not do:**

| Work | What it does | Gap we fill |
|---|---|---|
| PASER (He et al., ICLR 2026, arXiv 2502.12594) | Allocates recovery data in proportion to the *damage* of each capability cluster | Width/unstructured pruning only, 7B+ models, perplexity and multiple-choice evaluation, no scope choice, no recoverability model |
| On the Limits of Layer Pruning for Generative Reasoning (arXiv 2602.01997) | Shows math and code break first and barely recover | Diagnosis only, no allocation method, little coverage below 3B |
| What Breaks Under Pruning in Smart Homes (arXiv 2609.17515) | Order in which tool-calling breaks, over-refusal | One domain, untargeted fine-tuning (SFT) |
| Reassessing Layer Pruning (arXiv 2411.15558) | Tune only the last layers plus the output head | Fixed scope, no data allocation |
| CLP (arXiv 2510.23652), LinearPatch (arXiv 2505.24680), Ghosted Layers (arXiv 2605.15491) | Repairs at the cut point | Fixed scope, no capability awareness |
| Self-Data Distillation (arXiv 2410.09982), ShortOPD (arXiv 2607.13124) | Teacher-generated healing targets | Orthogonal: we *use* teacher targets as the shared recipe |
| Sheared-LLaMA DBL (arXiv 2310.06694) | Online reweighting by loss gap | Pretraining domains, not capabilities |

**Claims we can make:**
1. The first *joint* optimisation of healing data mixture, budget and adaptation scope for depth-pruned LLMs.
2. Evidence that damage ≠ recoverability, plus per-capability recovery curves.
3. The first breakage ordering for depth-pruned 1–2B instruct models that covers structured output, function calling, instruction following, safety and calibration together.
4. A zero-cost proxy for allocation, and the full pipeline running on an 8 GB GPU.

**Claims we must NOT make:**
- That capability- or damage-aware data selection is new (PASER).
- That math and code breaking first is new.
- That tuning the layers next to the cut is new.
- That teacher distillation for healing is new.
- That pruning causing over-refusal is new.

---

## 4. Method: Recoverability-Aware Healing (RAH)

**Notation**
- *M₀* is the unpruned instruct model. *Mₚ* is *M₀* with *k* blocks removed by Block Influence (ShortGPT-style), reusing our existing code.
- *C* is the set of capabilities: {structured output (fmt), function calling (tool), instruction following (inst), math, safety (safe), knowledge (know)}.
- Retention is *r_c(M) = score_c(M) / score_c(M₀)*.
- *D_c* is the healing pool for capability *c*. Targets are regenerated by *M₀* (self-distillation, the shared recipe for every method).

**Step 1: Pilots.** For each pool *D_c*, heal *Mₚ* at two small budgets (100k and 300k tokens) with a fixed scope, and run a fast evaluation on all capabilities. That is 12 runs per model.

**Step 2: Fit the recovery model.**
- Per-capability curve: *g_c(x) = a_c · (1 − e^(−x/τ_c))*. Here *a_c* is the recovery ceiling, *τ_c* is the recovery speed, and *x* is the effective tokens.
- Transfer matrix *T* (|C|×|C|): the effect of tokens from pool *c′* on capability *c*. It comes from the same pilots, because each pilot is evaluated on every capability.
- Effective tokens for capability *c* are *x_c = Σ_c′ T_cc′ t_c′*.

**Step 3: Allocate (solved in milliseconds).**
- **RAH-sum:** maximise Σ_c w_c · g_c(x_c), subject to Σ_c t_c = B and t_c ≥ 0. Here *w_c* are deployment priorities, uniform by default.
- **RAH-maxmin:** maximise min_c (r_c(Mₚ) + g_c(x_c)), i.e. no capability left behind.
- Both are concave programs, solved with `scipy.optimize`.

**Step 4: Choose the scope.** Run 3 scope pilots with a uniform mixture:
- (a) last-3 blocks plus the output head;
- (b) LoRA on the 2 blocks next to the cut;
- (c) all-layer LoRA.

Pick the scope with the highest predicted value for the optimised mixture.

**Step 5: Full heal.** One training run with the optimised mixture, budget *B* and scope.

**RAH-proxy (H4):** skip steps 1–2. Estimate *a_c* from the per-capability KL divergence between *Mₚ* and *M₀* on 50 prompts per capability, using a mapping calibrated on the other model family. The cost is about 1 minute.

---

## 5. Experimental setup

### 5.1 Models

| Role | Model | Blocks | Main pruning (about 25%) | Light pruning (about 12.5%) |
|---|---|---|---|---|
| Main | Qwen2.5-1.5B-Instruct | 28 | remove 7 | remove 4 (diagnosis only) |
| Main | Llama-3.2-1B-Instruct | 16 | remove 4 | remove 2 (diagnosis only) |
| Optional | Gemma-2-2B-it | 26 | remove 6 | — |

The pruning level is confirmed in week 2 so that no capability sits at chance level (a lesson from the previous paper).

### 5.2 Evaluation suite

Fixed subsets are used, so every checkpoint sees identical items. Generation lengths are capped to control cost.

| Capability | Benchmark | Full eval items | Pilot eval items | Max new tokens | Scoring |
|---|---|---|---|---|---|
| fmt | JSONSchemaBench subset | 150 | 60 | 256 | Schema-valid rate |
| tool | BFCL v3 simple + multiple | 200 | 60 | 128 | AST match |
| inst | IFEval | 300 | 60 | 384 | Strict prompt accuracy |
| math | GSM8K test | 200 | 60 | 256 | Exact match |
| safe | XSTest (100 unsafe + 100 safe) | 200 | 60 | 64 | Refusal and over-refusal (string-match classifier) |
| know | MMLU | 300 | 60 | — (log-likelihood) | Accuracy + ECE (calibration side metric) |

### 5.3 Healing data pools

About 1,000 prompts each, kept separate from the evaluation sets. Targets are regenerated by *M₀*.

| Pool | Candidate source (verify licence and access in week 1) |
|---|---|
| tool, fmt | glaiveai/glaive-function-calling-v2 (fmt: JSON-schema tasks derived from it) |
| inst | IFEval-style constraint data (e.g. argilla/ifeval-like-data) |
| math | GSM8K train / MetaMathQA |
| safe | Harmful prompts from PKU-SafeRLHF + benign look-alikes (not from XSTest) |
| general | Alpaca-cleaned |

### 5.4 Methods compared (all at an equal token budget, B = 1M tokens)

| # | Method | Seeds |
|---|---|---|
| 1 | Uniform mixture (Gromov-style recipe) | 3 |
| 2 | Damage-proportional (PASER-style allocation) | 3 |
| 3 | **RAH-sum (ours)** | 3 |
| 4 | RAH-maxmin (ours) | 1 |
| 5 | RAH-proxy (ours, zero-cost) | 1 |
| 6 | Last-3 + head tuning, uniform data (Reassessing Layer Pruning) | 1 |
| 7 | Training-free linear patch (LinearPatch-style) | 1 (deterministic) |

**Ablations (Qwen only, 1 seed):** RAH without the transfer matrix; RAH without scope selection; power-law vs exponential curve.
**Budget sweep (Qwen only):** RAH vs Uniform at 0.25M and 4M tokens.

### 5.5 Metrics and statistics

- **Metrics:** per-capability retention; mean retention; worst-case retention; retention gained per 1M tokens; ECE and over-refusal as side-effect checks.
- **Bootstrap:** 95% confidence intervals over items, with seeds pooled.
- **Paired tests:** RAH vs each baseline on the same items.
- **H1:** Spearman correlation of damage vs *a_c*.
- **RQ1:** Kendall's W for consistency of the breakage order across families.

---

## 6. Compute plan

### 6.1 Unit costs

These are estimates for a 1.5B model on the RTX 4060 and are **measured in week 1** (Gate G0).

| Unit | Estimate | Notes |
|---|---|---|
| Full evaluation (all 6 capabilities) | about 7 min | Batched HF generation, about 155k generated tokens |
| Pilot evaluation | about 2 min | 60 items per capability |
| LoRA heal, 1M tokens | about 6 min | bf16, sequence length 512 with packing |
| LoRA heal, 100–300k tokens (pilot) | 1–2 min | |
| Teacher target generation (6 pools × 1k prompts) | about 25 min per model | Done once |

### 6.2 Compute ledger (2 main models)

| Phase | Runs | GPU-hours |
|---|---|---|
| Teacher data generation | 2 models | 0.9 |
| Damage diagnosis (unpruned + 2 pruning levels, full evaluation) | 6 evaluations | 0.7 |
| Recovery pilots (12 per model) | 24 | 1.6 |
| Scope pilots (3 per model) | 6 | 1.3 |
| Main comparison (13 runs per model) | 26 | 5.6 |
| Ablations (Qwen) | 3 | 0.7 |
| Budget sweep (Qwen) | 4 | 1.3 |
| **Core total** | | **about 12.1** |
| 50% buffer (reruns, debugging) | | about 6 |
| **Planned total** | | **about 18–19** |
| Optional: Gemma replication (RAH vs Uniform vs PASER, 1 seed, with pilots) | | +3 (on Kaggle) |

### 6.3 Where the compute runs

- **Laptop (RTX 4060):** all training and Qwen evaluations, as 2–3 overnight sessions of 6–8 h each.
- **Kaggle (2× T4):** Llama evaluation in parallel, plus the optional Gemma runs. This uses less than 10 of the 30 weekly hours.

### 6.4 Levers if Gate G0 shows higher costs than estimated

Apply in order until the plan fits:
1. Cut full-evaluation subsets by a third.
2. Use 2 seeds instead of 3 for the core methods.
3. Drop the budget sweep down to one extra budget.
4. Drop method 6.
5. Drop light-pruning diagnosis.

---

## 7. Timeline (5 weeks)

| Week | Work | Deliverable | Gate |
|---|---|---|---|
| **1** | Evaluation harness for the 6 capabilities; data pools; teacher generation; pruning code reused; timing benchmark | Validated unpruned scores for both models; measured unit costs | **G0:** unit costs within 1.5× of §6.1, otherwise apply §6.4 |
| **2** | Damage diagnosis; recovery pilots; scope pilots; curve fitting; optimiser | Damage table, recovery curves, transfer matrix | **G1 (go/no-go):** check H1 (below) |
| **3** | Main comparison on Qwen and Llama in parallel (laptop + Kaggle); ablations | Main results table with confidence intervals | — |
| **4** | Budget sweep; RAH-proxy; statistics; figures; optional Gemma | All figures and tables | **G2:** RAH ≥ best baseline on mean *or* worst-case retention |
| **5** | Write the IEEE Access paper; internal review; submit | Submitted manuscript + code release | — |

**Gate G1 (end of week 2, after about 4 GPU-hours spent).**
- **Pass (continue with RAH):** the damage-vs-ceiling correlation is ρ < 0.8, so recoverability does not simply track damage.
- **Pivot:** if ρ ≥ 0.8, RAH cannot meaningfully beat damage-proportional (PASER-style) allocation. Turn the paper into a scope-per-capability study (H3) plus a diagnosis study, reusing all pilots. No compute is wasted.

**Gate G2.** If RAH doesn't win, report it honestly as a negative result with the recovery-curve analysis, which is still a contribution, and target a workshop or conference instead of IEEE Access.

---

## 8. Expected paper structure (IEEE Access)

1. **Introduction:** cost of small-LLM deployment; depth pruning; the healing-allocation problem; contributions.
2. **Related work:** depth pruning; recovery and healing; data selection for recovery; capability-specific compression effects.
3. **Problem formulation:** retention, budget, scope.
4. **Method (RAH):** pilots, recovery model with transfer, allocation (sum and max-min), scope selection, proxy.
5. **Experimental setup:** models, pruning, evaluation suite, data pools, baselines, compute.
6. **Results:**
   - RQ1 breakage order;
   - RQ2 damage ≠ recoverability;
   - RQ3 main comparison;
   - RQ4 scope;
   - RQ5 proxy;
   - ablations;
   - budget sweep;
   - side effects (calibration, over-refusal).
7. **Discussion:** practitioner recipe, cost analysis.
8. **Threats to validity:**
   - small-model scope;
   - evaluation subsets;
   - teacher-target recipe;
   - single pruning criterion.
9. **Conclusion.**

**Planned figures:**
1. Pipeline.
2. Damage vs recovery ceiling scatter, the key figure.
3. Recovery curves per capability.
4. Transfer-matrix heatmap.
5. Retention radar or bars by method.
6. Budget–retention Pareto plot.

**Planned tables:**
1. Models.
2. Evaluation suite.
3. Main results with confidence intervals.
4. Ablations.
5. Compute cost.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| H1 false (recoverability ≈ damage) | Medium | Gate G1 pivot to scope + diagnosis paper |
| Evaluation too slow on Windows (no vLLM) | Medium | Batched HF generation with capped lengths; Kaggle (Linux) can use vLLM |
| Gated or unlicensed datasets | Low | Listed alternatives; verify in week 1 |
| Pruned model at chance on some capability | Medium | Choose the pruning level in week 2 so every capability is above floor |
| Evaluation noise on small subsets | Medium | Fixed items, paired bootstrap, 3 seeds for the core methods |
| New competing preprint before submission | Medium | Weekly arXiv check (search terms: "pruned LLM recovery", "healing allocation"); post our own arXiv preprint at submission |
| Over-refusal or safety regressions after healing | Low–Medium | Tracked as a side metric; the safety pool is included in every mixture |

---

## 10. Deliverables

- A code repository: pruning, healing, the evaluation harness, the RAH optimiser, and figure scripts.
- All results: per-item predictions, pilot curves, the transfer matrix, and run configs.
- Pruned and healed LoRA adapters for both model families.
- The IEEE Access manuscript (LaTeX) and an arXiv preprint.

---

## 11. Decisions needed from you

1. Approve the **5-week timeline and about 19 GPU-hour budget**, or ask for the 4-week variant: drop the budget sweep and optional Gemma, use 2 seeds, about 12 GPU-hours.
2. Confirm the **model choice** (Qwen2.5-1.5B + Llama-3.2-1B). The official Llama/Gemma checkpoints need a Hugging Face login; otherwise we use the ungated mirrors as before.
3. Confirm the **project folder** for code and results (default: this `rah-healing` folder).
