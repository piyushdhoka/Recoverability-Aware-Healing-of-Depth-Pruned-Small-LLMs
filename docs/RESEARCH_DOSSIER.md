# Research Dossier: Recoverability-Aware Healing (RAH) for Depth-Pruned Small LLMs

**Working title:** *Recoverability-Aware Healing: Pilot-Guided Allocation of Data Mixture and Adaptation Scope for Depth-Pruned Small Language Models*
**Target venue:** IEEE Access (regular paper, about 10–12 pages)
**Document date:** 2026-10-05
**Status:** design approved; implementation starting (week 1)
**Companion files:** `docs/RESEARCH_PLAN.md` (short operational plan); this dossier is the long-form reference.

---

## Table of contents

1. Executive summary
2. Background primer
3. Problem statement and motivation
4. Literature review
5. Research gap and novelty claims
6. Research questions, hypotheses and falsification criteria
7. Method: Recoverability-Aware Healing
8. Experimental design
9. Compute plan for the two-laptop setup
10. Expected results and how each outcome is written up
11. Paper outline, figures and tables
12. Risks and mitigations
13. Glossary
14. References

---

## 1. Executive summary

**The setting.** Small instruction-tuned language models (1–2B parameters) are what industry actually deploys on phones, laptops, edge servers and cheap cloud instances, increasingly as *agents* that must emit valid JSON, call tools correctly, follow formatting instructions and refuse unsafe requests. *Depth pruning* — deleting entire Transformer blocks — is the simplest compression method that delivers real, hardware-agnostic savings: every removed block cuts parameters, memory and per-token compute proportionally, with no special kernels.

**The problem.** Pruning damages different capabilities by very different amounts, and a short fine-tune ("healing") is needed afterwards. Healing has a fixed budget (tokens, GPU time). Existing practice either spreads that budget *uniformly* over a generic instruction mixture, or — in the most advanced recent method, PASER (ICLR 2026) — allocates it *in proportion to how damaged* each capability cluster is.

**Our hypothesis.** *How damaged a capability is does not determine how recoverable it is.* Evidence already points this way: math and code are among the most damaged capabilities after layer pruning yet barely recover with fine-tuning (arXiv 2602.01997), whereas surface skills such as output formatting may be cheap to restore. If damage ≠ recoverability, damage-proportional allocation wastes budget on capabilities that will not come back.

**Our method.** *Recoverability-Aware Healing (RAH)*:
1. measures per-capability damage after pruning;
2. runs a dozen tiny pilot heals to fit a **recovery curve** per capability (ceiling and speed) and a **transfer matrix** (how data for one capability helps others);
3. solves a small concave optimisation that **jointly chooses the healing data mixture, the token budget split, and the adaptation scope** (which layers to train);
4. performs one full heal with that recipe.

A zero-cost variant (**RAH-proxy**) replaces the pilots with a one-minute divergence measurement.

**Scope and resources.**
- **Models:** two families, Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct.
- **Capabilities measured:** structured output, function calling, instruction following, math, safety (refusal and over-refusal) and knowledge/calibration.
- **Hardware:** one model per laptop (Qwen on an RTX 4070 laptop, Llama on an RTX 4060 laptop), with Kaggle as backup.
- **Compute:** about 12 GPU-hours core, about 18–19 with buffer.
- **Timeline:** about 4 weeks to submission.

**Built-in safety valve.** After about 4 GPU-hours (end of week 2), Gate G1 tests the central hypothesis. If damage and recoverability turn out to be strongly correlated, we pivot to a scope-per-capability + diagnosis paper that reuses every run.

---

## 2. Background primer

### 2.1 Decoder-only Transformers as a stack of residual blocks

A decoder-only LLM maps token embeddings $\mathbf{h}_0$ through $L$ blocks. Each block (pre-norm form) adds two residual updates:

$$\mathbf{u}_l = \mathbf{h}_{l-1} + \mathrm{Attn}_l(\mathrm{Norm}(\mathbf{h}_{l-1})),\qquad \mathbf{h}_l = \mathbf{u}_l + \mathrm{MLP}_l(\mathrm{Norm}(\mathbf{u}_l)).$$

A final norm and the language-model head produce next-token probabilities $p_\theta(x_{t+1}\mid x_{\le t}) = \mathrm{softmax}(W_{\mathrm{head}}\,\mathrm{Norm}(\mathbf{h}_L))$.

Because every block only *adds* an update to the residual stream, deleting a block leaves a well-defined (if degraded) network. This is why depth pruning works at all.

### 2.2 Depth (block) pruning

Depth pruning removes a set $\mathcal{S}\subset\{1,\dots,L\}$ of $|\mathcal{S}|=k$ blocks, giving $M_p = M_0^{\setminus\mathcal{S}}$.

**Savings per removed block** (for our models):
- about $1/L$ of decoder compute per token;
- a similar fraction of KV-cache memory;
- about 3% (Qwen2.5-1.5B) to 5% (Llama-3.2-1B) of all parameters.

This differs from width pruning (removing heads, channels or neurons), which keeps depth but slims each block, and from unstructured sparsity (zeroing individual weights), which needs sparse kernels to give speedups.

### 2.3 Choosing blocks: Block Influence

ShortGPT (arXiv 2403.03853) scores each block by how much it changes its input:

$$\mathrm{BI}_l = 1 - \mathbb{E}_{x,t}\!\left[\frac{\mathbf{h}_{l-1,t}^{\top}\mathbf{h}_{l,t}}{\lVert\mathbf{h}_{l-1,t}\rVert\,\lVert\mathbf{h}_{l,t}\rVert}\right],$$

and removes the $k$ blocks with the lowest BI. Our earlier study (the LB-BI paper in `../paper/`) supplies a validated, leak-free implementation. That includes two pitfalls we already found and fixed:
- capturing block outputs *before* the final norm;
- prepending BOS tokens for models that define one.

Alternatives in the literature are:
- angular distance over contiguous spans (Gromov et al., arXiv 2403.17887);
- reverse-order removal of the last layers (arXiv 2411.15558);
- learned or iterative criteria.

RAH is agnostic to the pruning criterion. We fix BI to isolate the healing question.

### 2.4 Healing (recovery) after pruning

A pruned model is "healed" by a short fine-tune on a modest amount of data. Formally, with a healing dataset $\mathcal{D}$ and a set of trainable parameters $\phi$ (the *scope*):

$$\phi^{\star} = \arg\min_{\phi}\ \mathbb{E}_{(x,y)\sim\mathcal{D}}\Big[-\sum_{t}\log p_{M_p,\phi}(y_t\mid x, y_{<t})\Big],$$

constrained by a token budget $B$ (the number of training tokens processed). Healing restores much, but rarely all, of the lost ability. How much comes back depends on *what data* is used, *how much*, and *which parameters* are trained.

### 2.5 Adaptation scopes: LoRA vs partial fine-tuning

- **LoRA** (Hu et al., 2022) freezes a weight $W\in\mathbb{R}^{d\times d'}$ and learns a low-rank update $\Delta W = BA$, with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times d'}$ and $r\ll d$. It is cheap in memory and easy to merge. QLoRA (arXiv 2305.14314) does the same on a 4-bit base.
- **Partial (layer-selective) fine-tuning** trains a subset of whole blocks at full rank. arXiv 2411.15558 reports that fine-tuning only the last few layers (plus the head) beats LoRA for healing layer-pruned models.
- **Cut-adjacent tuning** trains the blocks immediately before and after each removed span, where the "representation mismatch" is created. CLP (arXiv 2510.23652) uses this "cutoff endpoint tuning".
- **Linear patches** insert a (training-free or lightly trained) linear map at the cut to re-align hidden states: LinearPatch (arXiv 2505.24680) and Ghosted Layers (arXiv 2605.15491).

We define three trainable scopes plus one training-free baseline:

| Scope | Trainable parameters |
|---|---|
| (a) last-k | Full-rank last 3 blocks plus final norm (the output head is tied to the embeddings in both models, so it stays frozen) |
| (b) cut-LoRA | LoRA on the blocks adjacent to each cut |
| (c) all-LoRA | LoRA on all linear layers of all remaining blocks |
| Linear patch (baseline only) | Ridge-regression linear map at each cut, no gradient training |

### 2.6 Self-distillation targets

Healing data needs targets. Instead of human-written answers, we regenerate every target with the *unpruned* model $M_0$ (self-data distillation, arXiv 2410.09982). The healed model is thereby pulled back towards its own original behaviour, rather than towards a new distribution. This is known to retain more quality than plain SFT (91.2% vs 81.7% on Llama-3.1-8B with 6 blocks removed, as reported in 2410.09982).

**This is a shared recipe used by every method we compare, not our contribution.**

### 2.7 Capability evaluation

We track six capabilities $c\in C$ with dedicated benchmarks (§8.3). For a model $M$ we define **retention**

$$r_c(M) = \frac{\mathrm{score}_c(M)}{\mathrm{score}_c(M_0)},$$

so $r_c(M_0)=1$. **Damage** after pruning is $d_c = 1 - r_c(M_p)$. Retention normalises across benchmarks with different scales and baseline levels, which lets us aggregate (mean) and compare (worst case) across capabilities.

---

## 3. Problem statement and motivation

### 3.1 Industrial motivation

- **Small models are the deployment workhorse.** 0.5–3B instruct models run on laptops, phones and single small GPUs. Every block removed buys latency, memory and energy, and depth pruning does so without sparse kernels or new hardware.
- **Deployments are agentic.** Real products need valid JSON, correct function names and arguments, faithful instruction following and calibrated refusal — not just low perplexity. Recent evidence shows compression damage on exactly these skills: quantised agents show amplified failure modes under flat task scores; tool-calling accuracy of a 0.6B model falls from 62.3% to 16.8% under quantisation (arXiv 2608.22472); and pruned smart-home models lose argument-level specificity before intent and become heavily over-refusing (7.7% → 80.8% false refusals, arXiv 2609.17515).
- **Healing is a budgeted engineering step.** Teams have a fixed amount of GPU time and curated data. The question "*what should I heal with, how much, and where?*" is asked by every team that ships a pruned model, and today it is answered by intuition.

### 3.2 Formal problem

Given:
- an unpruned model $M_0$;
- a pruned model $M_p$;
- capability set $C$ with priorities $w_c\ge 0$;
- capability-targeted healing pools $\{D_j\}_{j\in P}$ (one per capability plus a general pool);
- a set of scopes $\Sigma$;
- a total token budget $B$;

choose a token allocation $\mathbf{t}=(t_j)_{j\in P}$ with $\sum_j t_j = B$, $t_j\ge 0$, and a scope $\sigma\in\Sigma$ that maximise the post-healing objective:

$$\max_{\mathbf{t},\sigma}\ \sum_{c\in C} w_c\, r_c\big(\mathrm{Heal}(M_p;\mathbf{t},\sigma)\big)\quad\text{or}\quad \max_{\mathbf{t},\sigma}\ \min_{c\in C} r_c\big(\mathrm{Heal}(M_p;\mathbf{t},\sigma)\big).$$

$\mathrm{Heal}(\cdot)$ is an expensive, noisy black box (a training run plus a full evaluation), so the problem cannot be solved by direct search over many recipes on a laptop. RAH replaces the black box with a cheap, fitted surrogate.

---

## 4. Literature review

Verification legend: IDs come from three saturation scans and two focused novelty checks run on 2026-10-05. [V] means the agent opened the arXiv page or abstract; [?] means some detail (authors, venue or ID) was not re-confirmed. Every [?] item must be checked before it is cited.

### 4.1 Depth pruning of LLMs

- **ShortGPT** (Men et al., arXiv 2403.03853) [V]: BI scoring, removes many blocks of large models. *Does not:* study per-capability effects or allocate healing.
- **The Unreasonable Ineffectiveness of the Deeper Layers** (Gromov et al., ICLR 2025, arXiv 2403.17887) [V]: prunes contiguous deep spans chosen by angular distance, heals with QLoRA on C4. *Does not:* target capabilities; healing data is generic.
- **Shortened LLaMA** (Kim et al., arXiv 2402.02834) [V]: compares retraining methods after depth pruning. *Does not:* optimise data mixture or scope.
- **LLM-Streamline** (Chen et al., ICLR 2025, arXiv 2403.19135) [V]: replaces pruned spans with lightweight modules.
- **Reassessing Layer Pruning in LLMs** (arXiv 2411.15558; authors [?], believed Lu et al.) [V]: reverse-order pruning of about 25% of layers, then tuning the lm_head and last three layers, beats LoRA. *Does not:* choose data or adapt scope to capability. **Mandatory baseline.**
- **CLP: The Structural Scalpel** (Lu, Li, Xie, Yu, Xuan, Zhu, Wen; arXiv 2510.23652) [V]: automated contiguous pruning plus "cutoff endpoint tuning" of the two cut-adjacent layers. *Does not:* allocate data. No public code found. We reimplement it as scope (b).

### 4.2 Repairs at the cut point

- **LinearPatch** (Chen, Bai, Yuan et al.; NeurIPS 2025, arXiv 2505.24680) [V]: Hadamard transform plus channel scaling at the pruning interface, optional KD on about 5K samples. Public code.
- **Ghosted Layers** (Yun, Jo, Karimireddy, Lee; arXiv 2605.15491) [V]: training-free optimal linear operator fitted on calibration data. Beats constrained patches. Public code.
- **OverRep** (EMNLP 2026, arXiv 2609.06974) [V]: over-parameterised recovery module merged afterwards. **RestoreLCC** (NeurIPS 2025, arXiv 2510.21834) [V].
- *None of these* are capability-aware or budget-allocating.

### 4.3 Healing data and recovery recipes

- **PASER** (He, Yin, Zhen, Zhang, Yuan, Ma; ICLR 2026, arXiv 2502.12594) [V]. Code: github.com/BokwaiHo/PASER.
  - **Method:** spectral clustering of the instruction pool into "capability clusters"; the budget is split across clusters **in proportion to measured degradation** (token-level JS divergence between original and pruned outputs); within clusters it picks the most degraded samples and filters conflicts.
  - **Scope:** width, semi-structured and unstructured pruning (LLM-Pruner, SliceGPT, Wanda 2:4, SparseGPT); LLaMA2/3 at 7–70B and Baichuan2; perplexity and commonsense multiple-choice evaluation.
  - **Does not:** depth pruning, models under 3B, generative/agentic capabilities, adaptation-scope choice, any model of recoverability.
  - **This is our primary methodological foil:** damage-proportional ≠ recoverability-aware.
- **Self-Data Distillation** (Thangarasa et al., Cerebras; arXiv 2410.09982) [V]: teacher-regenerated targets for recovery. We adopt it as the shared recipe.
- **ShortOPD** (Zhang, Yuan, Lin, Lu, Han, Sun, Xu, Li; arXiv 2607.13124) [V]: on-policy distillation with a rollout-length budget, recovering math, code and open-ended generation; 1.6–4.4× gains over SFT/KD. Orthogonal to allocation, and could be combined with RAH in future.
- **Sheared-LLaMA** (Xia et al., ICLR 2024, arXiv 2310.06694) [V]: *dynamic batch loading* reweights pretraining domains online by the gap between current and reference loss. **A direct precedent for adaptive mixtures during recovery**, but loss-based, pretraining-scale and domain-level, not capability-level. It must be cited, and is optionally compared against.
- **Damage Predicts Recovery** (Ye, Yu, Hazra, Wang; arXiv 2609.26241) [V]: in compressing financial LLMs, measure task-specific damage first and use specialised calibration data only when damage is large. Close in spirit (damage-gated data choice), but about calibration data in one domain, not depth healing.
- **Self-Distillation as a Performance Recovery Mechanism** (arXiv 2604.15794) [V].

### 4.4 What breaks under compression (capability-specific studies)

- **On the Limits of Layer Pruning for Generative Reasoning in LLMs** (Shrestha, Shrestha, Nepal, Kim, Ross; arXiv 2602.01997) [V].
  - **Setup:** Gemma2-2B-It, Llama-3.1-8B-It, Qwen2.5-7B-It and Mistral-7B; BI, reverse and iterative pruning; healing with QLoRA, self-generated data, GSM8K full fine-tuning, RLVR and KD.
  - **Findings:** math and code are the most fragile and barely recover; classification recovers to about 90%; arithmetic and parenthesis balancing break first.
  - **Does not:** JSON, function calling, IFEval, safety or calibration; allocation; much coverage below 3B.
  - It is the strongest evidence for our H1.
- **What Breaks Under Pruning in Smart Homes, and When?** (Zhang, Patil, Lange, Aleem; arXiv 2609.17515, EACL Industry) [V]: Qwen3/3.5/3.6 at 4B+; ShortGPT, Angular, width and expert pruning; about 50K-example full SFT. Grounded specificity breaks before schema-level intent; false refusals rise from 7.7% to 80.8%. *Does not:* targeted healing, other domains, models under 4B.
- **Can Compressed LLMs Truly Act? (ACBench)** (Dong et al., ICML 2025, arXiv 2505.19433) [V]: agentic evaluation of GPTQ, AWQ, Wanda and SparseGPT. 4-bit quantisation costs 1–3% on tool use but 10–15% on real applications. *No depth pruning, no healing.*
- **LLM-KICK: Compressing LLMs — The Truth Is Rarely Pure and Never Simple** (Jaiswal et al., ICLR 2024, arXiv 2310.01382) [V]: compressed models can look fine on perplexity yet fail knowledge-intensive tasks.
- **Width Pruning Dichotomy in Llama-3.2** (Martra; arXiv 2512.22671) [V]: IFEval *improves* under GLU width pruning of 1B/3B models. Capability effects can even be non-monotone.
- **When Fewer Layers Break More Chains** (arXiv 2510.22228) [?]: layer pruning and long-CoT reasoning at 7–8B.

### 4.5 Neighbouring compression lines

These are context, not competitors.
- **Compression ordering:** A Systematic Study of Compression Ordering for LLMs (arXiv 2511.19495) [V] — prune → KD → quantise is best on Qwen2.5-3B.
- **Quantisation and reasoning:** Quantization Hurts Reasoning? (arXiv 2504.04823) [V]; RAC (NeurIPS 2025, arXiv 2509.12464) [V].
- **Layer-selective PEFT** (LISA [?]; AdaGradSelect arXiv 2512.15764 [V]; CKA-based layer-wise LoRA arXiv 2602.05988 [V]): a crowded area, used here only as one dimension of scope.

### 4.6 Comparison table

| Work | Pruning type | Model size | Capabilities evaluated | Healing data choice | Budget model | Scope choice |
|---|---|---|---|---|---|---|
| ShortGPT 2403.03853 | Depth (BI) | 7B+ | PPL, MCQ | — | — | — |
| Gromov et al. 2403.17887 | Depth (angular) | 7B+ | MCQ | Generic (C4) | — | Fixed (QLoRA) |
| Reassessing LP 2411.15558 | Depth | 7B+ | MCQ, generation | Generic | — | Fixed (last-k + head) |
| CLP 2510.23652 | Depth (contiguous) | 7–70B | MCQ | Generic | — | Fixed (cut-adjacent) |
| LinearPatch / Ghosted | Depth | 7B+ | PPL, MCQ | — / calibration | — | Fixed (patch) |
| Self-Data Distill. 2410.09982 | Depth | 8B | MCQ, generation | Teacher-regenerated | — | Fixed |
| ShortOPD 2607.13124 | Depth/width | — | Math, code, open-ended | On-policy KD | Rollout length | Fixed |
| **PASER 2502.12594** | **Width/unstructured** | **7–70B** | PPL, MCQ | **Damage-proportional clusters** | **Proportional heuristic** | — |
| Sheared-LLaMA 2310.06694 | Structured | 1.3–2.7B | Broad | Loss-gap online reweighting | Heuristic | — |
| Limits of LP 2602.01997 | Depth | 2–8B | Math, code, summarisation, classification | Several (untargeted) | — | — |
| Smart homes 2609.17515 | Depth + width | 4B+ | Tool calling, refusal | Generic full SFT | — | — |
| **RAH (ours)** | **Depth (BI)** | **1–1.5B** | **JSON, tools, IFEval, math, safety, MMLU + ECE** | **Per-capability pools, optimised mixture** | **Fitted recovery curves + transfer** | **Optimised over 3 scopes** |

---

## 5. Research gap and novelty

### 5.1 The gap

The literature supplies three separate ingredients:
- *diagnosis* of what breaks (2602.01997, 2609.17515);
- *damage-weighted* data allocation (PASER, Sheared-LLaMA, 2609.26241);
- several *fixed* repair scopes (last-k, cut-adjacent, linear patch, LoRA).

No work models **how recoverable** each capability is, or chooses **data, budget and scope jointly**, for **depth-pruned models under 3B** evaluated on **generative and agentic capabilities**. The no-competitor search covered work up to October 2026.

### 5.2 Claims the paper can make

1. **First joint optimisation** of healing data mixture, token budget and adaptation scope for depth-pruned LLMs, based on fitted per-capability recovery curves with cross-capability transfer.
2. **Empirical evidence that damage ≠ recoverability** (if H1 holds), with per-capability recovery curves (ceiling, speed) and a transfer matrix for two model families.
3. **The first breakage ordering for depth-pruned 1–2B instruct models** that jointly covers structured output, function calling, instruction following, math, safety/over-refusal and knowledge/calibration, with a cross-family consistency test.
4. **A near-zero-cost proxy** that predicts the allocation without pilots, and the complete pipeline on 8 GB consumer GPUs.

### 5.3 Claims the paper must not make

- That capability- or damage-aware selection of recovery data is new → PASER, Sheared-LLaMA, 2609.26241.
- That math and code break first or barely recover → 2602.01997.
- That tuning the layers next to the cut is new → CLP.
- That teacher distillation for healing is new → 2410.09982, ShortOPD.
- That pruning causes over-refusal → 2609.17515.
- That LoRA is suboptimal for healing → 2411.15558.

---

## 6. Research questions, hypotheses and falsification criteria

| ID | Question | Hypothesis | Test | Falsified if |
|---|---|---|---|---|
| RQ1 | In what order, and by how much, do capabilities break under depth pruning in 1–2B instruct models? Is the order consistent across families? | Damage varies strongly across capabilities; the order is broadly shared | Retention per capability at two pruning levels; Kendall's $W$ across the two families' rankings | Spread of $d_c$ across capabilities < 0.1, or $W$ not significantly > chance |
| RQ2 | Does damage predict recoverability? | **H1:** weakly at best | Spearman $\rho(d_c, a_c)$ over capabilities × models; also $\rho(d_c, \tau_c)$ | $\rho \ge 0.8$, which triggers the G1 pivot |
| RQ3 | Does optimised allocation beat existing rules at equal budget? | **H2:** RAH-sum > Uniform and > Damage-proportional on mean retention; RAH-maxmin > both on worst-case retention | 3 seeds; paired bootstrap over items | RAH's 95% CI for the mean-retention gain over the best baseline includes 0 on both models |
| RQ4 | Is the best adaptation scope capability-dependent? | **H3:** yes (e.g. formatting via last-k, math via cut-LoRA or all-LoRA) | Scope pilots; per-capability arg-max scope | One scope is best for every capability |
| RQ5 | Can a cheap proxy replace pilots? | **H4:** the KL-divergence proxy gives an allocation within a small regret of RAH | RAH-proxy vs RAH, with the mapping calibrated on the other family | Proxy regret ≥ half the RAH-vs-uniform gap |

**Note on statistical power.** H1 rests on 6 capabilities × 2 models = 12 points. We therefore report it as a rank correlation *with a bootstrap CI over pilot items*, plus per-model scatter plots, and avoid over-claiming. The core contribution is H2 (method performance), which is tested on item-level paired data with much more power.

---

## 7. Method: Recoverability-Aware Healing

### 7.1 Notation

| Symbol | Meaning |
|---|---|
| $M_0$, $M_p$ | Unpruned and pruned model |
| $C$ | Capabilities {fmt, tool, inst, math, safe, know} |
| $P$ | Healing pools {fmt, tool, inst, math, safe, gen} |
| $\pi(c)$ | The "own" pool of capability $c$ (identity, except know → gen) |
| $r_c(M)$ | Retention of capability $c$ |
| $d_c = 1 - r_c(M_p)$ | Damage |
| $t_j$ | Healing tokens drawn from pool $j$, with $\sum_j t_j = B$ |
| $\sigma\in\Sigma$ | Adaptation scope |
| $w_c$ | Deployment priority weights (default $1/|C|$) |

### 7.2 Recovery model

**Recovery curve.** The retention gain of capability $c$ as a function of *effective* healing tokens $x$ is modelled as a saturating exponential:

$$g_c(x) = a_c\,\big(1 - e^{-x/\tau_c}\big),\qquad a_c\ge 0,\ \tau_c>0.$$

- $a_c$ is the **recovery ceiling**: the most that can come back under this recipe.
- $\tau_c$ is the **recovery timescale**: the tokens needed to recover about 63% of the ceiling.
- $g_c$ is increasing and concave, which captures diminishing returns.

A power-law alternative $g_c(x)=a_c\,x^{\beta_c}/(x^{\beta_c}+\kappa_c^{\beta_c})$ is tested as an ablation.

**Transfer and effective tokens.** Data for one capability can help (or hurt) another: instruction-following data may improve JSON validity; math data may slightly degrade refusal. We model the effective tokens for capability $c$ as a linear combination of pool tokens:

$$x_c(\mathbf{t}) = \sum_{j\in P} T_{cj}\, t_j,\qquad T_{c,\pi(c)} = 1.$$

Normalising the own-pool coefficient to 1 removes the scale ambiguity between $T$ and $\tau_c$. Off-diagonal entries may be negative (interference). Predicted post-healing retention is

$$\hat r_c(\mathbf{t}) = r_c(M_p) + g_c\big(x_c(\mathbf{t})\big).$$

### 7.3 Pilot design and fitting

**Pilots.** For each pool $j\in P$ and two small budgets $b\in\{100\mathrm{k}, 300\mathrm{k}\}$ tokens, heal $M_p$ with *only* pool $j$ (fixed pilot scope: all-LoRA), and run the *fast* evaluation (60 items per capability). This gives $|P|\times 2 = 12$ runs per model. **Each pilot is scored on every capability**, so we observe $\Delta r_c(j,b)$ for all $c$.

**Fitting.** For each capability $c$, the unknowns are $a_c$, $\tau_c$ and the off-diagonal row $T_{c,\cdot}$ (5 values): 7 parameters from 12 observations. We solve a bounded, regularised least-squares problem:

$$\min_{a_c,\tau_c,T_{c,\cdot}}\ \sum_{j,b}\Big(\Delta r_c(j,b) - g_c(T_{cj}\,b)\Big)^2 + \lambda\sum_{j\ne\pi(c)} T_{cj}^2,$$

subject to:
- $0\le a_c\le 1.1 - r_c(M_p)$: retention cannot meaningfully exceed the unpruned model;
- $\tau_c\in[10^4, 10^8]$;
- $T_{cj}\in[-1, 2]$.

**Why regularise.** With only two budget points per pool, $a_c$, $\tau_c$ and $T_{cj}$ are partly confounded. The ridge penalty shrinks transfer towards zero unless the data clearly show it, and the bounds rule out degenerate fits.

We report:
- fit residuals;
- a leave-one-pilot-out prediction error;
- bootstrap CIs over pilot evaluation items for $a_c$ and $\tau_c$.

These make the identifiability limits visible.

### 7.4 Allocation

**RAH-sum (utilitarian):**

$$\max_{\mathbf{t}\ge 0}\ \sum_{c\in C} w_c\, g_c\big(x_c(\mathbf{t})\big)\quad\text{s.t.}\quad \sum_{j}t_j = B.$$

Each $g_c$ is concave and non-decreasing, and $x_c$ is affine in $\mathbf{t}$, so each term is concave. Therefore the objective is concave even with negative transfer coefficients. The problem is a smooth convex program over a simplex. We solve it with `scipy.optimize` (SLSQP) from multiple starts, in milliseconds.

**RAH-maxmin (no capability left behind):**

$$\max_{\mathbf{t}\ge 0,\,s}\ s\quad\text{s.t.}\quad r_c(M_p) + g_c\big(x_c(\mathbf{t})\big)\ \ge\ s\ \ \forall c,\qquad \sum_j t_j = B.$$

This is also a convex program, because the superlevel sets of concave functions are convex.

**Baselines in the same notation:**
- **Uniform:** $t_j = B/|P|$.
- **Damage-proportional** (PASER-style allocation rule, applied to our pools): $t_{\pi(c)} \propto d_c$.

### 7.5 Scope selection

Scope pilots heal $M_p$ with the *uniform* mixture at the pilot budget under each scope (a) last-k, (b) cut-LoRA and (c) all-LoRA, and run the fast evaluation. From these we estimate a per-scope, per-capability efficiency multiplier $\eta_{c,\sigma}$ relative to the pilot scope (all-LoRA):

$$\eta_{c,\sigma} = \frac{\Delta r_c(\text{uniform},\sigma)}{\Delta r_c(\text{uniform},\text{all-LoRA})},$$

clipped to [0, 3]. Under scope $\sigma$ the predicted gain is $\eta_{c,\sigma}\,g_c(x_c)$.

RAH re-solves the allocation for each scope and picks $(\mathbf{t}^\star,\sigma^\star)$ with the highest predicted objective. This tests H3 directly: if the arg-max scope varies by capability, the optimal *global* scope depends on the damage profile and priorities.

### 7.6 RAH-proxy (zero-cost variant)

For each capability, take 50 pool prompts with teacher responses and compute the mean token-level KL divergence between $M_0$ and $M_p$ on the teacher responses:

$$\kappa_c = \frac{1}{N_c}\sum_{(x,y)}\sum_t \mathrm{KL}\big(p_{M_0}(\cdot\mid x,y_{<t})\,\|\,p_{M_p}(\cdot\mid x,y_{<t})\big).$$

A simple mapping $\hat a_c = \alpha + \beta\,\kappa_c$ (clipped to the bounds above) is calibrated on the *other* model family's fitted ceilings. The timescales use that family's median $\tau$, and transfer is set to the identity. Allocation then proceeds as in §7.4.

Cost: two forward passes over 300 prompts, about 1 minute. The cross-family calibration is what makes the proxy honest: it never sees its own model's pilots.

### 7.7 Full heal and evaluation

Train once with $(\mathbf{t}^\star,\sigma^\star)$ at budget $B=1\mathrm{M}$ tokens, then run the full evaluation (§8.3). Methods differ *only* in mixture and scope. Data, targets, optimiser, learning-rate schedule, sequence length and budget are shared.

### 7.8 Worked toy example: why damage-proportional allocation can fail

Consider three capabilities, budget $B = 1$ (in units of M tokens) and no transfer ($T = I$):

| Capability | Retention after pruning $r_c(M_p)$ | Damage $d_c$ | Ceiling $a_c$ | Timescale $\tau_c$ |
|---|---|---|---|---|
| fmt | 0.90 | 0.10 | 0.10 | 0.1 |
| inst | 0.70 | 0.30 | 0.25 | 0.5 |
| math | 0.40 | 0.60 | 0.08 | 3.0 |

Math is the most damaged but nearly unrecoverable (small ceiling, slow). Formatting is mildly damaged but fully and quickly recoverable.

**Damage-proportional** gives $t = (0.1, 0.3, 0.6)$:
- $g_{\mathrm{fmt}} = 0.10(1-e^{-1}) = 0.063$
- $g_{\mathrm{inst}} = 0.25(1-e^{-0.6}) = 0.113$
- $g_{\mathrm{math}} = 0.08(1-e^{-0.2}) = 0.015$
- Total gain **0.191**.

**Uniform** gives $t = (1/3, 1/3, 1/3)$:
- $g = 0.096,\ 0.122,\ 0.008$
- Total **0.227**.

**RAH-sum.** At the optimum, the marginal gains $g_c'(t)=(a_c/\tau_c)e^{-t/\tau_c}$ are equalised. Math's marginal gain at zero tokens, $0.08/3 = 0.027$, is lower than what the other two still offer at the optimum, so math receives nothing. Solving $e^{-10t_f} = 0.5\,e^{-2(1-t_f)}$ gives $t = (0.224, 0.776, 0)$:
- $g = 0.089,\ 0.197,\ 0$
- Total **0.286**.

| Allocation | fmt | inst | math | Mean retention | Worst-case retention |
|---|---|---|---|---|---|
| Pruned, unhealed | 0.900 | 0.700 | 0.400 | 0.667 | 0.400 |
| Damage-proportional | 0.963 | 0.813 | 0.415 | 0.730 | 0.415 |
| Uniform | 0.996 | 0.822 | 0.408 | 0.742 | 0.408 |
| **RAH-sum** | 0.989 | **0.897** | 0.400 | **0.762** | 0.400 |
| RAH-maxmin | 0.900 | 0.700 | **0.423** | 0.674 | **0.423** |

**Lessons:**
1. RAH-sum recovers **50% more total retention than damage-proportional** (0.286 vs 0.191) and 26% more than uniform, simply by not pouring tokens into an unrecoverable capability.
2. Damage-proportional is *worse than uniform* here. Allocating by damage is not just suboptimal — it can actively misallocate.
3. RAH-maxmin shows the cost of fairness: lifting the worst capability by 0.023 costs 0.088 of mean retention. RAH makes this trade-off *explicit and quantified*; heuristics hide it.
4. The example's numbers are illustrative. Whether real capabilities look like this is exactly H1 (RQ2), tested at Gate G1.

---

## 8. Experimental design

### 8.1 Models and device assignment

| Model | Blocks | Main pruning (≈25%) | Light pruning (≈12.5%, diagnosis only) | Device |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 28 | remove 7 | remove 4 | RTX 4070 laptop |
| Llama-3.2-1B-Instruct | 16 | remove 4 | remove 2 | RTX 4060 laptop |

- Gemma-2-2B-it is deferred: an optional replication only if time remains.
- Llama is loaded from `unsloth/Llama-3.2-1B-Instruct`, an ungated public mirror of the official `meta-llama` checkpoint (same architecture: 16 blocks, $d=2048$). No Hugging Face account is needed, and the mirror is disclosed in the paper. Qwen is loaded from the official `Qwen/Qwen2.5-1.5B-Instruct` repository, which is not gated.
- Blocks are selected by BI on a general calibration set (shared code from the LB-BI paper).
- **The pruning level is confirmed in week 2** so that no capability sits at its chance floor. This is the main lesson of our previous paper, where zero-shot Belebele collapsed to chance and became uninformative.

### 8.2 Pruning protocol

1. Compute BI on 200 calibration chunks of general text.
2. Remove the $k$ lowest-BI blocks permanently.
3. Re-index the attention layer indices and the config, so that KV-cache generation stays correct.

A pruned model is fully specified by (model ID, removed indices). Adapters are stored separately.

### 8.3 Evaluation suite

Items are fixed and seeded subsets, shared by every checkpoint of a model. Generation is greedy with capped lengths.

| Capability | Benchmark | Full eval | Pilot eval | Max new tokens | Scoring |
|---|---|---|---|---|---|
| fmt | JSONSchemaBench subset (arXiv 2501.10868 [?]) | 150 | 60 | 256 | % outputs that parse and validate against the schema |
| tool | BFCL v3 simple + multiple (Gorilla/BFCL) | 200 | 60 | 128 | AST match: function name + required args within accepted values |
| inst | IFEval (arXiv 2311.07911) | 300 | 60 | 384 | Strict prompt-level accuracy (official instruction checkers) |
| math | GSM8K test (arXiv 2110.14168) | 200 | 60 | 256 | Exact match of final number |
| safe | XSTest (arXiv 2308.01263): 100 unsafe + 100 safe | 200 | 60 | 64 | Refusal rate on unsafe (higher is better) and compliance on safe (1 − over-refusal); string-match refusal classifier |
| know | MMLU (arXiv 2009.03300) | 300 | 60 | — (log-likelihood) | Accuracy; ECE (15 bins) as a side metric |

**Prompting.** We use each model's own chat template. Tool and JSON tasks use a fixed, model-agnostic instruction format (function schemas in the system prompt, answer as JSON), so methods are compared on identical inputs. Absolute scores may be lower than leaderboard numbers. That is irrelevant to our retention-based comparisons, but we state it.

### 8.4 Healing pools

Each pool has about 1,000 prompts. Targets are regenerated by $M_0$ (greedy), so each model family gets its own teacher data.

| Pool | Source (licence and access verified in week 1) | Disjointness from evaluation |
|---|---|---|
| tool | glaiveai/glaive-function-calling-v2 (system function schemas + first user turn) | Different source from BFCL |
| fmt | JSON-instance tasks generated from the glaive function parameter schemas | Different source from JSONSchemaBench |
| inst | Synthetic verifiable-constraint prompts (Alpaca instructions + constraints such as "all lowercase", "exactly N bullets", "end with phrase X") | Constraint *types* overlap IFEval by design; prompts do not |
| math | GSM8K **train** split | Test split used for evaluation |
| safe | PKU-SafeRLHF harmful prompts + benign Alpaca look-alikes | Not XSTest |
| gen | Alpaca-cleaned | — |

### 8.5 Methods (equal budget $B=1$M tokens, shared training recipe)

| # | Method | Mixture | Scope | Seeds |
|---|---|---|---|---|
| 1 | Uniform | $B/|P|$ per pool | all-LoRA | 3 |
| 2 | Damage-proportional (PASER-style rule) | $\propto d_c$ | all-LoRA | 3 |
| 3 | **RAH-sum** | Optimised | Optimised | 3 |
| 4 | RAH-maxmin | Optimised (max-min) | Optimised | 1 |
| 5 | RAH-proxy | Proxy-optimised | all-LoRA | 1 |
| 6 | Last-k + uniform (2411.15558 recipe) | Uniform | last-k | 1 |
| 7 | Linear patch (training-free; LinearPatch / Ghosted-style ridge map) | — | Patch | 1 (deterministic) |

Optional if compute allows: Sheared-LLaMA-style loss-gap reweighting.

**Shared training recipe:**
- LoRA $r=16$, $\alpha=32$, on q, k, v, o, gate, up and down projections where LoRA is used;
- learning rate $2\times10^{-4}$ (LoRA) or $2\times10^{-5}$ (full-rank last-k);
- AdamW, 3% warm-up then cosine decay;
- sequence length 512;
- loss on response tokens only;
- bf16 with gradient checkpointing.

### 8.6 Ablations and budget sweep (Qwen only, 1 seed)

- RAH without the transfer matrix ($T=I$).
- RAH without scope selection (all-LoRA fixed).
- Power-law instead of exponential curve.
- Budget sweep: RAH vs Uniform at $B\in\{0.25\mathrm{M}, 4\mathrm{M}\}$, in addition to the main 1M, giving a budget–retention Pareto plot.

### 8.7 Metrics

- **Primary:** mean retention $\bar r = \frac{1}{|C|}\sum_c r_c$, and worst-case retention $\min_c r_c$.
- **Efficiency:** retention gained per 1M healing tokens.
- **Per-capability:** retention for each $c$.
- **Side effects:** ECE on MMLU; over-refusal on XSTest-safe; JSON validity.
- **Diagnosis:** damage $d_c$ at two pruning levels; breakage ranks.

### 8.8 Statistical tests

- **Item-level paired bootstrap** (2,000 resamples): CIs for each method's retention and for differences between RAH and each baseline. Items are resampled identically for both methods; seeds are pooled by averaging per-item scores across seeds.
- **Paired tests:** McNemar or paired permutation on binary item scores, Holm-corrected across capabilities.
- **H1:** Spearman $\rho(d_c, a_c)$ with a bootstrap CI over pilot items.
- **RQ1:** Kendall's $W$ across families.
- **Seed variability:** mean ± std over 3 seeds for methods 1–3.

### 8.9 Threats to validity

| Threat | Description | Mitigation |
|---|---|---|
| Internal: pilot-scale extrapolation | Curves fitted at 100–300k tokens are used to allocate 1M | Leave-one-out prediction checks; budget sweep tests extrapolation |
| Internal: evaluation-subset noise | Small subsets | Fixed items; paired tests |
| Internal: prompt-format effects | Shared format | Applied equally to all methods |
| Construct: string-match refusal | Imperfect classifier | Manual audit of 50 outputs |
| Construct: retention normalisation | Can inflate low baselines | Report absolute scores too |
| External | Two families, 1–1.5B only, BI pruning only, teacher-regenerated targets, English-only | Discussed explicitly; Gemma and other pruning criteria are future work |
| Conclusion validity | Few capabilities make H1 low-powered | H2 (the method claim) uses item-level tests |

---

## 9. Compute plan (two-laptop setup)

### 9.1 Devices

| Device | GPU | Role |
|---|---|---|
| Laptop A | RTX 4070 laptop (8 GB) | Qwen2.5-1.5B: all runs + Qwen-only ablations and budget sweep |
| Laptop B | RTX 4060 laptop (8 GB) | Llama-3.2-1B: all runs |
| Kaggle | 2× T4 (16 GB each), ~30 GPU-h/week | Backup/overflow; optional Gemma |

**Synchronisation:**
- Code lives in one git repository; the same commit runs on both laptops.
- Library versions are pinned in `requirements.txt`; seeds and data subsets are shipped in the repo.
- Results (small JSON/CSV) are committed; adapters stay out of git.
- Each run writes a completion marker, so interrupted runs resume and finished ones are skipped.

### 9.2 Unit-cost estimates

These are measured at Gate G0. The RTX 4070 laptop should be at least as fast as the 4060.

| Unit | Qwen-1.5B (4070) | Llama-1B (4060) |
|---|---|---|
| Full evaluation (6 capabilities, about 155k generated tokens) | ~7 min | ~6 min |
| Pilot evaluation | ~2 min | ~2 min |
| LoRA heal, 1M tokens | ~6 min | ~5 min |
| Pilot heal, 100–300k tokens | 1–2 min | 1–2 min |
| Teacher generation (6 pools × 1k prompts) | ~25 min | ~20 min |

### 9.3 Per-device ledger (GPU-hours)

| Phase | Laptop A (Qwen) | Laptop B (Llama) |
|---|---|---|
| Teacher generation | 0.45 | 0.4 |
| Damage diagnosis (3 full evaluations) | 0.35 | 0.3 |
| Recovery pilots (12) | 0.8 | 0.7 |
| Scope pilots (3) | 0.65 | 0.6 |
| Main comparison (13 runs) | 2.8 | 2.5 |
| Ablations (3) | 0.7 | — |
| Budget sweep (4) | 1.3 | — |
| **Core** | **≈7.1** | **≈4.5** |
| +50% buffer | ≈10.6 | ≈6.8 |

The budget sweep can move to Laptop B if Laptop A falls behind. Each laptop needs **about two overnight sessions**.

### 9.4 Timeline (4 weeks)

| Week | Laptop A (Qwen) | Laptop B (Llama) | Shared / writing | Gate |
|---|---|---|---|---|
| 1 | Environment setup; evaluation harness validated on $M_0$; teacher generation; timing benchmark | Same | Harness code, unit tests, data builders; dataset licence checks | **G0:** measured unit costs ≤ 1.5× estimates; unpruned scores sensible |
| 2 | Pruning + diagnosis; recovery pilots; scope pilots | Same | Curve fitting + optimiser; H1 analysis | **G1:** $\rho(d,a) < 0.8$ → continue; otherwise pivot |
| 3 | Main comparison (3 seeds core); ablations; budget sweep | Main comparison; proxy calibration swap | Statistics scripts; figure drafts; related-work section | **G2:** RAH ≥ best baseline on mean *or* worst-case retention |
| 4 | Reruns/buffer | Reruns/buffer | Full write-up, internal review, arXiv + IEEE Access submission | — |

### 9.5 Gates and pivot plan

- **G0 (end of week 1).** If costs exceed 1.5× the estimates, apply these levers in order:
  1. reduce full-evaluation subsets by one third;
  2. use 2 seeds for the core methods;
  3. shrink the budget sweep to one extra budget;
  4. drop method 6;
  5. drop light-pruning diagnosis.
- **G1 (end of week 2, about 4 GPU-hours spent).** If $\rho(d_c,a_c)\ge 0.8$, recoverability tracks damage and RAH cannot meaningfully beat damage-proportional allocation. **Pivot** to *"Where to heal: capability-dependent adaptation scope for depth-pruned small LLMs"*, a combined RQ1 + RQ4 paper using the diagnosis, recovery-pilot and scope-pilot data already collected. No compute is wasted.
- **G2 (end of week 3).** If RAH does not beat the best baseline, write an honest analysis paper (recovery curves, transfer, scope) and target a conference or workshop rather than IEEE Access.

---

## 10. Expected results and write-up per outcome

| Outcome | What we would see | How it is written up | Venue |
|---|---|---|---|
| **A. Full success** | Low $\rho(d,a)$; RAH-sum significantly > uniform and damage-proportional on mean retention; RAH-maxmin best on worst case; proxy close to RAH | Method paper: "most damaged ≠ most recoverable", with recipe and proxy | IEEE Access |
| **B. Method wins on one model only** | H1 holds on both, but H2 is significant on one family | Method paper with an honest heterogeneity analysis; emphasise when recoverability modelling pays off (damage profile spread) | IEEE Access (slightly weaker) |
| **C. H1 holds, H2 marginal** | Curves differ, but gains are within noise at $B=1$M | Analysis-first paper: curves + transfer + budget sweep showing gains appear at smaller budgets | IEEE Access / conference |
| **D. H1 fails (G1 pivot)** | Ceilings track damage | Scope-per-capability + breakage-ordering paper | Conference / workshop |

**Likely patterns we expect to observe**, all to be verified:
- fmt and tool damage is large at about 25% pruning but recovers quickly with own-pool data;
- math has a large damage but a low ceiling (consistent with 2602.01997);
- inst data transfers positively to fmt;
- over-refusal rises after pruning and falls with the safe + gen pools;
- ECE worsens after pruning and partially recovers.

---

## 11. Paper outline, figures and tables

1. **Introduction:** deployment cost and depth pruning; the healing-budget problem; damage vs recoverability; contributions.
2. **Related work:** depth pruning; cut repairs; recovery data and mixtures (PASER, Sheared-LLaMA, self-distillation, ShortOPD); capability-specific compression effects.
3. **Problem formulation:** retention, budget, scope.
4. **Method:** recovery model with transfer; pilots and regularised fitting; RAH-sum and RAH-maxmin; scope selection; RAH-proxy; complexity and cost.
5. **Experimental setup:** models and devices; pruning; evaluation suite; pools; baselines; training recipe; statistics.
6. **Results:**
   - 6.1 Breakage ordering (RQ1)
   - 6.2 Damage vs recoverability (RQ2)
   - 6.3 Main comparison (RQ3)
   - 6.4 Scope (RQ4)
   - 6.5 Proxy (RQ5)
   - 6.6 Ablations and budget sweep
   - 6.7 Side effects (calibration, over-refusal)
7. **Discussion:** practitioner recipe; when RAH helps most; cost analysis.
8. **Threats to validity and limitations.**
9. **Conclusion and future work:** Gemma/3B scale, other pruning criteria, on-policy KD + RAH, quantised bases.

**Figures:**
1. Pipeline diagram.
2. **Damage vs recovery-ceiling scatter** (headline figure).
3. Recovery curves per capability (pilot points plus fitted curves).
4. Transfer-matrix heatmap.
5. Retention by method (grouped bars with CIs) or radar.
6. Budget–retention Pareto plot.
7. Scope efficiency $\eta_{c,\sigma}$ heatmap.

**Tables:**
1. Models, pruning and devices.
2. Evaluation suite.
3. Damage and breakage ranks.
4. **Main results with 95% CIs.**
5. Ablations.
6. Compute cost per phase.

---

## 12. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| H1 false (recoverability ≈ damage) | Medium | High | G1 pivot plan using already-collected data |
| Recovery-curve fits unstable (2 budget points) | Medium | Medium | Ridge on transfer, bounds, leave-one-out checks; add a third pilot budget for one pool if needed |
| Evaluation too slow on Windows (no vLLM) | Medium | Medium | Batched HF generation with capped lengths; Kaggle (Linux, vLLM) overflow |
| Pruned models at chance on some capability | Medium | High | Week-2 pruning-level check; light pruning available |
| Dataset access, licence or field changes | Low–Medium | Medium | Week-1 loader checks; listed alternatives (e.g. glaive instead of xLAM) |
| Two-laptop inconsistency | Medium | Medium | Pinned requirements, same commit, environment check script, per-device timing reported |
| Competing preprint appears | Medium | High | Weekly arXiv watch ("pruned LLM recovery", "healing allocation", "recoverability"); post our preprint at submission |
| Safety regression after healing | Low–Medium | Medium | Safe pool in every mixture; over-refusal and refusal tracked as side metrics |
| Reviewer: "only 1–1.5B models" | Medium | Medium | Frame as the edge-deployment regime; optional Gemma/3B replication on Kaggle |

---

## 13. Glossary

| Term | Definition |
|---|---|
| **Adaptation scope** | Which parameters are trained during healing (last-k blocks, cut-adjacent LoRA, all-layer LoRA) |
| **Block Influence (BI)** | One minus the mean cosine similarity between a block's input and output hidden states; low BI means the block is a pruning candidate |
| **Ceiling ($a_c$)** | Maximum retention gain a capability can reach under the healing recipe |
| **Cut-adjacent** | The kept blocks immediately before and after a removed span |
| **Damage ($d_c$)** | Retention lost by pruning, $1-r_c(M_p)$ |
| **Depth pruning** | Removing whole Transformer blocks |
| **ECE** | Expected calibration error: the gap between confidence and accuracy |
| **Effective tokens ($x_c$)** | Transfer-weighted sum of pool tokens that count towards capability $c$ |
| **Healing / recovery** | Short fine-tune after pruning to restore lost ability |
| **LoRA** | Low-rank adapter fine-tuning |
| **Over-refusal** | Refusing benign requests that superficially resemble harmful ones |
| **Pilot** | A tiny healing run used only to fit recovery curves |
| **Retention ($r_c$)** | Capability score relative to the unpruned model |
| **Self-distillation** | Training targets produced by the unpruned model itself |
| **Timescale ($\tau_c$)** | Healing tokens needed to recover about 63% of the ceiling |
| **Transfer matrix ($T$)** | Effect of each pool's data on each capability |

---

## 14. References

[V] means verified during the 2026-10-05 scans; [?] means a detail is to be verified before citing. Our own prior work is listed for context.

**Depth pruning and repair**
1. X. Men et al., "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect," arXiv 2403.03853 (Findings ACL 2025). [V]
2. A. Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers," ICLR 2025, arXiv 2403.17887. [V]
3. B.-K. Kim et al., "Shortened LLaMA: Depth Pruning for LLMs with Comparison of Retraining Methods," arXiv 2402.02834. [V]
4. X. Chen et al., "Streamlining Redundant Layers to Compress Large Language Models (LLM-Streamline)," ICLR 2025, arXiv 2403.19135. [V]
5. (Lu et al.?) "Reassessing Layer Pruning in LLMs: New Insights and Methods," arXiv 2411.15558. [V for paper; authors ?]
6. Lu, Li, Xie, Yu, Xuan, Zhu, Wen, "The Structural Scalpel: Automated Contiguous Layer Pruning (CLP)," arXiv 2510.23652. [V]
7. Chen, Bai, Yuan et al., "A Simple Linear Patch Revives Layer-Pruned LLMs (LinearPatch)," NeurIPS 2025, arXiv 2505.24680. [V]
8. Yun, Jo, Karimireddy, Lee, "Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs," arXiv 2605.15491. [V]
9. "OverRep," EMNLP 2026, arXiv 2609.06974. [V]
10. "RestoreLCC," NeurIPS 2025, arXiv 2510.21834. [V]

**Recovery data, mixtures and distillation**
11. He, Yin, Zhen, Zhang, Yuan, Ma, "PASER: Post-Training Data Selection for Efficient Pruned LLM Recovery," ICLR 2026, arXiv 2502.12594. [V]
12. Thangarasa et al., "Self-Data Distillation for Recovering Quality in Pruned LLMs," arXiv 2410.09982. [V]
13. Zhang, Yuan, Lin, Lu, Han, Sun, Xu, Li, "ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation," arXiv 2607.13124. [V]
14. M. Xia et al., "Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning," ICLR 2024, arXiv 2310.06694. [V]
15. Ye, Yu, Hazra, Wang, "Damage Predicts Recovery: When Calibration Data Matters in Compressing Financial LLMs," arXiv 2609.26241. [V]
16. "Self-Distillation as a Performance Recovery Mechanism," arXiv 2604.15794. [V]

**Capability effects of compression**
17. Shrestha, Shrestha, Nepal, Kim, Ross, "On the Limits of Layer Pruning for Generative Reasoning in LLMs," arXiv 2602.01997. [V]
18. Zhang, Patil, Lange, Aleem, "What Breaks Under Pruning in Smart Homes, and When?," arXiv 2609.17515. [V]
19. Dong et al., "Can Compressed LLMs Truly Act? (ACBench)," ICML 2025, arXiv 2505.19433. [V]
20. A. Jaiswal et al., "Compressing LLMs: The Truth Is Rarely Pure and Never Simple (LLM-KICK)," ICLR 2024, arXiv 2310.01382. [V]
21. Martra, "Width Pruning Dichotomy in Llama-3.2," arXiv 2512.22671. [V]
22. "When Fewer Layers Break More Chains," arXiv 2510.22228. [?]
23. Taheri et al., "Small Reasoning Models are Instruction Followers in Function Calling," arXiv 2608.22472. [V]

**Neighbouring compression lines**
24. Chhawri, Mahadik, Rooj, "A Systematic Study of Compression Ordering for LLMs," arXiv 2511.19495. [V]
25. Liu et al., "Quantization Hurts Reasoning?," COLM 2025, arXiv 2504.04823. [V]
26. "Reasoning Models Can Be Accurately Pruned via CoT Reconstruction (RAC)," NeurIPS 2025, arXiv 2509.12464. [V]

**Fine-tuning methods**
27. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
28. T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023, arXiv 2305.14314.

**Benchmarks**
29. J. Zhou et al., "Instruction-Following Evaluation for Large Language Models (IFEval)," arXiv 2311.07911.
30. K. Cobbe et al., "Training Verifiers to Solve Math Word Problems (GSM8K)," arXiv 2110.14168.
31. D. Hendrycks et al., "Measuring Massive Multitask Language Understanding (MMLU)," ICLR 2021, arXiv 2009.03300.
32. P. Röttger et al., "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in LLMs," NAACL 2024, arXiv 2308.01263.
33. S. G. Patil et al., "Gorilla: LLM Connected with Massive APIs," arXiv 2305.15334; Berkeley Function Calling Leaderboard (BFCL v3).
34. S. Geng et al., "JSONSchemaBench / Generating Structured Outputs from Language Models: Benchmark and Studies," arXiv 2501.10868. [?]

**Models**
35. Qwen Team, "Qwen2.5 Technical Report," arXiv 2412.15115.
36. A. Dubey et al., "The Llama 3 Herd of Models," arXiv 2407.21783.

**Statistics**
37. B. Efron and R. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
38. M. G. Kendall and B. Babington Smith, "The Problem of m Rankings," *Ann. Math. Statist.*, 1939 (Kendall's W).

**Our prior work**
39. "Calibration Language Matters: Language-Balanced Depth Pruning of Small Multilingual LLMs for Hindi and Marathi" (this group, 2026; manuscript in `../paper/`). Source of the BI pruning code and lessons on evaluation floors.
