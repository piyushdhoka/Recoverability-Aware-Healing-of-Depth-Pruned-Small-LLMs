# Publishability Assessment: RAH (Recoverability-Aware Healing) Paper

**Paper:** *Recoverability-Aware Healing: Pilot-Guided Allocation of Data Mixture and Adaptation Scope for Depth-Pruned Small Language Models*
**Assessed:** 2026-10-05, against `docs/RESEARCH_PLAN.md`
**Tags:** [V] = checked against Crossref, IEEE or the publisher during this assessment. [?] = from third-party aggregators or memory, so confirm before relying on it.

---

## 1. Bottom line

- **IEEE Access suits this paper if RAH wins.** The topic is in scope. The journal makes a binary decision in about 4 weeks. It publishes empirical LLM-efficiency work done on 7–8B models with modest compute. I found no IEEE Access paper on *healing allocation for depth-pruned LLMs*, so the topic is open there.
- **The main risk is reviewer scepticism, not the venue.** The likely objections are: "only 1–2B models", "evaluation subsets of 150–300 items", "the gains sit inside the confidence intervals", and "this is PASER with extra steps". Because the decision is binary, one unconvinced reviewer usually means a reject (with resubmission allowed).
- **Estimated acceptance at IEEE Access on first submission:** about 60–75% if RAH clearly wins, 35–50% with mixed results, and 15–30% for a pure negative result. Section 4 gives the reasoning.

---

## 2. IEEE Access profile

| Item | Finding | Source |
|---|---|---|
| Scope | Multidisciplinary and application-oriented. LLM compression, PEFT and quantization papers appear regularly (see 2.2). | [V] IEEE Access, Crossref |
| Review model | **Single-anonymous** (reviewers see the authors). At least 2 reviewers. | [V] ieeeaccess.ieee.org |
| Decision | **Binary**: accept (often with minor revisions) or reject. A rejected paper can be resubmitted as a new submission. There is no multi-round major revision. | [V] |
| Time to decision | About 4 weeks on average. About 4–6 weeks from submission to publication. A third-party estimate puts the median first decision at 3–6 weeks. | [V] official; [?] Manusights |
| Acceptance rate | Official figure is about 20% (one IEEE page says 27%). Third-party estimates of 30–45% probably exclude desk rejections. | [V] / [?] |
| APC (2026) | **US$2,160**, the only charge (full OA). IEEE society members get 20% off; IEEE members who are not society members get 5% off. **No overlength charges.** | [V] IEEE 2026 APC list (30 Jul 2026) |
| Length | No hard limit. About 10–20 pages in the double-column template is normal, and under 20 is recommended. The planned 10–12 pages fits. | [?] |
| Impact | JIF about 3.4–4.2 (Q2). Sources disagree on the latest JCR year. Indexed in Scopus, WoS and DOAJ. | [?] |
| Preprints | IEEE allows posting on arXiv before submission. After acceptance, the IEEE copyright notice must be added to the preprint. | [V] IEEE preprint policy |

### 2.1 What reviewers typically demand, and common rejection reasons

Sources: Manusights guides [?], the IEEE Access reviewer guidance [V], and the norms visible in comparable papers.

- **Baselines on recognised benchmarks.** "ML papers without baselines" is one of the most common reasons for desk rejection. Your plan has 5 baselines plus ablations, which is good.
- **Technical correctness and reproducibility:** full hyper-parameters, pseudocode and released code.
- **Ablations and error analysis.**
- **Recent citations (2025–2026).** Missing these is a frequent complaint. You must cite PASER, the 2602.01997 limits paper, the 2609.17515 smart-home paper, Ghosted Layers, ShortOPD and the Llama-3.2 width-pruning paper (2512.22671).
- **Writing quality.** Poor English is a common reason for desk rejection.
- **Wrong subject-area selection.** This is a minor cause of desk rejections. Choose the AI/ML category.
- **Statistical rigour.** It is not always required, but your setup (3 seeds, paired bootstrap, confidence intervals) is *above* the norm for IEEE Access and is a selling point. Note that Alahmari et al. (below) is an IEEE Access paper showing that QLoRA runs are not repeatable across seeds. Cite it to justify using multiple seeds.

### 2.2 Recent IEEE Access papers on LLM efficiency

All DOIs below were resolved through Crossref (ISSN 2169-3536).

| # | Title | Authors | Year | DOI | Scale / notes |
|---|---|---|---|---|---|
| 1 | Repeatability of Fine-Tuning Large Language Models Illustrated Using QLoRA | S. S. Alahmari, L. O. Hall, P. R. Mouton, D. B. Goldgof | 2024 | 10.1109/ACCESS.2024.3470850 [V] | 4 LLMs × 7 runs on **one GPU**, 2 datasets. Shows small-compute empirical work gets in. |
| 2 | Activation-Guided Low-Rank Parameter Adaptation for Efficient Model Fine-Tuning | Q. Wang, S. Shen | 2025 | 10.1109/ACCESS.2025.3533701 [V] | A LoRA variant. Model scale not verified [?]. |
| 3 | Layer Pruning With Consensus: A Triple-Win Solution | L. G. Mugnaini, C. T. Duarte, A. H. R. Costa, A. Jordao | 2025 | 10.1109/ACCESS.2025.3601042 [V] | Layer (depth) pruning, CNN-focused. Precedent for depth-pruning work. |
| 4 | Entropy-Based Data Selection for Language Models | H. Li, Y. Liu, C. Huang | 2025 | 10.1109/ACCESS.2025.3605290 [V] | Data selection for fine-tuning under limited resources. Closest in *spirit* to RAH's allocation. |
| 5 | Sub 4-bit Power-of-Two-Based Mixed-Precision Quantization for Efficient LLM Compression and Acceleration | H. Cho, A. P. Padhy, F. Camacho, S. Mukhopadhyay | 2025 | 10.1109/ACCESS.2025.3625771 [V] | Quantization. Scale not verified [?]. |
| 6 | Make Large Language Models Efficient: A Review | A. Mussa, Z. Tuimebayev, M. Mansurova | 2025 | 10.1109/ACCESS.2025.3605110 [V] | Survey covering compression, PEFT and on-device inference. |
| 7 | Investigating the Impact of Quantization Methods on the Safety and Reliability of Large Language Models | A. Kharinaev, V. Moskvoretskii, E. Shvetsov, K. Studenikina, M. Bykov, E. Burnaev | 2026 | 10.1109/ACCESS.2026.3703899 [V] | **Llama-3.1-8B-Instruct (plus an abliterated variant) and Mistral-7B**. 66 quantized variants. About 120 GPU-h on A40/T4. |
| 8 | UDP: Up-or-Down Precision Quantization for LLMs via Evolutionary Search | A.-T. Mai, T.-S. Pham, X. T. Nguyen, T.-T. Dao | 2026 | 10.1109/ACCESS.2026.3680045 [V] | Mixed-precision search. |
| 9 | GroupLoRA: Enhancing Rank Effectiveness Through Group-Wise Decomposition for Low-Rank Adaptation | J. Jun, Y. Ro | 2026 | 10.1109/ACCESS.2026.3671790 [V] | PEFT evaluated on LLMs and VLMs. |
| 10 | Sensitivity-Guided Mixed-Precision Quantization-Aware Training for LLMs | G. K. Mooppil, A. Sasikumar, A. Mathur, S. Vairavasundaram | 2026 | 10.1109/ACCESS.2026.3698449 [V] | QAT. |
| 11 | A Hardware-Aware Efficient LLM for Multimedia Understanding via Dynamic Rank Adaptation and Structured Pruning | R. Yang, L. Liang | 2026 | 10.1109/ACCESS.2026.3654646 [V] | Pruning combined with rank adaptation. |

**Is a scale of 1–2B models with 2 families acceptable?** Mostly yes, but you have to argue for it. The typical IEEE Access LLM-efficiency paper uses 2–4 models at the 7–8B scale and often runs on a single GPU or a small cluster. Papers on 1–3B models are common in application papers, and there is a growing small-language-model literature in IEEE Access. Framing helps: present "edge / consumer-GPU deployment of SLMs" as the *motivation*, not as an apology. Still, at least one reviewer will probably ask "does this hold at 3B/7B?". A single 3B replication (Qwen2.5-3B or Llama-3.2-3B) defuses that question far more cheaply than a third 1–2B family does.

I did **not** find any IEEE Access paper on depth-pruning *healing* or *recovery allocation*. That means the topic is new there, but reviewers may be generalists, so the paper must explain PASER and the healing problem from first principles.

---

## 3. Alternative venues of comparable level

JIF figures come from aggregators [?] and should be treated as approximate. Timelines are typical first-decision times.

| Venue | Fit for RAH | Difficulty vs IEEE Access | Timeline | Cost | Impact (approx.) |
|---|---|---|---|---|---|
| **IEEE Access** | Good | Baseline | About 4–6 weeks; binary decision | $2,160 | JIF about 3.6–4.2, Q2 |
| IEEE Trans. on Artificial Intelligence (TAI) | Good: AI methods | Higher; multi-round review | 3–6+ months | Free unless OA ($2,800); 10 pages regular, then $200/page [V] | JIF [?], strong IEEE brand |
| IEEE TNNLS | Moderate | **Much higher.** 1–2B scale and a heuristic optimiser would likely fall short | 4–8 months | Free / OA $2,800 | Q1, JIF about 10 [?] |
| IEEE/CAA JAS | Weak (control/automation focus) | Very high | Long | — | Q1 |
| Neurocomputing (Elsevier) | Good | Somewhat higher; values method plus ablations | 3–6 months | Free (subscription) or OA about $2,470 [?] | JIF about 5.5–6.5, Q1 [?] |
| Expert Systems with Applications | Moderate; needs an "application/system" framing | Higher, high desk-reject rate | 3–6 months | Free or OA about $3,220 [?] | JIF about 7.5, Q1 [?] |
| Knowledge-Based Systems | Moderate | Higher | 3–6 months | Free or OA about $3,130 [?] | JIF about 7.6, Q1 [?] |
| Information Sciences | Weak to moderate | Higher | 4–8 months | Free or OA | Q1 [?] |
| Applied Intelligence (Springer) | Good | Similar to slightly higher | 3–8 months (slow) | Free (subscription) | JIF about 3.5, Q2 [?] |
| Neural Networks (Elsevier) | Moderate | Higher; expects more theory | 4–8 months | Free or OA | JIF about 6.3, Q1 [?] |
| Pattern Recognition Letters | Weak; short-paper format | Similar | 2–4 months | Free or OA | JIF about 3.3 [?] |
| ACM TIST | Moderate | Higher, slow | 6+ months | Free or OA | High JIF [?] |
| **TMLR** | **Very good**, especially for mixed or negative results: it accepts based on whether claims are supported by evidence, not on novelty or SOTA | Similar rigour bar, but no "beat SOTA" requirement | About 2–4 months, OpenReview | Free | Not JCR-indexed, but well respected in ML |
| ACL/EMNLP **Findings** (via ARR) | Very good on topic | Higher; reviewers know PASER and ShortGPT well | ARR cycle about 2 months. Last ARR window for ACL 2027 is about Jan 2027 [V] | Registration plus travel | High in NLP community, not JCR |
| Workshops (ENLSP, WANT, SLLM, efficient-ML) | Very good | Lower | NeurIPS 2026 workshop deadlines (about 29 Aug) have **passed** [V]. Next realistic: ICLR/ACL 2027 workshops (about Feb–May 2027) | Registration | Low archival weight. Usually non-archival, so it can be combined with a journal |

**Take-aways:**
- TNNLS and IEEE/CAA JAS are too high for this scope.
- Neurocomputing and Applied Intelligence are the realistic "same level or slightly above" journals, and they are free on the subscription route.
- TMLR is the best home if results are mixed or negative.
- Findings carries the most prestige among peers, but the paper will be judged harder against PASER.

---

## 4. Verdict: acceptance probability at IEEE Access

These are subjective estimates, based on the venue's norms and the planned design.

| Scenario | Definition | First-submission acceptance | Within 2 submissions | Comment |
|---|---|---|---|---|
| **(a) RAH wins** | RAH-sum beats Uniform and PASER-style allocation on mean *and/or* worst-case retention, with paired-bootstrap significance on both families, and H1 holds (damage–ceiling ρ < 0.5) | **60–75%** | 80–90% | Clear method, clear baselines and statistics above the norm. The remaining risk is a "small models / small eval" reviewer. |
| **(b) Mixed** | RAH wins on one family or one metric, or gains overlap the confidence intervals, while H1/H3 findings are solid | **35–50%** | 55–65% | Reframe as an analysis paper: "damage ≠ recoverability; scope matters per capability". TMLR is a better fit. |
| **(c) Negative** | RAH does no better than PASER-style allocation; only the diagnosis and curves remain | **15–30%** | 30–40% | IEEE Access reviewers rarely reward negative method results. Go to TMLR or a workshop instead, as planned at Gate G2. |

### 4.1 Additions that most raise the odds, ranked by gain per GPU-hour

1. **One 3B-scale replication** (Qwen2.5-3B-Instruct; RAH vs Uniform vs PASER; 1–2 seeds). This directly answers the most likely objection. It probably needs 4–6 GPU-h on Kaggle, which is more than Gemma-2-2B, but it is worth more.
2. **Full benchmarks for the final comparison** (full GSM8K test of 1,319 items, full IFEval of 541 items, at least 500 BFCL items), even if pilots stay on subsets. Subsets are acceptable for pilots but invite "cherry-picked items" criticism for headline numbers. A cheaper alternative: report statistical power and minimum detectable effect for each subset.
3. **Open-source release** of the code, per-item predictions, pilot curves and LoRA adapters, plus a reproducibility checklist. This is low cost and appeals strongly to IEEE Access reviewers.
4. **A stronger PASER baseline.** Implement PASER's actual clustering and allocation, or state precisely what "PASER-style" keeps and drops. A straw-man baseline is the most likely technical reject reason from an expert reviewer.
5. **A second pruning criterion** (for example, angular distance or a Shortened-LLaMA PPL criterion) on one model. This removes a stated threat to validity.
6. **Light theoretical analysis.** State that the allocation is concave with a KKT/water-filling closed form, and bound when damage-proportional allocation is optimal (only when a_c ∝ damage and τ_c is equal across capabilities). This turns H1 into a formal reason why RAH should help, and it costs no compute.
7. **A third 1–2B family (Gemma-2-2B).** This is useful but the least valuable of these additions; it is less valuable than a 3B run.
8. **Wall-clock and cost accounting**, including pilot overhead versus the gains. Reviewers will ask whether the pilots cost more than they save.

### 4.2 Risks

| Risk | Severity | Note |
|---|---|---|
| A competing preprint on capability- or recoverability-aware healing for *depth*-pruned SLMs | Medium | None found today. Related work: PASER; Llama-3.2 width-pruning dichotomy (2512.22671, which covers width pruning and has no healing); Task-specific pruning limits (2604.27115, 1.5B/7B, fine-tuning recovers); Free Lunch retraining (2510.14444). Keep checking weekly and post on arXiv at submission. |
| Small-model scope | Medium–High | Mitigate with the 3B replication and the edge-deployment framing. |
| Evaluation subsets and noise | Medium | Fixed items, paired tests, and full benchmarks for final numbers. |
| "Incremental over PASER" | Medium | Lead with H1 (the damage–recoverability scatter) and joint scope selection. Do not claim that capability-aware selection is new. |
| Single-anonymous review plus binary decision | Low–Medium | A single sceptical reviewer can sink the paper. Pre-empt the objections in a "Threats to validity" section and a limitations table. |
| String-match safety classifier | Low–Medium | Validate it against a small LLM-judge or human sample (κ). |

---

## 5. Recommended submission strategy

1. **Gate-dependent routing:**
   - Gate G2 passes (scenario a): **IEEE Access is the primary venue.** It is fast and binary, the APC is $2,160, and the planned 10–12 pages fits.
   - Scenario (b): **TMLR first.** It judges claims against evidence and gives OpenReview visibility. Fall back to IEEE Access with an analysis-first framing.
   - Scenario (c): **TMLR or a 2027 efficiency workshop.** Do not spend an IEEE Access APC attempt on it.
2. **Backup venues:** **Neurocomputing** (Q1, free on the subscription route, accepts an empirical-method paper with ablations) or **Applied Intelligence**. If you are willing to accept harsher expert review in exchange for prestige, consider **ARR → ACL 2027 Findings** (last window about Jan 2027).
3. **arXiv timing:** post the preprint **the same day you submit**. IEEE allows this. It establishes priority against competing preprints, which the plan rates as a medium risk. After acceptance, add the IEEE copyright notice. Release the code and adapters with the preprint.
4. **Before submitting:**
   - Add the 3B replication and full-benchmark final evaluation if time allows. This adds roughly a week.
   - Make sure the related-work section cites all 2025–2026 depth-pruning and recovery work.
   - Use the IEEE Access LaTeX template.
   - Choose the AI/ML subject area.
   - Have the English proofread.
5. **If IEEE Access rejects:** address the reviews and either resubmit (resubmission is allowed and common) or move to Neurocomputing within 2 weeks.

---

## Sources

- IEEE Access, Rapid Peer Review: https://ieeeaccess.ieee.org/about/rapid-peer-review/
- IEEE Access, Stages of Peer Review: https://ieeeaccess.ieee.org/authors/stages-of-peer-review/
- IEEE 2026 APC list (updated 30 Jul 2026): https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/2026-07-30-IEEE-Article-Processing-Charges-List.pdf
- IEEE Open, article processing charges: https://open.ieee.org/for-authors/article-processing-charges/
- IEEE Preprint Policy: https://cis.ieee.org/images/files/Publications/IEEE_Preprint_Policy.pdf
- Manusights, IEEE Access readiness, review time and APC: https://manusights.com/blog/is-my-paper-ready-for-ieee-access ; https://manusights.com/blog/ieee-access-review-time ; https://manusights.com/blog/ieee-access-apc-open-access
- Crossref API (IEEE Access ISSN 2169-3536) for DOI verification: https://api.crossref.org/works
- DOAJ record for Alahmari et al.: https://doaj.org/article/e050081bcadb46528f36ff3f5bd3dd35
- Kharinaev et al. (arXiv version): https://arxiv.org/abs/2502.15799
- Entropy-Based Data Selection (arXiv version): https://arxiv.org/abs/2602.17465
- Layer Pruning with Consensus (arXiv version): https://arxiv.org/abs/2411.14345
- PASER (ICLR 2026): https://iclr.cc/virtual/2026/poster/10007567 ; https://arxiv.org/abs/2502.12594
- Width pruning dichotomy in Llama-3.2: https://arxiv.org/abs/2512.22671
- Limits of pruning in task-specific LLMs: https://arxiv.org/abs/2604.27115
- A Free Lunch in LLM Compression: https://arxiv.org/abs/2510.14444
- TMLR acceptance criteria: https://jmlr.org/tmlr/acceptance-criteria.html
- ARR / ACL 2027 dates: https://www.opencurious.com/ai-conference-deadlines/acl-2027
- NeurIPS 2026 workshop dates: https://neurips.cc/Conferences/2026/WorkshopsGuidance
- Journal metrics (approximate JIFs): https://www.journalmetrics.org/journal/neurocomputing ; https://www.journalmetrics.org/journal/expert-systems-with-applications ; https://www.journalmetrics.org/journal/knowledge-based-systems ; https://www.journalmetrics.org/journal/applied-intelligence ; https://www.journalmetrics.org/journal/neural-networks ; https://www.scijournal.org/ieee-transactions-on-artificial-intelligence
