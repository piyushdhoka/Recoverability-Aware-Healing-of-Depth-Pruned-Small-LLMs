# RAH: Setup and Run Instructions

Two models run on two laptops at the same time.

| Model | Laptop | Config name |
|---|---|---|
| **Qwen2.5-1.5B-Instruct** | **Other laptop (RTX 4070)** | `qwen` |
| **Llama-3.2-1B-Instruct** | **This laptop (RTX 4060)** | `llama` |

Never run `qwen` on this laptop or `llama` on the other laptop.

---

## 1. Setup (do this on BOTH laptops)

Requirements: Windows 10/11, an NVIDIA driver supporting CUDA 12.1+, Python 3.12, Git, and about 15 GB of free disk space.

Open **PowerShell** in a folder **outside OneDrive** (for example `C:\work`). Virtual environments and model caches should not sync to OneDrive.

```powershell
# 1. Get the code
git clone <REPO_URL> rah-healing
cd rah-healing

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch (CUDA 12.1), then the pinned requirements
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Check that the GPU is visible (must print True)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 5. Check that every dataset is reachable (must end with "10/10 dataset checks passed")
python scripts/check_datasets.py
```

If `Activate.ps1` is blocked, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## 2. Qwen — run on the OTHER laptop (RTX 4070)

```powershell
cd rah-healing
.\.venv\Scripts\Activate.ps1
git pull

.\scripts\run_all.ps1 -Model qwen
```

- It runs all stages in order (00 → 07) and takes roughly 6–8 hours in total. Leaving it overnight is fine.
- **If it stops** (crash, sleep, power cut), run the same command again. Finished runs are skipped automatically.
- To restart from a specific stage, for example stage 5:
  ```powershell
  .\scripts\run_all.ps1 -Model qwen -From 5
  ```
- Keep the laptop **plugged in** and **disable sleep** while it runs.

**When it finishes, send the results back:**

```powershell
git add results/qwen
git commit -m "Qwen results"
git push
```

---

## 3. Llama — run on THIS laptop (RTX 4060)

**One-time step:** Llama is a gated model. On huggingface.co, open `meta-llama/Llama-3.2-1B-Instruct`, accept the licence, then log in:

```powershell
huggingface-cli login
```

Then run:

```powershell
cd rah-healing
.\.venv\Scripts\Activate.ps1
git pull

.\scripts\run_all.ps1 -Model llama
```

- To resume after a stop, run the same command again. To restart from a stage, add `-From <n>`:
  ```powershell
  .\scripts\run_all.ps1 -Model llama -From 5
  ```

**When it finishes:**

```powershell
git add results/llama
git commit -m "Llama results"
git push
```

---

## 4. After BOTH models finish (on either laptop)

```powershell
git pull

# The zero-cost proxy method needs the other model's results, so re-run stages 6–7 once per model
python scripts/06_fit_allocate.py --model llama
python scripts/07_main.py --model llama --only rah_proxy
#   (on the RTX 4070 laptop: the same two commands with --model qwen)

# Build tables and figures for the paper
python scripts/08_analyze.py
```

Output: `results/analysis/` (CSV/Markdown tables and PDF figures).

---

## 5. Important rules

- **The data is shared.** Stages 1 and 2 build the evaluation and healing data. Run them on one laptop, commit `data/`, and `git pull` on the other before starting. Both models must use identical items.
  - On this laptop: `python scripts/01_build_eval.py --model llama` and `python scripts/02_build_pools.py --model llama`
  - Then: `git add data` → `git commit -m "data"` → `git push`
- **Do not change the library versions** in `requirements.txt` on only one laptop.
- **Do not edit `configs/`** during a run.
- `artifacts/` (teacher data, adapters) stays local and is not committed.
