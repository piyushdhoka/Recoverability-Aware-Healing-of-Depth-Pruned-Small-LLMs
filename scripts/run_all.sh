#!/usr/bin/env bash
# Run the whole pipeline for one model (resumable). Usage: bash scripts/run_all.sh qwen|llama|smollm [from_stage_index]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; FROM="${2:-0}"
STAGES=(00_check_env 01_build_eval 02_build_pools 03_teacher 04_diagnose 05_pilots 06_fit_allocate 07_main)
for ((i = FROM; i < ${#STAGES[@]}; i++)); do
  echo "=== stage ${STAGES[$i]} ($MODEL) ==="
  python "scripts/${STAGES[$i]}.py" --model "$MODEL"
done
echo "Done. Commit results/$MODEL and push; then run: python scripts/08_analyze.py"
