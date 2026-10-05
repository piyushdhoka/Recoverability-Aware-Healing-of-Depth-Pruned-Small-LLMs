#!/usr/bin/env bash
# Run the whole pipeline for one model (resumable). Usage: bash scripts/run_all.sh qwen|llama|smollm [from_stage_index]
# One heal job per python process (GPU memory fragments across jobs); exit code 3 = restart me.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; FROM="${2:-0}"
export RAH_MAX_JOBS="${RAH_MAX_JOBS:-1}" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
STAGES=(00_check_env 01_build_eval 02_build_pools 03_teacher 04_diagnose 05_pilots 06_fit_allocate 07_main)
for ((i = FROM; i < ${#STAGES[@]}; i++)); do
  failures=0
  while true; do
    echo "=== stage ${STAGES[$i]} ($MODEL) $(date +%H:%M:%S) ==="
    python "scripts/${STAGES[$i]}.py" --model "$MODEL"; code=$?
    [ $code -eq 0 ] && break
    [ $code -eq 3 ] && continue
    failures=$((failures + 1))
    if [ $failures -ge 3 ]; then echo "STAGE FAILED: ${STAGES[$i]} (exit $code)"; exit 1; fi
    echo "stage ${STAGES[$i]} exited with $code; retrying ($failures/3) ..."
  done
done
echo "Done. Commit results/$MODEL and push; then run: python scripts/08_analyze.py"
