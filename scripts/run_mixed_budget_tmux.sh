#!/usr/bin/env bash
set -euo pipefail
stage="${1:-}"
case "$stage" in smoke|representative|remaining|evaluate) ;; *) echo 'Usage: bash scripts/run_mixed_budget_tmux.sh {smoke|representative|remaining|evaluate}'; exit 2;; esac
root=/ceph_data/yeonwoo/MHW_subset
study="$root/outputs/v7_gpu_mixed_budget_comparison_20261011"
python_bin=/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python
socket="$study/runtime/tmux.sock"
session="v7_${stage}"
mkdir -p "$study/logs" "$study/runtime/TMPDIR"
if tmux -S "$socket" has-session -t "$session" 2>/dev/null; then
  echo "Session exists: $session. Inspect it; no duplicate launch."; exit 1
fi
logfile="$study/logs/manager_${stage}_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S).log"
printf -v command 'cd %q && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=%q %q -B %q --stage %q > %q 2>&1' "$root" "$study/runtime/TMPDIR" "$python_bin" "$root/scripts/run_mixed_budget_stage.py" "$stage" "$logfile"
tmux -S "$socket" new-session -d -s "$session" "$command"
tmux -S "$socket" set-option -t "$session" remain-on-exit on
printf 'Started manual stage: %s\nLog: %s\nAttach: tmux -S %q attach -t %q\n' "$stage" "$logfile" "$socket" "$session"
