#!/usr/bin/env bash
# Optional private planner uses the existing GPU1 runner; deterministic mode is CPU-only.
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${MEETING_CONFIG_FILE:-$BASE_DIR/config/.env}"
if [ -f "$CONFIG_FILE" ]; then set -a; source "$CONFIG_FILE"; set +a; fi
PLAN_ARGS=("$@")
USE_GPU=0
PREFLIGHT=0
for arg in "${PLAN_ARGS[@]}"; do
  if [ "$arg" = --planner ]; then USE_GPU=1; fi
  if [ "$arg" = --preflight ]; then PREFLIGHT=1; fi
done
PYTHON="${MEETING_SUMMARY_PYTHON:-${AIHUB_GPU_RUNNER_PYTHON:-python3}}"
if [ "$USE_GPU" = 1 ] && [ "$PREFLIGHT" = 0 ]; then
  export AIHUB_GPU_STAGE_INPUT_FILES
  AIHUB_GPU_STAGE_INPUT_FILES="$("$PYTHON" "$BASE_DIR/bin/meeting_stage_inputs.py" "$PYTHON" "$BASE_DIR/bin/meeting_chunk_experiment.py" "${PLAN_ARGS[@]}")"
  source "$BASE_DIR/bin/with-gpu-lock.sh"
  AIHUB_GPU_STAGE_INPUT="${1}/chunks_out/transcript_chunks.jsonl" AIHUB_GPU_STAGE_SETTINGS_FILE="$CONFIG_FILE" \
    aihub_run_gpu_stage gpu1 "Optional meeting boundary planner" "$PYTHON" "$BASE_DIR/bin/meeting_chunk_experiment.py" "${PLAN_ARGS[@]}"
else
  exec "$PYTHON" "$BASE_DIR/bin/meeting_chunk_experiment.py" "${PLAN_ARGS[@]}"
fi
