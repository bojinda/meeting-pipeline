#!/usr/bin/env bash
# Historical map/reduce: existing index -> one single or explicit dual GPU stage.
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${MEETING_CONFIG_FILE:-$BASE_DIR/config/.env}"
HISTORICAL_OUTPUT_OVERRIDE="${MEETING_SUMMARIES_ROOT:-}"
if [ -f "$CONFIG_FILE" ]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi
if [ -n "$HISTORICAL_OUTPUT_OVERRIDE" ]; then
  export MEETING_SUMMARIES_ROOT="$HISTORICAL_OUTPUT_OVERRIDE"
fi
if [ "$#" -lt 1 ]; then
  echo "Usage: bash summarize-existing-meeting.sh TRANSCRIPT_DIR [map/reduce options]" >&2
  exit 64
fi
TRANSCRIPT_DIR="$1"
shift
if [ ! -s "$TRANSCRIPT_DIR/chunks_out/transcript_chunks.jsonl" ]; then
  echo "ERROR: an existing nonempty transcript chunk index is required" >&2
  exit 66
fi
SUMMARY_OPTIONS=()
DUAL_TARGET=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dual-gpu-target)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then echo "ERROR: named dual target required" >&2; exit 64; fi
      DUAL_TARGET="$2"; shift 2 ;;
    --keep-recap|--no-keep-recap|--meeting-notes)
      SUMMARY_OPTIONS+=("$1"); shift ;;
    --speaker-aliases|--map-model|--reduce-model|--map-num-ctx|--reduce-num-ctx|--keep-alive|--temperature|--ollama-url|--chunk-plan|--chunk-comparison-dir|--chunk-tokenizer|--meeting-notes-output-dir|--meeting-notes-tokenizer)
      if [ "$#" -lt 2 ]; then echo "ERROR: missing summary option value" >&2; exit 64; fi
      SUMMARY_OPTIONS+=("$1" "$2"); shift 2 ;;
    *) echo "ERROR: unsupported historical summary option; map/reduce only" >&2; exit 64 ;;
  esac
done
SUMMARY_RESOURCE=gpu1
if [ -n "$DUAL_TARGET" ]; then
  SUMMARY_RESOURCE=gpu0+gpu1
  export AIHUB_GPU_OLLAMA_TARGET="$DUAL_TARGET"
fi
source "$BASE_DIR/bin/with-gpu-lock.sh"
AIHUB_GPU_STAGE_INPUT="$TRANSCRIPT_DIR/chunks_out/transcript_chunks.jsonl" \
AIHUB_GPU_STAGE_SETTINGS_FILE="$CONFIG_FILE" \
aihub_run_gpu_stage "$SUMMARY_RESOURCE" "Historical meeting map/reduce ($SUMMARY_RESOURCE)" \
  "${MEETING_SUMMARY_PYTHON:-${AIHUB_GPU_RUNNER_PYTHON:-python3}}" \
  "$BASE_DIR/bin/ollama_meeting_summary.py" "$TRANSCRIPT_DIR" "${SUMMARY_OPTIONS[@]}"
