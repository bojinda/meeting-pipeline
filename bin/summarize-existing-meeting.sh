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
NOTES=0
NOTES_OUTPUT_SET=0
NOTES_CONTEXT_SET=0
NOTES_TOKENIZER_SET=0
NOTES_MODE_SET=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dual-gpu-target)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then echo "ERROR: named dual target required" >&2; exit 64; fi
      DUAL_TARGET="$2"; shift 2 ;;
    --meeting-notes)
      NOTES=1; SUMMARY_OPTIONS+=("$1"); shift ;;
    --meeting-notes-only|--no-meeting-notes-only)
      NOTES_MODE_SET=1; SUMMARY_OPTIONS+=("$1"); shift ;;
    --keep-recap|--no-keep-recap)
      SUMMARY_OPTIONS+=("$1"); shift ;;
    --speaker-aliases|--map-model|--reduce-model|--map-num-ctx|--reduce-num-ctx|--keep-alive|--temperature|--ollama-url|--chunk-plan|--chunk-comparison-dir|--chunk-tokenizer|--meeting-notes-output-dir|--meeting-notes-tokenizer|--meeting-notes-thinking|--meeting-notes-checkpoint|--meeting-notes-checkpoint-sha256)
      if [ "$#" -lt 2 ]; then echo "ERROR: missing summary option value" >&2; exit 64; fi
      case "$1" in
        --meeting-notes-output-dir) NOTES_OUTPUT_SET=1 ;;
        --reduce-num-ctx) NOTES_CONTEXT_SET=1 ;;
        --meeting-notes-tokenizer) NOTES_TOKENIZER_SET=1 ;;
      esac
      SUMMARY_OPTIONS+=("$1" "$2"); shift 2 ;;
    *) echo "ERROR: unsupported historical summary option; map/reduce only" >&2; exit 64 ;;
  esac
done
if [ "$NOTES" = 1 ]; then
  umask 077
  export MEETING_MAP_MODEL="${MEETING_MAP_MODEL:-qwen3.8:27b}"
  export MEETING_REDUCE_MODEL="${MEETING_REDUCE_MODEL:-qwen3.8:27b}"
  export MEETING_MAP_NUM_CTX="${MEETING_MAP_NUM_CTX:-16384}"
  # The generic reduce context may be 32768; notes use the established 196608.
  # Explicit CLI context/model/tokenizer/thinking choices retain precedence.
  if [ "$NOTES_CONTEXT_SET" = 0 ]; then SUMMARY_OPTIONS+=(--reduce-num-ctx 196608); fi
  if [ "$NOTES_TOKENIZER_SET" = 0 ]; then
    SUMMARY_OPTIONS+=(--meeting-notes-tokenizer "${MEETING_NOTES_TOKENIZER:-$HOME/.cache/meeting-tokenizers/qwen3.5-27b/tokenizer.json}")
  fi
  if [ "$NOTES_MODE_SET" = 0 ]; then SUMMARY_OPTIONS+=(--meeting-notes-only); fi
  if [ "$NOTES_OUTPUT_SET" = 0 ]; then
    NOTES_ROOT="${MEETING_NOTES_ROOT:-$BASE_DIR/ignore/meeting-notes}"
    mkdir -p "$NOTES_ROOT"
    NOTES_RUN="$(mktemp -d "$NOTES_ROOT/$(basename "$TRANSCRIPT_DIR").XXXXXX")"
    SUMMARY_OPTIONS+=(--meeting-notes-output-dir "$NOTES_RUN/notes")
  fi
fi
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
