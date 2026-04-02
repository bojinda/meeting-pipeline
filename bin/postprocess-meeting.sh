#!/usr/bin/env bash
set -euo pipefail

INPUT="$1"
BASE_DIR="$HOME/meeting-pipeline"
CONFIG_FILE="$BASE_DIR/config/.env"
OUT_ROOT="$BASE_DIR/meeting-transcripts"
LOG_ROOT="$BASE_DIR/meeting-summaries"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate whisperx

if [ -f "$CONFIG_FILE" ]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi

: "${WHISPERX_MODEL:=large-v2}"
: "${WHISPERX_BATCH_SIZE:=4}"
: "${WHISPERX_COMPUTE_TYPE:=float16}"
: "${WHISPERX_DEVICE:=cuda}"
: "${WHISPERX_LANGUAGE:=en}"

BASE="$(basename "$INPUT" .wav)"
OUTDIR="$OUT_ROOT/$BASE"
mkdir -p "$OUTDIR" "$LOG_ROOT"

LOGFILE="$OUTDIR/whisperx.log"
STATUSFILE="$OUTDIR/status.txt"

{
  echo "Started: $(date -Is)"
  echo "Input: $INPUT"
  echo "Output dir: $OUTDIR"
  echo "Model: $WHISPERX_MODEL"
  echo "Batch size: $WHISPERX_BATCH_SIZE"
  echo "Compute type: $WHISPERX_COMPUTE_TYPE"
  echo "Device: $WHISPERX_DEVICE"
} > "$STATUSFILE"

CMD=(
  whisperx "$INPUT"
  --model "$WHISPERX_MODEL"
  --batch_size "$WHISPERX_BATCH_SIZE"
  --compute_type "$WHISPERX_COMPUTE_TYPE"
  --device "$WHISPERX_DEVICE"
  --language "$WHISPERX_LANGUAGE"
  --diarize
  --hf_token "$HF_TOKEN"
  --output_dir "$OUTDIR"
)

if [ -n "${WHISPERX_MIN_SPEAKERS:-}" ]; then
  CMD+=(--min_speakers "$WHISPERX_MIN_SPEAKERS")
fi

if [ -n "${WHISPERX_MAX_SPEAKERS:-}" ]; then
  CMD+=(--max_speakers "$WHISPERX_MAX_SPEAKERS")
fi

SAFE_CMD=("${CMD[@]}")
for i in "${!SAFE_CMD[@]}"; do
  if [ "${SAFE_CMD[$i]}" = "--hf_token" ] && [ $((i+1)) -lt ${#SAFE_CMD[@]} ]; then
    SAFE_CMD[$((i+1))]="[REDACTED]"
  fi
done

{
  printf 'Command: '
  printf '%q ' "${SAFE_CMD[@]}"
  printf '\n\n'
} >> "$STATUSFILE"

if "${CMD[@]}" >"$LOGFILE" 2>&1; then
  {
    echo "Completed: $(date -Is)"
    echo "Success: yes"
  } >> "$STATUSFILE"

  LATEST_JSON="$(ls -t "$OUTDIR"/*.json 2>/dev/null | head -n 1)"
  if [ -n "$LATEST_JSON" ]; then
    if python "$BASE_DIR/bin/transcript_chunker.py" "$LATEST_JSON" \
        --target-words "${TRANSCRIPT_CHUNK_TARGET_WORDS:-1400}" \
        --max-words "${TRANSCRIPT_CHUNK_MAX_WORDS:-2200}" >> "$LOGFILE" 2>&1; then
      echo "Chunking: yes" >> "$STATUSFILE"
    else
      echo "Chunking: failed" >> "$STATUSFILE"
    fi
  else
    echo "Chunking: no JSON found" >> "$STATUSFILE"
  fi

  if [ -f "$OUTDIR/chunks_out/transcript_chunks.jsonl" ]; then
    if python "$BASE_DIR/bin/ollama_meeting_summary.py" "$OUTDIR" >> "$LOGFILE" 2>&1; then
      echo "Summaries: yes" >> "$STATUSFILE"
    else
      echo "Summaries: failed" >> "$STATUSFILE"
    fi
  else
    echo "Summaries: no chunks found" >> "$STATUSFILE"
  fi

else
  {
    echo "Completed: $(date -Is)"
    echo "Success: no"
    echo "See log: $LOGFILE"
  } >> "$STATUSFILE"
  exit 1
fi