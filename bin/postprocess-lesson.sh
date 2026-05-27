#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <recording.wav>"
  exit 1
fi

INPUT="$1"
BASE_DIR="$HOME/meeting-pipeline"
CONFIG_FILE="$BASE_DIR/config/.env"

if [[ ! -f "$INPUT" ]]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate whisperx

if [ -f "$CONFIG_FILE" ]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi

OUT_ROOT="${LESSON_TRANSCRIPTS_ROOT:-$BASE_DIR/lesson-transcripts}"
LOG_ROOT="${LESSON_SUMMARIES_ROOT:-$BASE_DIR/lesson-summaries}"

: "${WHISPERX_MODEL:=large-v2}"
: "${WHISPERX_BATCH_SIZE:=4}"
: "${WHISPERX_COMPUTE_TYPE:=float16}"
: "${WHISPERX_DEVICE:=cuda}"
: "${WHISPERX_LANGUAGE:=en}"
: "${HF_TOKEN:?HF_TOKEN is required}"
: "${DELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION:=0}"

BASE="$(basename "$INPUT" .wav)"
OUTDIR="$OUT_ROOT/$BASE"
METADATA_SIDECAR="${INPUT%.wav}.metadata.json"

mkdir -p "$OUTDIR" "$LOG_ROOT"

if [ -f "$METADATA_SIDECAR" ]; then
  python3 - "$METADATA_SIDECAR" "$OUTDIR/session-metadata.json" "$BASE.wav" "$BASE" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
recording_file = sys.argv[3]
session_name = sys.argv[4]

data = json.loads(src.read_text(encoding="utf-8"))
data["recording_file"] = recording_file
data["transcript_dir"] = session_name
data["summary_dir"] = session_name

dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
fi

refresh_final_metadata() {
  local session_meta="$OUTDIR/session-metadata.json"
  local history_json=""

  if [ -f "$session_meta" ]; then
    history_json="$(python3 - "$session_meta" "${LESSON_STATE_DIR:-$BASE_DIR/lesson-state}" <<'PY'
import json
import pathlib
import sys

session_meta = pathlib.Path(sys.argv[1])
state_dir = pathlib.Path(sys.argv[2])

session_id = ""
if session_meta.exists():
    try:
        session_id = (json.loads(session_meta.read_text(encoding="utf-8")).get("session_id") or "").strip()
    except Exception:
        session_id = ""

if session_id:
    history_path = state_dir / "sessions" / f"{session_id}.json"
    if history_path.exists():
        print(history_path)
PY
)"
  fi

  if [ -n "$history_json" ]; then
    python3 - "$history_json" "$OUTDIR/session-metadata.json" "$BASE.wav" "$BASE" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
recording_file = sys.argv[3]
session_name = sys.argv[4]

data = json.loads(src.read_text(encoding="utf-8"))
data["recording_file"] = recording_file
data["transcript_dir"] = session_name
data["summary_dir"] = session_name

dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
  fi

  if [ -f "$OUTDIR/session-metadata.json" ] && [ -d "$LOG_ROOT/$BASE" ]; then
    cp -f "$OUTDIR/session-metadata.json" "$LOG_ROOT/$BASE/session-metadata.json"
  fi
}

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
  echo "Delete source audio after transcription: $DELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION"
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
  CHUNKS_JSONL="$OUTDIR/chunks_out/transcript_chunks.jsonl"

  if [ "$DELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION" = "1" ]; then
    if ls "$OUTDIR"/*.json >/dev/null 2>&1; then
      if [ -f "$INPUT" ]; then
        rm -f -- "$INPUT"
        echo "Source audio deleted after transcription: yes" >> "$STATUSFILE"
      else
        echo "Source audio deleted after transcription: source already missing" >> "$STATUSFILE"
      fi
    else
      echo "Source audio deleted after transcription: skipped (no transcript JSON found)" >> "$STATUSFILE"
    fi
  else
    echo "Source audio deleted after transcription: no" >> "$STATUSFILE"
  fi

  if [ -n "$LATEST_JSON" ]; then
    if python "$BASE_DIR/bin/transcript_chunker.py" "$LATEST_JSON" \
        --target-words "${TRANSCRIPT_CHUNK_TARGET_WORDS:-1400}" \
        --max-words "${TRANSCRIPT_CHUNK_MAX_WORDS:-2200}" >> "$LOGFILE" 2>&1; then

      if [ ! -f "$CHUNKS_JSONL" ] || [ ! -s "$CHUNKS_JSONL" ]; then
        echo "Chunking: no speech detected" >> "$STATUSFILE"
        echo "Summaries: skipped (no chunks)" >> "$STATUSFILE"
        refresh_final_metadata
        exit 0
      else
        echo "Chunking: yes" >> "$STATUSFILE"
      fi

    else
      echo "Chunking: failed" >> "$STATUSFILE"
      exit 1
    fi
  else
    echo "Chunking: no JSON found" >> "$STATUSFILE"
    exit 1
  fi

  if python "$BASE_DIR/bin/ollama_lesson_summary.py" "$OUTDIR" >> "$LOGFILE" 2>&1; then
    echo "Summaries: yes" >> "$STATUSFILE"
    echo "Summary dir: $LOG_ROOT/$(basename "$OUTDIR")" >> "$STATUSFILE"
    mkdir -p "$LOG_ROOT/$BASE"
    refresh_final_metadata
  else
    echo "Summaries: failed" >> "$STATUSFILE"
    exit 1
  fi

else
  {
    echo "Completed: $(date -Is)"
    echo "Success: no"
    echo "See log: $LOGFILE"
  } >> "$STATUSFILE"
  exit 1
fi