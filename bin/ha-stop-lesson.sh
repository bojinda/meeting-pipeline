#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$HOME/meeting-pipeline"
CONFIG_FILE="$BASE_DIR/config/.env"

if [ -f "$CONFIG_FILE" ]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi

: "${WINDOWS_HOST:?WINDOWS_HOST is required}"
: "${WINDOWS_USER:?WINDOWS_USER is required}"
: "${WINDOWS_SSH_KEY:?WINDOWS_SSH_KEY is required}"
: "${WINDOWS_STOP_SCRIPT:?WINDOWS_STOP_SCRIPT is required}"
: "${LESSON_CONTROL_LOG:=$BASE_DIR/lesson-control.log}"

LESSON_RECORDINGS_ROOT="${LESSON_RECORDINGS_ROOT:-$BASE_DIR/lesson-recordings}"
LESSON_SOURCE_RECORDINGS_ROOT="${LESSON_SOURCE_RECORDINGS_ROOT:-$BASE_DIR/meeting-recordings}"

mkdir -p "$(dirname "$LESSON_CONTROL_LOG")"
mkdir -p "$LESSON_RECORDINGS_ROOT"

echo "[$(date -Is)] LESSON STOP requested" >> "$LESSON_CONTROL_LOG"

ssh \
  -i "$WINDOWS_SSH_KEY" \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  "$WINDOWS_USER@$WINDOWS_HOST" \
  "powershell -NoProfile -ExecutionPolicy Bypass -Command \"if (Test-Path '$WINDOWS_STOP_SCRIPT') { & '$WINDOWS_STOP_SCRIPT' } else { Write-Error 'Stop script not found: $WINDOWS_STOP_SCRIPT'; exit 1 }\"" \
  >> "$LESSON_CONTROL_LOG" 2>&1

echo "[$(date -Is)] LESSON STOP command sent successfully" >> "$LESSON_CONTROL_LOG"

sleep 3

LATEST_WAV="$(find "$LESSON_SOURCE_RECORDINGS_ROOT" -maxdepth 1 -type f -name '*.wav' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"

if [ -z "$LATEST_WAV" ]; then
  echo "[$(date -Is)] No WAV found in $LESSON_SOURCE_RECORDINGS_ROOT" >> "$LESSON_CONTROL_LOG"
  exit 1
fi

DEST_WAV="$LESSON_RECORDINGS_ROOT/$(basename "$LATEST_WAV")"

if [ "$LATEST_WAV" != "$DEST_WAV" ]; then
  cp -f "$LATEST_WAV" "$DEST_WAV"
else
  DEST_WAV="$LATEST_WAV"
fi

printf '%s\n' "$DEST_WAV" > "$LESSON_RECORDINGS_ROOT/last_recording.txt"

echo "[$(date -Is)] Lesson WAV selected: $DEST_WAV" >> "$LESSON_CONTROL_LOG"
echo "[$(date -Is)] Updated $LESSON_RECORDINGS_ROOT/last_recording.txt" >> "$LESSON_CONTROL_LOG"