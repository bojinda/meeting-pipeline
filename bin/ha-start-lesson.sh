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
: "${WINDOWS_START_TASK:?WINDOWS_START_TASK is required}"
: "${LESSON_CONTROL_LOG:=$BASE_DIR/lesson-control.log}"

mkdir -p "$(dirname "$LESSON_CONTROL_LOG")"
mkdir -p "${LESSON_RECORDINGS_ROOT:-$BASE_DIR/lesson-recordings}"

echo "[$(date -Is)] LESSON START requested" >> "$LESSON_CONTROL_LOG"

ssh \
  -i "$WINDOWS_SSH_KEY" \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  "$WINDOWS_USER@$WINDOWS_HOST" \
  "powershell -NoProfile -Command \"Start-ScheduledTask -TaskName '$WINDOWS_START_TASK'\"" \
  >> "$LESSON_CONTROL_LOG" 2>&1

echo "[$(date -Is)] LESSON START command sent successfully" >> "$LESSON_CONTROL_LOG"