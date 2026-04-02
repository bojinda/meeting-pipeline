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
: "${MEETING_CONTROL_LOG:=$BASE_DIR/meeting-control.log}"

mkdir -p "$(dirname "$MEETING_CONTROL_LOG")"

echo "[$(date -Is)] START requested" >> "$MEETING_CONTROL_LOG"

ssh \
  -i "$WINDOWS_SSH_KEY" \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  "$WINDOWS_USER@$WINDOWS_HOST" \
  "powershell -NoProfile -Command \"Start-ScheduledTask -TaskName '$WINDOWS_START_TASK'\"" \
  >> "$MEETING_CONTROL_LOG" 2>&1

echo "[$(date -Is)] START command sent successfully" >> "$MEETING_CONTROL_LOG"