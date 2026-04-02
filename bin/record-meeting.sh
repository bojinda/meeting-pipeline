#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$HOME/meeting-pipeline"
RECORDINGS_DIR="$BASE_DIR/meeting-recordings"
TRANSCRIPTS_DIR="$BASE_DIR/meeting-transcripts"
SUMMARIES_DIR="$BASE_DIR/meeting-summaries"

mkdir -p "$RECORDINGS_DIR" "$TRANSCRIPTS_DIR" "$SUMMARIES_DIR"

STAMP="$(date +%F-%H%M%S)"
OUT="$RECORDINGS_DIR/union-meeting-$STAMP.wav"

ffmpeg -nostdin -hide_banner -loglevel warning \
  -f s16le -ar 16000 -ac 1 \
  -i tcp://0.0.0.0:4000?listen=1 \
  -c:a pcm_s16le \
  "$OUT"

if [ -s "$OUT" ]; then
  printf '%s\n' "$OUT" > "$RECORDINGS_DIR/last_recording.txt"
  "$(dirname "$0")/postprocess-meeting.sh" "$OUT" &
fi