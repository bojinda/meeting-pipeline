#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%F-%H%M%S)"
OUT="$HOME/meeting-pipeline/meeting-recordings/union-meeting-$STAMP.wav"

mkdir -p "$HOME/meeting-pipeline/meeting-recordings" "$HOME/meeting-pipeline/meeting-transcripts" "$HOME/meeting-pipeline/meeting-summaries"
echo "$OUT" > "$HOME/meeting-pipeline/meeting-recordings/last_recording.txt"

ffmpeg -nostdin -hide_banner -loglevel warning \
  -f s16le -ar 16000 -ac 1 \
  -i tcp://0.0.0.0:4000?listen=1 \
  -c:a pcm_s16le \
  "$OUT"

if [ -s "$OUT" ]; then
  "$(dirname "$0")/postprocess-meeting.sh" "$OUT" &
fi