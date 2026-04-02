#!/usr/bin/env bash
set -euo pipefail

INPUT="$1"
BASE="$(basename "$INPUT" .wav)"
OUTDIR="$HOME/meeting-pipeline/meeting-transcripts/$BASE"

mkdir -p "$OUTDIR"

printf 'Recording saved: %s\n' "$INPUT" > "$OUTDIR/status.txt"
date >> "$OUTDIR/status.txt"