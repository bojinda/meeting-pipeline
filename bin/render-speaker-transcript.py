#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: render-speaker-transcript.py <whisperx-json>", file=sys.stderr)
    sys.exit(1)

src = Path(sys.argv[1])
data = json.loads(src.read_text(encoding="utf-8"))

segments = data.get("segments", [])
out_lines = []

last_speaker = None
buffer = []

def flush():
    global buffer, last_speaker
    if buffer and last_speaker:
        text = " ".join(buffer).strip()
        if text:
            out_lines.append(f"[{last_speaker}] {text}")
    buffer = []

for seg in segments:
    speaker = seg.get("speaker") or "UNKNOWN"
    text = (seg.get("text") or "").strip()

    if not text:
        continue

    if speaker != last_speaker:
        flush()
        last_speaker = speaker

    buffer.append(text)

flush()

out_path = src.with_name(src.stem + ".speaker.txt")
out_path.write_text("\n\n".join(out_lines) + "\n", encoding="utf-8")
print(out_path)