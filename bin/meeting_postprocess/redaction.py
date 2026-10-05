"""Exact spoken redaction before normalization, classification, or model calls."""
from __future__ import annotations

import bisect
import json
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass

from .qa import Finding


PRIVATE_REDACTION_FILENAME = "redactions.json"
_COMMAND = re.compile(r"(?<!\w)(?:redact\s+the\s+following|end\s+redaction)(?!\w)", re.IGNORECASE)
_SPEAKER = re.compile(r"^\[([^\]\n]+)\][ \t]*(.*)$")


def has_redaction_start(text: str) -> bool:
    return any(marker.group().lower().split()[0] == "redact" for marker in _COMMAND.finditer(text))


@dataclass
class Turn:
    chunk_index: int
    speaker: str
    text: str
    start: int


@dataclass
class RedactionResult:
    chunks: list[dict]
    redactions: list[dict]
    warnings: list[Finding]


def redact_chunks(chunks: list[dict]) -> RedactionResult:
    turns = []
    offset = 0
    speaker = ""
    for index, chunk in enumerate(chunks):
        for line in chunk.get("text", "").splitlines():
            match = _SPEAKER.match(line)
            if match:
                speaker, line = match.groups()
            turns.append(Turn(index, speaker, line, offset))
            offset += len(line) + 1
    stream = "\n".join(turn.text for turn in turns)
    starts = [turn.start for turn in turns]
    public: dict[int, str] = {}
    private: dict[int, str] = {}
    records = []
    warnings = []
    active: dict | None = None

    def containing(position: int) -> Turn:
        return turns[max(0, bisect.bisect_right(starts, position) - 1)]

    def location(turn: Turn) -> dict:
        chunk = chunks[turn.chunk_index]
        return {"chunk_id": chunk.get("chunk_id", turn.chunk_index + 1), "start_time": chunk.get("start_time"), "end_time": chunk.get("end_time")}

    def add_speakers(start: int, end: int) -> None:
        if active is not None:
            for turn in turns:
                if turn.start < end and turn.start + len(turn.text) > start:
                    if turn.speaker and turn.speaker not in active["speakers"]:
                        active["speakers"].append(turn.speaker)

    def collect(start: int, end: int, target: dict[int, str]) -> None:
        for index, turn in enumerate(turns):
            left, right = max(start, turn.start), min(end, turn.start + len(turn.text))
            if left < right:
                target[index] = target.get(index, "") + turn.text[left - turn.start:right - turn.start]
        if active is not None:
            add_speakers(start, end)

    def render(fragments: dict[int, str]) -> str:
        return "\n".join(
            (f"[{turns[index].speaker}] " if turns[index].speaker else "") + text.strip()
            for index, text in fragments.items() if text.strip()
        )

    cursor = 0
    for marker in _COMMAND.finditer(stream):
        before = marker.start()
        while before > cursor and stream[before - 1] in "(\"'“‘[":
            before -= 1
        collect(cursor, before, private if active is not None else public)
        turn = containing(marker.start())
        ending_turn = containing(marker.end() - 1)
        if marker.group().lower().split()[0] == "redact":
            if active is None:
                loc = location(turn)
                active = {
                    "number": len(records) + 1, "start_chunk_id": loc["chunk_id"],
                    "end_chunk_id": loc["chunk_id"], "start_time": loc["start_time"],
                    "end_time": loc["end_time"], "closed": False, "speakers": [],
                    "reason": "unclosed_at_eof", "content": "",
                }
                private = {}
            else:
                warnings.append(Finding("redaction_repeated_start", 0, f"Repeated start marker inside redaction {active['number']}; exclusion remains active.", ""))
            add_speakers(marker.start(), marker.end())
        elif active is None:
            warnings.append(Finding("redaction_unmatched_end", 0, "Unmatched end-redaction marker; surrounding content retained.", ""))
        else:
            add_speakers(marker.start(), marker.end())
            loc = location(ending_turn)
            active.update(end_chunk_id=loc["chunk_id"], end_time=loc["end_time"], closed=True, reason="explicit_spoken_redaction", content=render(private))
            records.append(active)
            active = None
            private = {}
        cursor = marker.end()
        # Punctuation attached to a command belongs to the command, not to the
        # adjacent content. Preserve quotes/punctuation after intervening space.
        while cursor < len(stream) and stream[cursor] in ".,;:!?…\"'”’)]}":
            cursor += 1
    collect(cursor, len(stream), private if active is not None else public)
    if active is not None:
        last = chunks[-1]
        active.update(end_chunk_id=last.get("chunk_id", len(chunks)), end_time=last.get("end_time"), content=render(private))
        records.append(active)
        warnings.append(Finding("redaction_unclosed", 0, f"Redaction {active['number']} was unclosed at EOF; excluded through the end of the transcript.", ""))

    sanitized = []
    for chunk_index, chunk in enumerate(chunks):
        fragments = {index: text for index, text in public.items() if turns[index].chunk_index == chunk_index}
        text = render(fragments)
        if not text:
            continue
        speakers = list(dict.fromkeys(turns[index].speaker for index, text in fragments.items() if text.strip() and turns[index].speaker))
        # An explicit schema prevents auxiliary/raw segment fields from leaking
        # excluded text into debug JSON or future prompt builders.
        sanitized.append({
            "chunk_id": chunk.get("chunk_id", chunk_index + 1),
            "file_name": chunk.get("file_name", ""),
            "start_time": chunk.get("start_time"), "end_time": chunk.get("end_time"),
            "speaker_span": " -> ".join(speakers), "speaker_count": len(speakers),
            "chunk_type": "single_speaker" if len(speakers) == 1 else "discussion",
            "word_count": len(text.split()), "text": text,
        })
    return RedactionResult(sanitized, records, warnings)


def write_private_redactions(directory: Path, records: list[dict]) -> None:
    path = directory / PRIVATE_REDACTION_FILENAME
    if path.is_symlink():
        raise OSError("Private redaction output cannot be a symbolic link")
    descriptor, name = tempfile.mkstemp(prefix=".private-redactions-", suffix=".tmp", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump({"redactions": records}, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
