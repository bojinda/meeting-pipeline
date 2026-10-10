"""Immutable source-turn bindings and explicitly approved, private corrections."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

from .speaker_suggestions import _plain_name, speaker_input, write_private_json

CORRECTIONS_FILE = "speaker_turn_corrections.json"
TURNS_FILE = "speaker-turns.json"
INSPECTION_FILE = "speaker-turns.private.txt"
CONFLICTS_FILE = "speaker-turn-conflicts.json"
TURN = re.compile(r"^\[([^\]\n]+)\]:?[ \t]*(.*)$")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_index(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "chunks_out" / "transcript_chunks.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _timed_turns(directory: Path) -> dict:
    """Use existing WhisperX metadata only when its text match is unambiguous."""
    from transcript_chunker import merge_segments_into_turns
    matches = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in {CORRECTIONS_FILE, "speaker_aliases.json", "speaker_roster.private.json", "speaker-suggestions.json", TURNS_FILE, CONFLICTS_FILE, "redactions.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        segments = data.get("segments") if isinstance(data, dict) else None
        if not isinstance(segments, list) or not all(isinstance(s, dict) and isinstance(s.get("text"), str) and isinstance(s.get("start"), (int, float)) and isinstance(s.get("end"), (int, float)) for s in segments):
            continue
        for turn in merge_segments_into_turns(segments):
            key = (turn["speaker"], " ".join(turn["text"].split()))
            matches.setdefault(key, []).append({"file": path.name, "start_time": turn["start"], "end_time": turn["end"]})
    return {key: rows[0] for key, rows in matches.items() if len(rows) == 1}


def turn_catalog(directory: Path, chunks: list[dict] | None = None) -> dict:
    chunks = read_index(directory) if chunks is None else chunks
    timed = _timed_turns(directory)
    originals = []
    current_speaker = "UNKNOWN"
    for chunk_position, chunk in enumerate(chunks):
        for line_number, line in enumerate(chunk.get("text", "").splitlines(), 1):
            if not line.strip():
                continue
            match = TURN.match(line)
            speaker, text = match.groups() if match else (current_speaker, line)
            current_speaker = speaker
            metadata = timed.get((speaker, " ".join(text.split())))
            coordinate = {"whisperx": metadata} if metadata else {"chunk_id": chunk.get("chunk_id", chunk_position + 1), "line": line_number, "start_time": chunk.get("start_time"), "end_time": chunk.get("end_time")}
            fingerprint = digest({"coordinates": coordinate, "source_speaker": speaker, "original_text": text})
            originals.append({"turn_id": "T" + fingerprint[:32], "source_fingerprint": fingerprint,
                              "source_chunk_id": chunk.get("chunk_id", chunk_position + 1), "chunk_position": chunk_position,
                              "source_line": line_number, "source_speaker": speaker,
                              "start_time": (metadata or chunk).get("start_time"), "end_time": (metadata or chunk).get("end_time"),
                              "time_precision": "whisperx_turn" if metadata else "containing_chunk",
                              "text": line})
    if len({row["turn_id"] for row in originals}) != len(originals):
        raise ValueError("Duplicate source-turn bindings; inspect source index before correcting speakers")
    # Assign bindings before redaction, then expose only surviving text. No private
    # redaction payload is ever read or included in turn/review artifacts.
    visible = speaker_input([{"chunk_id": row["turn_id"], "text": row["text"], "start_time": row["start_time"], "end_time": row["end_time"]} for row in originals])
    by_id = {row["turn_id"]: row for row in originals}
    turns = []
    for row in visible:
        source = by_id[row["chunk_id"]]
        text = "\n".join(TURN.sub(lambda match: match[2], line) for line in row["text"].splitlines())
        # Keep the existing review-body convention for colon-formatted labels,
        # but retain the exact redaction rendering for verified label-only edits.
        if source["text"].startswith(f"[{source['source_speaker']}]:"):
            text = text.removeprefix(":").lstrip(" \t")
        turns.append({**source, "text": text, "redacted_text": row["text"], "redaction_gap": row["speaker_review_gap"], "position": len(turns)})
    return {"version": 1, "meeting_id": str(directory.resolve()), "source_digest": digest(chunks), "turns": turns}


_DEFAULT_CORRECTION_SOURCE = object()


def correction_document(directory: Path, source_path=_DEFAULT_CORRECTION_SOURCE) -> dict:
    path = directory / CORRECTIONS_FILE if source_path is _DEFAULT_CORRECTION_SOURCE else source_path
    if path is None or not path.exists():
        return {"version": 1, "meeting_id": str(directory.resolve()), "corrections": {}}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("version") != 1 or data.get("meeting_id") != str(directory.resolve()) or not isinstance(data.get("corrections"), dict):
        raise ValueError("Turn corrections do not belong to this meeting or have an invalid schema")
    return data


def correction_conflicts(catalog: dict, document: dict, aliases: dict) -> list[dict]:
    known = {row["turn_id"]: row for row in catalog["turns"]}
    issues = []
    for tid, item in document["corrections"].items():
        source = known.get(tid)
        if not source or not isinstance(item, dict) or item.get("approved") is not True or item.get("source_fingerprint") != source["source_fingerprint"] or item.get("source_speaker") != source["source_speaker"]:
            issues.append({"turn_id": tid, "type": "stale_or_invalid_binding", "blocking": True})
            continue
        try:
            name = _plain_name(item.get("name"))
        except ValueError:
            issues.append({"turn_id": tid, "type": "invalid_name", "blocking": True})
            continue
        global_name = aliases.get(source["source_speaker"])
        if global_name and name != global_name:
            issues.append({"turn_id": tid, "type": "explicit_turn_override", "blocking": False, "global_alias": global_name, "approved_name": name})
    return issues


def effective_turns(catalog: dict, document: dict, aliases: dict) -> list[dict]:
    if any(issue["blocking"] for issue in correction_conflicts(catalog, document, aliases)):
        raise ValueError("Stale or invalid turn correction; inspect turn conflicts before processing")
    return [{**turn, "effective_speaker": document["corrections"].get(turn["turn_id"], {}).get("name", aliases.get(turn["source_speaker"], turn["source_speaker"])),
             "approved_turn_name": document["corrections"].get(turn["turn_id"], {}).get("name")} for turn in catalog["turns"]]


def approve_turn(directory: Path, catalog: dict, turn_id: str, name: str) -> None:
    source = next((row for row in catalog["turns"] if row["turn_id"] == turn_id), None)
    if source is None:
        raise ValueError("Selected source turn is absent, changed, or fully redacted")
    document = correction_document(directory)
    document["corrections"][turn_id] = {"name": _plain_name(name), "approved": True, "source_fingerprint": source["source_fingerprint"], "source_speaker": source["source_speaker"], "approved_at": datetime.now(timezone.utc).isoformat()}
    write_private_json(directory / CORRECTIONS_FILE, document)


def remove_turn(directory: Path, turn_id: str) -> None:
    document = correction_document(directory)
    if turn_id not in document["corrections"]:
        raise ValueError("No correction exists for the selected turn")
    del document["corrections"][turn_id]
    write_private_json(directory / CORRECTIONS_FILE, document)


def corrected_chunks(redacted: list[dict], catalog: dict, document: dict, aliases: dict) -> list[dict]:
    turns = effective_turns(catalog, document, aliases)
    if not document["corrections"]:
        return redacted
    by_chunk = {}
    for turn in turns:
        by_chunk.setdefault(str(turn["source_chunk_id"]), []).append(turn)
    if [str(row["chunk_id"]) for row in redacted] != list(by_chunk):
        raise ValueError("Source-to-redacted turn alignment cannot be verified")
    corrected = []
    for row in redacted:
        source_turns = by_chunk[str(row["chunk_id"])]
        if any(not isinstance(turn.get("redacted_text"), str) for turn in source_turns) or row["text"] != "\n".join(turn["redacted_text"] for turn in source_turns):
            raise ValueError("Source-to-redacted turn alignment cannot be verified")
        edits, offset = [], 0
        for turn in source_turns:
            name = turn["approved_turn_name"]
            if name:
                match = TURN.match(turn["redacted_text"])
                if not match or match[1] != turn["source_speaker"]:
                    raise ValueError("Selected source speaker attribution cannot be verified")
                edits.append((offset + match.start(1), offset + match.end(1), name))
            offset += len(turn["redacted_text"]) + 1
        text = row["text"]
        for start, end, name in reversed(edits):
            text = text[:start] + name + text[end:]
        corrected.append({**row, "text": text} if edits else row)
    return corrected
