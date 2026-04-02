#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class ChunkRecord:
    file_name: str
    chunk_id: int
    chunk_type: str
    speaker_span: str
    speaker_count: int
    start_time: float
    end_time: float
    word_count: int
    text: str


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "chunk"


def count_words(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text))


def format_ts(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02}:{m:02}:{s:02}"


def segment_speaker(seg: dict) -> str:
    return seg.get("speaker") or "UNKNOWN"


def segment_text(seg: dict) -> str:
    return (seg.get("text") or "").strip()

MAX_SAME_SPEAKER_GAP = 5.0
TAIL_MERGE_MAX_WORDS = 450


def split_oversized_turns(turns: List[dict], target_words: int, max_words: int) -> List[dict]:
    out: List[dict] = []

    for turn in turns:
        words = turn["text"].split()
        if len(words) <= max_words:
            out.append(turn)
            continue

        step = target_words
        for i in range(0, len(words), step):
            subtext = " ".join(words[i:i + step]).strip()
            out.append({
                "speaker": turn["speaker"],
                "start": turn["start"],
                "end": turn["end"],
                "text": subtext,
            })

    return out


def merge_short_tail_chunk(chunks: List[List[dict]], max_words: int, tail_merge_max_words: int) -> List[List[dict]]:
    if len(chunks) < 2:
        return chunks

    tail = chunks[-1]
    tail_words = sum(count_words(seg["text"]) for seg in tail)

    if tail_words > tail_merge_max_words:
        return chunks

    prev = chunks[-2]
    prev_words = sum(count_words(seg["text"]) for seg in prev)

    if prev_words + tail_words <= max_words:
        chunks[-2] = prev + tail
        chunks.pop()

    return chunks

def merge_segments_into_turns(segments: List[dict]) -> List[dict]:
    turns: List[dict] = []

    for seg in segments:
        text = segment_text(seg)
        if not text:
            continue

        speaker = segment_speaker(seg)
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        if not turns:
            turns.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            })
            continue

        last = turns[-1]
        gap = start - last["end"]

        if last["speaker"] == speaker and gap <= MAX_SAME_SPEAKER_GAP:
            last["end"] = end
            last["text"] = (last["text"].rstrip() + " " + text.lstrip()).strip()
        else:
            turns.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            })

    return turns

def merge_segments_into_chunks(
    segments: List[dict],
    target_words: int,
    max_words: int,
) -> List[List[dict]]:
    turns = merge_segments_into_turns(segments)
    turns = split_oversized_turns(turns, target_words=target_words, max_words=max_words)

    chunks: List[List[dict]] = []
    current: List[dict] = []
    current_words = 0

    for turn in turns:
        turn_words = count_words(turn["text"])

        if not current:
            current = [turn]
            current_words = turn_words
            continue

        would_exceed_max = current_words + turn_words > max_words
        near_target = current_words >= target_words

        # break only BETWEEN complete turns
        if would_exceed_max or near_target:
            chunks.append(current)
            current = [turn]
            current_words = turn_words
        else:
            current.append(turn)
            current_words += turn_words

    if current:
        chunks.append(current)

    chunks = merge_short_tail_chunk(
        chunks,
        max_words=max_words,
        tail_merge_max_words=TAIL_MERGE_MAX_WORDS,
    )

    return chunks


def chunk_type_for(chunk: List[dict]) -> str:
    speakers = {s["speaker"] for s in chunk}
    return "single_speaker" if len(speakers) == 1 else "discussion"


def speaker_span_for(chunk: List[dict]) -> str:
    speakers = []
    for seg in chunk:
        sp = seg["speaker"]
        if not speakers or speakers[-1] != sp:
            speakers.append(sp)
    return " -> ".join(speakers)


def render_chunk_markdown(chunk_id: int, chunk: List[dict]) -> str:
    start = chunk[0]["start"]
    end = chunk[-1]["end"]
    speakers = sorted({s["speaker"] for s in chunk})

    lines = [
        f"# Transcript Chunk {chunk_id:03d}",
        "",
        f"- Time range: {format_ts(start)} → {format_ts(end)}",
        f"- Speakers: {', '.join(speakers)}",
        f"- Chunk type: {chunk_type_for(chunk)}",
        "",
    ]

    for seg in chunk:
        sp = seg["speaker"]
        text = seg["text"].strip()
        seg_start = seg["start"]
        seg_end = seg["end"]
        lines.append(f"## [{sp}] {format_ts(seg_start)} → {format_ts(seg_end)}")
        lines.append("")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"

def write_indexes(out_dir: Path, records: List[ChunkRecord]) -> None:
    jsonl_path = out_dir / "transcript_chunks.jsonl"
    csv_path = out_dir / "transcript_chunks.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    fieldnames = list(asdict(records[0]).keys()) if records else [
        "file_name", "chunk_id", "chunk_type", "speaker_span",
        "speaker_count", "start_time", "end_time", "word_count", "text"
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def build_readme(out_dir: Path, source_json: Path, records: List[ChunkRecord]) -> None:
    readme = f"""# Transcript chunker output

Source JSON: `{source_json.name}`
Chunks written: `{len(records)}`

## Notes

- `single_speaker` chunks are typically longer reports/monologues.
- `discussion` chunks preserve short back-and-forth conversation.
- Use these chunks for Ollama map/reduce summarization or upload into AnythingLLM.

## Generated files

- `chunks/` — one markdown file per transcript chunk
- `transcript_chunks.jsonl` — machine-readable chunk index
- `transcript_chunks.csv` — spreadsheet-friendly chunk index
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk WhisperX meeting transcript JSON into reviewable markdown chunks.")
    parser.add_argument("json_file", type=Path, help="Path to WhisperX JSON transcript")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: alongside JSON in ./chunks_out)")
    parser.add_argument("--target-words", type=int, default=1400, help="Preferred chunk size in words")
    parser.add_argument("--max-words", type=int, default=2200, help="Hard max chunk size in words")
    args = parser.parse_args()

    source = args.json_file
    data = json.loads(source.read_text(encoding="utf-8"))
    segments = sorted(data.get("segments", []), key=lambda s: s.get("start", 0.0))

    out_dir = args.out or (source.parent / "chunks_out")
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    grouped = merge_segments_into_chunks(
        segments=segments,
        target_words=args.target_words,
        max_words=args.max_words,
    )

    records: List[ChunkRecord] = []

    for idx, chunk in enumerate(grouped, start=1):
        md = render_chunk_markdown(idx, chunk)
        chunk_type = chunk_type_for(chunk)
        speaker_span = speaker_span_for(chunk)
        speakers = {segment_speaker(s) for s in chunk}
        start_time = chunk[0].get("start", 0.0)
        end_time = chunk[-1].get("end", 0.0)
        text = "\n".join(f"[{s['speaker']}] {s['text']}" for s in chunk)
        word_count = count_words(text)

        base_slug = slugify(speaker_span)[:50]
        file_name = f"{idx:03d}-{chunk_type}-{base_slug}.md"
        (chunks_dir / file_name).write_text(md, encoding="utf-8")

        records.append(
            ChunkRecord(
                file_name=file_name,
                chunk_id=idx,
                chunk_type=chunk_type,
                speaker_span=speaker_span,
                speaker_count=len(speakers),
                start_time=start_time,
                end_time=end_time,
                word_count=word_count,
                text=text,
            )
        )

    write_indexes(out_dir, records)
    build_readme(out_dir, source, records)

    print(f"Wrote {len(records)} chunks to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())