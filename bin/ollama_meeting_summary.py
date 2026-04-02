#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any


def call_ollama(
    ollama_url: str,
    model: str,
    prompt: str,
    system: str = "",
    keep_alive: str = "30m",
    temperature: float = 0.2,
    num_ctx: int | None = None,
) -> str:
    options = {
        "temperature": temperature,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
    }

    req = urllib.request.Request(
        url=ollama_url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return (data.get("response") or "").strip()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_chunk_prompt(chunk: dict) -> str:
    return f"""You are summarizing one chunk from a meeting-style transcript.

Be accurate and conservative.
Do not invent names, decisions, motions, or action items.
If a point is unclear, say so.
Use speaker labels exactly as given if no real names are known.

Chunk metadata:
- Chunk ID: {chunk.get("chunk_id")}
- Chunk type: {chunk.get("chunk_type")}
- Speaker span: {chunk.get("speaker_span")}
- Time range: {chunk.get("start_time")} to {chunk.get("end_time")}

Transcript chunk:
{chunk.get("text", "")}

Write a concise markdown summary with exactly these sections:

## Topics
## Decisions or tentative decisions
## Motions or proposals mentioned
## Action items
## Open questions or unresolved issues
## Important notes

For any empty section, write: None noted.
"""


def build_reduce_input(chunk_summaries: list[dict]) -> str:
    parts: list[str] = []
    for item in chunk_summaries:
        parts.append(
            f"""# Chunk {item["chunk_id"]}
- File: {item["file_name"]}
- Time range: {item["start_time"]} to {item["end_time"]}
- Speaker span: {item["speaker_span"]}

{item["summary"]}
"""
        )
    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize meeting transcript chunks with Ollama.")
    parser.add_argument("transcript_dir", type=Path, help="Transcript output directory containing chunks_out/")
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://192.168.0.105:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--map-model",
        default=os.environ.get("OLLAMA_MAP_MODEL", "qwen2.5:32b"),
        help="Model for per-chunk summaries",
    )
    parser.add_argument(
       "--reduce-model",
        default=os.environ.get("OLLAMA_REDUCE_MODEL", "qwen2.5:32b"),
        help="Model for final synthesis",
    )    
    keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
    temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
    map_num_ctx = int(os.environ.get("OLLAMA_MAP_NUM_CTX", "16384"))
    reduce_num_ctx = int(os.environ.get("OLLAMA_REDUCE_NUM_CTX", "32768"))
    args = parser.parse_args()
    transcript_dir = args.transcript_dir
    chunks_jsonl = transcript_dir / "chunks_out" / "transcript_chunks.jsonl"
    if not chunks_jsonl.exists():
        print(f"Missing chunk index: {chunks_jsonl}", file=sys.stderr)
        return 1

    summary_root = Path(
        os.environ.get(
            "MEETING_SUMMARIES_ROOT",
            str(transcript_dir.parent.parent / "meeting-summaries"),
        )
    )
    summaries_dir = summary_root / transcript_dir.name
    summaries_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_jsonl(chunks_jsonl)
    if not chunks:
        print("No chunk records found.", file=sys.stderr)
        return 1

    chunk_system = (
        "You create careful, high-quality summaries from transcript chunks. "
        "You do not hallucinate. You preserve chronology."
    )

    chunk_summaries: list[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[map] summarizing chunk {idx}/{len(chunks)} (chunk_id={chunk['chunk_id']})", flush=True)
        summary = call_ollama(
            ollama_url=args.ollama_url,
            model=args.map_model,
            prompt=build_chunk_prompt(chunk),
            system=chunk_system,
            keep_alive=keep_alive,
            temperature=temperature,
            num_ctx=map_num_ctx,
        )

        row = {
            "chunk_id": chunk["chunk_id"],
            "file_name": chunk["file_name"],
            "speaker_span": chunk["speaker_span"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "summary": summary,
        }
        chunk_summaries.append(row)

    chunk_summaries_jsonl = summaries_dir / "chunk_summaries.jsonl"
    with chunk_summaries_jsonl.open("w", encoding="utf-8") as f:
        for row in chunk_summaries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    combined = build_reduce_input(chunk_summaries)

    reduce_system = (
        "You synthesize chunk summaries into accurate, professional meeting documents. "
        "Do not invent facts. Preserve chronology. Use speaker labels when names are unknown."
    )

    summary_prompt = f"""Using the ordered chunk summaries below, write a concise executive summary in markdown.

Requirements:
- 1 short title
- 1 brief overview paragraph
- 5 to 10 bullet points covering the main topics in chronological order
- Do not invent decisions or names

Chunk summaries:
{combined}
"""

    actions_prompt = f"""Using the ordered chunk summaries below, write an action-items document in markdown.

Requirements:
- Title: Action Items
- One bullet per action item
- Include owner only if clearly known
- Include status like 'tentative' if uncertain
- If none exist, write: No clear action items identified.

Chunk summaries:
{combined}
"""

    minutes_prompt = f"""Using the ordered chunk summaries below, write formal draft minutes in markdown.

Requirements:
- Title: Draft Minutes
- Sections:
  - Overview
  - Topics Discussed
  - Decisions / Tentative Decisions
  - Motions / Proposals Mentioned
  - Action Items
  - Open Questions
- Keep chronology clear
- Do not invent names, motions, or votes

Chunk summaries:
{combined}
"""
    print("[reduce] writing summary.md", flush=True)
    summary_md = call_ollama(
        ollama_url=args.ollama_url,
        model=args.reduce_model,
        prompt=summary_prompt,
        system=reduce_system,
        keep_alive=keep_alive,
        temperature=temperature,
        num_ctx=reduce_num_ctx,
    )
    (summaries_dir / "summary.md").write_text(summary_md + "\n", encoding="utf-8")

    print("[reduce] writing action-items.md", flush=True)
    actions_md = call_ollama(
        ollama_url=args.ollama_url,
        model=args.reduce_model,
        prompt=actions_prompt,
        system=reduce_system,
        keep_alive=keep_alive,
        temperature=temperature,
        num_ctx=reduce_num_ctx,
    )
    (summaries_dir / "action-items.md").write_text(actions_md + "\n", encoding="utf-8")

    print("[reduce] writing minutes-draft.md", flush=True)
    minutes_md = call_ollama(
        ollama_url=args.ollama_url,
        model=args.reduce_model,
        prompt=minutes_prompt,
        system=reduce_system,
        keep_alive=keep_alive,
        temperature=temperature,
        num_ctx=reduce_num_ctx,
    )
    (summaries_dir / "minutes-draft.md").write_text(minutes_md + "\n", encoding="utf-8")

    print(f"Wrote summaries to: {summaries_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())