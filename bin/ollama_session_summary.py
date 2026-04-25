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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def load_template(path: Path, **kwargs) -> str:
    text = path.read_text(encoding="utf-8")
    return text.format_map(SafeDict(**kwargs))


def build_chunk_prompt(prompt_dir: Path, chunk: dict) -> str:
    return load_template(
        prompt_dir / "chunk_prompt.txt",
        chunk_id=chunk.get("chunk_id", ""),
        chunk_type=chunk.get("chunk_type", ""),
        speaker_span=chunk.get("speaker_span", ""),
        start_time=chunk.get("start_time", ""),
        end_time=chunk.get("end_time", ""),
        chunk_text=chunk.get("text", ""),
    )

def build_reduce_input(chunk_summaries: list[dict]) -> str:
    parts: list[str] = []
    for item in chunk_summaries:
        parts.append(
            f"""# Chunk {item["chunk_id"]}
- File: {item["file_name"]}
- Time range: {item["start_time"]} to {item["end_time"]}
- Speaker span: {item["speaker_span"]}

{item["summary"]}"""
        )
    return "\n\n".join(parts)

def build_reduce_prompt(prompt_dir: Path, template_name: str, combined: str) -> str:
    return load_template(
        prompt_dir / template_name,
        chunk_summaries=combined,
    )


PROFILE_CONFIG = {
    "meeting": {
        "summary_root_env": "MEETING_SUMMARIES_ROOT",
        "default_summary_root_name": "meeting-summaries",
        "outputs": [
            ("summary.md", "summary_prompt.txt"),
            ("action-items.md", "action_items_prompt.txt"),
            ("minutes-draft.md", "minutes_prompt.txt"),
        ],
    },
    "lesson": {
        "summary_root_env": "LESSON_SUMMARIES_ROOT",
        "default_summary_root_name": "lesson-summaries",
        "outputs": [
            ("lesson-notes.md", "lesson_notes_prompt.txt"),
            ("flashcards.md", "flashcards_prompt.txt"),
            ("quiz.md", "quiz_prompt.txt"),
            ("review-sheet.md", "review_sheet_prompt.txt"),
        ],
    },
}


def resolve_summary_dir(
    transcript_dir: Path,
    summary_root: Path,
) -> Path:
    """
    Place summaries in:
      <summary_root>/<transcript_dir.name>

    Example:
      transcript_dir = /home/me/meeting-transcripts/session-123
      summary_root   = /home/me/meeting-summaries
      result         = /home/me/meeting-summaries/session-123
    """
    return summary_root / transcript_dir.name


def main(default_profile: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize transcript chunks with Ollama."
    )
    parser.add_argument(
        "transcript_dir",
        type=Path,
        help="Transcript output directory containing chunks_out/",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIG),
        default=default_profile or "meeting",
        help="Prompt/output profile to use.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://192.168.0.105:11434"),
        help="Base URL for Ollama.",
    )
    parser.add_argument(
        "--map-model",
        default=os.environ.get("OLLAMA_MAP_MODEL", "qwen2.5:32b"),
        help="Model used for chunk-level summaries.",
    )
    parser.add_argument(
        "--reduce-model",
        default=os.environ.get("OLLAMA_REDUCE_MODEL", "qwen2.5:32b"),
        help="Model used for final output documents.",
    )
    parser.add_argument(
        "--keep-alive",
        default=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
        help="Ollama keep_alive value.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--map-num-ctx",
        type=int,
        default=int(os.environ["OLLAMA_MAP_NUM_CTX"])
        if os.environ.get("OLLAMA_MAP_NUM_CTX")
        else None,
        help="Context window for map-stage chunk summaries.",
    )
    parser.add_argument(
        "--reduce-num-ctx",
        type=int,
        default=int(os.environ["OLLAMA_REDUCE_NUM_CTX"])
        if os.environ.get("OLLAMA_REDUCE_NUM_CTX")
        else None,
        help="Context window for reduce-stage final documents.",
    )

    args = parser.parse_args()

    transcript_dir = args.transcript_dir.resolve()
    profile = args.profile
    profile_cfg = PROFILE_CONFIG[profile]

    # Repo root = parent of bin/
    repo_root = Path(__file__).resolve().parent.parent
    prompt_dir = repo_root / "prompts" / profile

    if not prompt_dir.exists():
        print(f"ERROR: Prompt directory not found: {prompt_dir}", file=sys.stderr)
        return 2

    chunk_system_path = prompt_dir / "chunk_system.txt"
    reduce_system_path = prompt_dir / "reduce_system.txt"

    if not chunk_system_path.exists():
        print(f"ERROR: Missing chunk system prompt: {chunk_system_path}", file=sys.stderr)
        return 2
    if not reduce_system_path.exists():
        print(f"ERROR: Missing reduce system prompt: {reduce_system_path}", file=sys.stderr)
        return 2

    chunk_system = chunk_system_path.read_text(encoding="utf-8").strip()
    reduce_system = reduce_system_path.read_text(encoding="utf-8").strip()

    chunks_jsonl = transcript_dir / "chunks_out" / "transcript_chunks.jsonl"
    if not chunks_jsonl.exists():
        print(f"ERROR: Missing chunk file: {chunks_jsonl}", file=sys.stderr)
        return 2

    try:
        chunks = load_jsonl(chunks_jsonl)
    except Exception as exc:
        print(f"ERROR: Failed to read chunk JSONL: {chunks_jsonl}: {exc}", file=sys.stderr)
        return 2

    if not chunks:
        print(f"ERROR: No chunks found in {chunks_jsonl}", file=sys.stderr)
        return 2

    summary_root = Path(
        os.environ.get(
            profile_cfg["summary_root_env"],
            str(repo_root / profile_cfg["default_summary_root_name"]),
        )
    ).expanduser().resolve()

    summaries_dir = resolve_summary_dir(transcript_dir, summary_root)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    chunk_summaries_path = summaries_dir / "chunk_summaries.jsonl"

    print(f"[info] profile={profile}", flush=True)
    print(f"[info] transcript_dir={transcript_dir}", flush=True)
    print(f"[info] prompt_dir={prompt_dir}", flush=True)
    print(f"[info] summaries_dir={summaries_dir}", flush=True)
    print(f"[info] chunk_count={len(chunks)}", flush=True)
    print(f"[info] ollama_url={args.ollama_url}", flush=True)
    print(f"[info] map_model={args.map_model}", flush=True)
    print(f"[info] reduce_model={args.reduce_model}", flush=True)
    print(f"[info] keep_alive={args.keep_alive}", flush=True)
    print(f"[info] temperature={args.temperature}", flush=True)
    print(f"[info] map_num_ctx={args.map_num_ctx}", flush=True)
    print(f"[info] reduce_num_ctx={args.reduce_num_ctx}", flush=True)

    chunk_summaries: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk-{idx:03d}")
        print(f"[map] {idx}/{len(chunks)} summarizing {chunk_id}", flush=True)

        prompt = build_chunk_prompt(prompt_dir, chunk)

        try:
            summary_text = call_ollama(
                ollama_url=args.ollama_url,
                model=args.map_model,
                prompt=prompt,
                system=chunk_system,
                keep_alive=args.keep_alive,
                temperature=args.temperature,
                num_ctx=args.map_num_ctx,
            )
        except Exception as exc:
            print(f"ERROR: Ollama map-stage failed for {chunk_id}: {exc}", file=sys.stderr)
            return 1

        row = {
            "chunk_id": chunk_id,
            "file_name": chunk.get("file_name", transcript_dir.name),
            "start_time": chunk.get("start_time"),
            "end_time": chunk.get("end_time"),
            "speaker_span": chunk.get("speaker_span", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "summary": summary_text,
        }
        chunk_summaries.append(row)

    with chunk_summaries_path.open("w", encoding="utf-8") as f:
        for row in chunk_summaries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[info] wrote {chunk_summaries_path}", flush=True)

    combined = build_reduce_input(chunk_summaries)

    for output_filename, template_name in profile_cfg["outputs"]:
        template_path = prompt_dir / template_name
        if not template_path.exists():
            print(f"ERROR: Missing reduce prompt: {template_path}", file=sys.stderr)
            return 2

        print(f"[reduce] generating {output_filename}", flush=True)
        reduce_prompt = build_reduce_prompt(prompt_dir, template_name, combined)

        try:
            content = call_ollama(
                ollama_url=args.ollama_url,
                model=args.reduce_model,
                prompt=reduce_prompt,
                system=reduce_system,
                keep_alive=args.keep_alive,
                temperature=args.temperature,
                num_ctx=args.reduce_num_ctx,
            )
        except Exception as exc:
            print(
                f"ERROR: Ollama reduce-stage failed for {output_filename}: {exc}",
                file=sys.stderr,
            )
            return 1

        out_path = summaries_dir / output_filename
        out_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        print(f"[info] wrote {out_path}", flush=True)

    print("[done] summary generation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())