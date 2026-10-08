#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from meeting_postprocess.aliases import load_aliases
from meeting_postprocess.actions import filter_completed_request_tasks
from meeting_postprocess.commitments import commitment_evidence
from meeting_postprocess.motions import adjournment_announcements, correct_adjournment_roles
from meeting_postprocess.normalization import prepare_text
from meeting_postprocess.qa import check_minutes, write_report
from meeting_postprocess.rendering import insert_recap, strip_chunk_references
from meeting_postprocess.redaction import redact_chunks, write_private_redactions
from meeting_postprocess.publication import strip_private_references
from meeting_postprocess.sections import ADJOURNMENT, BUSINESS, RECAP, prepare_chunks
from meeting_postprocess.speaker_suggestions import PRIVATE_SUGGESTIONS_FILENAME, build_speaker_suggestions, load_roster, speaker_input, write_private_json
from meeting_postprocess.speaker_turn_review import review_turns, settings as speaker_review_settings
from meeting_postprocess.speaker_review import review_speakers
from meeting_postprocess.speaker_turns import turn_catalog, correction_document, effective_turns, corrected_chunks, CORRECTIONS_FILE, TURNS_FILE, CONFLICTS_FILE


def call_ollama(
    ollama_url: str,
    model: str,
    prompt: str,
    system: str = "",
    keep_alive: str = "30m",
    temperature: float = 0.2,
    num_ctx: int | None = None,
    usage_callback=None,
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

    began = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if usage_callback is not None:
        usage_callback({"model": model, "num_ctx": num_ctx, "prompt_eval_count": data.get("prompt_eval_count") if type(data.get("prompt_eval_count")) is int else None,
                        "eval_count": data.get("eval_count") if type(data.get("eval_count")) is int else None,
                        "runtime_seconds": round(time.monotonic() - began, 3)})
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
        meeting_section=chunk.get("meeting_section", ""),
        chunk_text=chunk.get("text", ""),
    )

def build_reduce_input(chunk_summaries: list[dict]) -> str:
    parts: list[str] = []
    for item in chunk_summaries:
        section = f"\n- Meeting section: {item['meeting_section']}" if "meeting_section" in item else ""
        parts.append(
            f"""# Chunk {item["chunk_id"]}
- File: {item["file_name"]}
- Time range: {item["start_time"]} to {item["end_time"]}
- Speaker span: {item["speaker_span"]}{section}

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


def resolve_model_defaults(args: argparse.Namespace) -> None:
    """Resolve only omitted CLI options after the final profile is selected."""
    settings = (
        ("map_model", "MAP_MODEL", "qwen2.5:32b", str),
        ("reduce_model", "REDUCE_MODEL", "qwen2.5:32b", str),
        ("map_num_ctx", "MAP_NUM_CTX", None, int),
        ("reduce_num_ctx", "REDUCE_NUM_CTX", None, int),
    )
    for attribute, suffix, fallback, convert in settings:
        if getattr(args, attribute) is not None:
            continue
        keys = [f"OLLAMA_{suffix}"]
        if args.profile == "meeting":
            keys.insert(0, f"MEETING_{suffix}")
        for key in keys:
            value = os.environ.get(key)
            if value:
                try:
                    setattr(args, attribute, convert(value))
                except ValueError as exc:
                    raise ValueError(f"{key} must be an integer, got {value!r}") from exc
                break
        else:
            setattr(args, attribute, fallback)


def main(default_profile: str | None = None, *, _args=None, _summary_dir=None, _usage_callback=None) -> int:
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
        help="Chunk model (meeting default: MEETING_MAP_MODEL, then OLLAMA_MAP_MODEL).",
    )
    parser.add_argument(
        "--reduce-model",
        help="Final-output model (meeting default: MEETING_REDUCE_MODEL, then OLLAMA_REDUCE_MODEL).",
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
        help="Map context (meeting default: MEETING_MAP_NUM_CTX, then OLLAMA_MAP_NUM_CTX).",
    )
    parser.add_argument(
        "--reduce-num-ctx",
        type=int,
        help="Reduce context (meeting default: MEETING_REDUCE_NUM_CTX, then OLLAMA_REDUCE_NUM_CTX).",
    )
    parser.add_argument(
        "--speaker-aliases", type=Path,
        help="Meeting speaker mapping JSON (default: <transcript_dir>/speaker_aliases.json).",
    )
    parser.add_argument(
        "--keep-recap", action=argparse.BooleanOptionalAction,
        default=os.environ.get("MEETING_KEEP_RECAP", "0").lower() in {"1", "true", "yes"},
        help="Include a separately labelled Recap of Previous Meeting in draft minutes.",
    )
    parser.add_argument("--suggest-speakers", action="store_true", help="Meeting only: write private advisory text-based speaker suggestions; never apply them.")
    parser.add_argument("--speaker-roster", type=Path, help="Private roster for the optional meeting suggestion stage (default: per-meeting speaker_roster.private.json).")
    parser.add_argument("--suggest-speakers-llm", action="store_true", help="Explicitly request bounded discovery and verification; requires --suggest-speakers.")
    parser.add_argument("--speaker-suggestion-model", help="Optional local speaker-review model override (default: selected meeting reduce model).")
    parser.add_argument("--speaker-review-mode", choices=("two-pass", "legacy"), default="two-pass", help="Default: two-pass source-turn review")
    parser.add_argument("--speaker-review-num-ctx", type=int, help="Independent speaker review context (default 98304 or SPEAKER_REVIEW_NUM_CTX)")
    parser.add_argument("--speaker-review-tokenizer", type=Path, help="Matching local tokenizer.json for speaker input budgeting")
    parser.add_argument("--speaker-review-window", type=int, help="Discovery window to review when the complete meeting exceeds budget")

    parser.add_argument("--synthesis-mode", choices=("map-reduce", "whole"), default="map-reduce", help="Meeting only: opt-in whole-meeting experiment; map/reduce remains default")
    parser.add_argument("--synthesis-num-ctx", type=int, help="Independent whole-meeting context (default 98304 or MEETING_SYNTHESIS_NUM_CTX)")
    parser.add_argument("--synthesis-model", help="Whole-meeting model override (default MEETING_SYNTHESIS_MODEL or selected reduce model)")
    parser.add_argument("--synthesis-tokenizer", type=Path, help="Matching local tokenizer.json for whole-meeting budgeting")
    parser.add_argument("--experiment-output-dir", type=Path, help="New isolated destination required for whole-meeting experiments")

    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument("--synthesis-preflight", action="store_true", help="Read-only whole-mode preparation and context accounting; no models, locks or outputs")

    inspection.add_argument("--synthesis-validate-response", type=Path, help="Read-only offline validation of a private source-bound evidence response; no models or public outputs")
    parser.add_argument("--synthesis-retain-response", action="store_true", help="Explicitly retain a private source-bound evidence response for offline debugging")

    args = _args if _args is not None else parser.parse_args()
    synthesis_options = args.synthesis_retain_response or args.synthesis_validate_response is not None or args.synthesis_preflight or any(getattr(args, name) is not None for name in ("synthesis_num_ctx", "synthesis_model", "synthesis_tokenizer", "experiment_output_dir"))
    if args.profile != "meeting" and (args.synthesis_mode != "map-reduce" or synthesis_options):
        parser.error("Whole-meeting synthesis is available only in meeting mode")
    if args.synthesis_mode == "map-reduce" and synthesis_options:
        parser.error("Experimental options require --synthesis-mode whole")
    if args.synthesis_mode == "whole" and args.experiment_output_dir is None and not args.synthesis_preflight and args.synthesis_validate_response is None:
        parser.error("Whole-meeting synthesis requires --experiment-output-dir")
    if args.synthesis_retain_response and (args.synthesis_preflight or args.synthesis_validate_response is not None):
        parser.error("Response retention requires an inference run; inspection modes never write artifacts")
    if args.synthesis_mode == "whole" and (args.suggest_speakers or args.suggest_speakers_llm or args.speaker_roster or args.speaker_suggestion_model):
        parser.error("Run advisory speaker review separately; whole synthesis uses approved identities only")
    review_options = args.speaker_review_mode != "two-pass" or args.speaker_review_num_ctx is not None or args.speaker_review_tokenizer is not None or args.speaker_review_window is not None
    if args.profile != "meeting" and (args.suggest_speakers or args.speaker_roster is not None or args.suggest_speakers_llm or args.speaker_suggestion_model or review_options):
        parser.error("Speaker suggestions are available only in meeting mode")
    if (args.suggest_speakers_llm or args.speaker_suggestion_model) and not args.suggest_speakers:
        parser.error("LLM speaker review requires --suggest-speakers")
    if args.speaker_suggestion_model and not args.suggest_speakers_llm:
        parser.error("Speaker-review model override requires --suggest-speakers-llm")
    if review_options and not args.suggest_speakers_llm:
        parser.error("Speaker-review options require --suggest-speakers-llm")
    try:
        resolve_model_defaults(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.synthesis_mode == "whole":
        from meeting_postprocess.whole_synthesis import run_experiment, run_preflight, run_offline_validation
        if args.synthesis_preflight:
            return run_preflight(args, sys.modules[__name__])
        if args.synthesis_validate_response is not None:
            return run_offline_validation(args, sys.modules[__name__])
        return run_experiment(args, sys.modules[__name__])

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

    summaries_dir = _summary_dir or resolve_summary_dir(transcript_dir, summary_root)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    aliases: dict[str, str] = {}
    redaction_warnings = []
    if profile == "meeting":
        try:
            redacted = redact_chunks(chunks)
            write_private_redactions(summaries_dir, redacted.redactions)
            redaction_warnings = redacted.warnings
            if redacted.redactions:
                # A rerun must not leave older, unredacted derived files visible
                # if a model/preparation error interrupts the new generation.
                for filename in (
                    "summary.md", "action-items.md", "minutes-draft.md",
                    "chunk_summaries.jsonl", "meeting_sections.jsonl",
                    "minutes-qa.md", "minutes-qa.json",
                    PRIVATE_SUGGESTIONS_FILENAME,
                    TURNS_FILE, CONFLICTS_FILE,
                ):
                    (summaries_dir / filename).unlink(missing_ok=True)
            if redaction_warnings:
                write_report(summaries_dir, redaction_warnings)
                print(f"[qa] {len(redaction_warnings)} spoken-redaction warning(s); see minutes-qa.md", flush=True)
            aliases = load_aliases(transcript_dir, args.speaker_aliases)
            catalog = None
            if args.suggest_speakers_llm and args.speaker_review_mode == "two-pass" or (transcript_dir / CORRECTIONS_FILE).exists():
                catalog = turn_catalog(transcript_dir, chunks)
                corrections = correction_document(transcript_dir)
                turns = effective_turns(catalog, corrections, aliases)
                redacted.chunks = corrected_chunks(redacted.chunks, catalog, corrections, aliases)
            if args.suggest_speakers:
                roster = load_roster(transcript_dir, args.speaker_roster)
                speaker_chunks = speaker_input(chunks)
                report = build_speaker_suggestions(speaker_chunks, aliases, roster, str(transcript_dir.resolve()))
                write_private_json(summaries_dir / PRIVATE_SUGGESTIONS_FILENAME, report)
                if args.suggest_speakers_llm and args.speaker_review_mode == "legacy":
                    report = review_speakers(report, speaker_chunks, aliases, roster, ollama_url=args.ollama_url, model=args.speaker_suggestion_model or args.reduce_model, num_ctx=args.reduce_num_ctx, keep_alive=args.keep_alive)
                    write_private_json(summaries_dir / PRIVATE_SUGGESTIONS_FILENAME, report)
                elif args.suggest_speakers_llm:
                    write_private_json(summaries_dir / TURNS_FILE, {**catalog, "turns": turns})
                    report = review_turns(report, turns, aliases, roster, ollama_url=args.ollama_url, model=args.speaker_suggestion_model or args.reduce_model, options=speaker_review_settings(args.speaker_review_num_ctx, args.speaker_review_tokenizer, args.speaker_review_window))
                    write_private_json(summaries_dir / PRIVATE_SUGGESTIONS_FILENAME, report)
                    print(f"[speaker-review] Local LLM review: {report['llm_review']['status']}; advisory only", flush=True)
            chunks = prepare_chunks(redacted.chunks, aliases)
        except (OSError, ValueError, TypeError) as exc:
            print(f"ERROR: Meeting preparation failed: {exc}", file=sys.stderr)
            return 2
        if not chunks:
            if not redacted.redactions:
                print("ERROR: No meeting text found in transcript chunks", file=sys.stderr)
                return 2
            # Deliberately withholding all content is a valid meeting run and
            # must not create a model prompt or leave stale outputs behind.
            (summaries_dir / "meeting_sections.jsonl").write_text("", encoding="utf-8")
            (summaries_dir / "chunk_summaries.jsonl").write_text("", encoding="utf-8")
            (summaries_dir / "summary.md").write_text("# Meeting Summary\nNo non-redacted meeting content available.\n", encoding="utf-8")
            (summaries_dir / "action-items.md").write_text("# Action Items\nNo clear action items identified.\n", encoding="utf-8")
            (summaries_dir / "minutes-draft.md").write_text("# Draft Minutes\nNo non-redacted meeting content available.\n", encoding="utf-8")
            write_report(summaries_dir, redaction_warnings)
            print("[done] all meeting content withheld; no Ollama calls made", flush=True)
            return 0
        if args.keep_recap and any(chunk["meeting_section"] == RECAP for chunk in chunks):
            if not (prompt_dir / "recap_prompt.txt").exists():
                print(f"ERROR: Missing reduce prompt: {prompt_dir / 'recap_prompt.txt'}", file=sys.stderr)
                return 2

    if profile == "meeting":
        # Keep the original transcript/chunk index intact and save evidence for
        # every classified run, including sections excluded from draft minutes.
        (summaries_dir / "meeting_sections.jsonl").write_text(
            "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
            encoding="utf-8",
        )

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
    if profile == "meeting":
        print(f"[info] speaker_alias_count={len(aliases)}", flush=True)
        print(f"[info] keep_recap={args.keep_recap}", flush=True)

    chunk_summaries: list[dict[str, Any]] = []
    current_source = "\n".join(
        chunk["text"] for chunk in chunks
        if chunk.get("meeting_section") in {BUSINESS, ADJOURNMENT}
    ) if profile == "meeting" else ""

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
                **({"usage_callback": _usage_callback} if _usage_callback is not None else {}),
            )
        except Exception as exc:
            print(f"ERROR: Ollama map-stage failed for {chunk_id}: {exc}", file=sys.stderr)
            return 1

        if profile == "meeting":
            summary_text = prepare_text(summary_text, aliases)
            if chunk["meeting_section"] in {BUSINESS, ADJOURNMENT}:
                summary_text = filter_completed_request_tasks(summary_text, current_source, action_sections_only=True)
                summary_text = correct_adjournment_roles(summary_text, current_source)

        row = {
            "chunk_id": chunk_id,
            "file_name": chunk.get("file_name", transcript_dir.name),
            "start_time": chunk.get("start_time"),
            "end_time": chunk.get("end_time"),
            "speaker_span": chunk.get("speaker_span", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "summary": summary_text,
        }
        if profile == "meeting":
            for key in ("source_chunk_id", "meeting_section", "section_evidence"):
                row[key] = chunk[key]
        chunk_summaries.append(row)

    with chunk_summaries_path.open("w", encoding="utf-8") as f:
        for row in chunk_summaries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[info] wrote {chunk_summaries_path}", flush=True)

    combined = build_reduce_input(chunk_summaries)
    current_meeting_combined = combined
    recap_combined = ""
    commitments = []
    if profile == "meeting":
        commitments = commitment_evidence(chunks)
        combined = build_reduce_input([
            row for row in chunk_summaries if row["meeting_section"] in {RECAP, BUSINESS, ADJOURNMENT}
        ]) or "No historical recap, current meeting business or adjournment was identified."
        current_meeting_combined = build_reduce_input([
            row for row in chunk_summaries if row["meeting_section"] in {BUSINESS, ADJOURNMENT}
        ]) or "No explicit current meeting business or adjournment was identified."
        recap_combined = build_reduce_input([
            row for row in chunk_summaries if row["meeting_section"] == RECAP
        ])

    for output_filename, template_name in profile_cfg["outputs"]:
        template_path = prompt_dir / template_name
        if not template_path.exists():
            print(f"ERROR: Missing reduce prompt: {template_path}", file=sys.stderr)
            return 2

        print(f"[reduce] generating {output_filename}", flush=True)
        reduce_prompt = build_reduce_prompt(
            prompt_dir, template_name,
            current_meeting_combined
            if output_filename in {"minutes-draft.md", "action-items.md"}
            else combined,
        )
        if profile == "meeting" and output_filename in {"minutes-draft.md", "action-items.md"} and commitments:
            reduce_prompt += "\n\nSource-backed future commitment evidence (current meeting only):\n"
            reduce_prompt += "\n".join("- " + evidence for evidence in commitments)
            reduce_prompt += "\nUse this supplemental source evidence to check for tasks omitted from chunk summaries. These quotations are not pre-approved action items. Preserve speaker ownership, conditions and qualifications such as 'try to'; do not infer additional owners or assignments."
            reduce_prompt += " Keep multi-step workflows separate by actor. Preserve an explicit recipient named in the speaker's own task; leave an unstated recipient unstated. A later third party's delivery or redaction step must not supply the speaker's recipient or redaction duty, even if the map summary merges those steps."
        if profile == "meeting" and output_filename in {"minutes-draft.md", "summary.md"}:
            announcements = adjournment_announcements(current_source)
            if announcements:
                reduce_prompt += "\n\nExplicit named adjournment announcements from the current transcript:\n"
                reduce_prompt += "\n".join("- " + announcement.render() for announcement in announcements)
                reduce_prompt += "\nThese names are the announced mover/seconder, not the announcing speaker's identity. Preserve the names when recording the motion."

        try:
            content = call_ollama(
                ollama_url=args.ollama_url,
                model=args.reduce_model,
                prompt=reduce_prompt,
                system=reduce_system,
                keep_alive=args.keep_alive,
                temperature=args.temperature,
                num_ctx=args.reduce_num_ctx,
                **({"usage_callback": _usage_callback} if _usage_callback is not None else {}),
            )
            if profile == "meeting" and output_filename == "minutes-draft.md" and args.keep_recap and recap_combined:
                recap = call_ollama(
                    ollama_url=args.ollama_url,
                    model=args.reduce_model,
                    prompt=build_reduce_prompt(prompt_dir, "recap_prompt.txt", recap_combined),
                    system=reduce_system,
                    keep_alive=args.keep_alive,
                    temperature=args.temperature,
                    num_ctx=args.reduce_num_ctx,
                    **({"usage_callback": _usage_callback} if _usage_callback is not None else {}),
                )
                content = insert_recap(content, recap)
        except Exception as exc:
            print(
                f"ERROR: Ollama reduce-stage failed for {output_filename}: {exc}",
                file=sys.stderr,
            )
            return 1

        if profile == "meeting":
            content = prepare_text(content, aliases)
            if output_filename in {"minutes-draft.md", "summary.md"}:
                content = correct_adjournment_roles(content, current_source)
            if output_filename in {"minutes-draft.md", "action-items.md"}:
                content = filter_completed_request_tasks(
                    content, current_source, action_sections_only=output_filename == "minutes-draft.md",
                )
            content = strip_chunk_references(content, [chunk["file_name"] for chunk in chunks if chunk.get("file_name")])
            content = strip_private_references(content)
        out_path = summaries_dir / output_filename
        out_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        print(f"[info] wrote {out_path}", flush=True)
        if profile == "meeting" and output_filename == "minutes-draft.md":
            source = "\n".join(chunk["text"] for chunk in chunks)
            findings = check_minutes(content, source=source) + redaction_warnings
            write_report(summaries_dir, findings)
            print(f"[qa] {len(findings)} finding(s); see {summaries_dir / 'minutes-qa.md'}", flush=True)

    print("[done] summary generation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
