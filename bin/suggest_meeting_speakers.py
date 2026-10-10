#!/usr/bin/env python3
"""Private text-only meeting suggestions and explicit operator approval."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from meeting_postprocess.aliases import load_aliases
from meeting_postprocess.normalization import SPEAKER_LABEL
from meeting_postprocess.speaker_review import reground_review, review_speakers
from meeting_postprocess.speaker_turn_review import review_turns, settings, source_events
from meeting_postprocess.speaker_turns import turn_catalog, correction_document, correction_conflicts, effective_turns, approve_turn, remove_turn, read_index, TURNS_FILE, CONFLICTS_FILE, INSPECTION_FILE
from ollama_session_summary import resolve_model_defaults
from meeting_postprocess.speaker_suggestions import (
    PRIVATE_SUGGESTIONS_FILENAME, _plain_name, build_speaker_suggestions,
    load_roster, source_digest, speaker_input, write_private_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    suggest = commands.add_parser("suggest", help="Write advisory suggestions without models or alias changes")
    suggest.add_argument("transcript_dir", type=Path)
    suggest.add_argument("--speaker-roster", type=Path)
    suggest.add_argument("--speaker-aliases", type=Path)
    suggest.add_argument("--output-dir", type=Path)
    suggest.add_argument("--llm", "--suggest-speakers-llm", action="store_true", help="Request bounded discovery and verification under one GPU1 lock")
    suggest.add_argument("--speaker-suggestion-model")
    suggest.add_argument("--speaker-review-mode", choices=("two-pass", "legacy"), default="two-pass", help="Default: two-pass turn review; legacy retains compatibility with older label reviews")
    suggest.add_argument("--speaker-review-num-ctx", type=int, help="Independent speaker context (default 98304 or SPEAKER_REVIEW_NUM_CTX)")
    suggest.add_argument("--speaker-review-tokenizer", type=Path, help="Matching local tokenizer.json; no downloads")
    suggest.add_argument("--speaker-review-window", type=int, help="Select an overlapping discovery window (default 0)")
    suggest.add_argument("--reduce-model", help="Meeting model override; otherwise use existing meeting environment defaults")
    suggest.add_argument("--reduce-num-ctx", type=int)
    suggest.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://192.168.0.105:11434"))
    approve = commands.add_parser("approve", help="Merge only explicitly selected suggestions into approved aliases")
    approve.add_argument("transcript_dir", type=Path)
    approve.add_argument("--suggestions", type=Path)
    approve.add_argument("--speaker-aliases", type=Path)
    approve.add_argument("--speaker-roster", type=Path, help="Use the same private roster as the reviewed suggestions")
    approve.add_argument("--approve", action="append", required=True, metavar="SPEAKER_XX")
    alias = commands.add_parser("set-alias", help="Record an operator-confirmed name for one label in this meeting; no model guesses")
    alias.add_argument("transcript_dir", type=Path)
    alias.add_argument("--speaker-aliases", type=Path)
    alias.add_argument("--speaker-label", required=True, metavar="SPEAKER_XX")
    alias.add_argument("--name", required=True, help="Use only when this label consistently represents the known person; otherwise approve-turn")
    for name in ("inspect-turns", "approve-turn", "remove-turn", "turn-conflicts"):
        action = commands.add_parser(name, help="Private source-turn inspection/correction; never runs WhisperX or models")
        action.add_argument("transcript_dir", type=Path)
        action.add_argument("--speaker-aliases", type=Path)
        if name in {"inspect-turns", "turn-conflicts"}:
            action.add_argument("--output-dir", type=Path)
        if name != "turn-conflicts":
            action.add_argument("--turn-id", required=name in {"approve-turn", "remove-turn"})
        if name == "approve-turn":
            action.add_argument("--name", required=True, help="Explicitly approved identity for this exact source turn")
    args = parser.parse_args()
    try:
        transcript = args.transcript_dir.resolve()
        approved = load_aliases(transcript, args.speaker_aliases)
        root = Path(os.environ.get("MEETING_SUMMARIES_ROOT", str(Path(__file__).resolve().parents[1] / "meeting-summaries"))).expanduser()
        output_dir = root / transcript.name
        if args.command == "set-alias":
            catalog = turn_catalog(transcript)
            if not SPEAKER_LABEL.fullmatch(args.speaker_label) or not any(row["source_speaker"] == args.speaker_label for row in catalog["turns"]):
                raise ValueError("Selected speaker label is absent or fully redacted in this meeting")
            updated = {**approved, args.speaker_label: _plain_name(args.name)}
            write_private_json(args.speaker_aliases or transcript / "speaker_aliases.json", updated)
            print("Recorded the operator-confirmed name for the selected meeting label. Generate notes after finishing speaker review.")
        elif args.command in {"inspect-turns", "approve-turn", "remove-turn", "turn-conflicts"}:
            if args.command == "remove-turn":
                remove_turn(transcript, args.turn_id)
                print("Removed the explicitly selected private turn correction.")
                return 0
            catalog = turn_catalog(transcript)
            if args.command == "approve-turn":
                approve_turn(transcript, catalog, args.turn_id, args.name)
                print("Approved the explicitly selected source turn. Rerun meeting postprocessing.")
                return 0
            document = correction_document(transcript)
            conflicts = correction_conflicts(catalog, document, approved)
            output_dir = args.output_dir or output_dir
            if args.command == "turn-conflicts":
                events = source_events(catalog["turns"], approved, load_roster(transcript))
                for label in sorted({event["speaker_label"] for event in events}):
                    items = [event for event in events if event["speaker_label"] == label]
                    names = sorted({name for event in items for name in event["candidate_names"]})
                    if len(names) > 1 or any(event.get("uncertain_source_attribution") for event in items):
                        conflicts.append({"type": "uncertain_or_conflicting_source_identity", "speaker_label": label, "candidates": names, "turn_ids": sorted({tid for event in items for tid in event["target_turn_ids"]}), "blocking": False})
                payload = {"meeting_id": catalog["meeting_id"], "source_digest": catalog["source_digest"], "conflicts": conflicts}
                filename = CONFLICTS_FILE
            else:
                rows = effective_turns(catalog, document, approved) if not any(row["blocking"] for row in conflicts) else catalog["turns"]
                if args.turn_id:
                    rows = [row for row in rows if row["turn_id"] == args.turn_id]
                    if not rows:
                        raise ValueError("Requested source turn is absent or redacted")
                payload, filename = {**catalog, "turns": rows, "conflicts": conflicts}, TURNS_FILE
            write_private_json(output_dir / filename, payload)
            if args.command == "inspect-turns":
                inspection = output_dir / INSPECTION_FILE
                if inspection.is_symlink():
                    raise OSError("Private speaker output cannot be a symbolic link")
                lines = ["PRIVATE SPEAKER REVIEW — this meeting only", "",
                         "Leave unknown speakers unchanged. Use set-alias only for a consistently identified label; use approve-turn for a specific passage.", ""]
                if any(row["blocking"] for row in conflicts):
                    lines += ["Existing corrections are stale or invalid; the identities below are original source labels.", ""]
                for row in rows:
                    lines += [f"Turn: {row['turn_id']} | Source label: {row['source_speaker']} | Current name: {row.get('effective_speaker', row['source_speaker'])}",
                              f"Time: {row['start_time']}–{row['end_time']} ({row['time_precision']})", row["text"], ""]
                with os.fdopen(os.open(inspection, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "w", encoding="utf-8") as stream:
                    stream.write("\n".join(lines) + "\n")
                os.chmod(inspection, 0o600)
                print(f"Readable private turn list: {inspection}; unknown speakers can be left unchanged.")
            print(f"Wrote private {filename}; inspect locally.")
        elif args.command == "suggest":
            output_dir = args.output_dir or output_dir
            source_chunks = read_index(transcript)
            chunks = speaker_input(source_chunks)
            if (output_dir / TURNS_FILE).exists():
                fresh_catalog = turn_catalog(transcript, source_chunks)
                write_private_json(output_dir / TURNS_FILE, fresh_catalog)
            (output_dir / CONFLICTS_FILE).unlink(missing_ok=True)
            report = build_speaker_suggestions(chunks, approved, load_roster(transcript, args.speaker_roster), str(transcript))
            write_private_json(output_dir / PRIVATE_SUGGESTIONS_FILENAME, report)
            if args.llm:
                defaults = argparse.Namespace(profile="meeting", map_model=None, reduce_model=args.reduce_model, map_num_ctx=None, reduce_num_ctx=args.reduce_num_ctx)
                resolve_model_defaults(defaults)
                if args.speaker_review_mode == "legacy":
                    report = review_speakers(report, chunks, approved, load_roster(transcript, args.speaker_roster), ollama_url=args.ollama_url, model=args.speaker_suggestion_model or defaults.reduce_model, num_ctx=defaults.reduce_num_ctx)
                else:
                    catalog = turn_catalog(transcript, source_chunks)
                    turns = effective_turns(catalog, correction_document(transcript), approved)
                    write_private_json(output_dir / TURNS_FILE, {**catalog, "turns": turns})
                    report = review_turns(report, turns, approved, load_roster(transcript, args.speaker_roster), ollama_url=args.ollama_url, model=args.speaker_suggestion_model or defaults.reduce_model, options=settings(args.speaker_review_num_ctx, args.speaker_review_tokenizer, args.speaker_review_window))
                write_private_json(output_dir / PRIVATE_SUGGESTIONS_FILENAME, report)
                print(f"Local LLM review: {report['llm_review']['status']}; inspect the private review.")
            print("Wrote private speaker-suggestions.json (advisory only; aliases unchanged).")
        else:
            review = args.suggestions or output_dir / PRIVATE_SUGGESTIONS_FILENAME
            report = json.loads(review.read_text(encoding="utf-8-sig"))
            if report.get("meeting_id") != str(transcript) or report.get("advisory_only") is not True:
                raise ValueError("Suggestion review does not belong to this meeting")
            index = transcript / "chunks_out" / "transcript_chunks.jsonl"
            current_chunks = [json.loads(line) for line in index.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if report.get("source_digest") != source_digest(current_chunks):
                raise ValueError("Visible transcript changed since review; regenerate suggestions before approval")
            grounded = reground_review(report, current_chunks, approved, load_roster(transcript, args.speaker_roster))
            records = {row["speaker_label"]: row for row in grounded["suggestions"]}
            displayed = {row["speaker_label"]: row for row in report["suggestions"]}
            updated = dict(approved)
            for label in args.approve:
                row = records.get(label)
                selected = displayed.get(label)
                if not selected or not selected.get("suggested_name") or selected.get("confidence") not in {"high", "medium"} or selected.get("ambiguity"):
                    raise ValueError("Selected review remains unresolved or low confidence; use set-alias or approve-turn after identifying the speaker")
                if not SPEAKER_LABEL.fullmatch(label) or not row or not row.get("suggested_name") or not row.get("evidence") or row.get("confidence") not in {"high", "medium"} or row.get("ambiguity") or len(row.get("candidates", [])) != 1:
                    raise ValueError("Selected label has no unambiguous, evidenced suggestion; use set-alias or approve-turn after identifying the speaker")
                if row["suggested_name"] != selected["suggested_name"]:
                    raise ValueError("Displayed identity does not match current grounded review")
                name = _plain_name(row["suggested_name"])
                if name not in row.get("candidates", []):
                    raise ValueError("Selected suggestion is not an evidenced candidate")
                if label in approved and approved[label] != name:
                    raise ValueError("Approval cannot overwrite an existing approved alias")
                updated[label] = name
            target = args.speaker_aliases or transcript / "speaker_aliases.json"
            write_private_json(target, updated)
            print(f"Approved {len(set(args.approve))} explicitly selected speaker suggestion(s). Rerun meeting postprocessing.")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"ERROR: Speaker review failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
