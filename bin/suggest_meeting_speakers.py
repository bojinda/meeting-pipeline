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
    suggest.add_argument("--llm", "--suggest-speakers-llm", action="store_true", help="Request one bounded local Ollama review under the GPU1 lock")
    suggest.add_argument("--speaker-suggestion-model")
    suggest.add_argument("--reduce-model", help="Meeting model override; otherwise use existing meeting environment defaults")
    suggest.add_argument("--reduce-num-ctx", type=int)
    suggest.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://192.168.0.105:11434"))
    approve = commands.add_parser("approve", help="Merge only explicitly selected suggestions into approved aliases")
    approve.add_argument("transcript_dir", type=Path)
    approve.add_argument("--suggestions", type=Path)
    approve.add_argument("--speaker-aliases", type=Path)
    approve.add_argument("--speaker-roster", type=Path, help="Use the same private roster as the reviewed suggestions")
    approve.add_argument("--approve", action="append", required=True, metavar="SPEAKER_XX")
    args = parser.parse_args()
    try:
        transcript = args.transcript_dir.resolve()
        approved = load_aliases(transcript, args.speaker_aliases)
        root = Path(os.environ.get("MEETING_SUMMARIES_ROOT", str(Path(__file__).resolve().parents[1] / "meeting-summaries"))).expanduser()
        output_dir = root / transcript.name
        if args.command == "suggest":
            output_dir = args.output_dir or output_dir
            index = transcript / "chunks_out" / "transcript_chunks.jsonl"
            chunks = [json.loads(line) for line in index.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            chunks = speaker_input(chunks)
            report = build_speaker_suggestions(chunks, approved, load_roster(transcript, args.speaker_roster), str(transcript))
            write_private_json(output_dir / PRIVATE_SUGGESTIONS_FILENAME, report)
            if args.llm:
                defaults = argparse.Namespace(profile="meeting", map_model=None, reduce_model=args.reduce_model, map_num_ctx=None, reduce_num_ctx=args.reduce_num_ctx)
                resolve_model_defaults(defaults)
                report = review_speakers(report, chunks, approved, load_roster(transcript, args.speaker_roster), ollama_url=args.ollama_url, model=args.speaker_suggestion_model or defaults.reduce_model, num_ctx=defaults.reduce_num_ctx)
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
                    raise ValueError("Selected review remains unresolved or low confidence; verify aliases manually")
                if not SPEAKER_LABEL.fullmatch(label) or not row or not row.get("suggested_name") or not row.get("evidence") or row.get("confidence") not in {"high", "medium"} or row.get("ambiguity") or len(row.get("candidates", [])) != 1:
                    raise ValueError("Selected label has no unambiguous, evidenced suggestion; edit aliases manually after review")
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
