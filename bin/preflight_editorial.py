#!/usr/bin/env python3
"""Read-only editorial context accounting from saved prepared source/maps/register.

No inference, GPU admission/locks, transcript preparation or output generation.
The register is an offline fixture, not a prediction of the next live response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from meeting_postprocess import editorial as ed
from meeting_postprocess.commitments import commitment_evidence, source_context_evidence
from meeting_postprocess.sections import BUSINESS, ADJOURNMENT, RECAP
from ollama_session_summary import build_reduce_input, load_jsonl


def preflight(source_dir, register_path, tokenizer, context, keep_recap=False):
    prompts = Path(__file__).resolve().parents[1] / "prompts/meeting"
    chunks = load_jsonl(source_dir / "meeting_sections.jsonl")
    summaries = load_jsonl(source_dir / "chunk_summaries.jsonl")
    register = json.loads(register_path.read_text(encoding="utf-8"))
    records = ed.source_records(chunks)
    if ed.digest(records) != register["source_hash"] or any(
            records.get(row["id"]) != row for item in register["items"] for row in item["evidence"]):
        raise ed.EditorialFailure("preflight_source_register_mismatch")
    combined = build_reduce_input(summaries)
    current = build_reduce_input([s for s in summaries if s["meeting_section"] in {BUSINESS, ADJOURNMENT}])
    recap = build_reduce_input([s for s in summaries if s["meeting_section"] == RECAP])
    commitments = commitment_evidence(chunks, include_context=True)
    context_evidence = source_context_evidence(chunks)
    budget = ed.RequestBudget(tokenizer, context, (prompts / "reduce_system.txt").read_text(encoding="utf-8").strip())
    budget.measure("register", ed.register_prompt(prompts, records, current, commitments, context_evidence), True)
    budget.measure("notes", ed.notes_prompt(prompts, records, combined, register), True)
    budget.measure("detailed", ed.detailed_prompt(prompts, records, current, register))
    if keep_recap and recap:
        budget.measure("recap", (prompts / "recap_prompt.txt").read_text(encoding="utf-8").replace("{chunk_summaries}", recap))
    compact = ed.model_register(register)
    return {"mode": "zero_inference_editorial_preflight", "inference_calls": 0, "source_hash": register["source_hash"],
            "register_file_sha256": ed.hashlib.sha256(register_path.read_bytes()).hexdigest(),
            "private_register_file_bytes": register_path.stat().st_size,
            "register_items": len(register["items"]), "compact_register_bytes": len(json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode()),
            "eligible_source_records": len(records), "source_excerpt_gaps": ed.source_excerpts(records)["omitted_oversized_source_ids"],
            "requests": budget.measurements, "all_fit": all(row["fits"] for row in budget.measurements),
            "limitation": "saved offline register; live reduction prompts must each pass the same check",
            "publication_status": "review_hold"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Saved prepared meeting_sections.jsonl and chunk_summaries.jsonl")
    parser.add_argument("register", type=Path, help="Full private canonical register for the same prepared source")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--num-ctx", type=int, required=True, help="Existing configured reduce context")
    parser.add_argument("--keep-recap", action="store_true")
    args = parser.parse_args()
    try:
        result = preflight(args.source_dir, args.register, args.tokenizer, args.num_ctx, args.keep_recap)
    except ed.EditorialFailure as exc:
        print(json.dumps({"error": str(exc), "inference_calls": 0, "publication_status": "review_hold"}))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["all_fit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
