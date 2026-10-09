#!/usr/bin/env python3
"""Private planning/preflight over an existing meeting index; no transcription."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import ollama_session_summary as engine
from meeting_postprocess import chunking
from meeting_postprocess.aliases import load_aliases
from meeting_postprocess.gpu_admission import approved_input, managed_generate
from meeting_postprocess.redaction import redact_chunks
from meeting_postprocess.sections import prepare_chunks
from meeting_postprocess.speaker_suggestions import write_private_json
from meeting_postprocess.speaker_turns import CORRECTIONS_FILE, turn_catalog, correction_document, corrected_chunks


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.exists() else None


def source_bindings(directory, alias_path, correction_path):
    return {"transcript_index": file_hash(directory / "chunks_out/transcript_chunks.jsonl"),
            "approved_aliases": file_hash(alias_path), "approved_turn_corrections": file_hash(correction_path)}


def prepare(directory, alias_option=None):
    raw = engine.load_jsonl(directory / "chunks_out/transcript_chunks.jsonl")
    redacted = redact_chunks(raw)
    bound, alias_path = approved_input("approved_aliases", alias_option)
    aliases = {} if bound and alias_path is None else load_aliases(directory, alias_path)
    _, correction_path = approved_input("approved_turn_corrections", directory / CORRECTIONS_FILE)
    if correction_path is not None and correction_path.exists():
        redacted.chunks = corrected_chunks(redacted.chunks, turn_catalog(directory, raw), correction_document(directory, correction_path), aliases)
    ids = [str(row.get("chunk_id", index)) for index, row in enumerate(raw, 1)]
    blocked = set()
    for record in redacted.redactions:
        blocked.update(ids[ids.index(str(record["start_chunk_id"])):ids.index(str(record["end_chunk_id"])) + 1])
    return prepare_chunks(redacted.chunks, aliases), source_bindings(directory, alias_path if bound else (alias_path or directory / "speaker_aliases.json"), correction_path), blocked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_dir", type=Path)
    parser.add_argument("--mode", choices=("baseline", "medium", "adaptive"), default="medium")
    parser.add_argument("--output-dir", type=Path, help="New private plan directory; required unless --preflight")
    parser.add_argument("--preflight", action="store_true", help="No models or file writes; report coverage and token accounting")
    parser.add_argument("--planner", action="store_true", help="One optional boundary-only inference, within an existing managed GPU1 stage")
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--map-output-tokens", type=int, default=4096)
    parser.add_argument("--map-model")
    parser.add_argument("--reduce-model")
    parser.add_argument("--map-num-ctx", type=int)
    parser.add_argument("--reduce-num-ctx", type=int)
    parser.add_argument("--planner-num-ctx", type=int, default=98304)
    parser.add_argument("--tokenizer", type=Path, default=os.environ.get("MEETING_CHUNK_TOKENIZER") or None)
    parser.add_argument("--speaker-aliases", type=Path)
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--keep-alive", default=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"))
    args = parser.parse_args(argv)
    args.profile = "meeting"
    try:
        engine.resolve_model_defaults(args)
        if not args.map_num_ctx or not 2048 <= args.map_num_ctx <= 98304 or not 2048 <= args.planner_num_ctx <= 98304:
            raise chunking.ChunkingFailure("explicit_safe_context_required")
        ceiling = args.max_words or (6000 if args.mode == "medium" else 10000)
        if not 1 <= ceiling <= 10000 or not 256 <= args.map_output_tokens < args.map_num_ctx:
            raise chunking.ChunkingFailure("invalid_word_or_output_budget")
        directory = args.transcript_dir.resolve()
        target = args.output_dir.resolve() if args.output_dir else None
        if not args.preflight and (target is None or target.exists() or target == directory or target.is_relative_to(directory)):
            raise chunking.ChunkingFailure("new_private_output_directory_required")
        _, tokenizer = approved_input("chunk_tokenizer", args.tokenizer) if os.environ.get("AIHUB_GPU_STAGE_INPUT_SNAPSHOTS") else (False, args.tokenizer)
        count = chunking.counter(tokenizer or "")
        chunks, bindings, blocked = prepare(directory, args.speaker_aliases)
        prompt_dir = Path(__file__).resolve().parents[1] / "prompts/meeting"
        prompt = lambda row: engine.build_chunk_prompt(prompt_dir, row)
        system = (prompt_dir / "chunk_system.txt").read_text().strip()
        settings = {"mode": args.mode, "max_words": ceiling, "map_context": args.map_num_ctx,
                    "map_output": args.map_output_tokens, "map_model": args.map_model,
                    "tokenizer_hash": file_hash(tokenizer), "planner": args.planner,
                    "planner_context": args.planner_num_ctx}
        def infer(text, instructions, schema, output, context):
            from meeting_postprocess.speaker_review import _local_url, _local_model
            _local_url(args.ollama_url)
            _local_model(args.reduce_model)
            result = managed_generate(args.ollama_url, {"model": args.reduce_model, "system": instructions,
                "prompt": text, "format": schema, "think": False, "stream": False, "keep_alive": args.keep_alive,
                "options": {"num_ctx": context, "num_predict": output, "temperature": 0}}, 3600)
            if result is None:
                raise chunking.ChunkingFailure("planner_not_in_managed_gpu1_stage")
            return result
        plan = chunking.make_plan(chunks, bindings, settings, count, prompt, system, blocked,
                                  planner_call=infer if args.planner and not args.preflight else None, planner_preflight=args.preflight)
        # Planning never creates public documents or modifies the source index.
        if not args.preflight:
            target.mkdir(mode=0o700, parents=True)
            write_private_json(target / chunking.PLAN_FILE, plan)
        print(json.dumps({"coverage": plan["coverage"], "groups": len(plan["groups"]), "budgets": plan["budgets"],
                          "token_count_method": plan["token_count_method"], "planner": plan["planner"], "source_hash": plan["source_hash"], "plan_hash": plan["plan_hash"]}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": str(exc) if isinstance(exc, chunking.ChunkingFailure) else "invalid_source_or_configuration"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
