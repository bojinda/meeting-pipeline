#!/usr/bin/env python3
"""Source-bound CPU rendering of completed saved replies; never submits a job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import ollama_session_summary as engine
from meeting_postprocess import editorial as ed, editorial_checkpoint as cp
from meeting_postprocess.commitments import commitment_evidence
from meeting_postprocess.qa import Finding


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(value, code):
    if not value:
        raise ed.EditorialFailure(code)


def load_bound(path, expected):
    raw = path.read_bytes()
    need(sha(raw) == expected, "offline_manifest_changed")
    manifest = json.loads(raw)
    source = Path(manifest["source_directory"]).resolve()
    files = {}
    for name, digest in manifest["files"].items():
        selected = source / name
        need(selected.resolve().is_relative_to(source) and not selected.is_symlink(), "offline_invalid_source_path")
        data = selected.read_bytes()
        need(sha(data) == digest, "offline_source_changed")
        files[name] = data
    job_raw = Path(manifest["source_job_path"]).read_bytes()
    need(sha(job_raw) == manifest["source_job_sha256"], "offline_job_changed")
    job = json.loads(job_raw)
    need(job["state"] in {"failed", "success"} and job.get("cleanup_verified") is True
         and job.get("host_requests") and all(r["state"] == "completed" for r in job["host_requests"]), "offline_source_job_unsettled")
    checkpoint = json.loads(files[cp.FILE])
    need(job["request_id"] == checkpoint["source_stage_id"], "offline_job_binding_mismatch")
    owners = json.loads(Path(manifest["ownership_path"]).read_bytes())["owners"]
    need(not any(owner["job_id"] == job["request_id"] for owner in owners.values()), "offline_source_still_owned")
    chunks = [json.loads(line) for line in files["meeting_sections.jsonl"].decode("utf-8").splitlines() if line.strip()]
    maps = [json.loads(line) for line in files["chunk_summaries.jsonl"].decode("utf-8").splitlines() if line.strip()]
    need(sha(files["meeting_sections.jsonl"]) == checkpoint["sections_file_sha256"]
         and sha(files["chunk_summaries.jsonl"]) == checkpoint["maps_file_sha256"]
         and cp.digest(chunks) == checkpoint["prepared_source_hash"]
         and cp.digest(maps) == checkpoint["maps_hash"], "offline_checkpoint_binding_mismatch")
    need(cp.code_identity(engine) == checkpoint["map_code"], "offline_map_code_changed")
    cp.validate_maps(chunks, checkpoint["map_sources"], maps)
    saved = json.loads(files[ed.RESPONSE])
    records = ed.source_records(chunks)
    need(saved["source_hash"] == ed.digest(records) and saved["register_assessment"]["sources"] == records, "offline_response_source_mismatch")
    need(set(saved["responses"]) == {"register", "notes"}, "offline_expected_completed_notes_replies")
    diagnostics = json.loads(files[ed.DIAGNOSTICS])["requests"]
    need([d["stage"] for d in diagnostics] == ["register", "notes"] and all(
        d.get("done") is True and d.get("done_reason") == "stop" and not d.get("failure_category")
        and not d.get("empty_final_answer") for d in diagnostics), "offline_unverified_response")
    return manifest, files, checkpoint, chunks, saved, records


def render(manifest_path, manifest_sha, output):
    manifest, files, checkpoint, chunks, saved, records = load_bound(manifest_path, manifest_sha)
    source_dir = Path(manifest["source_directory"]).resolve()
    need(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(source_dir)
         and not source_dir.is_relative_to(output.resolve()), "offline_fresh_destination_required")
    source = "\n".join(c["text"] for c in chunks if c["meeting_section"] in {ed.BUSINESS, ed.ADJOURNMENT})
    commitments = commitment_evidence(chunks, include_context=True, blocked_context_source_ids=set(checkpoint["blocked_context_source_ids"]))
    # The saved audit preserves original exclusions and private proposal details.
    # Triage revalidates the selected candidates through all existing hard guards.
    register = ed.triage_register(saved["register_assessment"], records, commitments, source)
    proposed, withheld = ed.triage_notes(json.loads(saved["responses"]["notes"]))
    visible = {k: r for k, r in records.items() if len(r["text"]) <= ed.LOCAL_CHARS}
    notes = ed.validate_notes(proposed, visible, source, reference_changes=withheld)
    for key in records.keys() - visible.keys():
        register["findings"].append({"code": "source_excerpt_coverage_review", "source_id": key})
    os.umask(0o077)
    output.mkdir(mode=0o700)
    for name in [ed.RESPONSE, cp.FILE, "meeting_sections.jsonl", "chunk_summaries.jsonl", ed.DIAGNOSTICS, ed.BUDGETS, "redactions.json"]:
        if name in files:
            with (output / name).open("xb") as stream:
                stream.write(files[name])
            os.chmod(output / name, 0o600)
    review = ed.finish(output, register, notes, None, source,
                       [Finding(**w) for w in checkpoint["warnings"]], records=records)
    load_bound(manifest_path, manifest_sha)  # Recheck every original byte and source job after rendering.
    need((output / ed.RESPONSE).read_bytes() == files[ed.RESPONSE], "offline_response_copy_changed")
    provenance = {"mode": "source_bound_saved_notes_render", "manifest_sha256": manifest_sha,
                  "source_directory": str(source_dir), "source_job_sha256": manifest["source_job_sha256"],
                  "source_stage_id": checkpoint["source_stage_id"], "inference_calls": 0, "map_calls": 0,
                  "job_submissions": 0, "original_files_unchanged": True,
                  "saved_response_sha256": sha(files[ed.RESPONSE]), "status": review["status"],
                  "word_count": review["notes_word_count"], "detailed_minutes": "not_generated_no_saved_response"}
    ed.write_private_json(output / "offline-completion.private.json", provenance)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    def forbidden(*a, **k):
        raise ed.EditorialFailure("offline_network_or_inference_forbidden")
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    socket.create_connection = forbidden
    engine.call_ollama = forbidden
    review = render(args.manifest, args.manifest_sha256, args.output)
    print(json.dumps({"status": review["status"], "notes_directory": str(args.output), "word_count": review["notes_word_count"], "inference_calls": 0}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc) if isinstance(exc, ed.EditorialFailure) else "offline_completion_failed", file=sys.stderr)
        raise SystemExit(1)
