"""One explicit editorial continuation from completed, source-bound maps."""
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path

from .editorial import EditorialFailure
from .gpu_admission import approved_input
from .speaker_suggestions import write_private_json
from .sections import BUSINESS, RECAP, ADJOURNMENT

FILE = "editorial-map-checkpoint.private.json"
VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
MAP_CODE = ("aliases", "redaction", "sections", "speaker_turns", "normalization", "actions", "motions", "commitments", "chunking")


class StagePrompts:
    def __init__(self, directory):
        self.directory = directory

    def __truediv__(self, name):
        if os.environ.get("AIHUB_GPU_STAGE_INPUT_SNAPSHOTS"):
            _, selected = approved_input("editorial_prompt_" + Path(name).stem, self.directory / name)
            if selected is None:
                raise EditorialFailure("editorial_prompt_snapshot_missing")
            return selected
        return self.directory / name

    def __str__(self):
        return str(self.directory)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(data):
    return sha(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def file_hash(path):
    return sha(Path(path).read_bytes()) if path is not None and Path(path).is_file() else None


def code_identity(engine):
    # Bind map preparation/normalization and map prompts. Editorial-only changes
    # are allowed; they are separately captured by the fresh stage's inputs.
    return {**{name: file_hash(ROOT / f"bin/meeting_postprocess/{name}.py") for name in MAP_CODE},
            **{name: file_hash(ROOT / "prompts/meeting" / name) for name in ("chunk_prompt.txt", "chunk_system.txt")},
            "map_prompt_builder": sha((inspect.getsource(engine.SafeDict) + inspect.getsource(engine.load_template) + inspect.getsource(engine.build_chunk_prompt)).encode()),
            "map_postprocessing": sha(map_body(Path(engine.__file__).read_text()).encode())}


def map_body(text):
    marker = '        chunk_id = chunk.get("chunk_id", f"chunk-{idx:03d}")'
    return text.replace("\r\n", "\n").split(marker, 1)[1].split("    combined = build_reduce_input(chunk_summaries)", 1)[0].strip()


def configuration(args, tokenizer):
    return {key: getattr(args, key) for key in ("map_model", "reduce_model", "map_num_ctx", "reduce_num_ctx", "temperature", "keep_alive", "keep_recap", "ollama_url")} | {"tokenizer_sha256": file_hash(tokenizer)}


def file_binding(path):
    return {"path": str(Path(path).resolve()) if path is not None else None, "sha256": file_hash(path)}


def selected_binding(role, original, selected):
    specs = json.loads(os.environ.get("AIHUB_GPU_STAGE_INPUT_FILES", "{}"))
    canonical = specs.get(role, {}).get("path", original)
    return {"path": str(Path(canonical).resolve()) if canonical is not None else None,
            "sha256": file_hash(selected)}


def validate_maps(chunks, map_sources, maps):
    if not maps or len(map_sources) != len(maps) or len({str(r["chunk_id"]) for r in maps}) != len(maps):
        raise EditorialFailure("checkpoint_incomplete_maps")
    source_ids = {str(c["chunk_id"]): c for c in chunks}
    if len(source_ids) != len(chunks):
        raise EditorialFailure("checkpoint_duplicate_source_chunk")
    covered = []
    for original, result in zip(map_sources, maps):
        if not isinstance(result.get("summary"), str) or not result["summary"].strip():
            raise EditorialFailure("checkpoint_empty_map")
        for key in ("chunk_id", "file_name", "source_chunk_id", "meeting_section", "section_evidence", "start_time", "end_time", "speaker_span", "chunk_type", "source_portions"):
            if original.get(key) != result.get(key):
                raise EditorialFailure("checkpoint_chunk_identity_mismatch")
        portions = original.get("source_portions") or [original]
        for portion in portions:
            item = {k: v for k, v in portion.items() if k != "portion_id"}
            if source_ids.get(str(item["chunk_id"])) != item:
                raise EditorialFailure("checkpoint_map_source_mismatch")
            covered.append(str(item["chunk_id"]))
    expected = [str(c["chunk_id"]) for c in chunks if c["meeting_section"] in {RECAP, BUSINESS, ADJOURNMENT}]
    # Baseline mapping also covers portions classified outside meeting business.
    if covered != list(source_ids) and covered != expected:
        raise EditorialFailure("checkpoint_source_coverage_mismatch")


def request_hashes(engine, args, map_sources, prompts, chunk_plan):
    hashes = []
    for chunk in map_sources:
        options = {"temperature": args.temperature}
        if args.map_num_ctx is not None:
            options["num_ctx"] = args.map_num_ctx
        prompt = engine.build_chunk_prompt(prompts, chunk)
        if chunk_plan:
            from .chunking import DETAILS
            prompt += DETAILS
            options["num_predict"] = chunk_plan["settings"]["map_output"]
        hashes.append(digest({"model": args.map_model, "prompt": prompt,
                              "system": (prompts / "chunk_system.txt").read_text().strip(),
                              "stream": False, "keep_alive": args.keep_alive, "options": options}))
    return hashes


def verify_stage(stage, hashes, bindings):
    if (stage.get("host_stage") is not True or stage.get("state") not in {"success", "failed"} or
            stage.get("cleanup_verified") is not True or stage.get("error_code") not in {None, "host_command_failed"} or
            stage.get("cleanup_proof", {}).get("completion_verified") is not True):
        raise EditorialFailure("checkpoint_source_stage_unresolved")
    requests = stage.get("host_requests", [])
    if len(requests) < len(hashes) or any(r.get("state") != "completed" for r in requests):
        raise EditorialFailure("checkpoint_backend_work_uncertain")
    if [r.get("request_hash") for r in requests[:len(hashes)]] != hashes:
        raise EditorialFailure("checkpoint_model_or_prompt_binding_mismatch")
    declared = stage.get("input_bindings", {})
    roles = ["transcript_index", "approved_aliases", "approved_turn_corrections"]
    roles += [role for role in ("chunk_plan", "chunk_tokenizer") if bindings.get(role, {}).get("sha256") is not None]
    if "meeting_configuration" in declared:
        roles.append("meeting_configuration")
    for role in roles:
        entry = declared.get(role, {})
        binding = bindings[role]
        if entry.get("sha256") != binding["sha256"] or entry.get("present") != (binding["sha256"] is not None):
            raise EditorialFailure("checkpoint_source_authority_mismatch")
        if binding["path"] is not None and entry.get("path_sha256") != sha(binding["path"].encode()):
            raise EditorialFailure("checkpoint_source_path_mismatch")


def verify_legacy(directory, args, engine, chunks, map_sources, bindings, prompts, chunk_plan):
    """Seal old maps only with independently supplied original stage/config/code."""
    import subprocess
    stage_data = args.meeting_notes_source_stage.read_bytes()
    if sha(stage_data) != args.meeting_notes_source_stage_sha256:
        raise EditorialFailure("checkpoint_source_stage_identity_mismatch")
    if not args.meeting_notes_source_config_sha256 or bindings["meeting_configuration"]["sha256"] != args.meeting_notes_source_config_sha256:
        raise EditorialFailure("checkpoint_original_configuration_unconfirmed")
    # Never infer the original code revision from the current checkout.
    import re
    if not args.meeting_notes_source_commit or not re.fullmatch(r"[0-9a-f]{7,40}", args.meeting_notes_source_commit):
        raise EditorialFailure("checkpoint_original_code_unconfirmed")
    paths = [f"bin/meeting_postprocess/{name}.py" for name in MAP_CODE] + ["prompts/meeting/chunk_prompt.txt", "prompts/meeting/chunk_system.txt"]
    for path in paths:
        old = subprocess.run(["git", "show", f"{args.meeting_notes_source_commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
        if old.replace(b"\r\n", b"\n") != (ROOT / path).read_bytes().replace(b"\r\n", b"\n"):
            raise EditorialFailure("checkpoint_original_map_code_changed")
    original_engine = subprocess.run(["git", "show", f"{args.meeting_notes_source_commit}:bin/ollama_session_summary.py"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    for function in (engine.SafeDict, engine.load_template, engine.build_chunk_prompt):
        if inspect.getsource(function).strip() not in original_engine.replace("\r\n", "\n"):
            raise EditorialFailure("checkpoint_original_map_prompt_changed")
    # Protect against changed map postprocessing as well as changed prompts.
    current_engine = Path(engine.__file__).read_text()
    if map_body(original_engine) != map_body(current_engine):
        raise EditorialFailure("checkpoint_original_map_processing_changed")
    maps = engine.load_jsonl(directory / "chunk_summaries.jsonl")
    if engine.load_jsonl(directory / "meeting_sections.jsonl") != chunks:
        raise EditorialFailure("checkpoint_original_preparation_changed")
    from .redaction import PRIVATE_REDACTION_FILENAME
    if json.loads((directory / PRIVATE_REDACTION_FILENAME).read_text()) != json.loads((args.meeting_notes_output_dir / PRIVATE_REDACTION_FILENAME).read_text()):
        raise EditorialFailure("checkpoint_redactions_changed")
    validate_maps(chunks, map_sources, maps)
    stage = json.loads(stage_data)
    verify_stage(stage, request_hashes(engine, args, map_sources, prompts, chunk_plan), bindings)
    if not isinstance(stage.get("request_id"), str) or not stage["request_id"]:
        raise EditorialFailure("checkpoint_source_stage_identity_missing")
    return maps, stage["request_id"]


def seal(directory, args, engine, chunks, map_sources, maps, bindings, aliases, approved, blocked, warnings, tokenizer, prompts, chunk_plan, *, stage_id=None, origin=None):
    validate_maps(chunks, map_sources, maps)
    record = {"version": VERSION, "source_stage_id": stage_id or os.environ.get("AIHUB_GPU_HOST_JOB_ID"),
              "configuration": configuration(args, tokenizer), "map_code": code_identity(engine),
              "inputs": bindings, "prepared_source_hash": digest(chunks), "maps_hash": digest(maps),
              "maps_file_sha256": file_hash(directory / "chunk_summaries.jsonl"),
              "sections_file_sha256": file_hash(directory / "meeting_sections.jsonl"),
              "map_sources": map_sources, "map_source_hashes": [digest(c) for c in map_sources],
              "map_request_hashes": request_hashes(engine, args, map_sources, prompts, chunk_plan),
              "aliases": aliases, "approved_passages": approved, "blocked_context_source_ids": sorted(blocked),
              "warnings": [asdict(w) for w in warnings], "chunk_plan": chunk_plan,
              "origin": origin or {"mode": "completed_maps_in_source_stage"},
              "publication_status": "review_hold"}
    write_private_json(directory / FILE, record)
    return record


def load(path, expected_sha, args, engine, tokenizer, chunks, bindings):
    _, snapshot = approved_input("editorial_checkpoint", path)
    data = Path(snapshot).read_bytes()
    if not expected_sha or sha(data) != expected_sha:
        raise EditorialFailure("checkpoint_identity_mismatch")
    record = json.loads(data)
    if record.get("version") != VERSION or not record.get("source_stage_id") or record.get("publication_status") != "review_hold":
        raise EditorialFailure("checkpoint_missing_source_authority")
    if record["configuration"] != configuration(args, tokenizer) or record["map_code"] != code_identity(engine):
        raise EditorialFailure("checkpoint_configuration_or_code_changed")
    if record["inputs"] != bindings or record["prepared_source_hash"] != digest(chunks):
        raise EditorialFailure("checkpoint_source_or_approvals_changed")
    _, maps_path = approved_input("editorial_maps", Path(path).parent / "chunk_summaries.jsonl")
    _, sections_path = approved_input("editorial_sections", Path(path).parent / "meeting_sections.jsonl")
    maps = engine.load_jsonl(Path(maps_path))
    if (file_hash(maps_path) != record["maps_file_sha256"] or file_hash(sections_path) != record["sections_file_sha256"] or
            engine.load_jsonl(Path(sections_path)) != chunks or digest(maps) != record["maps_hash"]):
        raise EditorialFailure("checkpoint_inputs_changed")
    if len(record["map_source_hashes"]) != len(maps):
        raise EditorialFailure("checkpoint_incomplete_maps")
    if [digest(c) for c in record["map_sources"]] != record["map_source_hashes"]:
        raise EditorialFailure("checkpoint_map_source_mismatch")
    validate_maps(chunks, record["map_sources"], maps)
    return record, maps


def require_fresh_stage(record, expected_checkpoint_sha):
    job_id = os.environ.get("AIHUB_GPU_HOST_JOB_ID")
    lease_id = os.environ.get("AIHUB_GPU_HOST_LEASE_ID")
    if not job_id or not lease_id or job_id == record["source_stage_id"]:
        raise EditorialFailure("checkpoint_fresh_coordinated_stage_required")
    from aihub_gpu_runner.config import admission_from_config
    _, targets, admission = admission_from_config(os.environ["AIHUB_GPU_RUNNER_CONFIG"])
    target = targets[os.environ.get("AIHUB_GPU_OLLAMA_TARGET", "ollama")]
    owner = admission.snapshot()["owners"].get("gpu1", {})
    if (target.resources not in (("gpu1",), ("gpu0", "gpu1")) or owner.get("job_id") != job_id or
            owner.get("lease_id") != lease_id or owner.get("target") != target.name or
            any(admission.snapshot()["owners"].get(resource) != owner for resource in target.resources)):
        raise EditorialFailure("checkpoint_stage_not_owned")
    active = admission.store.job(job_id)
    if active.get("host_stage") is not True or active.get("state") != "running" or active.get("host_requests") != []:
        raise EditorialFailure("checkpoint_stage_not_fresh")
    if not os.environ.get("AIHUB_GPU_STAGE_INPUT_SNAPSHOTS"):
        raise EditorialFailure("checkpoint_immutable_snapshots_required")
    captured = active.get("input_bindings", {})
    for role, expected in (("editorial_checkpoint", expected_checkpoint_sha), ("editorial_maps", record["maps_file_sha256"]), ("editorial_sections", record["sections_file_sha256"]),
                           *((role, value["sha256"]) for role, value in record["inputs"].items() if role in {"transcript_index", "approved_aliases", "approved_turn_corrections"} or value["sha256"] is not None)):
        if role not in captured or captured[role].get("sha256") != expected:
            raise EditorialFailure("checkpoint_stage_snapshot_binding_mismatch")
    if any(o.get("job_id") == record["source_stage_id"] for o in admission.snapshot()["owners"].values()):
        raise EditorialFailure("checkpoint_source_stage_still_owned")
    original = admission.store.job(record["source_stage_id"])
    verify_stage(original, record["map_request_hashes"], record["inputs"])
    if original.get("target") != target.name or tuple(original.get("resources", [])) != target.resources:
        raise EditorialFailure("checkpoint_runner_target_changed")
