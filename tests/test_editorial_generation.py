"""Private response counts and explicit completed-map continuation, offline."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import ollama_session_summary as engine
from meeting_postprocess import editorial as ed, editorial_checkpoint as cp
from meeting_postprocess.ollama_response import GenerationFailure
from meeting_postprocess.publication import publication_payload, export_documents
from meeting_stage_inputs import input_files


class ResponseDiagnosticsTests(unittest.TestCase):
    def call(self, data=None, error=None, **kwargs):
        records = []
        with patch.object(engine, "managed_generate", side_effect=error, return_value=data), patch("urllib.request.urlopen", side_effect=AssertionError("HTTP forbidden")):
            try:
                result = engine.call_ollama("http://unused", "synthetic", "PRIVATE_PROMPT_SENTINEL", num_predict=16,
                                            response_format="json", response_callback=records.append, **kwargs)
                return result, records
            except GenerationFailure as exc:
                return exc.category, records

    def test_empty_thinking_only_limit_and_valid_response_are_distinct(self):
        cases = [(dict(response="", done=True, done_reason="stop"), "ollama_empty_final_answer"),
                 (dict(response="", done=True, thinking="PRIVATE_TRACE_SENTINEL", done_reason="stop"), "ollama_thinking_only_answer"),
                 (dict(response="", done=True, thinking="PRIVATE_TRACE_SENTINEL", done_reason="length", eval_count=16), "ollama_generation_limit"),
                 (dict(response='{"items":[]}', done=True, thinking="PRIVATE_TRACE_SENTINEL", done_reason="stop", eval_count=3, prompt_eval_count=10), '{"items":[]}'),
                 (dict(response="<think>PRIVATE_TRACE_SENTINEL</think>", done=True), "ollama_unseparated_thinking")]
        for data, expected in cases:
            with self.subTest(expected=expected):
                result, metadata = self.call(data)
                self.assertEqual(result, expected)
                self.assertEqual(len(metadata), 1)
                self.assertEqual(metadata[0]["response_char_count"], len(data["response"]))
                self.assertEqual(metadata[0]["thinking_char_count"], len(data["thinking"]) if "thinking" in data else None)
                self.assertEqual(metadata[0]["num_predict"], 16)
                self.assertEqual(metadata[0]["requested_thinking"], "default")
                self.assertNotIn("PRIVATE_TRACE_SENTINEL", json.dumps(metadata))
                self.assertNotIn("PRIVATE_PROMPT_SENTINEL", json.dumps(metadata))

    def test_transport_and_invalid_backend_json_do_not_retain_exception_text(self):
        for error, expected in ((TimeoutError("PRIVATE_PROMPT_SENTINEL"), "ollama_transport_failure"),
                                (json.JSONDecodeError("bad", "PRIVATE_TRACE_SENTINEL", 0), "ollama_invalid_response")):
            result, metadata = self.call(error=error)
            self.assertEqual(result, expected)
            self.assertNotIn("SENTINEL", json.dumps(metadata))
        result, metadata = self.call({"response": None, "error": "PRIVATE_TRACE_SENTINEL"})
        self.assertEqual(result, "ollama_invalid_response")
        self.assertNotIn("SENTINEL", json.dumps(metadata))

    def test_editorial_thinking_is_explicit_and_does_not_change_default_payload(self):
        for value in (None, False, True):
            with patch.object(engine, "managed_generate", return_value={"response": "{}", "done": True, "done_reason": "stop"}) as backend:
                engine.call_ollama("http://unused", "synthetic", "prompt", thinking=value, response_callback=lambda row: None, response_format="json", num_predict=16)
                payload = backend.call_args.args[1]
                self.assertEqual(payload["format"], "json")
                self.assertEqual(payload.get("think"), value)
                self.assertEqual("think" in payload, value is not None)
                self.assertEqual(payload["options"]["num_predict"], 16)

    def test_done_reason_and_count_fields_are_allowlisted(self):
        _, rows = self.call({"response": "{}", "done": True, "done_reason": "PRIVATE_TRACE_SENTINEL", "eval_count": "PRIVATE_TRACE_SENTINEL", "prompt_eval_count": True})
        self.assertEqual(rows[0]["done_reason"], "other")
        self.assertIsNone(rows[0]["eval_count"])
        self.assertIsNone(rows[0]["prompt_eval_count"])
        self.assertNotIn("SENTINEL", json.dumps(rows))

    def test_only_terminal_success_can_supply_a_final_answer(self):
        valid = {"response": '{"items":[]}', "done": True, "done_reason": "stop"}
        self.assertEqual(self.call(valid)[0], valid["response"])
        # Older terminal envelopes may omit the optional stop reason.
        self.assertEqual(self.call({"response": "{}", "done": True})[0], "{}")
        for value in (False, None, 0, 1, "true"):
            with self.subTest(done=value):
                result, rows = self.call({**valid, "done": value})
                self.assertEqual(result, "ollama_incomplete_response")
                self.assertEqual(rows[0]["done"], value if type(value) is bool else None)
        self.assertEqual(self.call({"response": "{}"})[0], "ollama_incomplete_response")
        # Nonterminal output cannot be mistaken for generation-limit completion.
        self.assertEqual(self.call({**valid, "done": False, "eval_count": 16})[0], "ollama_incomplete_response")
        for value in (True, False):
            with self.subTest(error_done=value):
                result, rows = self.call({**valid, "done": value, "done_reason": "error", "thinking": "PRIVATE_TRACE_SENTINEL"})
                self.assertEqual(result, "ollama_error_termination")
                self.assertNotIn("SENTINEL", json.dumps(rows))


class EditorialContinuationTests(unittest.TestCase):
    def setUp(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "chunks_out").mkdir(parents=True)
        self.index = self.source / "chunks_out/transcript_chunks.jsonl"
        self.index.write_text(json.dumps({"chunk_id": 1, "file_name": "source.txt", "start_time": 0, "end_time": 60,
                                          "text": "[Chair] Let's start the meeting.\n[Taylor] I'll send the report to the committee."}) + "\n")
        self.tokenizer = self.root / "tokenizer.json"
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(self.tokenizer))
        self.settings = self.root / "original.env"
        self.settings.write_text("MEETING_MAP_MODEL=synthetic\n")
        self.env = {"MEETING_SUMMARIES_ROOT": str(self.root / "production"), "MEETING_KEEP_RECAP": "0",
                    "MEETING_CONFIG_FILE": str(self.settings), "AIHUB_GPU_HOST_JOB_ID": "original-stage"}
        self.payloads = []

    def tearDown(self):
        self.temp.cleanup()

    def argv(self, out, *extra):
        return ["summary", str(self.source), "--meeting-notes", "--meeting-notes-output-dir", str(out),
                "--meeting-notes-tokenizer", str(self.tokenizer), "--map-model", "synthetic", "--reduce-model", "synthetic",
                "--map-num-ctx", "32768", "--reduce-num-ctx", "196608", "--ollama-url", "http://unused", *extra]

    def backend(self, url, payload, timeout):
        self.payloads.append(payload)
        prompt = payload["prompt"]
        if "Transcript chunk:" in prompt:
            response = "## Topics\nThe committee report was discussed."
        elif "propose ONE private canonical" in prompt:
            response = json.dumps({"items": []})
        elif "summary-reduction call" in prompt:
            data = json.loads(prompt.split("Untrusted input JSON:\n", 1)[1])
            ref = next(r["id"] for r in data["source_excerpts"] if r["section"] == "current_meeting_business")
            response = json.dumps({"highlights": [], "previous_context": [], "issues": [{"heading": "Committee report", "paragraphs": [{"text": "The model generated this committee discussion.", "source_ids": [ref]}]}], "motions": [], "unresolved": [], "concerns": []})
        else:
            response = "# Detailed minutes\nThe committee report was discussed."
        return {"response": response, "done": True, "thinking": "PRIVATE_TRACE_SENTINEL", "done_reason": "stop", "eval_count": 20, "prompt_eval_count": 100}

    def run_main(self, argv, env=None, backend=None):
        with patch.dict(os.environ, self.env if env is None else env, clear=True), patch.object(sys, "argv", argv), \
             patch.object(engine, "managed_generate", side_effect=self.backend if backend is None else backend), \
             patch("urllib.request.urlopen", side_effect=AssertionError("HTTP forbidden")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return engine.main()

    def original(self):
        out = self.root / "original"
        self.assertEqual(self.run_main(self.argv(out)), 0)
        record = json.loads((out / cp.FILE).read_text())
        stage = {"request_id": "original-stage", "host_stage": True, "state": "failed", "cleanup_verified": True,
                 "target": "ollama", "resources": ["gpu1"],
                 "error_code": "host_command_failed", "cleanup_proof": {"completion_verified": True},
                 "host_requests": [{"state": "completed", "request_hash": cp.digest(p)} for p in self.payloads],
                 "input_bindings": {role: {"sha256": b["sha256"], "present": b["sha256"] is not None,
                                             "path_sha256": cp.sha(b["path"].encode())} for role, b in record["inputs"].items() if b["path"]}}
        self.payloads.clear()
        return out, record, stage

    def fresh(self, out, record, stage):
        owner = {"job_id": "new-stage", "lease_id": "new-lease", "target": "ollama"}
        active = {"host_stage": True, "state": "running", "host_requests": [], "input_bindings": {
            **{role: {"sha256": b["sha256"]} for role, b in record["inputs"].items()},
            "editorial_checkpoint": {"sha256": cp.file_hash(out / cp.FILE)}, "editorial_maps": {"sha256": record["maps_file_sha256"]}, "editorial_sections": {"sha256": record["sections_file_sha256"]}}}
        admission = types.SimpleNamespace(snapshot=lambda: {"owners": {"gpu1": owner}}, store=types.SimpleNamespace(job=lambda identity: stage if identity == "original-stage" else active))
        config = types.ModuleType("aihub_gpu_runner.config")
        config.admission_from_config = lambda path: ({}, {"ollama": types.SimpleNamespace(resources=("gpu1",), name="ollama")}, admission)
        snapshots = {"editorial_checkpoint": str(out / cp.FILE), "editorial_maps": str(out / "chunk_summaries.jsonl"),
                     "editorial_sections": str(out / "meeting_sections.jsonl"), "transcript_index": str(self.index),
                     "approved_aliases": None, "approved_turn_corrections": None, "editorial_tokenizer": str(self.tokenizer)}
        snapshots.update({"editorial_prompt_" + p.stem: str(p) for p in (ROOT / "prompts/meeting").glob("*.txt")})
        env = dict(self.env, AIHUB_GPU_HOST_JOB_ID="new-stage", AIHUB_GPU_HOST_LEASE_ID="new-lease", AIHUB_GPU_RUNNER_CONFIG="synthetic",
                   AIHUB_GPU_STAGE_INPUT_SNAPSHOTS=json.dumps(snapshots))
        return env, {"aihub_gpu_runner": types.ModuleType("aihub_gpu_runner"), "aihub_gpu_runner.config": config}, active

    def continuation_argv(self, original, out):
        return self.argv(out, "--meeting-notes-checkpoint", str(original / cp.FILE), "--meeting-notes-checkpoint-sha256", cp.file_hash(original / cp.FILE))

    def test_verified_continuation_uses_only_reductions_and_actual_model_notes(self):
        original, record, stage = self.original()
        before = {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}
        env, modules, _ = self.fresh(original, record, stage)
        out = self.root / "continued"
        with patch.dict(sys.modules, modules):
            self.assertEqual(self.run_main(self.continuation_argv(original, out), env), 0)
        self.assertEqual(len(self.payloads), 3)
        self.assertFalse(any("Transcript chunk:" in p["prompt"] for p in self.payloads))
        self.assertIn("The model generated", (out / "meeting-notes-draft.md").read_text())
        self.assertEqual({p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}, before)
        metadata = json.loads((out / ed.DIAGNOSTICS).read_text())
        self.assertEqual([r["stage"] for r in metadata["requests"]], ["register", "notes", "detailed"])
        self.assertNotIn("PRIVATE_TRACE_SENTINEL", "".join(p.read_text() for p in out.iterdir() if p.is_file()))
        with self.assertRaisesRegex(ValueError, "hold"):
            publication_payload(out)
        with self.assertRaisesRegex(ValueError, "hold"):
            export_documents(out, self.root / "export")

    def test_checkpoint_identity_changed_maps_source_approval_configuration_and_code_fail_before_generation(self):
        original, record, original_stage = self.original()
        original_index = self.index.read_bytes()
        maps_file = original / "chunk_summaries.jsonl"
        original_maps = maps_file.read_bytes()
        for change in ("identity", "maps", "source", "aliases", "context", "code", "stage", "uncertain_backend", "new_uncertain_backend", "snapshots", "same_stage", "uncoordinated", "target"):
            with self.subTest(change=change):
                try:
                    stage = copy.deepcopy(original_stage)
                    env, modules, active = self.fresh(original, record, stage)
                    failed = self.root / ("failed-" + change)
                    argv = self.continuation_argv(original, failed)
                    if change == "identity": argv[-1] = "0" * 64
                    if change == "maps": (original / "chunk_summaries.jsonl").write_text((original / "chunk_summaries.jsonl").read_text() + "\n")
                    if change == "source": self.index.write_text(self.index.read_text().replace("report", "invoice"))
                    if change == "aliases":
                        alias = self.source / "speaker_aliases.json"
                        alias.write_text('{"SPEAKER_01":"Jordan"}')
                        snapshots = json.loads(env["AIHUB_GPU_STAGE_INPUT_SNAPSHOTS"])
                        snapshots["approved_aliases"] = str(alias)
                        env["AIHUB_GPU_STAGE_INPUT_SNAPSHOTS"] = json.dumps(snapshots)
                    if change == "context": argv[argv.index("--reduce-num-ctx") + 1] = "98304"
                    if change == "stage": stage["cleanup_verified"] = False
                    if change == "uncertain_backend": stage["host_requests"][-1]["state"] = "submit_intent"
                    if change == "new_uncertain_backend": active["host_requests"] = [{"state": "submit_intent"}]
                    if change == "snapshots": active["input_bindings"]["editorial_maps"]["sha256"] = "bad"
                    if change == "same_stage": env["AIHUB_GPU_HOST_JOB_ID"] = "original-stage"
                    if change == "uncoordinated": env.pop("AIHUB_GPU_HOST_LEASE_ID")
                    if change == "target": stage["target"] = "different-target"
                    patches = [patch.dict(sys.modules, modules)]
                    if change == "code": patches.append(patch.object(cp, "code_identity", return_value={"changed": True}))
                    with contextlib.ExitStack() as stack:
                        for item in patches: stack.enter_context(item)
                        self.assertEqual(self.run_main(argv, env), 2)
                    self.assertEqual(self.payloads, [])
                    self.assertFalse((failed / "meeting-notes-draft.md").exists())
                finally:
                    self.index.write_bytes(original_index)
                    maps_file.write_bytes(original_maps)
                    (self.source / "speaker_aliases.json").unlink(missing_ok=True)

    def test_semantic_invalid_json_empty_and_thinking_only_register_failures_are_separate(self):
        cases = [("", "", "ollama_empty_final_answer"), ("", "PRIVATE_TRACE_SENTINEL", "ollama_thinking_only_answer"),
                 ("{", "", "editorial_register_invalid_json"), ('{"items":null}', "", "invalid_register_schema")]
        for index, (answer, trace, expected) in enumerate(cases):
            def backend(url, payload, timeout):
                if "propose ONE private canonical" in payload["prompt"]:
                    self.payloads.append(payload)
                    return {"response": answer, "done": True, "thinking": trace, "done_reason": "stop", "eval_count": 10}
                return self.backend(url, payload, timeout)
            out = self.root / f"failure-{index}"
            self.payloads.clear()
            self.assertEqual(self.run_main(self.argv(out), backend=backend), 1)
            self.assertEqual(len(self.payloads), 2)
            review = json.loads((out / ed.REVIEW).read_text())
            self.assertEqual(review["failure_category"], expected)
            self.assertEqual(review["failed_stage"], "register")
            self.assertTrue((out / cp.FILE).exists())
            self.assertFalse((out / "meeting-notes-draft.md").exists())
            self.assertNotIn("PRIVATE_TRACE_SENTINEL", "".join(p.read_text() for p in out.iterdir() if p.is_file()))

    def test_nonterminal_and_error_register_answers_stop_before_notes(self):
        for index, (done, reason, expected) in enumerate(((False, "stop", "ollama_incomplete_response"),
                                                        (True, "error", "ollama_error_termination"))):
            def backend(url, payload, timeout):
                result = self.backend(url, payload, timeout)
                if "propose ONE private canonical" in payload["prompt"]:
                    result.update(done=done, done_reason=reason)
                return result
            output = self.root / f"nonterminal-{index}"
            self.payloads.clear()
            self.assertEqual(self.run_main(self.argv(output), backend=backend), 1)
            self.assertEqual(len(self.payloads), 2)
            self.assertEqual(json.loads((output / ed.REVIEW).read_text())["failure_category"], expected)
            self.assertTrue((output / cp.FILE).is_file())
            self.assertFalse((output / "meeting-notes-draft.md").exists())
            self.assertNotIn("PRIVATE_TRACE_SENTINEL", "".join(p.read_text() for p in output.iterdir() if p.is_file()))

    def test_runner_declaration_snapshots_checkpoint_and_binds_current_code_prompts_config(self):
        original, _, _ = self.original()
        argv = self.continuation_argv(original, self.root / "fresh")
        with patch.dict(os.environ, self.env, clear=True):
            files = input_files([sys.executable, str(ROOT / "bin/ollama_meeting_summary.py"), *argv[1:]])
        for role in ("editorial_maps", "editorial_sections", "editorial_checkpoint", "transcript_index", "editorial_tokenizer", "editorial_prompt_meeting_notes_prompt"):
            self.assertTrue(files[role]["required"])
            self.assertTrue(files[role]["snapshot"])
        self.assertIn("editorial_engine", files)
        self.assertIn("meeting_configuration", files)
        self.assertLessEqual(len(files), 64)

    def test_legacy_sealing_checks_original_stage_config_code_redactions_and_never_infers(self):
        original, _, stage = self.original()
        stage_file = self.root / "source-stage.private.json"
        stage_file.write_text(json.dumps(stage))
        source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True, text=True).stdout.strip()
        for case in ("valid", "stage", "configuration", "source_authority", "redaction"):
            with self.subTest(case=case):
                altered = copy.deepcopy(stage)
                if case == "stage": altered["cleanup_verified"] = False
                if case == "source_authority": altered["input_bindings"]["approved_aliases"]["sha256"] = "changed"
                stage_file.write_text(json.dumps(altered))
                output = self.root / ("sealed-" + case)
                args = self.argv(output, "--meeting-notes-seal-maps", str(original),
                                 "--meeting-notes-source-stage", str(stage_file), "--meeting-notes-source-stage-sha256", cp.file_hash(stage_file),
                                 "--meeting-notes-source-commit", source_commit,
                                 "--meeting-notes-source-config-sha256", cp.file_hash(self.settings) if case != "configuration" else "0" * 64)
                if case == "redaction": (original / "redactions.json").write_text('{"redactions":[{"changed":true}]}')
                self.payloads.clear()
                self.assertEqual(self.run_main(args, backend=lambda *args: self.fail("sealing inferred")), 0 if case == "valid" else 2)
                self.assertEqual(self.payloads, [])
                self.assertFalse((output / "meeting-notes-draft.md").exists())
                if case == "valid":
                    self.assertEqual(json.loads((output / cp.FILE).read_text())["source_stage_id"], "original-stage")

    def test_saved_maps_are_complete_but_four_files_cannot_establish_checkpoint_authority(self):
        saved = ROOT / "ignore/october-editorial-triage.bioHNv/editorial-output"
        if not saved.exists(): self.skipTest("protected failed-run October artifacts unavailable")
        before = {p.name: cp.file_hash(p) for p in saved.iterdir() if p.is_file()}
        chunks, maps = engine.load_jsonl(saved / "meeting_sections.jsonl"), engine.load_jsonl(saved / "chunk_summaries.jsonl")
        self.assertEqual(len(maps), 27)
        cp.validate_maps(chunks, chunks, maps)
        response = json.loads((saved / ed.RESPONSE).read_text())
        self.assertEqual(response["source_hash"], ed.digest(ed.source_records(chunks)))
        self.assertEqual(response["responses"]["register"], "")
        self.assertFalse((saved / cp.FILE).exists(), "legacy maps must not be silently blessed")
        with self.assertRaisesRegex(ed.EditorialFailure, "stage_unresolved"):
            cp.verify_stage({}, [], {})
        self.assertEqual({p.name: cp.file_hash(p) for p in saved.iterdir() if p.is_file()}, before)


if __name__ == "__main__":
    unittest.main()
