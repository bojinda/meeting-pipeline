"""Source/request binding for the completion entry point; synthetic replies only."""
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed, editorial_checkpoint as cp
from test_editorial_notes import fixture
from aihub_gpu_runner.core import encode

spec = importlib.util.spec_from_file_location("response_continuation", ROOT / "bin/continue-editorial-responses.py")
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


class ResponseContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        chunks, _, _ = fixture()
        self.records = ed.source_records(chunks)
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_bytes(b"synthetic tokenizer")
        self.config = dict(reduce_model="synthetic", keep_alive="30m", temperature=.2, reduce_num_ctx=196608,
                           keep_recap=False, ollama_url="http://127.0.0.1:1", tokenizer_sha256=completion.sha(self.tokenizer.read_bytes()))
        maps = [{**c, "file_name": "synthetic.txt", "speaker_span": "unverified", "summary": "Synthetic map"} for c in chunks]
        sections = "\n".join(json.dumps(c) for c in chunks).encode()
        map_bytes = "\n".join(json.dumps(c) for c in maps).encode()
        record = dict(configuration=self.config, prepared_source_hash=cp.digest(chunks), map_code={"frozen": True},
                      map_sources=[], maps_file_sha256=completion.sha(map_bytes), sections_file_sha256=completion.sha(sections),
                      blocked_context_source_ids=[], aliases={}, approved_passages=[], warnings=[])
        saved = dict(source_hash=ed.digest(self.records), register_assessment={"sources": self.records},
                     responses={"register": "synthetic register", "notes": "Café — synthetic notes"})
        self.info = [{"stage": s, "done": True, "failure_category": None, "empty_final_answer": False} for s in ["register", "notes"]]
        raw = {"editorial_checkpoint": json.dumps(record).encode(), "completion_response": json.dumps(saved).encode(),
               "editorial_sections": sections, "editorial_maps": map_bytes, "completion_diagnostics": json.dumps({"requests": self.info}).encode()}
        self.files = {}
        for role, data in raw.items():
            path = self.root / (role + ".json")
            path.write_bytes(data)
            self.files[role] = {"path": str(path), "sha256": completion.sha(data)}
        system = (ROOT / "prompts/meeting/reduce_system.txt").read_text(encoding="utf-8").strip() + "\nReturn only the requested JSON; no Markdown wrapper."
        def request_hash(phase):
            return completion.sha(encode(dict(model="synthetic", prompt=phase, system=system, stream=False, keep_alive="30m",
                options=dict(temperature=.2, num_ctx=196608, num_predict=ed.OUTPUT_TOKENS[phase]), format="json")))
        self.job = dict(state="failed", cleanup_verified=True, host_requests=[{"state": "completed", "request_hash": request_hash(s)} for s in ["register", "notes"]])
        self.manifest_data = dict(files=self.files, tokenizer=str(self.tokenizer), source_stage="synthetic-source", source_job_sha256=completion.sha(encode(self.job)))
        self.manifest = self.root / "manifest.json"
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.manifest_data), encoding="utf-8")

    def invoke(self, *, preflight=False, prompt_changed=False, owners=None):
        controller = Mock()
        controller.store.job.return_value = self.job
        controller.snapshot.return_value = {"owners": owners or {}}
        def run(*args, **kwargs):
            generate = args[11]
            for phase in ["register", "notes", "detailed"]:
                generate(phase + (" changed" if prompt_changed else ""), structured=phase != "detailed", num_predict=ed.OUTPUT_TOKENS[phase])
            return 0
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["continuation", str(self.manifest), str(self.output)] + (["--preflight"] if preflight else [])))
            stack.enter_context(patch.dict(os.environ, {"AIHUB_GPU_RUNNER_CONFIG": "synthetic"}))
            stack.enter_context(patch.object(completion, "approved_input", side_effect=lambda role, path: (True, path)))
            stack.enter_context(patch.object(completion, "admission_from_config", return_value=(None, {}, controller)))
            stack.enter_context(patch.object(cp, "code_identity", return_value={"frozen": True}))
            stack.enter_context(patch.object(cp, "validate_maps"))
            stage = stack.enter_context(patch.object(cp, "require_fresh_stage"))
            stack.enter_context(patch.object(ed, "RequestBudget"))
            stack.enter_context(patch.object(ed, "run", side_effect=run))
            inference = stack.enter_context(patch.object(completion.engine, "call_ollama", return_value="Synthetic detail"))
            result = completion.main()
            return result, inference.call_count, stage.call_count

    def test_reuses_only_bound_completed_replies_and_generates_missing_detail(self):
        before = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        self.assertEqual(self.invoke(), (0, 1, 1))
        provenance = json.loads((self.output / "completion-provenance.private.json").read_text())
        self.assertEqual(provenance["reused_responses"], ["register", "notes"])
        self.assertEqual(provenance["new_inference_stages"], ["detailed"])
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_source_bytes_changed_fail_before_output_or_inference(self):
        Path(self.files["editorial_sections"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ed.EditorialFailure, "completion_source_changed"):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_changed_request_cannot_reuse_a_saved_reply(self):
        with self.assertRaisesRegex(ed.EditorialFailure, "completion_request_binding_mismatch"):
            self.invoke(prompt_changed=True)

    def test_existing_destination_is_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ed.EditorialFailure, "fresh_completion_output_required"):
            self.invoke()

    def test_unsettled_source_job_is_rejected(self):
        self.job["cleanup_verified"] = False
        self.manifest_data["source_job_sha256"] = completion.sha(encode(self.job))
        self.write_manifest()
        with self.assertRaisesRegex(ed.EditorialFailure, "completion_source_work_unresolved"):
            self.invoke()

    def test_source_ownership_is_rejected(self):
        with self.assertRaisesRegex(ed.EditorialFailure, "completion_source_still_owned"):
            self.invoke(owners={"gpu1": {"job_id": "synthetic-source"}})

    def test_preflight_stops_at_missing_detail(self):
        with self.assertRaisesRegex(ed.EditorialFailure, "preflight_missing_detail_not_executed"):
            self.invoke(preflight=True)
