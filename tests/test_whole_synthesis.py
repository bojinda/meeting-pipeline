"""Synthetic whole-meeting evidence, isolation, budgeting and fallback regressions."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import ollama_session_summary as engine
from meeting_postprocess import whole_synthesis as whole
from meeting_postprocess import speaker_turns as turns
from meeting_postprocess.sections import BUSINESS, ADJOURNMENT, RECAP
from meeting_postprocess.publication import publication_payload, export_archive, PUBLIC_MEETING_FILENAMES

SOURCE = """[SPEAKER_00] Hello, PRE_MEETING_SENTINEL personal discussion.
[SPEAKER_01] I'll invite Riley to the list tomorrow.
[SPEAKER_00] Okay guys, I think we'll get started here.
[SPEAKER_00] I'll start with the recap last month.
[SPEAKER_01] The last meeting, Taylor reported HISTORICAL_SENTINEL.
[SPEAKER_01] I'll send the old historical report tomorrow.
[SPEAKER_00] Okay, moving right along then.
[SPEAKER_01] I'll send the updated safety notice tomorrow.
[SPEAKER_02] We agreed to inspect the service track.
[SPEAKER_01] The brake inspection remains an outstanding safety concern.
[SPEAKER_02] I disagree with delaying the inspection.
[SPEAKER_00] Motion to adjourn.
[SPEAKER_00] Motion by Taylor, seconded by Casey.
[SPEAKER_00] The meeting is adjourned.
"""
ALIASES = {"SPEAKER_00": "Chair", "SPEAKER_01": "Taylor", "SPEAKER_02": "Casey"}


def create_session(root, text=SOURCE, aliases=True):
    directory = Path(root) / "session"
    index = directory / "chunks_out" / "transcript_chunks.jsonl"
    index.parent.mkdir(parents=True)
    chunks = [{"chunk_id": "1.1", "start_time": 0, "end_time": 30, "text": text}]
    index.write_text(json.dumps(chunks[0]) + "\n", encoding="utf-8")
    if aliases:
        (directory / "speaker_aliases.json").write_text(json.dumps(ALIASES), encoding="utf-8")
    return directory


def item(record, number, kind="topic", references=None, **kwargs):
    return {"id": "E" + str(number), "kind": kind, "section": record["section"], "statement": record["text"],
            "quotes": [{"record_id": row["id"], "text": row["text"]} for row in (references or [record])],
            "owners": [], "mover": None, "seconder": None, "outcome": None, **kwargs}


class FakeModel:
    def __init__(self, mutate_evidence=None, mutate_plan=None, fail_stage=None):
        self.requests = []
        self.mutate_evidence = mutate_evidence
        self.mutate_plan = mutate_plan
        self.fail_stage = fail_stage

    def __call__(self, request):
        self.requests.append(request)
        payload = json.loads(request["prompt"])
        stage = "evidence" if payload.get("format") == "meeting-source-v1" else "documents"
        if self.fail_stage == stage:
            return {"response": "PRIVATE_MODEL_BODY not JSON", "done_reason": "stop"}
        if stage == "evidence":
            # Decode the public model protocol independently of the ID resolver.
            records = [{"id": str(sid), "section": payload["sections"][section], "speaker": payload["speakers"][speaker], "text": text}
                       for section, rows in payload["runs"] for sid, speaker, text in rows]
            evidence = []
            for number, record in enumerate(records, 1):
                kind = "recap" if record["section"] == RECAP else "topic"
                fields = {}
                references = [record]
                if record["section"] != RECAP:
                    if "updated safety notice" in record["text"]:
                        kind, fields = "action", {"owners": [record["speaker"]]}
                    elif "We agreed" in record["text"]:
                        kind = "decision"
                    elif "safety concern" in record["text"]:
                        kind = "health_safety"
                    elif "I disagree" in record["text"]:
                        kind = "qualification"
                    elif record["text"] == "Motion to adjourn.":
                        kind = "motion"
                        references += [row for row in records if "Motion by Taylor" in row["text"] or row["text"] == "Carried."]
                        fields = {"mover": "Taylor", "seconder": "Casey", "outcome": "Carried" if any(row["text"] == "Carried." for row in references) else None}
                evidence.append(item(record, number, kind, references, **fields))
            data = {"items": evidence}
            if self.mutate_evidence:
                self.mutate_evidence(data, records)
            compact = []
            for row in data["items"]:
                primary = next((quote for quote in row["quotes"] if quote["text"] == row["statement"]), row["quotes"][0])
                entry = {key: row[key] for key in ("id", "kind", "section", "owners", "mover", "seconder", "outcome")}
                entry.update(primary_record_id=primary["record_id"], supporting_record_ids=[quote["record_id"] for quote in row["quotes"] if quote is not primary])
                # Free claims/quotes cannot be smuggled into the new protocol.
                if row["statement"] != primary["text"]:
                    entry["statement"] = row["statement"]
                if any(quote["record_id"] in {record["id"] for record in records} and quote["text"] != next(record["text"] for record in records if record["id"] == quote["record_id"]) for quote in row["quotes"]):
                    entry["quotes"] = row["quotes"]
                compact.append(entry)
            data = {"format": "meeting-evidence-v2", "items": compact}
        else:
            evidence = payload["validated_evidence"]
            data = {}
            for document in ("summary", "minutes", "actions"):
                grouped = {}
                for row in evidence:
                    if document == "actions" and row["kind"] != "action" or document == "minutes" and row["kind"] == "recap" and not payload["keep_recap"]:
                        continue
                    grouped.setdefault(whole.HEADINGS[row["kind"]], []).append(row["id"])
                data[document] = [{"section": key, "evidence_ids": values} for key, values in grouped.items()]
            if self.mutate_plan:
                self.mutate_plan(data, evidence)
        return {"response": json.dumps(data), "done_reason": "stop", "prompt_eval_count": 1000, "eval_count": 400, "total_duration": 2000000}


def invoke(root, directory, model=None, flags=(), env=None, summary=None, destination=None):
    output = Path(destination) if destination else Path(root) / "experiment"
    argv = ["summary", str(directory), "--synthesis-mode", "whole", "--experiment-output-dir", str(output), "--ollama-url", "http://localhost:11434", *flags]
    environment = {"MEETING_SUMMARIES_ROOT": str(Path(root) / "production"), "AIHUB_GPU1_LOCK_FILE": "/tmp/test-whole-gpu", "AIHUB_GPU_LOCK_HELD_FILE": "/tmp/test-whole-gpu", **(env or {})}
    captured = io.StringIO()
    with patch.dict(os.environ, environment, clear=True), patch.object(sys, "argv", argv), patch.object(whole.local, "_http_call", side_effect=model or FakeModel()), patch.object(engine, "call_ollama", side_effect=summary) as reduced, contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        status = engine.main()
    return status, output, captured.getvalue(), reduced


class WholeSynthesisTests(unittest.TestCase):
    def test_two_stages_full_coverage_and_source_index_without_audio(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            index = directory / "chunks_out" / "transcript_chunks.jsonl"
            before = index.read_bytes()
            production = Path(root) / "production" / "session"
            production.mkdir(parents=True)
            (production / "minutes-draft.md").write_text("PRODUCTION_SENTINEL")
            model = FakeModel()
            status, output, _, reduced = invoke(root, directory, model)
            self.assertEqual(status, 0)
            self.assertEqual(len(model.requests), 2)
            reduced.assert_not_called()
            self.assertEqual(index.read_bytes(), before)
            self.assertEqual((production / "minutes-draft.md").read_text(), "PRODUCTION_SENTINEL")
            report = json.loads((output / "whole-run.json").read_text())
            self.assertTrue(report["coverage"]["complete"])
            self.assertEqual(report["coverage"]["eligible_record_ids"], report["coverage"]["whole_included_record_ids"])
            self.assertEqual(report["processing_mode"], "whole")
            self.assertEqual(report["context"]["num_ctx"], 98304)
            self.assertEqual(report["calls"][0]["prompt_eval_count"], 1000)
            self.assertEqual(report["calls"][1]["eval_count"], 400)
            self.assertIn("runtime_seconds", report)
            self.assertTrue((output / "minutes-qa.md").exists())
            self.assertTrue((output / "meeting_sections.jsonl").exists())
            self.assertEqual((output / "chunk_summaries.jsonl").read_text(), "")
            public = json.dumps(publication_payload(output))
            for row in json.loads((output / "whole-source.json").read_text())["records"]:
                self.assertNotIn(row["id"], public)

    def test_natural_start_and_historical_recap_never_create_current_tasks(self):
        for keep in (False, True):
            with self.subTest(keep=keep), tempfile.TemporaryDirectory() as root:
                directory = create_session(root)
                model = FakeModel()
                status, output, _, _ = invoke(root, directory, model, flags=["--keep-recap"] if keep else ["--no-keep-recap"])
                self.assertEqual(status, 0)
                self.assertNotIn("PRE_MEETING_SENTINEL", json.dumps(model.requests))
                self.assertNotIn("invite Riley", json.dumps(model.requests))
                summary = (output / "summary.md").read_text()
                minutes = (output / "minutes-draft.md").read_text()
                actions = (output / "action-items.md").read_text()
                self.assertIn("Recap of Previous Meeting", summary)
                self.assertIn("HISTORICAL_SENTINEL", summary)
                self.assertEqual("HISTORICAL_SENTINEL" in minutes, keep)
                self.assertNotIn("historical report", actions)
                self.assertIn("Taylor – I'll send the updated safety notice tomorrow.", actions)
                for content in (summary, minutes, actions):
                    self.assertNotIn("PRE_MEETING_SENTINEL", content)
                    self.assertNotIn("Chunk", content)

    def test_explicit_motion_roles_preserve_only_source_outcome(self):
        for carried in (False, True):
            with self.subTest(carried=carried), tempfile.TemporaryDirectory() as root:
                text = SOURCE.replace("[SPEAKER_00] The meeting", "[SPEAKER_00] Carried.\n[SPEAKER_00] The meeting") if carried else SOURCE
                directory = create_session(root, text)
                status, output, _, _ = invoke(root, directory)
                self.assertEqual(status, 0)
                minutes = (output / "minutes-draft.md").read_text()
                self.assertIn("Taylor", minutes)
                self.assertIn("Casey", minutes)
                self.assertEqual("Carried" in minutes, carried)
                qa = json.loads((output / "minutes-qa.json").read_text())
                self.assertFalse(any(row["code"] == "questionable_motion" for row in qa["findings"]))

    def test_unsupported_motion_outcome_fails_without_public_documents(self):
        def mutate(data, records):
            next(row for row in data["items"] if row["kind"] == "motion")["outcome"] = "Carried"
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            model = FakeModel(mutate_evidence=mutate)
            status, output, _, _ = invoke(root, directory, model)
            self.assertEqual(status, 1)
            self.assertEqual(len(model.requests), 1)
            self.assertEqual(publication_payload(output)["documents"], [])
            report = json.loads((output / "whole-run.json").read_text())
            self.assertIn("unsupported_motion_outcome", json.dumps(report))

    def test_tentative_action_wrong_owner_and_historical_action_are_rejected(self):
        for case in ("owner", "tentative", "historical", "third_party"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                text = SOURCE
                if case == "tentative":
                    text = text.replace("I'll send the updated safety notice tomorrow.", "I think we should send the updated safety notice tomorrow.")
                if case == "third_party":
                    text = text.replace("I'll send the updated safety notice tomorrow.", "I will send the updated safety notice off, and if the office approves it, they will send it to Morgan.")
                directory = create_session(root, text)
                def mutate(data, records):
                    action = next(row for row in data["items"] if row["kind"] == "action")
                    if case == "owner":
                        action["owners"] = ["Taylor", "Riley"]
                    if case == "historical":
                        record = next(row for row in records if "old historical report" in row["text"])
                        action.update(section=BUSINESS, statement=record["text"], quotes=[{"record_id": record["id"], "text": record["text"]}])
                status, output, _, _ = invoke(root, directory, FakeModel(mutate_evidence=mutate))
                if case == "third_party":
                    self.assertEqual(status, 0)
                    actions = (output / "action-items.md").read_text()
                    self.assertIn("I will send the updated safety notice off", actions)
                    self.assertNotIn("Morgan", actions)
                    self.assertNotIn("office approves", actions)
                else:
                    self.assertEqual(status, 1)
                    self.assertEqual(publication_payload(output)["documents"], [])

    def test_missing_identities_are_preserved_and_flagged_without_using_suggestions(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root, aliases=False)
            (directory / "speaker-suggestions.json").write_text('{"SPEAKER_01":"Riley"}')
            status, output, _, _ = invoke(root, directory)
            self.assertEqual(status, 0)
            self.assertIn("SPEAKER_01", (output / "action-items.md").read_text())
            self.assertNotIn("Riley", (output / "action-items.md").read_text())
            self.assertIn("unresolved_speaker", (output / "minutes-qa.json").read_text())
            self.assertFalse((directory / "speaker_aliases.json").exists())

    def test_approved_turn_correction_overrides_alias_only_for_exact_turn(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            catalog = turns.turn_catalog(directory)
            target = next(row for row in catalog["turns"] if "updated safety notice" in row["text"])
            turns.approve_turn(directory, catalog, target["turn_id"], "Casey")
            correction = (directory / turns.CORRECTIONS_FILE).read_bytes()
            status, output, _, _ = invoke(root, directory)
            self.assertEqual(status, 0)
            self.assertIn("Casey – I'll send", (output / "action-items.md").read_text())
            self.assertEqual((directory / turns.CORRECTIONS_FILE).read_bytes(), correction)
            self.assertEqual(json.loads((directory / "speaker_aliases.json").read_text()), ALIASES)

    def test_spoken_redaction_is_applied_before_both_calls_and_all_public_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            text = SOURCE.replace("[SPEAKER_01] I'll send the updated", "[SPEAKER_01] Redact the following. PRIVATE_SECRET_ONE.\n[SPEAKER_02] PRIVATE_SECRET_TWO. End redaction.\n[SPEAKER_01] I'll send the updated")
            directory = create_session(root, text)
            model = FakeModel()
            status, output, _, _ = invoke(root, directory, model)
            self.assertEqual(status, 0)
            self.assertNotIn("PRIVATE_SECRET", json.dumps(model.requests))
            self.assertNotIn("PRIVATE_SECRET", json.dumps(publication_payload(output)))
            self.assertIn("PRIVATE_SECRET_ONE", (output / "redactions.json").read_text())
            for filename in (*whole.PRIVATE_FILES, "meeting_sections.jsonl", "chunk_summaries.jsonl", "minutes-qa.json"):
                self.assertNotIn("PRIVATE_SECRET", (output / filename).read_text())

    def test_fully_redacted_source_needs_no_model_calls(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root, "[SPEAKER_01] Redact the following. PRIVATE_SECRET.")
            model = Mock()
            status, output, _, _ = invoke(root, directory, model)
            self.assertEqual(status, 0)
            model.assert_not_called()
            self.assertNotIn("PRIVATE_SECRET", json.dumps(publication_payload(output)))

    def test_invalid_json_invalid_refs_and_second_stage_failure_never_publish_partial_outputs(self):
        for case in ("json", "reference", "quote", "claim", "plan", "plan_omission", "documents"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                directory = create_session(root)
                def mutate(data, records):
                    if case == "reference":
                        data["items"][0]["quotes"][0]["record_id"] = "Rmissing"
                    elif case == "quote":
                        data["items"][0]["quotes"][0]["text"] = "Invented statement."
                    elif case == "claim":
                        data["items"][0]["statement"] = "Invented statement."
                def plan(data, evidence):
                    if case == "plan":
                        data["minutes"][0]["evidence_ids"].append("Emissing")
                    elif case == "plan_omission":
                        data["actions"] = []
                model = FakeModel(mutate_evidence=mutate, mutate_plan=plan, fail_stage="evidence" if case == "json" else "documents" if case == "documents" else None)
                status, output, logs, _ = invoke(root, directory, model)
                self.assertEqual(status, 1)
                self.assertEqual(publication_payload(output)["documents"], [])
                self.assertNotIn("PRIVATE_MODEL_BODY", logs)
                self.assertNotIn("PRIVATE_MODEL_BODY", (output / "whole-run.json").read_text())

    def test_generation_limit_and_transport_failure_never_retry_or_accept_partial_json(self):
        for failure in (TimeoutError(), {"response": '{"items":[]}', "done_reason": "length"}):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as root:
                directory = create_session(root)
                model = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
                status, output, _, _ = invoke(root, directory, model)
                self.assertEqual(status, 1)
                model.assert_called_once()
                self.assertEqual(publication_payload(output)["documents"], [])

    def test_oversized_transcript_falls_back_in_isolation_and_records_actual_usage(self):
        with tempfile.TemporaryDirectory() as root:
            text = SOURCE + "[SPEAKER_01] " + "Current source material. " * 1800
            directory = create_session(root, text)
            model = Mock()
            calls = []
            def summary(**kwargs):
                calls.append(kwargs)
                kwargs["usage_callback"]({"prompt_eval_count": 20, "eval_count": 10, "runtime_seconds": 0.1})
                return "# Model output\nNone noted."
            status, output, _, reduced = invoke(root, directory, model, flags=["--synthesis-num-ctx", "8192", "--map-num-ctx", "12000", "--reduce-num-ctx", "16000"], summary=summary)
            self.assertEqual(status, 0)
            model.assert_not_called()
            self.assertGreater(reduced.call_count, 3)
            self.assertEqual({row["num_ctx"] for row in calls}, {12000, 16000})
            report = json.loads((output / "whole-run.json").read_text())
            self.assertEqual(report["processing_mode"], "map_reduce_fallback")
            self.assertEqual(report["fallback_reason"], "evidence_context_budget_exceeded")
            self.assertTrue(report["coverage"]["complete"])
            self.assertEqual(report["coverage"]["whole_included_record_ids"], [])
            self.assertEqual(report["calls"][0]["prompt_eval_count"], 20)
            self.assertFalse((Path(root) / "production" / "session").exists())

    def test_fallback_failure_removes_all_public_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root, SOURCE + "[SPEAKER_01] " + "Context. " * 3000)
            def fallback(**kwargs):
                if "Owner:" in kwargs["prompt"]:
                    raise TimeoutError()
                raise TimeoutError()
            status, output, _, _ = invoke(root, directory, Mock(), flags=["--synthesis-num-ctx", "8192"], summary=fallback)
            self.assertEqual(status, 1)
            self.assertEqual(publication_payload(output)["documents"], [])

    def test_environment_and_cli_context_are_independent_of_map_reduce(self):
        for flags, expected in (([], 65536), (["--synthesis-num-ctx", "98304"], 98304)):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as root:
                directory = create_session(root)
                model = FakeModel()
                status, output, _, _ = invoke(root, directory, model, flags=flags, env={"MEETING_SYNTHESIS_NUM_CTX": "65536", "MEETING_REDUCE_NUM_CTX": "32768"})
                self.assertEqual(status, 0)
                self.assertEqual(model.requests[0]["num_ctx"], expected)
                for request in model.requests:
                    self.assertLessEqual(whole._cost(request["prompt"], request["system"], request["format"], whole.TokenCounter(), request["num_predict"]), request["num_ctx"])

    def test_reuse_existing_gpu_lock_for_both_calls_and_no_nested_supervisor(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            model = FakeModel()
            with patch.object(whole.subprocess, "run") as process:
                status, _, _, _ = invoke(root, directory, model)
                self.assertEqual(status, 0)
                process.assert_not_called()
                self.assertEqual(len(model.requests), 2)

    def test_standalone_uses_one_existing_gpu1_supervisor(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            output = Path(root) / "experiment"
            argv = ["summary", str(directory), "--synthesis-mode", "whole", "--experiment-output-dir", str(output), "--ollama-url", "http://localhost"]
            model = FakeModel()
            def child(command):
                with patch.dict(os.environ, {"AIHUB_GPU_LOCK_HELD_FILE": "/tmp/test-whole-gpu"}):
                    result = engine.main()
                return subprocess.CompletedProcess(command, result)
            with patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(Path(root) / "production"), "AIHUB_GPU1_LOCK_FILE": "/tmp/test-whole-gpu"}, clear=True), patch.object(sys, "argv", argv), patch.object(whole.sys, "platform", "linux"), patch.object(whole.local, "_http_call", side_effect=model), patch.object(whole.subprocess, "run", side_effect=child) as process, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(engine.main(), 0)
                process.assert_called_once()
                self.assertEqual(process.call_args.args[0][2], "gpu1")
                self.assertEqual(len(model.requests), 2)

    def test_destination_and_lesson_guards_never_touch_production(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            production = Path(root) / "production"
            production.mkdir()
            (production / "sentinel").write_text("KEEP")
            status, _, _, _ = invoke(root, directory, destination=production / "new-experiment")
            self.assertEqual(status, 2)
            status, _, _, _ = invoke(root, directory, destination=production)
            self.assertEqual(status, 2)
            self.assertEqual((production / "sentinel").read_text(), "KEEP")
            with self.assertRaises(SystemExit):
                invoke(root, directory, flags=["--profile", "lesson"])

    def test_default_and_explicit_map_reduce_keep_identical_existing_behavior(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            runs = []
            for flags in ([], ["--synthesis-mode", "map-reduce"]):
                calls = []
                def summarize(**kwargs):
                    calls.append(kwargs)
                    return "# Model output\nNone noted."
                with patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(Path(root) / "production")}, clear=True), patch.object(sys, "argv", ["summary", str(directory), *flags]), patch.object(engine, "call_ollama", side_effect=summarize), patch.object(whole, "run_experiment") as experiment, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(engine.main(), 0)
                    experiment.assert_not_called()
                output = Path(root) / "production" / "session"
                self.assertFalse((output / "whole-run.json").exists())
                runs.append((calls, publication_payload(output)))
            self.assertEqual(runs[0], runs[1])

    def test_collective_commitment_has_no_invented_owner_and_tentative_group_is_rejected(self):
        for tentative in (False, True):
            with self.subTest(tentative=tentative), tempfile.TemporaryDirectory() as root:
                statement = "We should send the updated safety notice tomorrow." if tentative else "We will try to send the updated safety notice tomorrow."
                directory = create_session(root, SOURCE.replace("I'll send the updated safety notice tomorrow.", statement))
                def mutate(data, records):
                    next(row for row in data["items"] if row["kind"] == "action")["owners"] = []
                status, output, _, _ = invoke(root, directory, FakeModel(mutate_evidence=mutate))
                if tentative:
                    self.assertEqual(status, 1)
                    self.assertEqual(publication_payload(output)["documents"], [])
                else:
                    self.assertEqual(status, 0)
                    self.assertIn("Owner not recorded – We will try to send", (output / "action-items.md").read_text())
                    self.assertIn("unassigned_collective_action", (output / "minutes-qa.json").read_text())

    def test_configured_local_tokenizer_is_used_without_downloads(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            counter = Mock(side_effect=lambda text: max(1, len(text) // 4))
            counter.method = "configured_local_tokenizer"
            with patch.object(whole, "TokenCounter", return_value=counter) as tokenizer:
                status, output, _, _ = invoke(root, directory, flags=["--synthesis-tokenizer", str(Path(root) / "matching-tokenizer.json")])
                self.assertEqual(status, 0)
                tokenizer.assert_called_once_with(str(Path(root) / "matching-tokenizer.json"))
                self.assertEqual(json.loads((output / "whole-run.json").read_text())["token_count_method"], "configured_local_tokenizer")

    def test_complete_document_input_budget_failure_falls_back_without_second_call(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            counter = Mock(side_effect=lambda text: 100000 if '"validated_evidence"' in text else len(text.encode("utf-8")))
            counter.method = "synthetic_token_counter"
            model = FakeModel()
            def summarize(**kwargs):
                kwargs["usage_callback"]({"prompt_eval_count": 20, "eval_count": 10})
                return "# Model output\nNone noted."
            with patch.object(whole, "TokenCounter", return_value=counter):
                status, output, _, _ = invoke(root, directory, model, summary=summarize)
                self.assertEqual(status, 0)
                self.assertEqual(len(model.requests), 1)
                report = json.loads((output / "whole-run.json").read_text())
                self.assertEqual(report["fallback_reason"], "documents_context_budget_exceeded")
                self.assertTrue(report["coverage"]["complete"])
                self.assertTrue(report["coverage"]["whole_input_complete"])

    def test_motion_cannot_borrow_outcome_or_ignore_conflicting_named_roles(self):
        for case in ("other_motion", "role_conflict"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                text = SOURCE
                if case == "other_motion":
                    text += "[SPEAKER_00] I move to review the schedule.\n[SPEAKER_00] Carried."
                else:
                    text = text.replace("[SPEAKER_00] The meeting", "[SPEAKER_00] Motion by Riley, seconded by Casey.\n[SPEAKER_00] The meeting")
                directory = create_session(root, text)
                status, output, _, _ = invoke(root, directory)
                self.assertEqual(status, 1)
                self.assertEqual(publication_payload(output)["documents"], [])

    def test_qualified_commitment_is_preserved_and_omission_adds_qa(self):
        for omit in (False, True):
            with self.subTest(omit=omit), tempfile.TemporaryDirectory() as root:
                directory = create_session(root, SOURCE.replace("I'll send the updated", "When I get back, I'll try to send the updated"))
                def mutate(data, records):
                    if omit:
                        data["items"] = [row for row in data["items"] if row["kind"] != "action"]
                status, output, _, _ = invoke(root, directory, FakeModel(mutate_evidence=mutate))
                self.assertEqual(status, 0)
                if not omit:
                    self.assertIn("When I get back, I'll try to send", (output / "action-items.md").read_text())
                else:
                    self.assertIn("source_commitment_omission", (output / "minutes-qa.json").read_text())
                    self.assertTrue(json.loads((output / "whole-run.json").read_text())["unrepresented_commitment_record_ids"])

    def test_destination_collision_and_promotion_failure_leave_no_partial_documents(self):
        for collide in (True, False):
            with self.subTest(collide=collide), tempfile.TemporaryDirectory() as root:
                directory = create_session(root)
                output = Path(root) / "experiment"
                model = FakeModel()
                def call(request):
                    result = model(request)
                    if collide and len(model.requests) == 2:
                        output.mkdir()
                        (output / "sentinel").write_text("KEEP")
                    return result
                with patch.object(whole.os, "rename", side_effect=OSError("PRIVATE_ERROR")) if not collide else contextlib.nullcontext():
                    status, _, logs, _ = invoke(root, directory, call)
                self.assertIn(status, (1, 2))
                self.assertNotIn("PRIVATE_ERROR", logs)
                for stage in Path(root).glob(".whole-experiment-*"):
                    self.assertEqual(publication_payload(stage)["documents"], [])
                if collide:
                    self.assertEqual((output / "sentinel").read_text(), "KEEP")
                    self.assertEqual(publication_payload(output)["documents"], [])

    def test_fallback_usage_callback_records_api_counts_without_changing_response(self):
        data = {"response": " Model output ", "prompt_eval_count": 34, "eval_count": 12}
        usage = []
        with patch.object(engine.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(data).encode())):
            result = engine.call_ollama("http://localhost", "local", "Synthetic prompt", usage_callback=usage.append)
        self.assertEqual(result, "Model output")
        self.assertEqual(usage[0]["prompt_eval_count"], 34)
        self.assertEqual(usage[0]["eval_count"], 12)
        self.assertNotIn("Synthetic prompt", json.dumps(usage))

    def test_private_artifacts_and_references_are_explicitly_excluded_from_exports(self):
        with tempfile.TemporaryDirectory() as root:
            directory = create_session(root)
            status, output, _, _ = invoke(root, directory)
            self.assertEqual(status, 0)
            for name in whole.PRIVATE_FILES:
                ignored = subprocess.run(["git", "check-ignore", "--no-index", "private-experiment/" + name], cwd=ROOT, capture_output=True)
                self.assertEqual(ignored.returncode, 0)
                if os.name == "posix":
                    self.assertEqual((output / name).stat().st_mode & 0o777, 0o600)
            with (output / "minutes-draft.md").open("a") as doc:
                doc.write("\n" + "\n".join(f"[private]({name})" for name in whole.PRIVATE_FILES))
            payload = json.dumps(publication_payload(output))
            self.assertFalse(any(name in payload for name in whole.PRIVATE_FILES))
            export_archive(output, Path(root) / "public.zip")
            with zipfile.ZipFile(Path(root) / "public.zip") as archive:
                self.assertEqual(set(archive.namelist()), set(PUBLIC_MEETING_FILENAMES))


if __name__ == "__main__":
    unittest.main()
