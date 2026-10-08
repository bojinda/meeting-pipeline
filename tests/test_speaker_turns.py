"""Synthetic source bindings, explicit corrections, and bounded two-pass review."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import suggest_meeting_speakers as cli
import ollama_session_summary as engine
import speaker_review_worker as worker
from meeting_postprocess import speaker_turns as turns
from meeting_postprocess import speaker_turn_review as review
from meeting_postprocess import speaker_review as legacy
from meeting_postprocess.speaker_suggestions import build_speaker_suggestions
from meeting_postprocess.redaction import redact_chunks
from meeting_postprocess.sections import prepare_chunks
from meeting_postprocess.publication import publication_payload, export_archive, export_documents, PUBLIC_MEETING_FILENAMES


SOURCE = "[SPEAKER_00] Let's get started.\n[SPEAKER_03] My name is Taylor Morgan.\n[SPEAKER_04] Casey, could you give the report?\n[SPEAKER_03] Yes, I'll check the schedule."


def session(root, text=SOURCE):
    directory = Path(root) / "session"
    index = directory / "chunks_out" / "transcript_chunks.jsonl"
    index.parent.mkdir(parents=True)
    chunks = [{"chunk_id": "1.2", "start_time": 10.0, "end_time": 20.0, "text": text}]
    index.write_text(json.dumps(chunks[0]) + "\n", encoding="utf-8")
    return directory, chunks


def run_cli(root, args, env=None):
    with patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(Path(root) / "outputs"), **(env or {})}, clear=True), patch.object(sys, "argv", ["speakers", *map(str, args)]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return cli.main()


def claim(turn_id, name, refs=None, kind="self_identification", verification=False, verdict="supported"):
    row = {"turn_id": turn_id, "name": name, "confidence": "high", "evidence_turn_ids": refs or [turn_id], "evidence_type": kind, "conflicting_names": []}
    if verification:
        row["verdict"] = verdict
    return row


def model_reply(request):
    payload = json.loads(request["prompt"])
    verification = "proposed_assignments" in payload
    target = next(row for row in payload["transcript_turns"] if "My name is Taylor Morgan." in row["text"])
    return json.dumps({"candidates": [claim(target["turn_id"], "Taylor Morgan", verification=verification)]})


class TurnBindingTests(unittest.TestCase):
    def test_ids_are_repeatable_source_bound_and_use_containing_times(self):
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root)
            index = directory / "chunks_out" / "transcript_chunks.jsonl"
            original = index.read_bytes()
            first = turns.turn_catalog(directory)
            self.assertEqual(first, turns.turn_catalog(directory))
            self.assertEqual(len({row["turn_id"] for row in first["turns"]}), 4)
            self.assertEqual(first["turns"][0]["start_time"], 10.0)
            self.assertEqual(first["turns"][0]["time_precision"], "containing_chunk")
            changed = [{**chunks[0], "text": SOURCE.replace("schedule", "report")}]
            second = turns.turn_catalog(directory, changed)
            self.assertEqual(first["turns"][0]["turn_id"], second["turns"][0]["turn_id"])
            self.assertNotEqual(first["turns"][-1]["turn_id"], second["turns"][-1]["turn_id"])
            self.assertEqual(index.read_bytes(), original)

    def test_existing_whisperx_metadata_supplies_precise_time_without_transcribing(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _ = session(root, "[SPEAKER_03] My name is Taylor Morgan.")
            source = directory / "recording.json"
            source.write_text(json.dumps({"segments": [{"speaker": "SPEAKER_03", "text": "My name is Taylor Morgan.", "start": 12.5, "end": 13.7}]}))
            before = source.read_bytes()
            catalog = turns.turn_catalog(directory)
            self.assertEqual(catalog["turns"][0]["time_precision"], "whisperx_turn")
            self.assertEqual(catalog["turns"][0]["start_time"], 12.5)
            self.assertEqual(catalog["turns"][0]["end_time"], 13.7)
            self.assertEqual(before, source.read_bytes())

    def test_identical_text_in_distinct_source_turns_has_distinct_ids(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _ = session(root, "[SPEAKER_03] Yes.\n[SPEAKER_03] Yes.")
            rows = turns.turn_catalog(directory)["turns"]
            self.assertNotEqual(rows[0]["turn_id"], rows[1]["turn_id"])

    def test_redaction_hides_content_but_preserves_source_binding(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _ = session(root, "[SPEAKER_03] Visible. Redact the following. PRIVATE_TURN_SECRET.\n[SPEAKER_04] PRIVATE_OTHER_SECRET. End redaction. Back to business.")
            catalog = turns.turn_catalog(directory)
            self.assertNotIn("PRIVATE_", json.dumps(catalog))
            self.assertEqual(len(catalog["turns"]), 2)
            self.assertTrue(all(row["redaction_gap"] for row in catalog["turns"]))
            self.assertEqual(catalog, turns.turn_catalog(directory))

    def test_correction_overrides_global_alias_only_for_exact_turn(self):
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root)
            catalog = turns.turn_catalog(directory)
            target = catalog["turns"][-1]["turn_id"]
            aliases = {"SPEAKER_03": "Taylor Morgan"}
            turns.approve_turn(directory, catalog, target, "Casey")
            document = turns.correction_document(directory)
            corrected = turns.corrected_chunks(redact_chunks(chunks).chunks, catalog, document, aliases)
            prepared = prepare_chunks(corrected, aliases)
            text = "\n".join(row["text"] for row in prepared)
            self.assertIn("[Taylor Morgan] My name is Taylor Morgan.", text)
            self.assertIn("[Casey] Yes, I'll check the schedule.", text)
            self.assertNotIn(target, text)
            self.assertEqual(turns.read_index(directory), chunks)
            self.assertFalse((directory / "speaker_aliases.json").exists())
            turns.remove_turn(directory, target)
            self.assertEqual(turns.corrected_chunks(redact_chunks(chunks).chunks, catalog, turns.correction_document(directory), aliases), redact_chunks(chunks).chunks)

    def test_stale_or_foreign_corrections_fail_closed_and_can_be_removed(self):
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root)
            catalog = turns.turn_catalog(directory)
            target = catalog["turns"][-1]["turn_id"]
            turns.approve_turn(directory, catalog, target, "Casey")
            stale = turns.turn_catalog(directory, [{**chunks[0], "text": SOURCE.replace("schedule", "minutes")}])
            with self.assertRaises(ValueError):
                turns.effective_turns(stale, turns.correction_document(directory), {})
            other, _ = session(Path(root) / "other")
            (other / turns.CORRECTIONS_FILE).write_bytes((directory / turns.CORRECTIONS_FILE).read_bytes())
            with self.assertRaises(ValueError):
                turns.correction_document(other)
            turns.remove_turn(directory, target)
            self.assertEqual(turns.correction_document(directory)["corrections"], {})

    def test_inspect_approve_conflicts_and_remove_commands_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _ = session(root)
            run_cli(root, ["inspect-turns", directory])
            output = Path(root) / "outputs" / "session"
            catalog = json.loads((output / turns.TURNS_FILE).read_text())
            target = catalog["turns"][-1]["turn_id"]
            self.assertFalse((directory / turns.CORRECTIONS_FILE).exists())
            with self.assertRaises(SystemExit):
                run_cli(root, ["approve-turn", directory, "--turn-id", target])
            run_cli(root, ["approve-turn", directory, "--turn-id", target, "--name", "Casey"])
            run_cli(root, ["turn-conflicts", directory])
            self.assertTrue(json.loads((output / turns.CONFLICTS_FILE).read_text())["conflicts"])
            run_cli(root, ["remove-turn", directory, "--turn-id", target])
            self.assertEqual(turns.correction_document(directory)["corrections"], {})

    def test_fully_redacted_turn_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _ = session(root, "[SPEAKER_03] Redact the following. PRIVATE_TURN_SECRET.")
            catalog = turns.turn_catalog(directory)
            self.assertEqual(catalog["turns"], [])
            with self.assertRaises(ValueError):
                turns.approve_turn(directory, catalog, "Tmissing", "Taylor Morgan")

    def test_unlabelled_continuation_and_correction_permissions(self):
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root, "[SPEAKER_03] First sentence.\nContinued report.\n[SPEAKER_04] Reply.")
            catalog = turns.turn_catalog(directory)
            self.assertEqual(catalog["turns"][1]["source_speaker"], "SPEAKER_03")
            turns.approve_turn(directory, catalog, catalog["turns"][-1]["turn_id"], "Casey")
            corrected = turns.corrected_chunks(redact_chunks(chunks).chunks, catalog, turns.correction_document(directory), {})
            self.assertIn("[SPEAKER_03] Continued report.", corrected[0]["text"])
            if os.name == "posix":
                self.assertEqual((directory / turns.CORRECTIONS_FILE).stat().st_mode & 0o777, 0o600)

    def test_correction_preserves_colon_labels_punctuation_and_unrelated_lines(self):
        text = "[SPEAKER_03]: Original punctuation: unchanged!\n[SPEAKER_04]: Selected; report.\n[SPEAKER_03]: Another sentence?"
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root, text)
            source_bytes = (directory / "chunks_out" / "transcript_chunks.jsonl").read_bytes()
            catalog = turns.turn_catalog(directory)
            target = catalog["turns"][1]["turn_id"]
            turns.approve_turn(directory, catalog, target, "Casey")
            redacted = redact_chunks(chunks).chunks
            unchanged = copy.deepcopy(redacted)
            corrected = turns.corrected_chunks(redacted, catalog, turns.correction_document(directory), {})
            expected = copy.deepcopy(redacted)
            expected[0]["text"] = expected[0]["text"].replace("[SPEAKER_04]", "[Casey]", 1)
            self.assertEqual(corrected, expected)
            self.assertEqual(redacted, unchanged)
            self.assertEqual((directory / "chunks_out" / "transcript_chunks.jsonl").read_bytes(), source_bytes)

    def test_correction_of_unlabelled_continuation_leaves_neighboring_turns_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root, "[SPEAKER_03]: Report begins.\nUnlabelled continuation: details!\n[SPEAKER_04] Response.")
            catalog = turns.turn_catalog(directory)
            turns.approve_turn(directory, catalog, catalog["turns"][1]["turn_id"], "Casey")
            redacted = redact_chunks(chunks).chunks
            corrected = turns.corrected_chunks(redacted, catalog, turns.correction_document(directory), {})
            lines = redacted[0]["text"].splitlines()
            lines[1] = lines[1].replace("[SPEAKER_03]", "[Casey]", 1)
            self.assertEqual(corrected[0]["text"], "\n".join(lines))

    def test_repeated_turns_are_corrected_only_at_the_selected_source_position(self):
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root, "[SPEAKER_03]: Yes.\n[SPEAKER_03]: Yes.\n[SPEAKER_03]: Yes.")
            catalog = turns.turn_catalog(directory)
            turns.approve_turn(directory, catalog, catalog["turns"][1]["turn_id"], "Casey")
            redacted = redact_chunks(chunks).chunks
            corrected = turns.corrected_chunks(redacted, catalog, turns.correction_document(directory), {})
            lines = redacted[0]["text"].splitlines()
            lines[1] = lines[1].replace("[SPEAKER_03]", "[Casey]", 1)
            self.assertEqual(corrected[0]["text"], "\n".join(lines))

    def test_correction_preserves_cross_chunk_redaction_results_and_order(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _ = session(root)
            chunks = [
                {"chunk_id": "1.1", "text": "[SPEAKER_03]: Before! Redact the following. PRIVATE_FIRST.\n[SPEAKER_04] PRIVATE_SECOND."},
                {"chunk_id": "1.2", "text": "[SPEAKER_04]: PRIVATE_THIRD. End redaction. After?\n[SPEAKER_03]: Last; item."},
            ]
            catalog = turns.turn_catalog(directory, chunks)
            target = next(row for row in catalog["turns"] if "After?" in row["text"])
            turns.approve_turn(directory, catalog, target["turn_id"], "Casey")
            result = redact_chunks(chunks)
            private_before = copy.deepcopy(result.redactions)
            redacted_before = copy.deepcopy(result.chunks)
            corrected = turns.corrected_chunks(result.chunks, catalog, turns.correction_document(directory), {})
            expected = copy.deepcopy(result.chunks)
            expected[1]["text"] = expected[1]["text"].replace("[SPEAKER_04]", "[Casey]", 1)
            self.assertEqual(corrected, expected)
            self.assertNotIn("PRIVATE_", json.dumps(corrected))
            self.assertNotIn("PRIVATE_", json.dumps(catalog))
            self.assertEqual(result.redactions, private_before)
            self.assertEqual(result.chunks, redacted_before)

    def test_unverifiable_redacted_alignment_fails_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root)
            catalog = turns.turn_catalog(directory)
            turns.approve_turn(directory, catalog, catalog["turns"][-1]["turn_id"], "Casey")
            document = turns.correction_document(directory)
            redacted = redact_chunks(chunks).chunks
            for damaged in ([{**redacted[0], "text": redacted[0]["text"] + " Altered."}], [{**redacted[0], "chunk_id": "wrong"}], []):
                with self.subTest(damaged=bool(damaged)):
                    before = copy.deepcopy(damaged)
                    with self.assertRaisesRegex(ValueError, "alignment cannot be verified"):
                        turns.corrected_chunks(damaged, catalog, document, {})
                    self.assertEqual(damaged, before)
            legacy_catalog = copy.deepcopy(catalog)
            del legacy_catalog["turns"][0]["redacted_text"]
            with self.assertRaisesRegex(ValueError, "alignment cannot be verified"):
                turns.corrected_chunks(redacted, legacy_catalog, document, {})

    def test_private_artifacts_and_links_are_excluded_from_all_exports(self):
        import zipfile
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            for private in (turns.CORRECTIONS_FILE, turns.TURNS_FILE, turns.CONFLICTS_FILE):
                (output / private).write_text("PRIVATE_TURN_DATA")
                ignored = subprocess.run(["git", "check-ignore", "--no-index", "private-session/" + private], cwd=ROOT, capture_output=True)
                self.assertEqual(ignored.returncode, 0)
            for name in PUBLIC_MEETING_FILENAMES:
                (output / name).write_text("# Public\n" + "\n".join(f"[private]({name})" for name in (turns.CORRECTIONS_FILE, turns.TURNS_FILE, turns.CONFLICTS_FILE)))
            self.assertNotIn("speaker", json.dumps(publication_payload(output)))
            export_documents(output, output / "website")
            export_archive(output, output / "public.zip")
            with zipfile.ZipFile(output / "public.zip") as archive:
                self.assertEqual(set(archive.namelist()), set(PUBLIC_MEETING_FILENAMES))
                self.assertTrue(all(b"speaker" not in archive.read(name) for name in archive.namelist()))


class CandidateValidationTests(unittest.TestCase):
    def validate(self, row, verification=False):
        data = {"candidates": [row]}
        original = copy.deepcopy(data)
        result = review._valid_rows(data, {"Tsource": {"text": "Taylor Morgan introduced Casey."}}, verification)
        self.assertEqual(data, original)
        return result

    def test_supported_conflict_is_retained_without_becoming_a_proposal(self):
        row = {**claim("Tsource", "Taylor Morgan"), "conflicting_names": ["Casey"]}
        candidates, issues = self.validate(row)
        self.assertEqual(candidates, [row])
        self.assertFalse(issues)

    def test_unsupported_primary_is_not_replaced_by_supported_conflict(self):
        row = {**claim("Tsource", "Riley"), "conflicting_names": ["Casey"]}
        candidates, issues = self.validate(row)
        self.assertEqual(candidates, [])
        self.assertIn("name_absent_from_cited_source", issues)

    def test_null_primary_with_conflicts_remains_one_null_proposal(self):
        row = {**claim("Tsource", None), "conflicting_names": ["Taylor Morgan", "Casey"]}
        candidates, issues = self.validate(row)
        self.assertEqual(candidates, [row])
        self.assertFalse(issues)

    def test_unsupported_conflict_rejects_row_without_substitution(self):
        row = {**claim("Tsource", "Taylor Morgan"), "conflicting_names": ["Riley", "Casey"]}
        candidates, issues = self.validate(row)
        self.assertEqual(candidates, [])
        self.assertIn("name_absent_from_cited_source", issues)

    def test_repeated_conflict_names_do_not_duplicate_candidates(self):
        row = {**claim("Tsource", "Taylor Morgan"), "conflicting_names": ["Casey", "Casey", "Taylor Morgan"]}
        candidates, issues = self.validate(row)
        self.assertEqual(candidates, [row])
        self.assertFalse(issues)

    def test_verification_verdict_and_primary_are_preserved_with_conflicts(self):
        row = {**claim("Tsource", "Taylor Morgan", verification=True, verdict="unsupported"), "conflicting_names": ["Casey"]}
        candidates, issues = self.validate(row, verification=True)
        self.assertEqual(candidates, [row])
        self.assertFalse(issues)


class TwoPassTests(unittest.TestCase):
    def setup_review(self, root, text=SOURCE, aliases=None):
        directory, chunks = session(root, text)
        catalog = turns.turn_catalog(directory)
        return directory, build_speaker_suggestions(chunks, aliases), turns.effective_turns(catalog, turns.correction_document(directory), aliases or {})

    def test_two_passes_use_independent_context_and_smaller_verification(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"MEETING_REDUCE_NUM_CTX": "8192"}, clear=True):
            directory, report, rows = self.setup_review(root)
            model = Mock(side_effect=model_reply)
            result = review.review_turns(report, rows, {}, [], model="local-review-model", ollama_url="http://localhost:11434", options=review.settings(), call=model)
            self.assertEqual(model.call_count, 2)
            discovery, verification = [call.args[0] for call in model.call_args_list]
            self.assertEqual(discovery["num_ctx"], 98304)
            self.assertEqual(discovery["num_predict"], 16384)
            self.assertLessEqual(verification["num_predict"], 4096)
            self.assertLess(verification["num_ctx"], 98304)
            self.assertIn("Independently verify", verification["system"])
            proposed = json.loads(verification["prompt"])["proposed_assignments"]
            self.assertNotIn("confidence", proposed[0])
            self.assertEqual(result["llm_review"]["status"], "completed")
            self.assertTrue(result["llm_review"]["coverage"]["complete"])
            self.assertEqual(result["llm_review"]["assignments"][0]["status"], "grounded_advisory")
            self.assertFalse((directory / turns.CORRECTIONS_FILE).exists())
            self.assertFalse((directory / "speaker_aliases.json").exists())

    def test_full_coverage_with_large_input_and_reserved_discovery_output(self):
        text = SOURCE + "\n" + "\n".join(f"[SPEAKER_04] Routine report {i}." for i in range(454))
        class Counter:
            method = "synthetic_token_counter"
            def __call__(self, text):
                return 75943 if '"transcript_turns"' in text else len(text.encode("utf-8"))
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, text)
            def completion(request):
                return {"response": model_reply(request), "prompt_eval_count": 75943, "done_reason": "stop"}
            model = Mock(side_effect=completion)
            counter = Counter()
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(98304), call=model, counter=counter)
            self.assertEqual(len(rows), 458)
            self.assertEqual(model.call_count, 2)
            coverage = result["llm_review"]["coverage"]
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["window_count"], 1)
            self.assertEqual(len(coverage["included_turn_ids"]), 458)
            self.assertEqual(result["llm_review"]["status"], "completed")
            for call in model.call_args_list:
                request = call.args[0]
                self.assertLessEqual(review._cost(request["prompt"], request["system"], request["format"], counter, request["num_predict"]), request["num_ctx"])
            system = model.call_args_list[0].args[0]["system"]
            self.assertIn("Do not duplicate candidate proposals", system)
            self.assertIn("Preserve conflicting evidence", system)

    def test_discovery_output_limit_skips_verification_even_with_valid_json(self):
        for valid_json in (True, False):
            with self.subTest(valid_json=valid_json), tempfile.TemporaryDirectory() as root:
                directory, report, rows = self.setup_review(root)
                raw = json.dumps({"candidates": [claim(rows[1]["turn_id"], "Taylor Morgan")]}) if valid_json else '{"candidates":['
                model = Mock(return_value={"response": raw, "done_reason": "length", "eval_count": 16384})
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(98304), call=model)
                model.assert_called_once()
                self.assertEqual(model.call_args.args[0]["num_predict"], 16384)
                self.assertEqual(result["llm_review"]["status"], "incomplete")
                self.assertEqual(result["llm_review"]["passes"][0]["diagnostics"]["failure_category"], "generation_token_limit")
                self.assertEqual(result["llm_review"]["assignments"], [])
                self.assertEqual(result["suggestions"], report["suggestions"])
                self.assertFalse((directory / turns.CORRECTIONS_FILE).exists())
                self.assertFalse((directory / "speaker_aliases.json").exists())

    def test_larger_output_reserve_moves_over_budget_input_to_explicit_windows(self):
        text = SOURCE + "\n" + "\n".join(f"[SPEAKER_04] Routine report {i}." for i in range(12))
        class Counter:
            method = "synthetic_token_counter"
            def __call__(self, text):
                if '"transcript_turns"' in text:
                    return len(json.loads(text)["transcript_turns"]) * 5400
                return len(text.encode("utf-8"))
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, text)
            counter = Counter()
            self.assertLessEqual(review._cost(review._prompt(rows, []), review.DISCOVERY_SYSTEM, review.schema(), counter, 8192), 98304)
            self.assertGreater(review._cost(review._prompt(rows, []), review.DISCOVERY_SYSTEM, review.schema(), counter, 16384), 98304)
            model = Mock(return_value='{"candidates":[]}')
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(98304), call=model, counter=counter)
            coverage = result["llm_review"]["coverage"]
            self.assertFalse(coverage["complete"])
            self.assertGreater(coverage["window_count"], 1)
            self.assertTrue(coverage["omitted_turn_ids"])
            request = model.call_args.args[0]
            self.assertEqual(request["num_predict"], 16384)
            self.assertLessEqual(review._cost(request["prompt"], request["system"], request["format"], counter, request["num_predict"]), request["num_ctx"])

    def test_rejected_names_have_safe_field_specific_diagnostics(self):
        for field in ("name", "conflicting_names"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                _, report, rows = self.setup_review(root)
                target = rows[1]["turn_id"]
                row = claim(target, "Taylor Morgan")
                if field == "name":
                    row[field] = "**PRIVATE_DIAGNOSTIC_NAME**"
                else:
                    row[field] = ["SPEAKER_99"]
                model = Mock(return_value=json.dumps({"candidates": [row]}))
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                model.assert_called_once()
                diagnostics = result["llm_review"]["passes"][0]["diagnostics"]
                self.assertEqual(diagnostics["rejected_candidates"], [{"candidate_index": 0, "field": field, "rejection_category": "invalid_name", "turn_id": target}])
                encoded = json.dumps(diagnostics)
                for private in ("Taylor Morgan", "PRIVATE_DIAGNOSTIC_NAME", "My name is", "SPEAKER_99"):
                    self.assertNotIn(private, encoded)

    def test_independent_valid_candidate_reaches_verification_but_review_stays_partial(self):
        text = "[SPEAKER_03] My name is Taylor Morgan.\n[SPEAKER_04] My name is Casey."
        with tempfile.TemporaryDirectory() as root:
            directory, report, rows = self.setup_review(root, text)
            valid = claim(rows[0]["turn_id"], "Taylor Morgan")
            invalid = claim(rows[1]["turn_id"], "SPEAKER_04")
            verified = claim(rows[0]["turn_id"], "Taylor Morgan", verification=True)
            model = Mock(side_effect=[json.dumps({"candidates": [invalid, valid]}), json.dumps({"candidates": [verified]})])
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
            self.assertEqual(model.call_count, 2)
            info = result["llm_review"]
            self.assertEqual(info["status"], "incomplete")
            self.assertEqual(info["reason"], "partial_discovery_validation")
            self.assertEqual(len(info["assignments"]), 1)
            self.assertEqual(info["assignments"][0]["suggested_name"], "Taylor Morgan")
            self.assertFalse(info["assignments"][0]["auto_approvable"])
            self.assertTrue(info["assignments"][0]["requires_explicit_approval"])
            proposed = json.loads(model.call_args_list[1].args[0]["prompt"])["proposed_assignments"]
            self.assertEqual([row["name"] for row in proposed], ["Taylor Morgan"])
            self.assertEqual(result["suggestions"], report["suggestions"])
            self.assertFalse((directory / "speaker_aliases.json").exists())
            self.assertFalse((directory / turns.CORRECTIONS_FILE).exists())

    def test_rejected_candidate_dependencies_keep_discovery_fail_closed(self):
        cases = ("same_label", "overlap", "unknown_reference", "rejected_conflict", "valid_conflict", "description_link", "label_link", "source_conflict", "event_link", "empty_evidence", "same_identity")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                text = "[SPEAKER_03] My name is Taylor Morgan.\n[SPEAKER_04] My name is Casey."
                if case == "same_label":
                    text = text.replace("SPEAKER_04", "SPEAKER_03")
                if case == "same_identity":
                    text = text.replace("My name is Casey.", "My name is Taylor Morgan.")
                if case == "source_conflict":
                    text += "\n[SPEAKER_03] My name is Riley."
                if case == "event_link":
                    text += "\n[SPEAKER_04] Taylor, could you give the report?\n[SPEAKER_03] Yes, I'll check."
                directory, report, rows = self.setup_review(root, text)
                valid = claim(rows[0]["turn_id"], "Taylor Morgan")
                invalid = claim(rows[1]["turn_id"], "SPEAKER_04")
                if case == "overlap":
                    invalid["evidence_turn_ids"].append(rows[0]["turn_id"])
                if case == "unknown_reference":
                    invalid["evidence_turn_ids"].append("Tmissing")
                if case == "empty_evidence":
                    invalid["evidence_turn_ids"] = []
                if case == "rejected_conflict":
                    invalid["conflicting_names"] = ["Casey"]
                if case == "valid_conflict":
                    valid["conflicting_names"] = ["Casey"]
                    valid["evidence_turn_ids"].append(rows[1]["turn_id"])
                if case == "description_link":
                    invalid["name"] = "**Taylor / Casey**"
                if case == "label_link":
                    invalid["name"] = "SPEAKER_03"
                model = Mock(return_value=json.dumps({"candidates": [valid, invalid]}))
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                model.assert_called_once()
                self.assertEqual(result["llm_review"]["status"], "incomplete")
                self.assertEqual(result["llm_review"]["assignments"], [])
                self.assertFalse((directory / turns.CORRECTIONS_FILE).exists())

    def test_partial_verification_cannot_hide_rejected_same_label_identity(self):
        text = "[SPEAKER_03] My name is Taylor Morgan.\n[SPEAKER_04] My name is Casey."
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory() as root:
                _, report, rows = self.setup_review(root, text)
                proposed = claim(rows[0]["turn_id"], "Taylor Morgan")
                verified = claim(rows[0]["turn_id"], "Taylor Morgan", verification=True)
                invalid = claim(rows[0 if shared else 1]["turn_id"], "SPEAKER_04", verification=True)
                model = Mock(side_effect=[json.dumps({"candidates": [proposed]}), json.dumps({"candidates": [verified, invalid]})])
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                self.assertEqual(model.call_count, 2)
                info = result["llm_review"]
                self.assertEqual(info["status"], "incomplete")
                self.assertEqual(info["assignments"][0]["suggested_name"], None if shared else "Taylor Morgan")
                self.assertFalse(info["assignments"][0]["auto_approvable"])

    def test_context_env_and_cli_override_only_speaker_review(self):
        with patch.dict(os.environ, {"SPEAKER_REVIEW_NUM_CTX": "65536", "MEETING_REDUCE_NUM_CTX": "32768"}, clear=True):
            self.assertEqual(review.settings()["num_ctx"], 65536)
            self.assertEqual(review.settings(98304)["num_ctx"], 98304)
        with self.assertRaises(ValueError):
            review.settings(4096)

    def test_cli_context_overrides_environment(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _, _ = self.setup_review(root)
            with patch.object(legacy, "_http_call", side_effect=model_reply) as model:
                run_cli(root, ["suggest", directory, "--llm", "--speaker-review-num-ctx", "98304"], {"SPEAKER_REVIEW_NUM_CTX": "65536", "AIHUB_GPU1_LOCK_FILE": "/tmp/test-turn-lock", "AIHUB_GPU_LOCK_HELD_FILE": "/tmp/test-turn-lock"})
                self.assertEqual(model.call_args_list[0].args[0]["num_ctx"], 98304)

    def test_local_token_counter_can_fit_full_meeting_and_missing_file_fails_safely(self):
        import types
        text = SOURCE + "\n[SPEAKER_04] " + "Ordinary context. " * 6500
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, text)
            tokenizer_file = Path(root) / "tokenizer.json"
            tokenizer_file.write_text("{}")
            tokenizer = Mock()
            tokenizer.encode.side_effect = lambda text: types.SimpleNamespace(ids=list(range(len(text) // 4)))
            module = types.SimpleNamespace(Tokenizer=Mock())
            module.Tokenizer.from_file.return_value = tokenizer
            with patch.dict(sys.modules, {"tokenizers": module}):
                model = Mock(side_effect=model_reply)
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(tokenizer=tokenizer_file), call=model)
                self.assertTrue(result["llm_review"]["coverage"]["complete"])
                self.assertEqual(result["llm_review"]["coverage"]["token_count_method"], "configured_local_tokenizer")
                self.assertGreater(len(model.call_args_list[0].args[0]["prompt"]), 98304)
                model.reset_mock()
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(tokenizer=Path(root) / "missing.json"), call=model)
                model.assert_not_called()
                self.assertEqual(result["llm_review"]["diagnostics"]["failure_category"], "tokenizer_unavailable_or_invalid")

    def test_full_evidence_over_old_character_cap_is_not_truncated(self):
        text = SOURCE + "\n[SPEAKER_04] " + "Routine source context. " * 1200
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, text)
            model = Mock(side_effect=model_reply)
            result = review.review_turns(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
            prompt = model.call_args_list[0].args[0]["prompt"]
            self.assertGreater(len(prompt), 18000)
            self.assertIn(rows[-1]["text"], prompt)
            self.assertTrue(result["llm_review"]["coverage"]["complete"])

    def test_overlapping_windows_report_omissions_and_respect_budget(self):
        text = "\n".join(f"[SPEAKER_{i % 2:02d}] Routine report {i}. " + "Context. " * 180 for i in range(12))
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, text)
            counter = review.TokenCounter()
            planned, oversized = review.windows(rows, [], 8192, counter, 1024)
            self.assertGreater(len(planned), 1)
            self.assertFalse(oversized)
            self.assertTrue(set(row["turn_id"] for row in planned[0]) & set(row["turn_id"] for row in planned[1]))
            model = Mock(return_value='{"candidates":[]}')
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(8192, window=1), call=model)
            model.assert_called_once()
            coverage = result["llm_review"]["coverage"]
            self.assertFalse(coverage["complete"])
            self.assertTrue(coverage["omitted_turn_ids"])
            self.assertEqual(coverage["window_index"], 1)
            req = model.call_args.args[0]
            self.assertLessEqual(review._cost(req["prompt"], req["system"], req["format"], counter, req["num_predict"]), req["num_ctx"])

    def test_oversized_single_turn_is_reported_without_silent_truncation(self):
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, "[SPEAKER_03] " + "Large. " * 5000)
            model = Mock()
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(8192), call=model)
            model.assert_not_called()
            self.assertEqual(result["llm_review"]["coverage"]["oversized_turn_ids"], [rows[0]["turn_id"]])

    def test_redacted_content_and_links_never_enter_either_pass(self):
        text = "[SPEAKER_03] Redact the following. PRIVATE_SPEAKER_SECRET. End redaction. My name is Taylor Morgan.\n[SPEAKER_04] Public business."
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, text)
            model = Mock(side_effect=model_reply)
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
            self.assertEqual(model.call_count, 2)
            self.assertNotIn("PRIVATE_SPEAKER_SECRET", json.dumps(result))
            self.assertTrue(all("PRIVATE_SPEAKER_SECRET" not in call.args[0]["prompt"] for call in model.call_args_list))

    def test_verification_cannot_override_a_conflicting_approved_turn(self):
        with tempfile.TemporaryDirectory() as root:
            directory, report, rows = self.setup_review(root)
            catalog = turns.turn_catalog(directory)
            target = catalog["turns"][1]["turn_id"]
            turns.approve_turn(directory, catalog, target, "Casey")
            approved = (directory / turns.CORRECTIONS_FILE).read_bytes()
            rows = turns.effective_turns(catalog, turns.correction_document(directory), {})
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model_reply)
            assignment = result["llm_review"]["assignments"][0]
            self.assertIsNone(assignment["suggested_name"])
            self.assertFalse(assignment["auto_approvable"])
            self.assertEqual((directory / turns.CORRECTIONS_FILE).read_bytes(), approved)

    def test_identity_evidence_does_not_propagate_to_another_turn_of_same_label(self):
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root)
            source, target = rows[1]["turn_id"], rows[-1]["turn_id"]
            model = Mock(side_effect=[json.dumps({"candidates": [claim(target, "Taylor Morgan", [source])]}), json.dumps({"candidates": [claim(target, "Taylor Morgan", [source], verification=True)]})])
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
            self.assertIsNone(result["llm_review"]["assignments"][0]["suggested_name"])
            self.assertIn("Casey", result["llm_review"]["assignments"][0]["conflicting_names"])

    def test_adjacent_question_context_and_fragmented_identity_keep_exact_targets(self):
        text = "[Chair] Who's that?\n[SPEAKER_03] Yeah, that's me.\n[SPEAKER_03] Taylor Morgan.\n[Chair] Casey, I've got a question for you.\n[Chair] Do you have the report?\n[SPEAKER_03] Yes, I'll check."
        with tempfile.TemporaryDirectory() as root:
            _, _, rows = self.setup_review(root, text)
            events = review.source_events(rows, {}, [])
            addressed = [event for event in events if "Casey" in event["candidate_names"]]
            self.assertTrue(addressed)
            self.assertTrue(all(event["target_turn_ids"] == [rows[-1]["turn_id"]] for event in addressed))
            identified = [event for event in events if "Taylor Morgan" in event["candidate_names"]]
            self.assertTrue(identified)
            self.assertTrue(all(rows[-1]["turn_id"] not in event["target_turn_ids"] for event in identified))

    def test_roster_variant_needs_actual_relationship_and_cited_name(self):
        roster = [{"name": "Taylor Morgan", "aliases": ["Taylor"], "role": ""}]
        for text, supported in (("My name is Taylor.", True), ("Taylor sent a report.", False), ("The report arrived.", False)):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as root:
                _, report, rows = self.setup_review(root, "[SPEAKER_03] " + text)
                target = rows[0]["turn_id"]
                model = Mock(side_effect=[json.dumps({"candidates": [claim(target, "Taylor Morgan")]}), json.dumps({"candidates": [claim(target, "Taylor Morgan", verification=True)]})])
                result = review.run_two_pass(report, rows, {}, roster, model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                assignments = result["llm_review"]["assignments"]
                self.assertEqual(bool(assignments and assignments[0]["suggested_name"]), supported)

    def test_verification_failure_never_retries_or_promotes_discovery(self):
        for completion in (TimeoutError(), "malformed", {"response": "{}", "done_reason": "length"}):
            with self.subTest(completion=type(completion).__name__), tempfile.TemporaryDirectory() as root:
                _, report, rows = self.setup_review(root)
                model = Mock(side_effect=[json.dumps({"candidates": [claim(rows[1]["turn_id"], "Taylor Morgan")]}), completion])
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                self.assertEqual(model.call_count, 2)
                self.assertIsNone(result["llm_review"]["assignments"][0]["suggested_name"])
                self.assertEqual(result["suggestions"], report["suggestions"])

    def test_redaction_and_intervening_speakers_block_turn_relationships(self):
        for middle in ("[SPEAKER_04] Redact the following. Private text. End redaction.\n", "[SPEAKER_04] I have another question.\n"):
            with self.subTest(middle=middle), tempfile.TemporaryDirectory() as root:
                _, report, rows = self.setup_review(root, "[Chair] Casey, could you give the report?\n" + middle + "[SPEAKER_03] Yes, I'll check.")
                target = rows[-1]["turn_id"]
                refs = [row["turn_id"] for row in rows]
                kind = "direct_address_response"
                model = Mock(side_effect=[json.dumps({"candidates": [claim(target, "Casey", refs, kind)]}), json.dumps({"candidates": [claim(target, "Casey", refs, kind, verification=True)]})])
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                self.assertFalse(any(row["suggested_name"] for row in result["llm_review"]["assignments"]))

    def test_disabling_llm_adds_no_calls_and_clears_stale_turn_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _, _ = self.setup_review(root, "[SPEAKER_03] Redact the following. PRIVATE_SECRET. End redaction. Public.")
            output = Path(root) / "outputs" / "session"
            output.mkdir(parents=True)
            (output / turns.TURNS_FILE).write_text("PRIVATE_SECRET")
            (output / turns.CONFLICTS_FILE).write_text("PRIVATE_SECRET")
            with patch.object(legacy, "_http_call") as model:
                run_cli(root, ["suggest", directory])
                model.assert_not_called()
            self.assertNotIn("PRIVATE_SECRET", (output / turns.TURNS_FILE).read_text())
            self.assertFalse((output / turns.CONFLICTS_FILE).exists())

    def test_verification_disagreement_and_unrelated_mentions_remain_unverified(self):
        for verdict in ("unsupported", "uncertain"):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as root:
                _, report, rows = self.setup_review(root)
                target = rows[1]["turn_id"]
                model = Mock(side_effect=[json.dumps({"candidates": [claim(target, "Taylor Morgan")]}), json.dumps({"candidates": [claim(target, "Taylor Morgan", verification=True, verdict=verdict)]})])
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                self.assertIsNone(result["llm_review"]["assignments"][0]["suggested_name"])
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, "[SPEAKER_03] Taylor Morgan prepared the report.")
            target = rows[0]["turn_id"]
            model = Mock(side_effect=[json.dumps({"candidates": [claim(target, "Taylor Morgan")]}), json.dumps({"candidates": [claim(target, "Taylor Morgan", verification=True)]})])
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
            self.assertEqual(result["llm_review"]["assignments"][0]["status"], "unverified_or_conflicting")

    def test_primary_conflicts_and_verification_alternatives_are_not_promoted(self):
        for primary, conflict, verifier_name in (("Taylor Morgan", "Casey", "Taylor Morgan"), (None, "Casey", None), ("Taylor Morgan", "Casey", "Casey")):
            with self.subTest(primary=primary, verifier_name=verifier_name), tempfile.TemporaryDirectory() as root:
                _, report, rows = self.setup_review(root)
                target = rows[1]["turn_id"]
                refs = [row["turn_id"] for row in rows]
                proposal = {**claim(target, primary, refs), "conflicting_names": [conflict]}
                verified = {**claim(target, verifier_name, refs, verification=True), "conflicting_names": [conflict]}
                model = Mock(side_effect=[json.dumps({"candidates": [proposal]}), json.dumps({"candidates": [verified]})])
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                info = result["llm_review"]
                self.assertEqual(model.call_count, 2)
                self.assertEqual(len(info["passes"][0]["candidates"]), 1)
                self.assertEqual(len(info["passes"][1]["candidates"]), 1)
                self.assertEqual(len(info["assignments"]), 1)
                self.assertEqual(info["assignments"][0]["name"], primary)
                self.assertIsNone(info["assignments"][0]["suggested_name"])
                self.assertIn("Casey", info["assignments"][0]["conflicting_names"])
                self.assertFalse(info["assignments"][0]["auto_approvable"])
                self.assertEqual(len(info["verification_leads"]), int(verifier_name != primary))

    def test_timeout_invalid_json_and_fabricated_references_keep_safe_fallback(self):
        for completion in (TimeoutError(), "not JSON", json.dumps({"candidates": [claim("Tmissing", "Taylor Morgan")]})):
            with self.subTest(completion=type(completion).__name__), tempfile.TemporaryDirectory() as root:
                _, report, rows = self.setup_review(root)
                model = Mock(side_effect=completion) if isinstance(completion, Exception) else Mock(return_value=completion)
                result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(), call=model)
                model.assert_called_once()
                self.assertEqual(result["suggestions"], report["suggestions"])
                self.assertNotEqual(result["llm_review"]["status"], "completed")

    def test_existing_gpu1_lock_is_reused_for_both_passes(self):
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root)
            with patch.dict(os.environ, {"AIHUB_GPU1_LOCK_FILE": "/tmp/test-turn-lock", "AIHUB_GPU_LOCK_HELD_FILE": "/tmp/test-turn-lock"}, clear=True), patch.object(legacy, "_http_call", side_effect=model_reply) as http, patch.object(review.subprocess, "run") as process:
                result = review.review_turns(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings())
                self.assertEqual(http.call_count, 2)
                process.assert_not_called()
                self.assertEqual(result["llm_review"]["status"], "completed")

    def test_standalone_supervisor_covers_two_passes_in_one_worker(self):
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root)
            def supervisor(argv, **kwargs):
                packet = json.loads(kwargs["input"])
                with patch.object(sys, "stdin", io.StringIO(json.dumps(packet))), patch.object(worker, "_http_call", side_effect=model_reply) as http, contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(worker.main(), 0)
                    self.assertEqual(http.call_count, 2)
                return subprocess.CompletedProcess(argv, 0, out.getvalue())
            with patch.dict(os.environ, {}, clear=True), patch.object(review.sys, "platform", "linux"), patch.object(review.subprocess, "run", side_effect=supervisor) as process:
                result = review.review_turns(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings())
                process.assert_called_once()
                self.assertEqual(process.call_args.args[0][2], "gpu1")
                self.assertNotIn("Taylor Morgan", " ".join(process.call_args.args[0]))
                self.assertEqual(result["llm_review"]["status"], "completed")

    def test_new_cli_defaults_to_two_pass_without_map_reduce_or_alias_writes(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _, _ = self.setup_review(root)
            with patch.object(legacy, "_http_call", side_effect=model_reply) as model, patch.object(engine, "call_ollama") as summary:
                run_cli(root, ["suggest", directory, "--llm"], {"AIHUB_GPU1_LOCK_FILE": "/tmp/test-turn-lock", "AIHUB_GPU_LOCK_HELD_FILE": "/tmp/test-turn-lock", "MEETING_REDUCE_NUM_CTX": "32768"})
                self.assertEqual(model.call_count, 2)
                self.assertEqual(model.call_args_list[0].args[0]["num_ctx"], 98304)
                summary.assert_not_called()
            self.assertFalse((directory / turns.CORRECTIONS_FILE).exists())
            self.assertFalse((directory / "speaker_aliases.json").exists())
            with self.assertRaises(SystemExit):
                run_cli(root, ["approve", directory, "--approve", "SPEAKER_03"])

    def test_two_pass_option_preserves_existing_public_processing(self):
        with tempfile.TemporaryDirectory() as root:
            directory, _ = session(root)
            records = []
            for enabled in (False, True):
                calls = []
                output = Path(root) / ("enabled" if enabled else "disabled")
                argv = ["summary", str(directory), "--profile", "meeting"]
                if enabled:
                    argv.extend(["--suggest-speakers", "--suggest-speakers-llm"])
                def summarize(**kwargs):
                    calls.append(kwargs)
                    return "# Public output\nNo additional items."
                with patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(output), "MEETING_REDUCE_NUM_CTX": "32768", "AIHUB_GPU1_LOCK_FILE": "/tmp/test-turn-lock", "AIHUB_GPU_LOCK_HELD_FILE": "/tmp/test-turn-lock"}, clear=True), patch.object(sys, "argv", argv), patch.object(engine, "call_ollama", side_effect=summarize), patch.object(legacy, "_http_call", side_effect=model_reply) as model, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(engine.main(), 0)
                    self.assertEqual(model.call_count, 2 if enabled else 0)
                records.append((calls, publication_payload(output / "session")))
            self.assertEqual(records[0], records[1])

    def test_verification_includes_competing_source_evidence_outside_selected_window(self):
        text = "[SPEAKER_03] My name is Taylor Morgan.\n" + "\n".join("[SPEAKER_04] Routine context. " + "Report. " * 200 for _ in range(8)) + "\n[Chair] Casey, could you give the report?\n[SPEAKER_03] Yes, I'll check."
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, text)
            model = Mock(side_effect=model_reply)
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(8192), call=model)
            self.assertFalse(result["llm_review"]["coverage"]["complete"])
            self.assertEqual(model.call_count, 2)
            verification = json.loads(model.call_args_list[1].args[0]["prompt"])
            self.assertIn(rows[-1]["turn_id"], [row["turn_id"] for row in verification["transcript_turns"]])
            self.assertIn("Casey", model.call_args_list[1].args[0]["prompt"])

    def test_verification_budget_overflow_is_reported_without_dropping_conflicts(self):
        text = "[SPEAKER_03] My name is Taylor Morgan.\n" + "\n".join("[Chair] Casey, could you give the report?\n[SPEAKER_03] Yes. " + "Report. " * 200 for _ in range(10))
        with tempfile.TemporaryDirectory() as root:
            _, report, rows = self.setup_review(root, text)
            model = Mock(side_effect=model_reply)
            result = review.run_two_pass(report, rows, {}, [], model="local", ollama_url="http://localhost", options=review.settings(8192), call=model)
            model.assert_called_once()
            self.assertEqual(result["llm_review"]["reason"], "verification_budget_exceeded")
            self.assertFalse(result["llm_review"]["verification_coverage"]["complete"])

    def test_meeting_applies_correction_before_map_without_public_turn_ids(self):
        with tempfile.TemporaryDirectory() as root:
            directory, chunks = session(root)
            (directory / "speaker_aliases.json").write_text('{"SPEAKER_03":"Taylor Morgan"}')
            catalog = turns.turn_catalog(directory)
            target = catalog["turns"][-1]["turn_id"]
            turns.approve_turn(directory, catalog, target, "Casey")
            calls = []
            def model(**kwargs):
                calls.append(kwargs)
                return "# Public output\nNo additional items."
            with patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(Path(root) / "outputs")}, clear=True), patch.object(sys, "argv", ["summary", str(directory), "--profile", "meeting"]), patch.object(engine, "call_ollama", side_effect=model), patch.object(legacy, "call_local_ollama") as speaker_model, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(engine.main(), 0)
                speaker_model.assert_not_called()
            prompts = json.dumps(calls)
            self.assertIn("[Taylor Morgan] My name is Taylor Morgan.", prompts)
            self.assertIn("[Casey] Yes, I'll check the schedule.", prompts)
            self.assertNotIn(target, prompts)
            self.assertEqual(turns.read_index(directory), chunks)
            output = Path(root) / "outputs" / "session"
            self.assertNotIn(target, json.dumps(publication_payload(output)))


if __name__ == "__main__":
    unittest.main()
