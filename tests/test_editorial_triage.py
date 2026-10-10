"""Automatic held drafts after private register triage; no model/backend calls."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed
from meeting_postprocess.commitments import commitment_evidence
from meeting_postprocess.publication import export_archive, export_documents, publication_payload
from meeting_postprocess.sections import ADJOURNMENT, BUSINESS
from test_editorial_notes import portion


LIVE = Path(os.environ.get("OCTOBER_EDITORIAL_LIVE_DIR", ROOT / "ignore/october-editorial-live.wnfcYj/editorial-output"))
LIVE_FILES = (ed.RESPONSE, "meeting_sections.jsonl", "chunk_summaries.jsonl", ed.BUDGETS)
MODEL_TEXT = "The model returned this issue discussion for operator review."
MEMBER_FILES = ("meeting-notes-draft.md", "action-items.md", "minutes-draft.md", "summary.md")


def proposal(identifier, task, ref="1:L1", owner="Taylor", concerns=()):
    return {"id": identifier, "category": "undertaking", "task": task, "owners": [owner],
            "source_ids": [ref], "member_facing": True, "concerns": list(concerns)}


def notes_response(ref="1:L3", text=MODEL_TEXT, concerns=()):
    return {"highlights": [], "previous_context": [],
            "issues": [{"heading": "Staffing discussion", "paragraphs": [{"text": text, "source_ids": [ref]}]}],
            "motions": [], "unresolved": [], "concerns": list(concerns)}


class RegisterTriageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer, models, pre_tokenizers
        cls.tokenizer_directory = tempfile.TemporaryDirectory()
        cls.tokenizer_path = Path(cls.tokenizer_directory.name) / "tokenizer.json"
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(cls.tokenizer_path))

    @classmethod
    def tearDownClass(cls):
        cls.tokenizer_directory.cleanup()

    def setUp(self):
        self.chunks = [portion(1, "\n".join([
            "[Taylor] I'll send the report to the committee.",
            "[Morgan] Staffing shortages remain unresolved.",
            "[Chair] " + MODEL_TEXT,
        ]))]
        self.records = ed.source_records(self.chunks)
        self.source = self.chunks[0]["text"]
        self.commitments = commitment_evidence(self.chunks)
        self.raw = {"items": [proposal("A1", "Circulate the report to the committee."),
                              proposal("A2", "Recruit additional staff.", "1:L2", "Morgan")]}

    def assess(self, raw=None):
        return ed.audit_register(self.raw if raw is None else raw, self.records, self.commitments, self.source)

    def triage(self, raw=None):
        return ed.triage_register(self.assess(raw), self.records, self.commitments, self.source)

    def run_responses(self, directory, raw=None, notes=None, *, chunks=None, commitments=None, combined="saved map summaries"):
        raw = self.raw if raw is None else raw
        notes = notes_response() if notes is None else notes
        replies = [json.dumps(raw), json.dumps(notes), "# Detailed Minutes\n\nModel detailed response retained."]
        calls = []
        def generate(prompt, **kwargs):
            calls.append((prompt, kwargs))
            if len(calls) > len(replies):
                raise AssertionError("an additional reduction was attempted")
            return replies[len(calls) - 1]
        budget = ed.RequestBudget(self.tokenizer_path, 196608, "Synthetic offline editorial system")
        with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")), \
             patch("ollama_session_summary.call_ollama", side_effect=AssertionError("inference forbidden")), \
             patch.object(ed, "render_notes", wraps=ed.render_notes) as renderer, \
             contextlib.redirect_stdout(io.StringIO()):
            status = ed.run(directory, self.chunks if chunks is None else chunks, combined, combined, "",
                            self.commitments if commitments is None else commitments, [], {}, [], [],
                            ROOT / "prompts/meeting", generate, False, budget=budget)
        return status, calls, renderer, replies, budget

    def assert_held(self, directory):
        report = json.loads((directory / ed.REVIEW).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "review_hold")
        for action in (lambda: publication_payload(directory),
                       lambda: export_documents(directory, directory / "export"),
                       lambda: export_archive(directory, directory / "export.zip")):
            with self.assertRaisesRegex(ValueError, "hold"):
                action()
        self.assertFalse((directory / "export").exists())
        self.assertFalse((directory / "export.zip").exists())

    def test_projection_keeps_full_original_audit_and_only_unapproved_candidates(self):
        assessment = self.assess()
        before = ed.digest(assessment)
        original = ed.digest(self.raw)
        register = ed.triage_register(assessment, self.records, self.commitments, self.source)
        self.assertEqual(ed.digest(assessment), before)
        self.assertEqual(ed.digest(self.raw), original)
        self.assertEqual(ed.digest(register["original_assessment"]), before)
        self.assertEqual(assessment["hard_block_count"], 1)
        self.assertEqual({i["id"] for i in register["items"]}, {"A1"})
        for item in register["items"]:
            self.assertRegex(item["status"].lower(), "proposed|candidate")
            self.assertIn("review", item["status"].lower())
        compact = ed.model_register(register)
        self.assertEqual(compact["items"], [])
        self.assertNotIn("original_assessment", compact)
        self.assertNotIn(self.raw["items"][1]["task"], ed.undertaking_table(register))

    def test_even_exact_lexical_candidates_are_not_approved(self):
        raw = {"items": [proposal("A1", "Send the report to the committee.")]}
        register = self.triage(raw)
        self.assertEqual(len(register["items"]), 1)
        self.assertRegex(register["items"][0]["status"].lower(), "proposed|candidate")
        self.assertIn("review", register["items"][0]["status"].lower())
        self.assertNotEqual(register["items"][0]["status"], ed.STATUS["undertaking"])

    def test_duplicate_malformed_and_unsupplied_items_cannot_be_reselected_by_id(self):
        raw = {"items": [self.raw["items"][0], copy.deepcopy(self.raw["items"][0]), {},
                         proposal("A3", "Send the report to the committee.")]}
        assessment = ed.audit_register(raw, self.records, self.commitments, self.source, supplied={"1:L3"})
        self.assertEqual(len(assessment["outcomes"]), 4)
        register = ed.triage_register(assessment, self.records, self.commitments, self.source)
        self.assertEqual(register["items"], [])
        self.assertEqual(ed.digest(register["original_assessment"]["proposed_register"]), ed.digest(raw))
        # Without the supplied-ID restriction, the first duplicate still cannot
        # become a candidate while its conflicting duplicate is kept privately.
        assessment = self.assess({"items": raw["items"][:2]})
        register = ed.triage_register(assessment, self.records, self.commitments, self.source)
        self.assertEqual(register["items"], [])

    def test_partial_hardblock_continues_three_calls_and_renders_model_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            status, calls, renderer, replies, budget = self.run_responses(directory)
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual([row["stage"] for row in budget.measurements], ["register", "notes", "detailed"])
            self.assertEqual([kwargs["structured"] for _, kwargs in calls], [True, True, False])
            self.assertEqual([kwargs["num_predict"] for _, kwargs in calls], [16384, 8192, 16384])
            renderer.assert_called_once()
            self.assertEqual(renderer.call_args.args[0]["issues"][0]["paragraphs"][0]["text"], MODEL_TEXT)
            self.assertIn(MODEL_TEXT, (directory / "meeting-notes-draft.md").read_text(encoding="utf-8"))
            self.assertIn("Model detailed response retained.", (directory / "minutes-draft.md").read_text(encoding="utf-8"))
            payload = json.loads(calls[1][0].split("Untrusted input JSON:\n", 1)[1])
            self.assertEqual(payload["register"]["items"], [])
            self.assertIn("1:L2", {r["id"] for r in payload["source_excerpts"]})
            self.assertEqual(payload["summaries"], "saved map summaries")
            response = json.loads((directory / ed.RESPONSE).read_text(encoding="utf-8"))
            self.assertEqual(response["responses"]["register"], replies[0])
            self.assertEqual(response["responses"]["notes"], replies[1])
            self.assertEqual(response["responses"]["detailed"], replies[2])
            self.assertEqual(response["register_assessment"]["sources"], self.records)
            self.assertEqual(response["register_assessment"]["proposed_register"], self.raw)
            table = (directory / "action-items.md").read_text(encoding="utf-8")
            self.assertNotIn("Owner awaiting confirmation", table)
            self.assertNotIn(self.raw["items"][0]["task"], table)
            self.assertNotIn("Taylor", table)
            checklist = (directory / ed.CHECKLIST).read_text(encoding="utf-8")
            self.assertIn("A2", checklist)
            self.assertIn("unsupported_undertaking", checklist)
            self.assert_held(directory)

    def test_malformed_individual_proposal_is_retained_while_notes_continue(self):
        for item in ({}, {"id": []}, {"id": {}}):
            with self.subTest(item=item), tempfile.TemporaryDirectory() as tmp:
                raw = {"items": [item]}
                directory = Path(tmp)
                status, calls, renderer, _, _ = self.run_responses(directory, raw)
                self.assertEqual(status, 0)
                self.assertEqual(len(calls), 3)
                renderer.assert_called_once()
                saved = json.loads((directory / ed.RESPONSE).read_text(encoding="utf-8"))
                self.assertEqual(saved["register_assessment"]["proposed_register"], raw)
                self.assertEqual(saved["register_assessment"]["outcomes"][0]["failure_category"], "invalid_register_item")
                self.assertEqual(saved["register_assessment"]["hard_block_count"], 1)
                self.assertIn(MODEL_TEXT, (directory / "meeting-notes-draft.md").read_text(encoding="utf-8"))
                self.assert_held(directory)

    def test_all_actions_hardblocked_still_prepares_issue_based_held_notes(self):
        raw = {"items": [self.raw["items"][1]]}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            status, calls, _, _, _ = self.run_responses(directory, raw)
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 3)
            register = json.loads((directory / ed.REGISTER).read_text(encoding="utf-8"))
            self.assertEqual(register["items"], [])
            self.assertEqual(register["original_assessment"]["hard_block_count"], 1)
            document = (directory / "meeting-notes-draft.md").read_text(encoding="utf-8")
            self.assertIn(MODEL_TEXT, document)
            self.assertNotIn(raw["items"][0]["task"], (directory / "action-items.md").read_text(encoding="utf-8"))
            self.assert_held(directory)

    def test_unsafe_display_conflicts_and_sensitive_source_remain_private(self):
        cases = [
            ("[Taylor] I'll send the report to the committee.", proposal("A1", "Send the report to the committee.", concerns=["source_conflict"])),
            ("[Taylor] I'll send the report to the committee after the medical history review.", proposal("A1", "Send the report to the committee.")),
            ("[Taylor] I'll review my paystubs.", proposal("A1", "Review my paystubs.")),
            ("[Taylor] I'll review the pay-stubs.", proposal("A1", "Review the pay-stubs.")),
            ("[Taylor] I'll send you the report.", proposal("A1", "Send the report to SPEAKER_02.")),
            ("[Taylor] I'll send the report to the committee.", proposal("A1", "Send the report to the committee.\nExtra paragraph.")),
        ]
        for source, item in cases:
            with self.subTest(case=cases.index((source, item))):
                chunks = [portion(1, source)]
                records = ed.source_records(chunks)
                commitments = commitment_evidence(chunks)
                assessment = ed.audit_register({"items": [item]}, records, commitments, source)
                register = ed.triage_register(assessment, records, commitments, source)
                self.assertNotIn(item["task"], ed.undertaking_table(register))
                self.assertEqual(ed.digest(register["original_assessment"]), ed.digest(assessment))

    def test_notes_lexical_uncertainty_and_conflicts_remain_explicit_review_blockers(self):
        notes = notes_response(text="The warehouse renovation received a proposed budget.", concerns=[
            {"category": "source_conflict", "text": "The renovation claim requires checking against the committee discussion.", "source_ids": ["1:L3"]}])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            status, calls, _, _, _ = self.run_responses(directory, notes=notes)
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 3)
            report = json.loads((directory / ed.REVIEW).read_text(encoding="utf-8"))
            codes = {f["code"] for f in report["findings"]}
            self.assertTrue({"source_conflict", "notes_support_review"} <= codes)
            self.assert_held(directory)

    def test_invalid_and_sensitive_actual_notes_stop_before_render_and_retain_diagnostics(self):
        for notes, category in ((notes_response(ref="fabricated"), "invalid_source_references"),
                                (notes_response(text="Medical history discussed."), "confidential_notes_content"),
                                ({"highlights": []}, "invalid_notes_schema")):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                status, calls, renderer, replies, _ = self.run_responses(directory, notes=notes)
                self.assertEqual(status, 1)
                self.assertEqual(len(calls), 2)
                renderer.assert_not_called()
                self.assertFalse(any((directory / name).exists() for name in MEMBER_FILES))
                response = json.loads((directory / ed.RESPONSE).read_text(encoding="utf-8"))
                self.assertEqual(response["responses"]["notes"], replies[1])
                self.assertEqual(response["register_assessment"]["sources"], self.records)
                self.assertEqual(response["register_assessment"]["proposed_register"], self.raw)
                report = json.loads((directory / ed.REVIEW).read_text(encoding="utf-8"))
                self.assertEqual(report["failure_category"], category)
                self.assert_held(directory)

    def test_editing_review_status_cannot_enable_publication_or_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            status, _, _, _, _ = self.run_responses(directory)
            self.assertEqual(status, 0)
            (directory / ed.REVIEW).write_text('{"status":"approved"}', encoding="utf-8")
            for action in (lambda: publication_payload(directory),
                           lambda: export_documents(directory, directory / "export"),
                           lambda: export_archive(directory, directory / "export.zip")):
                with self.assertRaisesRegex(ValueError, "hold"):
                    action()

    @unittest.skipUnless(all((LIVE / name).is_file() for name in LIVE_FILES), "saved private October artifacts required")
    def test_saved_register_hardblocks_excluded_and_notes_stage_executes_offline(self):
        hashes = {name: hashlib.sha256((LIVE / name).read_bytes()).hexdigest() for name in LIVE_FILES}
        response = json.loads((LIVE / ed.RESPONSE).read_text(encoding="utf-8"))
        chunks = [json.loads(line) for line in (LIVE / "meeting_sections.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        maps = [json.loads(line) for line in (LIVE / "chunk_summaries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        json.loads((LIVE / ed.BUDGETS).read_text(encoding="utf-8"))
        self.assertEqual(len(maps), 27)
        records = ed.source_records(chunks)
        self.assertEqual(ed.digest(records), response["source_hash"])
        raw = json.loads(response["responses"]["register"])
        self.assertEqual(len(raw["items"]), 21)
        source = "\n".join(c["text"] for c in chunks if c["meeting_section"] in {BUSINESS, ADJOURNMENT})
        commitments = commitment_evidence(chunks)
        assessment = ed.audit_register(raw, records, commitments, source)
        blocked = {r["item_id"] for r in assessment["outcomes"] if r["outcome"] == "hard_block"}
        self.assertEqual(blocked, {"A2", "A10", "A11", "A12", "A13", "A14", "A18", "A19"})
        register = ed.triage_register(assessment, records, commitments, source)
        self.assertFalse(blocked & {i["id"] for i in ed.model_register(register)["items"]})
        self.assertEqual(ed.digest(register["original_assessment"]["proposed_register"]), ed.digest(raw))
        self.assertEqual(ed.digest(register["original_assessment"]["sources"]), ed.digest(records))
        table = ed.undertaking_table(register)
        for item in raw["items"]:
            if item["id"] in blocked:
                self.assertTrue(item["task"] not in table, "a hard-blocked October action entered the table")
        ref = next(key for key, row in records.items() if row["section"] == BUSINESS and len(row["text"]) <= ed.LOCAL_CHARS)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            status, calls, renderer, _, _ = self.run_responses(directory, raw, notes_response(ref), chunks=chunks,
                                                             commitments=commitments, combined=json.dumps(maps))
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 3)
            renderer.assert_called_once()
            self.assertEqual(renderer.call_args.args[0]["issues"][0]["paragraphs"][0]["text"], MODEL_TEXT)
            saved = json.loads((directory / ed.RESPONSE).read_text(encoding="utf-8"))
            self.assertEqual(len(saved["register_assessment"]["outcomes"]), 21)
            self.assertEqual(saved["register_assessment"]["hard_block_count"], 8)
            self.assertEqual(ed.digest(saved["register_assessment"]["proposed_register"]), ed.digest(raw))
            self.assertEqual(ed.digest(saved["register_assessment"]["sources"]), ed.digest(records))
            self.assert_held(directory)
        self.assertEqual({name: hashlib.sha256((LIVE / name).read_bytes()).hexdigest() for name in LIVE_FILES}, hashes)


if __name__ == "__main__":
    unittest.main()
