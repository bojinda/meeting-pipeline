"""Native, synthetic structural checks; no October artifacts or model calls."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed
from meeting_postprocess.commitments import commitment_evidence
from meeting_postprocess.publication import publication_payload, export_archive, export_documents
from meeting_postprocess.sections import PRE_MEETING
from test_editorial_notes import portion


class EditorialStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer, models, pre_tokenizers
        cls.tokenizer_directory = tempfile.TemporaryDirectory()
        cls.tokenizer = Path(cls.tokenizer_directory.name) / "tokenizer.json"
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(cls.tokenizer))

    @classmethod
    def tearDownClass(cls):
        cls.tokenizer_directory.cleanup()

    def setUp(self):
        self.chunks = [portion(1, "[Chair] Earlier staffing plans remained open.\n[Chair] Earlier committee reports were discussed.", ed.RECAP),
                       portion(2, "\n".join(["[Taylor] I'll send the draft report to the committee.",
                           "[Chair] Staffing remained unresolved.", "[Taylor] A medical history requires PRIVATE CONCERN review.",
                           *["[Chair] The committee discussion continued." for _ in range(7)]])),
                       portion(3, "[Chair] A motion was discussed.\n[Chair] Distribution remained unconfirmed.", ed.ADJOURNMENT)]
        self.records = ed.source_records(self.chunks)
        self.source = "\n".join(c["text"] for c in self.chunks if c["meeting_section"] != ed.RECAP)
        self.commitments = commitment_evidence(self.chunks)
        self.raw = {"items": [
            {"id": "A1", "category": "undertaking", "task": "Send the draft report to the committee.",
             "owners": ["Taylor"], "source_ids": ["2:L1"], "member_facing": True, "concerns": []},
            {"id": "A2", "category": "undertaking", "task": "Recruit additional staff.",
             "owners": ["Chair"], "source_ids": ["2:L2"], "member_facing": True, "concerns": []},
            {"id": "A3", "category": "business", "task": "Distribution remained unconfirmed.",
             "owners": [], "source_ids": ["3:L2"], "member_facing": False, "concerns": []}]}
        self.notes = {"highlights": [], "previous_context": [], "motions": [], "unresolved": [], "concerns": [],
                      "issues": [{"heading": "Staffing", "paragraphs": [{"text": "Staffing remained unresolved.", "source_ids": ["2:L2"]}]}]}

    def register(self):
        assessment = ed.audit_register(self.raw, self.records, self.commitments, self.source)
        return ed.triage_register(assessment, self.records, self.commitments, self.source)

    def run_responses(self, directory, notes):
        replies = [json.dumps(self.raw), json.dumps(notes), "# Detailed Minutes\n\nStaffing remained unresolved."]
        calls, output = [], io.StringIO()
        def generate(prompt, *, response_callback=None, **kwargs):
            calls.append(prompt)
            if response_callback:
                response_callback({"done": True, "done_reason": "stop", "response_char_count": len(replies[len(calls)-1])})
            return replies[len(calls)-1]
        budget = ed.RequestBudget(self.tokenizer, 196608, "Synthetic offline system")
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), contextlib.redirect_stdout(output):
            result = ed.run(directory, self.chunks, "Synthetic map summaries", "Synthetic current summaries", "", self.commitments,
                            [], {}, [], [], ROOT / "prompts/meeting", generate, False, budget=budget)
        return result, calls, replies, output.getvalue()

    def assert_held(self, directory):
        self.assertEqual(json.loads((directory / ed.REVIEW).read_text())["status"], "review_hold")
        for export in (lambda: publication_payload(directory), lambda: export_documents(directory, directory / "export"),
                       lambda: export_archive(directory, directory / "export.zip")):
            with self.assertRaisesRegex(ValueError, "hold"):
                export()

    def test_unapproved_named_and_unverified_owners_never_enter_member_assignments(self):
        register = self.register()
        before = copy.deepcopy(register)
        checked = ed.validate_notes(self.notes, self.records, self.source)
        for candidate_register in (register, ed.validate_register({"items": self.raw["items"][:1]}, self.records, self.commitments, self.source)):
            for owners in (["Taylor"], ["SPEAKER_01"], []):
                candidate = copy.deepcopy(candidate_register)
                candidate["items"][0]["owners"] = owners
                candidate["items"][0]["member_facing"] = True
                candidate["items"][0]["operator_approved"] = True  # Extra untrusted data is not an approval workflow.
                text = ed.render_notes(checked, candidate)
                self.assertNotIn(self.raw["items"][0]["task"], text)
                self.assertNotIn("Owner awaiting confirmation", text)
                self.assertNotIn("| Undertaking |", text)
        self.assertEqual(register, before)
        self.assertEqual(len(register["original_assessment"]["proposed_register"]["items"]), 3)
        self.assertEqual(len(register["exclusions"]), 2)

    def test_model_projection_withholds_undertakings_without_hiding_source_issues(self):
        register = self.register()
        before = copy.deepcopy(register)
        for prompt in (ed.notes_prompt(ROOT / "prompts/meeting", self.records, "Staffing remains unresolved.", register),
                       ed.detailed_prompt(ROOT / "prompts/meeting", self.records, "Staffing remains unresolved.", register)):
            payload = json.loads(prompt.split("Untrusted input JSON:\n", 1)[1])
            self.assertFalse(any(item["category"] == "undertaking" for item in payload["register"]["items"]))
            self.assertTrue({"2:L1", "2:L2", "3:L2"} <= {row["id"] for row in payload["source_excerpts"]})
        self.assertEqual(register, before)

    def test_strict_private_concerns_preserve_sensitive_detail_and_source_links(self):
        notes = copy.deepcopy(self.notes)
        private_text = "A medical history requires PRIVATE CONCERN review before distribution."
        notes["concerns"] = [{"category": "confidentiality", "text": private_text, "source_ids": ["2:L3"]}]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result, calls, replies, log = self.run_responses(directory, notes)
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 3)
            self.assertNotIn(private_text, log)
            for name in ("meeting-notes-draft.md", "summary.md", "action-items.md", "minutes-draft.md"):
                text = (directory / name).read_text()
                self.assertNotIn(private_text, text)
                self.assertNotIn(self.raw["items"][0]["task"], text)
            evidence = json.loads((directory / ed.NOTES_EVIDENCE).read_text())
            self.assertEqual(evidence["notes"]["concerns"], notes["concerns"])
            review = json.loads((directory / ed.REVIEW).read_text())
            flag = next(f for f in review["findings"] if f.get("concern_text") == private_text)
            self.assertEqual(flag["code"], "confidentiality")
            self.assertEqual(flag["source_ids"], ["2:L3"])
            self.assertEqual(flag["evidence_pointer"], f"{ed.NOTES_EVIDENCE}#/notes/concerns/0")
            group = next(g for g in review["review_groups"] if g["id"] == "confidentiality")
            self.assertTrue(any(e.get("concern_text") == private_text for e in group["entries"]))
            checklist = (directory / ed.CHECKLIST).read_text()
            self.assertNotIn(private_text, checklist)
            self.assertIn("Sources: 2:L3.", checklist)
            self.assertIn(flag["evidence_pointer"], checklist)
            raw = json.loads((directory / ed.RESPONSE).read_text())
            self.assertEqual(raw["responses"]["notes"], replies[1])
            self.assertEqual(raw["register_assessment"]["proposed_register"], self.raw)
            if os.name != "nt":
                for name in (ed.RESPONSE, ed.REVIEW, ed.NOTES_EVIDENCE, ed.CHECKLIST):
                    self.assertEqual((directory / name).stat().st_mode & 0o777, 0o600)
            self.assert_held(directory)

    def test_concern_markup_stays_in_json_and_never_enters_markdown(self):
        fragments = ["CONFIDENTIAL DETAIL", "[review](https://private.invalid/review)",
                     "![case](https://private.invalid/image.png)",
                     '<a href="https://private.invalid/detail">private HTML</a>',
                     '<img src="https://private.invalid/pixel.png">']
        concern = {"category": "confidentiality", "text": "\n".join(fragments), "source_ids": ["2:L3"]}
        notes = copy.deepcopy(self.notes)
        notes["concerns"] = [concern]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result, _, replies, log = self.run_responses(directory, notes)
            self.assertEqual(result, 0)
            response = json.loads((directory / ed.RESPONSE).read_text())
            self.assertEqual(response["responses"]["notes"], replies[1])
            self.assertEqual(json.loads(response["responses"]["notes"])["concerns"], [concern])
            evidence = json.loads((directory / ed.NOTES_EVIDENCE).read_text())
            self.assertEqual(evidence["notes"]["concerns"], [concern])
            review = json.loads((directory / ed.REVIEW).read_text())
            finding = next(f for f in review["findings"] if "concern_text" in f)
            self.assertEqual(finding["concern_text"], concern["text"])
            self.assertEqual(finding["source_ids"], concern["source_ids"])
            group = next(g for g in review["review_groups"] if g["id"] == "confidentiality")
            entry = next(e for e in group["entries"] if "concern_text" in e)
            self.assertEqual(entry["concern_text"], concern["text"])
            checklist = (directory / ed.CHECKLIST).read_text()
            self.assertIn("- [ ] confidentiality", checklist)
            self.assertIn("Sources: 2:L3.", checklist)
            self.assertIn(finding["evidence_pointer"], checklist)
            for content in [checklist, log, *[(directory / name).read_text() for name in
                    ("meeting-notes-draft.md", "summary.md", "action-items.md", "minutes-draft.md")]]:
                for fragment in fragments:
                    self.assertNotIn(fragment, content)
            self.assert_held(directory)

    def test_each_allowed_concern_category_has_explicit_nonempty_contract(self):
        for category in ed.CONCERNS:
            notes = copy.deepcopy(self.notes)
            notes["concerns"] = [{"category": category, "text": "The earlier plan and current staffing need comparison.", "source_ids": ["1:L1", "2:L2"]}]
            clean = ed.validate_notes(notes, self.records, self.source)
            self.assertEqual(clean["concerns"], notes["concerns"])

    def test_missing_inferred_unknown_or_malformed_concern_contract_fails(self):
        valid = {"category": "source_conflict", "text": "The reported plan needs review.", "source_ids": ["2:L2"]}
        malformed = ["source_conflict", {"text": "Historical shape without a category.", "source_ids": ["2:L2"]},
                     {**valid, "category": "invented"}, {**valid, "category": []}, {**valid, "category": None},
                     {**valid, "text": " "}, {**valid, "text": 3}, {**valid, "text": "x" * 6001},
                     {**valid, "extra": True}, {"category": "source_conflict", "text": "Missing references."}]
        for concern in malformed:
            with self.subTest(concern=concern):
                notes = copy.deepcopy(self.notes)
                notes["concerns"] = [concern]
                with self.assertRaisesRegex(ed.EditorialFailure, "invalid_notes_concern"):
                    ed.validate_notes(notes, self.records, self.source)

    def test_concern_invalid_ids_duplicates_and_nonmeeting_sections_fail(self):
        records = copy.deepcopy(self.records)
        records["outside:L1"] = {**records["2:L2"], "id": "outside:L1", "section": PRE_MEETING}
        for refs in ([], ["missing"], ["2:L2", "missing"], ["2:L2", "2:L2"], ["outside:L1"], "2:L2", [True]):
            notes = copy.deepcopy(self.notes)
            notes["concerns"] = [{"category": "source_conflict", "text": "Staffing needs review.", "source_ids": refs}]
            changes = []
            with self.assertRaisesRegex(ed.EditorialFailure, "invalid_source_references"):
                ed.validate_notes(notes, records, self.source, reference_changes=changes)
            self.assertEqual(changes, [])

    def test_notes_references_normalize_in_source_order_without_mutating_response(self):
        notes = copy.deepcopy(self.notes)
        for key in ("highlights", "motions", "unresolved"):
            notes[key] = [{"text": "Distribution and staffing remained open.", "source_ids": ["3:L2", "2:L2"]}]
        notes["previous_context"] = [{"text": "Earlier plans and committee reports were discussed.", "source_ids": ["1:L2", "1:L1"]}]
        notes["issues"][0]["paragraphs"][0]["source_ids"] = ["2:L10", "2:L2"]
        notes["concerns"] = [{"category": "source_conflict", "text": "Compare the earlier plan and current staffing.", "source_ids": ["2:L2", "1:L1"]}]
        before, changes = copy.deepcopy(notes), []
        clean = ed.validate_notes(notes, self.records, self.source, reference_changes=changes)
        self.assertEqual(notes, before)
        self.assertEqual(len(changes), 6)
        self.assertEqual(clean["issues"][0]["paragraphs"][0]["source_ids"], ["2:L2", "2:L10"])
        self.assertEqual(clean["previous_context"][0]["source_ids"], ["1:L1", "1:L2"])
        self.assertEqual(clean["concerns"][0]["source_ids"], ["1:L1", "2:L2"])
        for change in changes:
            self.assertEqual(change["code"], "notes_reference_order_normalized")
            self.assertNotEqual(change["original_source_ids"], change["canonical_source_ids"])
        unchanged = []
        ed.validate_notes(self.notes, self.records, self.source, reference_changes=unchanged)
        self.assertEqual(unchanged, [])

    def test_reordering_does_not_repair_invalid_or_nonmeeting_references(self):
        records = copy.deepcopy(self.records)
        records["outside:L1"] = {**records["2:L2"], "id": "outside:L1", "section": PRE_MEETING}
        for historical, refs in ((False, ["3:L2", "missing"]), (False, ["3:L2", "2:L2", "2:L2"]),
                                 (False, ["outside:L1", "2:L2"]), (True, ["missing"])):
            notes = copy.deepcopy(self.notes)
            block = {"text": "A report was discussed.", "source_ids": refs}
            if historical:
                notes["previous_context"] = [block]
            else:
                notes["issues"][0]["paragraphs"] = [block]
            changes = []
            with self.assertRaisesRegex(ed.EditorialFailure, "invalid_source_references"):
                ed.validate_notes(notes, records, self.source, reference_changes=changes)
            self.assertEqual(changes, [])
        # Register validation remains strict: normalization applies only to notes.
        with self.assertRaisesRegex(ed.EditorialFailure, "unordered_source_references"):
            ed._refs(["3:L2", "2:L2"], self.records)

    def test_section_mismatches_save_held_draft_without_relabelling_evidence(self):
        notes = copy.deepcopy(self.notes)
        notes["previous_context"] = [{"text": "The report was discussed earlier.", "source_ids": ["2:L1"]}]
        notes["issues"][0]["paragraphs"][0]["source_ids"] = ["2:L2", "1:L1"]
        original_notes, original_records = copy.deepcopy(notes), copy.deepcopy(self.records)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result, _, replies, _ = self.run_responses(directory, notes)
            self.assertEqual(result, 0)
            review = json.loads((directory / ed.REVIEW).read_text())
            flags = [f for f in review["findings"] if f["code"] == "notes_source_section_review"]
            self.assertEqual([f["mismatched_source_ids"] for f in flags], [["2:L1"], ["1:L1"]])
            self.assertTrue(all(f["validation_finding"] == "invalid_source_references" for f in flags))
            self.assertEqual(flags[0]["recorded_sections"], {"2:L1": ed.BUSINESS})
            self.assertEqual(flags[0]["expected_sections"], [ed.RECAP])
            evidence = json.loads((directory / ed.NOTES_EVIDENCE).read_text())
            self.assertEqual(evidence["sources"], original_records)
            self.assertEqual(evidence["notes"]["issues"][0]["paragraphs"][0]["source_ids"], ["1:L1", "2:L2"])
            self.assertEqual(json.loads((directory / ed.RESPONSE).read_text())["responses"]["notes"], replies[1])
            doc = (directory / "meeting-notes-draft.md").read_text()
            self.assertIn("DRAFT — REVIEW REQUIRED", doc)
            self.assertIn("Source citations require review", doc)
            self.assertIn(notes["previous_context"][0]["text"], doc)
            self.assertIn("notes_source_section_review", (directory / ed.CHECKLIST).read_text())
            self.assert_held(directory)
        self.assertEqual(notes, original_notes)
        self.assertEqual(self.records, original_records)

    def test_possible_identifying_cases_require_private_confidentiality_review(self):
        for text in ("Advice was given about the reporting process.", "Morgan received discipline."):
            notes = copy.deepcopy(self.notes)
            notes["issues"][0]["paragraphs"][0]["text"] = text
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                status, _, _, _ = self.run_responses(directory, notes)
                self.assertEqual(status, 0)
                review = json.loads((directory / ed.REVIEW).read_text())
                flags = [f for f in review["findings"] if f["code"] == "confidentiality"]
                self.assertEqual(flags[0]["notes_location"], "issues/0/paragraphs/0")
                self.assertEqual(flags[0]["source_ids"], ["2:L2"])
                self.assertIn(text, (directory / "meeting-notes-draft.md").read_text())
                self.assert_held(directory)

    def test_no_final_notes_answer_never_creates_a_document(self):
        for answer in ("", "   "):
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                replies = iter([json.dumps(self.raw), answer])
                budget = ed.RequestBudget(self.tokenizer, 196608, "Synthetic offline system")
                with contextlib.redirect_stdout(io.StringIO()):
                    status = ed.run(directory, self.chunks, "maps", "current", "", self.commitments, [], {}, [], [],
                                    ROOT / "prompts/meeting", lambda *a, **k: next(replies), False, budget=budget)
                self.assertEqual(status, 1)
                self.assertFalse((directory / "meeting-notes-draft.md").exists())
                self.assertEqual(json.loads((directory / ed.REVIEW).read_text())["failure_category"], "ollama_empty_final_answer")
                self.assert_held(directory)

    def test_production_preserves_raw_order_and_normalization_diagnostics(self):
        notes = copy.deepcopy(self.notes)
        notes["issues"][0]["paragraphs"][0]["source_ids"] = ["2:L10", "2:L2"]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            status, calls, replies, _ = self.run_responses(directory, notes)
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 3)
            raw = json.loads((directory / ed.RESPONSE).read_text())
            self.assertEqual(raw["responses"]["notes"], replies[1])
            diagnostics = json.loads((directory / ed.DIAGNOSTICS).read_text())
            self.assertEqual(len(diagnostics["requests"]), 3)
            self.assertEqual(diagnostics["notes_reference_changes"], [{"code": "notes_reference_order_normalized",
                "notes_location": "issues/0/paragraphs/0", "original_source_ids": ["2:L10", "2:L2"],
                "canonical_source_ids": ["2:L2", "2:L10"]}])
            checked = json.loads((directory / ed.NOTES_EVIDENCE).read_text())["notes"]
            self.assertEqual(checked["issues"][0]["paragraphs"][0]["source_ids"], ["2:L2", "2:L10"])
            self.assert_held(directory)

    def test_malformed_detailed_concerns_are_preserved_not_inferred_or_discarded(self):
        notes = copy.deepcopy(self.notes)
        notes["concerns"] = [{"text": f"Substantive private concern {i}.", "source_ids": ["2:L2"]} for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result, calls, replies, _ = self.run_responses(directory, notes)
            self.assertEqual(result, 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads((directory / ed.RESPONSE).read_text())["responses"]["notes"], replies[1])
            self.assertEqual(json.loads((directory / ed.REVIEW).read_text())["failure_category"], "invalid_notes_concern")
            self.assertFalse((directory / "meeting-notes-draft.md").exists())
            self.assert_held(directory)

    def test_ordering_changes_survive_later_validation_failure_without_a_draft(self):
        notes = copy.deepcopy(self.notes)
        notes["highlights"] = [{"text": "Staffing remains open.", "source_ids": ["3:L2", "2:L2"]}]
        notes["issues"][0]["paragraphs"][0]["source_ids"] = ["missing"]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result, calls, replies, _ = self.run_responses(directory, notes)
            self.assertEqual(result, 1)
            self.assertEqual(len(calls), 2)
            changes = json.loads((directory / ed.DIAGNOSTICS).read_text())["notes_reference_changes"]
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0]["notes_location"], "highlights/0")
            self.assertEqual(json.loads((directory / ed.RESPONSE).read_text())["responses"]["notes"], replies[1])
            self.assertFalse((directory / "meeting-notes-draft.md").exists())
            self.assert_held(directory)


if __name__ == "__main__":
    unittest.main()
