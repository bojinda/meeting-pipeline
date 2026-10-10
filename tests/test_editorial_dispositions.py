"""Offline selection cannot waive original or derived hard failures."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed
from meeting_postprocess import editorial_dispositions as disp
from meeting_postprocess.commitments import commitment_evidence
from meeting_postprocess.publication import publication_payload, export_documents, export_archive, strip_private_references


class DispositionTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"chunk_id": "1", "source_chunk_id": "1", "meeting_section": ed.BUSINESS,
                        "text": "[SPEAKER_01] If the office agrees, I'll send the report to the safety committee.\n"
                                "[SPEAKER_02] I am grieving the private case.\n"
                                "[SPEAKER_01] The safety report is still under discussion."}]
        self.records = ed.source_records(self.chunks)
        self.commitments = commitment_evidence(self.chunks)
        self.source = self.chunks[0]["text"]
        self.raw = {"items": [
            {"id": "A1", "category": "undertaking", "task": "If the office agrees, send the report to the safety committee.",
             "owners": ["SPEAKER_01"], "source_ids": ["1:L1"], "member_facing": True, "concerns": []},
            {"id": "A2", "category": "undertaking", "task": "File a new private grievance.",
             "owners": ["SPEAKER_02"], "source_ids": ["1:L2"], "member_facing": True, "concerns": []}]}
        a = ed.audit_register(self.raw, self.records, self.commitments, self.source)
        self.assertEqual(a["hard_block_count"], 1)
        self.decisions = {"version": 1, "scope": "held_draft_only", "source_hash": ed.digest(self.records), "commitments_hash": ed.digest(self.commitments),
                          "original_register_hash": ed.digest(self.raw), "original_outcomes_hash": ed.digest(a["outcomes"]),
                          "authorization": "Operator explicitly approved these two editorial dispositions for held drafting only.",
                          "rows": [
                              {"id": "A1", "treatment": "selected_undertaking", "classification": "conditional undertaking",
                               "reason": "Retain condition; identity pending.", "context_source_ids": ["1:L1"],
                               "candidate": {"task": self.raw["items"][0]["task"], "source_ids": ["1:L1"], "qualification": "Office agreement required; no completion confirmed."}},
                              {"id": "A2", "treatment": "private", "classification": "existing casework",
                               "reason": "Keep original failure and evidence privately.", "context_source_ids": ["1:L2"], "candidate": None}]}
        self.notes = {"highlights": [], "previous_context": [], "motions": [], "unresolved": [], "concerns": [],
                      "issues": [{"heading": "Safety report", "paragraphs": [{"text": "The safety report remains under discussion.", "source_ids": ["1:L3"]}]}]}

    def check(self, decisions=None, raw=None, records=None):
        return disp.reviewed_register(raw or self.raw, records or self.records, self.commitments, self.source, decisions or self.decisions)

    def test_private_hard_failure_retained_selected_derivative_checked(self):
        before = copy.deepcopy((self.raw, self.records, self.decisions))
        original, register, selections = self.check()
        self.assertEqual(original["outcomes"][1]["outcome"], "hard_block")
        self.assertIsNone(original["register"])
        self.assertEqual([i["id"] for i in register["items"]], ["A1"])
        self.assertEqual(selections[0]["candidate"]["owners"], ["SPEAKER_01"])
        self.assertEqual(before, (self.raw, self.records, self.decisions))

    def test_stale_source_original_and_outcome_hashes_rejected(self):
        for field in ("source_hash", "commitments_hash", "original_register_hash", "original_outcomes_hash"):
            with self.subTest(field=field):
                d = copy.deepcopy(self.decisions); d[field] = "changed"
                with self.assertRaisesRegex(ed.EditorialFailure, "stale_disposition_identity"): self.check(d)

    def test_only_explicit_held_scope_accepted(self):
        for key, value in (("scope", "approved_for_publication"), ("authorization", ""), ("version", True)):
            d = copy.deepcopy(self.decisions); d[key] = value
            with self.assertRaises(ed.EditorialFailure): self.check(d)

    def test_actual_source_change_invalidates_dispositions(self):
        r = copy.deepcopy(self.records); r["1:L1"]["body"] = "Changed source"
        with self.assertRaisesRegex(ed.EditorialFailure, "stale_disposition_identity"): self.check(records=r)

    def test_truncated_source_text_and_commitment_evidence_rejected(self):
        with self.assertRaisesRegex(ed.EditorialFailure, "disposition_source_text_mismatch"):
            disp.reviewed_register(self.raw, self.records, self.commitments, self.source.splitlines()[0], self.decisions)
        with self.assertRaisesRegex(ed.EditorialFailure, "stale_disposition_identity"):
            disp.reviewed_register(self.raw, self.records, [], self.source, self.decisions)

    def test_all_originals_need_explicit_disposition_no_unknown_or_duplicate(self):
        for rows in (self.decisions["rows"][:1], self.decisions["rows"] * 2, list(reversed(self.decisions["rows"]))):
            d = copy.deepcopy(self.decisions); d["rows"] = rows
            with self.assertRaisesRegex(ed.EditorialFailure, "incomplete_disposition_coverage"): self.check(d)

    def test_hard_block_cannot_be_selected_even_with_rewritten_candidate(self):
        d = copy.deepcopy(self.decisions)
        d["rows"][1].update(treatment="selected_undertaking", candidate=copy.deepcopy(d["rows"][0]["candidate"]))
        with self.assertRaisesRegex(ed.EditorialFailure, "selected_original_hard_block"): self.check(d)

    def test_lost_condition_changed_recipient_and_confidentiality_still_block(self):
        for task in ("Send the report to the safety committee.",
                     "If the office agrees, send the report to the finance committee.",
                     "If the office agrees, send the medical history report to the safety committee."):
            with self.subTest(task=task):
                d = copy.deepcopy(self.decisions); d["rows"][0]["candidate"]["task"] = task
                with self.assertRaises(ed.EditorialFailure): self.check(d)

    def test_fabricated_or_dropped_references_rejected(self):
        for refs in (["99:L1"], ["1:L3"], []):
            d = copy.deepcopy(self.decisions); d["rows"][0]["candidate"]["source_ids"] = refs
            with self.assertRaises(ed.EditorialFailure): self.check(d)

    def test_owner_alias_and_approval_override_fields_not_supported(self):
        for field, value in (("owners", ["Morgan"]), ("approved", True), ("skip_validation", True)):
            d = copy.deepcopy(self.decisions); d["rows"][0]["candidate"][field] = value
            with self.assertRaises(ed.EditorialFailure): self.check(d)

    def test_held_outputs_and_all_export_paths_remain_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "held"
            result = disp.prepare_held_draft(out, self.raw, self.records, self.commitments, self.source, self.decisions, self.notes)
            doc = (out / "meeting-notes-draft.md").read_text()
            self.assertNotIn("SPEAKER_", doc)
            self.assertNotIn("private grievance", doc)
            self.assertNotIn("Owner awaiting confirmation", doc)
            self.assertNotIn(self.raw["items"][0]["task"], doc)
            register = json.loads((out / ed.REGISTER).read_text())
            self.assertEqual(register["selected_records"][0]["qualification"], "Office agreement required; no completion confirmed.")
            self.assertFalse(result["publication_authorized"])
            saved = json.loads((out / ed.RESPONSE).read_text())["register_assessment"]
            self.assertEqual(saved["proposed_register"], self.raw)
            self.assertEqual(saved["sources"], self.records)
            for action in (lambda: publication_payload(out), lambda: export_documents(out, Path(tmp)/"export"), lambda: export_archive(out, Path(tmp)/"export.zip")):
                with self.assertRaises(ValueError): action()
            with self.assertRaises(FileExistsError):
                disp.prepare_held_draft(out, self.raw, self.records, self.commitments, self.source, self.decisions, self.notes)

    def test_invalid_derivative_leaves_private_audit_and_no_member_document(self):
        d = copy.deepcopy(self.decisions); d["rows"][0]["candidate"]["task"] = "Send the report."
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"failed"
            with self.assertRaises(ed.EditorialFailure):
                disp.prepare_held_draft(out, self.raw, self.records, self.commitments, self.source, d, self.notes)
            self.assertTrue((out/ed.RESPONSE).is_file())
            self.assertFalse((out/"meeting-notes-draft.md").exists())
            self.assertEqual(json.loads((out/ed.REVIEW).read_text())["phase"], "failed")

    def test_bad_notes_reference_blocks_member_files(self):
        n = copy.deepcopy(self.notes); n["issues"][0]["paragraphs"][0]["source_ids"] = ["99:L1"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"failed"
            with self.assertRaises(ed.EditorialFailure):
                disp.prepare_held_draft(out, self.raw, self.records, self.commitments, self.source, self.decisions, n)
            self.assertFalse((out/"meeting-notes-draft.md").exists())

    def test_source_linked_concerns_and_reference_changes_stay_private(self):
        notes = copy.deepcopy(self.notes)
        notes["concerns"] = [{"category": "confidentiality", "text": "Medical history requires private operator review.",
                              "source_ids": ["1:L3", "1:L1"]}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "held"
            result = disp.prepare_held_draft(out, self.raw, self.records, self.commitments, self.source, self.decisions, notes)
            self.assertEqual(json.loads((out / ed.RESPONSE).read_text())["notes_proposal"], notes)
            changes = json.loads((out / ed.DIAGNOSTICS).read_text())["notes_reference_changes"]
            self.assertEqual(changes[0]["original_source_ids"], ["1:L3", "1:L1"])
            self.assertEqual(changes[0]["canonical_source_ids"], ["1:L1", "1:L3"])
            concern = next(f for f in result["findings"] if "concern_text" in f)
            self.assertEqual(concern["source_ids"], ["1:L1", "1:L3"])
            self.assertNotIn(concern["concern_text"], (out / "meeting-notes-draft.md").read_text())

    def test_saved_selected_view_keeps_all_original_failures(self):
        live = ROOT / "ignore/october-editorial-live.wnfcYj/editorial-output"
        held = ROOT / "ignore/october-held-dispositions/review-ready"
        if not (live / ed.RESPONSE).exists() or not (held / disp.DISPOSITIONS).exists():
            self.skipTest("Optional protected October fixture unavailable")
        def read(p): return json.loads(p.read_text(encoding="utf-8"))
        raw = json.loads(read(live / ed.RESPONSE)["responses"]["register"])
        chunks = [json.loads(x) for x in (live/"meeting_sections.jsonl").read_text(encoding="utf-8").splitlines()]
        r = ed.source_records(chunks)
        s = "\n".join(c["text"] for c in chunks if c["meeting_section"] in {ed.BUSINESS, ed.ADJOURNMENT})
        original, selected, _ = disp.reviewed_register(raw, r, commitment_evidence(chunks), s, read(held/disp.DISPOSITIONS))
        self.assertEqual(original["hard_block_count"], 8)
        self.assertEqual(len(original["outcomes"]), 21)
        self.assertEqual(original["outcomes"], read(held/ed.RESPONSE)["register_assessment"]["outcomes"])
        self.assertEqual([i["id"] for i in selected["items"]], ["A1", "A3", "A4", "A16", "A17", "A20"])
        text = (held/"meeting-notes-draft.md").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"SPEAKER_|\bWayne\b|\bLee\b|\bJake\b")
        self.assertEqual(text.count("| Owner awaiting confirmation |"), 6)
        with self.assertRaises(ValueError): publication_payload(held)

    def test_disposition_file_alone_enforces_hold_and_private_links_stripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p/disp.DISPOSITIONS).write_text('{}')
            with self.assertRaises(ValueError): publication_payload(p)
        self.assertNotIn(disp.DISPOSITIONS, strip_private_references(f'[private]({disp.DISPOSITIONS})'))


if __name__ == "__main__": unittest.main()
