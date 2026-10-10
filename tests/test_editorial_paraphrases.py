"""Offline proposal assessment: lexical uncertainty never becomes approval."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed
from meeting_postprocess.commitments import commitment_evidence
from meeting_postprocess.sections import BUSINESS, ADJOURNMENT
from test_editorial_notes import portion


def proposal(task, owner="Taylor", category="undertaking", refs=None):
    return {"items": [{"id": "A1", "category": category, "task": task, "owners": [owner],
                       "source_ids": refs or ["1:L1"], "member_facing": True, "concerns": []}]}


def assess(text, raw):
    chunks = [portion(1, text)]
    return ed.validate_register(raw, ed.source_records(chunks), commitment_evidence(chunks, include_context=True), text)


class ParaphraseGuardTests(unittest.TestCase):
    def test_paraphrase_preserves_proposal_and_reaches_review_without_repeated_subset_failures(self):
        text = "[Taylor] I'll send the report to the committee."
        raw = proposal("Circulate the report to the committee.")
        original = copy.deepcopy(raw)
        register = assess(text, raw)
        self.assertEqual(raw, original)
        item = register["items"][0]
        self.assertEqual(item["task"], raw["items"][0]["task"])
        self.assertEqual(item["owners"], ["Taylor"])
        self.assertEqual(item["category"], "undertaking")
        self.assertIn("semantic support requires operator review", item["status"])
        self.assertIn("action_support_review", {f["code"] for f in register["findings"]})
        self.assertIn(item["status"], ed.undertaking_table(register))

    def test_source_owned_future_is_not_proof_of_an_unrelated_proposed_task(self):
        register = assess("[Taylor] I'll send the report.", proposal("Reserve a hotel."))
        self.assertEqual(register["items"][0]["support"]["status"], "uncertain_local_support")
        self.assertIn("Proposed undertaking", register["items"][0]["status"])
        self.assertIn("action_support_review", {f["code"] for f in register["findings"]})
        # No future commitment at all remains a hard substantive failure.
        with self.assertRaisesRegex(ed.EditorialFailure, "unsupported_undertaking"):
            assess("[Taylor] The report was discussed.", proposal("Reserve a hotel."))

    def test_named_assignment_paraphrase_retains_its_source_actor(self):
        register = assess("[Chair] Taylor will send the report to the committee.", proposal("Circulate the report to the committee."))
        self.assertEqual(register["items"][0]["owners"], ["Taylor"])
        with self.assertRaisesRegex(ed.EditorialFailure, "unsupported_owner"):
            assess("[Taylor] I'll send the report to the committee.", proposal("Circulate the report to the committee.", owner="Morgan"))

    def test_changed_named_and_collective_recipients_remain_hard_failures(self):
        for recipient in ("Morgan", "management"):
            with self.subTest(recipient=recipient), self.assertRaisesRegex(ed.EditorialFailure, "unsupported_recipient"):
                assess("[Taylor] I'll send the report to the committee.", proposal(f"Circulate the report to {recipient}."))
        with self.assertRaisesRegex(ed.EditorialFailure, "unsupported_recipient"):
            assess("[Taylor] I already sent the report to the committee.", proposal("Sent the report to management.", category="completed"))

    def test_future_negated_and_conditional_completion_are_not_completed_work(self):
        for text in ("[Taylor] I'll send the report.", "[Taylor] I have not sent the report.",
                     "[Taylor] If approved, the report will be sent."):
            with self.subTest(text=text), self.assertRaisesRegex(ed.EditorialFailure, "unsupported_completion"):
                assess(text, proposal("Dispatched the report.", category="completed"))
        register = assess("[Taylor] I already sent the report.", proposal("Dispatched the report.", category="completed"))
        self.assertIn("Proposed completed", register["items"][0]["status"])
        with self.assertRaisesRegex(ed.EditorialFailure, "unsupported_undertaking"):
            assess("[Taylor] I already sent the report.", proposal("Circulate the report."))

    def test_conditional_paraphrase_is_retained_but_lost_condition_or_attempt_is_blocked(self):
        text = "[Taylor] If the claim is denied, I'll try to file the grievance."
        register = assess(text, proposal("If the claim is rejected, attempt to submit the grievance."))
        self.assertEqual(register["items"][0]["support"]["status"], "uncertain_local_support")
        for task in ("Attempt to submit the grievance.", "If the claim is rejected, submit the grievance."):
            with self.subTest(task=task), self.assertRaisesRegex(ed.EditorialFailure, "lost_commitment_qualification"):
                assess(text, proposal(task))
        with self.assertRaisesRegex(ed.EditorialFailure, "contradictory_task_polarity"):
            assess("[Taylor] If the claim is not accepted, I'll file the grievance.", proposal("If the claim is accepted, submit the grievance."))

    def test_confidentiality_still_blocks_an_otherwise_plausible_paraphrase(self):
        for task, concern in (("Circulate the medical history report.", []),
                              ("Circulate the report.", ["confidentiality"])):
            raw = proposal(task)
            raw["items"][0]["concerns"] = concern
            with self.subTest(task=task), self.assertRaisesRegex(ed.EditorialFailure, "confidential_action_presentation"):
                assess("[Taylor] I'll send the report.", raw)

    def test_audit_reports_every_action_and_preserves_complete_proposals_and_sources(self):
        text = "[Taylor] I'll send the report to the committee."
        records = ed.source_records([portion(1, text)])
        raw = {"items": [proposal("Circulate the report to the committee.")["items"][0],
                         proposal("Circulate the report.", refs=["fabricated"])["items"][0],
                         proposal("Circulate the report.", owner="Morgan")["items"][0]]}
        for number, item in enumerate(raw["items"], 1):
            item["id"] = f"A{number}"
        original = copy.deepcopy(raw)
        result = ed.audit_register(raw, records, commitment_evidence([portion(1, text)]), text)
        self.assertEqual([r["outcome"] for r in result["outcomes"]], ["review_required", "hard_block", "hard_block"])
        self.assertEqual(result["hard_block_count"], 2)
        self.assertIsNone(result["register"])
        self.assertEqual(result["proposed_register"], original)
        self.assertEqual(result["sources"], records)
        self.assertEqual(raw, original)
        checklist = ed.register_review_checklist(result)
        for item in original["items"]:
            self.assertIn(item["id"], checklist)
            self.assertIn(item["task"], checklist)
        self.assertIn("action_support_review", checklist)
        self.assertIn("unsupported_owner", checklist)

    def test_audit_does_not_lose_duplicate_or_unsupplied_reference_restrictions(self):
        text = "[Taylor] I'll send the report."
        records = ed.source_records([portion(1, text)])
        raw = proposal("Circulate the report.")
        raw["items"].append(copy.deepcopy(raw["items"][0]))
        result = ed.audit_register(raw, records, commitment_evidence([portion(1, text)]), text)
        self.assertEqual(result["outcomes"][1]["failure_category"], "duplicate_register_id")
        result = ed.audit_register(proposal("Circulate the report."), records, [], text, supplied=set())
        self.assertEqual(result["outcomes"][0]["failure_category"], "unsupplied_action_evidence")

    def test_equivalent_negative_conditions_are_not_polarity_reversals(self):
        for before, after in (("does not respond", "fails to respond"), ("fails to respond", "does not respond"),
                              ("doesn't respond", "fails to respond")):
            with self.subTest(before=before, after=after):
                register = assess(f"[Taylor] If the office {before}, I'll file the grievance.",
                                  proposal(f"If the office {after}, file the grievance."))
                self.assertEqual(register["items"][0]["category"], "undertaking")
        for task in ("If the office responds, file the grievance.", "Do not file the grievance."):
            source = "[Taylor] If the office does not respond, I'll file the grievance." if task.startswith("If") else "[Taylor] I'll file the grievance."
            with self.subTest(task=task), self.assertRaisesRegex(ed.EditorialFailure, "contradictory_task_polarity"):
                assess(source, proposal(task))

    def test_each_qualification_stays_with_its_own_action_by_the_same_speaker(self):
        for separator in ("\n[Taylor] ", ". ", "; ", " and "):
            text = "[Taylor] If the office does not respond, I'll try to file the grievance" + separator + "I'll send the minutes."
            raw = proposal("Send the minutes.", refs=["1:L1", "1:L2"] if "\n" in separator else ["1:L1"])
            with self.subTest(separator=separator):
                register = assess(text, raw)
                self.assertEqual(register["items"][0]["task"], "Send the minutes.")
                self.assertNotIn("qualification_relationship_review", {f["code"] for f in register["findings"]})
            raw["items"][0]["task"] = "File the grievance."
            with self.subTest(lost_condition=separator), self.assertRaisesRegex(ed.EditorialFailure, "lost_commitment_qualification"):
                assess(text, raw)

    def test_ambiguous_action_condition_relationship_is_retained_for_review(self):
        text = "[Taylor] If the office does not respond, I'll file the grievance. I'll file the grievance at the next meeting."
        register = assess(text, proposal("File the grievance."))
        self.assertIn("qualification_relationship_review", {f["code"] for f in register["findings"]})
        self.assertIn("Proposed undertaking", register["items"][0]["status"])
        self.assertEqual(register["items"][0]["task"], "File the grievance.")

    def test_composite_action_does_not_lose_its_other_source_supported_recipient(self):
        source = "[Taylor] I'll send an email to Morgan. I'll bring the budget discussion up at the next meeting."
        register = assess(source, proposal("Send an email to Morgan, and bring the budget discussion up at the next meeting."))
        self.assertIn("qualification_relationship_review", {f["code"] for f in register["findings"]})
        self.assertEqual(register["items"][0]["source_ids"], ["1:L1"])

    def test_locally_grounded_recipient_initials_and_explicit_expansion_are_equivalent(self):
        for source, task in (("I'll send the report to the health and safety committee.", "Circulate the report to the H&S committee."),
                             ("I'll send the report to H&S (health and safety) committee.", "Send the report to the health and safety committee."),
                             ("I'll send the report to, uh, Riley.", "Send the report to Riley.")):
            with self.subTest(source=source):
                register = assess("[Taylor] " + source, proposal(task))
                self.assertNotIn("recipient_equivalence_review", {f["code"] for f in register["findings"]})
        with self.assertRaisesRegex(ed.EditorialFailure, "unsupported_recipient"):
            assess("[Taylor] I'll send the report to the health and safety committee.", proposal("Send the report to the finance committee."))

    def test_recipient_equivalence_is_reviewed_without_inventing_an_alias(self):
        for source, task in (("I'll send the report to the H&S committee.", "Send the report to the health and safety committee."),
                             ("I'll send you the report.", "Send the report to SPEAKER_02."),
                             ("I'll send the report to the health and safety committee. I'll send the report to the human and staffing committee.", "Send the report to H&S committee.")):
            raw = proposal(task)
            original = copy.deepcopy(raw)
            with self.subTest(source=source):
                register = assess("[Taylor] " + source, raw)
                self.assertIn("recipient_equivalence_review", {f["code"] for f in register["findings"]})
                self.assertIn("Proposed undertaking", register["items"][0]["status"])
                self.assertEqual(raw, original)


LIVE = Path(os.environ.get("OCTOBER_EDITORIAL_LIVE_DIR", ROOT / "ignore/october-editorial-live.wnfcYj/editorial-output"))


@unittest.skipUnless(all((LIVE / name).is_file() for name in (ed.RESPONSE, "meeting_sections.jsonl", "chunk_summaries.jsonl", ed.BUDGETS)),
                     "saved 21-action live response and its original source/maps/budgets are not available locally")
class SavedProposalTests(unittest.TestCase):
    def test_saved_response_has_complete_source_bound_action_audit(self):
        chunks = [json.loads(line) for line in (LIVE / "meeting_sections.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        response = json.loads((LIVE / ed.RESPONSE).read_text(encoding="utf-8"))
        records = ed.source_records(chunks)
        self.assertEqual(ed.digest(records), response["source_hash"])
        raw = json.loads(response["responses"]["register"])
        self.assertEqual(len(raw["items"]), 21)
        source = "\n".join(c["text"] for c in chunks if c["meeting_section"] in {BUSINESS, ADJOURNMENT})
        # Deliberately no cross-line recovery without the live redaction-boundary
        # snapshot. Any such support remains a blocker for exact-run validation.
        commitments = commitment_evidence(chunks)
        result = ed.audit_register(raw, records, commitments, source)
        self.assertEqual(len(result["outcomes"]), 21)
        result["context_boundary_verification"] = "single-line evidence only; original redaction/identity snapshots required for full-run equivalence"
        output = ROOT / "ignore/editorial-guard-correction"
        output.mkdir(parents=True, exist_ok=True)
        ed.write_private_json(output / "live-action-audit.private.json", result)


if __name__ == "__main__":
    unittest.main()
