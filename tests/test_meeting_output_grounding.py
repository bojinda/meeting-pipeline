"""Synthetic regressions for source-grounded union meeting output cleanup."""
import json
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess.commitments import commitment_evidence, source_context_evidence
from meeting_postprocess.motions import adjournment_announcements, correct_adjournment_roles
from meeting_postprocess.normalization import clean_speaker_annotations
from meeting_postprocess.qa import check_action_consistency, check_meeting_identity
from meeting_postprocess.sections import BUSINESS, RECAP, PRE_MEETING
from meeting_postprocess.actions import filter_completed_request_tasks
from meeting_postprocess.speaker_turns import turn_catalog, approve_turn
import test_meeting_postprocess as pipeline_tests


def portion(number, text, section=BUSINESS):
    return {"source_chunk_id": number, "chunk_id": number, "meeting_section": section, "text": text}


class OutputGroundingTests(unittest.TestCase):
    def test_speaking_order_followup_is_chairs_undertaking_not_beneficiary_assignment(self):
        topic = "[SPEAKER_06] Taylor has not had a chance to speak. We should give Taylor an opportunity to speak first."
        reply = "[SPEAKER_07] I'll bring that up at the next meeting and see if they want an earlier turn."
        chunks = [portion(1, topic), portion(2, reply)]
        result = commitment_evidence(chunks, include_context=True)
        self.assertEqual(result, [topic + "\n" + reply])
        self.assertEqual(commitment_evidence([portion(1, reply)], include_context=True), [])

    def test_followup_does_not_cross_missing_chunks_redaction_or_section_boundaries(self):
        topic = "[Morgan] We should give Taylor an opportunity to speak first."
        reply = "[Chair] I'll bring that up at the next meeting."
        for chunks, blocked in (
            ([portion(1, topic), portion(3, reply)], set()),
            ([portion(1, topic), portion(2, reply)], {"1"}),
            ([portion(1, topic), portion(2, reply)], {"2"}),
            ([portion(1, topic, RECAP), portion(2, reply)], set()),
            ([portion(1, topic, PRE_MEETING), portion(2, reply)], set()),
        ):
            with self.subTest(chunks=chunks, blocked=blocked):
                self.assertEqual(commitment_evidence(chunks, include_context=True, blocked_context_source_ids=blocked), [])
        hypothetical = "[Morgan] For example, we could give Taylor an opportunity to speak first."
        self.assertEqual(commitment_evidence([portion(1, hypothetical), portion(2, reply)], include_context=True), [])

    def test_exact_testing_roles_and_full_change_sequence_reach_source_supplement(self):
        lines = ["[Morgan] Management directed yardmasters to help administer efficiency tests of operating crews.",
                 "[Taylor] Approximately 70% of jobs were abolished at Mac Yard.",
                 "[Taylor] After that, jobs were reinstated and relief assignments changed to evenings.",
                 "[Taylor] Four jobs remained cut.",
                 "[Taylor] one will be brought back as a relief job, leaving three reductions."]
        result = source_context_evidence([portion(1, "\n".join(lines))])
        self.assertEqual(result, ["\n".join(lines)])
        for label in (RECAP, PRE_MEETING):
            self.assertEqual(source_context_evidence([portion(1, "\n".join(lines), label)]), [])
        self.assertEqual(source_context_evidence([portion(1, "\n".join(lines))], {"1"}), [])

    def test_reported_pending_review_preserves_named_owner_and_reporting_context(self):
        source = ("[SPEAKER_04] The case needs review.\n"
                  "[SPEAKER_04] I spoke to Morgan about the case.\n"
                  "[SPEAKER_04] Morgan said they would look into it.\n"
                  "[SPEAKER_04] That was last Thursday.\n"
                  "[SPEAKER_04] He just sent me an email this morning; he will review tomorrow.")
        self.assertEqual(source_context_evidence([portion(1, source)]), [source])
        for changed in (source.replace("I spoke to Morgan", "I spoke to them"),
                        source.replace("I spoke to Morgan", "I spoke to Morgan and Taylor"),
                        source.replace("He just sent", "For example, he just sent"),
                        source.replace("[SPEAKER_04] He just", "[SPEAKER_05] He just")):
            with self.subTest(changed=changed):
                self.assertEqual(source_context_evidence([portion(1, changed)]), [])

    def test_unknown_identity_annotations_are_removed_without_creating_aliases(self):
        source = "[SPEAKER_10] I described the procedure.\n[Chair] Taylor, any questions?"
        for content in ("SPEAKER_10 (Taylor) described the procedure.", "Taylor (SPEAKER_10) described the procedure."):
            self.assertEqual(clean_speaker_annotations(content, source, {}), "SPEAKER_10 described the procedure.")
        aliases = {"SPEAKER_10": "Morgan"}
        self.assertEqual(clean_speaker_annotations("SPEAKER_10 (Taylor) spoke.", source, aliases), "Morgan spoke.")
        self.assertEqual(aliases, {"SPEAKER_10": "Morgan"})
        self.assertEqual(clean_speaker_annotations("Moved by Taylor (SPEAKER_03).", "[Chair] If there were a motion by Taylor.", {}), "Moved by SPEAKER_03.")
        self.assertEqual(clean_speaker_annotations("SPEAKER_10 (PLBO Chair) spoke.", source, {}), "SPEAKER_10 (PLBO Chair) spoke.")

    def test_verified_turn_name_is_preserved_but_mixed_label_is_not_globally_assigned(self):
        source = "[Taylor] A verified turn."
        content = "Taylor (SPEAKER_03) A verified turn."
        approved = [("SPEAKER_03", "Taylor", "A verified turn.")]
        self.assertEqual(clean_speaker_annotations(content, source, {}), "SPEAKER_03 A verified turn.")
        self.assertEqual(clean_speaker_annotations(content, source, {}, approved_passages=approved), "Taylor A verified turn.")
        self.assertEqual(clean_speaker_annotations("**" + content.replace(") ", ")** "), source, {}, approved_passages=approved), "**Taylor** A verified turn.")
        self.assertEqual(clean_speaker_annotations(content, source, {"SPEAKER_03": "Morgan"}, approved_passages=approved), "Taylor A verified turn.")
        self.assertEqual(clean_speaker_annotations("Taylor spoke.", source, {}), "Taylor spoke.")
        mixed = source + "\n[SPEAKER_03] An unresolved turn."
        self.assertEqual(clean_speaker_annotations("Taylor (SPEAKER_03) An unresolved turn.", mixed, {}, approved_passages=approved), "SPEAKER_03 An unresolved turn.")

    def test_verified_name_elsewhere_never_promotes_invented_label_pairing(self):
        source = "[Taylor] Opened the report.\n[SPEAKER_06] Raised the safety issue."
        for content in ("SPEAKER_99 (Taylor) raised the safety issue.", "Taylor (SPEAKER_99) raised the safety issue.",
                        "SPEAKER_99 (Taylor) Opened the report."):
            with self.subTest(content=content):
                self.assertIn("SPEAKER_99", clean_speaker_annotations(content, source, {}))
                self.assertNotIn("Taylor", clean_speaker_annotations(content, source, {}))
        approved = [("SPEAKER_03", "Taylor", "Opened the report.")]
        self.assertEqual(clean_speaker_annotations("SPEAKER_99 (Taylor) Opened the report.", source, {}, approved_passages=approved), "SPEAKER_99 Opened the report.")
        duplicate = "[Taylor] A verified turn.\n[SPEAKER_06] A verified turn."
        self.assertEqual(clean_speaker_annotations("SPEAKER_03 (Taylor) A verified turn.", duplicate, {}, approved_passages=[("SPEAKER_03", "Taylor", "A verified turn.")]), "SPEAKER_03 A verified turn.")

    def test_unrelated_discussion_blocks_vague_speaking_followup_and_action(self):
        source = ("[Taylor] We should give Morgan an opportunity to speak first.\n"
                  "[Riley] We need to obtain a printer.\n[Chair] I will bring that up at the next meeting.")
        evidence = commitment_evidence([portion(1, source)], include_context=True)
        self.assertEqual(evidence, [])
        candidate = "# Action Items\n- Chair: Raise Morgan's opportunity to speak at the next meeting."
        self.assertNotIn("Morgan", filter_completed_request_tasks(candidate, source, future_evidence=evidence))
        table = "# Minutes\n## Action Items\n| Owner | Task |\n|---|---|\n| Chair | Raise Morgan's opportunity to speak at the next meeting |"
        self.assertNotIn("Morgan", filter_completed_request_tasks(table, source, True, future_evidence=evidence))
        chunks = [portion(1, source.split("\n[Chair]", 1)[0]), portion(2, "[Chair] I will bring that up at the next meeting.")]
        self.assertEqual(commitment_evidence(chunks, include_context=True), [])

    def test_explicit_speaking_reference_after_interruption_does_not_borrow_printer_context(self):
        source = ("[Taylor] We should give Morgan an opportunity to speak first.\n"
                  "[Riley] We need to obtain a printer.\n"
                  "[Chair] I will bring that up at the next meeting: Morgan's opportunity to speak first.")
        evidence = commitment_evidence([portion(1, source)], include_context=True)
        self.assertEqual(evidence, [source.splitlines()[-1]])
        self.assertNotIn("printer", evidence[0])
        candidate = "# Action Items\n- Chair: Raise Morgan's opportunity to speak at the next meeting."
        self.assertEqual(filter_completed_request_tasks(candidate, source, future_evidence=evidence), candidate)

    def test_harmless_acknowledgments_preserve_local_speaking_followup(self):
        source = "[Taylor] We should give Morgan an opportunity to speak first.\n[Chair] Yeah, for sure.\n[Chair] I will bring that up at the next meeting."
        self.assertTrue(commitment_evidence([portion(1, source)], include_context=True))

    def test_consistency_flags_different_objects_recipients_and_qualifications(self):
        for left, right in (
            ("Review unresolved wage claims", "Review unresolved safety claims"),
            ("File the grievance by Friday", "File the report by Friday"),
            ("Send the report to Taylor", "Send the report to Riley"),
            ("Try to file the grievance by Friday", "File the grievance by Friday"),
            ("File the grievance if the claim is denied", "File the grievance"),
            ("Review the report by Friday", "Review the report by Monday"),
            ("Review three claims", "Review four claims"),
        ):
            with self.subTest(left=left, right=right):
                minutes = "# Minutes\n## Action Items\n| Owner | Task |\n|---|---|\n| Morgan | " + left + " |"
                actions = "# Action Items\n- Morgan: " + right
                self.assertTrue(check_action_consistency(minutes, actions))
                self.assertIn(left, minutes)  # QA never rewrites either task.
                self.assertIn(right, actions)

    def test_consistency_allows_safe_formatting_and_deadline_equivalence(self):
        minutes = "# Minutes\n## Action Items\n| Owner | Task |\n|---|---|\n| Morgan | Review the unresolved wage claims by Friday. |"
        for task in ("REVIEW unresolved wage claims by Friday", "Review the unresolved wage claims no later than Friday."):
            self.assertEqual(check_action_consistency(minutes, "# Action Items\n- **Morgan**: " + task), [])

    def test_fragmented_announced_roles_do_not_prove_formal_closure(self):
        source = ("[SPEAKER_03] Motion to adjourn.\n[SPEAKER_00] I second.\n"
                  "[Chair] Okay, seconded by Casey.\n[Chair] Casey.\n[Chair] Perfect.\n"
                  "[Chair] And motion by Taylor.\n[Morgan] One last thing: can members read the minutes?")
        self.assertEqual(len(adjournment_announcements(source)), 1)
        result = correct_adjournment_roles("The meeting was adjourned on motion.\n- Motion to adjourn moved by SPEAKER_03, seconded by SPEAKER_00. Carried.\nDiscussion about minutes continued.", source)
        self.assertNotIn("was adjourned", result)
        self.assertNotIn("Carried", result)
        self.assertIn("Motion to adjourn moved by Taylor, seconded by Casey.", result)
        self.assertIn("Discussion about minutes continued", result)
        self.assertEqual(clean_speaker_annotations("Moved by Taylor (SPEAKER_03), seconded by Casey (SPEAKER_00).", source, {}), "Moved by Taylor, seconded by Casey.")
        self.assertEqual(clean_speaker_annotations("Moved by Taylor (SPEAKER_03). SPEAKER_03 (Taylor) spoke later.", source, {"SPEAKER_03": "Morgan"}), "Moved by Taylor. Morgan spoke later.")

    def test_fragmented_motion_does_not_borrow_roles_from_substantive_or_hypothetical_speech(self):
        start = "[Chair] Motion to adjourn.\n[Chair] Seconded by Casey.\n"
        for following in ("[Chair] We should review the finances.\n[Chair] Motion by Taylor.",
                          "[Chair] If there were a motion by Taylor.",
                          "[Morgan] Motion by Taylor."):
            self.assertEqual(adjournment_announcements(start + following), [])

    def test_explicit_closing_and_other_motion_outcomes_remain_intact(self):
        source = "[Chair] Motion to adjourn, moved by Taylor, seconded by Casey.\n[Chair] Carried.\n[Chair] The meeting is adjourned."
        content = "The meeting was adjourned on motion. The budget motion carried."
        self.assertEqual(correct_adjournment_roles(content, source), content)
        self.assertEqual(check_meeting_identity("# Health & Safety Committee Meeting", "[Chair] Welcome to the Health & Safety Committee meeting."), [])
        self.assertTrue(check_meeting_identity("# Health & Safety Committee Meeting", "[Chair] Let's get started.\n[Taylor] We discussed health and safety issues."))

    def test_minutes_table_and_standalone_bullets_are_compared_without_inventing_tasks(self):
        minutes = "# Minutes\n## Action Items\n| # | Assigned To | Action |\n|---|---|---|\n| 1 | Morgan | Review the failed test tomorrow (reported commitment) |"
        self.assertEqual(check_action_consistency(minutes, "# Action Items\n- Morgan: Review the failed test tomorrow (reported commitment)."), [])
        for actions in ("# Action Items\nNone noted.", "# Action Items\n- Taylor: Review the failed test tomorrow."):
            self.assertTrue(check_action_consistency(minutes, actions))

    def test_raising_proposal_guard_requires_each_owner_and_does_not_approve_implementation(self):
        source = ("[Morgan] We should give Taylor an opportunity to speak first.\n"
                  "[SPEAKER_07] I'll bring that up at the next meeting and see if they want an earlier turn.")
        evidence = commitment_evidence([portion(1, source)], include_context=True)
        candidate = "# Action Items\n- SPEAKER_07: Raise Taylor's opportunity to speak at the next meeting."
        self.assertEqual(filter_completed_request_tasks(candidate, source, future_evidence=evidence), candidate)
        table = "# Minutes\n## Action Items\n| Owner | Task |\n|---|---|\n| SPEAKER_07 | Raise Taylor's opportunity to speak at the next meeting |"
        self.assertEqual(filter_completed_request_tasks(table, source, True, future_evidence=evidence), table)
        for unsupported in (
            candidate.replace("SPEAKER_07:", "SPEAKER_07 and Riley:"),
            candidate.replace("SPEAKER_07:", "Riley:"),
            "# Action Items\n- SPEAKER_07: Give Taylor an opportunity to speak first.",
        ):
            self.assertNotIn("Taylor", filter_completed_request_tasks(unsupported, source, future_evidence=evidence))
        self.assertNotIn("Taylor", filter_completed_request_tasks(candidate, source, future_evidence=[]))

    def test_source_context_never_contains_excluded_text_or_historical_actions(self):
        source = "[Morgan] I spoke to Taylor about the case.\n[Morgan] He just sent me an email; he will review tomorrow."
        for label in (PRE_MEETING, RECAP):
            self.assertEqual(source_context_evidence([portion(1, source, label)]), [])
        self.assertEqual(source_context_evidence([portion(1, source)], {"1"}), [])


class GroundedPipelineTests(unittest.TestCase):
    def test_approved_turn_passage_overrides_global_alias_without_promoting_unknown_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "synthetic-session"
            (transcript / "chunks_out").mkdir(parents=True)
            index = transcript / "chunks_out/transcript_chunks.jsonl"
            original = json.dumps({"chunk_id": 1, "text": "[SPEAKER_03] I will check the report.\n[SPEAKER_03] I will review the budget.\n[SPEAKER_06] Raised the safety issue."}) + "\n"
            index.write_text(original)
            (transcript / "speaker_aliases.json").write_text(json.dumps({"SPEAKER_03": "Morgan"}))
            catalog = turn_catalog(transcript)
            approve_turn(transcript, catalog, catalog["turns"][0]["turn_id"], "Taylor")
            def response(**call):
                if "Transcript chunk:\n" in call["prompt"]:
                    self.assertIn("[Taylor] I will check the report.", call["prompt"])
                    self.assertIn("[Morgan] I will review the budget.", call["prompt"])
                return "# Generated Output\n- SPEAKER_03 (Taylor) I will check the report.\n- SPEAKER_03 (Morgan) I will review the budget.\n- SPEAKER_99 (Taylor) Raised the safety issue."
            env = {"MEETING_SUMMARIES_ROOT": str(root / "outputs"), "MEETING_KEEP_RECAP": "0"}
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", ["summary", str(transcript), "--profile", "meeting"]), patch.object(pipeline_tests.engine, "call_ollama", side_effect=response), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline_tests.engine.main(), 0)
            for filename in ("chunk_summaries.jsonl", "minutes-draft.md", "action-items.md", "summary.md"):
                output = (root / "outputs/synthetic-session" / filename).read_text()
                self.assertIn("Taylor " + "I will check the report", output)
                self.assertIn("Morgan " + "I will review the budget", output)
                self.assertIn("SPEAKER_99 Raised the safety issue", output)
                self.assertNotIn("Taylor Raised the safety issue", output)
            self.assertEqual(index.read_text(), original)

    def test_omitted_map_details_reach_all_reducers_without_new_calls_or_identity_guesses(self):
        source = ("[Chair] Let's get started.\n"
                  "[SPEAKER_06] We should give Taylor an opportunity to speak first.\n"
                  "[SPEAKER_07] I'll bring that up at the next meeting and see if they want an earlier turn.\n"
                  "[SPEAKER_10] Management directed yardmasters to help administer efficiency tests of operating crews.\n"
                  "[SPEAKER_04] Approximately 70% of jobs were abolished at Mac Yard.\n"
                  "[SPEAKER_04] Jobs were reinstated and relief jobs reassigned to evenings.\n"
                  "[SPEAKER_04] Four jobs remained cut. one will be brought back as a relief job, leaving three reductions.\n"
                  "[SPEAKER_04] We had a failed efficiency test.\n"
                  "[SPEAKER_04] I spoke to Morgan about the case.\n"
                  "[SPEAKER_04] Morgan said they would look into it.\n"
                  "[SPEAKER_04] That was last Thursday.\n"
                  "[SPEAKER_04] He just sent me an email this morning; he will review tomorrow.\n"
                  "[SPEAKER_03] Motion to adjourn.\n[SPEAKER_00] I second.\n"
                  "[Chair] Seconded by Casey.\n[Chair] Casey.\n[Chair] Perfect.\n[Chair] Motion by Taylor.\n"
                  "[SPEAKER_06] One last thing: can members read the minutes?")
        def response(call):
            prompt = call["prompt"]
            if "Transcript chunk:\n" in prompt:
                return "## Topics\n- SPEAKER_10 (Taylor) reported that yardmasters were tested.\n## Action Items\nNone noted."
            self.assertIn("help administer efficiency tests of operating crews", prompt)
            self.assertIn("one will be brought back as a relief job, leaving three reductions", prompt)
            self.assertIn("he will review tomorrow", prompt)
            self.assertIn("explicit", prompt.casefold())
            if "write an action-items document" in prompt or "write formal draft minutes" in prompt:
                self.assertIn("bring that up at the next meeting", prompt)
                return ("# Generated Document\n## Action Items\n"
                        "- SPEAKER_07: Raise Taylor's opportunity to speak earlier at the next meeting and ask if they want it.\n"
                        "- Morgan: Review the failed test tomorrow (reported pending undertaking).\n"
                        "## Overview\nThe meeting was adjourned on motion.\n"
                        "## Motions\n- Motion to adjourn moved by SPEAKER_03, seconded by SPEAKER_00. Carried.")
            return "# Health & Safety Committee Meeting\nThe meeting was adjourned on motion.\n- SPEAKER_10 (Taylor) discussed testing."
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = pipeline_tests.PipelineTests().run_pipeline(Path(directory), source, model_reply=response,
                environment={"MEETING_MAP_MODEL": "qwen3.8:27b", "MEETING_REDUCE_MODEL": "qwen3.8:27b"})
            self.assertEqual(status, 0)
            maps = [call for call in calls if "Transcript chunk:\n" in call["prompt"]]
            self.assertEqual(len(calls), len(maps) + 3)
            self.assertTrue(all(call["model"] == "qwen3.8:27b" for call in calls))
            for name in ("chunk_summaries.jsonl", "minutes-draft.md", "summary.md", "action-items.md"):
                content = (output / name).read_text(encoding="utf-8")
                self.assertNotIn("SPEAKER_10 (Taylor)", content)
                self.assertNotIn("was adjourned", content)
            summary = (output / "summary.md").read_text()
            self.assertTrue(summary.startswith("# Meeting Summary"))
            minutes, actions = [(output / name).read_text() for name in ("minutes-draft.md", "action-items.md")]
            self.assertEqual(check_action_consistency(minutes, actions), [])
            self.assertIn("Morgan: Review", minutes)
            self.assertIn("SPEAKER_07: Raise", actions)
            qa = json.loads((output / "minutes-qa.json").read_text())
            self.assertIn("unsupported_meeting_identity", {entry["code"] for entry in qa["findings"]})


if __name__ == "__main__":
    unittest.main()
