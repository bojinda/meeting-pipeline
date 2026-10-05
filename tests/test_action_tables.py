from __future__ import annotations

import contextlib
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

import ollama_session_summary as engine
from meeting_postprocess.actions import filter_completed_request_tasks


PROPOSAL = "[Chair] Taylor, I think we have to request another meeting with the manager."
TASK = "Request another meeting with the manager"


def table(headers, rows, separators=None, outer_pipes=True):
    def row(cells):
        text = " | ".join(cells)
        return "| " + text + " |" if outer_pipes else text
    return "\n".join([row(headers), row(separators or ["---"] * len(headers)), *(row(cells) for cells in rows)])


def minutes(actions):
    return "# Draft Minutes\n\n## Overview\nPublic business.\n\n## Action Items\n\n" + actions + "\n\n## Open Questions\nNone noted.\n"


class ActionTableTests(unittest.TestCase):
    def test_proposal_row_removed_in_each_common_column_order(self):
        for headers, cells in (
            (["#", "Assigned To", "Action"], ["1", "Taylor", TASK]),
            (["#", "Action Item", "Responsible"], ["1", TASK, "Taylor"]),
            (["Owner", "Task"], ["Taylor", TASK]),
        ):
            with self.subTest(headers=headers):
                cleaned = filter_completed_request_tasks(minutes(table(headers, [cells])), PROPOSAL, action_sections_only=True)
                section = cleaned.split("## Action Items", 1)[1].split("## Open Questions", 1)[0]
                self.assertEqual(section.strip(), "None noted.")
                self.assertNotIn("|", section)
                self.assertIn("## Overview\nPublic business.", cleaned)
                self.assertIn("## Open Questions\nNone noted.", cleaned)

    def test_explicit_followup_commitment_retains_same_table_row(self):
        source = PROPOSAL + "\n[Taylor] I'll request another meeting with the manager."
        for headers, cells in (
            (["#", "Assigned To", "Action"], ["1", "Taylor", TASK]),
            (["#", "Action Item", "Responsible"], ["1", TASK, "Taylor"]),
            (["Owner", "Task"], ["Taylor", TASK]),
        ):
            with self.subTest(headers=headers):
                content = minutes(table(headers, [cells]))
                self.assertEqual(filter_completed_request_tasks(content, source, action_sections_only=True), content)

    def test_supported_row_survives_partial_removal_with_original_table_structure(self):
        data = table(["#", "Assigned To", "Action"], [["1", "Taylor", TASK], ["2", "Morgan", "Approve both employees"]], [":---:", ":---", "---:"])
        content = minutes(data)
        source = PROPOSAL + "\n[Morgan] I'll approve access for both members."
        cleaned = filter_completed_request_tasks(content, source, action_sections_only=True)
        self.assertNotIn("| 1 | Taylor", cleaned)
        self.assertIn("| 2 | Morgan | Approve both employees |", cleaned)
        self.assertIn("| # | Assigned To | Action |\n| :---: | :--- | ---: |", cleaned)
        self.assertNotIn("None noted.", cleaned.split("## Action Items", 1)[1].split("## Open Questions", 1)[0])

    def test_completed_outreach_rows_filtered_with_supported_contact_task_retained(self):
        source = "[Chair] I've already messaged Casey and Riley for their contact details.\n[Morgan] I'll contact Casey Morgan for contact detail."
        data = table(["#", "Action Item", "Responsible"], [["1", "Provide contact details", "Casey and Riley"], ["2", "Contact Casey Morgan for contact detail", "Morgan"]])
        cleaned = filter_completed_request_tasks(minutes(data), source, action_sections_only=True)
        self.assertNotIn("| 1 |", cleaned)
        self.assertIn("| 2 | Contact Casey Morgan for contact detail | Morgan |", cleaned)

    def test_completed_outreach_recipient_commitment_keeps_table_row(self):
        source = "[Chair] I've already contacted Morgan for contact detail.\n[SPEAKER_06] I'll send my contact detail tomorrow."
        content = minutes(table(["Owner", "Task"], [["Morgan (SPEAKER_06)", "Provide contact detail"]]))
        self.assertEqual(filter_completed_request_tasks(content, source, action_sections_only=True), content)

    def test_owner_annotations_do_not_turn_collective_proposal_into_commitment(self):
        for owner in ("Taylor (SPEAKER_03)", "SPEAKER_04 (Committee Chair)", "Morgan (SPEAKER_06)"):
            with self.subTest(owner=owner):
                content = minutes(table(["Owner", "Task"], [[owner, TASK]]))
                cleaned = filter_completed_request_tasks(content, PROPOSAL, action_sections_only=True)
                self.assertNotIn(TASK, cleaned)
                self.assertIn("## Action Items\nNone noted.", cleaned)

    def test_annotated_owners_match_explicit_speaker_commitments(self):
        for owner, speaker in (("Taylor (SPEAKER_03)", "SPEAKER_03"), ("SPEAKER_04 (Committee Chair)", "SPEAKER_04"), ("Morgan (SPEAKER_06)", "SPEAKER_06")):
            with self.subTest(owner=owner):
                source = PROPOSAL + f"\n[{speaker}] I'll request another meeting with the manager."
                content = minutes(table(["Owner", "Task"], [[owner, TASK]]))
                self.assertEqual(filter_completed_request_tasks(content, source, action_sections_only=True), content)

    def test_annotated_addressee_can_accept_without_repeating_task(self):
        source = PROPOSAL + "\n[SPEAKER_03] Yes, I'll do it."
        content = minutes(table(["Owner", "Task"], [["Taylor (SPEAKER_03)", TASK]]))
        self.assertEqual(filter_completed_request_tasks(content, source, action_sections_only=True), content)

    def test_other_speakers_commitment_does_not_support_annotated_owner(self):
        source = PROPOSAL + "\n[SPEAKER_06] I'll request another meeting with the manager."
        content = minutes(table(["Owner", "Task"], [["Taylor (SPEAKER_03)", TASK]]))
        self.assertNotIn(TASK, filter_completed_request_tasks(content, source, action_sections_only=True))
        source = PROPOSAL + "\n[Taylor] I'll request another meeting with the manager."
        content = minutes(table(["Owner", "Task"], [["SPEAKER_04 (Committee Chair)", TASK]]))
        self.assertNotIn(TASK, filter_completed_request_tasks(content, source, action_sections_only=True))

    def test_all_rows_removed_from_standalone_actions_leaves_none_noted(self):
        content = "# Action Items\n\n" + table(["#", "Assigned To", "Action"], [["1", "Taylor", TASK]])
        cleaned = filter_completed_request_tasks(content, PROPOSAL)
        self.assertEqual(cleaned.strip(), "# Action Items\nNone noted.")

    def test_all_rows_removed_from_multiple_tables_replaces_whole_section(self):
        data = "Items discussed:\n" + table(["Owner", "Task"], [["Taylor", TASK]])
        data += "\n\nAdditional items:\n" + table(["Action Item", "Responsible"], [[TASK, "Taylor (SPEAKER_03)"]])
        cleaned = filter_completed_request_tasks(minutes(data), PROPOSAL, action_sections_only=True)
        self.assertEqual(cleaned.split("## Action Items", 1)[1].split("## Open Questions", 1)[0].strip(), "None noted.")

    def test_empty_filtered_table_does_not_erase_remaining_bullet(self):
        data = table(["Owner", "Task"], [["Taylor", TASK]]) + "\n\n- Morgan – approve access for both members."
        cleaned = filter_completed_request_tasks(minutes(data), PROPOSAL + "\n[Morgan] I'll approve access for both members.", action_sections_only=True)
        self.assertIn("- Morgan – approve access for both members.", cleaned)
        self.assertNotIn("|", cleaned)

    def test_table_outside_action_section_is_preserved(self):
        data = table(["Owner", "Task"], [["Taylor", TASK]])
        content = "# Draft Minutes\n## Topics Discussed\n" + data + "\n## Action Items\n" + data + "\n## Open Questions\nNone noted."
        cleaned = filter_completed_request_tasks(content, PROPOSAL, action_sections_only=True)
        self.assertIn(data, cleaned.split("## Action Items")[0])
        self.assertNotIn(data, cleaned.split("## Action Items")[1])

    def test_headers_without_outer_pipes_and_with_markdown_emphasis(self):
        data = table(["**#**", "**Action Item**", "**Responsible**"], [["1", TASK, "**Taylor** (SPEAKER_03)"]], outer_pipes=False)
        cleaned = filter_completed_request_tasks(minutes(data), PROPOSAL, action_sections_only=True)
        self.assertNotIn(TASK, cleaned)
        self.assertIn("None noted.", cleaned)

    def test_extra_columns_do_not_supply_commitment_evidence(self):
        data = table(["#", "Assigned To", "Action", "Status"], [["1", "Taylor", TASK, "Committed"]])
        cleaned = filter_completed_request_tasks(minutes(data), PROPOSAL, action_sections_only=True)
        self.assertNotIn("Committed", cleaned)
        self.assertNotIn(TASK, cleaned)

    def test_escaped_pipes_and_role_annotations_in_retained_rows_are_preserved(self):
        data = table(["Owner", "Task"], [["Taylor", TASK], ["Morgan (SPEAKER_06)", r"Check manual \| suppression procedure"]])
        cleaned = filter_completed_request_tasks(minutes(data), PROPOSAL, action_sections_only=True)
        self.assertIn(r"| Morgan (SPEAKER_06) | Check manual \| suppression procedure |", cleaned)
        self.assertNotIn(TASK, cleaned)

    def test_unknown_headers_and_fenced_table_examples_are_preserved(self):
        content = minutes(table(["Who", "What"], [["Taylor", TASK]]))
        self.assertEqual(filter_completed_request_tasks(content, PROPOSAL, action_sections_only=True), content)
        content = minutes("```markdown\n" + table(["Owner", "Task"], [["Taylor", TASK]]) + "\n```")
        self.assertEqual(filter_completed_request_tasks(content, PROPOSAL, action_sections_only=True), content)


class MultiOwnerActionTests(unittest.TestCase):
    def check_action(self, source, owner, task, retained):
        for content in ("# Action Items\n- " + owner + " – " + task + ".", minutes(table(["Owner", "Task"], [[owner, task]]))):
            with self.subTest(content=content):
                cleaned = filter_completed_request_tasks(content, source, action_sections_only=True)
                if retained:
                    self.assertEqual(cleaned, content)
                else:
                    self.assertNotIn(task, cleaned)
                    self.assertIn("None noted.", cleaned)

    def test_completed_outreach_requires_each_owner_commitment(self):
        source = "[Chair] Casey and Riley were already contacted for information.\n[Casey] I'll send my information tomorrow."
        self.check_action(source, "Casey and Riley", "provide information tomorrow", False)
        self.check_action(source + "\n[Riley] I'll send my information tomorrow.", "Casey and Riley", "provide information tomorrow", True)

    def test_collective_proposal_requires_each_owner_commitment(self):
        source = "[Chair] I think we should request another meeting.\n[Taylor] I'll request it."
        self.check_action(source, "Taylor and Morgan", "request another meeting", False)
        self.check_action(source + "\n[Morgan] I'll request it.", "Taylor and Morgan", "request another meeting", True)

    def test_joint_owners_each_accept_with_short_replies(self):
        for acceptance in ("Yes, I'll do it.", "Sure.", "Agreed.", "Okay, I'll do it.", "I will do that."):
            with self.subTest(acceptance=acceptance):
                source = "[Chair] I think we should request another meeting.\n[Taylor] " + acceptance + "\n[Morgan] " + acceptance
                self.check_action(source, "Taylor and Morgan", "request another meeting", True)

    def test_only_first_joint_owner_accepts_with_short_reply(self):
        source = "[Chair] I think we should request another meeting.\n[Taylor] Yes, I'll do it."
        self.check_action(source, "Taylor and Morgan", "request another meeting", False)

    def test_only_second_joint_owner_accepts_with_short_reply(self):
        source = "[Chair] I think we should request another meeting.\n[Morgan] Yes, I'll do it."
        self.check_action(source, "Taylor and Morgan", "request another meeting", False)

    def test_second_joint_owner_declines_or_remains_tentative(self):
        for reply in ("No, I won't do it.", "Okay, but I decline.", "Yes, but I think we should request another meeting."):
            with self.subTest(reply=reply):
                source = "[Chair] I think we should request another meeting.\n[Taylor] Yes, I'll do it.\n[Morgan] " + reply
                self.check_action(source, "Taylor and Morgan", "request another meeting", False)

    def test_joint_annotated_owners_have_separate_short_acceptances(self):
        for first, second in (("Taylor", "Morgan"), ("SPEAKER_03", "SPEAKER_06"), ("Taylor", "SPEAKER_06"), ("SPEAKER_03", "Morgan")):
            with self.subTest(first=first, second=second):
                source = f"[Chair] I think we should request another meeting.\n[{first}] Sure.\n[{second}] I will do that."
                self.check_action(source, "Taylor (SPEAKER_03) and Morgan (SPEAKER_06)", "request another meeting", True)

    def test_short_acceptance_scan_keeps_existing_bounded_reply_window(self):
        source = "[Chair] I think we should request another meeting.\n[Taylor] Sure.\n[Casey] I'll check the agenda.\n[Riley] I'm reviewing the schedule.\n[Morgan] Agreed."
        self.check_action(source, "Taylor and Morgan", "request another meeting", False)
        source = "[Chair] I think we should request another meeting.\n[Casey] I'll check the agenda.\n[Taylor] Sure.\n[Morgan] Agreed."
        self.check_action(source, "Taylor and Morgan", "request another meeting", True)

    def test_owner_neutral_reply_then_short_acceptance_is_supported(self):
        for owner, first, last in (("Taylor", "Taylor", "Taylor"), ("Taylor (SPEAKER_03)", "SPEAKER_03", "SPEAKER_03"), ("Taylor (SPEAKER_03)", "Taylor", "SPEAKER_03")):
            with self.subTest(owner=owner, first=first, last=last):
                source = f"[Chair] Taylor, I think we should request another meeting.\n[{first}] What time were you thinking?\n[Morgan] Agreed.\n[{last}] Sure."
                self.check_action(source, owner, "request another meeting", True)

    def test_owner_neutral_reply_without_acceptance_is_unsupported(self):
        source = "[Chair] I think we should request another meeting.\n[Taylor] What time were you thinking?\n[Morgan] Agreed.\n[Taylor] I'm checking the calendar."
        self.check_action(source, "Taylor", "request another meeting", False)
        self.check_action(source + "\n[Taylor] Sure.", "Taylor", "request another meeting", False)

    def test_owner_clear_decline_remains_unsupported(self):
        for replies in ("[Taylor] No, I won't do it.", "[Taylor] What time were you thinking?\n[Taylor] I decline.", "[Taylor] No, I won't do it.\n[Morgan] Agreed.\n[Taylor] Sure."):
            with self.subTest(replies=replies):
                source = "[Chair] I think we should request another meeting.\n" + replies
                self.check_action(source, "Taylor", "request another meeting", False)

    def test_joint_owners_accept_at_different_positions_after_neutral_reply(self):
        for first, second in (("Taylor", "Morgan"), ("Morgan", "Taylor")):
            with self.subTest(first=first, second=second):
                source = f"[Chair] I think we should request another meeting.\n[{first}] What time were you thinking?\n[{second}] Agreed.\n[{first}] Sure."
                self.check_action(source, "Taylor and Morgan", "request another meeting", True)
        source = "[Chair] I think we should request another meeting.\n[SPEAKER_03] What time were you thinking?\n[SPEAKER_06] Agreed.\n[SPEAKER_03] Sure."
        self.check_action(source, "Taylor (SPEAKER_03) and Morgan (SPEAKER_06)", "request another meeting", True)

    def test_single_annotated_owner_matches_either_identity(self):
        for identity in ("Taylor", "SPEAKER_03"):
            with self.subTest(identity=identity):
                source = "[Chair] Taylor was already contacted for information.\n[" + identity + "] I'll send my information tomorrow."
                self.check_action(source, "Taylor (SPEAKER_03)", "provide information tomorrow", True)
                source = "[Chair] Taylor, I think we should request another meeting.\n[" + identity + "] I'll request it."
                self.check_action(source, "Taylor (SPEAKER_03)", "request another meeting", True)

    def test_another_person_cannot_support_single_annotated_owner(self):
        self.check_action("[Chair] Taylor was already contacted for information.\n[Morgan] I'll send my information tomorrow.", "Taylor (SPEAKER_03)", "provide information tomorrow", False)
        self.check_action("[Chair] Taylor, I think we should request another meeting.\n[Morgan] I'll request it.", "Taylor (SPEAKER_03)", "request another meeting", False)

    def test_joint_annotated_owners_keep_aliases_separate(self):
        for separator in (" and ", " & ", ", ", " / "):
            with self.subTest(separator=separator):
                owners = "Taylor (SPEAKER_03)" + separator + "Morgan (SPEAKER_06)"
                source = "[Chair] Taylor and Morgan were already contacted for information.\n[SPEAKER_03] I'll send my information tomorrow."
                self.check_action(source, owners, "provide information tomorrow", False)
                self.check_action(source + "\n[Morgan] I'll send my information tomorrow.", owners, "provide information tomorrow", True)
                source = "[Chair] I think we should request another meeting.\n[SPEAKER_03] I'll request it."
                self.check_action(source, owners, "request another meeting", False)
                self.check_action(source + "\n[SPEAKER_06] I'll request it.", owners, "request another meeting", True)

    def test_joint_named_assignment_supports_each_owner(self):
        self.check_action("[Chair] Casey and Riley were already contacted for information.\n[Chair] Casey and Riley will provide information tomorrow.", "Casey and Riley", "provide information tomorrow", True)
        self.check_action("[Chair] I think we should request another meeting.\n[Chair] Taylor and Morgan will request another meeting.", "Taylor and Morgan", "request another meeting", True)

    def test_reference_to_one_owner_does_not_borrow_another_owners_verb(self):
        self.check_action("[Chair] Casey and Riley were already contacted for information.\n[Chair] Casey said Riley will provide information tomorrow.", "Casey and Riley", "provide information tomorrow", False)
        self.check_action("[Chair] I think we should request another meeting.\n[Chair] Taylor said Morgan will request another meeting.", "Taylor and Morgan", "request another meeting", False)

    def test_unrelated_or_tentative_reply_does_not_supply_commitment(self):
        for reply in ("[Taylor] I'll send that agenda.", "[Taylor] I'll skip it until after the meeting.", "[Taylor] I think I should request it.", "[Taylor] I won't request it."):
            with self.subTest(reply=reply):
                self.check_action("[Chair] I think we should request another meeting.\n" + reply, "Taylor", "request another meeting", False)


class ActionTablePipelineTests(unittest.TestCase):
    def test_erroneous_model_table_removed_or_retained_based_on_followup_commitment(self):
        row = "| 1 | Taylor | Request another meeting with the manager |"
        data = "| # | Assigned To | Action |\n| --- | --- | --- |\n" + row
        for committed in (False, True):
            with self.subTest(committed=committed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                transcript = root / "session"
                chunks = transcript / "chunks_out"
                chunks.mkdir(parents=True)
                source = "[Chair] Let's get started.\n" + PROPOSAL
                if committed:
                    source += "\n[Taylor] I'll request another meeting with the manager."
                original = json.dumps({"chunk_id": 1, "file_name": "001.md", "text": source, "start_time": 0, "end_time": 60}) + "\n"
                index = chunks / "transcript_chunks.jsonl"
                index.write_text(original, encoding="utf-8")
                calls = []

                def model_reply(**kwargs):
                    calls.append(kwargs)
                    prompt = kwargs["prompt"]
                    if "Transcript chunk:\n" in prompt:
                        return "## Topics\n- Proposal to request another meeting with the manager.\n## Action items\n" + data
                    if "write formal draft minutes" in prompt:
                        return minutes(data)
                    if "write an action-items document" in prompt:
                        return "# Action Items\n" + data
                    return "# Summary\n- Discussion of another meeting with the manager."

                with patch.object(sys, "argv", ["summarizer", str(transcript)]), patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(root / "outputs")}, clear=True), patch.object(engine, "call_ollama", side_effect=model_reply), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(engine.main(default_profile="meeting"), 0)
                output = root / "outputs" / "session"
                for filename in ("minutes-draft.md", "action-items.md"):
                    content = (output / filename).read_text(encoding="utf-8")
                    if committed:
                        self.assertIn(row, content)
                        self.assertIn("| # | Assigned To | Action |\n| --- | --- | --- |", content)
                    else:
                        self.assertNotIn(row, content)
                        self.assertIn("None noted.", content)
                        self.assertNotIn("| --- |", content)
                reduction = next(call["prompt"] for call in calls if "write formal draft minutes" in call["prompt"])
                if committed:
                    self.assertIn(row, reduction)
                else:
                    self.assertNotIn(row, reduction)
                self.assertEqual(index.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
