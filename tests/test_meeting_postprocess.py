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
from meeting_postprocess.aliases import load_aliases
from meeting_postprocess.normalization import normalize_text, prepare_text
from meeting_postprocess.qa import check_minutes
from meeting_postprocess.actions import filter_completed_request_tasks
from meeting_postprocess.motions import adjournment_announcements, correct_adjournment_roles
from meeting_postprocess.rendering import insert_recap, strip_chunk_references
from meeting_postprocess.sections import ADJOURNMENT, BUSINESS, PRE_MEETING, RECAP, prepare_chunks


def chunk(text: str, chunk_id: int = 1) -> dict:
    return {
        "chunk_id": chunk_id, "file_name": f"{chunk_id:03d}.md", "text": text,
        "speaker_span": "SPEAKER_00 -> SPEAKER_01", "chunk_type": "discussion",
        "start_time": 0.0, "end_time": 60.0,
    }


def opening_conversation_turns() -> list[dict]:
    fixture = ROOT / "tests" / "fixtures" / "opening_conversation_sections.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["turns"]


def opening_conversation_text() -> str:
    return "\n".join(f"[{turn['speaker']}] {turn['text']}" for turn in opening_conversation_turns())


def recap_interruption_turns() -> list[dict]:
    fixture = ROOT / "tests" / "fixtures" / "natural_start_recap_interruption.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["turns"]


def recap_interruption_text() -> str:
    return "\n".join(f"[{turn['speaker']}] {turn['text']}" for turn in recap_interruption_turns())


class NormalizationTests(unittest.TestCase):
    def test_artifacts_and_correct_unicode_on_same_line(self):
        source = "Mack Yard, Mackyard, Mac yard: Transport Canadaâ€™s hypodermical policy. café’s façade."
        expected = "Mac Yard, Mac Yard, Mac Yard: Transport Canada’s hypodermic policy. café’s façade."
        self.assertEqual(normalize_text(source), expected)
        self.assertEqual(normalize_text(expected), expected)

    def test_double_encoded_and_latin1_punctuation(self):
        for encoding in ("cp1252", "latin1"):
            with self.subTest(encoding=encoding):
                broken = "‘Canada’ — café".encode("utf-8").decode(encoding)
                self.assertEqual(normalize_text(broken), "‘Canada’ — café")
        broken = "Canada’s".encode("utf-8").decode("cp1252").encode("utf-8").decode("cp1252")
        self.assertEqual(normalize_text(broken), "Canada’s")

    def test_aliases_replace_whole_labels_only(self):
        self.assertEqual(
            prepare_text("[SPEAKER_01] SPEAKER_010, SPEAKER_02, XSPEAKER_01", {"SPEAKER_01": "Riley"}),
            "[Riley] SPEAKER_010, SPEAKER_02, XSPEAKER_01",
        )

    def test_alias_discovery_and_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(load_aliases(root), {})
            (root / "speaker_aliases.json").write_text('{"SPEAKER_00": "  Mórgan  "}', encoding="utf-8-sig")
            self.assertEqual(load_aliases(root), {"SPEAKER_00": "Mórgan"})
            explicit = root / "alternate.json"
            explicit.write_text('{"SPEAKER_00": "Morgan"}', encoding="utf-8")
            self.assertEqual(load_aliases(root, explicit), {"SPEAKER_00": "Morgan"})
            with self.assertRaises(OSError):
                load_aliases(root, root / "missing.json")

    def test_invalid_aliases_fail_instead_of_guessing(self):
        invalid = [[], {"Morgan": "Riley"}, {"SPEAKER_01": ""}, {"SPEAKER_01": 1},
                   {"SPEAKER_01": "SPEAKER_02"}, {"SPEAKER_01": "Morgan\nRiley"},
                   {"SPEAKER_01": "[Morgan]"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "speaker_aliases.json"
            for data in invalid:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_aliases(root)


class SectionTests(unittest.TestCase):
    def test_asr_fillers_preserve_opening_interruption_and_recap_resumption(self):
        turns = json.loads((ROOT / "tests" / "fixtures" / "asr_opening_recap_resumption.json").read_text(encoding="utf-8"))["turns"]
        source = [chunk(f"[{turn['speaker']}] {turn['text']}", index) for index, turn in enumerate(turns, 1)]
        original = json.dumps(source)
        rows = prepare_chunks(source, {})
        self.assertEqual([row["meeting_section"] for row in rows], [turn["expected"] for turn in turns])
        self.assertEqual([row["text"] for row in rows], [row["text"] for row in source])
        self.assertEqual(json.dumps(source), original)
        merged = prepare_chunks([chunk("\n".join(row["text"] for row in source))], {})
        self.assertEqual([row["meeting_section"] for row in merged], [PRE_MEETING, RECAP, BUSINESS, RECAP, BUSINESS, ADJOURNMENT])

    def test_filler_tolerant_cues_do_not_turn_task_starts_into_meeting_boundaries(self):
        for fillers in ("Uh, um, oh, ", "So uh yeah ", "Well, okay, "):
            with self.subTest(fillers=fillers):
                rows = prepare_chunks([chunk("[Morgan] The budget is approved.\n[Chair] " + fillers + "Let's get started on the next project.")], {})
                self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS])
                rows = prepare_chunks([chunk("[Morgan] Personal conversation.\n[Chair] " + fillers + "guys all set here um oh start off with the agenda.")], {})
                self.assertEqual([row["meeting_section"] for row in rows], [PRE_MEETING, BUSINESS])
                rows = prepare_chunks([chunk("[Chair] " + fillers + "the previous meeting, Taylor reported staffing concerns. Moving right along.")], {})
                self.assertEqual([row["meeting_section"] for row in rows], [RECAP, BUSINESS])

    def test_standalone_adjournment_motion_marks_close_but_hypotheticals_do_not(self):
        for cue in ("Motion to adjourn.", "Motion to adjourn the meeting.", "Okay, motion to adjourn."):
            with self.subTest(cue=cue):
                rows = prepare_chunks([chunk("[Chair] The budget is approved. " + cue + "\n[Taylor] I second it.")], {})
                self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS, ADJOURNMENT])
        for discussion in ("If we had a motion to adjourn, we could leave.", "We should discuss a future motion to adjourn.", "Motion to adjourn next week might be useful.", "Motion to adjourn?"):
            with self.subTest(discussion=discussion):
                rows = prepare_chunks([chunk("[Chair] The budget is approved. " + discussion)], {})
                self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS])

    def test_all_four_sections_inside_one_merged_turn(self):
        text = "[SPEAKER_00] Good morning everyone. I call the meeting to order. Let's review the previous meeting. We approved the OLD plan. Moving on to new business. We approved the NEW plan. The meeting is adjourned."
        rows = prepare_chunks([chunk(text)], {"SPEAKER_00": "Morgan"})
        self.assertEqual([row["meeting_section"] for row in rows], [PRE_MEETING, BUSINESS, RECAP, BUSINESS, ADJOURNMENT])
        self.assertIn("OLD plan", rows[2]["text"])
        self.assertIn("NEW plan", rows[3]["text"])
        self.assertTrue(all(row["source_chunk_id"] == 1 and row["start_time"] == 0.0 for row in rows))
        self.assertTrue(all("SPEAKER_00" not in row["text"] for row in rows))
        self.assertEqual(len({row["chunk_id"] for row in rows}), len(rows))
        # All text words survive, only the per-sentence speaker labels change.
        original_words = text.replace("[SPEAKER_00] ", "").split()
        prepared_words = " ".join(row["text"].replace("[Morgan] ", "") for row in rows).split()
        self.assertEqual(prepared_words, original_words)

    def test_state_survives_source_chunk_boundaries(self):
        rows = prepare_chunks([
            chunk("[SPEAKER_00] At the last meeting, we discussed the old budget."),
            chunk("[SPEAKER_01] It was approved. Next item on the agenda: the new budget.", 2),
        ], {})
        self.assertEqual([r["meeting_section"] for r in rows], [RECAP, RECAP, BUSINESS])

    def test_ambiguous_discussion_and_casual_motion_stay_current(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] We should revisit the previous meeting's budget. I move to adjourn. If the meeting is adjourned, can we continue tomorrow?")], {})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["meeting_section"], BUSINESS)

    def test_approval_of_previous_minutes_is_current_business(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] Let's review the minutes from our last meeting. Approval of the previous minutes. New business: the schedule.")], {})
        self.assertEqual([r["meeting_section"] for r in rows], [RECAP, BUSINESS])

    def test_greetings_during_business_do_not_become_pre_meeting(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] The budget is approved.\n[SPEAKER_01] Hello, I joined late.")], {})
        self.assertEqual([r["meeting_section"] for r in rows], [BUSINESS])

    def test_task_get_started_does_not_reclassify_earlier_business(self):
        for sentence in ("Let's get started on the next project.", "Let's get started on the investigation.", "Let's get started with the equipment check.", "We can get started on that report."):
            with self.subTest(sentence=sentence):
                rows = prepare_chunks([
                    chunk("[Chair] The budget is approved."),
                    chunk("[Morgan] " + sentence, 2),
                ], {})
                self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS, BUSINESS])
                self.assertIn("budget is approved", rows[0]["text"])

    def test_bare_and_explicit_meeting_get_started_preserve_formal_openings(self):
        for sentence in ("Okay, yeah, I guess we'll get started there.", "Let's get started.", "Let's get the meeting started.", "Let's start the meeting."):
            with self.subTest(sentence=sentence):
                rows = prepare_chunks([chunk("[Morgan] Casual conversation.\n[Chair] " + sentence + " The budget is approved.")], {})
                self.assertEqual([row["meeting_section"] for row in rows], [PRE_MEETING, BUSINESS])
                self.assertIn("Casual conversation", rows[0]["text"])
                self.assertIn("budget is approved", rows[1]["text"])

    def test_explicit_current_transition_ends_recap(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] At the last meeting, the budget was approved.\n[SPEAKER_01] Moving on to new business. We need to reconsider the budget.\n[SPEAKER_00] The deadline is Friday.")], {})
        self.assertEqual([r["meeting_section"] for r in rows], [RECAP, BUSINESS])
        self.assertIn("deadline is Friday", rows[1]["text"])

    def test_recap_persists_without_past_tense_or_with_task_wording(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] At the previous meeting, we discussed the deadline.\n[SPEAKER_01] The deadline is Friday. We need a new budget. I will send the old report.\n[SPEAKER_00] Next item on the agenda: the current budget.")], {})
        self.assertEqual([r["meeting_section"] for r in rows], [RECAP, BUSINESS])
        self.assertIn("I will send the old report", rows[0]["text"])

    def test_opening_conversation_sections_across_source_chunks(self):
        turns = opening_conversation_turns()
        chunks = [chunk(f"[{turn['speaker']}] {turn['text']}", index) for index, turn in enumerate(turns, start=1)]
        rows = prepare_chunks(chunks, {})
        self.assertEqual([row["meeting_section"] for row in rows], [turn["expected"] for turn in turns])
        for row, turn in zip(rows, turns):
            self.assertIn(turn["text"], row["text"])
            self.assertTrue(row["section_evidence"])

    def test_opening_conversation_boundaries_inside_merged_turn(self):
        turns = opening_conversation_turns()
        rows = prepare_chunks([chunk("[SPEAKER_00] " + " ".join(turn["text"] for turn in turns))], {})
        self.assertEqual([row["meeting_section"] for row in rows], [PRE_MEETING, BUSINESS, RECAP, BUSINESS, ADJOURNMENT])
        self.assertIn("Casey to add Morgan", rows[0]["text"])
        self.assertIn("proposal covers safety equipment", rows[2]["text"])
        self.assertIn("Okay, thanks a lot", rows[3]["text"])

    def test_each_informal_current_transition_ends_recap(self):
        for transition in ("Okay, thanks a lot. We'll move right along then.", "Anybody want to volunteer to go first?", "I can go first"):
            with self.subTest(transition=transition):
                rows = prepare_chunks([chunk("[SPEAKER_00] Okay, yeah, I guess we'll get started there. Do you have notes for last month's meeting? The topic is safety. " + transition)], {})
                self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS, RECAP, BUSINESS])
                self.assertIn(transition, rows[-1]["text"].replace("[SPEAKER_00] ", "").replace("\n", " "))

    def test_curly_apostrophes_and_lowercase_transcript_cues(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] i’ll call Casey to add Morgan to the list. okay, yeah, i guess we’ll get started there. do you have notes for last month’s meeting? if you wouldn’t mind doing a little recap. the deadline is Friday. okay, thanks a lot. we’ll move right along then.")], {})
        self.assertEqual([row["meeting_section"] for row in rows], [PRE_MEETING, BUSINESS, RECAP, BUSINESS])

    def test_generic_recap_request_immediately_after_start(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] Let's get started. If you wouldn't mind doing a little recap. We discussed safety. I can go first.")], {})
        self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS, RECAP, BUSINESS])

    def test_no_formal_start_keeps_mid_meeting_business_fallback(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] The current budget needs approval. I'll call Casey to add Morgan to the list.")], {})
        self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS])

    def test_generic_recap_of_current_topic_does_not_start_previous_recap(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] Let's get started. New business: current safety report. If you wouldn't mind doing a little recap. We discussed today's schedule.")], {})
        self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS])

    def test_formal_start_discussion_after_adjournment_does_not_reopen_business(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] Let's get started. New business: safety. The meeting is adjourned.\n[SPEAKER_01] Let's get started on the next project.")], {})
        self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS, ADJOURNMENT])

    def test_recap_intent_allows_current_business_interruption_across_chunks(self):
        turns = recap_interruption_turns()
        rows = prepare_chunks([chunk(f"[{turn['speaker']}] {turn['text']}", index) for index, turn in enumerate(turns, start=1)], {})
        self.assertEqual([row["meeting_section"] for row in rows], [turn["expected"] for turn in turns])
        self.assertTrue(all(row["meeting_section"] == PRE_MEETING for row in rows[:3]))
        self.assertIn("awaiting historical content", rows[4]["section_evidence"][0])
        self.assertTrue(all(row["meeting_section"] == BUSINESS for row in rows[5:8]))
        self.assertEqual(rows[8]["meeting_section"], RECAP)

    def test_natural_chair_start_excludes_prior_chatter(self):
        rows = prepare_chunks([chunk(recap_interruption_text())], {})
        self.assertEqual([row["meeting_section"] for row in rows], [PRE_MEETING, BUSINESS, RECAP, BUSINESS, RECAP, BUSINESS, ADJOURNMENT])
        self.assertIn("North Terminal", rows[0]["text"])
        self.assertIn("101", rows[0]["text"])
        for topic in ("East Terminal", "South Terminal", "brake valve", "suppression"):
            self.assertIn(topic, rows[3]["text"])
            self.assertNotIn(topic, rows[4]["text"])
        self.assertIn("Morgan reported", rows[4]["text"])

    def test_joined_chair_start_and_recap_intent_exclude_prior_chatter(self):
        text = recap_interruption_text().replace("Okay guys all set here\n[SPEAKER_02] start", "Okay guys all set here start")
        rows = prepare_chunks([chunk(text)], {})
        self.assertEqual([row["meeting_section"] for row in rows], [PRE_MEETING, RECAP, BUSINESS, RECAP, BUSINESS, ADJOURNMENT])
        self.assertIn("North Terminal", rows[0]["text"])
        self.assertIn("the freight operator", rows[0]["text"])
        self.assertIn("101", rows[0]["text"])
        self.assertIn("202", rows[0]["text"])
        self.assertIn("East Terminal", rows[2]["text"])
        self.assertIn("Morgan reported", rows[3]["text"])

    def test_conservative_natural_chair_start_variants(self):
        for cue in ("Okay guys, all set here", "Okay, everybody all set?", "start off with a recap", "we'll start with the agenda", "We'll start with a recap of the last meeting."):
            with self.subTest(cue=cue):
                rows = prepare_chunks([chunk("[SPEAKER_01] Prior conversation.\n[SPEAKER_02] " + cue)], {})
                self.assertEqual(rows[0]["meeting_section"], PRE_MEETING)
                self.assertNotEqual(rows[-1]["meeting_section"], PRE_MEETING)

    def test_ordinary_start_and_readiness_uses_do_not_create_boundaries(self):
        for text in ("I'll ask our team to start following this procedure.", "We'll start with the battery on unit 101.", "The locomotives start off with a battery check.", "Okay guys, all set for the North Terminal job?", "The employee asked everybody to start early."):
            with self.subTest(text=text):
                rows = prepare_chunks([chunk("[SPEAKER_01] Current business.\n[SPEAKER_02] " + text)], {})
                self.assertEqual([row["meeting_section"] for row in rows], [BUSINESS])

    def test_retrospective_cue_resumes_recap(self):
        text = "[SPEAKER_02] At the last meeting, Morgan reported staffing.\n[SPEAKER_03] Yesterday we had a red flag on the service track.\n[SPEAKER_04] The brake valve needs manual handling.\n[SPEAKER_02] So, yeah, the last meeting, Morgan reported staffing again. Moving right along."
        rows = prepare_chunks([chunk(text)], {})
        self.assertEqual([row["meeting_section"] for row in rows], [RECAP, BUSINESS, RECAP, BUSINESS])
        self.assertIn("red flag", rows[1]["text"])
        self.assertIn("brake valve", rows[1]["text"])

    def test_announcing_recap_does_not_force_an_unmarked_interruption_into_history(self):
        text = "[SPEAKER_02] Let's recap the previous meeting.\n[SPEAKER_03] The brake valve needs suppression handling.\n[SPEAKER_04] We discussed the brake valve during a recovery.\n[SPEAKER_02] The last meeting, Morgan reported staffing."
        rows = prepare_chunks([chunk(text)], {})
        self.assertEqual([row["meeting_section"] for row in rows], [RECAP, BUSINESS, RECAP])
        self.assertIn("We discussed the brake valve", rows[1]["text"])


class QATests(unittest.TestCase):
    def test_each_required_defect_is_flagged_with_a_line(self):
        findings = check_minutes("# Draft Minutes\n-SPEAKER_00 discussed Mackyard.\n- Transport Canadaâ€™s rules.\n- Moved by: Unknown; Seconded by: Unknown.")
        self.assertEqual({f.code for f in findings}, {"unresolved_speaker", "malformed_bullet", "mojibake", "mac_yard_spelling", "questionable_motion"})
        self.assertTrue(all(f.line >= 2 and f.excerpt for f in findings))

    def test_valid_markdown_and_explicit_motion_evidence(self):
        source = "[Morgan] I move to approve the budget.\n[Riley] I second the motion."
        minutes = "# Draft Minutes\n\n- Mac Yard budget approved.\n  - Transport Canada’s policy.\n- **Moved by:** Morgan; **Seconded by:** Riley.\n\n---\n\n*Emphasized paragraph*\n\n```text\n-not a markdown bullet\n```"
        self.assertEqual(check_minutes(minutes, source), [])

    def test_tentative_same_person_and_unsupported_attributions(self):
        source = "[Morgan] We discussed the budget.\n[Riley] I was present."
        minutes = "- Moved by: Morgan\n  Seconder: Morgan\n\n- Possibly moved by Riley."
        findings = check_minutes(minutes, source)
        self.assertTrue(any("same person" in f.message for f in findings))
        self.assertTrue(any("no explicit role evidence" in f.message for f in findings))
        self.assertTrue(any("tentative" in f.message for f in findings))

    def test_explicit_named_roles_and_no_guess_from_attendance(self):
        source = "[Chair] Moved by Morgan; seconded by Riley.\n[Riley] I second the question about attendance."
        self.assertEqual(check_minutes("- Mover: Morgan; Seconder: Riley.", source), [])
        findings = check_minutes("- Seconder: Riley.", "[Riley] I second the question about attendance.")
        self.assertEqual(len(findings), 1)

    def test_malformed_bullets(self):
        for text in ("-bad bullet", "+bad bullet", "*bad bullet", "- - doubled", "• non-markdown", "1.missing space", "- "):
            with self.subTest(text=text):
                self.assertIn("malformed_bullet", {f.code for f in check_minutes(text)})

    def test_recap_heading_is_owned_by_pipeline(self):
        result = insert_recap("# Draft Minutes\n\n## Overview\nCurrent business.", "# A model title\n- Earlier business.")
        self.assertEqual(result.count("## Recap of Previous Meeting"), 1)
        self.assertNotIn("A model title", result)
        self.assertLess(result.index("Earlier business"), result.index("## Overview"))

    def test_announced_adjournment_roles_ignore_discourse_particle(self):
        source = "[SPEAKER_02] I'll say a motion by Casey, seconded by Taylor there."
        minutes = "# Draft Minutes\n- Motion to adjourn moved by Casey, seconded by Taylor."
        self.assertEqual(check_minutes(minutes, source), [])
        for particle in ("there", "then", "there okay"):
            with self.subTest(particle=particle):
                self.assertEqual(check_minutes(minutes, source.replace("there.", particle + ".")), [])


class FinalOutputTests(unittest.TestCase):
    def test_summary_adjournment_exception_outcomes_become_neutral(self):
        source = "[Chair] Motion to adjourn. Moved by Taylor, seconded by Casey.\n[Chair] The meeting is adjourned."
        for outcome in ("passed", "carried", "approved", "adopted", "accepted", "ratified"):
            with self.subTest(outcome=outcome):
                content = f"No formal motions were {outcome} other than the motion to adjourn."
                self.assertEqual(correct_adjournment_roles(content, source), "The only formal motion recorded was the motion to adjourn.")

    def test_adjournment_prose_cleanup_preserves_other_motion_outcomes_and_ending(self):
        source = "[Chair] Motion to adjourn. Moved by Taylor, seconded by Casey.\n[Chair] The meeting is adjourned."
        for claim in ("The motion to adjourn was passed unanimously.", "The only formal motion passed was the motion to adjourn.", "The committee adopted the adjournment motion.", "The approved motion to adjourn ended the meeting.", "The adjournment motion passed; the meeting ended."):
            with self.subTest(claim=claim):
                content = "The budget motion passed. " + claim
                cleaned = correct_adjournment_roles(content, source)
                self.assertTrue(cleaned.startswith("The budget motion passed. "))
                self.assertNotRegex(cleaned[len("The budget motion passed. "):], r"\b(?:passed|carried|approved|adopted)\b")
                if "ended" in claim:
                    self.assertIn("ended", cleaned)
                if "meeting ended" in claim:
                    self.assertIn("meeting ended", cleaned)
        content = "No formal motions were passed other than the motion to adjourn. If the budget passes later, funding can proceed."
        self.assertEqual(correct_adjournment_roles(content, source), "The only formal motion recorded was the motion to adjourn. If the budget passes later, funding can proceed.")
        hypothetical = "If the motion to adjourn passed, the meeting would end."
        self.assertEqual(correct_adjournment_roles(hypothetical, source), hypothetical)

    def test_explicit_source_outcomes_are_preserved_in_summary_prose(self):
        source = "[Chair] Motion to adjourn. Moved by Taylor, seconded by Casey."
        for outcome in ("Carried", "Passed", "Approved", "Adopted", "Accepted", "Ratified", "Defeated", "Withdrawn", "Tabled", "Not carried"):
            with self.subTest(outcome=outcome):
                supported = source + f"\n[Chair] The motion was {outcome.lower()}."
                self.assertEqual(adjournment_announcements(supported)[0].outcome, outcome)
                content = f"The motion to adjourn was {outcome.lower()}."
                self.assertEqual(correct_adjournment_roles(content, supported), content)

    def test_adjournment_prose_remains_untouched_without_unique_source_record(self):
        claim = "No formal motions were passed other than the motion to adjourn."
        source = "[Chair] Motion to adjourn. Moved by Taylor, seconded by Casey."
        for evidence in ("", source + "\n[Chair] Motion to adjourn. Moved by Morgan, seconded by Riley."):
            with self.subTest(evidence=evidence):
                self.assertEqual(correct_adjournment_roles(claim, evidence), claim)
        content = "# Summary\n## Recap of Previous Meeting\n" + claim + "\n## Current Business\n" + claim
        cleaned = correct_adjournment_roles(content, source)
        self.assertIn(claim, cleaned.split("## Current Business", 1)[0])
        self.assertNotIn(claim, cleaned.split("## Current Business", 1)[1])

    def test_bold_adjournment_with_outcome_continuation_uses_source_only(self):
        content = "- **Motion to adjourn the meeting:** Moved by Taylor, seconded by Casey.\n  Carried; the meeting formally adjourned."
        source = "[Chair] Motion to adjourn. Moved by Taylor, seconded by Casey."
        for outcome in (None, "Carried", "Not carried", "Defeated", "Withdrawn", "Tabled"):
            with self.subTest(outcome=outcome):
                evidence = source + (f"\n[Chair] {outcome}." if outcome else "")
                expected = "- Motion to adjourn moved by Taylor, seconded by Casey." + (f" {outcome}." if outcome else "")
                self.assertEqual(correct_adjournment_roles(content, evidence), expected)

    def test_adjournment_titles_and_equivalent_model_outcomes_are_scoped(self):
        source = "[Chair] Motion to adjourn. Moved by Taylor, seconded by Casey."
        for title in ("Motion to adjourn", "Motion to adjourn the meeting", "Adjournment motion", "__Adjournment motion__", "**Motion to adjourn the meeting:**", "Adjournment –"):
            for outcome in ("Carried; the meeting formally adjourned.", "The motion passed.", "Approved.", "The motion was adopted.", "Accepted.", "Ratified.", "**Carried unanimously.**"):
                with self.subTest(title=title, outcome=outcome):
                    content = title + " Moved by Taylor, seconded by Casey.\n\n  " + outcome + "\nBudget motion carried.\nOther business stays."
                    self.assertEqual(correct_adjournment_roles(content, source), "Motion to adjourn moved by Taylor, seconded by Casey.\n\nBudget motion carried.\nOther business stays.")

    def test_explicit_carried_statement_with_semicolon_is_supported(self):
        for source in ("[Chair] Motion to adjourn. Moved by Taylor, seconded by Casey; Carried.", "[Chair] Motion to adjourn. Moved by Taylor, seconded by Casey.\n[Chair] Carried; the meeting is adjourned."):
            with self.subTest(source=source):
                content = "- **Motion to adjourn the meeting:** Moved by Taylor, seconded by Casey.\n  Carried; the meeting formally adjourned."
                self.assertEqual(correct_adjournment_roles(content, source), "- Motion to adjourn moved by Taylor, seconded by Casey. Carried.")

    def test_acknowledgment_turns_do_not_hide_adjournment_source_roles_or_outcome(self):
        source = "[Chair] Motion to adjourn.\n[Chair] Okay.\n[Chair] Yeah.\n[Chair] Uh, I'll say a motion by Taylor, seconded by Casey there."
        content = "- **Motion to adjourn the meeting:** Moved by Taylor, seconded by Casey.\n  Carried; the meeting formally adjourned."
        for outcome in (None, "Carried"):
            with self.subTest(outcome=outcome):
                evidence = source + ("\n[Chair] Okay.\n[Chair] Uh.\n[Chair] Yeah.\n[Chair] Carried." if outcome else "")
                original = evidence
                announcements = adjournment_announcements(evidence)
                self.assertEqual(len(announcements), 1)
                self.assertEqual(announcements[0].outcome, outcome)
                self.assertEqual(correct_adjournment_roles(content, evidence), "- Motion to adjourn moved by Taylor, seconded by Casey." + (" Carried." if outcome else ""))
                self.assertEqual(evidence, original)

    def test_chunk_citations_removed_without_losing_business_parentheses(self):
        content = "# Draft Minutes\n- Budget approved (Chunk 2.2).\n- Size confirmed [Source chunk ID: 3.1].\n- Contact Casey Morgan, from Chunk 4.\n- Safety review (includes workers; Chunk 5.2).\n- Report ready (source_chunk_id=8.2).\n# Chunk 7.1\n- File: 007-discussion.md\n"
        cleaned = strip_chunk_references(content, ["007-discussion.md"])
        self.assertNotIn("Chunk", cleaned)
        self.assertNotIn("chunk", cleaned)
        self.assertNotIn("007-discussion.md", cleaned)
        self.assertIn("Budget approved.", cleaned)
        self.assertIn("includes workers", cleaned)
        self.assertIn("Contact Casey Morgan.", cleaned)
        self.assertIn("Report ready.", cleaned)
        self.assertFalse(any(f.code == "malformed_bullet" for f in check_minutes(cleaned)))

    def test_completed_information_requests_do_not_assign_recipients(self):
        content = "# Action Items\n- Morgan – contact Casey Morgan for contact detail.\n- Casey and Riley – provide contact details."
        for source in (
            "[Chair] I've already messaged Casey and Riley for their contact details.\n[Morgan] I'll contact Casey Morgan for contact detail.",
            "[Chair] Casey and Riley have already been contacted for their contact details.\n[Morgan] I'll contact Casey Morgan for contact detail.",
            "[Chair] We already requested information from Casey and Riley about contact details.",
        ):
            with self.subTest(source=source):
                cleaned = filter_completed_request_tasks(content, source)
                self.assertIn("Morgan – contact Casey Morgan", cleaned)
                self.assertNotIn("Casey and Riley", cleaned)

    def test_recipient_explicit_later_commitment_is_retained(self):
        source = "[Chair] I've already messaged Casey and Riley for contact details.\n[Chair] Casey and Riley will provide contact details tomorrow."
        content = "# Action Items\n- Casey and Riley – provide contact details tomorrow."
        self.assertEqual(filter_completed_request_tasks(content, source), content)
        source = "[Chair] I've already messaged Casey for contact detail.\n[Casey] I'll send my contact detail tomorrow."
        self.assertNotIn("Casey and Riley", filter_completed_request_tasks(content, source))
        content = "# Action Items\n- Casey – provide contact detail tomorrow."
        self.assertEqual(filter_completed_request_tasks(content, source), content)

    def test_future_outreach_and_unrelated_tasks_are_retained(self):
        source = "[Morgan] I'll message Casey and Riley for their contact details."
        content = "# Action Items\n- Morgan – message Casey and Riley for contact details."
        self.assertEqual(filter_completed_request_tasks(content, source), content)
        source = "[Chair] I've already contacted Casey for contact detail."
        content = "# Action Items\n- Casey – send the meeting agenda."
        self.assertEqual(filter_completed_request_tasks(content, source), content)

    def test_negated_or_unidentified_outreach_does_not_remove_tasks(self):
        content = "# Action Items\n- Casey – provide contact detail."
        for source in ("[Chair] I have not messaged Casey for contact detail.", "[Chair] If I messaged Casey for contact detail, what would happen?", "[Chair] I've already messaged them for contact details."):
            with self.subTest(source=source):
                self.assertEqual(filter_completed_request_tasks(content, source), content)

    def test_completed_contacts_stay_as_discussion_in_minutes(self):
        source = "[Chair] I've already contacted Casey and Riley for contact details."
        content = "# Draft Minutes\n## Topics Discussed\n- Casey and Riley were contacted for contact details.\n## Action Items\n- Casey and Riley – provide contact details.\n## Important Notes\nNone noted."
        cleaned = filter_completed_request_tasks(content, source, action_sections_only=True)
        self.assertIn("were contacted", cleaned)
        self.assertNotIn("– provide", cleaned)
        self.assertIn("## Action Items\nNone noted.", cleaned)

    def test_only_unsupported_actions_produce_standard_empty_output(self):
        source = "[Chair] I've already contacted Casey for contact detail."
        self.assertEqual(filter_completed_request_tasks("# Action Items\n- Casey – provide contact detail.", source), "# Action Items\nNo clear action items identified.")

    def test_announced_adjournment_roles_override_announcer_label(self):
        source = "[SPEAKER_07] We need a motion to adjourn.\n[SPEAKER_07] Motion by Taylor, seconded by Casey.\n[SPEAKER_07] Carried."
        content = "# Draft Minutes\n- Motion to adjourn moved by Taylor, seconded by SPEAKER_07. Carried."
        cleaned = correct_adjournment_roles(content, source)
        self.assertIn("Motion to adjourn moved by Taylor, seconded by Casey. Carried.", cleaned)
        self.assertNotIn("SPEAKER_07", cleaned)
        self.assertEqual(check_minutes(cleaned, source), [])

    def test_multiline_roles_are_corrected_and_named_announcer_is_not_seconder(self):
        source = "[Riley] Motion to adjourn: motion by Taylor, seconded by Casey. Carried."
        content = "- Motion to adjourn\n  Mover: Taylor\n  Seconder: Riley\n  Carried.\n- Another motion moved by Morgan, seconded by Riley."
        cleaned = correct_adjournment_roles(content, source)
        self.assertEqual(cleaned, "- Motion to adjourn moved by Taylor, seconded by Casey. Carried.\n- Another motion moved by Morgan, seconded by Riley.")

    def test_adjournment_outcome_is_not_invented(self):
        for source in (
            "[Chair] Motion to adjourn, motion by Taylor, seconded by Casey.",
            "[Chair] Motion to adjourn, motion by Taylor, seconded by Casey. Not carried.",
        ):
            with self.subTest(source=source):
                cleaned = correct_adjournment_roles("- Motion to adjourn seconded by SPEAKER_07. Carried.", source)
                self.assertIn("seconded by Casey.", cleaned)
                self.assertNotIn("Carried", cleaned)

    def test_current_adjournment_roles_do_not_rewrite_historical_recap(self):
        source = "[Chair] Motion to adjourn, motion by Taylor, seconded by Casey. Carried."
        content = "# Draft Minutes\n## Recap of Previous Meeting\n- Motion to adjourn moved by Morgan, seconded by Riley.\n## Motions / Proposals Mentioned\n- Motion to adjourn seconded by SPEAKER_07."
        cleaned = correct_adjournment_roles(content, source)
        self.assertIn("moved by Morgan, seconded by Riley", cleaned)
        self.assertIn("moved by Taylor, seconded by Casey. Carried.", cleaned)

    def test_known_negative_motion_outcome_is_preserved(self):
        source = "[Chair] Motion to adjourn, motion by Taylor, seconded by Casey. Defeated."
        cleaned = correct_adjournment_roles("- Motion to adjourn moved by Taylor, seconded by SPEAKER_07. Carried.", source)
        self.assertIn("seconded by Casey. Defeated.", cleaned)
        self.assertNotIn("Carried", cleaned)

    def test_question_about_carried_outcome_does_not_establish_it(self):
        source = "[Chair] Motion to adjourn, motion by Taylor, seconded by Casey. Carried?"
        cleaned = correct_adjournment_roles("- Motion to adjourn moved by Taylor, seconded by SPEAKER_07. Carried.", source)
        self.assertNotIn("Carried", cleaned)

    def test_hypothetical_or_conflicting_roles_are_not_guessed(self):
        self.assertEqual(adjournment_announcements("[Chair] If we had a motion to adjourn, motion by Taylor, seconded by Casey."), [])
        source = "[Chair] Motion to adjourn, motion by Taylor, seconded by Casey.\n[Chair] Motion to adjourn, motion by Morgan, seconded by Riley."
        content = "- Motion to adjourn seconded by SPEAKER_07."
        self.assertEqual(correct_adjournment_roles(content, source), content)

    def test_collective_proposal_does_not_assign_addressee(self):
        source = "[Chair] Taylor, I think we have to request another meeting with the manager."
        candidate = "# Action Items\n- Taylor – request another meeting with the manager."
        self.assertEqual(filter_completed_request_tasks(candidate, source), "# Action Items\nNo clear action items identified.")
        minutes = "# Draft Minutes\n## Topics Discussed\n- Proposal to request another meeting with the manager.\n## Action Items\n- Taylor – request another meeting with the manager."
        cleaned = filter_completed_request_tasks(minutes, source, action_sections_only=True)
        self.assertIn("Proposal to request", cleaned)
        self.assertNotIn("Taylor – request", cleaned)

    def test_collective_hedging_variants_remain_proposals(self):
        for proposal in ("I think we should request another meeting with the manager.", "We probably need to request another meeting with the manager.", "I think we have to request another meeting with the manager."):
            with self.subTest(proposal=proposal):
                source = "[Chair] Taylor, " + proposal
                candidate = "# Action Items\n- Taylor – request another meeting with the manager."
                self.assertNotIn("Taylor – request", filter_completed_request_tasks(candidate, source))

    def test_proposal_becomes_action_only_with_supported_owner_agreement_or_assignment(self):
        proposal = "[Chair] Taylor, I think we have to request another meeting with the manager."
        candidate = "# Action Items\n- Taylor – request another meeting with the manager."
        for followup in ("[Taylor] I'll request another meeting with the manager.", "[Taylor] Yes, I'll do it.", "[Chair] Taylor will request another meeting with the manager.", "[Chair] Taylor, please request another meeting with the manager.", "[Chair] I assign Taylor to request another meeting with the manager."):
            with self.subTest(followup=followup):
                self.assertEqual(filter_completed_request_tasks(candidate, proposal + "\n" + followup), candidate)
        self.assertNotIn("Taylor – request", filter_completed_request_tasks(candidate, proposal + "\n[Morgan] Yes, I'll request another meeting with the manager."))
        self.assertNotIn("Taylor – request", filter_completed_request_tasks(candidate, proposal + "\n[Taylor] Yes, but I won't request that meeting."))

    def test_explicit_commitments_survive_unrelated_collective_proposals(self):
        commitments = ("I'll approve access for both members", "I'll follow up with the employee", "I'll ask our team to start following this procedure", "I'll contact the coordinator", "I'll try to identify relevant references ... and we'll share the notes")
        proposal = "[Chair] Taylor, I think we have to request another meeting with the manager."
        source = proposal + "\n" + "\n".join("[Morgan] " + text for text in commitments)
        content = "# Action Items\n" + "\n".join("- Morgan – " + text for text in commitments)
        self.assertEqual(filter_completed_request_tasks(content, source), content)

    def test_announced_roles_normalize_without_inventing_outcome(self):
        source = "[SPEAKER_02] Can we have a motion to adjourn?\n[SPEAKER_02] I'll say a motion by Casey, seconded by Taylor there."
        content = "- Motion to adjourn moved by Casey, seconded by SPEAKER_02. Carried."
        cleaned = correct_adjournment_roles(content, source)
        self.assertEqual(cleaned, "- Motion to adjourn moved by Casey, seconded by Taylor.")
        self.assertEqual(check_minutes(cleaned, source), [])

    def test_honorific_split_does_not_hide_explicit_owner_commitment(self):
        raw = "[Chair] Taylor, I think we have to request another meeting with Mr. Casey.\n[Taylor] I'll request another meeting with Mr. Casey."
        prepared = "\n".join(row["text"] for row in prepare_chunks([chunk(raw)], {}))
        content = "# Action Items\n- Taylor – request another meeting with Mr. Casey."
        self.assertEqual(filter_completed_request_tasks(content, prepared), content)
        content = "# Action Items\n- Owner: Taylor; Task: request another meeting with Mr. Casey."
        self.assertEqual(filter_completed_request_tasks(content, prepared), content)


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, root: Path, text: str, options: list[str] | None = None, aliases=None, profile="meeting", environment=None, default_profile=None, reduce_echo=False, model_reply=None):
        transcript = root / "session-123"
        chunk_dir = transcript / "chunks_out"
        chunk_dir.mkdir(parents=True)
        index = chunk_dir / "transcript_chunks.jsonl"
        original = json.dumps(chunk(text), ensure_ascii=False) + "\n"
        index.write_text(original, encoding="utf-8")
        if aliases is not None:
            (transcript / "speaker_aliases.json").write_text(json.dumps(aliases), encoding="utf-8")
        calls = []

        def fake_ollama(**kwargs):
            calls.append(kwargs)
            prompt = kwargs["prompt"]
            if model_reply is not None:
                return model_reply(kwargs)
            if "Transcript chunk:\n" in prompt:
                transcript_text = prompt.split("Transcript chunk:\n", 1)[1].split("\nWrite concise markdown", 1)[0]
                return "## Topics\n- " + transcript_text.strip()
            if reduce_echo:
                # Reflect supplied summaries to make leaked source content
                # observable in the generated files, without a real LLM.
                marker = "Previous-meeting chunk summaries:\n" if "Summarize only the supplied historical recap" in prompt else "Chunk summaries:\n"
                heading = "## Overview\n\n" if "write formal draft minutes" in prompt else ""
                return "# Generated Output\n\n" + heading + prompt.split(marker, 1)[1]
            if "Summarize only the supplied historical recap" in prompt:
                return "# Unexpected model title\n- Previously approved OLD Mac yard plan."
            if "write formal draft minutes" in prompt:
                return "# Draft Minutes\n\n## Topics Discussed\n- Morgan discussed Mackyard and Transport Canadaâ€™s hypodermical guidance.\n- SPEAKER_09 offered comments.\n-bad bullet\n- Moved by: Morgan; Seconded by: Riley."
            return "# Other output\n- SPEAKER_00 Mack Yard."

        argv = ["summarizer", str(transcript)]
        if profile is not None:
            argv.extend(["--profile", profile])
        argv.extend(options or [])
        env = {"MEETING_SUMMARIES_ROOT": str(root / "outputs"), "LESSON_SUMMARIES_ROOT": str(root / "outputs"), "MEETING_KEEP_RECAP": "0"}
        env.update(environment or {})
        with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True), patch.object(engine, "call_ollama", side_effect=fake_ollama), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = engine.main(default_profile=default_profile)
        self.assertEqual(index.read_text(encoding="utf-8"), original)
        return status, calls, root / "outputs" / transcript.name

    def test_summary_adjournment_outcome_synonym_is_cleaned_in_final_markdown(self):
        source = "[Chair] Let's get started. Motion to adjourn. Moved by Taylor, seconded by Casey. The meeting is adjourned."
        claim = "No formal motions were passed other than the motion to adjourn."
        def response(call):
            if "Transcript chunk:\n" in call["prompt"]:
                return "## Topics\n- Adjournment motion announced."
            return "# Generated document\n" + claim
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), source, model_reply=response)
            self.assertEqual(status, 0)
            summary = (output / "summary.md").read_text(encoding="utf-8")
            self.assertNotIn(claim, summary)
            self.assertIn("The only formal motion recorded was the motion to adjourn.", summary)

    def test_actor_separated_commitment_evidence_prevents_inferred_delivery_recipient(self):
        for recipient in (None, "Casey"):
            with self.subTest(recipient=recipient), tempfile.TemporaryDirectory() as directory:
                own = "I'll send the package " + ("off" if recipient is None else "to " + recipient)
                source = "[Chair] Let's get started.\n[Taylor] " + own + ", and if the office approves it, they will send it to Morgan."
                def response(call):
                    prompt = call["prompt"]
                    if "Transcript chunk:\n" in prompt:
                        return "## Topics\n- The office will later send the package to Morgan.\n## Action Items\n- Taylor – Send the package to Morgan."
                    if "write an action-items document" in prompt or "write formal draft minutes" in prompt:
                        evidence = prompt.split("Source-backed future commitment evidence (current meeting only):\n", 1)[1].split("\nUse this supplemental source evidence", 1)[0]
                        self.assertEqual(evidence, "- [Taylor] " + own)
                        self.assertIn("leave an unstated recipient unstated", prompt)
                        self.assertIn("even if the map summary merges those steps", prompt)
                        return "# Generated document\n## Action Items\n- Taylor – Send the package " + ("off." if recipient is None else "to " + recipient + ".")
                    return "# Summary\n- A package will be sent onward."
                status, calls, output = self.run_pipeline(Path(directory), source, model_reply=response)
                self.assertEqual(status, 0)
                self.assertEqual(len(calls), sum("Transcript chunk:\n" in call["prompt"] for call in calls) + 3)
                for filename in ("action-items.md", "minutes-draft.md"):
                    content = (output / filename).read_text(encoding="utf-8")
                    self.assertNotIn("Morgan", content)
                    self.assertIn("Send the package " + ("off." if recipient is None else "to " + recipient + "."), content)
                    self.assertNotIn("Chunk", content)

    def test_asr_opening_excludes_chatter_from_every_public_reduction(self):
        turns = json.loads((ROOT / "tests" / "fixtures" / "asr_opening_recap_resumption.json").read_text(encoding="utf-8"))["turns"]
        source = "\n".join(f"[{turn['speaker']}] {turn['text']}" for turn in turns)
        for keep_recap in (False, True):
            with self.subTest(keep_recap=keep_recap), tempfile.TemporaryDirectory() as directory:
                status, calls, output = self.run_pipeline(Path(directory), source, ["--keep-recap"] if keep_recap else [], reduce_echo=True)
                self.assertEqual(status, 0)
                self.assertTrue(any("PRIVATE_CHATTER_SENTINEL" in call["prompt"] for call in calls if "Transcript chunk:\n" in call["prompt"]))
                for call in calls:
                    if "Transcript chunk:\n" not in call["prompt"]:
                        self.assertNotIn("PRIVATE_CHATTER_SENTINEL", call["prompt"])
                for filename in ("summary.md", "minutes-draft.md", "action-items.md"):
                    self.assertNotIn("PRIVATE_CHATTER_SENTINEL", (output / filename).read_text(encoding="utf-8"))
                self.assertEqual([row["meeting_section"] for row in engine.load_jsonl(output / "meeting_sections.jsonl")], [PRE_MEETING, RECAP, BUSINESS, RECAP, BUSINESS, ADJOURNMENT])

    def test_current_commitments_omitted_by_map_are_available_to_action_reductions(self):
        source = ("[SPEAKER_03] I'll contact the PRE_SENTINEL coordinator.\n"
                  "[Chair] Let's get started.\n"
                  "[SPEAKER_03] At the last meeting, we discussed staffing. I'll send the RECAP_SENTINEL report.\n"
                  "[Chair] Moving right along.\n"
                  "[SPEAKER_03] I'll go first. I'll make it short. I think I should send the TENTATIVE_SENTINEL report.\n"
                  "[SPEAKER_03] When I get back, I'm going to contact the CURRENT_SENTINEL coordinator.\n"
                  "[SPEAKER_03] I'll try to isolate the QUALIFIED_SENTINEL references.\n"
                  "[SPEAKER_03] Redact the following. I'll send the REDACTED_SENTINEL report. End redaction.\n"
                  "[Chair] Motion to adjourn.")
        def omit_map_actions(call):
            if "Transcript chunk:\n" in call["prompt"]:
                return "## Topics\n- Staffing discussion.\n## Action Items\nNone noted."
            return "# Generated document\nNo clear action items identified."
        for aliases, speaker in ((None, "SPEAKER_03"), ({"SPEAKER_03": "Taylor"}, "Taylor")):
            with self.subTest(aliases=aliases), tempfile.TemporaryDirectory() as directory:
                status, calls, output = self.run_pipeline(Path(directory), source, aliases=aliases, model_reply=omit_map_actions)
                self.assertEqual(status, 0)
                maps = [call for call in calls if "Transcript chunk:\n" in call["prompt"]]
                self.assertEqual(len(calls), len(maps) + 3)
                self.assertNotIn("CURRENT_SENTINEL", (output / "chunk_summaries.jsonl").read_text(encoding="utf-8"))
                for instruction in ("write an action-items document", "write formal draft minutes"):
                    prompt = next(call["prompt"] for call in calls if instruction in call["prompt"])
                    evidence = prompt.split("Source-backed future commitment evidence (current meeting only):\n", 1)[1]
                    self.assertIn(f"[{speaker}] When I get back, I'm going to contact the CURRENT_SENTINEL coordinator.", evidence)
                    self.assertIn(f"[{speaker}] I'll try to isolate the QUALIFIED_SENTINEL references.", evidence)
                    for excluded in ("PRE_SENTINEL", "RECAP_SENTINEL", "TENTATIVE_SENTINEL", "REDACTED_SENTINEL", "I'll go first", "I'll make it short"):
                        self.assertNotIn(excluded, prompt)
                summary = next(call["prompt"] for call in calls if "write a concise executive summary" in call["prompt"])
                self.assertNotIn("Source-backed future commitment evidence", summary)
                self.assertTrue(all("REDACTED_SENTINEL" not in call["prompt"] for call in calls))
                for filename in ("action-items.md", "minutes-draft.md"):
                    self.assertNotIn("CURRENT_SENTINEL", (output / filename).read_text(encoding="utf-8"))

    def test_public_reductions_exclude_chatter_and_summary_keeps_historical_recap(self):
        source = ("[Morgan] PERSONAL_CHATTER_SENTINEL plans for vacation.\n"
                  "[Chair] Let's get started.\n"
                  "[Taylor] At the last meeting, HISTORICAL_RECAP_SENTINEL was approved.\n"
                  "[Chair] Moving on to new business. CURRENT_BUSINESS_SENTINEL is ready. The meeting is adjourned.")
        def echo_reply(call):
            prompt = call["prompt"]
            if "Transcript chunk:\n" in prompt:
                return "## Topics\n- " + prompt.split("Transcript chunk:\n", 1)[1].split("\nWrite concise markdown", 1)[0]
            marker = "Previous-meeting chunk summaries:\n" if "Summarize only the supplied historical recap" in prompt else "Chunk summaries:\n"
            heading = "# Generated Output\n\n## Overview\n\n"
            if "write a concise executive summary" in prompt:
                heading += "Recap of Previous Meeting: Historical context.\n\n"
            return heading + prompt.split(marker, 1)[1]
        for keep_recap in (False, True):
            with self.subTest(keep_recap=keep_recap), tempfile.TemporaryDirectory() as directory:
                status, calls, output = self.run_pipeline(Path(directory), source, ["--keep-recap"] if keep_recap else [], model_reply=echo_reply)
                self.assertEqual(status, 0)
                map_prompts = [call["prompt"] for call in calls if "Transcript chunk:\n" in call["prompt"]]
                self.assertTrue(any("PERSONAL_CHATTER_SENTINEL" in prompt for prompt in map_prompts))
                for instruction in ("write a concise executive summary", "write an action-items document", "write formal draft minutes"):
                    prompt = next(call["prompt"] for call in calls if instruction in call["prompt"])
                    self.assertNotIn("PERSONAL_CHATTER_SENTINEL", prompt)
                    self.assertNotIn("Meeting section: pre_meeting_chatter", prompt)
                    self.assertIn("CURRENT_BUSINESS_SENTINEL", prompt)
                    if instruction == "write a concise executive summary":
                        self.assertIn("HISTORICAL_RECAP_SENTINEL", prompt)
                        self.assertIn("previous_meeting_recap", prompt)
                        self.assertIn("historical context", prompt)
                    else:
                        self.assertNotIn("HISTORICAL_RECAP_SENTINEL", prompt)
                for filename in ("summary.md", "action-items.md", "minutes-draft.md"):
                    content = (output / filename).read_text(encoding="utf-8")
                    self.assertNotIn("PERSONAL_CHATTER_SENTINEL", content)
                summary = (output / "summary.md").read_text(encoding="utf-8")
                self.assertIn("HISTORICAL_RECAP_SENTINEL", summary)
                self.assertIn("Recap of Previous Meeting: Historical context.", summary)

    def test_default_minutes_exclude_chatter_and_recap_and_run_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), "[SPEAKER_00] Good morning. Let's review the previous meeting. OLD plan approved. Moving on to new business. NEW Mac yard plan discussed. The meeting is adjourned.", aliases={"SPEAKER_00": "Morgan"})
            self.assertEqual(status, 0)
            minutes_prompt = next(c["prompt"] for c in calls if "write formal draft minutes" in c["prompt"])
            self.assertNotIn("OLD plan", minutes_prompt)
            self.assertNotIn("Good morning", minutes_prompt)
            self.assertIn("NEW Mac Yard plan", minutes_prompt)
            self.assertNotIn("[SPEAKER_00]", minutes_prompt)
            minutes = (output / "minutes-draft.md").read_text(encoding="utf-8")
            self.assertNotIn("Recap of Previous Meeting", minutes)
            self.assertIn("Mac Yard and Transport Canada’s hypodermic", minutes)
            report = json.loads((output / "minutes-qa.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "needs_review")
            self.assertEqual({f["code"] for f in report["findings"]}, {"unresolved_speaker", "malformed_bullet", "questionable_motion"})
            self.assertTrue((output / "minutes-qa.md").exists())
            sections = engine.load_jsonl(output / "meeting_sections.jsonl")
            self.assertEqual({r["meeting_section"] for r in sections}, {PRE_MEETING, RECAP, BUSINESS, ADJOURNMENT})

    def test_keep_recap_uses_separate_historical_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), "[SPEAKER_00] At the last meeting, OLD plan was approved. Moving on to new business. NEW plan discussed.", ["--keep-recap"])
            self.assertEqual(status, 0)
            recap_prompt = next(c["prompt"] for c in calls if "Summarize only the supplied historical recap" in c["prompt"])
            self.assertIn("OLD plan", recap_prompt)
            self.assertNotIn("NEW plan", recap_prompt)
            minutes = (output / "minutes-draft.md").read_text(encoding="utf-8")
            self.assertIn("## Recap of Previous Meeting\n\n- Previously approved OLD Mac Yard plan.", minutes)

    def test_action_items_use_same_current_sections_as_minutes(self):
        text = "[SPEAKER_00] Good morning everyone. Let's review the previous meeting. OLD plan approved. Moving on to new business. NEW plan discussed. The meeting is adjourned. Riley will circulate the notes tomorrow."
        for options in ([], ["--keep-recap"]):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                status, calls, _ = self.run_pipeline(Path(directory), text, options)
                self.assertEqual(status, 0)
                action_prompt = next(c["prompt"] for c in calls if "write an action-items document" in c["prompt"])
                minutes_prompt = next(c["prompt"] for c in calls if "write formal draft minutes" in c["prompt"])
                action_input = action_prompt.split("Chunk summaries:\n", 1)[1]
                self.assertEqual(action_input, minutes_prompt.split("Chunk summaries:\n", 1)[1])
                self.assertNotIn("OLD plan", action_input)
                self.assertNotIn("Good morning", action_input)
                self.assertNotIn(f"Meeting section: {RECAP}", action_input)
                self.assertNotIn(f"Meeting section: {PRE_MEETING}", action_input)
                self.assertIn(f"Meeting section: {BUSINESS}", action_input)
                self.assertIn(f"Meeting section: {ADJOURNMENT}", action_input)
                self.assertIn("NEW plan", action_input)
                self.assertIn("Riley will circulate the notes tomorrow", action_input)
                summary_prompt = next(c["prompt"] for c in calls if "write a concise executive summary" in c["prompt"])
                self.assertIn("OLD plan", summary_prompt)
                self.assertIn(f"Meeting section: {RECAP}", summary_prompt)
                self.assertIn("clearly label any previous-meeting recap as historical context", summary_prompt)

    def test_action_items_with_only_recap_have_no_historical_reduce_input(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Good morning. At the last meeting, Morgan was assigned to circulate the OLD report.")
            self.assertEqual(status, 0)
            action_prompt = next(c["prompt"] for c in calls if "write an action-items document" in c["prompt"])
            action_input = action_prompt.split("Chunk summaries:\n", 1)[1]
            self.assertEqual(action_input.strip(), "No explicit current meeting business or adjournment was identified.")
            self.assertNotIn("OLD report", action_input)

    def test_pre_start_invitation_and_recap_stay_out_of_current_business(self):
        for options in ([], ["--keep-recap"]):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                status, calls, output = self.run_pipeline(Path(directory), opening_conversation_text(), options, reduce_echo=True)
                self.assertEqual(status, 0)
                action_prompt = next(c["prompt"] for c in calls if "write an action-items document" in c["prompt"])
                minutes_prompt = next(c["prompt"] for c in calls if "write formal draft minutes" in c["prompt"])
                action_input = action_prompt.split("Chunk summaries:\n", 1)[1]
                self.assertEqual(action_input, minutes_prompt.split("Chunk summaries:\n", 1)[1])
                for excluded in ("Casey", "Morgan", "time off", "equipment", "meeting tomorrow", "previous safety report", "Taylor will send"):
                    self.assertNotIn(excluded, action_input)
                self.assertIn("Riley will send the current safety report Friday", action_input)
                actions = (output / "action-items.md").read_text(encoding="utf-8")
                self.assertNotIn("Casey", actions)
                self.assertNotIn("Morgan", actions)
                self.assertNotIn("Taylor will send", actions)
                self.assertIn("Riley will send the current safety report Friday", actions)
                minutes = (output / "minutes-draft.md").read_text(encoding="utf-8")
                self.assertNotIn("Casey", minutes)
                self.assertNotIn("Morgan", minutes)
                if options:
                    self.assertEqual(minutes.count("## Recap of Previous Meeting"), 1)
                    historical, current = minutes.split("## Recap of Previous Meeting", 1)[1].split("## Overview", 1)
                    self.assertIn("previous safety report", historical)
                    self.assertNotIn("previous safety report", current)
                    recap_prompt = next(c["prompt"] for c in calls if "Summarize only the supplied historical recap" in c["prompt"])
                    self.assertNotIn("Casey", recap_prompt)
                    self.assertNotIn("current safety report", recap_prompt)
                else:
                    self.assertNotIn("Recap of Previous Meeting", minutes)
                    self.assertNotIn("previous safety report", minutes)
                summary_prompt = next(c["prompt"] for c in calls if "write a concise executive summary" in c["prompt"])
                self.assertIn("previous safety report", summary_prompt)
                self.assertIn(f"Meeting section: {RECAP}", summary_prompt)
                self.assertIn("clearly label any previous-meeting recap as historical context", summary_prompt)
                self.assertIn('prefix each historical recap bullet with "Recap of Previous Meeting:"', summary_prompt)

    def test_keep_recap_without_recap_does_not_add_an_extra_call(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), "[SPEAKER_00] Current discussion of the budget.", ["--keep-recap"])
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 4)
            self.assertNotIn("Recap of Previous Meeting", (output / "minutes-draft.md").read_text(encoding="utf-8"))

    def test_chunk_references_removed_from_all_final_outputs_in_both_recap_modes(self):
        def model_reply(call):
            prompt = call["prompt"]
            if "Transcript chunk:\n" in prompt:
                return "## Topics\n- Meeting discussion (Chunk 2.2)."
            if "Summarize only the supplied historical recap" in prompt:
                return "- Previous meeting discussed safety (Source chunk 1.3)."
            if "write formal draft minutes" in prompt:
                return "# Draft Minutes\n## Overview\n- Safety discussed (Chunk 2.2)."
            if "write an action-items document" in prompt:
                return "# Action Items\n- Riley – send safety report (Source chunk ID: 2.2)."
            return "# Summary\n- Recap of Previous Meeting: safety discussed [Chunk 1.3].\n- Current safety report discussed (Chunk 2.2)."

        for options in ([], ["--keep-recap"]):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                status, _, output = self.run_pipeline(Path(directory), opening_conversation_text(), options, model_reply=model_reply)
                self.assertEqual(status, 0)
                for filename in ("minutes-draft.md", "summary.md", "action-items.md"):
                    content = (output / filename).read_text(encoding="utf-8")
                    self.assertNotRegex(content, r"(?i)\b(?:source )?chunk(?: id)?\s*[:#]?\s*\d")
                    self.assertNotIn("(Chunk", content)
                self.assertIn("chunk_id", (output / "chunk_summaries.jsonl").read_text(encoding="utf-8"))
                self.assertIn("Chunk 2.2", (output / "chunk_summaries.jsonl").read_text(encoding="utf-8"))

    def test_contact_tasks_and_named_roles_survive_erroneous_model_output(self):
        source = "[SPEAKER_07] Let's get started.\n[Morgan] I'll contact Casey Morgan for contact detail.\n[SPEAKER_07] I've already messaged Casey and Riley for their contact details.\n[SPEAKER_07] We need a motion to adjourn.\n[SPEAKER_07] Motion by Taylor, seconded by Casey.\n[SPEAKER_07] Carried.\n[SPEAKER_07] The meeting is adjourned."
        tasks = "- Morgan – contact Casey Morgan for contact detail (Chunk 2.2).\n- Casey and Riley – provide contact details (Chunk 2.2)."
        motion = "- Motion to adjourn moved by Taylor, seconded by SPEAKER_07. Carried. (Chunk 3.1)"

        def model_reply(call):
            prompt = call["prompt"]
            if "Transcript chunk:\n" in prompt:
                return "## Topics\n- Casey and Riley have already been contacted for contact details.\n## Action items\n" + tasks + "\n## Motions or proposals mentioned\n" + motion
            if "write an action-items document" in prompt:
                return "# Action Items\n" + tasks
            if "write formal draft minutes" in prompt:
                return "# Draft Minutes\n## Action Items\n" + tasks + "\n## Motions / Proposals Mentioned\n" + motion
            return "# Summary\n" + motion

        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), source, model_reply=model_reply)
            self.assertEqual(status, 0)
            for filename in ("minutes-draft.md", "action-items.md"):
                content = (output / filename).read_text(encoding="utf-8")
                self.assertIn("Morgan – contact Casey Morgan for contact detail", content)
                self.assertNotIn("Casey and Riley – provide contact details", content)
                self.assertNotIn("Chunk", content)
            for filename in ("minutes-draft.md", "summary.md"):
                content = (output / filename).read_text(encoding="utf-8")
                self.assertIn("Motion to adjourn moved by Taylor, seconded by Casey. Carried.", content)
                self.assertNotIn("seconded by SPEAKER_07", content)
            report = json.loads((output / "minutes-qa.json").read_text(encoding="utf-8"))
            self.assertEqual(report["findings"], [])
            minutes_prompt = next(call["prompt"] for call in calls if "write formal draft minutes" in call["prompt"])
            self.assertIn("Explicit named adjournment announcements", minutes_prompt)
            self.assertIn("seconded by Casey", minutes_prompt)

    def test_recap_interruption_keeps_current_business_and_explicit_commitments(self):
        unsupported = "- Taylor – request another meeting with the manager."
        commitments = [
            "- Morgan – approve access for both members.",
            "- Riley – follow up with the employee.",
            "- Taylor – ask our team to start following this procedure.",
            "- Morgan – contact the coordinator.",
            "- Riley – try to identify relevant references and share the notes.",
        ]
        wrong_motion = "- Motion to adjourn moved by Casey, seconded by SPEAKER_02. Carried."

        def model_reply(call):
            prompt = call["prompt"]
            if "Transcript chunk:\n" in prompt:
                raw = prompt.split("Transcript chunk:\n", 1)[1].split("\nWrite concise markdown", 1)[0]
                summary = "## Topics\n" + raw
                if "Taylor, I think" in raw:
                    summary += "\n## Action items\n" + unsupported + "\n" + "\n".join(commitments)
                    summary += "\n## Motions or proposals mentioned\n" + wrong_motion
                return summary
            if "Summarize only the supplied historical recap" in prompt:
                return "- " + prompt.split("Previous-meeting chunk summaries:\n", 1)[1]
            if "write an action-items document" in prompt:
                return "# Action Items\n" + unsupported + "\n" + "\n".join(commitments)
            if "write formal draft minutes" in prompt:
                return "# Draft Minutes\n## Overview\nClarkson red flag / service track; South Terminal brake recovery; locomotive brake valve, manual and suppression handling.\n## Topics Discussed\n- Proposal to request another meeting with the manager.\n## Action Items\n" + unsupported + "\n" + "\n".join(commitments) + "\n## Motions / Proposals Mentioned\n" + wrong_motion
            return "# Summary\n- Recap of Previous Meeting: Morgan reported on crew staffing.\n- Current brake handling discussed."

        aliases = {"SPEAKER_02": "Chair", "SPEAKER_03": "Morgan", "SPEAKER_04": "Riley", "SPEAKER_05": "Taylor"}
        sources = (recap_interruption_text(), recap_interruption_text().replace("Okay guys all set here\n[SPEAKER_02] start", "Okay guys all set here start"))
        for source, options in ((source, options) for source in sources for options in ([], ["--keep-recap"])):
            with self.subTest(options=options, joined="all set here start" in source), tempfile.TemporaryDirectory() as directory:
                status, calls, output = self.run_pipeline(Path(directory), source, options, aliases=aliases, model_reply=model_reply)
                self.assertEqual(status, 0)
                action_prompt = next(call["prompt"] for call in calls if "write an action-items document" in call["prompt"])
                minutes_prompt = next(call["prompt"] for call in calls if "write formal draft minutes" in call["prompt"])
                action_input = action_prompt.split("Chunk summaries:\n", 1)[1]
                minutes_input = minutes_prompt.split("Chunk summaries:\n", 1)[1].split("\n\nExplicit named adjournment announcements", 1)[0]
                self.assertEqual(action_input, minutes_input)
                for prior_topic in ("North Terminal", "the freight operator", "101", "202", "Morgan reported"):
                    self.assertNotIn(prior_topic, action_input)
                for current_topic in ("East Terminal", "South Terminal", "brake valve", "suppression"):
                    self.assertIn(current_topic, action_input)
                self.assertNotIn(unsupported, action_input)
                self.assertNotIn(unsupported, (output / "chunk_summaries.jsonl").read_text(encoding="utf-8"))
                actions = (output / "action-items.md").read_text(encoding="utf-8")
                minutes = (output / "minutes-draft.md").read_text(encoding="utf-8")
                for document in (actions, minutes):
                    self.assertNotIn(unsupported, document)
                    for commitment in commitments:
                        self.assertIn(commitment, document)
                    for prior_topic in ("North Terminal", "the freight operator", "101", "202"):
                        self.assertNotIn(prior_topic, document)
                self.assertIn("Proposal to request another meeting", minutes)
                self.assertIn("Motion to adjourn moved by Casey, seconded by Taylor.", minutes)
                self.assertNotIn("Carried", minutes)
                self.assertNotIn("Morgan reported", minutes.split("## Overview", 1)[1])
                report = json.loads((output / "minutes-qa.json").read_text(encoding="utf-8"))
                self.assertEqual(report["findings"], [])
                if options:
                    recap_prompt = next(call["prompt"] for call in calls if "Summarize only the supplied historical recap" in call["prompt"])
                    self.assertIn("Morgan reported", recap_prompt)
                    for current_topic in ("East Terminal", "South Terminal", "brake valve", "suppression"):
                        self.assertNotIn(current_topic, recap_prompt)
                else:
                    self.assertNotIn("Recap of Previous Meeting", minutes)

    def test_lesson_profile_preserves_original_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), "[SPEAKER_00] Mackyard and Transport Canadaâ€™s rules.", ["--keep-recap"], aliases=[], profile="lesson")
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 5)
            self.assertIn("Mackyard", calls[0]["prompt"])
            self.assertFalse((output / "meeting_sections.jsonl").exists())
            self.assertFalse((output / "minutes-qa.json").exists())
            self.assertIn("SPEAKER_00 Mack Yard", (output / "lesson-notes.md").read_text(encoding="utf-8"))

    def assert_model_calls(self, calls, map_model, reduce_model, map_ctx, reduce_ctx):
        map_calls = [c for c in calls if "Transcript chunk:\n" in c["prompt"]]
        reduce_calls = [c for c in calls if c not in map_calls]
        self.assertTrue(map_calls)
        self.assertTrue(reduce_calls)
        for call in map_calls:
            self.assertEqual((call["model"], call["num_ctx"]), (map_model, map_ctx))
        for call in reduce_calls:
            self.assertEqual((call["model"], call["num_ctx"]), (reduce_model, reduce_ctx))

    def test_meeting_environment_defaults_override_shared_defaults_including_recap(self):
        environment = {
            "MEETING_MAP_MODEL": "qwen3.6:27b", "MEETING_REDUCE_MODEL": "qwen3.8:27b",
            "MEETING_MAP_NUM_CTX": "16384", "MEETING_REDUCE_NUM_CTX": "32768",
            "OLLAMA_MAP_MODEL": "shared-map", "OLLAMA_REDUCE_MODEL": "shared-reduce",
            "OLLAMA_MAP_NUM_CTX": "4096", "OLLAMA_REDUCE_NUM_CTX": "8192",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(
                Path(directory), "[SPEAKER_00] At the last meeting, OLD plan was approved. Moving on to new business. NEW plan discussed.",
                ["--keep-recap"], environment=environment, profile=None, default_profile="meeting",
            )
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "qwen3.6:27b", "qwen3.8:27b", 16384, 32768)
            self.assertEqual(len([c for c in calls if "Transcript chunk:\n" not in c["prompt"]]), 4)
            self.assertIn("## Recap of Previous Meeting", (output / "minutes-draft.md").read_text(encoding="utf-8"))

    def test_meeting_cli_overrides_environment_even_invalid_context_defaults(self):
        environment = {
            "MEETING_MAP_MODEL": "env-map", "MEETING_REDUCE_MODEL": "env-reduce",
            "MEETING_MAP_NUM_CTX": "invalid", "MEETING_REDUCE_NUM_CTX": "invalid",
            "OLLAMA_MAP_NUM_CTX": "also-invalid", "OLLAMA_REDUCE_NUM_CTX": "also-invalid",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(
                Path(directory), "[SPEAKER_00] Current business.",
                ["--map-model", "custom-fast:latest", "--reduce-model", "custom-final:latest",
                 "--map-num-ctx", "8192", "--reduce-num-ctx", "49152"], environment=environment,
            )
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "custom-fast:latest", "custom-final:latest", 8192, 49152)

    def test_meeting_partial_overrides_fall_back_per_setting(self):
        environment = {
            "MEETING_REDUCE_MODEL": "qwen3.8:27b", "MEETING_REDUCE_NUM_CTX": "32768",
            "MEETING_MAP_MODEL": "", "MEETING_MAP_NUM_CTX": "",
            "OLLAMA_MAP_MODEL": "qwen2.5:32b", "OLLAMA_MAP_NUM_CTX": "16384",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", environment=environment)
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "qwen2.5:32b", "qwen3.8:27b", 16384, 32768)

    def test_meeting_without_overrides_preserves_shared_environment_defaults(self):
        environment = {
            "OLLAMA_MAP_MODEL": "existing-fast", "OLLAMA_REDUCE_MODEL": "existing-final",
            "OLLAMA_MAP_NUM_CTX": "4096", "OLLAMA_REDUCE_NUM_CTX": "8192",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", environment=environment)
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "existing-fast", "existing-final", 4096, 8192)

    def test_unconfigured_profiles_keep_existing_builtin_defaults(self):
        for profile in ("meeting", "lesson"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current discussion.", profile=profile)
                self.assertEqual(status, 0)
                self.assert_model_calls(calls, "qwen2.5:32b", "qwen2.5:32b", None, None)

    def test_lesson_defaults_ignore_all_meeting_model_settings(self):
        environment = {
            "MEETING_MAP_MODEL": "qwen3.6:27b", "MEETING_REDUCE_MODEL": "qwen3.8:27b",
            "MEETING_MAP_NUM_CTX": "invalid", "MEETING_REDUCE_NUM_CTX": "invalid",
            "OLLAMA_MAP_MODEL": "lesson-map", "OLLAMA_REDUCE_MODEL": "lesson-reduce",
            "OLLAMA_MAP_NUM_CTX": "4096", "OLLAMA_REDUCE_NUM_CTX": "8192",
        }
        # Test both the lesson wrapper's default and an explicit lesson profile
        # overriding a meeting wrapper default; the selected profile must win.
        for profile, default_profile in ((None, "lesson"), ("lesson", "meeting")):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Lesson content.", profile=profile, default_profile=default_profile, environment=environment)
                self.assertEqual(status, 0)
                self.assert_model_calls(calls, "lesson-map", "lesson-reduce", 4096, 8192)
        meeting_only = {key: value for key, value in environment.items() if key.startswith("MEETING_")}
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Lesson content.", profile="lesson", environment=meeting_only)
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "qwen2.5:32b", "qwen2.5:32b", None, None)

    def test_explicit_meeting_profile_uses_meeting_defaults_from_lesson_wrapper(self):
        environment = {
            "MEETING_MAP_MODEL": "meeting-map", "MEETING_REDUCE_MODEL": "meeting-reduce",
            "MEETING_MAP_NUM_CTX": "16384", "MEETING_REDUCE_NUM_CTX": "32768",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", profile="meeting", default_profile="lesson", environment=environment)
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "meeting-map", "meeting-reduce", 16384, 32768)

    def test_invalid_selected_meeting_context_reports_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as error:
                self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", environment={"MEETING_MAP_NUM_CTX": "invalid"})
            self.assertEqual(error.exception.code, 2)

    def test_bad_alias_configuration_stops_before_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", aliases={"SPEAKER_00": ""})
            self.assertEqual(status, 2)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
