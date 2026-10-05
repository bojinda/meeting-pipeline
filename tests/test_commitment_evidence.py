from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from meeting_postprocess.commitments import commitment_evidence
from meeting_postprocess.sections import ADJOURNMENT, BUSINESS, PRE_MEETING, RECAP, prepare_chunks


def section(text, label=BUSINESS):
    return {"text": text, "meeting_section": label}


class CommitmentEvidenceTests(unittest.TestCase):
    def test_first_person_future_forms_keep_speaker_and_exact_qualification(self):
        for text in ("I'll approve access for both members.", "I will contact the coordinator.", "I'm going to prepare the report.", "I'm gonna send the notes.", "I am going to review the report.", "When I get back, I'm going to contact the coordinator.", "When I get back I'm gonna request another meeting.", "I'll try to isolate the relevant articles and share them.", "I'll get our team to start using the procedure.", "I’ll touch base with the employee."):
            with self.subTest(text=text):
                source = "[Taylor] " + text
                self.assertEqual(commitment_evidence([section(source)]), [source])

    def test_only_current_business_and_adjournment_can_supply_evidence(self):
        source = "[Taylor] I'll send the report tomorrow."
        for label in (BUSINESS, ADJOURNMENT, PRE_MEETING, RECAP, "", None):
            with self.subTest(label=label):
                self.assertEqual(commitment_evidence([section(source, label)]), [source] if label in {BUSINESS, ADJOURNMENT} else [])

    def test_quoted_hypothetical_self_speech_is_not_commitment_evidence(self):
        for text in (
            "And if they say yes, and I say, okay, I'll do it, and then I'll grieve it.",
            "If the schedule changes, I say I'll send it, and then I'll file the grievance.",
            "For example, I would say I'll send it, and then I'll file the grievance.",
            "Imagine the claim is denied, so I'll file the grievance.",
            "Let's say the claim is denied, and then I'll file the grievance.",
            "If the claim is denied, you would then reply I'll send it, and then I'll file the grievance.",
        ):
            with self.subTest(text=text):
                self.assertEqual(commitment_evidence([section("[Taylor] " + text)]), [])

    def test_genuine_conditional_undertakings_keep_exact_condition_and_qualification(self):
        for text in (
            "If the claim is denied, I'll file the grievance.",
            "If the claim is denied, I'll try to file the grievance.",
            "If they say yes, I'll file the grievance.",
            "If the claim is denied I will file the grievance.",
            "When the claim is denied, I'm going to file the grievance.",
            "Once the claim is denied, I'm gonna file the grievance.",
        ):
            with self.subTest(text=text):
                source = "[Taylor] " + text
                self.assertEqual(commitment_evidence([section(source)]), [source])

    def test_group_task_requires_a_connected_construction(self):
        for text in (
            "I'll get our guys to start doing the same thing.",
            "I'll get our team using the procedure.",
            "I'll get the staff to review the report.",
            "I'll try to get our people following the procedure.",
        ):
            with self.subTest(text=text):
                source = "[Taylor] " + text
                self.assertEqual(commitment_evidence([section(source)]), [source])

    def test_distant_group_task_words_do_not_supply_evidence(self):
        for text in (
            "I'll get our guys by being, as far as I know, some of them were doing that.",
            "I'll get our team a report about what others were doing.",
            "I'll get our people, after a discussion about the timetable, to start doing that.",
            "I'll get our staff to, as far as I know, start doing that.",
            "I'll get our team to me, where some of them were doing that.",
        ):
            with self.subTest(text=text):
                self.assertEqual(commitment_evidence([section("[Taylor] " + text)]), [])

    def test_notification_commitments_keep_conditions_and_qualifications(self):
        for text in (
            "I'll inform the members about the decision.",
            "I'll notify the members when the report arrives.",
            "I'll let the members know what the report contains.",
            "Once I get the list, I'll let all the members know what it contains.",
            "If the claim is denied, I'll notify our members about the decision.",
            "I'll try to let our team know what the updated procedure requires.",
        ):
            with self.subTest(text=text):
                source = "[SPEAKER_03] " + text
                self.assertEqual(commitment_evidence([section(source)]), [source])

    def test_conversational_let_you_know_is_excluded_conservatively(self):
        for text in ("I'll let you know.", "I'll let you know later.", "Once I get back, I'll let you know."):
            with self.subTest(text=text):
                self.assertEqual(commitment_evidence([section("[Taylor] " + text)]), [])

    def test_later_third_party_delivery_does_not_supply_speaker_recipient(self):
        own = "I'll send the package off"
        for step in (
            ", and if the office approves it, they will send it to Morgan.",
            ", and if the reviewing body agrees then they give it to the employer redacted.",
            ", and then they will forward it to Morgan.",
            "; the office forwards it to Morgan.",
            ", and then Casey will deliver it to Morgan.",
            ", and then they will pass it to Morgan.",
            ", and then they will redact it for Morgan.",
        ):
            with self.subTest(step=step):
                source = "[Taylor] " + own + step
                self.assertEqual(commitment_evidence([section(source)]), ["[Taylor] " + own])
                self.assertEqual(source, "[Taylor] " + own + step)

    def test_own_explicit_recipient_and_qualifications_survive_later_delivery(self):
        own = "If the claim is denied, I'll try to send the package to Casey"
        source = "[SPEAKER_03] " + own + ", and if the office approves it, they will send it to Morgan."
        self.assertEqual(commitment_evidence([section(source)]), ["[SPEAKER_03] " + own])

    def test_first_person_followup_keeps_its_explicit_recipient(self):
        for text in (
            "I'll send the package off, and if the office approves it, I'll send it to Morgan.",
            "I'll prepare the package, and then we will send it to Morgan.",
            "I'll prepare the package, and then Taylor will send it to Morgan.",
        ):
            with self.subTest(text=text):
                source = "[Taylor] " + text
                self.assertEqual(commitment_evidence([section(source)]), [source])

    def test_clear_clause_prefaces_and_asr_repetition_keep_undertaking_wording(self):
        for text in ("The schedule is unclear, but I'll contact the coordinator.", "And so I'll try to isolate the references.", "Okay, I'll, I'll get our team following the procedure.", "When the list arrives, I'll, I'll approve access for both members.", "I'll be filing the grievance tomorrow.", "I will find out which report is missing."):
            with self.subTest(text=text):
                source = "[Taylor] " + text
                self.assertEqual(commitment_evidence([section(source)]), [source])

    def test_conversation_capability_jokes_hypotheticals_and_tentative_words_are_excluded(self):
        for text in ("I'll go first.", "I'll make it short.", "I'll be able to send the report.", "I can send the report.", "I think I should send the report.", "I think I'll send the report.", "If I had time, I'll send the report.", "If I had time, but I'll send the report.", "I'll send the report if I could.", "I'll send the report, just kidding.", "I'm gonna contact the coordinator as a joke.", "Morgan said I'll send the report.", "Morgan said, but I'll send the report.", "I think the schedule works, so I'll contact the coordinator."):
            with self.subTest(text=text):
                self.assertEqual(commitment_evidence([section("[Taylor] " + text)]), [])

    def test_speaker_labels_and_existing_alias_preparation_remain_intact(self):
        raw = [{"chunk_id": 1, "text": "[SPEAKER_03] I'll try to isolate the references."}]
        for aliases, speaker in (({}, "SPEAKER_03"), ({"SPEAKER_03": "Taylor"}, "Taylor")):
            with self.subTest(aliases=aliases):
                self.assertEqual(commitment_evidence(prepare_chunks(raw, aliases)), [f"[{speaker}] I'll try to isolate the references."])

    def test_evidence_is_bounded_deduplicated_and_does_not_truncate_conditions(self):
        source = "[Taylor] I'll send the report."
        self.assertEqual(commitment_evidence([section(source), section(source)]), [source])
        candidates = [section(f"[Taylor] I'll send report {number}.") for number in range(40)]
        self.assertEqual(commitment_evidence(candidates), [row["text"] for row in candidates[:32]])
        self.assertEqual(commitment_evidence([section("[Taylor] I'll send " + "the report " * 120)]), [])


if __name__ == "__main__":
    unittest.main()
