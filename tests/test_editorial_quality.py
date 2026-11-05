"""Frozen synthetic editorial counterexamples; no production text or inference."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed
from meeting_postprocess.editorial_quality import TRACE, review_quality, source_review
from meeting_postprocess.publication import publication_payload, strip_private_references
from test_editorial_notes import fixture


class EditorialQualityTests(unittest.TestCase):
    def test_frozen_role_allegation_comparison_and_chronology_warnings(self):
        for case in json.loads((ROOT / "tests/fixtures/editorial_quality_cases.json").read_text(encoding="utf-8")):
            with self.subTest(code=case["expected"]):
                notes = {"highlights": [], "previous_context": [], "motions": [], "unresolved": [], "concerns": [],
                         "issues": [{"heading": "Business", "paragraphs": [{"text": case["claim"], "source_ids": ["1:L1"]}]}]}
                records = {"1:L1": {"text": case["source"]}}
                before = copy.deepcopy((notes, records))
                _, flags = review_quality(notes, records, case["claim"])
                finding = next(f for f in flags if f["code"] == case["expected"])
                self.assertEqual(finding["source_ids"], ["1:L1"])
                self.assertEqual(finding["evidence_pointer"], TRACE + "#block-1")
                self.assertEqual((notes, records), before)

    def test_length_never_truncates_and_short_meetings_are_not_padded(self):
        chunks, _, notes = fixture()
        records = ed.source_records(chunks)
        before = copy.deepcopy(notes)
        metrics, flags = review_quality(notes, records, "business " * 3001)
        warning = next(f for f in flags if f["code"] == "length_review")
        self.assertEqual(warning["word_count"], 3001)
        self.assertIn("Never truncate", warning["instruction"])
        self.assertIn("separate copy", warning["instruction"])
        self.assertFalse(metrics["automatic_revision"])
        self.assertEqual(notes, before)
        for words in (120, 2600, 3000):
            self.assertNotIn("length_review", [f["code"] for f in review_quality(notes, records, "business " * words)[1]])
        records["3:L1"]["text"] = "source " * 4001
        self.assertIn("length_review", [f["code"] for f in review_quality(notes, records, "business " * 1500)[1]])

    def test_recap_and_repetition_are_reviewed_without_erasing_business(self):
        chunks, _, notes = fixture()
        notes["previous_context"] = [{"text": "history " * 181, "source_ids": ["3:L1"]}]
        notes["unresolved"] = copy.deepcopy(notes["issues"][0]["paragraphs"])
        codes = {f["code"] for f in review_quality(notes, ed.source_records(chunks), "business " * 2000)[1]}
        self.assertIn("previous_context_length_review", codes)
        self.assertIn("repeated_notes_block", codes)

    def test_private_trace_preserves_utf8_and_fences_untrusted_source(self):
        chunks, _, notes = fixture()
        records = ed.source_records(chunks)
        key = notes["issues"][0]["paragraphs"][0]["source_ids"][0]
        records[key]["text"] = 'Café — naïve “quotation”.\n```\n![untrusted](https://invalid.example)'
        trace = source_review(notes, records, "frozen-hash")
        self.assertIn(records[key]["text"], trace)
        self.assertIn("````text", trace)
        self.assertIn("containing chunk bounds", trace)
        self.assertIn("frozen-hash", trace)
        self.assertNotIn(TRACE, strip_private_references(f"[private]({TRACE})\nPublic business."))

    def test_full_generation_preserves_responses_private_permissions_and_hold(self):
        chunks, register, notes = fixture()
        paragraph = notes["issues"][0]["paragraphs"][0]
        paragraph["text"] = "The presiding officer alleged false information. " + "Substantive business. " * 180
        notes["issues"][0]["paragraphs"] = [copy.deepcopy(paragraph) for _ in range(9)]
        original = json.dumps(notes)
        calls = []
        detailed = "# Detailed Minutes\n" + "Comprehensive detail. " * 1500
        def generate(prompt, **kwargs):
            calls.append(prompt)
            return [json.dumps(register), original, detailed][len(calls)-1]
        class Budget:
            measurements = []
            def measure(self, *args):
                return {"fits": True, "reserved_output": 32768}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(ed.run(directory, chunks, "maps", "current", "", [], [], {}, [], [],
                                    ROOT / "prompts/meeting", generate, False, budget=Budget()), 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads((directory / ed.RESPONSE).read_text(encoding="utf-8"))["responses"]["notes"], original)
            self.assertIn(detailed.rstrip(), (directory / "minutes-draft.md").read_text(encoding="utf-8"))
            review = json.loads((directory / ed.REVIEW).read_text(encoding="utf-8"))
            self.assertEqual(review["status"], "review_hold")
            self.assertIn("length_review", {f["code"] for f in review["findings"]})
            checklist = (directory / ed.CHECKLIST).read_text(encoding="utf-8")
            self.assertIn(f"[{TRACE}]({TRACE})", checklist)
            self.assertIn("separate copy", checklist)
            self.assertIn("Substantive business.", (directory / TRACE).read_text(encoding="utf-8"))
            if sys.platform.startswith("linux"):
                self.assertEqual((directory / TRACE).stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, "hold"):
                publication_payload(directory)

    def test_mojibake_is_flagged_and_original_is_not_silently_rewritten(self):
        from meeting_postprocess.qa import check_minutes
        original = "CafÃ© â€” reported wording."
        self.assertIn("mojibake", [f.code for f in check_minutes(original)])
        self.assertNotIn("mojibake", [f.code for f in check_minutes("Café — reported wording.")])
        self.assertEqual(original, "CafÃ© â€” reported wording.")
