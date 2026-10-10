"""General prompt contracts and synthetic input boundaries, not model-quality claims."""
import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed
from test_editorial_notes import portion


def segment(number, text, section=ed.BUSINESS):
    return {**portion(number, text, section), "chunk_id": f"{number}.1"}


class EditorialInstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = ROOT / "prompts/meeting"
        cls.template = (cls.prompts / "meeting_notes_prompt.txt").read_text(encoding="utf-8")
        cls.instructions = " ".join(cls.template.lower().split())

    def setUp(self):
        # Unequal discussion lengths: the brief dispute must still reach the
        # drafting request. These are synthetic, not benchmark/source facts.
        self.chunks = [segment(1, "[Chair] Staffing was discussed at the prior meeting.", ed.RECAP),
                       segment(2, "[Delegate] The union disputes the overtime interpretation.\n"
                               "[Delegate] A proposed law's effect remains uncertain.\n"
                               "[Chair] " + "The equipment discussion continued. " * 40),
                       segment(3, "[Chair] A motion was seconded; the outcome was not stated.", ed.ADJOURNMENT)]
        self.records = ed.source_records(self.chunks)
        self.register = {"items": [], "findings": []}

    def test_material_union_topics_and_brief_discussion_priority_are_explicit(self):
        for topic in ("collective-agreement disputes", "health and safety", "legislative developments",
                      "grievances", "motions", "unresolved operational problems"):
            self.assertIn(topic, self.instructions)
        self.assertRegex(self.instructions, r"cover material topics .*discussed briefly")
        self.assertIn("discussion time or repetition is not a measure of importance", self.instructions)
        self.assertIn("do not invent coverage for topics absent from the evidence", self.instructions)

    def test_attribution_privacy_and_status_contracts_are_explicit(self):
        for kind in ("participant statements", "allegations", "rumours", "interpretations"):
            self.assertIn(kind, self.instructions)
        self.assertIn("not a verified legal conclusion", self.instructions)
        self.assertIn("once per coherent paragraph or issue", self.instructions)
        self.assertIn("avoid repetitive qualifications", self.instructions)
        for private_topic in ("personal medical information", "disciplinary details", "fatality investigations",
                              "sensitive grievance information", "identifying details"):
            self.assertIn(private_topic, self.instructions)
        self.assertIn("explicitly establishes their suitability", self.instructions)
        self.assertIn("retain sensitive explanations only in private concerns", self.instructions)
        for status in ("recorded motions", "supported decisions", "reported work", "tentative proposals", "unresolved matters"):
            self.assertIn(status, self.instructions)
        self.assertIn("never turn an unapproved proposal", self.instructions)
        register_instructions = " ".join((self.prompts / "register_prompt.txt").read_text(encoding="utf-8").lower().split())
        self.assertIn("member_facing is not operator approval", register_instructions)
        self.assertNotIn("enter the public table", register_instructions)

    def test_issue_structure_readability_and_flexible_length_contract(self):
        self.assertIn("organize by issue, not speaker or transcript chunk", self.instructions)
        self.assertIn("readable paragraphs", self.instructions)
        self.assertIn("highlights should concisely identify the most important business", self.instructions)
        self.assertIn("1,800–2,400", self.template)
        self.assertIn("do not pad it or invent context", self.instructions)
        self.assertIn("shorter meeting may need far fewer words", self.instructions)

    def test_synthetic_brief_topics_and_exact_references_survive_prompt_building(self):
        records, register = copy.deepcopy(self.records), copy.deepcopy(self.register)
        summaries = "A lengthy equipment report and brief overtime and legislative discussions."
        prompt = ed.notes_prompt(self.prompts, self.records, summaries, self.register)
        payload = json.loads(prompt.split("Untrusted input JSON:\n", 1)[1])
        self.assertEqual(payload["summaries"], summaries)
        self.assertEqual({r["id"] for r in payload["source_excerpts"]}, set(self.records))
        for row in payload["source_excerpts"]:
            self.assertEqual(row["text"], self.records[row["id"]]["text"])
        self.assertIn("2.1:L1", {r["id"] for r in payload["source_excerpts"]})
        self.assertIn("2.1:L2", {r["id"] for r in payload["source_excerpts"]})
        self.assertEqual(self.records, records)
        self.assertEqual(self.register, register)

    def test_documented_json_examples_still_match_strict_validator_contract(self):
        fields = re.search(r"Return JSON only with exactly:\s*(.*?)\. Each", self.template, re.S)[1]
        self.assertEqual({field.strip() for field in fields.split(",")},
                         {"highlights", "previous_context", "issues", "motions", "unresolved", "concerns"})
        concern = json.loads(re.search(r'^\{"category":.*\}', self.template, re.M)[0])
        self.assertEqual(set(concern), {"category", "text", "source_ids"})
        self.assertIn(concern["category"], ed.CONCERNS)
        block = json.loads(re.search(r'\{"text":"Plain prose\.".*?\}', self.template)[0])
        # The syntax IDs are examples, not evidence; make that ID real only in
        # this synthetic fixture before validating the documented schema.
        records = ed.source_records([segment(1, "[Chair] A report was discussed.\n[Chair] Its status requires review.")])
        notes = {"highlights": [], "previous_context": [], "issues": [{"heading": "Reporting", "paragraphs": [block]}],
                 "motions": [], "unresolved": [], "concerns": [concern]}
        clean = ed.validate_notes(notes, records, "A report was discussed.")
        self.assertEqual(clean["concerns"], [concern])
        self.assertNotIn(concern["text"], ed.render_notes(clean, self.register))
        invalid = copy.deepcopy(notes)
        invalid["concerns"] = [concern["category"]]
        with self.assertRaisesRegex(ed.EditorialFailure, "invalid_notes_concern"):
            ed.validate_notes(invalid, records, "A report was discussed.")

    def test_reusable_prompt_has_no_october_specific_examples_or_benchmark_dependency(self):
        for literal in ("October", "2026", "Mac Yard", "MRS", "CAB", "Wayne", "E-test", "70%",
                        "efficiency tests", "first slot", "earlier speaking opportunity"):
            self.assertNotRegex(self.template, re.compile(re.escape(literal), re.I))
        read_paths, original_read = [], Path.read_text
        def tracked_read(path, *args, **kwargs):
            read_paths.append(path.resolve())
            return original_read(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = Path(tmp) / "accepted-editorial-benchmark.md"
            benchmark.write_text("PRIVATE QUALITY BENCHMARK SENTINEL", encoding="utf-8")
            with patch.object(Path, "read_text", tracked_read):
                prompt = ed.notes_prompt(self.prompts, self.records, "Synthetic summaries", self.register)
            self.assertEqual(read_paths, [(self.prompts / "meeting_notes_prompt.txt").resolve()])
            self.assertNotIn("PRIVATE QUALITY BENCHMARK SENTINEL", prompt)


if __name__ == "__main__":
    unittest.main()
