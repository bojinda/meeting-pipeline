"""Synthetic selective evidence, exact hydration, omission QA and safe limits."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import whole_synthesis as whole
from meeting_postprocess import whole_evidence as refs
from meeting_postprocess import whole_source as source
from meeting_postprocess.sections import BUSINESS, RECAP, ADJOURNMENT
from meeting_postprocess.publication import publication_payload
import test_whole_synthesis as existing


def evidence(record, number, kind="topic", support=(), **fields):
    return {"id": "E" + str(number), "kind": kind, "section": record["section"], "primary_record_id": record["id"], "supporting_record_ids": list(support), "owners": [], "mover": None, "seconder": None, "outcome": None, **fields}


class SelectiveModel:
    def __init__(self, drop_critical=False, duplicate=False):
        self.requests = []
        self.drop_critical = drop_critical
        self.duplicate = duplicate

    def __call__(self, request):
        self.requests.append(request)
        payload = json.loads(request["prompt"])
        if payload.get("format") == "meeting-source-v1":
            records = [{"id": str(sid), "section": payload["sections"][code], "speaker": payload["speakers"][speaker], "text": text} for code, run in payload["runs"] for sid, speaker, text in run]
            items, grouped = [], {}
            for record in records:
                text = record["text"]
                if record["section"] == RECAP:
                    grouped.setdefault("recap", []).append(record)
                elif text.startswith("Motion to adjourn"):
                    support = [row["id"] for row in records if "Motion by Taylor" in row["text"] or row["text"] == "Carried."]
                    if not self.drop_critical:
                        items.append(evidence(record, len(items) + 1, "motion", support, mover="Taylor", seconder="Casey", outcome="Carried" if any(row["text"] == "Carried." for row in records) else None))
                elif "I'll send" in text:
                    if not self.drop_critical:
                        items.append(evidence(record, len(items) + 1, "action", owners=[record["speaker"]]))
                elif text.startswith("We agreed"):
                    items.append(evidence(record, len(items) + 1, "decision"))
                elif "I disagree" in text:
                    items.append(evidence(record, len(items) + 1, "qualification"))
                elif "safety concern" in text:
                    items.append(evidence(record, len(items) + 1, "health_safety"))
                elif "brake report" in text:
                    grouped.setdefault("brakes", []).append(record)
                elif "schedule report" in text:
                    grouped.setdefault("schedule", []).append(record)
            for topic, group in grouped.items():
                items.append(evidence(group[0], len(items) + 1, "recap" if topic == "recap" else "topic", [row["id"] for row in group[1:]]))
            if self.duplicate:
                item = copy.deepcopy(next(row for row in items if row["kind"] == "topic"))
                item["id"] = "E999"
                items.append(item)
            data = {"format": refs.FORMAT, "items": items}
        else:
            hydrated = payload["validated_evidence"]
            data = {}
            for document in ("summary", "minutes", "actions"):
                groups = {}
                for item in hydrated:
                    if document == "actions" and item["kind"] != "action" or document == "minutes" and item["kind"] == "recap" and not payload["keep_recap"]:
                        continue
                    groups.setdefault(whole.HEADINGS[item["kind"]], []).append(item["id"])
                data[document] = [{"section": key, "evidence_ids": values} for key, values in groups.items()]
        return {"response": json.dumps(data), "done_reason": "stop", "prompt_eval_count": 1200, "eval_count": 500}


def discussion():
    start = "[SPEAKER_00] Hello, PRE_CHAT_SENTINEL.\n[SPEAKER_00] Let's get started.\n"
    lines = [start]
    for i in range(50):
        lines.append(f"[SPEAKER_{1 + i % 2:02d}] The brake report concerns inspection detail {i}.\n")
        if i == 15:
            lines.append("[SPEAKER_01] I'll send the updated safety notice tomorrow.\n")
        if i == 25:
            lines.append("[SPEAKER_02] We agreed to inspect the service track.\n")
        if i == 35:
            lines.append("[SPEAKER_02] I disagree with delaying the inspection.\n")
    lines += ["[SPEAKER_01] The brake inspection remains an outstanding safety concern.\n",
              "[SPEAKER_01] The schedule report concerns the next inspection.\n",
              "[SPEAKER_02] The schedule report includes a conditional staffing concern.\n",
              "[SPEAKER_00] Motion to adjourn.\n[SPEAKER_00] Motion by Taylor, seconded by Casey.\n[SPEAKER_00] Carried.\n"]
    return "".join(lines)


class SelectionTests(unittest.TestCase):
    def test_long_discussion_and_distinct_topics_use_meaningful_groups_with_critical_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root, discussion())
            model = SelectiveModel()
            status, output, _, _ = existing.invoke(root, directory, model)
            self.assertEqual(status, 0)
            self.assertEqual(len(model.requests), 2)
            report = json.loads((output / "whole-run.json").read_text())
            counts = report["accepted_evidence_counts_by_category"]
            self.assertEqual(counts["topic"], 2)
            for kind in ("action", "motion", "decision", "qualification", "health_safety"):
                self.assertEqual(counts[kind], 1)
            data = json.loads((output / "whole-evidence.json").read_text())
            brake = next(row for row in data["items"] if row["kind"] == "topic" and "brake report" in row["statement"])
            self.assertEqual(len(brake["quotes"]), 50)
            self.assertNotIn("maxItems", whole.extraction_schema()["properties"]["items"])
            properties = whole.extraction_schema()["properties"]["items"]["items"]["properties"]
            self.assertNotIn("statement", properties)
            self.assertEqual(properties["kind"]["enum"][:3], ["motion", "action", "decision"])
            self.assertEqual(properties["kind"]["enum"][-1], "topic")
            self.assertTrue(report["context_coverage"]["complete"])
            semantic = report["semantic_evidence_coverage"]
            self.assertEqual(semantic["known_source_checks"], "passed")
            self.assertEqual(semantic["semantic_completeness"], "not_certified")
            self.assertTrue(report["calls"][0]["output_token_estimate"] > 0)
            self.assertEqual(report["calls"][0]["eval_count"], 500)
            self.assertEqual(model.requests[0]["num_ctx"], 98304)
            self.assertEqual(model.requests[0]["num_predict"], 16384)
            self.assertNotIn("PRE_CHAT_SENTINEL", json.dumps(model.requests))
            self.assertIn("inspection detail 49", (output / "minutes-draft.md").read_text())
            self.assertIn("I disagree with delaying", (output / "minutes-draft.md").read_text())

    def test_primary_attribution_and_all_quotes_are_restored_exactly(self):
        records = [{"id": "R1", "position": 0, "section": BUSINESS, "speaker": "Taylor", "text": "The brake report needs review."},
                   {"id": "R2", "position": 1, "section": BUSINESS, "speaker": "Casey", "text": "However, it must retain the staffing condition."}]
        data = {"format": refs.FORMAT, "items": [evidence({**records[1], "id": "2"}, 1, support=["1"])]}
        original = copy.deepcopy(data)
        accepted, rejected = whole.validate_evidence(data, records, source.encode(records), expected_format=refs.FORMAT)
        self.assertFalse(rejected)
        self.assertEqual(data, original)
        self.assertEqual(accepted[0]["primary_record_id"], "R2")
        self.assertEqual(accepted[0]["statement"], records[1]["text"])
        self.assertEqual({row["record_id"]: row["text"] for row in accepted[0]["quotes"]}, {row["id"]: row["text"] for row in records})

    def test_reference_protocol_has_no_silent_item_cap_and_rejects_free_claims(self):
        records = [{"id": "R" + str(i), "position": i, "section": BUSINESS, "speaker": "Taylor", "text": f"Substantive report detail {i}."} for i in range(520)]
        data = {"format": refs.FORMAT, "items": [evidence({**record, "id": str(i + 1)}, i + 1) for i, record in enumerate(records)]}
        accepted, rejected = whole.validate_evidence(data, records, source.encode(records), expected_format=refs.FORMAT)
        self.assertFalse(rejected)
        self.assertEqual(len(accepted), 520)
        bad = {"format": refs.FORMAT, "items": [{**data["items"][0], "statement": "An unsupported interpretation."}]}
        accepted, rejected = whole.validate_evidence(bad, records, source.encode(records), expected_format=refs.FORMAT)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["category"], "missing_or_extra_fields")

    def test_valid_source_ids_cannot_support_unrelated_interpretations(self):
        for kind, fields in (("action", {"owners": ["Taylor"]}), ("decision", {}), ("motion", {"mover": "Taylor", "outcome": "Carried"}), ("issue", {}), ("health_safety", {}), ("qualification", {})):
            with self.subTest(kind=kind):
                records = [{"id": "R1", "position": 0, "section": BUSINESS, "speaker": "Taylor", "text": "The cafeteria served lunch."}]
                data = {"format": refs.FORMAT, "items": [evidence({**records[0], "id": "1"}, 1, kind, **fields)]}
                accepted, rejected = whole.validate_evidence(data, records, source.encode(records), expected_format=refs.FORMAT)
                self.assertFalse(accepted)
                self.assertTrue(rejected)

    def test_historical_refs_cannot_create_current_action_or_decision(self):
        record = {"id": "R1", "position": 0, "section": RECAP, "speaker": "Taylor", "text": "I'll send the historical report."}
        for kind in ("action", "decision"):
            with self.subTest(kind=kind):
                data = {"format": refs.FORMAT, "items": [evidence({**record, "id": "1", "section": BUSINESS}, 1, kind, owners=["Taylor"] if kind == "action" else [])]}
                accepted, rejected = whole.validate_evidence(data, [record], source.encode([record]), expected_format=refs.FORMAT)
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["category"], "historical_or_cross_section_claim")

    def test_missing_actions_and_motions_add_private_qa_without_claiming_semantic_completeness(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root, discussion())
            status, output, _, _ = existing.invoke(root, directory, SelectiveModel(drop_critical=True))
            self.assertEqual(status, 0)
            report = json.loads((output / "whole-run.json").read_text())
            self.assertTrue(report["context_coverage"]["complete"])
            semantic = report["semantic_evidence_coverage"]
            self.assertFalse(semantic["omitted_commitment_record_ids"] == [])
            self.assertFalse(semantic["omitted_motion_record_ids"] == [])
            self.assertEqual(semantic["known_source_checks"], "needs_review")
            qa = (output / "minutes-qa.json").read_text()
            self.assertIn("source_commitment_omission", qa)
            self.assertIn("source_motion_omission", qa)
            self.assertNotIn("I'll send", (output / "action-items.md").read_text())

    def test_duplicate_topics_fail_closed_without_silent_discard(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root, discussion())
            model = SelectiveModel(duplicate=True)
            status, output, _, _ = existing.invoke(root, directory, model)
            self.assertEqual(status, 1)
            self.assertEqual(len(model.requests), 1)
            report = json.loads((output / "whole-run.json").read_text())
            self.assertEqual(report["duplicate_rejected_item_count"], 1)
            self.assertEqual(publication_payload(output)["documents"], [])

    def test_generation_limit_retains_private_buffer_but_never_accepts_partial_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root, discussion())
            model = Mock(return_value={"response": '{"format":"meeting-evidence-v2","items":[', "done_reason": "length", "eval_count": 16384})
            status, output, logs, _ = existing.invoke(root, directory, model, flags=["--synthesis-retain-response"])
            self.assertEqual(status, 1)
            model.assert_called_once()
            self.assertEqual(publication_payload(output)["documents"], [])
            self.assertTrue((output / whole.PRIVATE_RESPONSE_FILENAME).exists())
            report = json.loads((output / "whole-run.json").read_text())
            self.assertEqual(report["failure_category"], "generation_token_limit")
            self.assertEqual(sum(report["accepted_evidence_counts_by_category"].values()), 0)
            self.assertTrue(report["context_coverage"]["complete"])
            self.assertFalse(report["semantic_evidence_coverage"]["extraction_completed_and_validated"])
            self.assertTrue(report["unrepresented_motion_record_ids"])
            self.assertNotIn('"items":[', logs)

    def test_source_own_task_span_does_not_assign_a_later_actor_or_drop_a_condition(self):
        record = {"id": "R1", "position": 0, "section": BUSINESS, "speaker": "Taylor", "text": "When I get back, I'll try to send the package off, and Morgan will deliver it to the employer."}
        for owner in ("Taylor", "Morgan"):
            with self.subTest(owner=owner):
                data = {"format": refs.FORMAT, "items": [evidence({**record, "id": "1"}, 1, "action", owners=[owner])]}
                accepted, rejected = whole.validate_evidence(data, [record], source.encode([record]), expected_format=refs.FORMAT)
                if owner == "Morgan":
                    self.assertFalse(accepted)
                    self.assertTrue(rejected)
                else:
                    self.assertFalse(rejected)
                    self.assertEqual(accepted[0]["statement"], "When I get back, I'll try to send the package off")
                    self.assertEqual(accepted[0]["quotes"][0]["text"], record["text"])


if __name__ == "__main__":
    unittest.main()
