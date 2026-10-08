"""Synthetic strict schema contracts and private source-bound offline debugging."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import ollama_session_summary as engine
from meeting_postprocess import whole_synthesis as whole
from meeting_postprocess import whole_source as source
from meeting_postprocess.sections import BUSINESS, RECAP, ADJOURNMENT
from meeting_postprocess.publication import publication_payload, export_archive, PUBLIC_MEETING_FILENAMES
import test_whole_synthesis as existing
import test_whole_source as fixtures


class EvidenceContractTests(unittest.TestCase):
    def validate(self, rows):
        records = fixtures.originals()
        original = copy.deepcopy(rows)
        result = whole.validate_evidence({"items": rows}, records, source.encode(records))
        self.assertEqual(rows, original)
        return result

    def example(self):
        return fixtures.evidence(fixtures.originals()[0])["items"][0]

    def test_model_schema_and_validator_share_id_and_canonical_section_contract(self):
        props = whole.extraction_schema()["properties"]["items"]["items"]["properties"]
        self.assertEqual(props["id"]["pattern"], whole.EVIDENCE_ID_PATTERN)
        self.assertEqual((props["id"]["minLength"], props["id"]["maxLength"]), (2, 7))
        self.assertEqual(set(props["section"]["enum"]), {BUSINESS, RECAP, ADJOURNMENT})
        self.assertIn("never output B/R/A", whole.EXTRACT_SYSTEM)
        for value in ("E0", "E1", "E15", "E999999", "1", "topic_1", "e1", "E1234567"):
            with self.subTest(value=value):
                row = self.example()
                row["id"] = value
                accepted, rejected = self.validate([row])
                self.assertEqual(bool(accepted), bool(re.fullmatch(props["id"]["pattern"], value)))
                if rejected:
                    self.assertEqual(rejected[0]["category"], "invalid_evidence_id_format")

    def test_compact_source_number_as_evidence_id_and_section_code_are_not_reinterpreted(self):
        row = self.example()
        row.update(id="1", section="R")
        accepted, rejected = self.validate([row])
        self.assertFalse(accepted)
        self.assertEqual(rejected, [{"item_index": 0, "category": "invalid_evidence_id_format", "field": "id"}, {"item_index": 0, "category": "invalid_section", "field": "section"}])
        self.assertEqual(whole.rejection_summary(rejected), {"rejected_item_count": 1, "category_counts": {"invalid_evidence_id_format": 1, "invalid_section": 1}})

    def test_field_specific_rejections_never_include_untrusted_values(self):
        mutations = (
            ("missing_or_extra_fields", lambda row: row.pop("owners")),
            ("missing_or_extra_fields", lambda row: row.update(PRIVATE_EXTRA_KEY="PRIVATE_MODEL_VALUE")),
            ("invalid_kind", lambda row: row.update(kind="PRIVATE_KIND")),
            ("invalid_section", lambda row: row.update(section="B")),
            ("invalid_statement", lambda row: row.update(statement=None)),
            ("invalid_statement", lambda row: row.update(statement=" ")),
            ("invalid_statement", lambda row: row.update(statement="x" * 6001)),
            ("invalid_owners", lambda row: row.update(owners="Taylor")),
            ("invalid_owners", lambda row: row.update(owners=[3])),
            ("invalid_owners", lambda row: row.update(owners=[" "])),
            ("invalid_role_fields", lambda row: row.update(mover=["PRIVATE_MOVER"])),
            ("invalid_role_fields", lambda row: row.update(seconder=" ")),
            ("invalid_role_fields", lambda row: row.update(outcome="PRIVATE_OUTCOME")),
        )
        for category, mutate in mutations:
            with self.subTest(category=category):
                row = self.example()
                mutate(row)
                accepted, rejected = self.validate([row])
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["category"], category)
                self.assertEqual(rejected[0]["item_index"], 0)
                encoded = json.dumps(rejected)
                for value in ("PRIVATE_", "Taylor", "Repeated exact text", "Mac yard"):
                    self.assertNotIn(value, encoded)

    def test_duplicate_id_is_reported_even_when_first_claim_is_rejected(self):
        row = self.example()
        bad = {**row, "statement": "Unsupported statement."}
        accepted, rejected = self.validate([bad, row])
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["category"], "unsupported_or_paraphrased_claim")
        self.assertEqual(rejected[1], {"item_index": 1, "category": "duplicate_evidence_id", "field": "id"})

    def test_multiple_bad_role_fields_count_once_per_item_in_aggregate(self):
        row = self.example()
        row.update(mover=3, seconder=4)
        accepted, rejected = self.validate([row])
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(whole.rejection_summary(rejected)["category_counts"], {"invalid_role_fields": 1})

    def test_fifteen_schema_rejections_are_safe_and_do_not_start_document_synthesis(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root, existing.SOURCE + "[SPEAKER_01] " + "Routine report. " * 15)
            def mutate(data, records):
                data["items"] = data["items"][:15]
                for row in data["items"]:
                    row.update(id="PRIVATE_INVALID_ID", section="B")
            model = existing.FakeModel(mutate_evidence=mutate)
            status, output, logs, _ = existing.invoke(root, directory, model)
            self.assertEqual(status, 1)
            self.assertEqual(len(model.requests), 1)
            report = json.loads((output / "whole-run.json").read_text())
            self.assertEqual(report["evidence_validation"]["rejected_item_count"], 15)
            self.assertEqual(report["evidence_validation"]["category_counts"], {"invalid_evidence_id_format": 15, "invalid_section": 15})
            self.assertNotIn("PRIVATE_INVALID_ID", logs + json.dumps(report["evidence_validation"]) + json.dumps(report["rejected_evidence"]))
            self.assertEqual(publication_payload(output)["documents"], [])


class PrivateResponseTests(unittest.TestCase):
    def replay(self, root, directory, path):
        stdout = io.StringIO()
        with patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(Path(root) / "production")}, clear=True), patch.object(sys, "argv", ["summary", str(directory), "--synthesis-mode", "whole", "--synthesis-validate-response", str(path)]), patch.object(whole.local, "_http_call") as model, patch.object(engine, "call_ollama") as reduce, patch.object(whole.subprocess, "run") as gpu, patch.object(whole, "write_private_json") as writer, patch.object(whole, "make_counter") as tokenizer, contextlib.redirect_stdout(stdout):
            status = engine.main()
            for tool in (model, reduce, gpu, writer, tokenizer):
                tool.assert_not_called()
        return status, json.loads(stdout.getvalue())

    def test_retention_is_explicit_and_response_is_private_gitignored_and_not_exported(self):
        for retain in (False, True):
            with self.subTest(retain=retain), tempfile.TemporaryDirectory() as root:
                directory = existing.create_session(root)
                status, output, _, _ = existing.invoke(root, directory, flags=["--synthesis-retain-response"] if retain else [])
                self.assertEqual(status, 0)
                path = output / whole.PRIVATE_RESPONSE_FILENAME
                self.assertEqual(path.exists(), retain)
                if retain:
                    if os.name == "posix":
                        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    cache = json.loads(path.read_text())
                    self.assertIsInstance(json.loads(cache["response"]), dict)
                    self.assertFalse(any("session" in value for value in cache["binding"].values()))
                    ignored = subprocess.run(["git", "check-ignore", "--no-index", "private/" + path.name], cwd=ROOT, capture_output=True)
                    self.assertEqual(ignored.returncode, 0)
                    with (output / "summary.md").open("a") as doc:
                        doc.write("\n[private](" + path.name + ")")
                    self.assertNotIn(path.name, json.dumps(publication_payload(output)))
                    export_archive(output, Path(root) / "public.zip")
                    with zipfile.ZipFile(Path(root) / "public.zip") as archive:
                        self.assertEqual(set(archive.namelist()), set(PUBLIC_MEETING_FILENAMES))

    def test_successful_offline_validation_is_read_only_and_does_not_generate_documents(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root)
            status, output, _, _ = existing.invoke(root, directory, flags=["--synthesis-retain-response"])
            self.assertEqual(status, 0)
            cache = output / whole.PRIVATE_RESPONSE_FILENAME
            # Old schema/prompt digests are provenance, not permission to relax
            # current validation. Offline debugging must survive validator fixes.
            data = json.loads(cache.read_text())
            data["schema_digest"] = "0" * 64
            cache.write_text(json.dumps(data))
            before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in Path(root).rglob("*") if path.is_file()}
            status, report = self.replay(root, directory, cache)
            self.assertEqual(status, 0)
            self.assertEqual(report["status"], "passed")
            after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in Path(root).rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            self.assertNotIn("Taylor", json.dumps(report))
            self.assertNotIn("My name", json.dumps(report))

    def test_failed_schema_response_can_be_revalidated_without_inference_or_publication(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root)
            def mutate(data, records):
                for row in data["items"]:
                    row["id"] = "invalid-source-style-id"
            model = existing.FakeModel(mutate_evidence=mutate)
            status, output, _, _ = existing.invoke(root, directory, model, flags=["--synthesis-retain-response"])
            self.assertEqual(status, 1)
            self.assertEqual(len(model.requests), 1)
            status, report = self.replay(root, directory, output / whole.PRIVATE_RESPONSE_FILENAME)
            self.assertEqual(status, 1)
            self.assertEqual(report["status"], "needs_review")
            self.assertIn("invalid_evidence_id_format", report["category_counts"])
            self.assertEqual(publication_payload(output)["documents"], [])
            self.assertFalse((directory / "speaker_turn_corrections.json").exists())

    def test_stale_foreign_or_edited_response_binding_cannot_be_reused(self):
        for case in ("source", "alias", "foreign", "body"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                directory = existing.create_session(root)
                _, output, _, _ = existing.invoke(root, directory, flags=["--synthesis-retain-response"])
                path = output / whole.PRIVATE_RESPONSE_FILENAME
                if case == "source":
                    index = directory / "chunks_out" / "transcript_chunks.jsonl"
                    index.write_text(index.read_text().replace("safety notice", "safety report"))
                elif case == "alias":
                    (directory / "speaker_aliases.json").write_text(json.dumps({**existing.ALIASES, "SPEAKER_01": "Riley"}))
                elif case == "foreign":
                    other = Path(root) / "other"
                    directory = existing.create_session(other)
                else:
                    data = json.loads(path.read_text())
                    data["response"] = '{"items":[]}'
                    path.write_text(json.dumps(data))
                status, report = self.replay(root, directory, path)
                self.assertEqual(status, 2)
                self.assertIn(report["failure_category"], {"retained_source_binding_mismatch", "retained_response_integrity_mismatch"})

    def test_generation_limit_response_stays_unusable_even_when_json_is_complete(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root)
            model = Mock(return_value={"response": '{"items":[]}', "done_reason": "length", "eval_count": 16384})
            status, output, _, _ = existing.invoke(root, directory, model, flags=["--synthesis-retain-response"])
            self.assertEqual(status, 1)
            model.assert_called_once()
            status, report = self.replay(root, directory, output / whole.PRIVATE_RESPONSE_FILENAME)
            self.assertEqual(status, 2)
            self.assertEqual(report["failure_category"], "generation_token_limit")

    def test_retention_does_not_disable_safe_context_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            directory = existing.create_session(root, existing.SOURCE + "[SPEAKER_01] " + "Context. " * 3000)
            def summarize(**kwargs):
                return "# Output\nNone noted."
            model = Mock()
            status, output, _, _ = existing.invoke(root, directory, model, flags=["--synthesis-retain-response", "--synthesis-num-ctx", "8192"], summary=summarize)
            self.assertEqual(status, 0)
            model.assert_not_called()
            self.assertFalse((output / whole.PRIVATE_RESPONSE_FILENAME).exists())
            self.assertEqual(json.loads((output / "whole-run.json").read_text())["processing_mode"], "map_reduce_fallback")


if __name__ == "__main__":
    unittest.main()
