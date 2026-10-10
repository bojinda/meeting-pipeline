"""Optional private fixture: exercise production reductions with saved answers only.

Set EDITORIAL_SAVED_RESPONSE_DIR, EDITORIAL_TEST_TOKENIZER and optionally
EDITORIAL_TEST_OUTPUT_DIR (a new private directory). No fixture text is in Git.
This is an offline regression test, not a live continuation or checkpoint importer.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import ollama_session_summary as engine
from meeting_postprocess import editorial as ed
from meeting_postprocess.commitments import commitment_evidence, source_context_evidence
from meeting_postprocess.publication import publication_payload, export_documents, export_archive


@unittest.skipUnless(os.environ.get("EDITORIAL_SAVED_RESPONSE_DIR") and os.environ.get("EDITORIAL_TEST_TOKENIZER"),
                     "requires an explicitly selected private saved response and local tokenizer")
class SavedResponseTests(unittest.TestCase):
    def test_unchanged_saved_response_reaches_held_markdown_through_production(self):
        original = Path(os.environ["EDITORIAL_SAVED_RESPONSE_DIR"])
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir() if p.is_file()}
        saved = json.loads((original / ed.RESPONSE).read_text(encoding="utf-8"))
        chunks = engine.load_jsonl(original / "meeting_sections.jsonl")
        summaries = engine.load_jsonl(original / "chunk_summaries.jsonl")
        records = ed.source_records(chunks)
        self.assertEqual(ed.digest(records), saved["source_hash"])
        self.assertEqual(records, saved["register_assessment"]["sources"])
        self.assertEqual({str(c["chunk_id"]) for c in chunks}, {str(s["chunk_id"]) for s in summaries})
        checkpoint = json.loads((original / "editorial-map-checkpoint.private.json").read_text(encoding="utf-8"))
        self.assertEqual(before["meeting_sections.jsonl"], checkpoint["sections_file_sha256"])
        self.assertEqual(before["chunk_summaries.jsonl"], checkpoint["maps_file_sha256"])
        blocked = set(checkpoint["blocked_context_source_ids"])
        prompts = ROOT / "prompts/meeting"
        budget = ed.RequestBudget(os.environ["EDITORIAL_TEST_TOKENIZER"], 196608,
                                  (prompts / "reduce_system.txt").read_text(encoding="utf-8").strip())
        calls = []
        metadata = json.loads((original / ed.DIAGNOSTICS).read_text(encoding="utf-8"))["requests"]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(os.environ["EDITORIAL_TEST_OUTPUT_DIR"]) if os.environ.get("EDITORIAL_TEST_OUTPUT_DIR") else Path(tmp) / "output"
            output.mkdir(mode=0o700)  # Never overwrite a prior run.
            def saved_answer(prompt, *, structured, num_predict, response_callback):
                stage = list(ed.OUTPUT_TOKENS)[len(calls)]
                calls.append(stage)
                self.assertEqual(num_predict, ed.OUTPUT_TOKENS[stage])
                if stage not in saved["responses"]:
                    self.assertTrue((output / "meeting-notes-draft.md").is_file())
                    raise ed.EditorialFailure("offline_no_saved_response")
                response_callback(next({k: v for k, v in row.items() if k != "stage"}
                                       for row in metadata if row["stage"] == stage))
                return saved["responses"][stage]
            with patch.object(socket.socket, "connect", side_effect=AssertionError("inference/network forbidden")), contextlib.redirect_stdout(io.StringIO()):
                status = ed.run(output, chunks, engine.build_reduce_input(summaries),
                    engine.build_reduce_input([s for s in summaries if s["meeting_section"] in {ed.BUSINESS, ed.ADJOURNMENT}]),
                    engine.build_reduce_input([s for s in summaries if s["meeting_section"] == ed.RECAP]),
                    commitment_evidence(chunks, include_context=True, blocked_context_source_ids=blocked),
                    source_context_evidence(chunks, blocked), checkpoint["aliases"], checkpoint["approved_passages"], [],
                    prompts, saved_answer, False, budget=budget)
            self.assertEqual(calls[:2], ["register", "notes"])
            review = json.loads((output / ed.REVIEW).read_text(encoding="utf-8"))
            self.assertEqual(review["status"], "review_hold")
            if "detailed" not in saved["responses"]:
                self.assertEqual(status, 1)  # Missing ancillary fixture is not a successful generation.
                self.assertEqual(review["failure_category"], "offline_no_saved_response")
            else:
                self.assertEqual(status, 0)
            copied = json.loads((output / ed.RESPONSE).read_text(encoding="utf-8"))
            for stage in ("register", "notes"):
                self.assertEqual(copied["responses"][stage], saved["responses"][stage])
            original_notes = json.loads(saved["responses"]["notes"])
            evidence = json.loads((output / ed.NOTES_EVIDENCE).read_text(encoding="utf-8"))
            self.assertEqual(evidence["sources"], records)
            section_flags = [f for f in review["findings"] if f["code"] == "notes_source_section_review"]
            expected = []
            for location, block in ed.notes_blocks(original_notes):
                allowed = {ed.RECAP} if location.startswith("previous_context/") else {ed.BUSINESS, ed.ADJOURNMENT}
                wrong = [key for key in block["source_ids"] if records[key]["section"] not in allowed]
                if wrong:
                    expected.append((location, wrong))
            self.assertEqual([(f["notes_location"], f["mismatched_source_ids"]) for f in section_flags], expected)
            document = (output / "meeting-notes-draft.md").read_text(encoding="utf-8")
            self.assertIn("DRAFT — REVIEW REQUIRED", document)
            for _, block in ed.notes_blocks(evidence["notes"]):
                self.assertIn(block["text"], document)
            self.assertNotIn("| Undertaking |", document)
            for export in (lambda: publication_payload(output), lambda: export_documents(output, output / "export"),
                           lambda: export_archive(output, output / "export.zip")):
                with self.assertRaisesRegex(ValueError, "hold"):
                    export()
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir() if p.is_file()})
            ed.write_private_json(output / "offline-test.private.json", {
                "mode": "saved_responses_in_production_editorial_run", "inference_calls": 0,
                "response_sha256": hashlib.sha256(saved["responses"]["notes"].encode()).hexdigest(),
                "response_characters": len(saved["responses"]["notes"]), "source_hash": ed.digest(records),
                "word_count": len(document.split()), "section_review_blocks": len(section_flags),
                "mismatched_references": sum(len(f["mismatched_source_ids"]) for f in section_flags),
                "original_files_unchanged": True, "publication_status": review["status"], "pipeline_status": status,
                "limitation": "Offline responses only; absent ancillary responses are deliberately unavailable. No historical authority is inferred."})


if __name__ == "__main__":
    unittest.main()
