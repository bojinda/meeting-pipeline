"""Synthetic session-index ingestion, completion diagnostics, and private leads."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import speaker_review_worker as worker
import transcript_chunker as chunker
import test_speaker_suggestions as existing
from test_speaker_llm_review import proposal, SELF_SOURCE
from meeting_postprocess import speaker_review as review
from meeting_postprocess.speaker_suggestions import build_speaker_suggestions, PRIVATE_SUGGESTIONS_FILENAME


def record(text, identifier=1):
    return {"chunk_id": identifier, "file_name": f"chunk-{identifier}.md", "chunk_type": "discussion",
            "speaker_span": "SPEAKER_00 -> SPEAKER_15", "speaker_count": 2,
            "start_time": float(identifier * 10), "end_time": float(identifier * 10 + 9),
            "word_count": len(text.split()), "text": text}


class SessionIndexTests(unittest.TestCase):
    def run_index(self, directory, records, model_response=None):
        root = Path(directory)
        helper = existing.SpeakerCommandTests()
        transcript = helper.setup_meeting(root)
        index = transcript / "chunks_out" / "transcript_chunks.jsonl"
        index.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        before = index.read_bytes()
        with patch.object(review, "call_local_ollama", return_value=model_response) as model:
            helper.run_command(root, ["suggest", transcript, *(["--llm"] if model_response is not None else [])])
            if model_response is None:
                model.assert_not_called()
            else:
                model.assert_called_once()
        self.assertEqual(index.read_bytes(), before)
        self.assertFalse((transcript / "speaker_aliases.json").exists())
        path = root / "outputs" / transcript.name / PRIVATE_SUGGESTIONS_FILENAME
        return json.loads(path.read_text()), transcript, path

    def test_bare_questions_require_independent_prior_identity(self):
        for word in ("Pardon", "Right", "Taylor"):
            with self.subTest(word=word), tempfile.TemporaryDirectory() as directory:
                report, _, _ = self.run_index(directory, [record(f"[SPEAKER_00] {word}?\n[SPEAKER_15] Yes.")])
                self.assertFalse(any(row["candidates"] for row in report["suggestions"]))

    def test_prior_introduction_or_roster_can_support_bare_address(self):
        source = "[SPEAKER_00] Taylor?\n[SPEAKER_15] Yes."
        for records, roster in (([record("[SPEAKER_00] Let me introduce Taylor Morgan.\n[SPEAKER_04] The report is missing."), record(source, 2)], []),
                                ([record(source)], [{"name": "Taylor Morgan", "aliases": [], "role": ""}])):
            with self.subTest(roster=bool(roster)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                # The command discovers the per-session private roster.
                (root / "session").mkdir()
                (root / "session" / "speaker_roster.private.json").write_text(json.dumps(roster))
                report, _, _ = self.run_index(directory, records)
                row = next(row for row in report["suggestions"] if row["speaker_label"] == "SPEAKER_15")
                self.assertEqual(row["candidates"], ["Taylor Morgan"])

    def test_future_mention_does_not_retroactively_validate_bare_question(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _ = self.run_index(directory, [record("[SPEAKER_00] Taylor?\n[SPEAKER_15] Yes.\n[SPEAKER_00] Let me introduce Taylor Morgan.")])
            row = next(row for row in report["suggestions"] if row["speaker_label"] == "SPEAKER_15")
            self.assertEqual(row["candidates"], [])

    def test_non_identity_demonstrative_does_not_attest_a_bare_question(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _ = self.run_index(directory, [record("[SPEAKER_00] This is Right.\n[SPEAKER_04] The report is missing.\n[SPEAKER_00] Right?\n[SPEAKER_15] Yes.")])
            self.assertFalse(any(row["candidates"] for row in report["suggestions"]))

    def test_chunker_merged_exchange_retains_identity_and_later_collision(self):
        segments = [{"speaker": "SPEAKER_15", "start": index, "end": index + 0.9, "text": text}
                    for index, text in enumerate(("That was me. I'm just, I don't know if I missed the introduction.",
                                                  "Who's that with Morgan and Riley in the room there? That's Taylor. Taylor?",
                                                  "Yeah, that's me.", "Taylor Morgan."))]
        turns = chunker.merge_segments_into_turns(segments)
        self.assertEqual(len(turns), 1)
        source = "\n".join(f"[{turn['speaker']}] {turn['text']}" for turn in turns)
        address = "[SPEAKER_00] Okay, if there's no questions for Morgan, Casey, I've just got a question for you there. Do you happen to have the financials?\n[SPEAKER_15] Yes, sorry, I can't access them today."
        with tempfile.TemporaryDirectory() as directory:
            report, transcript, path = self.run_index(directory, [record(source), record(address, 2)])
            row = next(row for row in report["suggestions"] if row["speaker_label"] == "SPEAKER_15")
            self.assertEqual(set(row["candidates"]), {"Taylor Morgan", "Casey"})
            self.assertIsNone(row["suggested_name"])
            self.assertTrue(row["ambiguity"])
            self.assertTrue(any(item["uncertain_source_attribution"] for item in row["evidence"]))
            row.update(suggested_name="Taylor Morgan", confidence="high", ambiguity=[], candidates=["Taylor Morgan"])
            path.write_text(json.dumps(report))
            with self.assertRaises(SystemExit):
                existing.SpeakerCommandTests().run_command(Path(directory), ["approve", transcript, "--approve", "SPEAKER_15"])

    def test_merged_exchange_alone_is_not_high_confidence_or_approvable(self):
        with tempfile.TemporaryDirectory() as directory:
            report, transcript, path = self.run_index(directory, [record("[SPEAKER_15] Who's that? Yeah, that's me. Taylor Morgan.")])
            row = report["suggestions"][0]
            self.assertEqual(row["candidates"], ["Taylor Morgan"])
            self.assertIsNone(row["suggested_name"])
            row.update(suggested_name="Taylor Morgan", confidence="high", ambiguity=[])
            path.write_text(json.dumps(report))
            with self.assertRaises(SystemExit):
                existing.SpeakerCommandTests().run_command(Path(directory), ["approve", transcript, "--approve", "SPEAKER_15"])

    def test_direct_self_name_inside_merged_identity_exchange_is_uncertain(self):
        for source in ("[SPEAKER_15] Who's that? I'm Taylor Morgan.", "[SPEAKER_15] Who is speaking? My name is Taylor Morgan.",
                       "[SPEAKER_15] [SPEAKER_04] Hello. My name is Taylor Morgan."):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                report, _, _ = self.run_index(directory, [record(source)])
                row = report["suggestions"][0]
                self.assertEqual(row["candidates"], ["Taylor Morgan"])
                self.assertIsNone(row["suggested_name"])
                self.assertTrue(row["ambiguity"])

    def test_redaction_and_unrelated_sentences_do_not_join_merged_identity(self):
        for middle in ("The budget is ready.", "Redact the following. PRIVATE_MERGED_IDENTITY. End redaction."):
            with self.subTest(middle=middle), tempfile.TemporaryDirectory() as directory:
                report, _, _ = self.run_index(directory, [record("[SPEAKER_15] Who's that? Yeah, that's me. " + middle + " Taylor Morgan.")])
                self.assertEqual(report["suggestions"][0]["candidates"], [])
                self.assertNotIn("PRIVATE_MERGED_IDENTITY", json.dumps(report))

    def test_unverified_llm_lead_survives_privately_but_not_fresh_approval(self):
        source = "[SPEAKER_00] The next account belongs to Taylor Morgan.\n[SPEAKER_15] Those figures are ready."
        response = json.dumps(proposal("Taylor Morgan", "SPEAKER_15", ["E000000", "E000001"], "introduction", "low"))
        with tempfile.TemporaryDirectory() as directory:
            report, transcript, path = self.run_index(directory, [record(source)], response)
            row = next(row for row in report["suggestions"] if row["speaker_label"] == "SPEAKER_15")
            self.assertEqual(row["candidates"], [])
            self.assertIsNone(row["suggested_name"])
            self.assertEqual(row["unverified_leads"][0]["name"], "Taylor Morgan")
            self.assertEqual(row["unverified_leads"][0]["evidence_ids"], ["E000000", "E000001"])
            self.assertFalse(row["unverified_leads"][0]["approvable"])
            row.update(suggested_name="Taylor Morgan", confidence="high", ambiguity=[], candidates=["Taylor Morgan"])
            path.write_text(json.dumps(report))
            with self.assertRaises(SystemExit):
                existing.SpeakerCommandTests().run_command(Path(directory), ["approve", transcript, "--approve", "SPEAKER_15"])


class CompletionDiagnosticsTests(unittest.TestCase):
    def review(self, completion, text=SELF_SOURCE):
        chunks = [record(text)]
        call = Mock(side_effect=completion) if isinstance(completion, Exception) else Mock(return_value=completion)
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            result = review.review_speakers(build_speaker_suggestions(chunks), chunks, {}, [], ollama_url="http://127.0.0.1:11434", model="local-review-model", num_ctx=32768, call=call)
        call.assert_called_once()
        self.assertEqual(out.getvalue() + err.getvalue(), "")
        return result

    def test_empty_malformed_schema_and_grounding_have_distinct_categories(self):
        for completion, category in ((" ", "empty_generated_json"), ("PRIVATE_BROKEN_MODEL_OUTPUT", "malformed_generated_json"),
                                     ('{"suggestions":{}}', "invalid_schema"), (json.dumps(proposal("Casey")), "grounding_validation_failure"),
                                     (json.dumps(proposal(confidence=[])), "invalid_schema"),
                                     (json.dumps(proposal(references="E000000")), "invalid_schema"),
                                     (json.dumps(proposal(label="SPEAKER_99")), "grounding_validation_failure")):
            with self.subTest(category=category):
                result = self.review(completion)
                info = result["llm_review"]
                self.assertEqual(info["diagnostics"]["failure_category"], category)
                self.assertEqual(info["diagnostics"]["output_length"], len(completion))
                self.assertNotIn("PRIVATE_BROKEN_MODEL_OUTPUT", json.dumps(result))
                if category in {"invalid_schema", "grounding_validation_failure"}:
                    self.assertTrue(info["diagnostics"]["validation_error_categories"])

    def test_token_limit_is_visible_even_with_parseable_output(self):
        for response in ("{", json.dumps(proposal())):
            with self.subTest(response=response):
                result = self.review({"response": response, "done_reason": "length", "eval_count": 8192, "thinking": "PRIVATE_THINKING"})
                info = result["llm_review"]
                self.assertEqual(info["diagnostics"]["failure_category"], "generation_token_limit")
                self.assertEqual(info["diagnostics"]["done_reason"], "length")
                self.assertEqual(info["diagnostics"]["thinking_length"], len("PRIVATE_THINKING"))
                self.assertEqual(info["diagnostics"]["eval_count"], 8192)
                self.assertIsNone(info["response"])
                self.assertNotIn("PRIVATE_THINKING", json.dumps(result))
                self.assertEqual(result["suggestions"][0]["suggested_name"], "Taylor Morgan")

    def test_model_transport_and_timeout_errors_do_not_echo_error_text(self):
        for completion, category in (({"error": "PRIVATE_MODEL_ERROR"}, "ollama_model_error"), (TimeoutError("PRIVATE_TIMEOUT"), "ollama_timeout"),
                                     (urllib.error.URLError("PRIVATE_TRANSPORT"), "ollama_transport_error"),
                                     (urllib.error.HTTPError("http://localhost", 500, "PRIVATE_HTTP_ERROR", {}, None), "ollama_http_error")):
            with self.subTest(category=category):
                result = self.review(completion)
                self.assertEqual(result["llm_review"]["status"], "unavailable")
                self.assertEqual(result["llm_review"]["diagnostics"]["failure_category"], category)
                self.assertNotIn("PRIVATE_", json.dumps(result))

    def test_success_retains_only_safe_completion_metadata(self):
        result = self.review({"response": json.dumps(proposal()), "done_reason": "stop", "eval_count": 123, "thinking": "PRIVATE_THOUGHT"})
        self.assertEqual(result["llm_review"]["status"], "completed")
        self.assertIsNone(result["llm_review"]["diagnostics"]["failure_category"])
        self.assertEqual(result["llm_review"]["diagnostics"]["done_reason"], "stop")
        self.assertNotIn("PRIVATE_THOUGHT", json.dumps(result))

    def test_leads_require_existing_ids_labels_and_transcript_names(self):
        for response in (proposal("Casey", references=["fake"]), proposal("Casey"), proposal("Taylor Morgan", label="SPEAKER_99")):
            with self.subTest(response=response):
                result = self.review(json.dumps(response))
                self.assertFalse(any(row.get("unverified_leads") for row in result["suggestions"]))

    def test_worker_carries_completion_metadata_and_safe_failure_category(self):
        completion = {"response": "{}", "done_reason": "length", "eval_count": 8192}
        with patch.object(sys, "stdin", io.StringIO("{}")), patch.object(worker, "_http_call", return_value=completion), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(worker.main(), 0)
        self.assertEqual(json.loads(out.getvalue()), completion)
        with patch.object(sys, "stdin", io.StringIO("{}")), patch.object(worker, "_http_call", side_effect=TimeoutError("PRIVATE_EXCEPTION")), contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(worker.main(), 1)
        self.assertEqual(json.loads(out.getvalue())["failure"]["failure_category"], "ollama_timeout")
        self.assertNotIn("PRIVATE_EXCEPTION", out.getvalue() + err.getvalue())

    def test_supervised_worker_metadata_survives_parent_boundary(self):
        request = {"ollama_url": "http://localhost:11434", "model": "local-review-model", "num_ctx": 32768, "prompt": "{}"}
        for packet, code in (({"response": "{}", "done_reason": "length", "eval_count": 8192}, 0), ({"failure": {"failure_category": "ollama_timeout"}}, 1)):
            with self.subTest(code=code), patch.dict(os.environ, {}, clear=True), patch.object(review.sys, "platform", "linux"), patch.object(review.subprocess, "run", return_value=subprocess.CompletedProcess([], code, json.dumps(packet), "")) as process:
                if code:
                    with self.assertRaises(review.ReviewFailure) as failure:
                        review.call_local_ollama(request)
                    self.assertEqual(failure.exception.category, "ollama_timeout")
                else:
                    self.assertEqual(review.call_local_ollama(request), packet)
                process.assert_called_once()

    def test_thinking_setting_and_large_bounded_budget_are_review_only(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return b'{"response":"{}","done_reason":"stop"}'
        for value, setting in (("false", False), ("true", True), ("default", None)):
            with self.subTest(value=value), patch.dict(os.environ, {"SPEAKER_REVIEW_THINK": value}, clear=True), patch.object(review.urllib.request, "build_opener") as builder:
                builder.return_value.open.return_value = Response()
                chunks = [record(SELF_SOURCE)]
                review.review_speakers(build_speaker_suggestions(chunks), chunks, {}, [], ollama_url="http://localhost:11434", model="local-review-model", num_ctx=32768, call=review._http_call)
                builder.return_value.open.assert_called_once()
                payload = json.loads(builder.return_value.open.call_args.args[0].data)
                self.assertEqual(payload.get("think"), setting)
                self.assertEqual(payload["format"], review.RESPONSE_SCHEMA)
                self.assertEqual(payload["options"]["num_predict"], 8192)


if __name__ == "__main__":
    unittest.main()
