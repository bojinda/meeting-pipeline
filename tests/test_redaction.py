from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import export_meeting
import ollama_session_summary as engine
from meeting_postprocess.publication import export_archive, export_documents, publication_payload, public_meeting_files
from meeting_postprocess.redaction import redact_chunks, write_private_redactions


def chunk(text, identifier=1, start=0.0, end=60.0, **extras):
    return {"chunk_id": identifier, "file_name": f"{identifier}.md", "speaker_span": "SPEAKER_01 -> SPEAKER_02", "chunk_type": "discussion", "start_time": start, "end_time": end, "text": text, **extras}


class RedactionTests(unittest.TestCase):
    def test_redaction_within_one_turn_preserves_surrounding_text(self):
        result = redact_chunks([chunk("[SPEAKER_01] Current business. Redact the following. Sensitive information. End redaction. Back to the agenda.")])
        self.assertRegex(result.chunks[0]["text"], r"Current business\.\s+Back to the agenda\.")
        self.assertEqual(result.redactions[0]["content"], "[SPEAKER_01] Sensitive information.")
        self.assertEqual(result.warnings, [])

    def test_markers_in_middle_of_turn_without_sentence_breaks(self):
        result = redact_chunks([chunk("[SPEAKER_01] Before redact the following SECRET words end redaction after")])
        self.assertRegex(result.chunks[0]["text"], r"Before\s+after")
        self.assertEqual(result.redactions[0]["content"], "[SPEAKER_01] SECRET words")

    def test_redaction_across_speakers_and_turns(self):
        result = redact_chunks([chunk("[SPEAKER_01] Current business. Redact the following. Sensitive information.\n[SPEAKER_02] More sensitive information.\n[SPEAKER_01] End redaction. Back to the agenda.")])
        self.assertEqual(result.chunks[0]["text"], "[SPEAKER_01] Current business.\n[SPEAKER_01] Back to the agenda.")
        self.assertEqual(result.redactions[0]["content"], "[SPEAKER_01] Sensitive information.\n[SPEAKER_02] More sensitive information.")
        self.assertEqual(result.redactions[0]["speakers"], ["SPEAKER_01", "SPEAKER_02"])

    def test_redaction_across_source_chunks_has_container_metadata(self):
        result = redact_chunks([
            chunk("[SPEAKER_03] Public. Redact the following. SECRET A.", "4.2", 1832.4, 1880.0),
            chunk("[SPEAKER_07] SECRET B. End redaction. Public again.", "5.1", 1880.0, 1944.8),
        ])
        record = result.redactions[0]
        self.assertEqual(record, {"number": 1, "start_chunk_id": "4.2", "end_chunk_id": "5.1", "start_time": 1832.4, "end_time": 1944.8, "closed": True, "speakers": ["SPEAKER_03", "SPEAKER_07"], "reason": "explicit_spoken_redaction", "content": "[SPEAKER_03] SECRET A.\n[SPEAKER_07] SECRET B."})
        self.assertNotIn("SECRET", json.dumps(result.chunks))

    def test_capitalization_and_punctuation_around_exact_commands(self):
        result = redact_chunks([chunk('[SPEAKER_01] Public. (ReDaCt THE FoLLoWiNg): SECRET. [END REDACTION]! Back.')])
        self.assertEqual(result.redactions[0]["content"], "[SPEAKER_01] SECRET.")
        self.assertNotIn("SECRET", result.chunks[0]["text"])
        self.assertNotIn("[END", result.chunks[0]["text"])
        self.assertNotIn("(", result.chunks[0]["text"])

    def test_exact_phrase_can_span_source_boundaries(self):
        result = redact_chunks([
            chunk("[SPEAKER_01] Before. redact the", 1),
            chunk("[SPEAKER_01] following. SECRET.", 2),
            chunk("[SPEAKER_02] end", 3),
            chunk("[SPEAKER_02] redaction. After.", 4),
        ])
        self.assertEqual(result.redactions[0]["start_chunk_id"], 1)
        self.assertEqual(result.redactions[0]["end_chunk_id"], 4)
        self.assertEqual(result.redactions[0]["content"], "[SPEAKER_01] SECRET.")
        self.assertNotIn("SECRET", json.dumps(result.chunks))

    def test_no_fuzzy_or_semantic_marker_matching(self):
        text = "[SPEAKER_01] redact following. redact, the following. redact the followings. stop redacting. end-redaction."
        result = redact_chunks([chunk(text)])
        self.assertEqual(result.redactions, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.chunks[0]["text"], text)

    def test_repeated_start_does_not_close_or_restart_redaction(self):
        result = redact_chunks([chunk("[SPEAKER_01] Redact the following. SECRET ONE. Redact the following. SECRET TWO. End redaction. Public.")])
        self.assertEqual(len(result.redactions), 1)
        self.assertTrue(result.redactions[0]["closed"])
        self.assertIn("SECRET ONE", result.redactions[0]["content"])
        self.assertIn("SECRET TWO", result.redactions[0]["content"])
        self.assertNotIn("SECRET", result.chunks[0]["text"])
        self.assertEqual([warning.code for warning in result.warnings], ["redaction_repeated_start"])
        self.assertEqual(result.warnings[0].excerpt, "")

    def test_unmatched_end_keeps_surrounding_content(self):
        result = redact_chunks([chunk("[SPEAKER_01] Before. End redaction. After.")])
        self.assertEqual(result.redactions, [])
        self.assertRegex(result.chunks[0]["text"], r"Before\.\s+After\.")
        self.assertEqual([warning.code for warning in result.warnings], ["redaction_unmatched_end"])

    def test_unclosed_redaction_extends_through_eof(self):
        result = redact_chunks([chunk("[SPEAKER_01] Public. Redact the following. SECRET.", 2), chunk("[SPEAKER_02] PRIVATE EOF.", 3, 60, 90)])
        record = result.redactions[0]
        self.assertFalse(record["closed"])
        self.assertEqual(record["reason"], "unclosed_at_eof")
        self.assertEqual(record["end_chunk_id"], 3)
        self.assertEqual(record["end_time"], 90)
        self.assertEqual(record["content"], "[SPEAKER_01] SECRET.\n[SPEAKER_02] PRIVATE EOF.")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0]["text"], "[SPEAKER_01] Public.")

    def test_multiple_records_are_sequential_with_optional_timestamps(self):
        result = redact_chunks([chunk("[SPEAKER_01] redact the following SECRET ONE end redaction Public. redact the following SECRET TWO end redaction", start=None, end=None)])
        self.assertEqual([record["number"] for record in result.redactions], [1, 2])
        self.assertIsNone(result.redactions[0]["start_time"])
        self.assertEqual(result.redactions[1]["content"], "[SPEAKER_01] SECRET TWO")

    def test_exact_private_text_is_not_normalized_or_aliased(self):
        text = "Mackyard  Transport Canadaâ€™s hypodermical policy."
        result = redact_chunks([chunk("[SPEAKER_03] redact the following " + text + " end redaction")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_private_redactions(path, result.redactions)
            document = json.loads((path / "redactions.json").read_text(encoding="utf-8"))
            self.assertEqual(document["redactions"][0]["content"], "[SPEAKER_03] " + text)
            self.assertEqual(document["redactions"][0]["speakers"], ["SPEAKER_03"])

    def test_auxiliary_raw_metadata_cannot_leak(self):
        result = redact_chunks([chunk("[SPEAKER_01] Public. Redact the following. SECRET. End redaction.", raw_text="SECRET", segments=[{"text": "SECRET"}], speaker_span="PRIVATE LABEL")])
        self.assertNotIn("SECRET", json.dumps(result.chunks))
        self.assertNotIn("PRIVATE LABEL", json.dumps(result.chunks))
        self.assertNotIn("raw_text", result.chunks[0])

    @unittest.skipIf(os.name == "nt", "Windows output inherits directory ACLs; chmod does not enforce POSIX ownership")
    def test_private_file_permissions_are_owner_only_even_when_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "redactions.json").write_text("old", encoding="utf-8")
            os.chmod(path / "redactions.json", 0o644)
            write_private_redactions(path, [])
            self.assertEqual((path / "redactions.json").stat().st_mode & 0o777, 0o600)


class RedactionPipelineTests(unittest.TestCase):
    def run_pipeline(self, directory, chunks, profile="meeting", fail_map=False):
        root = Path(directory)
        transcript = root / "transcript" / "private-session"
        chunk_dir = transcript / "chunks_out"
        chunk_dir.mkdir(parents=True)
        original = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks)
        index = chunk_dir / "transcript_chunks.jsonl"
        index.write_text(original, encoding="utf-8")
        (transcript / "speaker_aliases.json").write_text('{"SPEAKER_01":"Morgan","SPEAKER_02":"Riley"}', encoding="utf-8")
        recording = root / "original.wav"
        recording.write_bytes(b"original private recording")
        output = root / "summaries" / transcript.name
        output.mkdir(parents=True)
        # Simulate a previous run before spoken redaction was enabled.
        for name in ("summary.md", "action-items.md", "minutes-draft.md", "chunk_summaries.jsonl", "meeting_sections.jsonl", "minutes-qa.json"):
            (output / name).write_text("STALE PRIVATE CONTENT", encoding="utf-8")
        calls = []

        def fake_ollama(**kwargs):
            calls.append(kwargs)
            if fail_map:
                raise RuntimeError("Mocked model failure")
            # Echo all inputs, making any input leak visible downstream too.
            prompt = kwargs["prompt"]
            if "Transcript chunk:\n" in prompt:
                return "## Topics\n" + prompt.split("Transcript chunk:\n", 1)[1].split("\nWrite concise markdown", 1)[0]
            if "Summarize only the supplied historical recap" in prompt:
                return "- Recap of Previous Meeting: " + prompt.split("Previous-meeting chunk summaries:\n", 1)[1]
            if "write formal draft minutes" in prompt:
                return "# Draft Minutes\n## Overview\n" + prompt.split("Chunk summaries:\n", 1)[1]
            return "# Generated Document\n" + prompt.split("Chunk summaries:\n", 1)[1]

        capture = io.StringIO()
        with patch.object(sys, "argv", ["summarizer", str(transcript), "--profile", profile, "--keep-recap"]), patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(root / "summaries"), "LESSON_SUMMARIES_ROOT": str(root / "summaries")}, clear=True), patch.object(engine, "call_ollama", side_effect=fake_ollama), contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            status = engine.main()
        self.assertEqual(index.read_text(encoding="utf-8"), original)
        self.assertEqual(recording.read_bytes(), b"original private recording")
        return status, calls, output, capture.getvalue()

    def test_sensitive_text_never_reaches_any_prompt_or_derived_artifact(self):
        chunks = [
            chunk("[SPEAKER_01] Let's get started. At the last meeting, public safety was discussed. Redact the following. UNIQUE_PRIVATE_RECAP.", "4.2"),
            chunk("[SPEAKER_02] UNIQUE_PRIVATE_DETAIL. End redaction. We'll move right along then. Public budget approved.", "5.1", 60, 120),
        ]
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output, logs = self.run_pipeline(directory, chunks)
            self.assertEqual(status, 0)
            self.assertTrue(any("Summarize only the supplied historical recap" in call["prompt"] for call in calls))
            for secret in ("UNIQUE_PRIVATE_RECAP", "UNIQUE_PRIVATE_DETAIL", "STALE PRIVATE CONTENT"):
                self.assertNotIn(secret, json.dumps(calls))
                self.assertNotIn(secret, logs)
                for path in output.iterdir():
                    if path.name != "redactions.json":
                        self.assertNotIn(secret, path.read_text(encoding="utf-8"), path.name)
            private = json.loads((output / "redactions.json").read_text(encoding="utf-8"))
            self.assertEqual(private["redactions"][0]["content"], "[SPEAKER_01] UNIQUE_PRIVATE_RECAP.\n[SPEAKER_02] UNIQUE_PRIVATE_DETAIL.")
            self.assertEqual(private["redactions"][0]["speakers"], ["SPEAKER_01", "SPEAKER_02"])
            self.assertIn("## Recap of Previous Meeting", (output / "minutes-draft.md").read_text(encoding="utf-8"))
            payload = json.dumps(publication_payload(output))
            self.assertNotIn("UNIQUE_PRIVATE", payload)
            self.assertNotIn("redactions.json", payload)
            website = Path(directory) / "website"
            export_documents(output, website)
            self.assertFalse((website / "redactions.json").exists())
            self.assertTrue(all("UNIQUE_PRIVATE" not in path.read_text(encoding="utf-8") for path in website.iterdir()))

    def test_generated_markdown_cannot_link_private_review_record(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _, output, _ = self.run_pipeline(directory, [chunk("[SPEAKER_01] Public. See [private review](redactions.json). Redact the following. SECRET. End redaction.")])
            self.assertEqual(status, 0)
            for name in ("summary.md", "action-items.md", "minutes-draft.md"):
                self.assertNotIn("redactions.json", (output / name).read_text(encoding="utf-8").casefold())

    def test_warnings_have_no_sensitive_excerpts_or_text(self):
        source = "[SPEAKER_01] End redaction. Public. Redact the following. PRIVATE ONE. Redact the following.\n[SPEAKER_02] PRIVATE TWO through EOF."
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output, logs = self.run_pipeline(directory, [chunk(source)])
            self.assertEqual(status, 0)
            report = json.loads((output / "minutes-qa.json").read_text(encoding="utf-8"))
            warnings = [finding for finding in report["findings"] if finding["code"].startswith("redaction_")]
            self.assertEqual({warning["code"] for warning in warnings}, {"redaction_unmatched_end", "redaction_repeated_start", "redaction_unclosed"})
            self.assertTrue(all(warning["excerpt"] == "" for warning in warnings))
            for public in (json.dumps(report), (output / "minutes-qa.md").read_text(encoding="utf-8"), json.dumps(calls), logs):
                self.assertNotIn("PRIVATE ONE", public)
                self.assertNotIn("PRIVATE TWO", public)

    def test_all_content_redacted_makes_no_model_calls_and_replaces_stale_outputs(self):
        for text in ("[SPEAKER_01] redact the following SECRET end redaction", "[SPEAKER_01] redact the following SECRET"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                status, calls, output, _ = self.run_pipeline(directory, [chunk(text)])
                self.assertEqual(status, 0)
                self.assertEqual(calls, [])
                for path in output.iterdir():
                    if path.name != "redactions.json":
                        self.assertNotIn("SECRET", path.read_text(encoding="utf-8"))
                        self.assertNotIn("STALE PRIVATE", path.read_text(encoding="utf-8"))
                self.assertEqual((output / "meeting_sections.jsonl").read_text(encoding="utf-8"), "")
                self.assertEqual((output / "chunk_summaries.jsonl").read_text(encoding="utf-8"), "")

    def test_model_failure_preserves_private_review_but_not_stale_public_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _, output, _ = self.run_pipeline(directory, [chunk("[SPEAKER_01] Public. redact the following SECRET end redaction")], fail_map=True)
            self.assertEqual(status, 1)
            self.assertIn("SECRET", (output / "redactions.json").read_text(encoding="utf-8"))
            self.assertEqual(public_meeting_files(output), [])
            self.assertFalse((output / "chunk_summaries.jsonl").exists())

    def test_lesson_mode_does_not_apply_spoken_redaction(self):
        source = "[SPEAKER_01] redact the following LESSON CONTENT end redaction"
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output, _ = self.run_pipeline(directory, [chunk(source)], profile="lesson")
            self.assertEqual(status, 0)
            self.assertIn("LESSON CONTENT", calls[0]["prompt"])
            self.assertFalse((output / "redactions.json").exists())
            self.assertIn("LESSON CONTENT", (output / "lesson-notes.md").read_text(encoding="utf-8"))


class PublicationPrivacyTests(unittest.TestCase):
    def populate(self, directory):
        root = Path(directory)
        source = root / "private"
        source.mkdir()
        for name in ("summary.md", "action-items.md", "minutes-draft.md"):
            (source / name).write_text("# Safe meeting document", encoding="utf-8")
        (source / "redactions.json").write_text('{"redactions":[{"content":"PRIVATE REVIEW TEXT"}]}', encoding="utf-8")
        (source / "unlisted.md").write_text("PRIVATE UNLISTED TEXT", encoding="utf-8")
        (source / "source-transcript.json").write_text("PRIVATE ORIGINAL TEXT", encoding="utf-8")
        return source

    def test_directory_export_payload_and_archive_explicitly_exclude_private_record(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.populate(directory)
            public = Path(directory) / "website"
            export_documents(source, public)
            self.assertEqual({path.name for path in public.iterdir()}, {"summary.md", "action-items.md", "minutes-draft.md"})
            payload = publication_payload(source)
            self.assertNotIn("redactions.json", json.dumps(payload))
            self.assertNotIn("PRIVATE", json.dumps(payload))
            archive = Path(directory) / "published.zip"
            export_archive(source, archive)
            with zipfile.ZipFile(archive) as document:
                self.assertEqual(set(document.namelist()), {"summary.md", "action-items.md", "minutes-draft.md"})
                self.assertTrue(all(b"PRIVATE" not in document.read(name) for name in document.namelist()))

    def test_publication_never_reads_private_json_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.populate(directory)
            original = Path.read_text

            def checked_read(path, *args, **kwargs):
                self.assertNotEqual(path.name.casefold(), "redactions.json")
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", checked_read):
                self.assertEqual(len(publication_payload(source)["documents"]), 3)

    def test_private_record_links_are_removed_from_all_public_export_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.populate(directory)
            (source / "summary.md").write_text('# Summary\nPublic facts. [Private review](redactions.json)\n<a href="../REDACTIONS.JSON">Private link</a>\n[review]: redactions.json\nMore public facts.', encoding="utf-8")
            payload = json.dumps(publication_payload(source))
            self.assertNotIn("redactions.json", payload.casefold())
            self.assertIn("More public facts", payload)
            destination = Path(directory) / "website"
            export_documents(source, destination)
            self.assertNotIn("redactions.json", (destination / "summary.md").read_text(encoding="utf-8").casefold())
            archive = Path(directory) / "public.zip"
            export_archive(source, archive)
            with zipfile.ZipFile(archive) as document:
                self.assertNotIn(b"redactions.json", document.read("summary.md").lower())

    def test_hardlinked_private_record_is_not_exported_under_public_name(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.populate(directory)
            (source / "summary.md").unlink()
            os.link(source / "redactions.json", source / "summary.md")
            self.assertNotIn("summary.md", {path.name for path in public_meeting_files(source)})
            self.assertNotIn("PRIVATE", json.dumps(publication_payload(source)))

    def test_private_path_name_is_denied_even_if_it_is_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / "redactions.json"
            private.mkdir()
            (private / "summary.md").write_text("PRIVATE", encoding="utf-8")
            self.assertEqual(public_meeting_files(private), [])

    def test_export_cli_uses_same_private_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.populate(directory)
            destination = Path(directory) / "website"
            with patch.object(sys, "argv", ["export_meeting", str(source), str(destination)]):
                self.assertEqual(export_meeting.main(), 0)
            self.assertFalse((destination / "redactions.json").exists())
            self.assertFalse((destination / "unlisted.md").exists())

    def test_git_ignores_private_records_outside_default_output_root(self):
        git = shutil.which("git")
        if not git:
            self.skipTest("Git is unavailable")
        result = subprocess.run([git, "check-ignore", "--no-index", "custom-output/session/redactions.json"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
